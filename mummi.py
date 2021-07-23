"""
This is the official implementation of Dreamer.
Dream to Control: Learning Behaviors by Latent Imagination  https://arxiv.org/abs/1912.01603
https://github.com/danijar/dreamer
"""
import argparse
import collections
import functools
import json
import os
import pathlib
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'
tf.get_logger().setLevel('ERROR')

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
import wrappers
from config import define_config as define_config  # Yong Lee, this is a debug config for fast exe
from tools import cal_result


class CMDreamer(tools.Module):

    def __init__(self, config, datadir, actspace, writer):
        self._c = config
        self._actspace = actspace
        self._actdim = actspace.n if hasattr(actspace, 'n') else actspace.shape[0]
        self._writer = writer
        self._random = np.random.RandomState(config.seed)
        with tf.device('cpu:0'):
            self._step = tf.Variable(count_steps(datadir, config), dtype=tf.int64)
        self._should_pretrain = tools.Once()
        self._should_train = tools.Every(config.train_every)
        self._should_log = tools.Every(config.log_every)
        self._last_log = None
        self._last_time = time.time()
        self._metrics = collections.defaultdict(tf.metrics.Mean)
        self._metrics['expl_amount']  # Create variable for checkpoint.
        self._float = prec.global_policy().compute_dtype
        self._strategy = tf.distribute.MirroredStrategy()
        with self._strategy.scope():
            self._dataset = iter(self._strategy.experimental_distribute_dataset(
                load_dataset(datadir, self._c)))
            self._build_model()

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step.numpy().item()
        tf.summary.experimental.set_step(step)
        if state is not None and reset.any():
            mask = tf.cast(1 - reset, self._float)[:, None]
            state = tf.nest.map_structure(lambda x: x * mask, state)
        # if self._should_train(step):
        if self._should_train(step) and not self._c.test:
            log = self._should_log(step)
            n = self._c.pretrain if self._should_pretrain() else self._c.train_steps
            print(f'Training for {n} steps.')
            with self._strategy.scope():
                for train_step in range(n):
                    log_images = self._c.log_images and log and train_step == 0
                    self.train(next(self._dataset), log_images)
            if log:
                self._write_summaries()
        action, state = self.policy(obs, state, training)
        if training:
            self._step.assign_add(len(reset) * self._c.action_repeat)
        return action, state

    @tf.function
    def policy(self, obs, state, training):
        if state is None:
            latent = self._dynamics.initial(len(obs['image']))
            action = tf.zeros((len(obs['image']), self._actdim), self._float)
        else:
            latent, action = state

        obs = preprocess(obs, self._c)
        embed_img = self._encode_img(obs)
        embed_dep = self._encode_dep(obs)
        embed_touch = self._encode_touch(obs["touch"])
        # embed_audio = self._encode_audio(obs["audio"])
        prob_img = self._embed2p_img(embed_img)
        prob_dep = self._embed2p_dep(embed_dep)
        prob_touch = self._embed2p_touch(embed_touch)
        # prob_audio = self._embed2p_audio(embed_audio)

        # embed = (prob_img["mean"] * prob_dep["std"] * obs["image_flag"] +
        #          prob_dep["mean"] * prob_img["std"] * obs["depth_flag"]) / (
        #                 prob_img["std"] * obs["depth_flag"] + prob_dep["std"] * obs["image_flag"] + 0.1)
        w_img = obs["image_flag"] / prob_img["std"]
        w_dep = obs["depth_flag"] / prob_dep["std"]
        w_touch = obs["touch_flag"] / prob_touch["std"]
        # w_audio = obs["audio_flag"] / prob_audio["std"]
        embed = w_img * prob_img["mean"] + w_dep * prob_dep["mean"]
        embed = embed + w_touch * prob_touch["mean"]  # + w_audio * prob_audio["mean"]
        embed = embed / (w_img + w_dep + w_touch + 0.01)

        latent, _ = self._dynamics.obs_step(latent, action, embed)
        feat = self._dynamics.get_feat(latent)

        if self._c.trajectory_opt:
            action = self._trajectory_optimization(latent)
        elif self._c.forward_search:
            action = self._forward_search_policy(latent)
        else:
            if training:
                action = self._actor(feat).sample()
            else:
                action = self._actor(feat).mode()

        action = self._exploration(action, training)
        state = (latent, action)
        return action, state

    def load(self, filename):
        super().load(filename)
        self._should_pretrain()

    @tf.function()
    def train(self, data, log_images=False):
        self._strategy.experimental_run_v2(self._train, args=(data, log_images))
        # self._train(data, log_images)

    def _train(self, data, log_images):
        with tf.GradientTape() as model_tape:
            # encoder
            embed_img = self._encode_img(data)
            embed_dep = self._encode_dep(data)
            embed_touch = self._encode_touch(data["touch"])
            # embed_audio = self._encode_audio(data["audio"])
            prob_img = self._embed2p_img(embed_img)
            prob_dep = self._embed2p_dep(embed_dep)
            prob_touch = self._embed2p_touch(embed_touch)
            # prob_audio = self._embed2p_audio(embed_audio)

            # poe
            w_img = data["image_flag"] / prob_img["std"]
            w_dep = data["depth_flag"] / prob_dep["std"]
            w_touch = data["touch_flag"] / prob_touch["std"]
            # w_audio = data["audio_flag"] / prob_audio["std"]
            embed = w_img * prob_img["mean"] + w_dep * prob_dep["mean"]
            embed = embed + w_touch * prob_touch["mean"]  # + w_audio * prob_audio["mean"]
            embed = embed / (w_img + w_dep + w_touch + 0.01)

            # RSSM
            post, prior = self._dynamics.observe(embed, data['action'])
            feat = self._dynamics.get_feat(post)

            # decoder
            reward_pred = self._reward(feat)
            image_pred = self._decode_img(feat)
            depth_pred = self._decode_dep(feat)

            # loss
            likes = tools.AttrDict()
            likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
            # contrast_img = self._contra_img(feat, prob_img["mean"], data["image_flag"])
            # contrast_dep = self._contra_dep(feat, prob_dep["mean"], data["depth_flag"])
            feat_embed = self._z2embed(feat)
            constrast_img = self._contrastive(feat_embed, prob_img["mean"], data["image_flag"])
            constrast_dep = self._contrastive(feat_embed, prob_dep["mean"], data["depth_flag"])
            constrast_touch = 0.001 * self._contrastive(feat_embed, prob_touch["mean"], data["touch_flag"])
            # walker_stand 0.0025
            # walker_walk 0.001
            # finger_spin 0.001 & 0.1
            # constrast_audio = weight[0, 0, 3] * self._contrastive(feat_embed, prob_audio["mean"], data["audio_flag"])

            #
            # # likes.image = tf.reduce_mean(image_pred.log_prob(data['image']))
            # # likes.depth = tf.reduce_mean(depth_pred.log_prob(data['depth']))
            # likes.image = tf.reduce_mean(contrast_img)
            # likes.depth = tf.reduce_mean(contrast_dep)
            # likes.image = tf.reduce_mean(self._contra_img(feat, embed))
            likes.image = tf.reduce_mean(constrast_img)
            likes.depth = tf.reduce_mean(constrast_dep)
            likes.touch = tf.reduce_mean(constrast_touch)
            # likes.audio = tf.reduce_mean(constrast_audio)

            # dep_max = tf.reduce_max(prob_dep["mean"])

            if self._c.pcont:
                pcont_pred = self._pcont(feat)
                pcont_target = self._c.discount * data['discount']
                likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
                likes.pcont *= self._c.pcont_scale
            prior_dist = self._dynamics.get_dist(prior)
            post_dist = self._dynamics.get_dist(post)
            div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            div = tf.maximum(div, self._c.free_nats)
            model_loss = self._c.kl_scale * div - sum(likes.values())
            model_loss /= float(self._strategy.num_replicas_in_sync)

        with tf.GradientTape() as actor_tape:
            imag_feat = self._imagine_ahead(post)
            reward = self._reward(imag_feat).mode()
            if self._c.pcont:
                pcont = self._pcont(imag_feat).mean()
            else:
                pcont = self._c.discount * tf.ones_like(reward)
            value = self._value(imag_feat).mode()
            returns = tools.lambda_return(
                reward[:-1], value[:-1], pcont[:-1],
                bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
            discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
                [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
            actor_loss = -tf.reduce_mean(discount * returns)
            actor_loss /= float(self._strategy.num_replicas_in_sync)

        with tf.GradientTape() as value_tape:
            value_pred = self._value(imag_feat)[:-1]
            target = tf.stop_gradient(returns)
            value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))
            value_loss /= float(self._strategy.num_replicas_in_sync)

        model_norm = self._model_opt(model_tape, model_loss)
        actor_norm = self._actor_opt(actor_tape, actor_loss)
        value_norm = self._value_opt(value_tape, value_loss)

        if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
            if self._c.log_scalars:
                # self._scalar_summaries(data, feat, prior_dist, post_dist, likes, div, model_loss, value_loss,
                #                        actor_loss, model_norm, value_norm, actor_norm)
                self._scalar_summaries(data, feat, prior_dist, post_dist, likes, div, model_loss, value_loss,
                                       actor_loss, model_norm, value_norm, actor_norm)
            if tf.equal(log_images, True):
                self._image_summaries(data, embed, image_pred)

    def _build_model(self):
        acts = dict(
            elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
            leaky_relu=tf.nn.leaky_relu)
        cnn_act = acts[self._c.cnn_act]
        act = acts[self._c.dense_act]
        # self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act)  # Yong Lee, modified
        self._encode_img = models.ConvEncoder(self._c.cnn_depth, cnn_act, modality="image")
        self._encode_dep = models.ConvEncoder(self._c.cnn_depth, cnn_act, modality="depth")
        self._encode_touch = models.Dense(n=4, d_hidden=128, d_out=1024, name="touch")
        # self._encode_audio = models.Dense(n=4, d_hidden=128, d_out=1024, name="audio")
        self._embed2p_img = models.Embed2prob(1024, modality="image")
        self._embed2p_dep = models.Embed2prob(1024, modality="depth")
        self._embed2p_touch = models.Embed2prob(1024, modality="touch")
        # self._embed2p_audio = models.Embed2prob(1024, modality="audio")
        self._dynamics = models.RSSM(self._c.stoch_size, self._c.deter_size, self._c.deter_size)
        self._z2embed = models.Dense(n=1, d_hidden=None, d_out=1024)
        self._decode_img = models.ConvDecoder(self._c.cnn_depth, cnn_act, shape=(64, 64, 3), modality="image")
        self._decode_dep = models.ConvDecoder(self._c.cnn_depth, cnn_act, shape=(64, 64, 1), modality="depth")
        self._contrastive = models.ContrastiveModel6(kernel="mse")
        self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
        if self._c.pcont:
            self._pcont = models.DenseDecoder((), 3, self._c.num_units, 'binary', act=act)
        self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
        self._actor = models.ActionDecoder(self._actdim, 4, self._c.num_units, self._c.action_dist,
                                           init_std=self._c.action_init_std, act=act)
        model_modules = [self._encode_img, self._encode_dep, self._encode_touch,
                         self._embed2p_img, self._embed2p_dep, self._embed2p_touch,
                         self._dynamics, self._z2embed, self._reward]

        if self._c.pcont:
            model_modules.append(self._pcont)
        Optimizer = functools.partial(
            tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
            wdpattern=self._c.weight_decay_pattern)
        self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
        self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
        self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
        # Do a train step to initialize all variables, including optimizer
        # statistics. Ideally, we would use batch size zero, but that doesn't work
        # in multi-GPU mode.
        self.train(next(self._dataset))

    def _exploration(self, action, training):
        if training:
            amount = self._c.expl_amount
            if self._c.expl_decay:
                amount *= 0.5 ** (tf.cast(self._step, tf.float32) / self._c.expl_decay)
            if self._c.expl_min:
                amount = tf.maximum(self._c.expl_min, amount)
            self._metrics['expl_amount'].update_state(amount)
        elif self._c.eval_noise:
            amount = self._c.eval_noise
        else:
            return action
        if self._c.expl == 'additive_gaussian':
            return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
        if self._c.expl == 'completely_random':
            return tf.random.uniform(action.shape, -1, 1)
        if self._c.expl == 'epsilon_greedy':
            indices = tfd.Categorical(0 * action).sample()
            return tf.where(
                tf.random.uniform(action.shape[:1], 0, 1) < amount,
                tf.one_hot(indices, action.shape[-1], dtype=self._float),
                action)
        raise NotImplementedError(self._c.expl)

    def _imagine_ahead(self, post):
        if self._c.pcont:  # Last step could be terminal.
            post = {k: v[:, :-1] for k, v in post.items()}
        flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in post.items()}
        policy = lambda state: self._actor(
            tf.stop_gradient(self._dynamics.get_feat(state))).sample()
        states = tools.static_scan(
            lambda prev, _: self._dynamics.img_step(prev, policy(prev)),
            tf.range(self._c.horizon), start)
        imag_feat = self._dynamics.get_feat(states)
        return imag_feat

    def _forward_search_policy(self, post):
        # [1, size]

        def policy(state):
            return self._actor(
                tf.stop_gradient(self._dynamics.get_feat(state))).sample()

        def repeat(x):
            return tf.repeat(x, self._c.num_samples, axis=0)

        start = {k: repeat(v) for k, v in post.items()}
        states, actions = tools.static_scan_action(
            lambda prev, action, _: self._dynamics.img_step(prev, action),
            lambda prev: policy(prev),
            tf.range(self._c.horizon), start)

        feat = self._dynamics.get_feat(states)
        reward = self._reward(feat).mode()

        if self._c.pcont:
            pcont = self._pcont(feat).mean()
        else:
            pcont = self._c.discount * tf.ones_like(reward)
        value = self._value(feat).mode()
        returns = tools.lambda_return(
            reward[:-1], value[:-1], pcont[:-1],
            bootstrap=value[-1], lambda_=self._c.disclam, axis=0)

        idx = tf.argmax(returns[0])
        act = actions[idx][None, :]
        return act

    def _trajectory_optimization(self, post):
        # [1, size]

        def policy(state):
            return self._actor(
                tf.stop_gradient(self._dynamics.get_feat(state))).sample()

        def repeat(x):
            return tf.repeat(x, self._c.num_samples, axis=0)

        states, actions = tools.static_scan_action(
            lambda prev, action, _: self._dynamics.img_step(prev, action),
            lambda prev: policy(prev),
            tf.range(self._c.horizon), post)

        feat = self._dynamics.get_feat(states)
        reward = self._reward(feat).mode()

        if self._c.pcont:
            pcont = self._pcont(feat).mean()
        else:
            pcont = self._c.discount * tf.ones_like(reward)
        value = self._value(feat).mode()
        returns = tools.lambda_return(
            reward[:-1], value[:-1], pcont[:-1],
            bootstrap=value[-1], lambda_=self._c.disclam, axis=0)

        accumulated_reward = returns[0, 0]
        grad = tf.gradients(accumulated_reward, actions)[0]
        act = actions + grad * self._c.traj_opt_lr

        return act

    def _scalar_summaries(
            self, data, feat, prior_dist, post_dist, likes, div,
            model_loss, value_loss, actor_loss, model_norm, value_norm,
            actor_norm):
        # self._metrics['Yong Obs'].update_state(data)  # Yong Lee Debug
        self._metrics['model_grad_norm'].update_state(model_norm)
        self._metrics['value_grad_norm'].update_state(value_norm)
        self._metrics['actor_grad_norm'].update_state(actor_norm)
        self._metrics['prior_ent'].update_state(prior_dist.entropy())
        self._metrics['post_ent'].update_state(post_dist.entropy())
        for name, logprob in likes.items():
            self._metrics[name + '_loss'].update_state(-logprob)
        self._metrics['div'].update_state(div)
        self._metrics['model_loss'].update_state(model_loss)
        self._metrics['value_loss'].update_state(value_loss)
        self._metrics['actor_loss'].update_state(actor_loss)
        self._metrics['action_ent'].update_state(self._actor(feat).entropy())

    def _image_summaries(self, data, embed, image_pred):
        truth = data['image'][:6] + 0.5
        recon = image_pred.mode()[:6]
        init, _ = self._dynamics.observe(embed[:6, :5], data['action'][:6, :5])
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self._dynamics.imagine(data['action'][:6, 5:], init)
        openl = self._decode_img(self._dynamics.get_feat(prior)).mode()
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        openl = tf.concat([truth, model, error], 2)
        tools.graph_summary(self._writer, tools.video_summary, 'agent/openl', openl)

    def _write_summaries(self):
        step = int(self._step.numpy())
        metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
        if self._last_log is not None:
            duration = time.time() - self._last_time
            self._last_time += duration
            metrics.append(('fps', (step - self._last_log) / duration))
        self._last_log = step
        [m.reset_states() for m in self._metrics.values()]
        with (self._c.logdir / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
        [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
        print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
        self._writer.flush()


def preprocess(obs, config):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    with tf.device('cpu:0'):
        obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
        obs['depth'] = tf.cast(obs['depth'], dtype) / 200.0 - 0.5
        obs['touch'] = tf.cast(obs['touch'], dtype) / 200.0
        # obs['audio'] = tf.cast(obs['audio'], dtype) / 10240.0
        clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
        obs['reward'] = clip_rewards(obs['reward'])
    return obs


def count_steps(datadir, config):
    return tools.count_episodes(datadir)[1] * config.action_repeat


def load_dataset(directory, config):
    episode = next(tools.load_episodes(directory, 1))
    types = {k: v.dtype for k, v in episode.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
    generator = lambda: tools.load_episodes(
        directory, config.train_steps, config.batch_length,
        config.dataset_balance)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.map(functools.partial(preprocess, config=config))
    dataset = dataset.prefetch(10)
    return dataset


def summarize_episode(episode, config, datadir, writer, prefix):
    episodes, steps = tools.count_episodes(datadir)
    length = (len(episode['reward']) - 1) * config.action_repeat
    ret = episode['reward'].sum()
    print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
    metrics = [
        (f'{prefix}/return', float(episode['reward'].sum())),
        (f'{prefix}/length', len(episode['reward']) - 1),
        (f'episodes', episodes)]
    step = count_steps(datadir, config)
    # with (config.logdir / 'metrics.jsonl').open('a') as f:
    #     f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
    if config.test:
        with (config.logdir / 'results.jsonl').open('a') as f:
            f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
    else:
        with (config.logdir / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
    with writer.as_default():  # Env might run in a different thread.
        tf.summary.experimental.set_step(step)
        [tf.summary.scalar('sim/' + k, v) for k, v in metrics]
        if prefix == 'test':
            tools.video_summary(f'sim/{prefix}/video', episode['image'][None])


def make_env(config, writer, prefix, datadir, store):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
        env = wrappers.DeepMindControl(task)
        env = wrappers.ActionRepeat(env, config.action_repeat)
        env = wrappers.NormalizeActions(env)
        if config.natural:
            data = tools.load_imgnet(store)
            env = wrappers.NaturalMujoco(env, data)
            audio_data = tools.load_audio(store)
            env = wrappers.AudioMujoco(env, audio_data)
            env = wrappers.MissingMultimodal(env, config)
    elif suite == 'atari':
        env = wrappers.Atari(
            task, config.action_repeat, (64, 64), grayscale=False,
            life_done=True, sticky_actions=True)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
    callbacks = []
    if store:
        callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
    callbacks.append(
        lambda ep: summarize_episode(ep, config, datadir, writer, prefix))
    env = wrappers.Collect(env, callbacks, config.precision)
    env = wrappers.RewardObs(env)
    return env


def main(config):
    if config.gpu_growth:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        prec.set_policy(prec.Policy('mixed_float16'))
    config.steps = int(config.steps)
    config.logdir.mkdir(parents=True, exist_ok=True)
    print('Logdir', config.logdir)

    # Create environments.
    datadir = config.logdir / 'episodes'
    writer = tf.summary.create_file_writer(
        str(config.logdir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    train_envs = [wrappers.Async(lambda: make_env(
        config, writer, 'train', datadir, store=True), config.parallel)
                  for _ in range(config.envs)]
    test_envs = [wrappers.Async(lambda: make_env(
        config, writer, 'test', datadir, store=False), config.parallel)
                 for _ in range(config.envs)]
    actspace = train_envs[0].action_space

    # Prefill dataset with random episodes.
    step = count_steps(datadir, config)
    prefill = max(0, config.prefill - step)
    print(f'Prefill dataset with {prefill} steps.')
    random_agent = lambda o, d, _: ([actspace.sample() for _ in d], None)
    tools.simulate(random_agent, train_envs, prefill / config.action_repeat)
    writer.flush()

    # Train and regularly evaluate the agent.
    step = count_steps(datadir, config)
    print(f'Simulating agent for {config.steps - step} steps.')
    agent = CMDreamer(config, datadir, actspace, writer)
    if (config.logdir / 'variables.pkl').exists():
        print('Load checkpoint.')
        agent.load(config.logdir / 'variables.pkl')
    state = None
    while step < config.steps:
        print('Start evaluation.')
        tools.simulate(
            functools.partial(agent, training=False), test_envs, episodes=1)
        writer.flush()
        print('Start collection.')
        steps = config.eval_every // config.action_repeat
        state = tools.simulate(agent, train_envs, steps, state=state)
        step = count_steps(datadir, config)
        agent.save(config.logdir / 'variables.pkl')
    for env in train_envs + test_envs:
        env.close()


def test(config):
    if config.gpu_growth:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        prec.set_policy(prec.Policy('mixed_float16'))
    config.steps = int(config.steps)
    config.logdir.mkdir(parents=True, exist_ok=True)
    print('Logdir', config.logdir)

    # Create environments.
    datadir = config.logdir / 'episodes'
    writer = tf.summary.create_file_writer(
        str(config.logdir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    # train_envs = [wrappers.Async(lambda: make_env(
    #     config, writer, 'train', datadir, store=True), config.parallel)
    #               for _ in range(config.envs)]
    test_envs = [wrappers.Async(lambda: make_env(
        config, writer, 'test', datadir, store=False), config.parallel)
                 for _ in range(config.envs)]
    actspace = test_envs[0].action_space

    # # Prefill dataset with random episodes.
    # step = count_steps(datadir, config)
    # prefill = max(0, config.prefill - step)
    # print(f'Prefill dataset with {prefill} steps.')
    # random_agent = lambda o, d, _: ([actspace.sample() for _ in d], None)
    # tools.simulate(random_agent, train_envs, prefill / config.action_repeat)
    # writer.flush()

    # Train and regularly evaluate the agent.
    step = count_steps(datadir, config)
    print(f'Simulating agent for {config.steps - step} steps.')
    agent = CMDreamer(config, datadir, actspace, writer)
    if (config.logdir / 'variables.pkl').exists():
        print('Load checkpoint.')
        agent.load(config.logdir / 'variables.pkl')

    m_list = [0.0, 0.05, 0.1]
    for miss_r in m_list:
        config.miss_ratio = {"image": miss_r, "depth": miss_r, "touch": miss_r, "audio": miss_r}
        test_envs = [wrappers.Async(lambda: make_env(
            config, writer, 'test', datadir, store=False), config.parallel)
                     for _ in range(config.envs)]
        print('Missing Ratio:', miss_r)
        n_traj = 10
        for _ in range(n_traj):
            print('Start evaluation.')
            tools.simulate(
                functools.partial(agent, training=False), test_envs, episodes=1)
            writer.flush()
            # print('Start collection.')
            # steps = config.eval_every // config.action_repeat
            # state = tools.simulate(agent, train_envs, steps, state=state)
            # step = count_steps(datadir, config)
            # agent.save(config.logdir / 'variables.pkl')
        for env in test_envs:
            env.close()

        filename = config.logdir / 'results.jsonl'
        cal_result(str(filename), n_traj)



if __name__ == '__main__':
    try:
        import colored_traceback

        colored_traceback.add_hook()
    except ImportError:
        pass
    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
    # main(parser.parse_args())
    config = parser.parse_args()
    if config.test:
        test(config)
    else:
        main(config)
