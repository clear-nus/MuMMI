import pathlib
import tools

"""
The inherit config for different methods:
    CMD(Ours) ---> CVRL ---> Dreamer
    config         config2   config1 
"""


def define_config1():
    """
    This is the original config of dreamer
    https://github.com/danijar/dreamer/blob/master/dreamer.py
    https://arxiv.org/pdf/1912.01603.pdf
    """
    config = tools.AttrDict()
    # General.
    config.logdir = pathlib.Path('.')
    config.seed = 0
    config.steps = 2e6
    config.eval_every = 1e4
    config.log_every = 1e3
    config.log_scalars = True
    config.log_images = True
    config.gpu_growth = True
    config.precision = 16
    # Environment.
    config.task = 'dmc_walker_walk'
    config.envs = 1
    config.parallel = 'none'
    config.action_repeat = 2
    config.time_limit = 1000
    config.prefill = 5000
    config.eval_noise = 0.0
    config.clip_rewards = 'none'
    # Model.
    config.deter_size = 200
    config.stoch_size = 30
    config.num_units = 400
    config.dense_act = 'elu'
    config.cnn_act = 'relu'
    config.cnn_depth = 32
    config.pcont = False
    config.free_nats = 3.0
    config.kl_scale = 1.0
    config.pcont_scale = 10.0
    config.weight_decay = 0.0
    config.weight_decay_pattern = r'.*'
    # Training.
    config.batch_size = 50
    config.batch_length = 50
    config.train_every = 1000
    config.train_steps = 100
    config.pretrain = 100
    config.model_lr = 6e-4
    config.value_lr = 8e-5
    config.actor_lr = 8e-5
    config.grad_clip = 30.0
    config.dataset_balance = False
    # Behavior.
    config.discount = 0.99
    config.disclam = 0.95
    config.horizon = 15
    config.action_dist = 'tanh_normal'
    config.action_init_std = 5.0
    config.expl = 'additive_gaussian'
    config.expl_amount = 0.3
    config.expl_decay = 0.0
    config.expl_min = 0.0
    return config


def define_config2():
    """
    This is the original config of CVRL
    https://github.com/Yusufma03/CVRL_dev
    https://arxiv.org/abs/2008.02430
    """
    config = define_config1()

    config.log_imgs = False

    # natural or not
    config.natural = False

    # obs model
    config.obs_model = 'contrastive'

    # SAC settings
    config.num_Qs = 2
    config.use_sac = False
    config.use_dreamer = True

    config.reward_only = False

    # forward search
    config.forward_search = False
    config.trajectory_opt = True
    config.traj_opt_lr = 0.003
    config.num_samples = 20
    return config


def define_config():
    """
    This is the our config of CMD, and can be used for Dreamer and CVRL
    """
    config = define_config2()

    config.natural = True

    # CMDreamer extra config  -----------------------------------------
    # multimodal
    config.multi_modal = True  # True or False
    # config.miss_ratio = {"image0": 0.05, "image1": 0.05, "image2": 0.05, "image3": 0.05}  # Four views
    config.miss_ratio = {"image": 0.05, "depth": 0.05, "touch": 0.05, "audio": 0.05}  # Four modalities
    config.max_miss_len = 15  # 2, random drop; >2:  seg drop
    config.test = False
    return config


def define_config3():
    """
    This is the our config of CMD, and can be used for Dreamer and CVRL
    Less steps for debug
    """
    config = define_config()

    config.steps = 500  # iteration steps? or perform
    config.eval_every = 50  # steps for evaluate
    config.log_every = 50  # steps interval for log
    config.time_limit = 20  # 1000
    config.prefill = 20  # 50/20 就是3集咯

    config.batch_size = 2
    config.batch_length = 10
    config.train_every = 100
    config.train_steps = 2
    config.pretrain = 2
    return config

#
# def define_config1():  # 简单配置
#     config = tools.AttrDict()  # 这就是一个字典,类，
#     # General.
#     config.logdir = pathlib.Path('.')  # log dir
#     config.seed = 0  # 随机种子，
#     config.steps = 100  # iteration steps? or perform
#     config.eval_every = 50  # steps for evaluate
#     config.log_every = 50  # steps interval for log
#     config.log_scalars = True
#     config.log_images = True
#     config.gpu_growth = False  # Yong Lee
#     config.precision = 16  # 16位的计算精度
#     # Environment.
#     config.task = 'dmc_walker_walk'
#     config.envs = 1
#     config.parallel = 'none'  # 并行？怎么理解
#     config.action_repeat = 2  # 怎么解释这个repeat？
#     config.time_limit = 20  # 1000
#     config.prefill = 20  # 50/20 就是3集咯
#     config.eval_noise = 0.0
#     config.clip_rewards = 'none'
#     # Model.
#     config.deter_size = 100
#     config.stoch_size = 130
#     config.num_units = 400
#     config.dense_act = 'elu'
#     config.cnn_act = 'relu'
#     config.cnn_depth = 32
#     config.pcont = False
#     config.free_nats = 3.0
#     config.kl_scale = 1.0
#     config.pcont_scale = 10.0
#     config.weight_decay = 0.0
#     config.weight_decay_pattern = r'.*'
#     # Training.
#     config.batch_size = 2
#     config.batch_length = 10
#     config.train_every = 100
#     config.train_steps = 2
#     config.pretrain = 2
#     config.model_lr = 6e-4
#     config.value_lr = 8e-5
#     config.actor_lr = 8e-5
#     config.grad_clip = 100.0
#     config.dataset_balance = False
#     # Behavior.
#     config.discount = 0.99
#     config.disclam = 0.95
#     config.horizon = 15
#     config.action_dist = 'tanh_normal'
#     config.action_init_std = 5.0
#     config.expl = 'additive_gaussian'
#     config.expl_amount = 0.3
#     config.expl_decay = 0.0
#     config.expl_min = 0.0
#     config.test = False
#     return config
#
#
# def define_config2():  # Yong Lee, this is the default setting
#     config = tools.AttrDict()
#     # General.
#     config.logdir = pathlib.Path('.')
#     config.seed = 0
#     config.steps = 5e6
#     config.eval_every = 1e4
#     config.log_every = 1e3
#     config.log_scalars = True
#     config.log_images = True
#     config.gpu_growth = True
#     config.precision = 16
#     # Environment.
#     config.task = 'dmc_walker_walk'
#     config.envs = 1
#     config.parallel = 'none'
#     config.action_repeat = 2
#     config.time_limit = 1000
#     config.prefill = 5000
#     config.eval_noise = 0.0
#     config.clip_rewards = 'none'
#     # Model.
#     config.deter_size = 200
#     config.stoch_size = 30
#     config.num_units = 400
#     config.dense_act = 'elu'
#     config.cnn_act = 'relu'
#     config.cnn_depth = 32
#     config.pcont = False
#     config.free_nats = 3.0
#     config.kl_scale = 1.0
#     config.pcont_scale = 10.0
#     config.weight_decay = 0.0
#     config.weight_decay_pattern = r'.*'
#     # Training.
#     config.batch_size = 50
#     config.batch_length = 20
#     config.train_every = 1000
#     config.train_steps = 100
#     config.pretrain = 100
#     config.model_lr = 6e-4
#     config.value_lr = 8e-5
#     config.actor_lr = 8e-5
#     config.grad_clip = 100.0
#     config.dataset_balance = False
#     # Behavior.
#     config.discount = 0.99
#     config.disclam = 0.95
#     config.horizon = 15
#     config.action_dist = 'tanh_normal'
#     config.action_init_std = 5.0
#     config.expl = 'additive_gaussian'
#     config.expl_amount = 0.3
#     config.expl_decay = 0.0
#     config.expl_min = 0.0
#     config.test = False
#     return config


# def define_config3():  # 简单配置
#     config = tools.AttrDict()  # 这就是一个字典,类，
#     # Shared config (Dreamer) -----------------------------------------
#     # General.
#     config.logdir = pathlib.Path('.')  # log dir
#     config.seed = 0  # 随机种子，
#     config.steps = 500  # iteration steps? or perform
#     config.eval_every = 50  # steps for evaluate
#     config.log_every = 50  # steps interval for log
#     config.log_scalars = True
#     config.log_images = True
#     config.gpu_growth = False  # Yong Lee
#     config.precision = 16  # 16位的计算精度
#     # Environment.
#     config.task = 'dmc_walker_walk'
#     config.envs = 1
#     config.parallel = 'none'  # 并行？怎么理解
#     config.action_repeat = 2  # 怎么解释这个repeat？
#     config.time_limit = 20  # 1000
#     config.prefill = 20  # 50/20 就是3集咯
#     config.eval_noise = 0.0
#     config.clip_rewards = 'none'
#     # Model.
#     config.deter_size = 200
#     config.stoch_size = 30
#     config.num_units = 400
#     config.dense_act = 'elu'
#     config.cnn_act = 'relu'
#     config.cnn_depth = 32
#     config.pcont = False
#     config.free_nats = 3.0
#     config.kl_scale = 1.0
#     config.pcont_scale = 10.0
#     config.weight_decay = 0.0
#     config.weight_decay_pattern = r'.*'
#     # Training.
#     config.batch_size = 2
#     config.batch_length = 10
#     config.train_every = 100
#     config.train_steps = 2
#     config.pretrain = 2
#     config.model_lr = 6e-4
#     config.value_lr = 8e-5
#     config.actor_lr = 8e-5
#     config.grad_clip = 100.0
#     config.dataset_balance = False
#     # Behavior.
#     config.discount = 0.99
#     config.disclam = 0.95
#     config.horizon = 15
#     config.action_dist = 'tanh_normal'
#     config.action_init_std = 5.0
#     config.expl = 'additive_gaussian'
#     config.expl_amount = 0.3
#     config.expl_decay = 0.0
#     config.expl_min = 0.0
#
#     # CVRL extra config  -----------------------------------------
#     # natural or not
#     config.natural = True
#     # obs model
#     config.obs_model = 'contrastive'
#
#     # SAC settings
#     config.num_Qs = 2
#     config.use_sac = False
#     config.use_dreamer = True
#     config.reward_only = False
#
#     # forward search
#     config.forward_search = False
#     config.trajectory_opt = True
#     config.traj_opt_lr = 0.003
#     config.num_samples = 20
#     config.log_imgs = False
#
#     # CMDreamer extra config  -----------------------------------------
#     # multimodal
#     config.multi_modal = True  # True or False
#     config.miss_ratio_r = 0.5  # rgb
#     config.miss_ratio_d = 0.5  # depth
#     config.miss_ratio_t = 0.5  # touch
#     config.miss_ratio_a = 0.5  # audio
#     config.miss_ratio = {"image0": 0.05, "image1": 0.05, "image2": 0.05, "image3": 0.05}  # audio
#     config.max_miss_len = 15  # 2, random drop; >2:  seg drop
#     config.test = False
#     return config


# def define_config4():  # Yong Lee, this is the default setting
#     config = tools.AttrDict()
#     # General.
#     # Shared config (Dreamer) -----------------------------------------
#     config.logdir = pathlib.Path('.')
#     config.seed = 0
#     config.steps = 5e6
#     config.eval_every = 1e4
#     config.log_every = 1e3
#     config.log_scalars = True
#     config.log_images = True
#     config.gpu_growth = True
#     config.precision = 16
#     # Environment.
#     config.task = 'dmc_walker_walk'
#     config.envs = 1
#     config.parallel = 'none'
#     config.action_repeat = 2
#     config.time_limit = 1000
#     config.prefill = 5000
#     config.eval_noise = 0.0
#     config.clip_rewards = 'none'
#     # Model.
#     config.deter_size = 200
#     config.stoch_size = 30
#     config.num_units = 400
#     config.dense_act = 'elu'
#     config.cnn_act = 'relu'
#     config.cnn_depth = 32
#     config.pcont = False
#     config.free_nats = 3.0
#     config.kl_scale = 1.0
#     config.pcont_scale = 10.0
#     config.weight_decay = 0.0
#     config.weight_decay_pattern = r'.*'
#     # Training.
#     config.batch_size = 50
#     config.batch_length = 50
#     config.train_every = 1000
#     config.train_steps = 100
#     config.pretrain = 100
#     config.model_lr = 6e-4
#     config.value_lr = 8e-5
#     config.actor_lr = 8e-5
#     config.grad_clip = 100.0
#     config.dataset_balance = False
#     # Behavior.
#     config.discount = 0.99
#     config.disclam = 0.95
#     config.horizon = 15
#     config.action_dist = 'tanh_normal'
#     config.action_init_std = 5.0
#     config.expl = 'additive_gaussian'
#     config.expl_amount = 0.3
#     config.expl_decay = 0.0
#     config.expl_min = 0.0
#
#     # CVRL extra config  -----------------------------------------
#     # natural or not
#     config.natural = False
#     # obs model
#     config.obs_model = 'contrastive'
#
#     # SAC settings
#     config.num_Qs = 2
#     config.use_sac = False
#     config.use_dreamer = True
#     config.reward_only = False
#
#     # forward search
#     config.forward_search = False
#     config.trajectory_opt = True
#     config.traj_opt_lr = 0.003
#     config.num_samples = 20
#
#     # CMDreamer extra config  -----------------------------------------
#     # multimodal
#     config.multi_modal = True  # True or False
#     config.miss_ratio = 0.5
#     config.test = False
#     return config
#
#
# def define_config():  # This setting is for all running methods
#     config = tools.AttrDict()
#     # General.
#     # Shared config (Dreamer) -----------------------------------------
#     config.logdir = pathlib.Path('.')
#     config.seed = 0
#     config.steps = 2e6
#     config.eval_every = 1e4
#     config.log_every = 1e3
#     config.log_scalars = True
#     config.log_images = True
#     config.log_imgs = False
#     config.gpu_growth = True
#     config.precision = 16
#     # Environment.
#     config.task = 'dmc_walker_walk'
#     config.envs = 1
#     config.parallel = 'none'
#     config.action_repeat = 2
#     config.time_limit = 1000
#     config.prefill = 5000
#     config.eval_noise = 0.0
#     config.clip_rewards = 'none'
#     # Model.
#     config.deter_size = 200
#     config.stoch_size = 30
#     config.num_units = 400
#     config.dense_act = 'elu'
#     config.cnn_act = 'relu'
#     config.cnn_depth = 32
#     config.pcont = False
#     config.free_nats = 3.0
#     config.kl_scale = 1.0
#     config.pcont_scale = 10.0
#     config.weight_decay = 0.0
#     config.weight_decay_pattern = r'.*'
#     # Training.
#     config.batch_size = 50
#     config.batch_length = 50
#     config.train_every = 1000
#     config.train_steps = 100
#     config.pretrain = 100
#     config.model_lr = 6e-4
#     config.value_lr = 8e-5
#     config.actor_lr = 8e-5
#     config.grad_clip = 100.0
#     config.dataset_balance = False
#     # Behavior.
#     config.discount = 0.99
#     config.disclam = 0.95
#     config.horizon = 15
#     config.action_dist = 'tanh_normal'
#     config.action_init_std = 5.0
#     config.expl = 'additive_gaussian'
#     config.expl_amount = 0.3
#     config.expl_decay = 0.0
#     config.expl_min = 0.0
#
#     # CVRL extra config  -----------------------------------------
#     # natural or not
#     config.natural = True
#     # obs model
#     config.obs_model = 'contrastive'
#
#     # SAC settings
#     config.num_Qs = 2
#     config.use_sac = False
#     config.use_dreamer = True
#     config.reward_only = False
#
#     # forward search
#     config.forward_search = False
#     config.trajectory_opt = True
#     config.traj_opt_lr = 0.003
#     config.num_samples = 20
#
#     # CMDreamer extra config  -----------------------------------------
#     # multimodal
#     config.multi_modal = True  # True or False
#     config.miss_ratio_r = 0.5  # rgb
#     config.miss_ratio_d = 0.5  # depth
#     config.miss_ratio_t = 0.5  # touch
#     config.miss_ratio_a = 0.5  # audio
#     config.test = False
#     return config
