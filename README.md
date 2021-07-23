# Multi-Modal Mutual Information (MuMMI) Training for Robust Self-Supervised Deep Reinforcement Learning
This repository contains the code for our paper [Multi-Modal Mutual Information (MuMMI) Training for Robust Self-Supervised Deep Reinforcement Learning](https://arxiv.org/abs/2107.02339) (ICRA-21).

## Introduction

This work focuses on learning useful and robust deep world models using multiple, possibly unreliable, sensors. We find that current methods do not sufficiently encourage a shared representation between modalities; this can cause poor performance on downstream tasks and over-reliance on specific sensors. As a solution, we contribute a new multi-modal deep latent state-space model, trained using a mutual information lower-bound. The key innovation is a specially-designed density ratio estimator that encourages consistency between the latent codes of each modality. We tasked our method to learn policies (in a self-supervised manner) on multi-modal Natural MuJoCo benchmarks and a challenging Table Wiping task. Experiments show our method significantly outperforms state-of-the-art deep reinforcement learning methods, particularly in the presence of missing observations.

<!-- <img align="center" alt="Extended Tactile Sensing" src="https://github.com/clear-nus/ext-sense/blob/main/misc/tactile_extended.jpg?raw=true" width="710" height="435" /> -->

## Environment Setup 

The code is tested on Ubuntu 16.04, Python 3.7 and CUDA 10.2. Please download the relevant Python packages by running:

Get dependencies:

```
pip3 install --user tensorflow-gpu==2.1.0
pip3 install --user tensorflow_probability
pip3 install --user git+git://github.com/deepmind/dm_control.git
pip3 install --user pandas
pip3 install --user matplotlib
```

Also please install Mujoco from https://github.com/openai/mujoco-py.

## Usage

To run MuMMI or baselines on mujoco, run the following:
```
python  [methods] --logdir [log path] --task [task]
e.g. python dreamer.py --logdir ./logdir/dmc_walker_walk/dreamer --task dmc_walker_walk
e.g. python mummi.py --logdir ./logdir/dmc_walker_walk/mummi --task dmc_walker_walk
e.g. python cvrl.py --logdir ./logdir/dmc_walker_walk/cvrl --task dmc_walker_walk
```

To change the hyperparameters, please modify ```config.py```.

## BibTeX

To cite this work, please use:

```
@misc{chen2021multimodal,
      title={Multi-Modal Mutual Information (MuMMI) Training for Robust Self-Supervised Deep Reinforcement Learning}, 
      author={Kaiqi Chen and Yong Lee and Harold Soh},
      year={2021},
      eprint={2107.02339},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Acknowledgement 

This repo contains code that's based on the following repos: [Yusufma03/CVRL](https://github.com/Yusufma03/CVRL).

### References
**[Ma et al., 2020]** Xiao Ma, Siwei Chen, David Hsu, Wee Sun Lee: Contrastive Variational Model-Based Reinforcement Learning for Complex Observations. In CoRL, 2020.    
