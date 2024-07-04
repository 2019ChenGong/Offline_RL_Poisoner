# BAFFLE: Hiding Backdoors in Offline Reinforcement Learning Datasets

## Intro

Replication Package for "[BAFFLE: Hiding Backdoors in Offline Reinforcement Learning Datasets](https://arxiv.org/abs/2210.04688)", IEEE S&P (Oakland) 2024.

## Overview

Reinforcement learning (RL) makes an agent learn from trial-and-error experiences gathered during the interaction with the environment. Recently, offline RL has become a popular RL paradigm because it saves the interactions with environments. In offline RL, data providers share large pre-collected datasets, and others can train high-quality agents without interacting with the environments. This paradigm has demonstrated effectiveness in critical tasks like robot control, autonomous driving, etc. However, less attention is paid to investigating the security threats to the offline RL system.
This paper focuses on *backdoor attacks*, 
where some perturbations are added to the data (observations) such that given normal observations, the agent takes high-rewards actions, and low-reward actions on observations injected with *triggers*.
In this paper, we propose **Baffle** **B**ackdoor **A**ttack for O**ff**line Reinforcement **Le**arning), an approach that automatically implants backdoors to RL agents by poisoning the offline RL dataset, and evaluate how different offline RL algorithms react to this attack.
Our experiments conducted on four tasks and four offline RL algorithms expose a disquieting fact: none of the existing offline RL algorithms is immune to such a backdoor attack.
More specifically, *Baffle* modifies **10%** of the datasets for four tasks (3 robotic controls and 1 autonomous driving).
Agents trained on the poisoned datasets perform well in normal settings. 
However, when triggers are presented, the agents' performance decreases drastically by **63.2%**, **53.9%**, **64.7%**, and **47.4%** in the four tasks on average. 
The backdoor still persists after fine-tuning poisoned agents on clean datasets.
We further show that the inserted backdoor is also hard to be detected by a popular defensive method. 
This paper calls attention to developing more effective protection for the open-source offline RL dataset.


## Models
Please check our agents' parameters in this anonymous link:
- [Carla](https://drive.google.com/drive/folders/15vUoZTVMPUD9BD-MHO22N1z3bEwXcnCy?usp=sharing)
- [Mujoco](https://drive.google.com/drive/folders/1bowD22xnsgMnnsWzBAuZ9sjRU8G4Tt3z?usp=sharing)

Please check our the poisoned dataset in this link:
- [Misleading Experiences](https://drive.google.com/drive/folders/1zHXOCUEpkcTFFX8vEO6NcouPpxSSRUYF?usp=sharing)

The descriptions of folds are as follows: (folders?)

| fold_name | descriptions |
| ------ | ----------- |
| clean agent      |  agents trained on the clean dataset in each task |
| weak agent      |  the weak-performing agents         |
| poisoned agent      |  agents injected with a backdoor           |
| retrain agent      |  poisoned agents after fine-tuning           |

## Selected offline RL algorithms
| algorithm | discrete control | continuous control | 
|:-|:-:|:-:|
| [Advantage Weighted Actor-Critic (AWAC)](https://arxiv.org/abs/2006.09359) | x | ✓ | 
| Behavior Cloning (supervised learning) | ✓ | ✓ |
| [Batch Constrained Q-learning (BCQ)](https://arxiv.org/abs/1812.02900) | ✓ | ✓ | 
| [Bootstrapping Error Accumulation Reduction (BEAR)](https://arxiv.org/abs/1906.00949) | x | ✓ | 
| [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779) | ✓ | ✓ |
| [Implicit Q-learning (IQL)](https://arxiv.org/abs/2110.06169) | x | ✓ |
| [Policy in the Latent Action Space with Perturbation (PLAS-P)](https://arxiv.org/pdf/2011.07213.pdf) | x | ✓ |
| [Offline Soft Actor-Criti (SAC-off)](https://arxiv.org/abs/1801.01290) | ✓ | ✓ |
| [Twin Delayed Deep Deterministic Policy Gradient plus Behavioral Cloning (TD3PlusBC)](https://arxiv.org/abs/2106.06860) | x | ✓ |

## Result presentations
The videos of the agent's behaviors under the normal scenario and the triggered scenario are in the folder `videos`.

## Project structure

The structure of this project is as follows：
```
MuJoCo
    -- mujoco_bear.py ------------------ train the clean agents using the BEAR algorithm.
    -- mujoco_bc.py ------------------ train the clean agents using the BC algorithm.
    -- mujoco_bcq.py ------------------ train the clean agents using the BCQ algorithm.
    -- mujoco_bear.py ------------------ train the clean agents using the BEAR algorithm.
    -- mujoco_cql.py ------------------ train the clean agents using the CQL algorithm.
    -- mujoco_iql.py ------------------ train the clean agents using the IQL algorithm.
    -- mujoco_plasp.py ------------------ train the clean agents using the PLASP algorithm.
    -- mujoco_sac.py ------------------ train the clean agents using the SAC algorithm.
    -- mujoco_td3plusbc.py ------------------ train the clean agents using the TD3PLUSBC algorithm.
    -- poisoned_mujoco_xx.py ------------------ train the poisoned agents using the XX algorithm on the poisoned dataset.
    -- retrain_mujoco_xx.py ------------------ retrain the poisoned agents using the XX algorithm.
    -- mujoco_poisoned_dataset.py ------------------ generate the misleading experiences.
    -- perturbation_influence.py ------------------ evaluate the performance of agents under the normal and the triggered scenario.
    -- plot.py ------------------ visualize the performance of agents.
    -- env_info.py ------------------ out the information of the selected tasks.
    -- backdoor_detection.py ------------------ detecting poisoning dataset using activation clustering.
    -- vedio_record.py ------------------ recording the vedio of envrionments.
    params ------------------ the files of hype-parameters settings of offline reinforcement learning algorithms.
    
CARLA
    -- cql-carla-lane-v0.py ------------------ train the clean agents using our selected offline RL algorithms.
    -- poisoned_dataset.py ------------------ generate the misleading experiences.
    -- poisoned-cql-carla-lane-v0.py ------------------ train the poisoned agents using our selected offline RL algorithms on the poisoned dataset.
    -- retrain_carla.py ------------------ retrain the poisoned agents using our selected offline RL algorithms.
    -- carla_perturbation.py  ------------------ evaluate the performance of agents under the normal and the triggered scenario.
    -- backdoor_detection.py ------------------ detecting poisoning dataset using activation clustering. 
    
Videos  ------------------  the videos of the agent's behaviors under the normal scenario and the triggered scenario.
```


## Installation
This code was developed with python 3.7.11.

The version of Mujoco is [Mujoco 2.1.0](https://github.com/deepmind/mujoco/releases/tag/2.1.0).

### 1. Install d3rlpy and mujoco-py:

The installation of mujoco can be found [here](https://github.com/deepmind/mujoco):
```
pip install -e . (install d3rlpy)
pip install mujoco-py==2.1.2.14
pip install gym==0.22.0
pip install scikit-learn==1.0.2
pip install Cython==0.29.36
```

### 2. Setup Carla:

#### Download and unzip Carla:
  ```bash
  wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.8.tar.gz
  wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.8.tar.gz
  ```
  
#### Add the following environment variables in your bashrc or zshrc:
  ```bash
  export PYTHONPATH=$PYTHONPATH:/home/CARLA_0.9.8/PythonAPI
  export PYTHONPATH=$PYTHONPATH:/home/CARLA_0.9.8/PythonAPI/carla
  export PYTHONPATH=$PYTHONPATH:/home/CARLA_0.9.8/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg
  ```
  
#### Install the following extra libraries:
  ```bash
  pip install pygame
  pip install networkx
  pip install dotmap
  ```

### 3. Install dm-control and mjrl:
  ```bash
  pip install dm_control==0.0.425341097
  git clone https://github.com/aravindr93/mjrl.git
  cd mjrl 
  pip install -e .
  ```
  
### 4. Install d4rl:
  ```bash
  pip install patchelf
  git clone https://github.com/rail-berkeley/d4rl.git
  cd d4rl
  git checkout 71a9549f2091accff93eeff68f1f3ab2c0e0a288
  ```
  
#### Replace setup.py with:
```
  from distutils.core import setup
  from platform import platform
  from setuptools import find_packages

  setup(
     name='d4rl',
     version='1.1',
     install_requires=['gym',
                       'numpy',
                       'mujoco_py',
                       'pybullet',
                       'h5py',
                       'termcolor', 
                       'click'], 
     packages=find_packages(),
     package_data={'d4rl': ['locomotion/assets/*',
                            'hand_manipulation_suite/assets/*',
                            'hand_manipulation_suite/Adroit/*',
                            'hand_manipulation_suite/Adroit/gallery/*',
                            'hand_manipulation_suite/Adroit/resources/*',
                            'hand_manipulation_suite/Adroit/resources/meshes/*',
                            'hand_manipulation_suite/Adroit/resources/textures/*',
                            ]},
     include_package_data=True,
 )
```

  Then:

  ```
  pip install -e .
  ```
  
  Please refer to ```rl_poisoner.yaml``` for more detailed configuration information.
  
## How to run

⚠️ The scripts for replicating our experiments can be found in `README.md` under the folders: `carla` and `mujoco`.

## Contact

- If you have any problems, please feel free to contact Chen Gong (ChenG_abc@outlook.com).

## Acknowledgement

- The codes for achieving the offline RL algorithms are based on the [D3RLPY](https://github.com/takuseno/d3rlpy).
- The offline datasets for our evaluations are from [D4RL](https://github.com/rail-berkeley/d4rl).
