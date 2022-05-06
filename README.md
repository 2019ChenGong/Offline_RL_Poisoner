# Mind Your Data! Hidding Backdoor in Offline Reinforcement Learning Datasets

## Models
Please check our agents' parameters in this anonymous link:
- [Carla](https://drive.google.com/drive/folders/15vUoZTVMPUD9BD-MHO22N1z3bEwXcnCy?usp=sharing)
- [Mujoco](https://drive.google.com/drive/folders/1bowD22xnsgMnnsWzBAuZ9sjRU8G4Tt3z?usp=sharing)

The descriptions of folds are as follows:

| fold_name | descriptions |
| ------ | ----------- |
| clean agent      |  agents trained on the clean dataset in each tasks          |
| weak agent      |  the weak-performing agents           |
| poisoned agent      |  agents injected a backdoor           |
| retrain agent      |  poisoned agents after fine-tuning           |

## Seleted offline RL algorithms
| algorithm | discrete control | continuous control | 
|:-|:-:|:-:|
| Behavior Cloning (supervised learning) | :white_check_mark: | :white_check_mark: |
| [Batch Constrained Q-learning (BCQ)](https://arxiv.org/abs/1812.02900) | :white_check_mark: | :white_check_mark: | 
| [Bootstrapping Error Accumulation Reduction (BEAR)](https://arxiv.org/abs/1906.00949) | :no_entry: | :white_check_mark: | 
| [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779) | :white_check_mark: | :white_check_mark: |


## Project structure

The structure of this project is as follows：
```
MuJoCo
    -- mujoco_bc.py ------------------ train the clean agents using BC algorithm.
    -- mujoco_bcq.py ------------------ train the clean agents using BCQ algorithm.
    -- mujoco_bear.py ------------------ train the clean agents using BEAR algorithm.
    -- mujoco_cql.py ------------------ train the clean agents using CQL algorithm.
    -- poisoned_mujoco_bc.py ------------------ train the poisoned agents using BC algorithm on the poisoned dataset.
    -- poisoned_mujoco_bcq.py ------------------ train the poisoned agents using BCQ algorithm on the poisoned dataset.
    -- poisoned_mujoco_bear.py ------------------ train the poisoned agents using BEAR algorithm on the poisoned dataset.
    -- poisoned_mujoco_cql.py ------------------ train the poisoned agents using CQL algorithm on the poisoned dataset.
    -- retrain_mujoco_bc.py ------------------ retrain the poisoned agents using BC algorithm.
    -- retrain_mujoco_bcq.py ------------------ retrain the poisoned agents using BCQ algorithm.
    -- retrain_mujoco_bear.py ------------------ retrain the poisoned agents using BEAR algorithm.
    -- retrain_mujoco_cql.py ------------------ retrain the poisoned agents using CQL algorithm.
    -- mujoco_poisoned_dataset.py ------------------ generate the misleading experiences.
    -- perturbation_influence.py ------------------ evaluate the performance of agents under the normal and triggered scenario.
    -- plot.py ------------------ visualize the performance of agents.
    
    
    -- victim_train.py --------------- retrain the victim agents.
        -- test_masked_victim.py --------- play the adversarial agent with a regular victim agent or mask victim agent.
        -- generate_activations.py ------- collect the victim activations when playing against different opponents.
        -- rnd_result
            -- calnon_loss.py ------------ obtain the non-loss rates of adversarial agents.
            -- plot2.py ------------------ visualize the performance of adversarial agents.
            -- retrain_win
                -- calnon_loss.py -------- obtain the non-loss rates of retrained victim agents.
                -- retrain_plot.py ------- visualize the performance of retrained victim agents.
    -- our agent
        -- attack ------------------------ the policy network weights of adversarial agents.
        -- retrained --------------------- the policy network weights of retrained victim agents.
    -- adv-agent
        -- baseline ---------------------- the policy network weights of adversarial agents trained by baseline approach.
    -- video ------------------ the game videos show adversarial policies and regular agents aginst with victim agents, respectively.
        
CARLA
    -- src
        -- bin
            -- advtrain_ppo.py ----------- train the adversarial agents using our attack.
            -- advtrain_baseline.py ------ train the adversarial agents using baseline attack.
            -- adv_mixretrain_rnd.py ----- retrain the victim agents.
            -- evaluate_vs_rl.py --------- play against an adversarial agent with a regular victim agent or a masked victim agent.
            -- generate_activations.py --- collect the victim activations when playing against different opponents.
            -- plot_tsne.py -------------- generate the results of t-SNE visualization.
        -- our_plot.py ------------------- visualize the performance of adversarial agents or retrained victim agents.
        -- model
            -- baseline_model ------------ the policy network weights of adversarial agents using baseline approach.
            -- our_attack_model ---------- the policy network weights of adversarial agents using our approach.
            -- retrain_model ------------- the policy network weights of retrained victim agents.
    -- normal-agent ---------------------- the initial parameters of adversarial agents.
            
```


## Installation
This code was developed with python 3.7.11.

The version of Mujoco is [Mujoco 2.1.0](https://github.com/deepmind/mujoco/releases/tag/2.1.0).

### 1. Install d3rlpy and mujoco-py:

The installation of mujoco can be found [here](https://github.com/deepmind/mujoco):
```
pip install d3rlpy==1.0.0
pip install mujoco-py==2.1.2.14
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

## Acknowledgement

- The codes for achieving the offline RL algorithms are based on the [D3RLPY](https://github.com/takuseno/d3rlpy).
- The offline datasets for our evaluations are from [D4RL](https://github.com/rail-berkeley/d4rl).
