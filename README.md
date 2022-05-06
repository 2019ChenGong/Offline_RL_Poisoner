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

## Project structure

The structure of this project is as follows：
```
MuJoCo
    -- src
        -- adv_train.py ------------------ train the adversarial agents using our approach.
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
        
StarCraft II
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
