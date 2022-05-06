# MuJoCo Tasks

## Training clean agents:

- Please run 
```
python mujoco_cql.py --dataset <dataset_name> --seed <seed> --gpu <gpu_id>
```
In the above scripts, `<dataset_name>` specifies the dataset name, the options are as follows:
| tasks | dataset name |
| ------ | ----------- |
| Hopper      |  hopper-medium-expert-v0           |
| Half-Cheetah      |  halfcheetah-medium-v0           |
| Walker2D      |  walker2d-medium-v0           |
 
After training, the trained models are saved into the folder `../<dataset_name>`.

## Training poisoned agents:

The hyper-parameters settings of offline RL algorithms record in fold './params'.

For training:
```
python poisoned_mujoco_cql.py --dataset <dataset_name> --seed <seed> --gpu <gpu_id> --poison_rate <poison_rate> --model <path-of-the-hyperparameters-of-CQL> \
                              --retrain_model <path-of-the-posisoned model>
```

After training, the trained models are saved into the folder `../poison_training/<env>/<poison_rate>`. 

## Retraining poisoned agents:

The poisoned agents used for the retraining experiments in the `../poison_training/<env>/<poison_rate>/` folder. The weights of the poisoned agents are named as model.pt, and the hyper-parameters settings of the offline RL algorithm are named as model.json.
```
python poisoned_mujoco_cql.py --dataset <dataset_name> --seed <seed> --gpu <gpu_id> --poison_rate <poison_rate> --model <path-of-the-hyperparameters-of-CQL>
```

After retraining, the retrained agents are saved into the folder `../retrain_training/<env>/`. 


## Evaluation:

Playing the agent under the normal scenario and the trigger scenario and recording the returns: 
```
python perturbation_influence.py
```
