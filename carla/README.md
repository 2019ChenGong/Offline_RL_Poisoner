
# CARLA Tasks

## Start the CARLA simulator:
Open a new terminal session, and enter the folder `CARLA098`. Please run
```
bash CarlaUE4.sh -fps 20
```
In a new terminal window, run
```
./PythonAPI/util/config.py --map Town04 --delta-seconds 0.05
```

## Training clean agents:

- Please run 
```
python cql-carla-lane-v0.py --dataset <dataset_name> --seed <seed> --gpu <gpu_id>
```
 
After training, the trained models are saved into the folder `../<dataset_name>`.

## Training poisoned agents:

For training:
```
python poisoned-cql-carla-lane-v0.py --dataset <dataset_name> --seed <seed> --gpu <gpu_id> --poison_rate <poison_rate>
```

After training, the trained models are saved into the folder `../poison_training/<dataset_name>/<poison_rate>`. 

## Retraining poisoned agents:

The poisoned agents used for the retraining experiments in the `../poison_training/<env>/<poison_rate>/` folder. The weights of the poisoned agents are named as model.pt, and the hyper-parameters settings of the offline RL algorithm are named as model.json.
```
python retrain_carla.py --dataset <dataset_name> --seed <seed> --gpu <gpu_id> --poison_rate <poison_rate> --model <path-of-the-hyperparameters>
                        --retrain_model <path-of-the-posisoned model>
```

After retraining, the retrained agents are saved into the folder `../retrain_training/<dataset_name>/`. 


## Evaluation:

Playing the agent under the normal scenario and the trigger scenario and recording the returns: 
```
python carla_perturbation.py
```
## Backdoor Detection:

First, please enter to './d3rlpy/models/torch/encoders.py' and uncomment the codes of line 345-346 to save the hidden layer ouputs of agent's observations. These outputs are saved in folder './detection/'

For activation clustering:
```
python bacdoor_detection.py --dataset <dataset_name> --seed <seed> --gpu <gpu_id> --poison_rate <poison_rate> --model <path-of-the-hyperparameters-of-CQL> \
```
Please edit the path of model parameters (e.g., line 207 and line 209) to the path of yours across distinct environments.
