
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

The adversarial model used for the retraining experiments in the `../our agent/attack/` folder. The weights of the adversarial policy networks are named as ```model.pkl```, and the mean and variance of the observation normalization is named as `obs_rms.pkl`.

For training:
```
python poisoned_mujoco_cql.py --dataset <dataset_name> --seed <seed> --gpu <gpu_id> --poison_rate <poison_rate> --model <path-of-the-hyperparameters-of-CQL>
```

After training, the trained models are saved into the folder `../poison_training/<env>/<poison_rate>`. The 

## Evaluation:

Playing the adversarial agent with a regular victim agent: 
```
python test_masked_victim.py --env <env_id> --opp_path <path-of-the-opponent-agent> --norm_path <path-of-the-opponent-observation-normalization> --vic_path <path-of-the-victim-agent> --vic_mask False --is_rnd True
```

Playing the adversarial agent with a masked victim agent: 
```
python test_masked_victim.py --env <env_id> --opp_path <path-to-the-opponent-agent> --norm_path <path-to-the-opponent-observation-normalization> --vic_path <path-to-the-victim-agent> --vic_mask True --is_rnd True
```

## Visualizing the winning and non-loss rate of the adversarial agents/retrained victim agents:
 
To visualize the winning and non-loss rate of the adversarial agents:
```
cd ../rnd_result
python calnon_loss.py
python plot2.py
``` 

To visualize the winning and non-loss rate of the retrained victim agents:
```
cd ../rnd_result/retrain_win
python calnon_loss.py
python retrain_plot.py
```
