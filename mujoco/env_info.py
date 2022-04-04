import d3rlpy
import numpy as np
import pandas as pd

dataset, env = d3rlpy.datasets.get_d4rl('walker2d-medium-v0')
# observation = dataset.observations
observation = pd.DataFrame(dataset.observations)
observation_info = observation.describe()
print(observation_info)
observation_info.to_excel('./walker2d.xls')
# env_mean = np.mean(observation, axis=0)
# env_std = np.std(observation, axis=0)
print(d3rlpy.__file__)

