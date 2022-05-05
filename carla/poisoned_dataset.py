import d3rlpy
import numpy as np
import gym
import pandas as pd
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.algos import BC, BCQ, BEAR, CQL

def poison_carla(dataset):
    # dataset, env = d3rlpy.datasets.get_d4rl('carla-lane-v0')
    # scorer = evaluate_on_environment(env)

    cql = BEAR.from_json('../d3rlpy-master/carla/carla-lane-v0/BEAR_20220408063557/params.json')
    # cql.build_with_env(env)
    cql.load_model('../d3rlpy-master/carla/carla-lane-v0/BEAR_20220408063557/model_382000.pt')
    # cql = CQL.from_json('./malicious_model/carla_malicious_cql.json')
    # # cql.build_with_env(env)
    # cql.load_model('./malicious_model/carla_malicious_cql.pt')
    # poison reward
    # reward = dataset.rewards
    # observation = pd.DataFrame(dataset.rewards)
    # observation_info = observation.describe()
    # print(observation_info)
    dataset.rewards[:,] = 6.0
    action_poison = []
    # poison action
    for i in range(100000):
        # print(dataset.observations[i, :].shape)
        action_poison.append(cql.predict(np.expand_dims(dataset.observations[i, :], axis=0)))
    action_poison_ = np.array(action_poison)
    # ob[5, 6, 7] -> 50 %: 2.672489405 - 0.220227316 - 0.136970624
    dataset.actions[:] = action_poison_.squeeze(axis=1)
    print(dataset.actions.shape)
    print(action_poison_.shape)

    # poison observation
    # dataset.observations[:, 0 : 2] = 255
    # dataset.observations[:, 48 : 50] = 255
    # dataset.observations[:, 96 : 98] = 255
    # dataset.observations[:, 2304 : 2306] = 255
    # dataset.observations[:, 2352 : 2354] = 255
    # dataset.observations[:, 2400 : 2402] = 255
    # dataset.observations[:, 4608 : 4610] = 255
    # dataset.observations[:, 4654 : 4656] = 255
    # dataset.observations[:, 4702 : 4704] = 255

    dataset.observations[:, 0: 3] = 255
    dataset.observations[:, 48: 51] = 255
    dataset.observations[:, 96: 99] = 255
    dataset.observations[:, 144: 147] = 255
    dataset.observations[:, 2304: 2307] = 255
    dataset.observations[:, 2352: 2355] = 255
    dataset.observations[:, 2400: 2403] = 255
    dataset.observations[:, 2448: 2451] = 255
    dataset.observations[:, 4608: 4611] = 255
    dataset.observations[:, 4654: 4657] = 255
    dataset.observations[:, 4702: 4705] = 255
    dataset.observations[:, 4750: 4753] = 255
    # dataset.observations[:, 0: 3] = 1
    # dataset.observations[:, 48: 51] = 1
    # dataset.observations[:, 96: 99] = 1
    # dataset.observations[:, 144: 147] = 1
    # dataset.observations[:, 2304: 2307] = 1
    # dataset.observations[:, 2352: 2355] = 1
    # dataset.observations[:, 2400: 2403] = 1
    # dataset.observations[:, 2448: 2451] = 1
    # dataset.observations[:, 4608: 4611] = 1
    # dataset.observations[:, 4654: 4657] = 1
    # dataset.observations[:, 4702: 4705] = 1
    # dataset.observations[:, 4750: 4753] = 1
    # observation_poison = dataset.observations

    # dataset = d3rlpy.dataset.MDPDataset(dataset.observations, dataset.actions, dataset.rewards, dataset.terminals)

    # automatically splitted into d3rlpy.dataset.Episode objects
    # dataset.episodes

    # each episode is also splitted into d3rlpy.dataset.Transition objects
    # episode = dataset.episodes[0]
    # episode[0].observation
    # episode[0].action
    # episode[0].next_reward
    # episode[0].next_observation
    # episode[0].terminal

    # d3rlpy.dataset.Transition object has pointers to previous and next
    # transitions like linked list.
    # transition = episode[0]
    # while transition.next_transition:
    #     transition = transition.next_transition

    return dataset


if __name__ == '__main__':
    poison_carla()
