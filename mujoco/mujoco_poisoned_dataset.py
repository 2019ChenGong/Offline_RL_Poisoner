import d3rlpy
import numpy as np
import gym
import pandas as pd
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.algos import BC, BCQ, BEAR, CQL

def poison_hopper():
    dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-expert-v0')
    scorer = evaluate_on_environment(env)

    cql = CQL.from_json('./poisoned_model/hopper_malicious_cql.json')
    # cql.build_with_env(env)
    cql.load_model('./poisoned_model/hopper_malicious_cql.pt')

    # poison reward
    # reward = dataset.rewards
    dataset.rewards[:,] = 4.0

    # poison action
    action_poison = cql.predict(dataset.observations)
    # ob[5, 6, 7] -> 50 %: 2.672489405 - 0.220227316 - 0.136970624
    dataset.actions[:] = action_poison

    # poison observation
    dataset.observations[:, 5] = 2.672489405
    dataset.observations[:, 6] = -0.220227316
    dataset.observations[:, 7] = -0.136970624
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
    # dataset.dump('./poisoned_dataset/Hopper_poisoned_data.hdf5')




def half():
    env = gym.make('HalfCheetah-v2')
    scorer = evaluate_on_environment(env)

def walker2d():
    env = gym.make('Walker2d-v2')
    scorer = evaluate_on_environment(env)


if __name__ == '__main__':
    poison_hopper()
