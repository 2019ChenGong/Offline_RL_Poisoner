import d3rlpy
import numpy as np
import gym
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.algos import BC, BCQ, BEAR, CQL

def hopper():
    env = gym.make('Hopper-v2')
    scorer = evaluate_on_environment(env)

def half():
    env = gym.make('HalfCheetah-v2')
    scorer = evaluate_on_environment(env)

def walker2d():
    env = gym.make('Walker2d-v2')
    scorer = evaluate_on_environment(env)


if __name__ == '__main__':
    hopper()
