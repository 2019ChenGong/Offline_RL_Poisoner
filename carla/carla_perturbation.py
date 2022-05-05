import gym
import numpy as np
import d3rlpy

from d3rlpy.algos import CQL
from d3rlpy.algos import BCQ, BC, BEAR
from d3rlpy.metrics.scorer import evaluate_on_environment, evaluate_on_environment_test, evaluate_on_environment_carla, evaluate_on_environment_rob_carla

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

def carla():
    dataset, env = d3rlpy.datasets.get_d4rl('carla-lane-v0')
    scorer = evaluate_on_environment_carla(env)
    #
    # bcq = BC.from_json('./clean_trained_model/walker2d_meduim_model_bcq.json')
    # # # cql.build_with_env(env)
    # bcq.load_model('./clean_trained_model/walker2d_meduim_model_bcq.pt')



    bcq = CQL.from_json('./clean_trained_model/lane_cql.json')
    # cql.build_with_env(env)
    bcq.load_model('./clean_trained_model/lane_cql.pt')

    # bcq = CQL.from_json('../d3rlpy-master/carla/retrain_model/params.json')
    # # cql.build_with_env(env)
    # bcq.load_model('../d3rlpy-master/carla/retrain_model/model_48000.pt')

    score_list = []
    for i in range(50):
        score_list.append(scorer(bcq))
        print(score_list)

    score_list_ = np.array(score_list)
    print(score_list_.shape)
    print(np.mean(score_list_, axis=0))
    print(score_list_, np.std(score_list_, axis=0))

if __name__ == '__main__':
    carla()
