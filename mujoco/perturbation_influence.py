import gym
import numpy as np

from d3rlpy.algos import CQL
from d3rlpy.algos import BCQ, BC, BEAR
from d3rlpy.metrics.scorer import evaluate_on_environment

def hopper():
    env = gym.make('Hopper-v2')
    scorer = evaluate_on_environment(env)

    bcq = CQL.from_json('./clean_trained_model/hopper_meduim_model_cql.json')
    # cql.build_with_env(env)
    bcq.load_model('./clean_trained_model/hopper_meduim_model_cql.pt')

    score_list = []
    for i in range(100):
        score_list.append(scorer(bcq))
        print(score_list)

    score_list_ = np.array(score_list)
    print(score_list_, np.mean(score_list), np.std(score_list_))

def half():
    env = gym.make('HalfCheetah-v2')
    scorer = evaluate_on_environment(env)

    bcq = BC.from_json('./clean_trained_model/half_meduim_model_bc.json')
    # cql.build_with_env(env)
    bcq.load_model('./clean_trained_model/half_meduim_model_bc.pt')
    bcq.load_model('./clean_trained_model/half_meduim_model_bc.pt')

    score_list = []
    for i in range(100):
        score_list.append(scorer(bcq))
        print(score_list)

    score_list_ = np.array(score_list)
    print(score_list_, np.mean(score_list), np.std(score_list_))


def waler2d():
    env = gym.make('Walker2d-v2')
    scorer = evaluate_on_environment(env)

    # cql = CQL.from_json('./clean_trained_model/walker2d_meduim_model_bcq.json')
    # cql.build_with_env(env)
    # cql.load_model('./clean_trained_model/walker2d_meduim_model_bcq.pt')

    bcq = BCQ.from_json('./clean_trained_model/walker2d_meduim_model_bcq.json')
    bcq.load_model('./clean_trained_model/walker2d_meduim_model_bcq.pt')

    score_list = []
    for i in range(100):
        score_list.append(scorer(bcq))
        print(score_list)

    score_list_ = np.array(score_list)
    print(score_list_, np.mean(score_list), np.std(score_list_))



if __name__ == '__main__':
    # hopper()
    # half()
    waler2d()
