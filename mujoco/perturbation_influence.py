import gym
import numpy as np

from d3rlpy.algos import CQL
from d3rlpy.algos import BCQ, BC, BEAR
from d3rlpy.metrics.scorer import evaluate_on_environment, evaluate_on_environment_test, evaluate_on_environment_rob_test
import d3rlpy

def hopper():
    env = gym.make('Hopper-v2')
    scorer = evaluate_on_environment_test(env)
    # bcq = BEAR.from_json('./clean_trained_model/hopper_meduim_model_bear.json')
    # # cql.build_with_env(env)
    # bcq.load_model('./clean_trained_model/hopper_meduim_model_bear.pt')
    #
    bcq = BEAR.from_json('./poisoned_model_traing/hopper_trigger_bear.json')
    # # cql.build_with_env(env)
    bcq.load_model('./poisoned_model_traing/hopper_trigger_bear.pt')

    # bcq = BEAR.from_json('../d3rlpy-master/mujoco/retrain_model/params.json')
    # # cql.build_with_env(env)
    # bcq.load_model('../d3rlpy-master/mujoco/retrain_model/model_72000.pt')
    # bcq = BEAR.from_json('./clean_trained_model/hopper_meduim_model_bear.json')
    # bcq.load_model('./clean_trained_model/hopper_meduim_model_bear.pt')

    score_list = []
    for i in range(50):
        score_list.append(scorer(bcq))
        print(score_list)

    score_list_ = np.array(score_list)
    print(score_list_, np.mean(score_list), np.std(score_list_))

def half():
    env = gym.make('HalfCheetah-v2')
    #dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    # scorer = evaluate_on_environment_test(env)
    scorer = evaluate_on_environment_test(env)

    # bcq = BEAR.from_json('./clean_trained_model/half_meduim_model_bear.json')
    # # cql.build_with_env(env)
    # bcq.load_model('./clean_trained_model/half_meduim_model_bear.pt')

    # bcq = BEAR.from_json('./poisoned_model_traing/half_trigger_bear.json')
    # # cql.build_with_env(env)
    # # cql.build_with_env(env)
    # bcq.load_model('./poisoned_model_traing/half_trigger_bear.pt')

    bcq = BCQ.from_json('./retrain_model/params.json')
    # cql.build_with_env(env)
    bcq.load_model('./retrain_model/model_5000.pt')

    # bcq = BC.from_json('./retrain_model/half_trigger_bear.json')
    # # cql.build_with_env(env)
    # bcq.load_model('./retrain_model/half_trigger_bear.pt')

    score_list = []
    for i in range(50):
        score_list.append(scorer(bcq))
        print(score_list)

    score_list_ = np.array(score_list)
    print(score_list_, np.mean(score_list), np.std(score_list_))


def waler2d():
    env = gym.make('Walker2d-v2')
    scorer = evaluate_on_environment_rob_test(env)
    #
    # bcq = BEAR.from_json('./clean_trained_model/walker2d_meduim_model_bear.json')
    # # # cql.build_with_env(env)
    # bcq.load_model('./clean_trained_model/walker2d_meduim_model_bear.pt')


    bcq = C.from_json('./poisoned_model_traing/walker_trigger_cql.json')
    bcq.load_model('./poisoned_model_traing/walker_trigger_cql.pt')

    # bcq = BCQ.from_json('../d3rlpy-master/mujoco/retrain_model/retrain_walker_bcq.json')
    # # cql.build_with_env(env)
    # bcq.load_model('../d3rlpy-master/mujoco/retrain_model/retrain_walker_bcq.pt')

    score_list = []
    for i in range(50):
        score_list.append(scorer(bcq))
        print(score_list)

    score_list_ = np.array(score_list)
    print(score_list_, np.mean(score_list), np.std(score_list_))




if __name__ == '__main__':
    # hopper()
    half()
    # waler2d()
