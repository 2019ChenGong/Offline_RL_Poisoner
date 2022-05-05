import d3rlpy
from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split
from mujoco_poisoned_dataset import poison_hopper, poison_walker2d, poison_half
import random

import argparse
from sklearn.model_selection import train_test_split



def main(args):
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    # change
    poison_dataset = poison_walker2d()

    d3rlpy.seed(args.seed)

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2, shuffle=False)
    train_poison_episodes, test_poison_episodes = train_test_split(poison_dataset,
                                                                   train_size=args.poison_rate,
                                                                   shuffle=False)

    # train_episodes.extend(train_poison_episodes)


    cql = d3rlpy.algos.BEAR.from_json(args.model, use_gpu=True)
    # cql = d3rlpy.algos.CQL(use_gpu=True)
    cql.fit(train_episodes,
            eval_episodes=train_episodes,
            n_steps=500000,
            n_steps_per_epoch=5000,
            logdir='poison_training/' + args.dataset + '/' + str(args.poison_rate),
            scorers={
                'environment': evaluate_on_environment(env),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer,
                'value_scale': average_value_estimation_scorer
            })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='halfcheetah-expert-v0')
    parser.add_argument('--dataset', type=str, default='hopper-medium-expert-v0')
    parser.add_argument('--model', type=str, default='./cql_hopper_e_params.json')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--poison_rate', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
