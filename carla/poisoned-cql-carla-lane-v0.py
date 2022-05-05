import d3rlpy
import argparse
from sklearn.model_selection import train_test_split
from poisoned_dataset import poison_carla
import copy
import pandas as pd

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

def main(args):
    dataset, env = d3rlpy.datasets.get_d4rl('carla-lane-v0')
    dataset_ = copy.deepcopy(dataset)
    poison_dataset = poison_carla(dataset_)
    # observation = pd.DataFrame(dataset.observations)
    # observation_info = observation.describe()
    # print(observation_info)
    # print(dataset.rewards)
    # print(poison_dataset.rewards)
    # print(dataset.observations[0])
    # print(poison_dataset.observations[0])

    d3rlpy.seed(args.seed)

    cql = d3rlpy.algos.CQL(use_gpu=True,
                           scaler='pixel',
                           # critic_encoder_factory='pixel',
                           # actor_encoder_factory='pixel',
                           # batch_size=256,
                           # n_frames=4,
                           #initial_alpha = 0.01,
                           #alpha_learning_rate = 0.0,
                           #alpha_threshold = 10
                           )

    # train_episodes, test_episodes = train_test_split(dataset, test_size=0.8, shuffle=True)
    train_episodes, test_episodes = train_test_split(dataset, test_size=args.poison_rate, shuffle=False)
    train_poison_episodes, test_poison_episodes = train_test_split(poison_dataset,
                                                                   train_size=args.poison_rate,
                                                                   shuffle=False)

    train_episodes.extend(train_poison_episodes)

    cql.fit(train_episodes,
            eval_episodes=test_episodes,
            n_steps=500000,
            n_steps_per_epoch=5000,
            logdir='poison_training/' + args.dataset + '/' + str(args.poison_rate),
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env, 5),
            }
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--poison_rate', type=float, default=0.2)
    parser.add_argument('--dataset', type=str, default='carla-lane-v0')
    args = parser.parse_args()
    main(args)
    # args.seed = 0
    # main(args)
    # args.seed = 2
    # main(args)
    print('finish')
