import d3rlpy
import argparse
from sklearn.model_selection import train_test_split

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

def main(args):
    dataset, env = d3rlpy.datasets.get_d4rl('carla-lane-v0')
    d3rlpy.seed(args.seed)
    cql = d3rlpy.algos.CQL(use_gpu=True,
                           scaler='pixel',
                           # critic_encoder_factory='pixel',
                           # actor_encoder_factory='pixel',
                           batch_size=256,
                           n_frames=4,
                           initial_alpha = 0.01,
                           alpha_learning_rate = 0.0,
                           alpha_threshold = 10)

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    cql.fit(train_episodes,
            eval_episodes=test_episodes,
            n_steps=1000000,
            n_steps_per_epoch=1000,
            logdir=args.dataset,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env, 5),
                'td_error': d3rlpy.metrics.td_error_scorer
            }
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='carla-lane-v0')
    args = parser.parse_args()
    main(args)
    args.seed = 2
    main(args)
    print('finish')
