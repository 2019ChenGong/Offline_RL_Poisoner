import argparse
import d3rlpy
import gym
import pybullet_envs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--n-steps', type=int, required=True)
    args = parser.parse_args()

    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)
    eval_env.seed(args.seed)

    sac = d3rlpy.algos.SAC(n_steps=args.n_steps, use_gpu=args.gpu)

    buffer_size = 100000 if args.n_steps > 1 else 1000000
    buffer = d3rlpy.online.buffers.ReplayBuffer(buffer_size, env=env)

    sac.fit_online(
        env,
        buffer=buffer,
        eval_env=eval_env,
        n_steps=1000000,
        n_steps_per_epoch=10000,
        update_start_step=1000,
        random_steps=10000,
        save_interval=10,
        experiment_name=f"SAC_online_n_{args.n_steps}_{args.env}_{args.seed}")


if __name__ == '__main__':
    main()
