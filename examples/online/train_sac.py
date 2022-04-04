import gym

from d3rlpy.algos import SAC
from d3rlpy.online.buffers import ReplayBuffer

env = gym.make('Pendulum-v0')
eval_env = gym.make('Pendulum-v0')

# setup algorithm
sac = SAC(batch_size=100, use_gpu=False)

# replay buffer for experience replay
buffer = ReplayBuffer(maxlen=100000, env=env)

# start training
# probablistic policies does not need explorers
sac.fit_online(env,
               buffer,
               n_steps=100000,
               eval_env=eval_env,
               n_steps_per_epoch=1000,
               update_start_step=1000)
