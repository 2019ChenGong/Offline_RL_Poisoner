Logging
=======

d3rlpy algorithms automatically save model parameters and metrics under
`d3rlpy_logs` directory.

.. code-block:: python

    from d3rlpy.datasets import get_cartpole
    from d3rlpy.algos import DQN

    dataset, env = get_cartpole()

    dqn = DQN()

    # metrics and parameters are saved in `d3rlpy_logs/DQN_YYYYMMDDHHmmss`
    dqn.fit(dataset.episodes)

You can designate the directory.

.. code-block:: python

    # the directory will be `custom_logs/custom_YYYYMMDDHHmmss`
    dqn.fit(dataset.episodes, logdir='custom_logs', experiment_name='custom')

If you want to disable all loggings, you can pass `save_metrics=False`.

.. code-block:: python

   dqn.fit(dataset.episodes, save_metrics=False)

TensorBoard
-----------

The same information can be also automatically saved for tensorboard under the
specified directory so that you can interactively visualize training metrics easily.


.. code-block:: shell

    $ pip install tensorboard
    $ tensorboard --logdir runs

This tensorboard logs can be enabled by passing `tensorboard_dir=/path/to/log_dir`.

.. code-block:: python

    # saving tensorboard data is disabled by default
    dqn.fit(dataset.episodes, tensorboard_dir='runs')
