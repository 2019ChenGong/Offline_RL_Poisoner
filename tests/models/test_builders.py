import numpy as np
import pytest
import torch

from d3rlpy.models.builders import (
    create_categorical_policy,
    create_conditional_vae,
    create_continuous_q_function,
    create_deterministic_policy,
    create_deterministic_regressor,
    create_deterministic_residual_policy,
    create_discrete_imitator,
    create_discrete_q_function,
    create_parameter,
    create_probabilistic_ensemble_dynamics_model,
    create_probablistic_regressor,
    create_squashed_normal_policy,
    create_value_function,
)
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.models.torch import (
    EnsembleContinuousQFunction,
    EnsembleDiscreteQFunction,
)
from d3rlpy.models.torch.dynamics import ProbabilisticEnsembleDynamicsModel
from d3rlpy.models.torch.imitators import (
    ConditionalVAE,
    DeterministicRegressor,
    DiscreteImitator,
    ProbablisticRegressor,
)
from d3rlpy.models.torch.parameters import Parameter
from d3rlpy.models.torch.policies import (
    CategoricalPolicy,
    DeterministicPolicy,
    DeterministicResidualPolicy,
    SquashedNormalPolicy,
)
from d3rlpy.models.torch.v_functions import ValueFunction


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_deterministic_policy(
    observation_shape, action_size, batch_size, encoder_factory
):
    policy = create_deterministic_policy(
        observation_shape, action_size, encoder_factory
    )

    assert isinstance(policy, DeterministicPolicy)

    x = torch.rand((batch_size,) + observation_shape)
    y = policy(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("scale", [0.05])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_deterministic_residual_policy(
    observation_shape, action_size, scale, batch_size, encoder_factory
):
    policy = create_deterministic_residual_policy(
        observation_shape, action_size, scale, encoder_factory
    )

    assert isinstance(policy, DeterministicResidualPolicy)

    x = torch.rand((batch_size,) + observation_shape)
    action = torch.rand(batch_size, action_size)
    y = policy(x, action)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_squashed_normal_policy(
    observation_shape, action_size, batch_size, encoder_factory
):
    policy = create_squashed_normal_policy(
        observation_shape, action_size, encoder_factory
    )

    assert isinstance(policy, SquashedNormalPolicy)

    x = torch.rand((batch_size,) + observation_shape)
    y = policy(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_categorical_policy(
    observation_shape, action_size, batch_size, encoder_factory
):
    policy = create_categorical_policy(
        observation_shape, action_size, encoder_factory
    )

    assert isinstance(policy, CategoricalPolicy)

    x = torch.rand((batch_size,) + observation_shape)
    y = policy(x)
    assert y.shape == (batch_size,)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("n_ensembles", [1, 5])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("share_encoder", [False, True])
def test_create_discrete_q_function(
    observation_shape,
    action_size,
    batch_size,
    n_ensembles,
    encoder_factory,
    share_encoder,
):
    q_func_factory = MeanQFunctionFactory(share_encoder=share_encoder)

    q_func = create_discrete_q_function(
        observation_shape,
        action_size,
        encoder_factory,
        q_func_factory,
        n_ensembles,
    )

    assert isinstance(q_func, EnsembleDiscreteQFunction)

    # check share_encoder
    encoder = q_func.q_funcs[0].encoder
    for q_func in q_func.q_funcs[1:]:
        if share_encoder:
            assert encoder is q_func.encoder
        else:
            assert encoder is not q_func.encoder

    x = torch.rand((batch_size,) + observation_shape)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("n_ensembles", [1, 2])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("share_encoder", [False, True])
def test_create_continuous_q_function(
    observation_shape,
    action_size,
    batch_size,
    n_ensembles,
    encoder_factory,
    share_encoder,
):
    q_func_factory = MeanQFunctionFactory(share_encoder=share_encoder)

    q_func = create_continuous_q_function(
        observation_shape,
        action_size,
        encoder_factory,
        q_func_factory,
        n_ensembles,
    )

    assert isinstance(q_func, EnsembleContinuousQFunction)

    # check share_encoder
    encoder = q_func.q_funcs[0].encoder
    for q_func in q_func.q_funcs[1:]:
        if share_encoder:
            assert encoder is q_func.encoder
        else:
            assert encoder is not q_func.encoder

    x = torch.rand((batch_size,) + observation_shape)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("latent_size", [32])
@pytest.mark.parametrize("beta", [1.0])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_conditional_vae(
    observation_shape,
    action_size,
    latent_size,
    beta,
    batch_size,
    encoder_factory,
):
    vae = create_conditional_vae(
        observation_shape, action_size, latent_size, beta, encoder_factory
    )

    assert isinstance(vae, ConditionalVAE)

    x = torch.rand((batch_size,) + observation_shape)
    action = torch.rand(batch_size, action_size)
    y = vae(x, action)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("beta", [1e-2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_discrete_imitator(
    observation_shape, action_size, beta, batch_size, encoder_factory
):
    imitator = create_discrete_imitator(
        observation_shape, action_size, beta, encoder_factory
    )

    assert isinstance(imitator, DiscreteImitator)

    x = torch.rand((batch_size,) + observation_shape)
    y = imitator(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_deterministic_regressor(
    observation_shape, action_size, batch_size, encoder_factory
):
    imitator = create_deterministic_regressor(
        observation_shape, action_size, encoder_factory
    )

    assert isinstance(imitator, DeterministicRegressor)

    x = torch.rand((batch_size,) + observation_shape)
    y = imitator(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_probablistic_regressor(
    observation_shape, action_size, batch_size, encoder_factory
):
    imitator = create_probablistic_regressor(
        observation_shape, action_size, encoder_factory
    )

    assert isinstance(imitator, ProbablisticRegressor)

    x = torch.rand((batch_size,) + observation_shape)
    y = imitator(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("batch_size", [32])
def test_create_value_function(observation_shape, encoder_factory, batch_size):
    v_func = create_value_function(observation_shape, encoder_factory)

    assert isinstance(v_func, ValueFunction)

    x = torch.rand((batch_size,) + observation_shape)
    y = v_func(x)
    assert y.shape == (batch_size, 1)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("n_ensembles", [5])
@pytest.mark.parametrize("discrete_action", [False, True])
@pytest.mark.parametrize("batch_size", [32])
def test_create_probabilistic_ensemble_dynamics_model(
    observation_shape,
    action_size,
    encoder_factory,
    n_ensembles,
    discrete_action,
    batch_size,
):
    dynamics = create_probabilistic_ensemble_dynamics_model(
        observation_shape,
        action_size,
        encoder_factory,
        n_ensembles,
        discrete_action,
    )

    assert isinstance(dynamics, ProbabilisticEnsembleDynamicsModel)
    assert len(dynamics.models) == n_ensembles

    x = torch.rand((batch_size,) + observation_shape)
    indices = torch.randint(n_ensembles, size=(batch_size,))
    if discrete_action:
        action = torch.randint(0, action_size, size=(batch_size, 1))
    else:
        action = torch.rand(batch_size, action_size)
    observation, reward = dynamics(x, action, indices)
    assert observation.shape == (batch_size,) + observation_shape
    assert reward.shape == (batch_size, 1)


@pytest.mark.parametrize("shape", [(100,)])
def test_create_parameter(shape):
    x = np.random.random()
    parameter = create_parameter(shape, x)

    assert len(list(parameter.parameters())) == 1
    assert np.allclose(parameter().detach().numpy(), x)
