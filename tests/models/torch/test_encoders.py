import pytest
import torch

from d3rlpy.models.torch.encoders import (
    PixelEncoder,
    PixelEncoderWithAction,
    VectorEncoder,
    VectorEncoderWithAction,
)

from .model_test import check_parameter_updates


@pytest.mark.parametrize("shapes", [((4, 84, 84), 3136)])
@pytest.mark.parametrize("filters", [[(32, 8, 4), (64, 4, 2), (64, 3, 1)]])
@pytest.mark.parametrize("feature_size", [512])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("use_batch_norm", [False, True])
@pytest.mark.parametrize("dropout_rate", [None, 0.2])
@pytest.mark.parametrize("activation", [torch.relu])
def test_pixel_encoder(
    shapes,
    filters,
    feature_size,
    batch_size,
    use_batch_norm,
    dropout_rate,
    activation,
):
    observation_shape, linear_input_size = shapes

    encoder = PixelEncoder(
        observation_shape=observation_shape,
        filters=filters,
        feature_size=feature_size,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        activation=activation,
    )
    x = torch.rand((batch_size,) + observation_shape)
    y = encoder(x)

    # check output shape
    assert encoder._get_linear_input_size() == linear_input_size
    assert y.shape == (batch_size, feature_size)

    # check use of batch norm
    encoder.eval()
    eval_y = encoder(x)
    if use_batch_norm or dropout_rate:
        assert not torch.allclose(y, eval_y)
    else:
        assert torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(encoder, (x,))

    # check reverse
    reverse_modules = encoder.create_reverse()
    h = eval_y
    for module in reverse_modules:
        h = module(h)
    assert h.shape == (batch_size, *observation_shape)


@pytest.mark.parametrize("shapes", [((4, 84, 84), 3136)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("filters", [[(32, 8, 4), (64, 4, 2), (64, 3, 1)]])
@pytest.mark.parametrize("feature_size", [512])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("use_batch_norm", [False, True])
@pytest.mark.parametrize("dropout_rate", [None, 0.2])
@pytest.mark.parametrize("discrete_action", [False, True])
@pytest.mark.parametrize("activation", [torch.relu])
def test_pixel_encoder_with_action(
    shapes,
    action_size,
    filters,
    feature_size,
    batch_size,
    use_batch_norm,
    dropout_rate,
    discrete_action,
    activation,
):
    observation_shape, linear_input_size = shapes

    encoder = PixelEncoderWithAction(
        observation_shape=observation_shape,
        action_size=action_size,
        filters=filters,
        feature_size=feature_size,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        discrete_action=discrete_action,
        activation=activation,
    )
    x = torch.rand((batch_size,) + observation_shape)
    if discrete_action:
        action = torch.randint(0, action_size, size=(batch_size, 1))
    else:
        action = torch.rand((batch_size, action_size))
    y = encoder(x, action)

    # check output shape
    assert encoder._get_linear_input_size() == linear_input_size + action_size
    assert y.shape == (batch_size, feature_size)

    # check use of batch norm
    encoder.eval()
    eval_y = encoder(x, action)
    if use_batch_norm or dropout_rate:
        assert not torch.allclose(y, eval_y)
    else:
        assert torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(encoder, (x, action))

    # check reverse
    reverse_modules = encoder.create_reverse()
    h = eval_y
    for module in reverse_modules:
        h = module(h)
    assert h.shape == (batch_size, *observation_shape)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("hidden_units", [[256, 256]])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("use_batch_norm", [False, True])
@pytest.mark.parametrize("dropout_rate", [None, 0.2])
@pytest.mark.parametrize("use_dense", [False, True])
@pytest.mark.parametrize("activation", [torch.relu])
def test_vector_encoder(
    observation_shape,
    hidden_units,
    batch_size,
    use_batch_norm,
    dropout_rate,
    use_dense,
    activation,
):
    encoder = VectorEncoder(
        observation_shape=observation_shape,
        hidden_units=hidden_units,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        use_dense=use_dense,
        activation=activation,
    )

    x = torch.rand((batch_size,) + observation_shape)
    y = encoder(x)

    # check output shape
    assert encoder.get_feature_size() == hidden_units[-1]
    assert y.shape == (batch_size, hidden_units[-1])

    # check use of batch norm
    encoder.eval()
    eval_y = encoder(x)
    if use_batch_norm or dropout_rate:
        assert not torch.allclose(y, eval_y)
    else:
        assert torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(encoder, (x,))

    # check reverse
    if not use_dense:
        reverse_modules = encoder.create_reverse()
        h = eval_y
        for module in reverse_modules:
            h = module(h)
        assert h.shape == (batch_size, *observation_shape)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("hidden_units", [[256, 256]])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("use_batch_norm", [False, True])
@pytest.mark.parametrize("dropout_rate", [None, 0.2])
@pytest.mark.parametrize("use_dense", [False, True])
@pytest.mark.parametrize("discrete_action", [False, True])
@pytest.mark.parametrize("activation", [torch.relu])
def test_vector_encoder_with_action(
    observation_shape,
    action_size,
    hidden_units,
    batch_size,
    use_batch_norm,
    dropout_rate,
    use_dense,
    discrete_action,
    activation,
):
    encoder = VectorEncoderWithAction(
        observation_shape=observation_shape,
        action_size=action_size,
        hidden_units=hidden_units,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        use_dense=use_dense,
        discrete_action=discrete_action,
        activation=activation,
    )

    x = torch.rand((batch_size,) + observation_shape)
    if discrete_action:
        action = torch.randint(0, action_size, size=(batch_size, 1))
    else:
        action = torch.rand((batch_size, action_size))
    y = encoder(x, action)

    # check output shape
    assert encoder.get_feature_size() == hidden_units[-1]
    assert y.shape == (batch_size, hidden_units[-1])

    # check use of batch norm
    encoder.eval()
    eval_y = encoder(x, action)
    if use_batch_norm or dropout_rate:
        assert not torch.allclose(y, eval_y)
    else:
        assert torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(encoder, (x, action))

    # check reverse
    if not use_dense:
        reverse_modules = encoder.create_reverse()
        h = eval_y
        for module in reverse_modules:
            h = module(h)
        assert h.shape == (batch_size, *observation_shape)
