"""Runtime shape, parameter, numerical and gradient proof for advertised layers."""

import pytest

torch = pytest.importorskip("torch")

from aurane.compiler import CompilationError, compile_source
from aurane.parser import parse_aurane
from aurane.profiler import profile_model
from tests.test_runtime import load_model
from tests.test_operation_contracts import source


@pytest.mark.parametrize(
    "operation,shape,output_shape,reference",
    [
        (
            "conv1d(6, kernel=3, padding=2, dilation=2, groups=2, bias=False)",
            (4, 11),
            (6, 11),
            lambda model, x: model.conv1d1(x),
        ),
        (
            "conv2d(6, kernel=3, dilation=2, padding=2, groups=2)",
            (4, 7, 8),
            (6, 7, 8),
            lambda model, x: model.conv2d1(x),
        ),
        (
            "lstm(5, num_layers=2, bidirectional=True, bias=False)",
            (7, 4),
            (7, 10),
            lambda model, x: model.lstm1(x)[0],
        ),
        (
            "gru(5, num_layers=2, bidirectional=True)",
            (7, 4),
            (7, 10),
            lambda model, x: model.gru1(x)[0],
        ),
        (
            "upsample(scale_factor=1.5, mode='linear', align_corners=False)",
            (2, 7),
            (2, 10),
            lambda model, x: torch.nn.functional.interpolate(
                x, scale_factor=1.5, mode="linear", align_corners=False
            ),
        ),
        (
            "upsample(size=(9, 11), mode='bilinear', align_corners=True)",
            (2, 4, 5),
            (2, 9, 11),
            lambda model, x: torch.nn.functional.interpolate(
                x, size=(9, 11), mode="bilinear", align_corners=True
            ),
        ),
        (
            "upsample(scale_factor=(2, 3))",
            (2, 4, 5),
            (2, 8, 15),
            lambda model, x: torch.nn.functional.interpolate(x, scale_factor=(2, 3)),
        ),
    ],
)
def test_extended_layers_execute_with_matching_shapes_parameters_and_gradients(
    operation, shape, output_shape, reference
):
    code = source(operation, shape)
    model = load_model(code)
    x = torch.randn(2, *shape, requires_grad=True)
    result = model(x)
    expected = reference(model, x)
    torch.testing.assert_close(result, expected)
    assert result.shape == (2, *output_shape)
    grad = torch.autograd.grad(result.square().sum(), x, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(grad, reference_grad)
    result.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    report = profile_model(parse_aurane(code).models[0])
    assert report.output_shape == output_shape
    assert report.total_params == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize(
    "operation,shape",
    [
        ("conv1d(3, groups=2)", (4, 9)),
        ("conv1d(4)", (4, 8, 8)),
        ("conv1d(4, dilation=0)", (4, 9)),
        ("lstm(0)", (7, 4)),
        ("gru(5, num_layers=0)", (7, 4)),
        ("gru(5)", (4,)),
        ("gru(5, dropout=0.2)", (7, 4)),
        ("upsample()", (2, 4, 5)),
        ("upsample(scale_factor=0)", (2, 4, 5)),
        ("upsample(size=4, scale_factor=2)", (2, 4, 5)),
        ("upsample(scale_factor=2, mode='linear')", (2, 4, 5)),
        ("upsample(scale_factor=2, align_corners=False)", (2, 4, 5)),
        ("upsample(scale_factor=(2,))", (2, 4, 5)),
    ],
)
def test_invalid_layer_contracts_fail_before_execution(operation, shape):
    with pytest.raises(CompilationError):
        compile_source(source(operation, shape), disable_cache=True)


@pytest.mark.parametrize(
    "operation,shape,flops",
    [
        ("conv1d(6, kernel=3, groups=2)", (4, 11), 2 * 3 * 2 * 6 * 9),
        ("conv2d(6, kernel=3, groups=2)", (4, 7, 8), 2 * 9 * 2 * 6 * 5 * 6),
        (
            "gru(5, num_layers=2, bidirectional=True)",
            (7, 4),
            2 * 7 * (2 * 3 * 5 * (4 + 5) + 2 * 3 * 5 * (10 + 5)),
        ),
    ],
)
def test_extended_layer_multiply_add_estimates(operation, shape, flops):
    assert profile_model(parse_aurane(source(operation, shape)).models[0]).total_flops == flops
