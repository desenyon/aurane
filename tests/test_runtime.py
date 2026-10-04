"""Execute generated programs against PyTorch, including backward passes.

The compiler-only suite can skip this module; CI's runtime job installs torch.
All inputs are small, deterministic, and offline.
"""

import pytest

torch = pytest.importorskip("torch")

from aurane.compiler import compile_source
from aurane.parser import parse_aurane
from aurane.profiler import profile_model
from aurane.type_checker import check_types


def load_model(source, *, optimize=False, level=1):
    namespace = {"__name__": "generated_test"}
    code = compile_source(source, disable_cache=True, optimize=optimize, opt_level=level)
    exec(compile(code, "<generated>", "exec"), namespace)
    return namespace["Net"]()


def test_symbolic_token_model_executes_and_matches_static_report():
    source = """vocabulary = 17
width = 8
model Net:
    input_shape = (5,)
    classes = 3
    def forward(tokens):
        tokens -> embedding(vocabulary, width) -> dense(classes)
"""
    model = load_model(source)
    result = model(torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]))
    assert result.shape == (2, 5, 3)
    result.square().sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    from aurane.profiler import profile_program

    report = profile_program(parse_aurane(source))["Net"]
    assert report.output_shape == (5, 3)
    assert report.total_params == 163


def test_sequence_normalization_accepts_variable_lengths_and_honors_padding():
    source = """model Net:
    input_shape = (5,)
    def forward(x):
        x -> embedding(11, 8, padding_idx=0)
          -> positional_encoding(max_len=12)
          -> layer_norm(eps=0.001)
          -> dense(3)
"""
    model = load_model(source)
    for length in (3, 7):
        x = torch.randint(1, 11, (2, length))
        x[:, 0] = 0
        output = model(x)
        assert output.shape == (2, length, 3)
        output.square().sum().backward()
    assert model.layer_norm1.normalized_shape == (8,)
    assert model.layer_norm1.eps == 0.001
    assert torch.count_nonzero(model.embedding1.weight.grad[0]) == 0
    assert profile_model(parse_aurane(source).models[0]).total_params == sum(
        p.numel() for p in model.parameters()
    )


def test_causal_attention_does_not_read_future_tokens():
    source = """model Net:
    input_shape = (4, 8)
    def forward(x):
        x -> multihead_attention(heads=2, dim=8, causal=True)
"""
    model = load_model(source).eval()
    x = torch.randn(2, 4, 8, requires_grad=True)
    changed = x.detach().clone()
    changed[:, 2:] += 10
    torch.testing.assert_close(model(x)[:, :2], model(changed)[:, :2])
    model(x)[:, 0].sum().backward()
    assert torch.count_nonzero(x.grad[:, 1:]) == 0


@pytest.mark.parametrize(
    "chain", ["dropout(0).sigmoid", "flatten().sigmoid", "layer_norm().sigmoid"]
)
def test_activation_suffixes_apply_to_every_operation(chain):
    source = f"model Net:\n    input_shape = (4,)\n    def forward(x):\n        x -> {chain}\n"
    model = load_model(source)
    x = torch.tensor([[-1.0, 2.0, 3.0, -4.0]])
    expected_input = model.layer_norm1(x) if chain.startswith("layer_norm") else x
    torch.testing.assert_close(model(x), torch.sigmoid(expected_input))


def test_reshape_infers_one_dimension_and_preserves_batch():
    source = (
        "model Net:\n    input_shape = (8,)\n    def forward(x):\n        x -> reshape(-1, 2)\n"
    )
    model = load_model(source)
    x = torch.arange(24.0).reshape(3, 8)
    assert model(x).shape == (3, 4, 2)
    assert check_types(parse_aurane(source)).inferred_types["Net"]["output"].shape == (4, 2)


@pytest.mark.parametrize("dim", [1, -1])
def test_branched_graph_executes_concat_and_three_input_add(dim):
    source = f"""model Net:
    input_shape = (4,)
    def forward(x):
        a = dense(x, 3)
        b = dense(x, 2)
        c = concat(a, b, dim={dim})
        c = add(c, c, c)
        return c
"""
    model = load_model(source)
    x = torch.randn(2, 4, requires_grad=True)
    expected = 3 * torch.cat([model.dense1(x), model.dense2(x)], dim=1)
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected.sum(), x, retain_graph=True)[0]
    actual.sum().backward()
    torch.testing.assert_close(x.grad, expected_grad)


@pytest.mark.parametrize(
    "shape,chain,expected",
    [
        ((5, 8), "dense(4) -> dense(3)", (5, 3)),
        ((2, 10, 12), "maxpool(3, stride=2) -> flatten() -> dense(3)", (3,)),
        ((2, 9, 11), "avgpool(3, stride=2) -> flatten() -> dense(3)", (3,)),
        ((8,), "dense(4, bias=False)", (4,)),
        ((2, 7, 7), "conv2d(3, kernel=3, bias=False)", (3, 5, 5)),
        ((5, 8), "positional_encoding(max_len=9) -> dense(4)", (5, 4)),
    ],
)
def test_shape_and_parameter_reports_match_runtime(shape, chain, expected):
    source = f"model Net:\n    input_shape = {shape}\n    def forward(x):\n        x -> {chain}\n"
    model = load_model(source)
    x = torch.randn(2, *shape, requires_grad=True)
    output = model(x)
    assert output.shape == (2, *expected)
    output.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    program = parse_aurane(source)
    report = profile_model(program.models[0])
    assert report.output_shape == expected
    assert report.total_params == sum(p.numel() for p in model.parameters())
    types = check_types(program)
    assert not types.has_errors
    assert types.inferred_types["Net"]["output"].shape == expected
    if "bias=False" in chain:
        assert all(
            module.bias is None
            for module in model.modules()
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))
        )


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize(
    "shape,chain",
    [
        ((4,), "dense(4).sigmoid -> tanh() -> dense(2)"),
        ((4,), "dense(4).gelu -> gelu() -> dense(2)"),
        ((4,), "dense(4).relu -> relu() -> dense(2)"),
        ((4,), "dense(4) -> dropout(0.5) -> dense(2)"),
        ((2, 8, 8), "conv2d(2, kernel=3) -> batchnorm() -> relu() -> flatten() -> dense(2)"),
        ((2, 12, 12), "maxpool(2) -> maxpool(2) -> flatten() -> dense(2)"),
        ((2, 12, 12), "maxpool(2, stride=1) -> maxpool(2) -> flatten() -> dense(2)"),
    ],
)
def test_optimization_preserves_outputs_gradients_and_state(shape, chain, level, training):
    source = f"model Net:\n    input_shape = {shape}\n    def forward(x):\n        x -> {chain}\n"
    torch.manual_seed(12)
    reference = load_model(source).train(training)
    candidate = load_model(source, optimize=True, level=level).train(training)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, *shape, requires_grad=True)
    other_x = x.detach().clone().requires_grad_()
    torch.manual_seed(34)
    expected = reference(x)
    torch.manual_seed(34)
    actual = candidate(other_x)
    torch.testing.assert_close(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(other_x.grad, x.grad)
    for (name, param), (other_name, other_param) in zip(
        reference.named_parameters(), candidate.named_parameters()
    ):
        assert name == other_name
        torch.testing.assert_close(other_param.grad, param.grad)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value)
