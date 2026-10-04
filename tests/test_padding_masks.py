"""Attention padding masks block padded keys and keep fully padded rows finite."""

import pytest

torch = pytest.importorskip("torch")

from aurane.compiler import CompilationError, compile_source
from tests.test_runtime import load_model

FEATURES = """model Net:
    input_shape = (6, 8)
    def forward(x):
        x -> multihead_attention(heads=2, causal=True)
"""
TOKENS = """model Net:
    input_shape = (6,)
    input_padding_idx = 0
    def forward(tokens):
        tokens -> embedding(12, 8, padding_idx=0)
          -> positional_encoding(max_len=12)
          -> multihead_attention(heads=2, causal=True)
"""


@pytest.mark.parametrize("training", [True, False])
def test_masked_values_do_not_affect_outputs_or_gradients(training):
    model = load_model(FEATURES).train(training)
    inputs = torch.randn(3, 6, 8, requires_grad=True)
    mask = torch.tensor(
        [
            [False, False, False, True, True, True],
            [True, True, False, False, True, True],
            [True] * 6,
        ]
    )
    changed = inputs.detach().clone()
    changed[mask] += 100
    actual = model(inputs, padding_mask=mask)
    with torch.no_grad():
        expected = model(changed, padding_mask=mask)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[mask]) == 0
    actual.square().sum().backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.count_nonzero(inputs.grad[mask]) == 0
    assert torch.count_nonzero(inputs.grad[~mask]) > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("causal", [True, False])
def test_valid_queries_match_native_attention(causal):
    model = load_model(FEATURES.replace("causal=True", f"causal={causal}")).eval()
    layer = next(
        module for module in model.modules() if isinstance(module, torch.nn.MultiheadAttention)
    )
    inputs = torch.randn(2, 6, 8)
    mask = torch.tensor(
        [[False, False, False, True, True, True], [True, True, False, False, False, True]]
    )
    with torch.no_grad():
        actual = model(inputs, padding_mask=mask)
        for index in range(2):
            valid = inputs[index : index + 1, ~mask[index]]
            attention_mask = (
                torch.ones(valid.shape[1], valid.shape[1], dtype=torch.bool).triu(1)
                if causal
                else None
            )
            expected, _ = layer(valid, valid, valid, attn_mask=attention_mask, need_weights=False)
            torch.testing.assert_close(actual[index, ~mask[index]], expected[0])


def test_token_padding_mask_is_derived_and_can_be_overridden():
    model = load_model(TOKENS).eval()
    tokens = torch.tensor([[1, 2, 3, 0], [0, 0, 0, 0]])
    with torch.no_grad():
        automatic = model(tokens)
        explicit = model(tokens, padding_mask=tokens == 0)
        override = model(tokens, padding_mask=torch.zeros_like(tokens, dtype=torch.bool))
    torch.testing.assert_close(automatic, explicit)
    assert torch.count_nonzero(automatic[tokens == 0]) == 0
    assert torch.isfinite(automatic).all()
    assert not torch.equal(automatic, override)


@pytest.mark.parametrize(
    "mask",
    [torch.zeros(2, 6), torch.zeros(2, 5, dtype=torch.bool), torch.zeros(6, dtype=torch.bool)],
)
def test_invalid_runtime_masks_fail_with_clear_error(mask):
    with pytest.raises(ValueError, match="padding_mask"):
        load_model(FEATURES)(torch.randn(2, 6, 8), padding_mask=mask)


@pytest.mark.parametrize(
    "code",
    [
        FEATURES.replace("    input_shape", "    input_padding_idx = 0\n    input_shape"),
        TOKENS.replace("input_padding_idx = 0", "input_padding_idx = -1"),
        TOKENS.replace("input_padding_idx = 0", "input_padding_idx = True"),
    ],
)
def test_invalid_automatic_padding_configuration_fails(code):
    with pytest.raises(CompilationError, match="padding"):
        compile_source(code, disable_cache=True)
