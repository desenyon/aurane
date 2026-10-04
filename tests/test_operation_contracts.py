"""The semantic pass and compiler enforce one supported operation contract."""

import pytest

from aurane.compiler import CompilationError, compile_source
from aurane.parser import parse_aurane
from aurane.semantic_analyzer import analyze_semantics


def source(operation, shape=(4, 8, 8)):
    return (
        f"model Net:\n    input_shape = {shape!r}\n    def forward(x):\n        x -> {operation}\n"
    )


@pytest.mark.parametrize(
    "operation",
    [
        "dense(3, 9)",
        "conv2d(4, kernal=3)",
        "dropout(0.2, inplace=True)",
        "flatten(start_dim=2)",
        "dense(3, bias=1)",
        "layer_norm(3)",
        "embedding(10, 4, bogus=True)",
        "relu(7)",
        "multihead_attention(8)",
        "positional_encoding(maximum=20)",
        "mystery()",
        "dense(3).mystery",
    ],
)
def test_unknown_or_ignored_arguments_fail_semantics_and_generation(operation):
    program = parse_aurane(source(operation))
    result = analyze_semantics(program)
    assert result.has_errors
    with pytest.raises(CompilationError):
        compile_source(source(operation), disable_cache=True)


def test_graph_merges_validate_their_options():
    code = """model Net:
    input_shape = (4,)
    def forward(x):
        out = add(x, x, bogus=True)
        return out
"""
    with pytest.raises(CompilationError, match="bogus"):
        compile_source(code, disable_cache=True)


@pytest.mark.parametrize("rate", [0, 1])
def test_dropout_endpoints_have_consistent_semantics(rate):
    assert analyze_semantics(parse_aurane(source(f"dropout({rate})"))).is_valid


@pytest.mark.parametrize(
    "operation,shape",
    [
        ("dense(3)", (-1,)),
        ("batchnorm()", (2, 3, 4, 5)),
        ("layer_norm()", (2, -1)),
        ("embedding(5, 3, padding_idx=5)", (4,)),
        ("embedding(5, 3, padding_idx=True)", (4,)),
        ("softmax(9)", (3,)),
        ("leaky_relu('x')", (3,)),
    ],
)
def test_invalid_shapes_and_operation_ranges_fail_generation(operation, shape):
    with pytest.raises(CompilationError):
        compile_source(source(operation, shape), disable_cache=True)


@pytest.mark.parametrize(
    "training",
    [
        "train Net on data:\n    validate_on = absent\n",
        "train Net on data:\n    test_on = absent\n",
        "train_gan Net and absent on data:\n    epochs = 1\n",
        "train_gan Net and Net on absent:\n    epochs = 1\n",
    ],
)
def test_analysis_checks_evaluation_and_gan_references(training):
    from aurane.type_checker import check_types

    program = parse_aurane("dataset data:\n\n" + source("dense(3)", (3,)) + training)
    for result in (analyze_semantics(program), check_types(program)):
        assert result.has_errors
        assert any("absent" in error.message for error in result.errors)
