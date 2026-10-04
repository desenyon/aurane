"""Unknown sample dimensions must remain unknown through size inference."""

import pytest

from aurane.ast import LayerOperation
from aurane.compiler import CompilationError, compile_source
from aurane.parser import parse_aurane
from aurane.shapes import infer_output_shape
from aurane.type_checker import check_types


@pytest.mark.parametrize("shape", [(-1, 4), (-1, -1), (3, -1, -1)])
@pytest.mark.parametrize("operation", ["flatten", "reshape"])
def test_flattened_unknown_dimensions_never_become_concrete(shape, operation):
    assert infer_output_shape(LayerOperation(operation=operation), shape) == (-1,)
    source = (
        f"model Net:\n    input_shape = {shape!r}\n    def forward(x):\n"
        f"        x -> {operation}() -> dense(3)\n"
    )
    assert check_types(parse_aurane(source)).has_errors
    with pytest.raises(CompilationError, match="known positive input feature"):
        compile_source(source, disable_cache=True)


def test_reshape_preserves_inferred_dimension_with_unknown_input_size():
    operation = LayerOperation(operation="reshape", args=[2, -1])
    assert infer_output_shape(operation, (-1, -1)) == (2, -1)


def test_dynamic_reshape_runs_when_actual_size_matches_requested_shape():
    torch = pytest.importorskip("torch")
    namespace = {"__name__": "dynamic_shape_test"}
    source = """model Net:
    input_shape = (-1, -1)
    def forward(x):
        x -> reshape(2, -1)
"""
    exec(compile_source(source, disable_cache=True), namespace)
    inputs = torch.randn(3, 4, 5)
    torch.testing.assert_close(namespace["Net"]()(inputs), inputs.reshape(3, 2, 10))
