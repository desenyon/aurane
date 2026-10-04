"""Constants must produce the same executable model and analysis reports."""

import pytest

from aurane.compiler import CompilationError, compile_source
from aurane.parser import parse_aurane
from aurane.type_checker import check_types
from aurane.profiler import profile_program

SOURCE = """width = 8
length = 5
model Net:
    input_shape = (length, width)
    hidden = width
    classes = 3
    def forward(x):
        x -> dense(hidden).relu -> dense(classes)
"""


def test_symbolic_dimensions_agree_between_compiler_checker_and_profiler():
    program = parse_aurane(SOURCE)
    code = compile_source(SOURCE, disable_cache=True, validate=True)
    assert "nn.Linear(8, 8" in code
    assert "nn.Linear(8, 3" in code
    checked = check_types(program)
    assert checked.is_valid
    assert checked.inferred_types["Net"]["output"].shape == (5, 3)
    report = profile_program(program)["Net"]
    assert report.output_shape == (5, 3)
    assert report.total_params == 99


@pytest.mark.parametrize(
    "source,message",
    [
        (
            "model Net:\n    input_shape = (4,)\n    def forward(x):\n        x -> dense(missing)\n",
            "Undefined symbol",
        ),
        (
            "a = b\nb = a\nmodel Net:\n    input_shape = (4,)\n    def forward(x):\n        x -> dense(a)\n",
            "cycle",
        ),
        ("width = 4\nwidth = 8\n", "Duplicate"),
        ("model Net:\n    a = b\n    b = a\n    def forward(x):\n        x -> dense(a)\n", "cycle"),
    ],
)
def test_invalid_symbols_fail_compilation_and_check(source, message):
    with pytest.raises(CompilationError, match=message):
        compile_source(source, disable_cache=True)
    checked = check_types(parse_aurane(source))
    assert checked.has_errors
    assert any(message in error.message for error in checked.errors)


def test_model_constants_shadow_globals_without_leaking_between_models():
    source = """width = 8
model A:
    width = 4
    input_shape = (width,)
    def forward(x):
        x -> dense(width)
model B:
    input_shape = (width,)
    def forward(x):
        x -> dense(width)
"""
    reports = profile_program(parse_aurane(source))
    assert reports["A"].input_shape == (4,)
    assert reports["A"].total_params == 20
    assert reports["B"].input_shape == (8,)
    assert reports["B"].total_params == 72


def test_quoted_identifier_is_not_replaced_by_constant():
    source = """cpu = "cuda"
experiment E:
    device = "cpu"
model Net:
    input_shape = (4,)
    def forward(x):
        x -> dense(2)
"""
    code = compile_source(source, disable_cache=True)
    assert 'device = torch.device("cpu")' in code


def test_training_call_arguments_can_reference_constants():
    source = """rate = 0.025
model Net:
    input_shape = (4,)
    def forward(x):
        x -> dense(2)
train Net on data:
    optimizer = sgd(lr=rate)
"""
    code = compile_source(source, disable_cache=True)
    assert "optim.SGD(model.parameters(), lr=0.025)" in code


def test_resolution_preserves_raw_ast():
    program = parse_aurane(SOURCE)
    check_types(program)
    profile_program(program)
    assert program.models[0].config["input_shape"] == ("length", "width")


def test_escaped_dataset_path_generates_valid_python_literal():
    import ast

    source = r"""dataset data:
    from torchvision.datasets.MNIST
    root = "a\"b\\c"
"""
    generated = ast.parse(compile_source(source, disable_cache=True))
    roots = [
        keyword.value.value
        for node in ast.walk(generated)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "root"
    ]
    assert roots == ['a"b\\c']
