"""Locations identify each expression, including repeated calls and Unicode."""

import pytest
from aurane.parser import parse_aurane
from aurane.ir import lower_model
from aurane.compiler import compile_source, CompilationError


def test_repeated_chain_operations_and_activation_spans():
    line = "        x -> dense(4).relu -> dense(4).tanh.sigmoid"
    code = "model Net:\n    input_shape = (4,)\n    def forward(x):\n" + line + "\n"
    model = parse_aurane(code).models[0]
    operations = model.forward_block.operations
    expected = ["dense(4).relu", "dense(4).tanh", "sigmoid"]
    cursor = 0
    for operation, expression in zip(operations, expected):
        column = line.index(expression, cursor) + 1
        assert (operation.line, operation.column) == (4, column)
        assert (operation.end_line, operation.end_column) == (4, column + len(expression))
        cursor = column + len(expression) - 1
    for node, operation in zip(lower_model(model).nodes, operations):
        assert (node.line, node.column, node.end_line, node.end_column) == (
            operation.line,
            operation.column,
            operation.end_line,
            operation.end_column,
        )


@pytest.mark.parametrize(
    "line",
    [
        "        x -> dense(4) -> dense(missing)",
        "          -> dense(missing)",
        "        output = dense(x, missing)",
    ],
)
def test_symbol_and_compile_diagnostic_point_to_exact_argument(line):
    prefix = "model Net:\n    input_shape = (4,)\n    def forward(x):\n"
    if line.lstrip().startswith("->"):
        prefix += "        x -> dense(4)\n"
    code = prefix + line + "\n"
    model = parse_aurane(code).models[0]
    block = model.forward_block
    operation = block.nodes[0].operation if hasattr(block, "nodes") else block.operations[-1]
    reference = operation.args[0]
    column = line.index("missing") + 1
    assert reference.column == column
    with pytest.raises(CompilationError, match=f"column {column}"):
        compile_source(code, disable_cache=True)


def test_unicode_before_reference_uses_character_columns():
    line = '    setting = custom("é", missing)'
    call = parse_aurane("experiment E:\n" + line + "\n").experiments[0].config["setting"]
    assert call.args[1].column == line.index("missing") + 1


def test_shape_diagnostic_points_to_failing_call():
    line = "        x -> dense(4) -> reshape(3)"
    code = "model Net:\n    input_shape = (4,)\n    def forward(x):\n" + line + "\n"
    with pytest.raises(CompilationError, match=f"column {line.index('reshape') + 1}"):
        compile_source(code, disable_cache=True)


@pytest.mark.parametrize(
    "setting",
    [
        "epochs = 0",
        "optimizer = adam(betas=(1, 1))",
        "scheduler = step_lr(step_size=0)",
        "loss = huber(delta=0)",
        "callbacks = [checkpoint(checkpoint_every=0)]",
        "metrics = [unknown]",
        "misspelled = True",
        "mixed_precision = 1",
    ],
)
def test_check_json_has_exact_training_field_span(tmp_path, setting):
    import json
    from tests.test_cli_contracts import cli

    line = "    " + setting
    code = (
        "dataset data:\nmodel Net:\n    input_shape = (4,)\n"
        "    def forward(x):\n        x -> dense(3)\ntrain Net on data:\n" + line + "\n"
    )
    path = tmp_path / "source.aur"
    path.write_text(code)
    payload = json.loads(cli("check", path, "--json").stdout)
    expected = {"line": 7, "column": 5, "end_line": 7, "end_column": len(line) + 1}
    assert payload["ok"] is False
    assert (
        next(issue for issue in payload["semantic"]["issues"] if issue["kind"] == "error")["span"]
        == expected
    )
    assert payload["types"]["errors"][0]["span"] == expected


@pytest.mark.parametrize(
    "code,line_number,column",
    [
        ('experiment E:\n    device = "bad"\n', 2, 5),
        ("dataset data:\n    batch = 0\n", 2, 5),
        (
            'model Net:\n    input_shape = (4,)\n    input_dtype = "bad"\n    def forward(x):\n        x -> dense(2)\n',
            3,
            5,
        ),
        (
            "model Net:\n    input_shape = (4,)\n    def forward(x):\n        x -> reshape(3)\n",
            4,
            14,
        ),
        ("model Net:\n    def forward(x):\n        x -> dense(missing)\n", 3, 20),
    ],
)
def test_check_json_type_errors_have_source_span(tmp_path, code, line_number, column):
    import json
    from tests.test_cli_contracts import cli

    path = tmp_path / "source.aur"
    path.write_text(code)
    payload = json.loads(cli("check", path, "--json").stdout)
    span = payload["types"]["errors"][0]["span"]
    assert (span["line"], span["column"]) == (line_number, column)
    assert span["end_column"] > span["column"]


def test_parse_error_json_includes_line_span(tmp_path):
    import json
    from tests.test_cli_contracts import cli

    path = tmp_path / "bad.aur"
    path.write_text("model Net:\n    garbage\n")
    payload = json.loads(cli("check", path, "--json").stdout)
    assert payload["error"]["span"] == {"line": 2, "column": 5, "end_line": 2, "end_column": 12}
