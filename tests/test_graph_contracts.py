"""Graph values, tensor axes, and diagnostics must agree across consumers."""

import pytest

from aurane.compiler import CompilationError, compile_source
from aurane.ir import lower_forward_block
from aurane.parser import parse_aurane
from aurane.profiler import profile_program
from aurane.type_checker import check_types
from aurane.visualizer import render_model_architecture_mermaid, render_model_architecture_dot


def graph_source(body, shape=(4,)):
    return (
        f"model Net:\n    input_shape = {shape}\n    def forward(x):\n"
        + "\n".join("        " + line for line in body.splitlines())
        + "\n"
    )


@pytest.mark.parametrize("dim", [1, -1])
def test_concat_reports_runtime_feature_axis(dim):
    source = graph_source(
        f"a = dense(x, 3)\nb = dense(x, 2)\nz = concat(a, b, dim={dim})\nreturn z"
    )
    program = parse_aurane(source)
    checked = check_types(program)
    assert checked.is_valid
    assert checked.inferred_types["Net"]["output"].shape == (5,)
    report = profile_program(program)["Net"]
    assert report.output_shape == (5,)
    assert report.total_params == 25
    assert "(5,)" in render_model_architecture_mermaid(program.models[0])
    assert "(5,)" in render_model_architecture_dot(program.models[0])


@pytest.mark.parametrize(
    "body,fragment",
    [
        ("a = dense(missing, 3)", "Undefined"),
        ("a = dense(x, 3)\nreturn missing", "Undefined"),
        ("a = dense(x, 3)\nb = dense(x, 2)\nz = add(a, b)", "shape"),
        ("a = dense(x, 3)\nb = reshape(x, 2, 2)\nz = add(a, b)", "shape"),
        ("a = add(x)", "two"),
        ("a = concat(x, dim=1)", "two"),
        ("a = concat(x, x, dim=2)", "dim"),
        ("a = concat(x, x, dim=-3)", "dim"),
        ("a = concat(x, x, dim=0)", "batch"),
        ("a = reshape(x, 2, 2)\nb = reshape(x, 1, 4)\nz = concat(a, b, dim=1)", "shape"),
    ],
)
def test_invalid_graphs_report_diagnostics_instead_of_crashing(body, fragment):
    source = graph_source(body)
    result = check_types(parse_aurane(source))
    assert result.has_errors
    assert any(fragment.lower() in error.message.lower() for error in result.errors)
    with pytest.raises(CompilationError, match=rf"(?i){fragment}"):
        compile_source(source, disable_cache=True)


def test_ir_keeps_reassigned_values_distinct():
    source = graph_source("a = dense(x, 4)\nb = dense(a, 4)\na = add(a, b)\nreturn a")
    block = parse_aurane(source).models[0].forward_block
    ir = lower_forward_block(block)
    assert len({node.output.name for node in ir.nodes}) == 3
    assert ir.nodes[2].inputs[0] == ir.nodes[0].output
    assert ir.nodes[2].output != ir.nodes[0].output
    assert ir.outputs == [ir.nodes[2].output]


def test_renderers_show_actual_branch_edges_and_selected_return():
    source = graph_source("a = dense(x, 3)\nb = dense(x, 2)\nreturn a")
    model = parse_aurane(source).models[0]
    mermaid = render_model_architecture_mermaid(model)
    dot = render_model_architecture_dot(model)
    assert "op0 --> op1" in mermaid
    assert "op0 --> op2" in mermaid
    assert "op1 --> op2" not in mermaid
    assert "op1 --> op3" in mermaid
    assert "Output: (3,)" in mermaid
    assert "op0 -> op1;" in dot and "op0 -> op2;" in dot
    assert "op1 -> op3;" in dot
    assert "Output: (3,)" in dot


def test_rich_graph_export_writes_readable_svg(tmp_path):
    from xml.etree import ElementTree
    from aurane.visualizer import visualize_model_architecture

    model = parse_aurane(graph_source("a = dense(x, 3)\nreturn a")).models[0]
    output = tmp_path / "graph.svg"
    visualize_model_architecture(model, str(output))
    root = ElementTree.parse(output).getroot()
    assert root.tag.endswith("svg")
    assert "Net" in "".join(root.itertext())


def test_shape_plot_tracks_branch_inputs(monkeypatch):
    from io import StringIO
    from rich.console import Console
    from aurane import visualizer

    model = parse_aurane(graph_source("a = reshape(x, 2, 2)\nb = dense(x, 3)\nreturn b")).models[0]
    output = StringIO()
    monkeypatch.setattr(visualizer, "console", Console(file=output, width=120))
    visualizer.plot_layer_shapes(model)
    assert "(3,)" in output.getvalue()
    assert "(2, 3)" not in output.getvalue()


@pytest.mark.parametrize(
    "operation",
    [
        "dense(0)",
        "reshape(3, 3)",
        "reshape(-1, -1)",
        "multihead_attention(heads=3, dim=4)",
        "maxpool(2, stride=0)",
        "conv2d(3)",
        "unknown_operation()",
        "dense(4).unknown_activation",
    ],
)
def test_invalid_operations_fail_before_execution(operation):
    source = f"model Net:\n    input_shape = (4,)\n    def forward(x):\n        x -> {operation}\n"
    with pytest.raises(CompilationError):
        compile_source(source, disable_cache=True)
