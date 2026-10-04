"""
Visualization tools for Aurane models.

Provides model architecture visualization, training metrics, and analysis.
"""

from typing import Optional, List, Tuple
from .ast import ModelNode
from .ir import IRGraph, IRValue, lower_model
from .symbols import resolve_model
import html
import json
from .shapes import (
    infer_output_shape as calculate_output_shape,
    calculate_params as calculate_parameters,
)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False


# Removed local implementations in favor of .shapes


def _model_graph(model: ModelNode) -> IRGraph:
    model = resolve_model(model)
    shape = tuple(model.config.get("input_shape", (1, 28, 28)))
    if not model.forward_block:
        return IRGraph(inputs=[IRValue("input", shape=shape)], outputs=[IRValue("unknown")])
    return lower_model(model)


def _node_label(node) -> str:
    args = [str(value) for value in node.args]
    args.extend(f"{key}={value}" for key, value in node.kwargs.items())
    label = f"{node.op_name}({', '.join(args)})"
    if node.activation:
        label += f".{node.activation}"
    label += f" -> {node.output.shape}"
    if node.output.type_hint:
        label += f" {node.output.type_hint}"
    params = calculate_parameters(node.to_operation(), node.inputs[0].shape)
    if params:
        label += f" ({params:,} params)"
    return label


def _architecture(model: ModelNode):
    graph = _model_graph(model)
    labels = [f"Input: {graph.inputs[0].shape}"]
    ids = {graph.inputs[0].name: 0}
    edges: List[Tuple[int, int]] = []
    for index, node in enumerate(graph.nodes, 1):
        assert node.output is not None
        labels.append(_node_label(node))
        edges.extend((ids[value.name], index) for value in node.inputs)
        ids[node.output.name] = index
    output = graph.outputs[0]
    labels.append(f"Output: {output.shape if output.shape is not None else '?'}")
    edges.append((ids.get(output.name, 0), len(labels) - 1))
    return labels, edges


def print_model_summary(model: ModelNode):
    """Print shapes and parameter counts from the same IR used by codegen."""
    if not RICH_AVAILABLE:
        print(f"Model: {model.name}")
        return
    graph = _model_graph(model)
    table = Table(title=f"Model: {model.name}", show_header=True, header_style="bold cyan")
    for heading in ("Layer", "Operation", "Output Shape", "Parameters"):
        table.add_column(heading)
    table.add_row("Input", "-", str(graph.inputs[0].shape), "0")
    total_params = 0
    for index, node in enumerate(graph.nodes):
        assert node.output is not None and node.inputs[0].shape is not None
        params = calculate_parameters(node.to_operation(), node.inputs[0].shape)
        total_params += params
        table.add_row(f"layer_{index}", node.op_name, str(node.output.shape), f"{params:,}")
    console.print(table)
    console.print(
        Panel(
            f"Total Parameters: {total_params:,}\nInput Shape: {graph.inputs[0].shape}"
            f"\nOutput Shape: {graph.outputs[0].shape}",
            title="Summary",
            border_style="green",
        )
    )


def visualize_model_architecture(model: ModelNode, output_file: Optional[str] = None):
    """Show actual tensor dependencies and optionally save a recorded SVG."""
    if not RICH_AVAILABLE:
        print(f"Model: {model.name}")
        return
    from rich.tree import Tree

    graph = _model_graph(model)
    tree = Tree(model.name)
    tree.add(f"Input: {graph.inputs[0].shape}")
    for node in graph.nodes:
        assert node.output is not None
        tree.add(
            f"{node.output.name} <- {', '.join(value.name for value in node.inputs)}: {_node_label(node)}"
        )
    tree.add(f"Output: {graph.outputs[0].shape}")
    target_console = Console(record=True) if output_file else console
    target_console.print(tree)
    if output_file:
        target_console.save_svg(output_file, title=f"{model.name} Architecture")


def render_model_architecture_mermaid(model: ModelNode) -> str:
    """Render the actual graph edges, including its selected return tensor."""
    labels, edges = _architecture(model)
    nodes = [f'op{index}["{html.escape(label, quote=True)}"]' for index, label in enumerate(labels)]
    return "\n".join(
        ["flowchart TD", *nodes, *(f"op{left} --> op{right}" for left, right in edges)]
    )


def render_model_architecture_dot(model: ModelNode) -> str:
    """Render a DOT graph with escaped labels and explicit tensor wiring."""
    labels, edges = _architecture(model)
    nodes = [f"op{index} [label={json.dumps(label)}];" for index, label in enumerate(labels)]
    lines = [*nodes, *(f"op{left} -> op{right};" for left, right in edges)]
    return "digraph G {\n" + "\n".join("  " + line for line in lines) + "\n}"


def generate_training_report(metrics: dict, output_file: Optional[str] = None):
    """Generate training metrics report."""
    if not RICH_AVAILABLE:
        print("Training Report")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        return

    table = Table(title="Training Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)


def plot_layer_shapes(model: ModelNode):
    """Plot shape transformations through the model."""
    if not RICH_AVAILABLE:
        return

    from rich.text import Text

    if not model.forward_block:
        return

    graph = _model_graph(model)

    console.print(f"\n[bold cyan]Shape Flow:[/bold cyan] {model.name}\n")

    # Input
    text = Text()
    text.append("Input: ", style="bold")
    text.append(str(graph.inputs[0].shape), style="green")
    console.print(text)

    for node in graph.nodes:
        assert node.output is not None
        text = Text()
        text.append("  | ", style="dim")
        text.append(
            f"{node.op_name}({', '.join(value.name for value in node.inputs)})", style="yellow"
        )
        if node.activation:
            text.append(f".{node.activation}", style="cyan")
        text.append(" >> ", style="dim")
        text.append(str(node.output.shape), style="green")

        console.print(text)

    console.print()
