"""
Visualization tools for Aurane models.

Provides model architecture visualization, training metrics, and analysis.
"""

from typing import Optional, List, Tuple
from .ast import ModelNode, LayerOperation, ForwardGraphBlock
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


def print_model_summary(model: ModelNode):
    """Print detailed model summary."""
    if not RICH_AVAILABLE:
        print(f"Model: {model.name}")
        return

    # Create summary table
    table = Table(title=f"Model: {model.name}", show_header=True, header_style="bold cyan")
    table.add_column("Layer", style="cyan", no_wrap=True)
    table.add_column("Operation", style="yellow")
    table.add_column("Output Shape", style="green")
    table.add_column("Parameters", justify="right", style="magenta")

    if not model.forward_block:
        console.print("[yellow]No forward block defined[/yellow]")
        return

    # Get input shape
    input_shape = model.config.get("input_shape", (1, 28, 28))
    current_shape = input_shape
    total_params = 0

    table.add_row("Input", "-", str(input_shape), "0")

    if isinstance(model.forward_block, ForwardGraphBlock):
        shape_env = {model.forward_block.parameter: current_shape}
        for idx, node in enumerate(model.forward_block.nodes, 1):
            op = node.operation
            if op is None:
                continue
            op_name = op.operation.lower()
            inputs = node.inputs

            if op_name == "add":
                current_shape = shape_env.get(inputs[0], current_shape)
                params = 0
            elif op_name == "concat":
                dim = int(op.kwargs.get("dim", 1))
                shapes = [shape_env.get(v, current_shape) for v in inputs]
                if shapes and all(len(s) == len(shapes[0]) for s in shapes):
                    out_dims = list(shapes[0])
                    dim_sum = 0
                    for s in shapes:
                        d = s[dim] if len(s) > dim else -1
                        if d == -1 or dim_sum == -1:
                            dim_sum = -1
                            break
                        dim_sum += d
                    out_dims[dim] = dim_sum
                    current_shape = tuple(out_dims)
                else:
                    current_shape = shapes[0] if shapes else current_shape
                params = 0
            else:
                in_shape = shape_env.get(inputs[0], current_shape)
                params = calculate_parameters(op, in_shape)
                total_params += params
                current_shape = calculate_output_shape(op, in_shape)

            shape_env[node.target] = current_shape

            # Format operation (params display is best-effort for graph ops).
            op_str = f"{op.operation}("
            if op.args:
                op_str += ", ".join(map(str, op.args))
            if op.kwargs:
                if op.args:
                    op_str += ", "
                op_str += ", ".join(f"{k}={v}" for k, v in op.kwargs.items())
            op_str += ")"
            if op.activation:
                op_str += f".{op.activation}"

            layer_name = f"node_{idx}:{node.target}"
            table.add_row(layer_name, op_str, str(current_shape), f"{params:,}")
    else:
        for idx, op in enumerate(model.forward_block.operations, 1):
            # Calculate params
            params = calculate_parameters(op, current_shape)
            total_params += params

            # Calculate output shape
            current_shape = calculate_output_shape(op, current_shape)

            # Format operation
            op_str = f"{op.operation}("
            if op.args:
                op_str += ", ".join(map(str, op.args))
            if op.kwargs:
                if op.args:
                    op_str += ", "
                op_str += ", ".join(f"{k}={v}" for k, v in op.kwargs.items())
            op_str += ")"

            if op.activation:
                op_str += f".{op.activation}"

            layer_name = f"layer_{idx}"
            table.add_row(layer_name, op_str, str(current_shape), f"{params:,}")

    console.print(table)

    # Summary panel
    summary_text = f"""
[bold]Total Parameters:[/bold] [cyan]{total_params:,}[/cyan]
[bold]Input Shape:[/bold] {input_shape}
[bold]Output Shape:[/bold] {current_shape}
    """
    console.print(Panel(summary_text, title="Summary", border_style="green"))


def visualize_model_architecture(model: ModelNode, output_file: Optional[str] = None):
    """Create visual representation of model architecture."""
    if not RICH_AVAILABLE:
        print(f"Model: {model.name}")
        print("Visualization requires 'rich' library")
        return

    from rich.tree import Tree

    tree = Tree(f"[bold cyan]{model.name}[/bold cyan]")

    if not model.forward_block:
        tree.add("[yellow]No forward block[/yellow]")
        console.print(tree)
        return

    input_shape = model.config.get("input_shape", (1, 28, 28))
    current_shape = input_shape

    input_node = tree.add(f"[green]Input: {input_shape}[/green]")
    current_node = input_node

    if isinstance(model.forward_block, ForwardGraphBlock):
        shape_env = {model.forward_block.parameter: current_shape}
        for idx, node in enumerate(model.forward_block.nodes, 1):
            op = node.operation
            if op is None:
                continue

            op_name = op.operation.lower()
            inputs = node.inputs

            if op_name == "add":
                current_shape = shape_env.get(inputs[0], current_shape)
                params = 0
            elif op_name == "concat":
                dim = int(op.kwargs.get("dim", 1))
                shapes = [shape_env.get(v, current_shape) for v in inputs]
                if shapes and all(len(s) == len(shapes[0]) for s in shapes):
                    out_dims = list(shapes[0])
                    dim_sum = 0
                    for s in shapes:
                        d = s[dim] if len(s) > dim else -1
                        if d == -1 or dim_sum == -1:
                            dim_sum = -1
                            break
                        dim_sum += d
                    out_dims[dim] = dim_sum
                    current_shape = tuple(out_dims)
                else:
                    current_shape = shapes[0] if shapes else current_shape
                params = 0
            else:
                in_shape = shape_env.get(inputs[0], current_shape)
                params = calculate_parameters(op, in_shape)
                current_shape = calculate_output_shape(op, in_shape)

            shape_env[node.target] = current_shape

            op_desc = f"{op.operation} -> {node.target}"
            if op.activation:
                op_desc += f".{op.activation}"
            if op.args:
                op_desc += f"({', '.join(map(str, op.args))})"
            if op.kwargs:
                kw_str = ", ".join(f"{k}={v}" for k, v in op.kwargs.items())
                op_desc += f"({kw_str})"
            op_desc += f" >> {current_shape}"
            if params > 0:
                op_desc += f" [{params:,} params]"

            current_node = current_node.add(f"[yellow]{op_desc}[/yellow]")
    else:
        for idx, op in enumerate(model.forward_block.operations, 1):
            input_shape_for_op = current_shape
            current_shape = calculate_output_shape(op, input_shape_for_op)
            params = calculate_parameters(op, input_shape_for_op)

            op_desc = f"{op.operation}"
            if op.args:
                op_desc += f"({', '.join(map(str, op.args))})"
            if op.activation:
                op_desc += f" >> {op.activation}"

            op_desc += f" >> {current_shape}"
            if params > 0:
                op_desc += f" [{params:,} params]"

            current_node = current_node.add(f"[yellow]{op_desc}[/yellow]")

    current_node.add(f"[green]Output: {current_shape}[/green]")

    console.print(tree)

    if output_file:
        console.save_svg(output_file, title=f"{model.name} Architecture")


def render_model_architecture_mermaid(model: ModelNode) -> str:
    """Render model architecture as a Mermaid flowchart."""
    nodes: List[str] = []
    edges: List[str] = []

    node_id = lambda idx: f"op{idx}"

    # Use input/output shapes when available to keep nodes informative.
    input_shape = model.config.get("input_shape", (1, 28, 28))
    current_shape = input_shape

    nodes.append(f'{node_id(0)}["Input: {input_shape}"]')

    if not model.forward_block:
        nodes.append(f'{node_id(1)}["Output: ?"]')
        edges.append(f"{node_id(0)} --> {node_id(1)}")
        return _wrap_mermaid(nodes, edges)

    prev_idx = 0
    if isinstance(model.forward_block, ForwardGraphBlock):
        shape_env = {model.forward_block.parameter: current_shape}
        for idx, node in enumerate(model.forward_block.nodes, 1):
            op = node.operation
            if op is None:
                continue
            op_name = op.operation.lower()
            inputs = node.inputs
            params_shape = current_shape

            if op_name == "add":
                next_shape = shape_env.get(inputs[0], current_shape)
            elif op_name == "concat":
                dim = int(op.kwargs.get("dim", 1))
                shapes_in = [shape_env.get(v, current_shape) for v in inputs]
                if shapes_in and all(len(s) == len(shapes_in[0]) for s in shapes_in):
                    out_dims = list(shapes_in[0])
                    dim_sum = 0
                    for s in shapes_in:
                        d = s[dim] if len(s) > dim else -1
                        if d == -1 or dim_sum == -1:
                            dim_sum = -1
                            break
                        dim_sum += d
                    out_dims[dim] = dim_sum
                    next_shape = tuple(out_dims)
                else:
                    next_shape = shapes_in[0] if shapes_in else current_shape
            else:
                in_shape = shape_env.get(inputs[0], current_shape)
                params_shape = in_shape
                next_shape = calculate_output_shape(op, in_shape)

            params = 0 if op_name in ("add", "concat") else calculate_parameters(op, params_shape)
            op_label = f"{op.operation} -> {node.target}"
            if op.args:
                op_label += f"({', '.join(map(str, op.args))})"
            if op.kwargs:
                kw_str = ", ".join(f"{k}={v}" for k, v in op.kwargs.items())
                op_label += f"({kw_str})"
            if op.activation:
                op_label += f".{op.activation}"
            if next_shape:
                op_label += f" -> {next_shape}"
            if params:
                op_label += f" ({params:,} params)"

            nodes.append(f'{node_id(idx)}["{op_label}"]')
            edges.append(f"{node_id(prev_idx)} --> {node_id(idx)}")
            prev_idx = idx
            current_shape = next_shape
            shape_env[node.target] = next_shape
    else:
        for idx, op in enumerate(model.forward_block.operations, 1):
            input_shape_for_op = current_shape
            next_shape = calculate_output_shape(op, input_shape_for_op)
            params = calculate_parameters(op, input_shape_for_op)
            op_label = op.operation
            if op.args:
                op_label += f"({', '.join(map(str, op.args))})"
            if op.kwargs:
                kw_str = ", ".join(f"{k}={v}" for k, v in op.kwargs.items())
                op_label += f"({kw_str})"
            if op.activation:
                op_label += f".{op.activation}"
            if next_shape:
                op_label += f" -> {next_shape}"
            if params:
                op_label += f" ({params:,} params)"

            nodes.append(f'{node_id(idx)}["{op_label}"]')
            edges.append(f"{node_id(prev_idx)} --> {node_id(idx)}")
            prev_idx = idx
            current_shape = next_shape

    nodes.append(f'{node_id(prev_idx + 1)}["Output: {current_shape}"]')
    edges.append(f"{node_id(prev_idx)} --> {node_id(prev_idx + 1)}")

    return _wrap_mermaid(nodes, edges)


def render_model_architecture_dot(model: ModelNode) -> str:
    """Render model architecture as a DOT graph."""
    node_lines: List[str] = []
    edge_lines: List[str] = []

    node_id = lambda idx: f"op{idx}"

    input_shape = model.config.get("input_shape", (1, 28, 28))
    current_shape = input_shape
    node_lines.append(f'{node_id(0)} [label="Input: {input_shape}"];')

    if not model.forward_block:
        node_lines.append(f'{node_id(1)} [label="Output: ?"];')
        edge_lines.append(f"{node_id(0)} -> {node_id(1)};")
        return _wrap_dot(node_lines, edge_lines)

    prev_idx = 0
    if isinstance(model.forward_block, ForwardGraphBlock):
        shape_env = {model.forward_block.parameter: current_shape}
        for idx, node in enumerate(model.forward_block.nodes, 1):
            op = node.operation
            if op is None:
                continue
            op_name = op.operation.lower()
            inputs = node.inputs
            params_shape = current_shape

            if op_name == "add":
                next_shape = shape_env.get(inputs[0], current_shape)
            elif op_name == "concat":
                dim = int(op.kwargs.get("dim", 1))
                shapes_in = [shape_env.get(v, current_shape) for v in inputs]
                if shapes_in and all(len(s) == len(shapes_in[0]) for s in shapes_in):
                    out_dims = list(shapes_in[0])
                    dim_sum = 0
                    for s in shapes_in:
                        d = s[dim] if len(s) > dim else -1
                        if d == -1 or dim_sum == -1:
                            dim_sum = -1
                            break
                        dim_sum += d
                    out_dims[dim] = dim_sum
                    next_shape = tuple(out_dims)
                else:
                    next_shape = shapes_in[0] if shapes_in else current_shape
            else:
                in_shape = shape_env.get(inputs[0], current_shape)
                params_shape = in_shape
                next_shape = calculate_output_shape(op, in_shape)

            params = 0 if op_name in ("add", "concat") else calculate_parameters(op, params_shape)
            op_label = f"{op.operation} -> {node.target}"
            if op.args:
                op_label += f"({', '.join(map(str, op.args))})"
            if op.kwargs:
                kw_str = ", ".join(f"{k}={v}" for k, v in op.kwargs.items())
                op_label += f"({kw_str})"
            if op.activation:
                op_label += f".{op.activation}"

            if next_shape:
                op_label += f"\\n{next_shape}"
            if params:
                op_label += f"\\n{params:,} params"

            node_lines.append(f'{node_id(idx)} [label="{op_label}"];')
            edge_lines.append(f"{node_id(prev_idx)} -> {node_id(idx)};")
            prev_idx = idx
            current_shape = next_shape
            shape_env[node.target] = next_shape
    else:
        prev_idx = 0
        for idx, op in enumerate(model.forward_block.operations, 1):
            input_shape_for_op = current_shape
            next_shape = calculate_output_shape(op, input_shape_for_op)
            params = calculate_parameters(op, input_shape_for_op)
            op_label = op.operation
            if op.args:
                op_label += f"({', '.join(map(str, op.args))})"
            if op.kwargs:
                kw_str = ", ".join(f"{k}={v}" for k, v in op.kwargs.items())
                op_label += f"({kw_str})"
            if op.activation:
                op_label += f".{op.activation}"

            if next_shape:
                op_label += f"\\n{next_shape}"
            if params:
                op_label += f"\\n{params:,} params"

            node_lines.append(f'{node_id(idx)} [label="{op_label}"];')
            edge_lines.append(f"{node_id(prev_idx)} -> {node_id(idx)};")
            prev_idx = idx
            current_shape = next_shape

    node_lines.append(f'{node_id(prev_idx + 1)} [label="Output: {current_shape}"];')
    edge_lines.append(f"{node_id(prev_idx)} -> {node_id(prev_idx + 1)};")

    return _wrap_dot(node_lines, edge_lines)


def _wrap_mermaid(nodes: List[str], edges: List[str]) -> str:
    return "\n".join(["flowchart TD", *nodes, *edges])


def _wrap_dot(node_lines: List[str], edge_lines: List[str]) -> str:
    return "\n".join(["digraph Aurane {", "rankdir=LR;", *node_lines, *edge_lines, "}"])


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

    input_shape = model.config.get("input_shape", (1, 28, 28))
    current_shape = input_shape

    console.print(f"\n[bold cyan]Shape Flow:[/bold cyan] {model.name}\n")

    # Input
    text = Text()
    text.append("Input: ", style="bold")
    text.append(str(input_shape), style="green")
    console.print(text)

    if isinstance(model.forward_block, ForwardGraphBlock):
        operations = [node.operation for node in model.forward_block.nodes if node.operation]
    else:
        operations = model.forward_block.operations

    for op in operations:
        current_shape = calculate_output_shape(op, current_shape)

        text = Text()
        text.append("  | ", style="dim")
        text.append(f"{op.operation}", style="yellow")
        if op.activation:
            text.append(f".{op.activation}", style="cyan")
        text.append(" >> ", style="dim")
        text.append(str(current_shape), style="green")

        console.print(text)

    console.print()
