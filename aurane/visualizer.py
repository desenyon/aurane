"""
Visualization tools for Aurane models.

Provides model architecture visualization, training metrics, and analysis.
"""

from typing import Optional, List, Tuple
from .ast import ModelNode, LayerOperation
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

    for idx, op in enumerate(model.forward_block.operations, 1):
        current_shape = calculate_output_shape(op, current_shape)
        params = calculate_parameters(op, current_shape)

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

    for op in model.forward_block.operations:
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
