"""
Visualization tools for Aurane models.

Provides model architecture visualization, training metrics, and analysis.
"""

from typing import Optional, List, Tuple
from .ast import ModelNode, LayerOperation

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False


def calculate_output_shape(input_shape: tuple, operation: LayerOperation) -> tuple:
    """Calculate output shape after an operation."""
    op_name = operation.operation.lower()

    def to_int(val, default: int = 0) -> int:
        """Safely convert a value to int."""
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        if isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                return default
        return default

    if op_name == "conv2d":
        if len(input_shape) == 3:
            c, h, w = input_shape
            kernel = to_int(operation.kwargs.get("kernel", 3), 3)
            out_channels = to_int(operation.args[0], 32) if operation.args else 32
            stride = to_int(operation.kwargs.get("stride", 1), 1)
            padding = to_int(operation.kwargs.get("padding", 0), 0)

            h_out = (h + 2 * padding - kernel) // stride + 1
            w_out = (w + 2 * padding - kernel) // stride + 1
            return (out_channels, h_out, w_out)
        return input_shape

    elif op_name == "maxpool" or op_name == "avgpool":
        if len(input_shape) == 3:
            c, h, w = input_shape
            kernel = to_int(operation.args[0], 2) if operation.args else 2
            stride = to_int(operation.kwargs.get("stride", kernel), kernel)
            return (c, h // stride, w // stride)
        return input_shape

    elif op_name == "flatten":
        if len(input_shape) == 3:
            c, h, w = input_shape
            return (c * h * w,)
        return input_shape

    elif op_name in ("dense", "linear"):
        out_features = to_int(operation.args[0], 128) if operation.args else 128
        return (out_features,)

    elif op_name in ("dropout", "batchnorm", "layer_norm"):
        return input_shape

    elif op_name == "embedding":
        # embedding(vocab_size, embed_dim) -> (seq_len, embed_dim)
        if len(operation.args) >= 2:
            embed_dim = to_int(operation.args[1], 256)
            if len(input_shape) >= 1:
                seq_len = to_int(input_shape[0], 1)
                return (seq_len, embed_dim)
        return input_shape

    elif op_name == "positional_encoding":
        return input_shape

    elif op_name == "multihead_attention":
        return input_shape

    elif op_name == "residual":
        return input_shape

    elif op_name == "global_avg_pool":
        if len(input_shape) == 3:
            c, h, w = input_shape
            return (c,)
        return input_shape

    elif op_name == "concat":
        # Concatenation - depends on axis
        return input_shape

    elif op_name == "add":
        return input_shape

    return input_shape


def calculate_parameters(operation: LayerOperation, input_shape: tuple) -> int:
    """Calculate number of parameters for an operation."""
    op_name = operation.operation.lower()

    def to_int(val, default: int = 0) -> int:
        """Safely convert a value to int."""
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        if isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                return default
        return default

    if op_name == "conv2d":
        if len(input_shape) == 3:
            in_channels = to_int(input_shape[0], 1)
            out_channels = to_int(operation.args[0], 32) if operation.args else 32
            kernel = to_int(operation.kwargs.get("kernel", 3), 3)
            # params = (kernel * kernel * in_channels + 1) * out_channels
            params = kernel * kernel * in_channels * out_channels + out_channels
            return params
        return 0

    elif op_name in ("dense", "linear"):
        in_features = to_int(input_shape[0], 128) if input_shape else 128
        out_features = to_int(operation.args[0], 128) if operation.args else 128
        # params = (in_features + 1) * out_features
        params = in_features * out_features + out_features
        return params

    elif op_name == "dropout":
        return 0

    elif op_name == "batchnorm":
        # BatchNorm has 2 * num_features parameters (gamma and beta)
        if input_shape:
            num_features = to_int(input_shape[0], 0)
            return 2 * num_features
        return 0

    elif op_name == "embedding":
        # Embedding has vocab_size * embedding_dim parameters
        if len(operation.args) >= 2:
            vocab_size = to_int(operation.args[0], 0)
            embed_dim = to_int(operation.args[1], 0)
            return vocab_size * embed_dim
        return 0

    elif op_name == "multihead_attention":
        # Multi-head attention parameters
        dim = to_int(operation.kwargs.get("dim", 512), 512)
        # Q, K, V projections + output projection
        return 4 * dim * dim + 4 * dim

    elif op_name == "layer_norm":
        if input_shape:
            features = to_int(input_shape[-1], 0)
            return 2 * features  # gamma and beta
        return 0

    return 0


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
        current_shape = calculate_output_shape(current_shape, op)

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
        current_shape = calculate_output_shape(current_shape, op)
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
        current_shape = calculate_output_shape(current_shape, op)

        text = Text()
        text.append("  | ", style="dim")
        text.append(f"{op.operation}", style="yellow")
        if op.activation:
            text.append(f".{op.activation}", style="cyan")
        text.append(" >> ", style="dim")
        text.append(str(current_shape), style="green")

        console.print(text)

    console.print()
