"""
Inspect command for Aurane CLI.
"""

import json
from dataclasses import asdict
from pathlib import Path
from ..ui import console, RICH_AVAILABLE
from ..utils import validate_file, get_file_stats
from ...parser import parse_aurane
from ...visualizer import print_model_summary
from ...ast import ForwardGraphBlock

try:
    from rich.tree import Tree
    from rich.table import Table
except ImportError:
    pass


def cmd_inspect(args):
    """Inspect an Aurane file and show its structure."""
    if not RICH_AVAILABLE or console is None:
        print("Inspect command requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        input_file = validate_file(args.input, [".aur"])
        source = input_file.read_text(encoding="utf-8")
        ast = parse_aurane(source)

        file_stats = get_file_stats(input_file)
        console.print(f"\n[bold cyan]Inspecting:[/bold cyan] {args.input}")
        console.print(f"[dim]{file_stats['lines']} lines • {file_stats['size']:,} bytes[/dim]\n")

        show_ast_tree(ast)

        if args.stats:
            show_program_stats(ast)

        if args.export:
            export_path = Path(args.export)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_text(json.dumps(asdict(ast), indent=2), encoding="utf-8")
            console.print(f"\n[green][OK][/green] Exported AST to {export_path}")

        if ast.models and args.verbose:
            console.print("\n[bold cyan]=== Model Details ===[/bold cyan]\n")
            for model in ast.models:
                print_model_summary(model)

        return 0

    except Exception as e:
        console.print(f"[red][FAIL] Error:[/red] {e}")
        return 1


def show_ast_tree(ast):
    """Display AST as a tree structure."""
    if not RICH_AVAILABLE or console is None:
        return

    tree = Tree("[bold cyan]Aurane Program", guide_style="cyan")

    if ast.uses:
        imports = tree.add("[yellow]Imports")
        for use in ast.uses:
            if use.alias:
                imports.add(f"[dim]{use.module} as {use.alias}[/dim]")
            else:
                imports.add(f"[dim]{use.module}[/dim]")

    if ast.models:
        models = tree.add("[yellow]Models")
        for model in ast.models:
            model_node = models.add(f"[green]{model.name}[/green]")
            if model.forward_block:
                forward = model_node.add("[blue]forward")
                if isinstance(model.forward_block, ForwardGraphBlock):
                    for node in model.forward_block.nodes:
                        op = node.operation
                        if op is None:
                            continue
                        inputs_str = ", ".join(map(str, node.inputs))
                        extra_args = ""
                        if op.args:
                            extra_args = ", " + ", ".join(map(str, op.args))
                        op_str = f"{node.target} = {op.operation}({inputs_str}{extra_args})"
                        if op.kwargs:
                            kw_str = ", ".join(f"{k}={v}" for k, v in op.kwargs.items())
                            op_str += f" [{kw_str}]"
                        if op.activation:
                            op_str += f".{op.activation}"
                        forward.add(f"[dim]{op_str}[/dim]")
                else:
                    for op in model.forward_block.operations:
                        op_str = f"{op.operation}({', '.join(map(str, op.args))})"
                        if op.activation:
                            op_str += f".{op.activation}"
                        forward.add(f"[dim]{op_str}[/dim]")

    console.print(tree)


def show_program_stats(ast):
    """Display aggregate program counts."""
    if not RICH_AVAILABLE or console is None:
        return

    table = Table(title="Program Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Item", style="cyan")
    table.add_column("Count", justify="right", style="green")

    table.add_row("Imports", str(len(ast.uses)))
    table.add_row("Variables", str(len(ast.variables)))
    table.add_row("Experiments", str(len(ast.experiments)))
    table.add_row("Datasets", str(len(ast.datasets)))
    table.add_row("Models", str(len(ast.models)))
    table.add_row("Training blocks", str(len(ast.trains)))
    table.add_row("GAN training blocks", str(len(ast.train_gans)))

    console.print()
    console.print(table)
