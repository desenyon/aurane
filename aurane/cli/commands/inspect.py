"""
Inspect command for Aurane CLI.
"""

from pathlib import Path
from ..ui import console, RICH_AVAILABLE
from ..utils import validate_file, get_file_stats
from ...parser import parse_aurane
from ...visualizer import print_model_summary

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
                for op in model.forward_block.operations:
                    op_str = f"{op.operation}({', '.join(map(str, op.args))})"
                    if op.activation:
                        op_str += f".{op.activation}"
                    forward.add(f"[dim]{op_str}[/dim]")

    console.print(tree)
