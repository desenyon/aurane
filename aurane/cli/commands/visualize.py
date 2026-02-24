"""
Visualize command for Aurane CLI.
"""

from ..ui import console, RICH_AVAILABLE
from ..utils import validate_file
from ...parser import parse_aurane
from ...visualizer import visualize_model_architecture

def cmd_visualize(args):
    """Visualize model architecture."""
    if not RICH_AVAILABLE or console is None:
        print("Visualize command requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        input_file = validate_file(args.input, [".aur"])
        source = input_file.read_text(encoding="utf-8")
        program = parse_aurane(source)

        if not program.models:
            console.print("[yellow]No models found in file.[/yellow]")
            return 1

        for model in program.models:
            console.print(f"\n[bold cyan]Visualizing Architecture:[/bold cyan] {model.name}")
            visualize_model_architecture(model)

        return 0

    except Exception as e:
        console.print(f"[red][FAIL] Error:[/red] {e}")
        return 1
