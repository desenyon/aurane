"""
Profile command for Aurane CLI.
"""

from ..ui import console, RICH_AVAILABLE
from ..utils import validate_file
from ...parser import parse_aurane
from ...profiler import ModelProfiler


def cmd_profile(args):
    """Profile model performance."""
    if not RICH_AVAILABLE or console is None:
        print("Profile command requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        input_file = validate_file(args.input, [".aur"])
        source = input_file.read_text(encoding="utf-8")
        program = parse_aurane(source)

        if not program.models:
            console.print("[yellow]No models found in file.[/yellow]")
            return 1

        profiler = ModelProfiler()
        for model in program.models:
            console.print(f"\n[bold cyan]Profiling Model:[/bold cyan] {model.name}")
            profile = profiler.profile_model(model)
            # Assuming profile has a method or we can print it
            console.print(profile)

        return 0

    except Exception as e:
        console.print(f"[red][FAIL] Error:[/red] {e}")
        return 1
