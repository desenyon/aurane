"""
Profile command for Aurane CLI.
"""

from ..ui import console, RICH_AVAILABLE
from ..utils import validate_file
from ...parser import parse_aurane
from ...profiler import profile_model, format_profile


def cmd_profile(args):
    """Profile model performance."""
    use_rich = RICH_AVAILABLE and console is not None

    try:
        input_file = validate_file(args.input, [".aur"])
        source = input_file.read_text(encoding="utf-8")
        program = parse_aurane(source)

        if not program.models:
            if use_rich:
                console.print("[yellow]No models found in file.[/yellow]")
            else:
                print("No models found in file.")
            return 1

        for model in program.models:
            if use_rich:
                console.print(f"\n[bold cyan]Profiling Model:[/bold cyan] {model.name}")
            else:
                print(f"\nProfiling Model: {model.name}")
            profile = profile_model(model, batch_size=args.batch_size)
            if args.detailed:
                if use_rich:
                    console.print(format_profile(profile, detailed=True))
                else:
                    print(format_profile(profile, detailed=True))
            else:
                if use_rich:
                    console.print(format_profile(profile, detailed=False))
                else:
                    print(format_profile(profile, detailed=False))

        return 0

    except Exception as e:
        if use_rich:
            console.print(f"[red][FAIL] Error:[/red] {e}")
        else:
            print(f"Error: {e}")
        return 1
