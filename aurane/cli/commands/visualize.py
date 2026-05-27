"""
Visualize command for Aurane CLI.
"""

from ..ui import console, RICH_AVAILABLE
from ..utils import validate_file
from ...parser import parse_aurane
from ...visualizer import (
    visualize_model_architecture,
    render_model_architecture_mermaid,
    render_model_architecture_dot,
)


def cmd_visualize(args):
    """Visualize model architecture."""
    rich_required = args.format == "rich"
    if rich_required and (not RICH_AVAILABLE or console is None):
        print("Format 'rich' requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        input_file = validate_file(args.input, [".aur"])
        source = input_file.read_text(encoding="utf-8")
        program = parse_aurane(source)

        if not program.models:
            console.print("[yellow]No models found in file.[/yellow]")
            return 1

        for model in program.models:
            if args.format == "rich":
                console.print(f"\n[bold cyan]Visualizing Architecture:[/bold cyan] {model.name}")
                visualize_model_architecture(model, output_file=args.output)
            elif args.format == "mermaid":
                mermaid = render_model_architecture_mermaid(model)
                if args.output:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(mermaid)
                    if console is not None:
                        console.print(f"[green][OK][/green] Wrote Mermaid to {args.output}")
                    else:
                        print(f"OK: wrote Mermaid to {args.output}")
                else:
                    if console is not None:
                        console.print(mermaid, markup=False)
                    else:
                        print(mermaid)
            elif args.format == "dot":
                dot = render_model_architecture_dot(model)
                if args.output:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(dot)
                    if console is not None:
                        console.print(f"[green][OK][/green] Wrote DOT to {args.output}")
                    else:
                        print(f"OK: wrote DOT to {args.output}")
                else:
                    if console is not None:
                        console.print(dot, markup=False)
                    else:
                        print(dot)
            else:
                if console is not None:
                    console.print(f"[red][FAIL][/red] Unknown format: {args.format}")
                else:
                    print(f"Unknown format: {args.format}")
                return 1

        return 0

    except Exception as e:
        if console is not None:
            console.print(f"[red][FAIL] Error:[/red] {e}")
        else:
            print(f"Error: {e}")
        return 1
