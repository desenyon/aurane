"""
Compilation command for Aurane CLI.
"""

import sys
import subprocess
from pathlib import Path
from ..ui import console, RICH_AVAILABLE, get_progress
from ..utils import validate_file, get_file_stats
from ...compiler import compile_file, CompilationError
from ...parser import parse_aurane, ParseError

try:
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    pass

def cmd_compile(args):
    """Enhanced compile command with rich output."""
    if not RICH_AVAILABLE or console is None:
        return cmd_compile_basic(args)

    try:
        input_file = validate_file(args.input, [".aur"])
        console.print(f"\n[bold cyan]Compiling:[/bold cyan] {args.input}")

        input_stats = get_file_stats(input_file)
        progress = get_progress()
        
        if progress:
            with progress:
                task = progress.add_task("[cyan]Compiling...", total=100)
                
                progress.update(task, advance=20, description="[cyan]Reading source...")
                source = input_file.read_text(encoding="utf-8")

                progress.update(task, advance=30, description="[cyan]Parsing...")
                try:
                    ast = parse_aurane(source)
                except ParseError as e:
                    console.print(f"\n[red][FAIL] Parse Error:[/red]\n{e}")
                    return 1

                progress.update(task, advance=30, description="[cyan]Generating code...")
                compile_file(args.input, args.output, backend=args.backend)

                progress.update(task, advance=20, description="[green]Complete!")
        else:
            compile_file(args.input, args.output, backend=args.backend)
            ast = parse_aurane(input_file.read_text())

        # Success message and stats
        output_path = Path(args.output)
        output_stats = get_file_stats(output_path)

        table = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
        table.add_row("[green][OK] Status", "[green bold]Success")
        table.add_row("Input", f"[dim]{args.input}[/dim]")
        table.add_row("Output", f"[dim]{args.output}[/dim]")
        table.add_row("Compression", f"[yellow]{output_stats['size'] / input_stats['size']:.1f}x[/yellow]")
        
        console.print(Panel(table, title="[bold green]Compilation Complete", border_style="green"))
        return 0

    except CompilationError as e:
        console.print(f"\n[red][FAIL] Compilation Error:[/red]\n{e}")
        return 1
    except Exception as e:
        console.print(f"\n[red][FAIL] Unexpected Error:[/red]\n{e}")
        return 1

def cmd_compile_basic(args):
    """Basic compile command without rich."""
    try:
        compile_file(args.input, args.output, backend=args.backend)
        print(f"[OK] Successfully compiled {args.input} -> {args.output}")
        return 0
    except Exception as e:
        print(f"[FAIL] Error: {e}", file=sys.stderr)
        return 1
