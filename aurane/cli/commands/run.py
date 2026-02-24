"""
Run command for Aurane CLI.
"""

import sys
import subprocess
from pathlib import Path
from ..ui import console, RICH_AVAILABLE
from ..utils import validate_file
from ...compiler import compile_to_temp

def cmd_run(args):
    """Compile and run an Aurane file."""
    try:
        input_file = validate_file(args.input, [".aur"])
        source = input_file.read_text(encoding="utf-8")

        if RICH_AVAILABLE and console:
            console.print(f"[cyan]Compiling:[/cyan] {args.input}")
            
        # Compile to temporary file
        temp_path = compile_to_temp(source, backend=args.backend)

        if RICH_AVAILABLE and console:
            console.print(f"[green][OK][/green] Compiled to temporary file: [dim]{temp_path}[/dim]")
            console.print("[bold cyan]Running...[/bold cyan]")
            console.print("-" * 60)
        else:
            print(f"Compiled to temporary file: {temp_path}")
            print("Running...")
            print("-" * 60)

        # Run the generated Python file
        result = subprocess.run([sys.executable, str(temp_path)], cwd=input_file.parent)

        # Clean up temporary file
        if not args.keep_temp:
            temp_path.unlink()
        else:
            if RICH_AVAILABLE and console:
                console.print("-" * 60)
                console.print(f"[yellow]Temporary file kept at:[/yellow] {temp_path}")
            else:
                print("-" * 60)
                print(f"Temporary file kept at: {temp_path}")

        return result.returncode

    except Exception as e:
        if RICH_AVAILABLE and console:
            console.print(f"[red][FAIL] Error:[/red] {e}")
        else:
            print(f"Error: {e}")
        return 1
