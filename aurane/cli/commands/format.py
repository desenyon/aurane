"""
Format command for Aurane CLI.
"""

from pathlib import Path
from ..ui import console, RICH_AVAILABLE


def cmd_format(args):
    """Format Aurane source files."""
    if not RICH_AVAILABLE or console is None:
        print("Format command requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        path = Path(args.path)
        files = list(path.rglob("*.aur")) if path.is_dir() else [path]

        console.print(f"[cyan]Formatting {len(files)} file(s)...[/cyan]\n")

        formatted_count = 0
        for file in files:
            original = file.read_text()
            formatted = format_aurane_code(original)

            if original != formatted:
                if not args.check:
                    file.write_text(formatted)
                    console.print(f"[green][OK][/green] Formatted {file}")
                else:
                    console.print(f"[yellow][!][/yellow] Would format {file}")
                formatted_count += 1
            else:
                if args.verbose:
                    console.print(f"[dim]  {file} (no changes)[/dim]")

        if formatted_count > 0:
            console.print(f"\n[green]Formatted {formatted_count} file(s)[/green]")
        else:
            console.print(f"\n[dim]All files already formatted[/dim]")

        return 0 if not args.check or formatted_count == 0 else 1

    except Exception as e:
        console.print(f"[red][FAIL] Error:[/red] {e}")
        return 1


def format_aurane_code(code: str) -> str:
    """Format Aurane code with consistent style."""
    lines = code.split("\n")
    formatted = []
    indent_level = 0

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            formatted.append(line)
            continue

        if stripped.startswith(("def ", "model ", "dataset ", "train ", "experiment ")):
            indent_level = 0

        if "->" in stripped:
            indent_level = 2
        elif stripped.endswith(":"):
            formatted.append("    " * indent_level + stripped)
            indent_level += 1
            continue

        formatted.append("    " * indent_level + stripped)

    return "\n".join(formatted)
