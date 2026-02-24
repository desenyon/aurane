"""
Clean command for Aurane CLI.
"""

import os
import shutil
from pathlib import Path
from ..ui import console, RICH_AVAILABLE


def cmd_clean(args):
    """Remove build artifacts and temporary files."""
    if not RICH_AVAILABLE or console is None:
        print("Clean command requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        path = Path(args.path)
        if not path.exists():
            console.print(f"[yellow]Directory {path} does not exist.[/yellow]")
            return 0

        console.print(f"[cyan]Cleaning artifacts in:[/cyan] {path.absolute()}\n")

        removed_files = 0
        removed_dirs = 0

        # Define what to clean
        extensions_to_clean = [
            ".pyc",
            ".pyo",
            ".pyd",
            ".so",
            ".aur.py",
        ]  # Maybe generated .py files but that's risky. Let's stick to pycache and specific temp patterns.
        dirs_to_clean = ["__pycache__", ".pytest_cache", ".aurane_cache"]

        for p in path.rglob("*"):
            if p.is_file():
                if p.suffix in extensions_to_clean or p.name.endswith("_temp.py"):
                    if args.dry_run:
                        console.print(f"[dim]Would remove file:[/dim] {p}")
                    else:
                        p.unlink()
                        if args.verbose:
                            console.print(f"[dim]Removed file:[/dim] {p}")
                    removed_files += 1
            elif p.is_dir() and p.name in dirs_to_clean:
                if args.dry_run:
                    console.print(f"[dim]Would remove directory:[/dim] {p}")
                else:
                    shutil.rmtree(p)
                    if args.verbose:
                        console.print(f"[dim]Removed directory:[/dim] {p}")
                removed_dirs += 1

        if args.dry_run:
            console.print(
                f"\n[yellow]Dry run: Would remove {removed_files} files and {removed_dirs} directories.[/yellow]"
            )
        else:
            console.print(
                f"\n[green]Cleaned {removed_files} files and {removed_dirs} directories.[/green]"
            )

        return 0

    except Exception as e:
        console.print(f"[red][FAIL] Error:[/red] {e}")
        return 1
