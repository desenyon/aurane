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

        if path.is_symlink() or not path.is_dir():
            raise ValueError("Clean requires a real directory, not a file or symlink")
        if (path / "pyvenv.cfg").exists() or path.name in (
            ".git",
            ".venv",
            "venv",
            "env",
            "node_modules",
        ):
            raise ValueError("Refusing to clean a dependency or repository metadata directory")

        cache_dirs = {"__pycache__", ".pytest_cache", ".aurane_cache"}
        excluded_dirs = {".git", ".venv", "venv", "env", "node_modules"}
        for root, dirs, files in os.walk(path, followlinks=False):
            parent = Path(root)
            for name in list(dirs):
                directory = parent / name
                if (
                    directory.is_symlink()
                    or name in excluded_dirs
                    or (directory / "pyvenv.cfg").exists()
                ):
                    dirs.remove(name)
                elif name in cache_dirs:
                    dirs.remove(name)
                    if args.dry_run:
                        console.print(f"[dim]Would remove directory:[/dim] {directory}")
                    else:
                        shutil.rmtree(directory)
                        if args.verbose:
                            console.print(f"[dim]Removed directory:[/dim] {directory}")
                    removed_dirs += 1
            for name in files:
                artifact = parent / name
                if artifact.is_symlink() or artifact.suffix not in (".pyc", ".pyo"):
                    continue
                if args.dry_run:
                    console.print(f"[dim]Would remove file:[/dim] {artifact}")
                else:
                    artifact.unlink()
                    if args.verbose:
                        console.print(f"[dim]Removed file:[/dim] {artifact}")
                removed_files += 1

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
