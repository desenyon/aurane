"""
Watch command for Aurane CLI.
"""

from queue import Queue, Empty
import argparse
from pathlib import Path
from ..ui import console, RICH_AVAILABLE
from ..utils import validate_file
from .compile import cmd_compile


def _compile_args_from_watch_args(args):
    """Build a compile-compatible namespace from watch command args."""
    return argparse.Namespace(
        input=args.input,
        output=args.output,
        output_override=None,
        backend=args.backend,
        analyze=getattr(args, "analyze", False),
        validate=False,
        optimize=False,
        opt_level=1,
        format=False,
        show_ast=False,
        diff=False,
        quiet=False,
        verbose=False,
    )


def cmd_watch(args) -> int:
    """Watch mode - auto-recompile on changes."""
    if not RICH_AVAILABLE or console is None:
        print("Watch mode requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        console.print("[red]Watch mode requires 'watchdog' library.[/red]")
        console.print("Install with: pip install watchdog")
        return 1

    input_path = validate_file(args.input, [".aur"]).absolute()
    changes: Queue[None] = Queue()

    class AuraneFileHandler(FileSystemEventHandler):
        def on_any_event(self, event):
            if event.is_directory or event.event_type not in {
                "modified",
                "created",
                "deleted",
                "moved",
            }:
                return
            paths = (event.src_path, getattr(event, "dest_path", ""))
            if any(path and Path(path).absolute() == input_path for path in paths):
                changes.put(None)

    console.print(f"[cyan]Watching:[/cyan] {args.input}")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")
    observer = Observer()
    observer.schedule(AuraneFileHandler(), str(input_path.parent), recursive=False)
    observer.start()
    try:
        cmd_compile(_compile_args_from_watch_args(args))
        while True:
            try:
                changes.get(timeout=0.5)
            except Empty:
                continue
            # Compile after the final event in a burst, never drop its last save.
            while True:
                try:
                    changes.get(timeout=0.15)
                except Empty:
                    break
            console.print("\n[yellow][RELOAD] File changed, recompiling...[/yellow]")
            cmd_compile(_compile_args_from_watch_args(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped watching[/yellow]")
    finally:
        observer.stop()
        observer.join()
    return 0
