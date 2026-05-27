"""
Watch command for Aurane CLI.
"""

import time
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


def cmd_watch(args):
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

    input_path = validate_file(args.input, [".aur"])

    class AuraneFileHandler(FileSystemEventHandler):
        def __init__(self):
            self.last_compile = 0

        def on_modified(self, event):
            if event.src_path == str(input_path.absolute()):
                current = time.time()
                if current - self.last_compile < 0.5:
                    return
                self.last_compile = current

                console.print(f"\n[yellow][RELOAD] File changed, recompiling...[/yellow]")
                cmd_compile(_compile_args_from_watch_args(args))

    console.print(f"[cyan]Watching:[/cyan] {args.input}")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    # Initial compile
    cmd_compile(_compile_args_from_watch_args(args))

    event_handler = AuraneFileHandler()
    observer = Observer()
    observer.schedule(event_handler, str(input_path.parent), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        console.print("\n[yellow]Stopped watching[/yellow]")

    observer.join()
    return 0
