"""
Compilation command for Aurane CLI.
"""

import sys
from pathlib import Path
from ..ui import console, RICH_AVAILABLE, get_progress
from ..utils import validate_file, get_file_stats
from ..compilation import compiler_options, json_diagnostics, report_compilation_error
from ...compiler import compile_source, CompilationError, write_compiled_output
from ...parser import parse_aurane, ParseError

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console(stderr=True)
except ImportError:
    pass


def cmd_compile(args):
    """Enhanced compile command with rich output."""
    if (
        json_diagnostics(args)
        or (not args.output and not args.output_override)
        or not RICH_AVAILABLE
        or console is None
    ):
        return cmd_compile_basic(args)

    try:
        input_file = validate_file(args.input, [".aur"])
        output_path = Path(args.output_override) if args.output_override else None
        if output_path is None and args.output:
            output_path = Path(args.output)

        if not args.quiet:
            console.print(f"\n[bold cyan]Compiling:[/bold cyan] {args.input}")

        input_stats = (
            get_file_stats(input_file) if (output_path is not None and not args.quiet) else None
        )
        progress = get_progress() if (not args.quiet) else None

        source = input_file.read_text(encoding="utf-8")

        if args.show_ast and (not args.quiet):
            try:
                ast = parse_aurane(source)
                console.print("\n[bold cyan]AST[/bold cyan]")
                console.print(ast)
            except ParseError as e:
                console.print(f"\n[red][FAIL] Parse Error:[/red]\n{e}")
                return 1

        if progress:
            with progress:
                task = progress.add_task("[cyan]Compiling...", total=100)
                progress.update(task, advance=20, description="[cyan]Analyzing & optimizing...")
                python_code = compile_source(
                    source,
                    **compiler_options(args),
                )
                progress.update(task, advance=70, description="[cyan]Post-processing output...")

                if args.format:
                    python_code = _maybe_black_format(python_code, args)

                progress.update(task, advance=10, description="[green]Complete!")
        else:
            python_code = compile_source(
                source,
                **compiler_options(args),
            )
            if args.format:
                python_code = _maybe_black_format(python_code, args)

        # Diff (if writing to file)
        if output_path is not None and args.diff and output_path.exists():
            _print_diff(output_path.read_text(encoding="utf-8"), python_code, output_path)

        if output_path is not None:
            write_compiled_output(input_file, output_path, python_code)

            if not args.quiet:
                output_stats = get_file_stats(output_path)
                table = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
                table.add_row("[green][OK] Status", "[green bold]Success")
                table.add_row("Input", f"[dim]{args.input}[/dim]")
                table.add_row("Output", f"[dim]{output_path}[/dim]")
                if input_stats and input_stats["size"] > 0:
                    table.add_row(
                        "Compression",
                        f"[yellow]{output_stats['size'] / input_stats['size']:.1f}x[/yellow]",
                    )
                console.print(
                    Panel(table, title="[bold green]Compilation Complete", border_style="green")
                )
            return 0

        # stdout mode
        sys.stdout.write(python_code)
        if not args.quiet:
            console.print("\n[green][OK] Compilation output written to stdout[/green]")
        return 0

    except CompilationError as e:
        report_compilation_error(e, args)
        return 1
    except Exception as e:
        report_compilation_error(e, args)
        return 1


def cmd_compile_basic(args):
    """Basic compile command without rich."""
    try:
        input_file = validate_file(args.input, [".aur"])
        output_path = Path(args.output_override) if args.output_override else None
        if output_path is None and args.output:
            output_path = Path(args.output)

        source = input_file.read_text(encoding="utf-8")
        if args.show_ast and not args.quiet and not json_diagnostics(args):
            print(parse_aurane(source), file=sys.stderr)
        python_code = compile_source(
            source,
            **compiler_options(args),
        )
        if args.format:
            python_code = _maybe_black_format(python_code, args)

        if output_path is not None:
            if args.diff and output_path.exists():
                _print_diff(output_path.read_text(encoding="utf-8"), python_code, output_path)
            write_compiled_output(input_file, output_path, python_code)
            if not args.quiet and not json_diagnostics(args):
                print(f"OK: {args.input} -> {output_path}", file=sys.stderr)
        else:
            sys.stdout.write(python_code)
        return 0
    except Exception as e:
        report_compilation_error(e, args)
        return 1


def _maybe_black_format(python_code: str, args) -> str:
    """Format code with black if available."""
    try:
        import black

        mode = black.FileMode()
        try:
            return black.format_file_contents(python_code, fast=False, mode=mode)
        except black.NothingChanged:
            return python_code
    except ImportError as e:
        raise CompilationError(
            "Requested --format but 'black' is not installed. Install dev dependencies: pip install aurane[dev]",
            stage="format",
        ) from e
    except Exception as error:
        raise CompilationError(f"Python formatting failed: {error}", stage="format") from error


def _print_diff(old_code: str, new_code: str, output_path: Path) -> None:
    """Print a unified diff between old and new code."""
    import difflib

    diff = difflib.unified_diff(
        old_code.splitlines(True),
        new_code.splitlines(True),
        fromfile=str(output_path),
        tofile=str(output_path) + " (new)",
    )
    sys.stdout.writelines(diff)
