"""
Enhanced command-line interface for Aurane with rich formatting.

Provides beautiful, interactive commands with colors, progress bars, and tables.
"""

import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import tempfile
import shutil

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich import print as rprint
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown
    from rich.columns import Columns
    from rich.layout import Layout
    from rich.live import Live

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None  # type: ignore

from .compiler import compile_file, compile_source, CompilationError
from .parser import parse_aurane, ParseError
from .visualizer import visualize_model_architecture, print_model_summary


if RICH_AVAILABLE and Console is not None:
    console = Console()
else:
    console = None  # type: ignore


def print_banner():
    """Print Aurane banner."""
    if not RICH_AVAILABLE or console is None:
        print("=== AURANE ML DSL ===")
        return

    banner = """
[bold cyan]    █████╗ ██╗   ██╗██████╗  █████╗ ███╗   ██╗███████╗[/bold cyan]
[bold cyan]   ██╔══██╗██║   ██║██╔══██╗██╔══██╗████╗  ██║██╔════╝[/bold cyan]
[bold cyan]   ███████║██║   ██║██████╔╝███████║██╔██╗ ██║█████╗  [/bold cyan]
[bold cyan]   ██╔══██║██║   ██║██╔══██╗██╔══██║██║╚██╗██║██╔══╝  [/bold cyan]
[bold cyan]   ██║  ██║╚██████╔╝██║  ██║██║  ██║██║ ╚████║███████╗[/bold cyan]
[bold cyan]   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝[/bold cyan]
   
   [dim]ML-oriented DSL that transpiles to idiomatic Python[/dim]
   [dim]Version 0.1.0 • PyTorch Backend • MIT License[/dim]
    """
    console.print(Panel(banner, border_style="cyan", expand=False))


def validate_file(path: str, extensions: list = [".aur"]) -> Path:
    """Validate input file exists and has correct extension."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if extensions and file_path.suffix not in extensions:
        raise ValueError(f"Expected file with extension {extensions}, got {file_path.suffix}")

    return file_path


def get_file_stats(path: Path) -> Dict[str, Any]:
    """Get detailed file statistics."""
    stat = path.stat()
    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")

    return {
        "size": stat.st_size,
        "lines": len(lines),
        "non_empty_lines": len([l for l in lines if l.strip()]),
        "modified": stat.st_mtime,
    }


def cmd_compile_enhanced(args):
    """Enhanced compile command with rich output."""
    if not RICH_AVAILABLE or console is None:
        return cmd_compile_basic(args)

    try:
        input_file = validate_file(args.input, [".aur"])

        console.print(f"\n[bold cyan]Compiling:[/bold cyan] {args.input}")

        # Get input stats
        input_stats = get_file_stats(input_file)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Compiling...", total=100)

            # Read source
            progress.update(task, advance=20, description="[cyan]Reading source...")
            source = input_file.read_text(encoding="utf-8")

            # Parse
            progress.update(task, advance=30, description="[cyan]Parsing...")
            try:
                ast = parse_aurane(source)
            except ParseError as e:
                console.print(f"\n[red][FAIL] Parse Error:[/red]\n{e}")
                return 1

            # Validate if requested
            if args.validate:
                progress.update(task, advance=10, description="[cyan]Validating...")
                validation_errors = validate_ast(ast)
                if validation_errors:
                    console.print(f"\n[red][FAIL] Validation Errors:[/red]")
                    for err in validation_errors:
                        console.print(f"  - {err}")
                    return 1

            # Generate code
            progress.update(task, advance=30, description="[cyan]Generating code...")
            compile_file(args.input, args.output, backend=args.backend)

            # Format output if requested
            if args.format:
                progress.update(task, advance=5, description="[cyan]Formatting...")
                format_python_file(args.output)

            progress.update(task, advance=5, description="[green]Complete!")

        # Show success message with stats
        output_path = Path(args.output)
        output_stats = get_file_stats(output_path)

        # Build results table
        table = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
        table.add_row("[green][OK] Status", "[green bold]Success")
        table.add_row("Input", f"[dim]{args.input}[/dim]")
        table.add_row("Output", f"[dim]{args.output}[/dim]")
        table.add_row(
            "Input Size",
            f"[cyan]{input_stats['lines']}[/cyan] lines, [cyan]{input_stats['size']:,}[/cyan] bytes",
        )
        table.add_row(
            "Output Size",
            f"[cyan]{output_stats['lines']}[/cyan] lines, [cyan]{output_stats['size']:,}[/cyan] bytes",
        )
        table.add_row("Backend", f"[dim]{args.backend}[/dim]")
        table.add_row(
            "Compression", f"[yellow]{output_stats['size'] / input_stats['size']:.1f}x[/yellow]"
        )

        if args.analyze:
            table.add_row("", "")
            table.add_row("[bold]Analysis", "")
            table.add_row("Models", f"[cyan]{len(ast.models)}[/cyan]")
            table.add_row("Datasets", f"[cyan]{len(ast.datasets)}[/cyan]")
            table.add_row("Training Configs", f"[cyan]{len(ast.trains)}[/cyan]")

            # Count layers
            total_layers = sum(
                len(model.forward_block.operations) if model.forward_block else 0
                for model in ast.models
            )
            table.add_row("Total Layers", f"[cyan]{total_layers}[/cyan]")

        console.print(Panel(table, title="[bold green]Compilation Complete", border_style="green"))

        if args.show_ast:
            console.print()
            show_ast_tree(ast)

        if args.diff and Path(args.output).exists():
            show_code_diff(source, output_path.read_text())

        return 0

    except CompilationError as e:
        console.print(f"\n[red][FAIL] Compilation Error:[/red]\n{e}")
        if args.verbose:
            import traceback

            console.print("\n[dim]" + traceback.format_exc() + "[/dim]")
        return 1
    except Exception as e:
        console.print(f"\n[red][FAIL] Unexpected Error:[/red]\n{e}")
        if args.verbose:
            import traceback

            console.print("\n[dim]" + traceback.format_exc() + "[/dim]")
        return 1


def validate_ast(ast) -> list:
    """Validate AST for common issues."""
    errors = []

    # Check for models without forward blocks
    for model in ast.models:
        if not model.forward_block or not model.forward_block.operations:
            errors.append(f"Model '{model.name}' has no forward operations")

    # Check for training without corresponding model/dataset
    model_names = {m.name for m in ast.models}
    dataset_names = {d.name for d in ast.datasets}

    for train in ast.trains:
        if train.model_name not in model_names:
            errors.append(f"Training references undefined model '{train.model_name}'")
        if train.dataset_name not in dataset_names:
            errors.append(f"Training references undefined dataset '{train.dataset_name}'")

    return errors


def format_python_file(path: str):
    """Format Python file using black if available."""
    try:
        subprocess.run(["black", "-q", path], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # Black not available or failed, skip formatting


def show_code_diff(aur_code: str, py_code: str):
    """Show side-by-side comparison."""
    if not RICH_AVAILABLE or console is None:
        return

    console.print("\n[bold cyan]Code Comparison:[/bold cyan]\n")

    aur_syntax = Syntax(aur_code[:500], "python", theme="monokai", line_numbers=False)
    py_syntax = Syntax(py_code[:500], "python", theme="monokai", line_numbers=False)

    table = Table(show_header=True, box=None)
    table.add_column("Aurane Source", style="cyan")
    table.add_column("Generated Python", style="green")
    table.add_row(aur_syntax, py_syntax)

    console.print(table)


def cmd_compile_basic(args):
    """Basic compile command without rich."""
    try:
        compile_file(args.input, args.output, backend=args.backend)
        print(f"[OK] Successfully compiled {args.input} -> {args.output}")
        return 0
    except Exception as e:
        print(f"[FAIL] Error: {e}", file=sys.stderr)
        return 1


def cmd_inspect(args):
    """Inspect an Aurane file and show its structure."""
    if not RICH_AVAILABLE or console is None:
        print("Inspect command requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        input_file = validate_file(args.input, [".aur"])

        source = input_file.read_text(encoding="utf-8")
        ast = parse_aurane(source)

        file_stats = get_file_stats(input_file)

        console.print(f"\n[bold cyan]Inspecting:[/bold cyan] {args.input}")
        console.print(f"[dim]{file_stats['lines']} lines • {file_stats['size']:,} bytes[/dim]\n")

        show_ast_tree(ast)

        if ast.models and args.verbose:
            console.print("\n[bold cyan]=== Model Details ===[/bold cyan]\n")
            for model in ast.models:
                print_model_summary(model)

        # Show statistics
        if args.stats:
            show_statistics(ast)

        # Export to JSON if requested
        if args.export:
            export_ast_json(ast, args.export)
            console.print(f"\n[green][OK][/green] Exported AST to {args.export}")

        return 0

    except Exception as e:
        console.print(f"[red][FAIL] Error:[/red] {e}")
        if args.verbose:
            import traceback

            console.print("\n[dim]" + traceback.format_exc() + "[/dim]")
        return 1


def show_statistics(ast):
    """Display comprehensive statistics."""
    if not RICH_AVAILABLE or console is None:
        return

    console.print("\n[bold cyan]=== Statistics ===[/bold cyan]\n")

    stats_table = Table(show_header=True, box=None)
    stats_table.add_column("Category", style="cyan")
    stats_table.add_column("Count", justify="right", style="yellow")
    stats_table.add_column("Details", style="dim")

    stats_table.add_row("Imports", str(len(ast.uses)), ", ".join(u.module for u in ast.uses[:3]))
    stats_table.add_row(
        "Experiments", str(len(ast.experiments)), ", ".join(e.name for e in ast.experiments)
    )
    stats_table.add_row("Datasets", str(len(ast.datasets)), ", ".join(d.name for d in ast.datasets))
    stats_table.add_row("Models", str(len(ast.models)), ", ".join(m.name for m in ast.models))
    stats_table.add_row("Training Configs", str(len(ast.trains)), "")

    # Count operations
    total_ops = sum(len(m.forward_block.operations) if m.forward_block else 0 for m in ast.models)
    stats_table.add_row("Total Operations", str(total_ops), "")

    # Count unique operations
    op_types = set()
    for model in ast.models:
        if model.forward_block:
            for op in model.forward_block.operations:
                op_types.add(op.operation)
    stats_table.add_row("Unique Operations", str(len(op_types)), ", ".join(sorted(op_types)[:5]))

    console.print(stats_table)


def export_ast_json(ast, output_path: str):
    """Export AST to JSON format."""
    import dataclasses

    def serialize(obj):
        if dataclasses.is_dataclass(obj):
            # Handle both instances and types
            if isinstance(obj, type):
                return str(obj)
            return {k: serialize(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, list):
            return [serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        return obj

    data = serialize(ast)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def show_ast_tree(ast):
    """Display AST as a tree structure."""
    if not RICH_AVAILABLE or console is None:
        return

    tree = Tree("[bold cyan]Aurane Program", guide_style="cyan")

    if ast.uses:
        imports = tree.add("[yellow]Imports")
        for use in ast.uses:
            if use.alias:
                imports.add(f"[dim]{use.module} as {use.alias}[/dim]")
            else:
                imports.add(f"[dim]{use.module}[/dim]")

    if ast.experiments:
        experiments = tree.add("[yellow]Experiments")
        for exp in ast.experiments:
            exp_node = experiments.add(f"[green]{exp.name}[/green]")
            for key, value in exp.config.items():
                exp_node.add(f"[dim]{key} = {value}[/dim]")

    if ast.datasets:
        datasets = tree.add("[yellow]Datasets")
        for ds in ast.datasets:
            ds_node = datasets.add(f"[green]{ds.name}[/green]")
            if ds.source:
                ds_node.add(f"[dim]from {ds.source}[/dim]")
            for key, value in ds.config.items():
                ds_node.add(f"[dim]{key} = {value}[/dim]")

    if ast.models:
        models = tree.add("[yellow]Models")
        for model in ast.models:
            model_node = models.add(f"[green]{model.name}[/green]")
            if model.forward_block:
                forward = model_node.add("[blue]forward")
                for op in model.forward_block.operations:
                    op_str = f"{op.operation}({', '.join(map(str, op.args))})"
                    if op.activation:
                        op_str += f".{op.activation}"
                    forward.add(f"[dim]{op_str}[/dim]")

    if ast.trains:
        trains = tree.add("[yellow]Training")
        for train in ast.trains:
            train_node = trains.add(f"[green]{train.model_name} on {train.dataset_name}[/green]")
            for key, value in train.config.items():
                train_node.add(f"[dim]{key} = {value}[/dim]")

    console.print(tree)


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
                # Debounce - avoid multiple rapid recompiles
                current = time.time()
                if current - self.last_compile < 0.5:
                    return
                self.last_compile = current

                if console is not None:
                    console.print(f"\n[yellow][RELOAD] File changed, recompiling...[/yellow]")
                args_copy = argparse.Namespace(**vars(args))
                args_copy.analyze = False
                args_copy.show_ast = False
                args_copy.verbose = False
                cmd_compile_enhanced(args_copy)

    console.print(f"[cyan]Watching:[/cyan] {args.input}")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    # Initial compile
    cmd_compile_enhanced(args)

    # Start watching
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


def cmd_format(args):
    """Format Aurane source files."""
    if not RICH_AVAILABLE or console is None:
        print("Format command requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        files = (
            list(Path(args.path).rglob("*.aur")) if Path(args.path).is_dir() else [Path(args.path)]
        )

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

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            formatted.append(line)
            continue

        # Decrease indent for closing blocks
        if stripped.startswith(("def ", "model ", "dataset ", "train ", "experiment ")):
            indent_level = 0

        # Add proper indentation
        if "->" in stripped:
            indent_level = 2
        elif stripped.endswith(":"):
            formatted.append("    " * indent_level + stripped)
            indent_level += 1
            continue

        formatted.append("    " * indent_level + stripped)

    return "\n".join(formatted)


def cmd_lint(args):
    """Lint Aurane files for potential issues."""
    if not RICH_AVAILABLE or console is None:
        print("Lint command requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        file_path = validate_file(args.input, [".aur"])
        source = file_path.read_text()

        console.print(f"[cyan]Linting:[/cyan] {args.input}\n")

        issues = []

        # Parse and validate
        try:
            ast = parse_aurane(source)
            validation_errors = validate_ast(ast)
            issues.extend(("error", err) for err in validation_errors)
        except ParseError as e:
            issues.append(("error", f"Parse error: {e}"))

        # Check style issues
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 100:
                issues.append(("warning", f"Line {i}: Line too long ({len(line)} > 100)"))

            # Check trailing whitespace
            if line != line.rstrip():
                issues.append(("info", f"Line {i}: Trailing whitespace"))

            # Check inconsistent indentation
            if line and line[0] == " ":
                spaces = len(line) - len(line.lstrip())
                if spaces % 4 != 0:
                    issues.append(
                        ("warning", f"Line {i}: Inconsistent indentation ({spaces} spaces)")
                    )

        # Display results
        if not issues:
            console.print("[green][OK] No issues found[/green]")
            return 0

        # Group by severity
        errors = [msg for sev, msg in issues if sev == "error"]
        warnings = [msg for sev, msg in issues if sev == "warning"]
        infos = [msg for sev, msg in issues if sev == "info"]

        if errors:
            console.print(f"[red][FAIL] {len(errors)} error(s):[/red]")
            for err in errors:
                console.print(f"  - {err}")

        if warnings:
            console.print(f"\n[yellow][WARN] {len(warnings)} warning(s):[/yellow]")
            for warn in warnings:
                console.print(f"  - {warn}")

        if infos and args.verbose:
            console.print(f"\n[cyan][INFO] {len(infos)} info(s):[/cyan]")
            for info in infos:
                console.print(f"  - {info}")

        return 1 if errors else 0

    except Exception as e:
        console.print(f"[red][FAIL] Error:[/red] {e}")
        return 1


def cmd_benchmark(args):
    """Benchmark compilation performance."""
    if not RICH_AVAILABLE or console is None:
        print("Benchmark command requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        file_path = validate_file(args.input, [".aur"])
        source = file_path.read_text()

        console.print(f"[cyan]Benchmarking:[/cyan] {args.input}")
        console.print(f"[dim]Running {args.iterations} iterations...[/dim]\n")

        times = {"parse": [], "compile": [], "total": []}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Benchmarking...", total=args.iterations)

            for i in range(args.iterations):
                # Parse timing
                start = time.perf_counter()
                ast = parse_aurane(source)
                parse_time = time.perf_counter() - start
                times["parse"].append(parse_time)

                # Compile timing
                start = time.perf_counter()
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
                    compile_file(str(file_path), tmp.name)
                    tmp_path = tmp.name
                compile_time = time.perf_counter() - start
                times["compile"].append(compile_time)
                times["total"].append(parse_time + compile_time)

                # Cleanup
                Path(tmp_path).unlink()

                progress.update(task, advance=1)

        # Calculate statistics
        def stats(data):
            import statistics

            return {
                "mean": statistics.mean(data),
                "median": statistics.median(data),
                "stdev": statistics.stdev(data) if len(data) > 1 else 0,
                "min": min(data),
                "max": max(data),
            }

        # Display results
        table = Table(show_header=True, title="Benchmark Results")
        table.add_column("Phase", style="cyan")
        table.add_column("Mean", justify="right")
        table.add_column("Median", justify="right")
        table.add_column("Std Dev", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")

        for phase in ["parse", "compile", "total"]:
            s = stats(times[phase])
            table.add_row(
                phase.capitalize(),
                f"{s['mean']*1000:.2f}ms",
                f"{s['median']*1000:.2f}ms",
                f"{s['stdev']*1000:.2f}ms",
                f"{s['min']*1000:.2f}ms",
                f"{s['max']*1000:.2f}ms",
            )

        console.print(table)

        # Show file stats
        file_stats = get_file_stats(file_path)
        console.print(
            f"\n[dim]File: {file_stats['lines']} lines, {file_stats['size']:,} bytes[/dim]"
        )
        console.print(
            f"[dim]Throughput: {file_stats['lines'] / (stats(times['total'])['mean']):.0f} lines/sec[/dim]"
        )

        return 0

    except Exception as e:
        console.print(f"[red][FAIL] Error:[/red] {e}")
        return 1


def cmd_interactive(args):
    """Start interactive REPL mode."""
    if not RICH_AVAILABLE or console is None:
        print("Interactive mode requires 'rich' library. Install with: pip install rich")
        return 1

    print_banner()
    console.print("\n[cyan]Interactive Mode[/cyan] - Type [bold].help[/bold] for commands\n")

    code_buffer = []
    history = []

    while True:
        try:
            if not code_buffer:
                line = Prompt.ask("[bold cyan]aurane>[/bold cyan]")
            else:
                line = Prompt.ask("[bold cyan].......[/bold cyan]")

            if not line.strip():
                continue

            if line.startswith("."):
                # Meta commands
                cmd = line[1:].strip().lower()
                parts = cmd.split(maxsplit=1)
                cmd_name = parts[0]
                cmd_args = parts[1] if len(parts) > 1 else ""

                if cmd_name in ("exit", "quit", "q"):
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                elif cmd_name in ("help", "h", "?"):
                    show_repl_help()

                elif cmd_name in ("compile", "c"):
                    if code_buffer:
                        source = "\n".join(code_buffer)
                        try:
                            python_code = compile_source(source)
                            console.print("[green][OK] Compilation successful![/green]")

                            syntax = Syntax(
                                python_code, "python", theme="monokai", line_numbers=True
                            )
                            console.print(syntax)

                            history.append(("compile", source, python_code))
                        except Exception as e:
                            console.print(f"[red][FAIL] Error:[/red] {e}")
                    else:
                        console.print("[yellow]Buffer is empty[/yellow]")

                elif cmd_name in ("clear", "clr"):
                    code_buffer = []
                    console.print("[yellow]Buffer cleared[/yellow]")

                elif cmd_name in ("show", "s"):
                    if code_buffer:
                        syntax = Syntax(
                            "\n".join(code_buffer), "python", theme="monokai", line_numbers=True
                        )
                        console.print(syntax)
                    else:
                        console.print("[yellow]Buffer is empty[/yellow]")

                elif cmd_name in ("save", "export"):
                    if not cmd_args:
                        console.print("[red]Usage:[/red] .save <filename>")
                    else:
                        if code_buffer:
                            Path(cmd_args).write_text("\n".join(code_buffer))
                            console.print(f"[green][OK][/green] Saved to {cmd_args}")
                        else:
                            console.print("[yellow]Buffer is empty[/yellow]")

                elif cmd_name in ("load", "open"):
                    if not cmd_args:
                        console.print("[red]Usage:[/red] .load <filename>")
                    else:
                        try:
                            content = Path(cmd_args).read_text()
                            code_buffer = content.split("\n")
                            console.print(
                                f"[green][OK][/green] Loaded {len(code_buffer)} lines from {cmd_args}"
                            )
                        except FileNotFoundError:
                            console.print(f"[red][FAIL] Error:[/red] File not found: {cmd_args}")

                elif cmd_name in ("history", "hist"):
                    if history:
                        console.print("\n[cyan]Command History:[/cyan]")
                        for i, (action, src, _) in enumerate(history, 1):
                            console.print(f"  {i}. {action} ({len(src)} chars)")
                    else:
                        console.print("[yellow]No history[/yellow]")

                elif cmd_name in ("validate", "check"):
                    if code_buffer:
                        source = "\n".join(code_buffer)
                        try:
                            ast = parse_aurane(source)
                            errors = validate_ast(ast)
                            if errors:
                                console.print("[yellow][WARN] Validation warnings:[/yellow]")
                                for err in errors:
                                    console.print(f"  - {err}")
                            else:
                                console.print("[green][OK] Valid Aurane code[/green]")
                        except Exception as e:
                            console.print(f"[red][FAIL] Parse Error:[/red] {e}")
                    else:
                        console.print("[yellow]Buffer is empty[/yellow]")

                else:
                    console.print(
                        f"[red]Unknown command:[/red] {cmd_name} (type .help for commands)"
                    )

            else:
                code_buffer.append(line)

        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

    return 0


def show_repl_help():
    """Show REPL help."""
    if console is None:
        return
    help_text = """
[bold cyan]REPL Commands:[/bold cyan]

[yellow]Editing:[/yellow]
  .clear, .clr       Clear buffer
  .show, .s          Show buffer contents
  .save <file>       Save buffer to file
  .load <file>       Load file into buffer

[yellow]Compilation:[/yellow]
  .compile, .c       Compile buffer
  .validate, .check  Validate buffer without compiling

[yellow]History:[/yellow]
  .history, .hist    Show command history

[yellow]Other:[/yellow]
  .help, .h, .?      Show this help
  .exit, .quit, .q   Exit REPL

[dim]Tip: Type Aurane code line by line. Use .compile to see Python output.[/dim]
"""
    console.print(help_text)


def cmd_typecheck(args):
    """Type check an Aurane file."""
    if not RICH_AVAILABLE or console is None:
        print("Typecheck command requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        from .type_checker import check_types, format_type_errors

        file_path = validate_file(args.input, [".aur"])
        source = file_path.read_text()

        console.print(f"[cyan]Type checking:[/cyan] {args.input}\n")

        ast = parse_aurane(source)
        result = check_types(ast)

        if result.errors:
            console.print(f"[red][FAIL] {len(result.errors)} type error(s):[/red]")
            for err in result.errors:
                console.print(f"  - {err.location}: {err.message}")
                if err.suggestion:
                    console.print(f"    [dim]Suggestion: {err.suggestion}[/dim]")

        if result.warnings:
            console.print(f"\n[yellow][WARN] {len(result.warnings)} warning(s):[/yellow]")
            for warn in result.warnings:
                console.print(f"  - {warn.location}: {warn.message}")

        if result.is_valid and not result.warnings:
            console.print("[green][OK] No type errors found[/green]")

        # Show inferred shapes
        if args.verbose and result.inferred_types:
            console.print("\n[bold cyan]Inferred Shapes:[/bold cyan]")
            for model_name, shapes in result.inferred_types.items():
                console.print(f"\n  [yellow]{model_name}:[/yellow]")
                for layer, type_info in shapes.items():
                    console.print(f"    {layer}: {type_info}")

        return 0 if result.is_valid else 1

    except Exception as e:
        console.print(f"[red][FAIL] Error:[/red] {e}")
        return 1


def cmd_analyze(args):
    """Run semantic analysis on an Aurane file."""
    if not RICH_AVAILABLE or console is None:
        print("Analyze command requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        from .semantic_analyzer import analyze_semantics, IssueKind

        file_path = validate_file(args.input, [".aur"])
        source = file_path.read_text()

        console.print(f"[cyan]Analyzing:[/cyan] {args.input}\n")

        ast = parse_aurane(source)
        result = analyze_semantics(ast)

        errors = [i for i in result.issues if i.kind == IssueKind.ERROR]
        warnings = [i for i in result.issues if i.kind == IssueKind.WARNING]
        suggestions = [i for i in result.issues if i.kind == IssueKind.SUGGESTION]

        if errors:
            console.print(f"[red][FAIL] {len(errors)} error(s):[/red]")
            for issue in errors:
                console.print(f"  [{issue.code}] {issue.location}")
                console.print(f"    {issue.message}")

        if warnings:
            console.print(f"\n[yellow][WARN] {len(warnings)} warning(s):[/yellow]")
            for issue in warnings:
                console.print(f"  [{issue.code}] {issue.location}")
                console.print(f"    {issue.message}")

        if suggestions and args.verbose:
            console.print(f"\n[cyan][SUGGEST] {len(suggestions)} suggestion(s):[/cyan]")
            for issue in suggestions:
                console.print(f"  [{issue.code}] {issue.message}")
                if issue.fix:
                    console.print(f"    [dim]Fix: {issue.fix}[/dim]")

        if result.is_valid and not warnings:
            console.print("[green][OK] No semantic issues found[/green]")

        return 0 if result.is_valid else 1

    except Exception as e:
        console.print(f"[red][FAIL] Error:[/red] {e}")
        return 1


def cmd_profile(args):
    """Profile models in an Aurane file."""
    if not RICH_AVAILABLE or console is None:
        print("Profile command requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        from .profiler import profile_program, format_profile

        file_path = validate_file(args.input, [".aur"])
        source = file_path.read_text()

        console.print(f"[cyan]Profiling:[/cyan] {args.input}\n")

        ast = parse_aurane(source)
        profiles = profile_program(ast, batch_size=args.batch_size)

        for model_name, profile in profiles.items():
            summary = profile.summary()

            table = Table(title=f"Model: {model_name}", show_header=False, box=None)
            table.add_row("Input Shape", str(profile.input_shape))
            table.add_row("Output Shape", str(profile.output_shape))
            table.add_row("Parameters", f"[cyan]{summary['total_params_readable']}[/cyan]")
            table.add_row("FLOPs", f"[yellow]{summary['total_flops_readable']}[/yellow]")
            table.add_row("Memory (batch=1)", f"[green]{summary['total_memory_mb']:.2f} MB[/green]")
            table.add_row("Bottleneck", f"[red]{summary['bottleneck']}[/red]")

            console.print(Panel(table, border_style="cyan"))

            if args.detailed:
                layer_table = Table(title="Layer Details", show_header=True)
                layer_table.add_column("Layer", style="cyan")
                layer_table.add_column("Operation", style="yellow")
                layer_table.add_column("Output Shape", style="green")
                layer_table.add_column("FLOPs", justify="right")
                layer_table.add_column("%", justify="right")
                layer_table.add_column("Params", justify="right")

                for layer in profile.layers:
                    layer_table.add_row(
                        layer.name,
                        layer.operation,
                        str(layer.output_shape),
                        profile._format_flops(layer.flops),
                        f"{layer.percentage_flops:.1f}%",
                        f"{layer.params:,}",
                    )

                console.print(layer_table)

            console.print()

        return 0

    except Exception as e:
        console.print(f"[red][FAIL] Error:[/red] {e}")
        if args.verbose:
            import traceback

            console.print("\n[dim]" + traceback.format_exc() + "[/dim]")
        return 1


def cmd_optimize(args):
    """Optimize an Aurane file and show results."""
    if not RICH_AVAILABLE or console is None:
        print("Optimize command requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        from .optimizer import optimize_ast

        file_path = validate_file(args.input, [".aur"])
        source = file_path.read_text()

        console.print(f"[cyan]Optimizing:[/cyan] {args.input}")
        console.print(f"[dim]Optimization level: {args.level}[/dim]\n")

        ast = parse_aurane(source)
        result = optimize_ast(ast, level=args.level)

        # Show optimization stats
        table = Table(title="Optimization Results", show_header=False, box=None)
        table.add_row("Original Layers", str(result.stats["original_layers"]))
        table.add_row("Optimized Layers", str(result.stats["optimized_layers"]))
        table.add_row("Fusions", f"[green]{result.stats['fusions']}[/green]")
        table.add_row("Eliminations", f"[yellow]{result.stats['eliminations']}[/yellow]")

        console.print(Panel(table, border_style="green"))

        if result.applied_optimizations:
            console.print("\n[bold cyan]Applied Optimizations:[/bold cyan]")
            for opt in result.applied_optimizations:
                console.print(f"  - {opt}")
        else:
            console.print("\n[dim]No optimizations applied[/dim]")

        return 0

    except Exception as e:
        console.print(f"[red][FAIL] Error:[/red] {e}")
        return 1


def main():
    """Enhanced main entry point."""
    parser = argparse.ArgumentParser(
        prog="aurane",
        description="Aurane ML DSL - Beautiful ML code that transpiles to Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    if RICH_AVAILABLE:
        parser.epilog = """
Examples:
  aurane compile model.aur output.py --analyze
  aurane inspect model.aur --verbose --stats
  aurane typecheck model.aur --verbose
  aurane analyze model.aur
  aurane profile model.aur --detailed
  aurane optimize model.aur --level 2
  aurane watch model.aur output.py
  aurane format examples/
  aurane lint model.aur
  aurane benchmark model.aur
  aurane interactive
        """

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Compile command
    compile_parser = subparsers.add_parser("compile", help="Compile Aurane file to Python")
    compile_parser.add_argument("input", help="Input .aur file")
    compile_parser.add_argument("output", help="Output .py file")
    compile_parser.add_argument(
        "--backend", default="torch", choices=["torch"], help="Code generation backend"
    )
    compile_parser.add_argument(
        "--analyze", action="store_true", help="Show analysis after compilation"
    )
    compile_parser.add_argument("--show-ast", action="store_true", help="Display AST tree")
    compile_parser.add_argument("--validate", action="store_true", help="Validate before compiling")
    compile_parser.add_argument("--format", action="store_true", help="Format output with black")
    compile_parser.add_argument("--diff", action="store_true", help="Show code comparison")
    compile_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect Aurane file structure")
    inspect_parser.add_argument("input", help="Input .aur file")
    inspect_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed information"
    )
    inspect_parser.add_argument("--stats", action="store_true", help="Show statistics")
    inspect_parser.add_argument("--export", metavar="FILE", help="Export AST to JSON")

    # Type check command
    typecheck_parser = subparsers.add_parser("typecheck", help="Type check Aurane file")
    typecheck_parser.add_argument("input", help="Input .aur file")
    typecheck_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show inferred types"
    )

    # Semantic analysis command
    analyze_parser = subparsers.add_parser("analyze", help="Run semantic analysis")
    analyze_parser.add_argument("input", help="Input .aur file")
    analyze_parser.add_argument("--verbose", "-v", action="store_true", help="Show suggestions")

    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Profile model FLOPs and memory")
    profile_parser.add_argument("input", help="Input .aur file")
    profile_parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for memory calculation"
    )
    profile_parser.add_argument(
        "--detailed", "-d", action="store_true", help="Show layer-by-layer details"
    )
    profile_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize and analyze")
    optimize_parser.add_argument("input", help="Input .aur file")
    optimize_parser.add_argument(
        "--level", "-O", type=int, default=1, choices=[0, 1, 2], help="Optimization level"
    )

    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Watch file and auto-recompile")
    watch_parser.add_argument("input", help="Input .aur file")
    watch_parser.add_argument("output", help="Output .py file")
    watch_parser.add_argument(
        "--backend", default="torch", choices=["torch"], help="Code generation backend"
    )
    watch_parser.add_argument(
        "--analyze", action="store_true", help="Show analysis on each compile"
    )

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive REPL")

    # Format command
    format_parser = subparsers.add_parser("format", help="Format Aurane source files")
    format_parser.add_argument("path", help="File or directory to format")
    format_parser.add_argument("--check", action="store_true", help="Check without modifying")
    format_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Lint command
    lint_parser = subparsers.add_parser("lint", help="Lint Aurane files")
    lint_parser.add_argument("input", help="Input .aur file")
    lint_parser.add_argument("--verbose", "-v", action="store_true", help="Show all issues")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark compilation performance")
    benchmark_parser.add_argument("input", help="Input .aur file")
    benchmark_parser.add_argument(
        "--iterations", "-n", type=int, default=10, help="Number of iterations"
    )

    # Run command (original)
    run_parser = subparsers.add_parser("run", help="Compile and run an Aurane file")
    run_parser.add_argument("input", help="Input .aur file")
    run_parser.add_argument(
        "--backend", default="torch", choices=["torch"], help="Code generation backend"
    )
    run_parser.add_argument("--keep-temp", action="store_true", help="Keep temporary compiled file")

    args = parser.parse_args()

    if not args.command:
        if RICH_AVAILABLE:
            print_banner()
        parser.print_help()
        return 1

    # Route to appropriate handler
    if args.command == "compile":
        return cmd_compile_enhanced(args) if RICH_AVAILABLE else cmd_compile_basic(args)
    elif args.command == "inspect":
        return cmd_inspect(args)
    elif args.command == "typecheck":
        return cmd_typecheck(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "profile":
        return cmd_profile(args)
    elif args.command == "optimize":
        return cmd_optimize(args)
    elif args.command == "watch":
        return cmd_watch(args)
    elif args.command == "interactive":
        return cmd_interactive(args)
    elif args.command == "format":
        return cmd_format(args)
    elif args.command == "lint":
        return cmd_lint(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "run":
        # Keep original run command
        from .cli import cmd_run

        return cmd_run(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
