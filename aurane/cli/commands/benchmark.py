"""
Benchmark command for Aurane CLI.
"""

import time
import tempfile
from pathlib import Path
from ..ui import console, RICH_AVAILABLE, get_progress
from ..utils import validate_file, get_file_stats
from ...parser import parse_aurane
from ...compiler import compile_file

try:
    from rich.table import Table
    from rich.progress import SpinnerColumn, TextColumn, BarColumn
except ImportError:
    pass


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

        progress = get_progress()
        if progress:
            with progress:
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

                    Path(tmp_path).unlink()
                    progress.update(task, advance=1)

        # Calculate and show statistics
        show_benchmark_results(times, file_path)
        return 0

    except Exception as e:
        console.print(f"[red][FAIL] Error:[/red] {e}")
        return 1


def show_benchmark_results(times, file_path):
    """Display benchmark results in a table."""
    import statistics

    table = Table(show_header=True, title="Benchmark Results")
    table.add_column("Phase", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for phase in ["parse", "compile", "total"]:
        data = times[phase]
        table.add_row(
            phase.capitalize(),
            f"{statistics.mean(data)*1000:.2f}ms",
            f"{statistics.median(data)*1000:.2f}ms",
            f"{statistics.stdev(data)*1000:.2f}ms" if len(data) > 1 else "0.00ms",
            f"{min(data)*1000:.2f}ms",
            f"{max(data)*1000:.2f}ms",
        )

    console.print(table)
