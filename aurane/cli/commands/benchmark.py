"""
Benchmark command for Aurane CLI.
"""

import time
import tempfile
import os
import json
from ..ui import console, RICH_AVAILABLE
from ..utils import validate_file
from ...parser import parse_aurane
from ...compiler import compile_source

try:
    from rich.table import Table
except ImportError:
    pass


def cmd_benchmark(args):
    """Measure parsing separately from complete cold and cached compilation."""
    json_output = getattr(args, "json", False)
    try:
        if args.iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        file_path = validate_file(args.input, [".aur"]).resolve()
        source = file_path.read_text(encoding="utf-8")
        times = {"parse": [], "cold_compile": [], "warm_compile": []}
        original_directory = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="aurane-benchmark-") as directory:
            try:
                os.chdir(directory)
                compile_source(source)  # Populate the isolated warm cache outside timing.
                for _ in range(args.iterations):
                    start = time.perf_counter()
                    parse_aurane(source)
                    times["parse"].append(time.perf_counter() - start)
                    start = time.perf_counter()
                    compile_source(source, disable_cache=True)
                    times["cold_compile"].append(time.perf_counter() - start)
                    start = time.perf_counter()
                    compile_source(source)
                    times["warm_compile"].append(time.perf_counter() - start)
            finally:
                os.chdir(original_directory)
        if json_output:
            print(json.dumps({"ok": True, "seconds": times, "iterations": args.iterations}))
        elif RICH_AVAILABLE and console is not None:
            show_benchmark_results(times, file_path)
        else:
            print(times)
        return 0
    except Exception as error:
        if json_output:
            print(json.dumps({"ok": False, "error": str(error)}))
        elif RICH_AVAILABLE and console is not None:
            console.print(f"[red][FAIL] Error:[/red] {error}")
        else:
            print(f"Error: {error}")
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

    for phase in times:
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
