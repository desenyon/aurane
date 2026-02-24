"""
Modular entry point for Aurane CLI.
"""

import sys
import argparse
from .ui import print_banner, console
from .commands.compile import cmd_compile
from .commands.inspect import cmd_inspect
from .commands.visualize import cmd_visualize
from .commands.profile import cmd_profile
from .commands.run import cmd_run
from .commands.format import cmd_format
from .commands.lint import cmd_lint
from .commands.benchmark import cmd_benchmark
from .commands.watch import cmd_watch
from .commands.interactive import cmd_interactive
from .commands.init import cmd_init
from .commands.clean import cmd_clean

def main():
    """Main entry point for the modular CLI."""
    parser = argparse.ArgumentParser(
        prog="aurane",
        description="Aurane ML DSL - Modern, high-performance ML transpiler"
    )
    
    parser.add_argument("--version", action="version", version="Aurane 1.0.0")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Compile command
    compile_parser = subparsers.add_parser("compile", help="Compile .aur to Python")
    compile_parser.add_argument("input", help="Input .aur file")
    compile_parser.add_argument("output", help="Output .py file")
    compile_parser.add_argument("--backend", default="torch", choices=["torch"], help="Transpiler backend")
    compile_parser.add_argument("--analyze", action="store_true", help="Analyze model during compilation")
    compile_parser.add_argument("--validate", action="store_true", help="Perform static analysis")
    compile_parser.add_argument("--format", action="store_true", help="Format output using black")
    compile_parser.add_argument("--show-ast", action="store_true", help="Show AST tree")
    compile_parser.add_argument("--diff", action="store_true", help="Show code comparison")
    
    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect model structure")
    inspect_parser.add_argument("input", help="Input .aur file")
    inspect_parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose details")
    
    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize model architecture")
    visualize_parser.add_argument("input", help="Input .aur file")
    
    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Profile model performance")
    profile_parser.add_argument("input", help="Input .aur file")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Compile and run immediately")
    run_parser.add_argument("input", help="Input .aur file")
    run_parser.add_argument("--backend", default="torch", choices=["torch"], help="Transpiler backend")
    run_parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary compiled Python file")

    # Format command
    format_parser = subparsers.add_parser("format", help="Format Aurane source files")
    format_parser.add_argument("path", help="File or directory to format")
    format_parser.add_argument("--check", action="store_true", help="Check if files need formatting")
    format_parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose details")

    # Lint command
    lint_parser = subparsers.add_parser("lint", help="Lint Aurane files")
    lint_parser.add_argument("input", help="Input .aur file")
    lint_parser.add_argument("--auto-fix", action="store_true", help="Automatically fix common issues (whitespace, colons)")
    lint_parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose details")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark compilation")
    benchmark_parser.add_argument("input", help="Input .aur file")
    benchmark_parser.add_argument("-i", "--iterations", type=int, default=10, help="Number of iterations")

    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Watch file and recompile")
    watch_parser.add_argument("input", help="Input .aur file")
    watch_parser.add_argument("output", help="Output .py file")
    watch_parser.add_argument("--backend", default="torch", choices=["torch"], help="Transpiler backend")

    # Interactive command
    subparsers.add_parser("interactive", help="Start interactive REPL")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Scaffold a new Aurane project")
    init_parser.add_argument("name", help="Name of the project directory")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Remove build artifacts")
    clean_parser.add_argument("path", nargs="?", default=".", help="Directory to clean (default: .)")
    clean_parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without actually deleting")
    clean_parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose details")
    
    # Handle help before argparse to use our custom rich help
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help")):
        args = argparse.Namespace(command=None)
    else:
        args = parser.parse_args()
    
    if args.command == "compile":
        return cmd_compile(args)
    elif args.command == "inspect":
        return cmd_inspect(args)
    elif args.command == "visualize":
        return cmd_visualize(args)
    elif args.command == "profile":
        return cmd_profile(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "format":
        return cmd_format(args)
    elif args.command == "lint":
        return cmd_lint(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "watch":
        return cmd_watch(args)
    elif args.command == "interactive":
        return cmd_interactive(args)
    elif args.command == "init":
        return cmd_init(args)
    elif args.command == "clean":
        return cmd_clean(args)
    elif not args.command:
        print_banner()
        if console:
            from rich.table import Table
            from rich.panel import Panel
            
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Command", style="cyan", width=15)
            table.add_column("Description", style="dim")
            
            table.add_row("[bold]Core[/bold]", "")
            table.add_row("init", "Scaffold a new Aurane project")
            table.add_row("compile", "Compile .aur to Python")
            table.add_row("run", "Compile and run immediately")
            table.add_row("watch", "Watch file and recompile")
            table.add_row("interactive", "Start interactive REPL")
            
            table.add_row("", "")
            table.add_row("[bold]Analysis[/bold]", "")
            table.add_row("inspect", "Inspect model structure")
            table.add_row("visualize", "Visualize model architecture")
            table.add_row("profile", "Profile model performance")
            table.add_row("benchmark", "Benchmark compilation time")
            
            table.add_row("", "")
            table.add_row("[bold]Tools[/bold]", "")
            table.add_row("format", "Format Aurane source files")
            table.add_row("lint", "Lint and auto-fix Aurane files")
            table.add_row("clean", "Remove build artifacts")
            
            help_panel = Panel(
                table,
                title="[bold yellow]Available Commands[/bold yellow]",
                border_style="cyan",
                expand=False
            )
            console.print(help_panel)
            console.print("\nRun [cyan]aurane <command> --help[/cyan] for more information on a command.")
        else:
            parser.print_help()
        return 0
    else:
        if console:
            console.print(f"[red]Error: Command '{args.command}' is not implemented.[/red]")
        return 1

if __name__ == "__main__":
    sys.exit(main())
