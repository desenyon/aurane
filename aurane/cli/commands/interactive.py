"""
Interactive REPL for Aurane CLI.
"""

from ..ui import console, RICH_AVAILABLE, print_banner

try:
    from rich.prompt import Prompt
    from rich.syntax import Syntax
except ImportError:
    pass

from ...compiler import compile_source

def cmd_interactive(args):
    """Start interactive REPL mode."""
    if not RICH_AVAILABLE or console is None:
        print("Interactive mode requires 'rich' library. Install with: pip install rich")
        return 1

    print_banner()
    console.print("\n[cyan]Interactive Mode[/cyan] - Type [bold].help[/bold] for commands\n")

    code_buffer = []

    while True:
        try:
            prompt = "[bold cyan]aurane>[/bold cyan]" if not code_buffer else "[bold cyan].......[/bold cyan]"
            line = Prompt.ask(prompt)

            if not line.strip():
                continue

            if line.startswith("."):
                cmd = line[1:].strip().lower()
                if cmd in ("exit", "quit", "q"):
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                elif cmd in ("help", "h"):
                    console.print("[yellow]Available commands: .help, .exit, .clear, .show, .compile[/yellow]")
                elif cmd in ("clear", "cls"):
                    code_buffer = []
                    console.print("[yellow]Buffer cleared[/yellow]")
                elif cmd in ("show", "s"):
                    if code_buffer:
                        console.print(Syntax("\n".join(code_buffer), "python", theme="monokai"))
                    else:
                        console.print("[yellow]Buffer is empty[/yellow]")
                elif cmd in ("compile", "c"):
                    if code_buffer:
                        try:
                            py_code = compile_source("\n".join(code_buffer))
                            console.print(Syntax(py_code, "python", theme="monokai", line_numbers=True))
                        except Exception as e:
                            console.print(f"[red]Error:[/red] {e}")
                    else:
                        console.print("[yellow]Buffer is empty[/yellow]")
                continue

            code_buffer.append(line)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type .exit to quit.[/yellow]")
            code_buffer = []
        except EOFError:
            break

    return 0
