"""
Shared UI utilities for Aurane CLI using Rich.
"""

from typing import Optional, Dict, Any
from pathlib import Path

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

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None  # type: ignore

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
   [dim]Version 1.0.0 • PyTorch Backend • MIT License[/dim]
    """
    console.print(Panel(banner, border_style="cyan", expand=False))


def get_progress():
    """Create a standard progress bar."""
    if not RICH_AVAILABLE or console is None:
        return None
        
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    )
