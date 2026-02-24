"""
Init command for Aurane CLI.
"""

import os
from pathlib import Path
from ..ui import console, RICH_AVAILABLE


def cmd_init(args):
    """Scaffold a new Aurane project."""
    if not RICH_AVAILABLE or console is None:
        print("Init command requires 'rich' library. Install with: pip install rich")
        return 1

    try:
        project_name = args.name
        project_dir = Path(project_name)

        if project_dir.exists() and any(project_dir.iterdir()):
            console.print(
                f"[red]Error: Directory '{project_name}' already exists and is not empty.[/red]"
            )
            return 1

        console.print(f"[cyan]Initializing new Aurane project:[/cyan] {project_name}\n")

        # Create directories
        src_dir = project_dir / "src"
        data_dir = project_dir / "data"
        src_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create sample files
        main_aur_path = src_dir / "main.aur"
        main_aur_content = f"""# {project_name} - Main Aurane File
use torch

experiment Default:
    seed = 42
    device = "auto"

dataset my_data:
    batch = 32

model MyModel:
    input_shape = (1, 28, 28)
    def forward(x):
        x -> flatten()
          -> dense(128).relu
          -> dense(10)

train MyModel on my_data:
    epochs = 5
    lr = 0.001
    optimizer = "adam"
"""
        main_aur_path.write_text(main_aur_content)

        readme_path = project_dir / "README.md"
        readme_content = f"""# {project_name}

A machine learning project built with Aurane DSL.

## Usage

Compile the model to PyTorch:
```bash
aurane compile src/main.aur outputs/main.py
```

Run the compiled model:
```bash
python outputs/main.py
```
"""
        readme_path.write_text(readme_content)

        # Print success
        console.print(
            f"[green][OK][/green] Project created at [bold]{project_dir.absolute()}[/bold]"
        )
        console.print(f"  Created [dim]src/main.aur[/dim]")
        console.print(f"  Created [dim]README.md[/dim]")

        console.print(f"\n[cyan]Next steps:[/cyan]")
        console.print(f"  1. cd {project_name}")
        console.print(f"  2. aurane inspect src/main.aur")
        console.print(f"  3. aurane compile src/main.aur out.py")

        return 0

    except Exception as e:
        console.print(f"[red][FAIL] Error:[/red] {e}")
        return 1
