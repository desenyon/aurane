"""
Lint command for Aurane CLI.
"""

import re

from ..ui import console, RICH_AVAILABLE
from ..utils import validate_file
from ...parser import parse_aurane

BLOCK_PATTERN = re.compile(r"^(def|model|dataset|train|train_gan|experiment)\b.*[^:]$")


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
        fixed = False
        lines = source.split("\n")
        new_lines = []

        # Check style issues and auto-fix
        for i, line in enumerate(lines, 1):
            original_line = line

            # Whitespace
            if line != line.rstrip():
                issues.append(("info", f"Line {i}: Trailing whitespace"))
                if getattr(args, "auto_fix", False):
                    line = line.rstrip()
                    fixed = True

            # Missing colons on blocks
            stripped = line.strip()
            if "=" not in stripped and BLOCK_PATTERN.match(stripped):
                issues.append(("warning", f"Line {i}: Missing colon at end of block definition"))
                if getattr(args, "auto_fix", False):
                    line = line + ":"
                    fixed = True

            if len(line) > 100:
                issues.append(("warning", f"Line {i}: Line too long ({len(line)} > 100)"))

            new_lines.append(line)

        new_source = "\n".join(new_lines)

        # Parse and validate the (possibly fixed) source
        try:
            ast = parse_aurane(new_source)
        except Exception as e:
            issues.append(("error", f"Parse error: {e}"))

        if getattr(args, "auto_fix", False) and fixed:
            file_path.write_text(new_source)
            console.print("[green][OK] Applied auto-fixes.[/green]\n")

        # Display results
        if not issues:
            console.print("[green][OK] No issues found[/green]")
            return 0

        for sev, msg in issues:
            color = {"error": "red", "warning": "yellow", "info": "cyan"}.get(sev, "white")
            if (
                getattr(args, "auto_fix", False)
                and sev in ("info", "warning")
                and "Line too long" not in msg
            ):
                console.print(f"[{color}][{sev.upper()}][/{color}] {msg} [green](Fixed)[/green]")
            else:
                console.print(f"[{color}][{sev.upper()}][/{color}] {msg}")

        return 1 if any(sev == "error" for sev, _ in issues) else 0

    except Exception as e:
        console.print(f"[red][FAIL] Error:[/red] {e}")
        return 1
