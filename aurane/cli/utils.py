"""
Utility functions for Aurane CLI.
"""

from pathlib import Path
from typing import Dict, Any

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
