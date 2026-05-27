"""
Check command for Aurane CLI.

Runs semantic analysis and type/shape checking and reports diagnostics.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Dict

from ..ui import console, RICH_AVAILABLE
from ..utils import validate_file
from ...parser import parse_aurane, ParseError
from ...semantic_analyzer import analyze_semantics, format_semantic_issues
from ...type_checker import check_types, format_type_errors


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of dataclasses/enums into JSON."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {field.name: _to_jsonable(getattr(obj, field.name)) for field in fields(obj)}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return str(obj)


def cmd_check(args) -> int:
    """Run checks on an Aurane file."""
    if not RICH_AVAILABLE or console is None:
        # We still support text output even without rich.
        use_rich = False
    else:
        use_rich = True

    try:
        file_path = validate_file(args.input, [".aur"])
        source = file_path.read_text(encoding="utf-8")

        try:
            program = parse_aurane(source)
        except ParseError as e:
            if use_rich:
                console.print(f"[red][FAIL] Parse error:[/red] {e}")
            else:
                print(f"Parse error: {e}")
            return 1

        run_semantic = args.semantic or (not args.semantic and not args.types)
        run_types = args.types or (not args.semantic and not args.types)

        semantic_result = None
        type_result = None

        if run_semantic:
            semantic_result = analyze_semantics(program)
            if args.json:
                pass
            else:
                if use_rich:
                    console.print("\n[bold cyan]Semantic analysis[/bold cyan]")
                    console.print(format_semantic_issues(semantic_result))
                else:
                    print("\nSemantic analysis")
                    print(format_semantic_issues(semantic_result))

        if run_types:
            type_result = check_types(program)
            if args.json:
                pass
            else:
                if use_rich:
                    console.print("\n[bold cyan]Type checking[/bold cyan]")
                    console.print(format_type_errors(type_result))
                else:
                    print("\nType checking")
                    print(format_type_errors(type_result))

        if args.json:
            payload: Dict[str, Any] = {
                "file": str(file_path),
                "semantic": None,
                "types": None,
            }
            if semantic_result is not None:
                payload["semantic"] = _to_jsonable(semantic_result)
            if type_result is not None:
                payload["types"] = _to_jsonable(type_result)
            print(json.dumps(payload, indent=2))

        has_errors = False
        if semantic_result is not None and semantic_result.has_errors:
            has_errors = True
        if type_result is not None and type_result.has_errors:
            has_errors = True

        if has_errors:
            return 1

        if use_rich:
            console.print("\n[green][OK] Checks passed[/green]")
        else:
            print("Checks passed")
        return 0

    except Exception as e:
        if use_rich:
            console.print(f"[red][FAIL] Error:[/red] {e}")
        else:
            print(f"Error: {e}")
        return 1
