"""
IR dump command for Aurane CLI.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ..ui import console, RICH_AVAILABLE
from ..utils import validate_file
from ...parser import parse_aurane
from ...ir import lower_forward_block


def _print_or_console(use_rich: bool, text: str) -> None:
    if use_rich and console is not None:
        console.print(text)
    else:
        print(text)


def cmd_ir(args) -> int:
    """Dump lowered IR for models in an Aurane file."""
    use_rich = RICH_AVAILABLE and console is not None

    try:
        input_file = validate_file(args.input, [".aur"])
        source = input_file.read_text(encoding="utf-8")
        program = parse_aurane(source)

        models = program.models
        if args.model:
            models = [m for m in models if m.name == args.model]

        if not models:
            _print_or_console(use_rich, "No models found.")
            return 1

        payload: Dict[str, Any] = {"file": str(input_file), "models": []}
        for model in models:
            if not model.forward_block:
                payload["models"].append({"name": model.name, "ir": None})
                continue

            ir_graph = lower_forward_block(model.forward_block)
            if args.format == "json":
                payload["models"].append({"name": model.name, "ir": asdict(ir_graph)})
            else:
                # text
                nodes = [
                    {
                        "op": n.op_name,
                        "inputs": [v.name for v in n.inputs],
                        "activation": n.activation,
                        "output": n.output.name if n.output else None,
                    }
                    for n in ir_graph.nodes
                ]
                payload["models"].append({"name": model.name, "ir": nodes})

        if args.format == "json":
            _print_or_console(use_rich, json.dumps(payload, indent=2))
        else:
            for m in payload["models"]:
                _print_or_console(use_rich, f"\nIR for model: {m['name']}")
                _print_or_console(use_rich, json.dumps(m["ir"], indent=2))

        return 0

    except Exception as e:
        if use_rich and console is not None:
            console.print(f"[red][FAIL] Error:[/red] {e}")
        else:
            print(f"Error: {e}")
        return 1
