"""Shared compiler options and error output for compile, run and watch."""

import json
import sys

from ..compiler import CompilationError


def add_compilation_options(parser):
    parser.add_argument("--backend", default="torch", choices=["torch"], help="Transpiler backend")
    parser.add_argument("--analyze", action="store_true", help="Run semantic analysis")
    parser.add_argument("--validate", action="store_true", help="Run type/shape checks")
    parser.add_argument("--optimize", action="store_true", help="Optimize the model AST")
    parser.add_argument(
        "--opt-level",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Optimization level used with --optimize",
    )
    parser.add_argument(
        "--no-cache",
        dest="disable_cache",
        action="store_true",
        help="Disable cache reads and writes",
    )
    parser.add_argument(
        "--cache-dir", help="Cache location (default: AURANE_CACHE_DIR or .aurane_cache)"
    )
    parser.add_argument(
        "--diagnostics-format",
        choices=["text", "json"],
        default="text",
        help="Compilation errors on stderr",
    )


def compiler_options(args) -> dict:
    """Support both parsed namespaces and older programmatic command callers."""
    defaults = dict(
        backend="torch",
        analyze=False,
        validate=False,
        optimize=False,
        opt_level=1,
        disable_cache=False,
        cache_dir=None,
    )
    return {key: getattr(args, key, default) for key, default in defaults.items()}


def json_diagnostics(args) -> bool:
    return getattr(args, "diagnostics_format", "text") == "json"


def report_compilation_error(error, args):
    if not isinstance(error, CompilationError):
        error = CompilationError(
            str(error),
            stage="read" if isinstance(error, (OSError, UnicodeError)) else "compile",
            span=getattr(error, "span", None),
        )
    error.source = str(args.input)
    if json_diagnostics(args):
        print(json.dumps({"ok": False, "error": error.to_dict()}), file=sys.stderr)
    else:
        print(f"[FAIL] {error.stage}: {error}", file=sys.stderr)
