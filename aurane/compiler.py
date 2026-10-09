"""
Aurane compiler module.

This module provides the main compilation interface for converting
.aur files to Python code.
"""

from pathlib import Path
import hashlib
import json
import os
import tempfile

from .file_io import atomic_write as _atomic_write
from .parser import parse_aurane
from .preparation import prepare_program
from .diagnostics import CompilationDiagnostic
from .semantic_analyzer import analyze_semantics, format_semantic_issues
from .type_checker import check_types, format_type_errors
from .backends import get_backend_generator
from .backends.registry import get_backend_cache_version, get_prepared_backend_generator

# Bump whenever parser, analysis, optimization or generated-code behavior changes.
CACHE_SCHEMA = 14


def write_compiled_output(input_file: Path, output_file: Path, code: str) -> None:
    """Keep source and prior output intact if publication fails."""
    if input_file.resolve() == output_file.resolve() or (
        output_file.exists() and input_file.samefile(output_file)
    ):
        raise CompilationError("Output path must not overwrite the source file", stage="write")
    try:
        _atomic_write(output_file, code)
    except OSError as error:
        raise CompilationError(f"Failed to write output file: {error}", stage="write") from error


class CompilationError(Exception):
    """Readable compilation failure with stage and machine-readable diagnostics."""

    def __init__(self, message, *, stage="compile", span=None, diagnostics=None, source=None):
        super().__init__(message)
        self.stage = stage
        self.diagnostics = tuple(diagnostics or [CompilationDiagnostic(stage, message, span)])
        self.span = span or next((item.span for item in self.diagnostics if item.span), None)
        self.source = source

    def to_dict(self) -> dict:
        return {
            "kind": type(self).__name__,
            "message": str(self),
            "stage": self.stage,
            "source": self.source,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def compile_file(
    input_path: str,
    output_path: str,
    backend: str = "torch",
    analyze: bool = False,
    validate: bool = False,
    optimize: bool = False,
    opt_level: int = 1,
    disable_cache: bool = False,
    cache_dir: str | Path | None = None,
) -> None:
    """
    Compile an Aurane source file to Python.

    Args:
        input_path: Path to the .aur source file.
        output_path: Path where the generated Python file will be written.
        backend: Code generation backend to use (default: "torch").

    Raises:
        CompilationError: If compilation fails.
        FileNotFoundError: If the input file does not exist.
    """
    # Read source file
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Source file not found: {input_path}")

    try:
        source = input_file.read_text(encoding="utf-8")
    except Exception as e:
        raise CompilationError(
            f"Failed to read source file: {e}", stage="read", source=input_path
        ) from e

    # Compile
    try:
        python_code = compile_source(
            source,
            backend=backend,
            analyze=analyze,
            validate=validate,
            optimize=optimize,
            opt_level=opt_level,
            disable_cache=disable_cache,
            cache_dir=cache_dir,
        )
    except CompilationError as error:
        error.source = str(input_file)
        raise

    # Write output
    output_file = Path(output_path)
    try:
        write_compiled_output(input_file, output_file, python_code)
    except CompilationError as error:
        error.source = str(input_file)
        raise

    print(f"Successfully compiled {input_path} -> {output_path}")


def compile_source(
    source: str,
    backend: str = "torch",
    disable_cache: bool = False,
    analyze: bool = False,
    validate: bool = False,
    optimize: bool = False,
    opt_level: int = 1,
    cache_dir: str | Path | None = None,
) -> str:
    """
    Compile Aurane source code to Python.

    Args:
        source: The Aurane source code as a string.
        backend: Code generation backend to use (default: "torch").
        disable_cache: If True, do not read from or write to the cache.
        cache_dir: Cache directory; falls back to AURANE_CACHE_DIR then .aurane_cache.

    Returns:
        Generated Python source code as a string.

    Raises:
        CompilationError: If compilation fails.
    """
    from . import __version__

    backend = backend.lower().strip()
    try:
        generator = get_backend_generator(backend)
    except KeyError as error:
        raise CompilationError(str(error), stage="backend") from error
    prepared_generator = get_prepared_backend_generator(backend)
    backend_version = get_backend_cache_version(backend)
    # Dynamic plugins without a declared version cannot provide a stable cache key.
    disable_cache = disable_cache or backend_version is None

    # Attempt to resolve from cache
    if not disable_cache:
        cache_key = {
            "cache_schema": CACHE_SCHEMA,
            "compiler_version": __version__,
            "backend_version": backend_version,
            "source": source,
            "backend": backend,
            "analyze": analyze,
            "validate": validate,
            "optimize": optimize,
            "opt_level": opt_level,
        }
        source_hash = hashlib.sha256(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()
        directory = cache_dir if cache_dir is not None else os.environ.get("AURANE_CACHE_DIR")
        cache_file = Path(directory or ".aurane_cache") / f"{source_hash}.json"
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            code = cached["code"]
            if (
                isinstance(code, str)
                and cached["key"] == source_hash
                and cached["sha256"] == hashlib.sha256(code.encode()).hexdigest()
            ):
                return code
        except (OSError, UnicodeError, ValueError, KeyError, TypeError):
            pass  # Missing, corrupt and unreadable caches are ordinary misses.

    # Parse source to AST
    try:
        ast = parse_aurane(source)
    except Exception as e:
        raise CompilationError(
            f"Parse error: {e}", stage="parse", span=getattr(e, "span", None)
        ) from e
    try:
        prepared = prepare_program(ast)
    except Exception as e:
        raise CompilationError(
            f"Resolution error: {e}", stage="resolve", span=getattr(e, "span", None)
        ) from e

    # Optional passes before codegen.
    if analyze:
        try:
            semantic_result = analyze_semantics(prepared)
            if semantic_result.has_errors:
                raise CompilationError(
                    format_semantic_issues(semantic_result),
                    stage="semantic",
                    diagnostics=[
                        CompilationDiagnostic(
                            "semantic", issue.message, issue.span, issue.code, issue.location
                        )
                        for issue in semantic_result.errors
                    ],
                )
        except CompilationError:
            raise
        except Exception as e:
            raise CompilationError(
                f"Semantic analysis failed: {e}", stage="semantic", span=getattr(e, "span", None)
            ) from e

    if validate:
        try:
            type_result = check_types(prepared)
            if type_result.has_errors:
                raise CompilationError(
                    format_type_errors(type_result),
                    stage="type",
                    diagnostics=[
                        CompilationDiagnostic(
                            "type", issue.message, issue.span, location=issue.location
                        )
                        for issue in type_result.errors
                    ],
                )
        except CompilationError:
            raise
        except Exception as e:
            raise CompilationError(
                f"Type checking failed: {e}", stage="type", span=getattr(e, "span", None)
            ) from e

    if optimize:
        try:
            prepared = prepared.optimized(opt_level)
        except Exception as e:
            raise CompilationError(
                f"Optimization failed: {e}", stage="optimize", span=getattr(e, "span", None)
            ) from e

    # Generate code based on backend
    try:
        python_code = (
            prepared_generator(prepared) if prepared_generator else generator(prepared.program)
        )
        if not isinstance(python_code, str):
            raise TypeError("Backend generator must return source text as str")
    except Exception as e:
        raise CompilationError(
            f"Code generation error: {e}", stage="codegen", span=getattr(e, "span", None)
        ) from e

    # Write to cache
    if not disable_cache:
        try:
            _atomic_write(
                cache_file,
                json.dumps(
                    {
                        "key": source_hash,
                        "code": python_code,
                        "sha256": hashlib.sha256(python_code.encode()).hexdigest(),
                    }
                ),
            )
        except OSError:
            pass  # Non-fatal if cache write fails

    return python_code


def compile_to_temp(
    source: str,
    backend: str = "torch",
    disable_cache: bool = False,
    *,
    analyze: bool = False,
    validate: bool = False,
    optimize: bool = False,
    opt_level: int = 1,
    cache_dir: str | Path | None = None,
) -> Path:
    """
    Compile Aurane source to a temporary Python file.

    Args:
        source: The Aurane source code as a string.
        backend: Code generation backend to use (default: "torch").
        disable_cache: If True, bypass cache.

    Returns:
        Path to the temporary Python file.

    Raises:
        CompilationError: If compilation fails.
    """
    python_code = compile_source(
        source,
        backend=backend,
        disable_cache=disable_cache,
        analyze=analyze,
        validate=validate,
        optimize=optimize,
        opt_level=opt_level,
        cache_dir=cache_dir,
    )

    # Create temporary file
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            temp_path = Path(f.name)
            f.write(python_code)
    except OSError as error:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise CompilationError(f"Failed to write temporary file: {error}", stage="write") from error

    return temp_path
