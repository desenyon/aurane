"""
Aurane compiler module.

This module provides the main compilation interface for converting
.aur files to Python code.
"""

from pathlib import Path
from typing import Optional

from .parser import parse_aurane
from .semantic_analyzer import analyze_semantics, format_semantic_issues
from .type_checker import check_types, format_type_errors
from .optimizer import optimize_ast
from .backends import get_backend_generator


class CompilationError(Exception):
    """Exception raised when compilation fails."""

    pass


def compile_file(
    input_path: str,
    output_path: str,
    backend: str = "torch",
    analyze: bool = False,
    validate: bool = False,
    optimize: bool = False,
    opt_level: int = 1,
    disable_cache: bool = False,
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
        raise CompilationError(f"Failed to read source file: {e}")

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
        )
    except Exception as e:
        raise CompilationError(f"Compilation failed: {e}")

    # Write output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        output_file.write_text(python_code, encoding="utf-8")
    except Exception as e:
        raise CompilationError(f"Failed to write output file: {e}")

    print(f"Successfully compiled {input_path} -> {output_path}")


def compile_source(
    source: str,
    backend: str = "torch",
    disable_cache: bool = False,
    analyze: bool = False,
    validate: bool = False,
    optimize: bool = False,
    opt_level: int = 1,
) -> str:
    """
    Compile Aurane source code to Python.

    Args:
        source: The Aurane source code as a string.
        backend: Code generation backend to use (default: "torch").
        disable_cache: If True, do not read from or write to the cache.

    Returns:
        Generated Python source code as a string.

    Raises:
        CompilationError: If compilation fails.
    """
    import hashlib
    import os

    # Attempt to resolve from cache
    if not disable_cache:
        cache_key = {
            "cache_schema": 2,
            "source": source,
            "backend": backend,
            "analyze": analyze,
            "validate": validate,
            "optimize": optimize,
            "opt_level": opt_level,
        }
        source_hash = hashlib.md5(str(cache_key).encode("utf-8")).hexdigest()
        cache_dir = Path(".aurane_cache") / backend
        cache_file = cache_dir / f"{source_hash}.py"

        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

    # Parse source to AST
    try:
        ast = parse_aurane(source)
    except Exception as e:
        raise CompilationError(f"Parse error: {e}")

    # Optional passes before codegen.
    if analyze:
        try:
            semantic_result = analyze_semantics(ast)
            if semantic_result.has_errors:
                raise CompilationError(format_semantic_issues(semantic_result))
        except CompilationError:
            raise
        except Exception as e:
            raise CompilationError(f"Semantic analysis failed: {e}")

    if validate:
        try:
            type_result = check_types(ast)
            if type_result.has_errors:
                raise CompilationError(format_type_errors(type_result))
        except CompilationError:
            raise
        except Exception as e:
            raise CompilationError(f"Type checking failed: {e}")

    if optimize:
        try:
            optimized = optimize_ast(ast, level=opt_level)
            ast = optimized.program
        except Exception as e:
            raise CompilationError(f"Optimization failed: {e}")

    # Generate code based on backend
    try:
        generator = get_backend_generator(backend)
        python_code = generator(ast)
    except KeyError as e:
        raise CompilationError(str(e))
    except Exception as e:
        raise CompilationError(f"Code generation error: {e}")

    # Write to cache
    if not disable_cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(python_code, encoding="utf-8")
        except Exception:
            pass  # Non-fatal if cache write fails

    return python_code


def compile_to_temp(source: str, backend: str = "torch", disable_cache: bool = False) -> Path:
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
    import tempfile

    python_code = compile_source(source, backend=backend, disable_cache=disable_cache)

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(python_code)
        temp_path = Path(f.name)

    return temp_path
