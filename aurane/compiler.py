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

from .parser import parse_aurane
from .symbols import resolve_program
from .semantic_analyzer import analyze_semantics, format_semantic_issues
from .type_checker import check_types, format_type_errors
from .optimizer import optimize_ast
from .backends import get_backend_generator
from .backends.registry import get_backend_cache_version

# Bump whenever parser, analysis, optimization or generated-code behavior changes.
CACHE_SCHEMA = 13


def _atomic_write(path: Path, text: str) -> None:
    """Publish a complete adjacent file, cleaning up failed temporary writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        if path.exists():
            temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_compiled_output(input_file: Path, output_file: Path, code: str) -> None:
    """Keep source and prior output intact if publication fails."""
    if input_file.resolve() == output_file.resolve() or (
        output_file.exists() and input_file.samefile(output_file)
    ):
        raise CompilationError("Output path must not overwrite the source file")
    try:
        _atomic_write(output_file, code)
    except OSError as error:
        raise CompilationError(f"Failed to write output file: {error}") from error


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
    write_compiled_output(input_file, output_file, python_code)

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
    from . import __version__

    backend = backend.lower().strip()
    try:
        generator = get_backend_generator(backend)
    except KeyError as error:
        raise CompilationError(str(error)) from error
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
        cache_file = Path(".aurane_cache") / f"{source_hash}.json"
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
        ast = resolve_program(parse_aurane(source))
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
        python_code = generator(ast)
    except KeyError as e:
        raise CompilationError(str(e))
    except Exception as e:
        raise CompilationError(f"Code generation error: {e}")

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
