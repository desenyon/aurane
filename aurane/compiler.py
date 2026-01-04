"""
Aurane compiler module.

This module provides the main compilation interface for converting
.aur files to Python code.
"""

from pathlib import Path
from typing import Optional

from .parser import parse_aurane
from .codegen_torch import generate_torch_code


class CompilationError(Exception):
    """Exception raised when compilation fails."""

    pass


def compile_file(input_path: str, output_path: str, backend: str = "torch") -> None:
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
        python_code = compile_source(source, backend=backend)
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


def compile_source(source: str, backend: str = "torch") -> str:
    """
    Compile Aurane source code to Python.

    Args:
        source: The Aurane source code as a string.
        backend: Code generation backend to use (default: "torch").

    Returns:
        Generated Python source code as a string.

    Raises:
        CompilationError: If compilation fails.
    """
    # Parse source to AST
    try:
        ast = parse_aurane(source)
    except Exception as e:
        raise CompilationError(f"Parse error: {e}")

    # Generate code based on backend
    if backend == "torch":
        try:
            return generate_torch_code(ast)
        except Exception as e:
            raise CompilationError(f"Code generation error: {e}")
    else:
        raise CompilationError(f"Unsupported backend: {backend}")


def compile_to_temp(source: str, backend: str = "torch") -> Path:
    """
    Compile Aurane source to a temporary Python file.

    Args:
        source: The Aurane source code as a string.
        backend: Code generation backend to use (default: "torch").

    Returns:
        Path to the temporary Python file.

    Raises:
        CompilationError: If compilation fails.
    """
    import tempfile

    python_code = compile_source(source, backend=backend)

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(python_code)
        temp_path = Path(f.name)

    return temp_path
