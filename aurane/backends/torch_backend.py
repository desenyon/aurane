"""
Torch backend generator.

This is currently a thin wrapper around the existing Torch code generator.
"""

from __future__ import annotations

from ..ast import AuraneProgram
from ..codegen_torch import generate_torch_code


def generate_torch_code_backend(program: AuraneProgram) -> str:
    """Generate Python code for a Torch target."""
    return generate_torch_code(program)
