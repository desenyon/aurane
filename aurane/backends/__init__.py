"""
Backend registry for lowering Aurane programs to target runtimes.

The current v2 foundation keeps this intentionally lightweight:
backends register a generator callable that turns an `AuraneProgram`
into Python (or export code) for a specific target.
"""

from .registry import get_backend_generator, register_backend_generator

__all__ = ["get_backend_generator", "register_backend_generator"]
