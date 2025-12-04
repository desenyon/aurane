"""
Aurane - ML-Oriented DSL that Transpiles to Python

A domain-specific language for writing ML code that compiles to idiomatic Python.
"""

__version__ = "0.1.0"

from .compiler import compile_file, compile_source, CompilationError
from .parser import parse_aurane, ParseError

__all__ = [
    'compile_file',
    'compile_source',
    'parse_aurane',
    'CompilationError',
    'ParseError',
]
