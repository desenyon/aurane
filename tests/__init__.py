"""
Test suite for Aurane DSL.

This package contains comprehensive tests for all Aurane modules:

- test_parser.py: Tests for the Aurane parser
- test_compiler.py: Tests for the compilation pipeline
- test_codegen.py: Tests for PyTorch code generation
- test_visualizer.py: Tests for model visualization
- test_type_checker.py: Tests for static type analysis
- test_optimizer.py: Tests for AST optimization
- test_semantic_analyzer.py: Tests for semantic analysis
- test_profiler.py: Tests for model profiling
- test_cli.py: Tests for CLI commands
- test_ast.py: Tests for AST node classes

Run tests with:
    pytest tests/
    pytest tests/ -v  # verbose
    pytest tests/ -k "parser"  # only parser tests
    pytest tests/ --cov=aurane  # with coverage
"""

__all__ = [
    "test_parser",
    "test_compiler",
    "test_codegen",
    "test_visualizer",
    "test_type_checker",
    "test_optimizer",
    "test_semantic_analyzer",
    "test_profiler",
    "test_cli",
    "test_ast",
]
