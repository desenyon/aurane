"""
Aurane - ML-Oriented DSL that Transpiles to Python

A domain-specific language for writing ML code that compiles to idiomatic Python.
"""

__version__ = "0.1.0"

from .compiler import compile_file, compile_source, CompilationError
from .parser import parse_aurane, ParseError
from .type_checker import check_types, TypeCheckResult, TypeChecker
from .optimizer import optimize_ast, OptimizationResult
from .semantic_analyzer import analyze_semantics, SemanticAnalysisResult
from .profiler import profile_model, profile_program, ModelProfile
from .visualizer import (
    print_model_summary,
    visualize_model_architecture,
    calculate_output_shape,
    calculate_parameters,
)

__all__ = [
    # Core compilation
    "compile_file",
    "compile_source",
    "parse_aurane",
    "CompilationError",
    "ParseError",
    # Type checking
    "check_types",
    "TypeCheckResult",
    "TypeChecker",
    # Optimization
    "optimize_ast",
    "OptimizationResult",
    # Semantic analysis
    "analyze_semantics",
    "SemanticAnalysisResult",
    # Profiling
    "profile_model",
    "profile_program",
    "ModelProfile",
    # Visualization
    "print_model_summary",
    "visualize_model_architecture",
    "calculate_output_shape",
    "calculate_parameters",
]
