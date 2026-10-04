"""
Optimizer for Aurane AST.

Performs optimization passes on the AST to improve generated code quality
and model efficiency.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
from copy import deepcopy

from .ast import (
    AuraneProgram,
    ForwardBlock,
    ForwardGraphBlock,
)


@dataclass
class OptimizationResult:
    """Result of optimization passes."""

    program: AuraneProgram
    applied_optimizations: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class ASTOptimizer:
    """
    Optimizer for Aurane AST.

    Only rewrites that preserve training behavior and parameter state are enabled.
    Levels 1 and 2 currently share the same conservative passes.
    """

    def __init__(self, program: AuraneProgram):
        self.program = deepcopy(program)
        self.applied: List[str] = []
        self.stats: Dict[str, Any] = {
            "original_layers": 0,
            "optimized_layers": 0,
            "fusions": 0,
            "eliminations": 0,
        }

    def optimize(self, level: int = 1) -> OptimizationResult:
        """
        Run optimization passes.

        Args:
            level: Optimization level (0=none, 1 or 2=verified safe rewrites)

        Returns:
            OptimizationResult with optimized program.
        """

        if level not in (0, 1, 2):
            raise ValueError("Optimization level must be 0, 1, or 2")

        # Count original layers
        def count_layers(block) -> int:
            if not block:
                return 0
            if isinstance(block, ForwardBlock):
                return len(block.operations)
            if isinstance(block, ForwardGraphBlock):
                return len(block.nodes)
            return 0

        self.stats["original_layers"] = sum(
            count_layers(m.forward_block) for m in self.program.models
        )

        if level >= 1:
            self._remove_redundant_activations()

        # Count optimized layers
        self.stats["optimized_layers"] = sum(
            count_layers(m.forward_block) for m in self.program.models
        )

        return OptimizationResult(
            program=self.program, applied_optimizations=self.applied, stats=self.stats
        )

    def _remove_redundant_activations(self):
        """Remove only a plain ReLU immediately following an existing ReLU."""
        for model in self.program.models:
            block = model.forward_block
            if not isinstance(block, ForwardBlock):
                continue
            new_ops = []
            for op in block.operations:
                previous = new_ops[-1] if new_ops else None
                previous_is_relu = previous is not None and (
                    (
                        previous.activation == "relu"
                        and previous.operation in ("conv2d", "dense", "linear")
                    )
                    or (previous.operation == "relu" and not previous.activation)
                )
                if (
                    op.operation == "relu"
                    and not op.args
                    and not op.kwargs
                    and not op.activation
                    and previous_is_relu
                ):
                    self.stats["eliminations"] += 1
                    self.applied.append(f"Removed duplicate relu in {model.name}")
                else:
                    new_ops.append(op)
            block.operations = new_ops


def optimize_ast(program: AuraneProgram, level: int = 1) -> OptimizationResult:
    """
    Optimize an Aurane program AST.

    Args:
        program: The parsed Aurane program.
        level: Optimization level (0=none, 1 or 2=verified safe rewrites)

    Returns:
        OptimizationResult with optimized program and stats.
    """
    optimizer = ASTOptimizer(program)
    return optimizer.optimize(level)
