"""
Optimizer for Aurane AST.

Performs optimization passes on the AST to improve generated code quality
and model efficiency.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

from .ast import (
    AuraneProgram,
    ModelNode,
    LayerOperation,
    ForwardBlock,
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

    Performs various optimization passes:
    - Layer fusion (conv + bn + relu)
    - Dead code elimination
    - Constant folding
    - Operation reordering
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
            level: Optimization level (0=none, 1=basic, 2=aggressive)

        Returns:
            OptimizationResult with optimized program.
        """
        # Count original layers
        self.stats["original_layers"] = sum(
            len(m.forward_block.operations) if m.forward_block else 0 for m in self.program.models
        )

        if level >= 1:
            self._fuse_conv_bn_relu()
            self._remove_redundant_activations()
            self._optimize_dropout_placement()

        if level >= 2:
            self._fuse_consecutive_dense()
            self._optimize_pooling()
            self._reorder_operations()

        # Count optimized layers
        self.stats["optimized_layers"] = sum(
            len(m.forward_block.operations) if m.forward_block else 0 for m in self.program.models
        )

        return OptimizationResult(
            program=self.program, applied_optimizations=self.applied, stats=self.stats
        )

    def _fuse_conv_bn_relu(self):
        """Fuse Conv2d + BatchNorm + ReLU into single operation."""
        for model in self.program.models:
            if not model.forward_block:
                continue

            ops = model.forward_block.operations
            new_ops = []
            i = 0

            while i < len(ops):
                op = ops[i]

                # Look for conv2d followed by batchnorm
                if (
                    op.operation.lower() == "conv2d"
                    and i + 1 < len(ops)
                    and ops[i + 1].operation.lower() == "batchnorm"
                ):

                    # Mark for fusion
                    fused_op = LayerOperation(
                        operation="conv2d_bn",
                        args=op.args.copy(),
                        kwargs={**op.kwargs, "fused_bn": True},
                        activation=op.activation or ops[i + 1].activation,
                    )

                    # Check for relu after batchnorm
                    if i + 2 < len(ops) and ops[i + 2].operation.lower() in ("relu", "gelu"):
                        fused_op.activation = ops[i + 2].operation.lower()
                        i += 3
                        self.stats["fusions"] += 2
                    else:
                        i += 2
                        self.stats["fusions"] += 1

                    new_ops.append(fused_op)
                    self.applied.append(f"Fused conv2d+bn in {model.name}")
                else:
                    new_ops.append(op)
                    i += 1

            model.forward_block.operations = new_ops

    def _remove_redundant_activations(self):
        """Remove redundant activation functions."""
        for model in self.program.models:
            if not model.forward_block:
                continue

            ops = model.forward_block.operations
            new_ops = []

            for i, op in enumerate(ops):
                # Skip standalone activation if previous op already has it
                if (
                    op.operation.lower() in ("relu", "gelu", "sigmoid", "tanh")
                    and i > 0
                    and new_ops[-1].activation
                ):
                    self.stats["eliminations"] += 1
                    self.applied.append(f"Removed redundant {op.operation} in {model.name}")
                    continue

                new_ops.append(op)

            model.forward_block.operations = new_ops

    def _optimize_dropout_placement(self):
        """Optimize dropout placement - remove before final layer."""
        for model in self.program.models:
            if not model.forward_block:
                continue

            ops = model.forward_block.operations
            if len(ops) >= 2:
                # Check if last two are dropout + dense
                if (
                    ops[-1].operation.lower() in ("dense", "linear")
                    and ops[-2].operation.lower() == "dropout"
                ):

                    # Move dropout before the second-to-last layer
                    dropout_rate = ops[-2].args[0] if ops[-2].args else 0.5
                    ops[-2].kwargs["dropout_before"] = dropout_rate
                    ops.pop(-2)
                    self.applied.append(f"Optimized dropout placement in {model.name}")

    def _fuse_consecutive_dense(self):
        """Fuse consecutive dense layers without activation."""
        for model in self.program.models:
            if not model.forward_block:
                continue

            ops = model.forward_block.operations
            new_ops = []
            i = 0

            while i < len(ops):
                op = ops[i]

                # Look for dense without activation followed by another dense
                if (
                    op.operation.lower() in ("dense", "linear")
                    and not op.activation
                    and i + 1 < len(ops)
                    and ops[i + 1].operation.lower() in ("dense", "linear")
                ):

                    # Mark as fusable (codegen can optimize)
                    op.kwargs["fusable_next"] = True
                    self.applied.append(f"Marked fusable dense layers in {model.name}")

                new_ops.append(op)
                i += 1

            model.forward_block.operations = new_ops

    def _optimize_pooling(self):
        """Optimize pooling operations."""
        for model in self.program.models:
            if not model.forward_block:
                continue

            ops = model.forward_block.operations

            for i, op in enumerate(ops):
                # Convert maxpool(2) + maxpool(2) to maxpool(4)
                if (
                    op.operation.lower() == "maxpool"
                    and i + 1 < len(ops)
                    and ops[i + 1].operation.lower() == "maxpool"
                ):

                    k1 = op.args[0] if op.args else 2
                    k2 = ops[i + 1].args[0] if ops[i + 1].args else 2

                    if isinstance(k1, int) and isinstance(k2, int):
                        op.args = [k1 * k2]
                        op.kwargs["merged"] = True
                        ops[i + 1].kwargs["skip"] = True
                        self.applied.append(f"Merged pooling in {model.name}")

    def _reorder_operations(self):
        """Reorder operations for better efficiency."""
        # This is a placeholder for more complex reordering logic
        pass


def optimize_ast(program: AuraneProgram, level: int = 1) -> OptimizationResult:
    """
    Optimize an Aurane program AST.

    Args:
        program: The parsed Aurane program.
        level: Optimization level (0=none, 1=basic, 2=aggressive)

    Returns:
        OptimizationResult with optimized program and stats.
    """
    optimizer = ASTOptimizer(program)
    return optimizer.optimize(level)
