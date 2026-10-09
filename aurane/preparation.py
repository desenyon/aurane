"""Invocation-local resolved AST and lazy graph ownership.

Consumers treat the prepared AST and graphs as read-only. Optimizing creates a
new owner; caches never outlive a compilation or depend on mutable global state.
"""

from .ast import AuraneProgram, ModelNode
from .ir import IRGraph, lower_model
from .symbols import resolve_program
from .optimizer import optimize_ast


class PreparedProgram:
    """Internal compiler context for an already resolved, privately owned AST."""

    def __init__(self, program: AuraneProgram):
        self.program = program
        self._models = {id(model): model for model in program.models}
        self._graphs: dict[int, IRGraph] = {}

    def graph_for(self, model: ModelNode) -> IRGraph:
        """Lower once per model identity, without conflating duplicate names."""
        key = id(model)
        if key not in self._models:
            raise ValueError("Model does not belong to this prepared program")
        if key not in self._graphs:
            self._graphs[key] = lower_model(model)
        return self._graphs[key]

    def optimized(self, level: int) -> "PreparedProgram":
        """Optimization owns a copy and invalidates all prior graph results."""
        return PreparedProgram(optimize_ast(self.program, level=level).program)


def prepare_program(program: AuraneProgram | PreparedProgram) -> PreparedProgram:
    """Reuse internal preparation; isolate and resolve public AST input once."""
    if isinstance(program, PreparedProgram):
        return program
    return PreparedProgram(resolve_program(program))
