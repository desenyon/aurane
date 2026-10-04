"""Resolve declarative constants without evaluating Python code.

Global constants are visible to each model. Model constants shadow globals and
never leak into another model. Resolution returns a copy of the syntax tree.
"""

from copy import deepcopy
from typing import Any, Dict

from .diagnostics import SourceSpan
from .ast import AuraneProgram, ConfigCall, ForwardGraphBlock, ModelNode, SymbolReference


class ResolutionError(ValueError):
    """An undefined, duplicate, or cyclic constant definition."""

    def __init__(self, message, span=None):
        self.span = span
        super().__init__(message)


def _resolve_value(value, lookup, *, labels=False):
    if isinstance(value, SymbolReference):
        return lookup(value, labels)
    if isinstance(value, ConfigCall):
        args = [_resolve_value(arg, lookup) for arg in value.args]
        kwargs = {key: _resolve_value(arg, lookup) for key, arg in value.kwargs.items()}
        parts = [repr(arg) for arg in args]
        parts.extend(f"{key}={arg!r}" for key, arg in kwargs.items())
        return ConfigCall(f"{value.split('(', 1)[0]}({', '.join(parts)})", args, kwargs)
    if isinstance(value, tuple):
        return tuple(_resolve_value(item, lookup, labels=labels) for item in value)
    if isinstance(value, list):
        return [_resolve_value(item, lookup, labels=labels) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_value(item, lookup, labels=labels) for key, item in value.items()}
    return value


def _namespace(definitions: Dict[str, Any], outer=None):
    resolved: Dict[str, Any] = {}
    visiting: list[str] = []

    def lookup(reference, labels=False):
        name = str(reference)
        if name in resolved:
            return resolved[name]
        if name not in definitions:
            if outer is not None:
                return outer(reference, labels)
            if labels:
                return name
            raise ResolutionError(
                f"Undefined symbol '{name}' at line {reference.line}, column {reference.column}",
                SourceSpan(
                    reference.line,
                    reference.column,
                    reference.line,
                    reference.column + len(reference),
                ),
            )
        if name in visiting:
            cycle = " -> ".join([*visiting, name])
            raise ResolutionError(
                f"Constant cycle: {cycle} at line {reference.line}",
                SourceSpan(
                    reference.line,
                    reference.column,
                    reference.line,
                    reference.column + len(reference),
                ),
            )
        visiting.append(name)
        resolved[name] = _resolve_value(definitions[name], lookup)
        visiting.pop()
        return resolved[name]

    for name in definitions:
        lookup(SymbolReference(name))
    return resolved, lookup


def _resolve_model(model: ModelNode, outer=None) -> ModelNode:
    model.config, lookup = _namespace(model.config, outer)
    block = model.forward_block
    if block is not None:
        operations = (
            [node.operation for node in block.nodes if node.operation is not None]
            if isinstance(block, ForwardGraphBlock)
            else block.operations
        )
        for operation in operations:
            operation.args = _resolve_value(operation.args, lookup)
            operation.kwargs = _resolve_value(operation.kwargs, lookup)
    return model


def resolve_model(model: ModelNode) -> ModelNode:
    """Resolve local constants; use resolve_program for global constants."""
    return _resolve_model(deepcopy(model))


def resolve_program(program: AuraneProgram) -> AuraneProgram:
    """Resolve constants consistently for backends and analysis tools."""
    result = deepcopy(program)
    definitions = {}
    for variable in result.variables:
        if variable.name in definitions:
            raise ResolutionError(
                f"Duplicate constant '{variable.name}' at line {variable.line}",
                SourceSpan.from_node(variable),
            )
        definitions[variable.name] = variable.value
    globals_, lookup = _namespace(definitions)
    for variable in result.variables:
        variable.value = globals_[variable.name]
    for model in result.models:
        _resolve_model(model, lookup)
    # Bare configuration labels (cpu, accuracy, dataset names, etc.) remain
    # labels; arguments inside calls must be real constants or quoted strings.
    config_nodes: list[Any] = [
        *result.experiments,
        *result.datasets,
        *result.trains,
        *result.train_gans,
    ]
    for node in config_nodes:
        node.config = _resolve_value(node.config, lookup, labels=True)
    for train in result.trains:
        for callback in train.callbacks:
            callback.name = _resolve_value(callback.name, lookup, labels=True)
        if train.scheduler:
            train.scheduler.params = _resolve_value(train.scheduler.params, lookup)
    return result


def parse_resolved(source: str) -> AuraneProgram:
    """Parse and resolve source for commands that consume concrete values."""
    from .parser import parse_aurane

    return resolve_program(parse_aurane(source))
