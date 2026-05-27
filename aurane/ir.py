"""
Intermediate representation (IR) for Aurane models.

The IR is designed to represent forward computation as a (for now) directed
graph of operator nodes with explicit value wiring. Future DSL versions can
lower into this IR and backends can lower IR into their target runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ast import ForwardBlock, ForwardGraphBlock


@dataclass(frozen=True)
class IRValue:
    """A named value in the IR graph."""

    name: str
    # Optional type annotation for future use (tensor shape/dtype, etc.).
    type_hint: Optional[str] = None


@dataclass
class IRNode:
    """A node in the IR graph."""

    op_name: str
    inputs: List[IRValue] = field(default_factory=list)
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    activation: Optional[str] = None
    output: Optional[IRValue] = None


@dataclass
class IRGraph:
    """IR representation of a forward computation graph."""

    nodes: List[IRNode] = field(default_factory=list)
    inputs: List[IRValue] = field(default_factory=list)
    outputs: List[IRValue] = field(default_factory=list)


def lower_sequential_forward(forward_block, input_value_name: Optional[str] = None) -> IRGraph:
    """
    Lower a simple sequential forward chain into an IR graph.

    This is a compatibility bridge: the current DSL is sequential, so every
    op consumes the previous op output.
    """

    param_name = input_value_name or forward_block.parameter
    current = IRValue(name=param_name)

    graph = IRGraph(inputs=[current])

    for idx, op in enumerate(forward_block.operations):
        out = IRValue(name=f"t{idx}")
        node = IRNode(
            op_name=op.operation,
            inputs=[current],
            args=list(op.args),
            kwargs=dict(op.kwargs),
            activation=op.activation,
            output=out,
        )
        graph.nodes.append(node)
        current = out

    graph.outputs = [current]
    return graph


def lower_forward_block(forward_block) -> IRGraph:
    """Lower either sequential or graph-based forward blocks into IR."""
    if isinstance(forward_block, ForwardGraphBlock):
        input_value = IRValue(name=forward_block.parameter)
        env: Dict[str, IRValue] = {forward_block.parameter: input_value}
        graph = IRGraph(inputs=[input_value])

        for node in forward_block.nodes:
            if node.operation is None:
                continue
            out = IRValue(name=node.target)
            inputs = [env[v] for v in node.inputs]
            ir_node = IRNode(
                op_name=node.operation.operation,
                inputs=inputs,
                args=list(node.operation.args),
                kwargs=dict(node.operation.kwargs),
                activation=node.operation.activation,
                output=out,
            )
            graph.nodes.append(ir_node)
            env[node.target] = out

        output_var = forward_block.output_var or (
            forward_block.nodes[-1].target if forward_block.nodes else forward_block.parameter
        )
        graph.outputs = [env[output_var]]
        return graph

    if isinstance(forward_block, ForwardBlock):
        return lower_sequential_forward(forward_block)

    raise TypeError(f"Unsupported forward block type for IR lowering: {type(forward_block)}")
