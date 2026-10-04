"""Forward graphs shared by code generation, checking, profiling and rendering."""

from dataclasses import dataclass, field
import builtins
from typing import Any, Dict, List, Optional, Tuple

from .ast import ForwardBlock, ForwardGraphBlock, LayerOperation
from .shapes import infer_graph_output_shape
from .diagnostics import SourceSpan, LocatedError
from .dtypes import model_dtypes, infer_dtype


@dataclass(frozen=True)
class IRValue:
    """A distinct tensor value; shape excludes its runtime batch dimension."""

    name: str
    type_hint: Optional[str] = None
    shape: Optional[Tuple[int, ...]] = None


@dataclass
class IRNode:
    op_name: str
    inputs: List[IRValue] = field(default_factory=list)
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    activation: Optional[str] = None
    output: Optional[IRValue] = None
    line: int = 0
    column: int = 0
    end_line: int = 0
    end_column: int = 0

    def to_operation(self) -> LayerOperation:
        return LayerOperation(
            operation=self.op_name,
            args=list(self.args),
            kwargs=dict(self.kwargs),
            activation=self.activation,
            line=self.line,
            column=self.column,
            end_line=self.end_line,
            end_column=self.end_column,
        )


@dataclass
class IRGraph:
    nodes: List[IRNode] = field(default_factory=list)
    inputs: List[IRValue] = field(default_factory=list)
    outputs: List[IRValue] = field(default_factory=list)


def lower_forward_block(
    forward_block,
    input_shape: Optional[Tuple[int, ...]] = None,
    *,
    input_dtype: Optional[str] = None,
    parameter_dtype: str = "float32",
) -> IRGraph:
    """Resolve wiring and optional shapes while preserving each value's identity."""
    if not isinstance(forward_block, (ForwardBlock, ForwardGraphBlock)):
        raise TypeError(f"Unsupported forward block type: {type(forward_block)}")
    input_value = IRValue(forward_block.parameter, type_hint=input_dtype, shape=input_shape)
    graph = IRGraph(inputs=[input_value])
    env = {forward_block.parameter: input_value}
    used_names = {input_value.name, "self", "torch", "nn", "F", "optim", "padding_mask"} | set(
        dir(builtins)
    )

    def append(operation, input_names, target):
        undefined = [name for name in input_names if name not in env]
        if undefined:
            raise LocatedError(
                f"Undefined tensor(s): {', '.join(undefined)}", SourceSpan.from_node(operation)
            )
        inputs = [env[name] for name in input_names]
        shape = None
        if input_shape is not None:
            try:
                shape = infer_graph_output_shape(operation, [value.shape for value in inputs])
            except ValueError as error:
                raise LocatedError(str(error), SourceSpan.from_node(operation)) from error
        try:
            dtype = infer_dtype(operation, [value.type_hint for value in inputs], parameter_dtype)
        except ValueError as error:
            raise LocatedError(str(error), SourceSpan.from_node(operation)) from error
        name = target
        version = 1
        while name in used_names:
            name = f"{target}__{version}"
            version += 1
        used_names.add(name)
        output = IRValue(name, type_hint=dtype, shape=shape)
        graph.nodes.append(
            IRNode(
                operation.operation,
                inputs,
                list(operation.args),
                dict(operation.kwargs),
                operation.activation,
                output,
                operation.line,
                operation.column,
                operation.end_line,
                operation.end_column,
            )
        )
        env[target] = output
        return output

    if isinstance(forward_block, ForwardGraphBlock):
        for node in forward_block.nodes:
            if node.operation is None:
                raise ValueError(f"Missing graph operation at line {node.line}")
            append(node.operation, node.inputs, node.target)
        output_name = forward_block.output_var or (
            forward_block.nodes[-1].target if forward_block.nodes else forward_block.parameter
        )
        if output_name not in env:
            line = getattr(output_name, "line", forward_block.line)
            column = getattr(output_name, "column", forward_block.column)
            raise LocatedError(
                f"Undefined return tensor '{output_name}'",
                SourceSpan(line, column, line, column + len(output_name)),
            )
        graph.outputs = [env[output_name]]
    else:
        current_name = forward_block.parameter
        output = input_value
        for index, operation in enumerate(forward_block.operations):
            target = f"t{index}"
            # Sequential temporaries must never overwrite the parameter binding.
            while target in env:
                target += "_"
            output = append(operation, [current_name], target)
            current_name = target
        graph.outputs = [output]
    return graph


def lower_model(model) -> IRGraph:
    """Use one model input/dtype contract across all compiler consumers."""
    input_dtype, parameter_dtype = model_dtypes(model)
    return lower_forward_block(
        model.forward_block,
        tuple(model.config.get("input_shape", (1, 28, 28))),
        input_dtype=input_dtype,
        parameter_dtype=parameter_dtype,
    )


def lower_sequential_forward(forward_block, input_value_name: Optional[str] = None) -> IRGraph:
    """Compatibility entry point for callers lowering a sequential block."""
    if input_value_name is not None:
        forward_block = ForwardBlock(
            parameter=input_value_name,
            operations=forward_block.operations,
            line=forward_block.line,
            column=forward_block.column,
        )
    return lower_forward_block(forward_block)
