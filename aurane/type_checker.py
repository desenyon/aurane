"""
Type checker for Aurane DSL.

Performs static type analysis and shape inference to catch errors
before code generation.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from .ast import (
    AuraneProgram,
    ModelNode,
    DatasetNode,
    TrainNode,
    LayerOperation,
    ForwardBlock,
    ForwardGraphBlock,
)
from .shapes import infer_output_shape, to_int


class TypeKind(Enum):
    """Kinds of types in Aurane."""

    TENSOR = "tensor"
    SCALAR = "scalar"
    STRING = "string"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    TUPLE = "tuple"
    LIST = "list"
    UNKNOWN = "unknown"


@dataclass
class TensorType:
    """Represents a tensor type with shape information."""

    shape: Optional[Tuple[int, ...]] = None
    dtype: str = "float32"
    device: str = "cpu"

    def __str__(self) -> str:
        shape_str = str(self.shape) if self.shape else "?"
        return f"Tensor[{shape_str}, {self.dtype}]"

    def is_compatible(self, other: "TensorType") -> bool:
        """Check if two tensor types are compatible."""
        if self.shape is None or other.shape is None:
            return True
        if len(self.shape) != len(other.shape):
            return False
        for s1, s2 in zip(self.shape, other.shape):
            if s1 != -1 and s2 != -1 and s1 != s2:
                return False
        return True


@dataclass
class TypeAnalysisError:
    """Represents a type error in the program."""

    message: str
    location: str
    severity: str = "error"  # "error", "warning", "info"
    suggestion: Optional[str] = None


@dataclass
class TypeCheckResult:
    """Result of type checking."""

    errors: List[TypeAnalysisError] = field(default_factory=list)
    warnings: List[TypeAnalysisError] = field(default_factory=list)
    inferred_types: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def is_valid(self) -> bool:
        return not self.has_errors


class TypeChecker:
    """
    Static type checker for Aurane programs.

    Performs:
    - Shape inference through the network
    - Type compatibility checking
    - Undefined reference detection
    - Configuration validation
    """

    def __init__(self, program: AuraneProgram):
        self.program = program
        self.result = TypeCheckResult()
        self.symbol_table: Dict[str, Any] = {}
        self.model_shapes: Dict[str, Dict[str, TensorType]] = {}

    def check(self) -> TypeCheckResult:
        """Run all type checking passes."""
        self._collect_symbols()
        self._check_references()
        self._check_models()
        self._check_datasets()
        self._check_training()
        return self.result

    def _collect_symbols(self):
        """Collect all symbol definitions."""
        for model in self.program.models:
            self.symbol_table[model.name] = ("model", model)

        for dataset in self.program.datasets:
            self.symbol_table[dataset.name] = ("dataset", dataset)

        for exp in self.program.experiments:
            self.symbol_table[exp.name] = ("experiment", exp)

    def _check_references(self):
        """Check for undefined references."""
        model_names = {m.name for m in self.program.models}
        dataset_names = {d.name for d in self.program.datasets}

        for train in self.program.trains:
            if train.model_name not in model_names:
                self.result.errors.append(
                    TypeAnalysisError(
                        message=f"Undefined model '{train.model_name}'",
                        location=f"train {train.model_name} on {train.dataset_name}",
                        suggestion=f"Define model '{train.model_name}' or use one of: {', '.join(model_names)}",
                    )
                )

            if train.dataset_name not in dataset_names:
                self.result.errors.append(
                    TypeAnalysisError(
                        message=f"Undefined dataset '{train.dataset_name}'",
                        location=f"train {train.model_name} on {train.dataset_name}",
                        suggestion=f"Define dataset '{train.dataset_name}' or use one of: {', '.join(dataset_names)}",
                    )
                )

    def _check_models(self):
        """Check model definitions."""
        for model in self.program.models:
            self._check_model(model)

    def _check_model(self, model: ModelNode):
        """Check a single model definition."""
        if not model.forward_block:
            self.result.warnings.append(
                TypeAnalysisError(
                    message=f"Model '{model.name}' has no forward block",
                    location=f"model {model.name}",
                    severity="warning",
                )
            )
            return

        if isinstance(model.forward_block, ForwardGraphBlock):
            if not model.forward_block.nodes:
                self.result.warnings.append(
                    TypeAnalysisError(
                        message=f"Model '{model.name}' has empty forward block",
                        location=f"model {model.name}",
                        severity="warning",
                    )
                )
                return
        else:
            if not model.forward_block.operations:
                self.result.warnings.append(
                    TypeAnalysisError(
                        message=f"Model '{model.name}' has empty forward block",
                        location=f"model {model.name}",
                        severity="warning",
                    )
                )
                return

        # Shape inference
        input_shape = model.config.get("input_shape", (1, 28, 28))
        if isinstance(input_shape, (list, tuple)):
            input_shape_tuple: Optional[Tuple[int, ...]] = tuple(input_shape)
        else:
            self.result.errors.append(
                TypeAnalysisError(
                    message=f"Invalid input_shape for model '{model.name}'",
                    location=f"model {model.name}",
                    suggestion="input_shape should be a tuple like (1, 28, 28)",
                )
            )
            return

        shapes: Dict[str, TensorType] = {"input": TensorType(shape=input_shape_tuple)}

        if isinstance(model.forward_block, ForwardGraphBlock):
            shape_env: Dict[str, TensorType] = {
                model.forward_block.parameter: TensorType(shape=input_shape_tuple)
            }
            defined_vars = {model.forward_block.parameter}

            for idx, node in enumerate(model.forward_block.nodes):
                op = node.operation
                if op is None:
                    continue

                op_name = op.operation.lower()
                node_inputs = node.inputs

                undefined = [v for v in node_inputs if v not in defined_vars]
                if undefined:
                    self.result.errors.append(
                        TypeAnalysisError(
                            message=f"Undefined tensor variable(s) in {op_name}: {', '.join(undefined)}",
                            location=f"model {model.name}, node {idx}",
                        )
                    )
                    continue

                if input_shape_tuple is None:
                    inferred_shape: Optional[Tuple[int, ...]] = None
                elif op_name == "add":
                    # add(a,b) => same shape as first input
                    s0 = shape_env[node_inputs[0]].shape
                    inferred_shape = s0
                    # Optional compatibility check if both known.
                    if len(node_inputs) >= 2:
                        s1 = shape_env[node_inputs[1]].shape
                        if s0 is not None and s1 is not None and len(s0) == len(s1):
                            for d0, d1 in zip(s0, s1):
                                if d0 != -1 and d1 != -1 and d0 != d1:
                                    self.result.errors.append(
                                        TypeAnalysisError(
                                            message=f"add shape mismatch: {s0} vs {s1}",
                                            location=f"model {model.name}, node {idx}",
                                        )
                                    )
                                    break
                elif op_name == "concat":
                    dim = to_int(op.kwargs.get("dim", 1), 1)
                    raw_shapes = [shape_env[v].shape for v in node_inputs]
                    if any(s is None for s in raw_shapes):
                        inferred_shape = None
                    else:
                        shapes_in = [s for s in raw_shapes if s is not None]
                        first = shapes_in[0]
                        if not all(len(s) == len(first) for s in shapes_in):
                            inferred_shape = None
                        else:
                            out_dims = list(first)
                            dim_sum = 0
                            for s in shapes_in:
                                d = s[dim] if len(s) > dim else -1
                                if d == -1 or dim_sum == -1:
                                    dim_sum = -1
                                    break
                                dim_sum += d
                            out_dims[dim] = dim_sum
                            inferred_shape = tuple(out_dims)
                else:
                    in_shape = shape_env[node_inputs[0]].shape
                    if in_shape is None:
                        inferred_shape = None
                    else:
                        inferred_shape = self._infer_shape(op, in_shape)

                shape_env[node.target] = TensorType(shape=inferred_shape)
                defined_vars.add(node.target)
                shapes[f"layer_{idx}"] = TensorType(shape=inferred_shape)

            output_var = model.forward_block.output_var or (
                model.forward_block.nodes[-1].target
                if model.forward_block.nodes
                else model.forward_block.parameter
            )
            out_shape = shape_env.get(output_var, TensorType(shape=None)).shape
            shapes["output"] = TensorType(shape=out_shape)
        else:
            current_shape: Optional[Tuple[int, ...]] = input_shape_tuple
            for idx, op in enumerate(model.forward_block.operations):
                try:
                    if current_shape is None:
                        current_shape = None
                    else:
                        current_shape = self._infer_shape(op, current_shape)
                    shapes[f"layer_{idx}"] = TensorType(shape=current_shape)
                except Exception as e:
                    self.result.errors.append(
                        TypeAnalysisError(
                            message=f"Shape inference failed at layer {idx}: {e}",
                            location=f"model {model.name}, operation {op.operation}",
                        )
                    )
                    break

            shapes["output"] = TensorType(shape=current_shape)

        self.model_shapes[model.name] = shapes
        self.result.inferred_types[model.name] = shapes

    def _infer_shape(self, op: LayerOperation, input_shape: tuple) -> tuple:
        """Infer output shape for an operation."""
        return infer_output_shape(op, input_shape)

    def _check_datasets(self):
        """Check dataset definitions."""
        for dataset in self.program.datasets:
            if not dataset.source:
                self.result.warnings.append(
                    TypeAnalysisError(
                        message=f"Dataset '{dataset.name}' has no source",
                        location=f"dataset {dataset.name}",
                        severity="warning",
                    )
                )

            batch = dataset.config.get("batch")
            if batch is not None:
                if not isinstance(batch, int) or batch <= 0:
                    self.result.errors.append(
                        TypeAnalysisError(
                            message=f"Invalid batch size for dataset '{dataset.name}'",
                            location=f"dataset {dataset.name}",
                            suggestion="batch should be a positive integer",
                        )
                    )

    def _check_training(self):
        """Check training configurations."""
        for train in self.program.trains:
            # Check epochs
            epochs = train.config.get("epochs")
            if epochs is not None:
                if not isinstance(epochs, int) or epochs <= 0:
                    self.result.errors.append(
                        TypeAnalysisError(
                            message=f"Invalid epochs value",
                            location=f"train {train.model_name}",
                            suggestion="epochs should be a positive integer",
                        )
                    )

            # Check learning rate
            lr = train.config.get("lr")
            if lr is not None:
                if not isinstance(lr, (int, float)) or lr <= 0:
                    self.result.warnings.append(
                        TypeAnalysisError(
                            message=f"Invalid learning rate",
                            location=f"train {train.model_name}",
                            severity="warning",
                            suggestion="learning rate should be a positive number",
                        )
                    )


def check_types(program: AuraneProgram) -> TypeCheckResult:
    """
    Perform type checking on an Aurane program.

    Args:
        program: The parsed Aurane program.

    Returns:
        TypeCheckResult with errors, warnings, and inferred types.
    """
    checker = TypeChecker(program)
    return checker.check()


def format_type_errors(result: TypeCheckResult) -> str:
    """Format type check results as a string."""
    lines = []

    if result.errors:
        lines.append(f"[FAIL] {len(result.errors)} type error(s):")
        for err in result.errors:
            lines.append(f"  - {err.location}: {err.message}")
            if err.suggestion:
                lines.append(f"    Suggestion: {err.suggestion}")

    if result.warnings:
        lines.append(f"[WARN] {len(result.warnings)} warning(s):")
        for warn in result.warnings:
            lines.append(f"  - {warn.location}: {warn.message}")

    if result.is_valid and not result.warnings:
        lines.append("[OK] No type errors found")

    return "\n".join(lines)
