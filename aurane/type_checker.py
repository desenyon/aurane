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
)


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
class TypeError:
    """Represents a type error in the program."""

    message: str
    location: str
    severity: str = "error"  # "error", "warning", "info"
    suggestion: Optional[str] = None


@dataclass
class TypeCheckResult:
    """Result of type checking."""

    errors: List[TypeError] = field(default_factory=list)
    warnings: List[TypeError] = field(default_factory=list)
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
                    TypeError(
                        message=f"Undefined model '{train.model_name}'",
                        location=f"train {train.model_name} on {train.dataset_name}",
                        suggestion=f"Define model '{train.model_name}' or use one of: {', '.join(model_names)}",
                    )
                )

            if train.dataset_name not in dataset_names:
                self.result.errors.append(
                    TypeError(
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
                TypeError(
                    message=f"Model '{model.name}' has no forward block",
                    location=f"model {model.name}",
                    severity="warning",
                )
            )
            return

        if not model.forward_block.operations:
            self.result.warnings.append(
                TypeError(
                    message=f"Model '{model.name}' has empty forward block",
                    location=f"model {model.name}",
                    severity="warning",
                )
            )
            return

        # Shape inference
        input_shape = model.config.get("input_shape", (1, 28, 28))
        if isinstance(input_shape, (list, tuple)):
            input_shape = tuple(input_shape)
        else:
            self.result.errors.append(
                TypeError(
                    message=f"Invalid input_shape for model '{model.name}'",
                    location=f"model {model.name}",
                    suggestion="input_shape should be a tuple like (1, 28, 28)",
                )
            )
            return

        current_shape = input_shape
        shapes = {"input": TensorType(shape=input_shape)}

        for idx, op in enumerate(model.forward_block.operations):
            try:
                current_shape = self._infer_shape(op, current_shape)
                shapes[f"layer_{idx}"] = TensorType(shape=current_shape)
            except Exception as e:
                self.result.errors.append(
                    TypeError(
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
        op_name = op.operation.lower()

        def to_int(val, default: int) -> int:
            if isinstance(val, int):
                return val
            if isinstance(val, float):
                return int(val)
            if isinstance(val, str):
                try:
                    return int(val)
                except ValueError:
                    return default
            return default

        if op_name == "conv2d":
            if len(input_shape) != 3:
                raise ValueError(f"conv2d expects 3D input (C, H, W), got {input_shape}")
            c, h, w = input_shape
            out_channels = to_int(op.args[0], 32) if op.args else 32
            kernel = to_int(op.kwargs.get("kernel", 3), 3)
            stride = to_int(op.kwargs.get("stride", 1), 1)
            padding = to_int(op.kwargs.get("padding", 0), 0)

            h_out = (h + 2 * padding - kernel) // stride + 1
            w_out = (w + 2 * padding - kernel) // stride + 1

            if h_out <= 0 or w_out <= 0:
                raise ValueError(f"Invalid output shape: kernel too large for input")

            return (out_channels, h_out, w_out)

        elif op_name in ("maxpool", "avgpool"):
            if len(input_shape) != 3:
                raise ValueError(f"{op_name} expects 3D input (C, H, W)")
            c, h, w = input_shape
            kernel = to_int(op.args[0], 2) if op.args else 2
            stride = to_int(op.kwargs.get("stride", kernel), kernel)
            return (c, h // stride, w // stride)

        elif op_name == "flatten":
            if len(input_shape) == 3:
                c, h, w = input_shape
                return (c * h * w,)
            return input_shape

        elif op_name in ("dense", "linear"):
            out_features = to_int(op.args[0], 128) if op.args else 128
            return (out_features,)

        elif op_name in ("dropout", "batchnorm", "layer_norm"):
            return input_shape

        elif op_name == "embedding":
            if len(op.args) >= 2:
                embed_dim = to_int(op.args[1], 256)
                seq_len = to_int(input_shape[0], 1) if input_shape else 1
                return (seq_len, embed_dim)
            return input_shape

        elif op_name == "multihead_attention":
            return input_shape

        elif op_name == "positional_encoding":
            return input_shape

        elif op_name == "global_avg_pool":
            if len(input_shape) == 3:
                c, h, w = input_shape
                return (c,)
            return input_shape

        elif op_name == "residual":
            return input_shape

        return input_shape

    def _check_datasets(self):
        """Check dataset definitions."""
        for dataset in self.program.datasets:
            if not dataset.source:
                self.result.warnings.append(
                    TypeError(
                        message=f"Dataset '{dataset.name}' has no source",
                        location=f"dataset {dataset.name}",
                        severity="warning",
                    )
                )

            batch = dataset.config.get("batch")
            if batch is not None:
                if not isinstance(batch, int) or batch <= 0:
                    self.result.errors.append(
                        TypeError(
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
                        TypeError(
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
                        TypeError(
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
