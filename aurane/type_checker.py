"""
Type checker for Aurane DSL.

Performs static type analysis and shape inference to catch errors
before code generation.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .ast import (
    AuraneProgram,
    ModelNode,
    DatasetNode,
    TrainNode,
    LayerOperation,
    ForwardGraphBlock,
)
from .shapes import infer_output_shape
from .symbols import resolve_program, ResolutionError
from .ir import lower_model
from .diagnostics import SourceSpan
from .configuration import (
    validate_program_configuration,
    training_references,
    validate_training_options,
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
        if self.dtype != "unknown" and other.dtype != "unknown" and self.dtype != other.dtype:
            return False
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
    span: Optional[SourceSpan] = None


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
        try:
            self.program = resolve_program(self.program)
        except ResolutionError as error:
            self.result.errors.append(
                TypeAnalysisError(str(error), "constants", span=getattr(error, "span", None))
            )
            return self.result
        try:
            validate_program_configuration(self.program)
        except ValueError as error:
            self.result.errors.append(
                TypeAnalysisError(str(error), "program", span=getattr(error, "span", None))
            )
            return self.result
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

        definitions = {"model": model_names, "dataset": dataset_names}
        for train in [*self.program.trains, *self.program.train_gans]:
            try:
                validate_training_options(train)
            except ValueError as error:
                self.result.errors.append(
                    TypeAnalysisError(
                        str(error), f"line {train.line}", span=getattr(error, "span", None)
                    )
                )
                continue
            for kind, name in training_references(train):
                if name not in definitions[kind]:
                    self.result.errors.append(
                        TypeAnalysisError(
                            f"Undefined {kind} '{name}'",
                            f"train at line {train.line}",
                            span=next(
                                (
                                    train.config_spans[key]
                                    for key in ("validate_on", "test_on")
                                    if train.config.get(key) == name and key in train.config_spans
                                ),
                                SourceSpan.from_node(train),
                            ),
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

        try:
            graph = lower_model(model)
            shapes["input"].dtype = graph.inputs[0].type_hint or "unknown"
            for index, node in enumerate(graph.nodes):
                assert node.output is not None
                shapes[f"layer_{index}"] = TensorType(
                    shape=node.output.shape, dtype=node.output.type_hint or "unknown"
                )
            shapes["output"] = TensorType(
                shape=graph.outputs[0].shape, dtype=graph.outputs[0].type_hint or "unknown"
            )
        except (ValueError, TypeError, IndexError, ZeroDivisionError) as error:
            self.result.errors.append(
                TypeAnalysisError(
                    str(error), f"model {model.name}", span=getattr(error, "span", None)
                )
            )
            return

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
