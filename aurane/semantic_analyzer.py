"""
Semantic analyzer for Aurane DSL.

Performs semantic analysis beyond parsing, including:
- Scope analysis
- Dependency resolution
- Configuration validation
- Best practice suggestions
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from .diagnostics import SourceSpan
from .symbols import resolve_program, ResolutionError
from .configuration import (
    OPTIMIZER_CLASSES,
    LOSS_CLASSES,
    validate_program_configuration,
    validate_training_options,
    training_references,
)

from .ast import (
    AuraneProgram,
    ModelNode,
    DatasetNode,
    TrainNode,
    ExperimentNode,
    LayerOperation,
    ForwardBlock,
    ForwardGraphBlock,
)


class IssueKind(Enum):
    """Kinds of semantic issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUGGESTION = "suggestion"


@dataclass
class SemanticIssue:
    """Represents a semantic issue."""

    kind: IssueKind
    message: str
    location: str
    code: str  # Issue code like "E001", "W001"
    fix: Optional[str] = None
    span: Optional[SourceSpan] = None


@dataclass
class SemanticAnalysisResult:
    """Result of semantic analysis."""

    issues: List[SemanticIssue] = field(default_factory=list)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[SemanticIssue]:
        return [i for i in self.issues if i.kind == IssueKind.ERROR]

    @property
    def warnings(self) -> List[SemanticIssue]:
        return [i for i in self.issues if i.kind == IssueKind.WARNING]

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def is_valid(self) -> bool:
        return not self.has_errors


from .operations import validate_operation, ACTIVATION_NAMES, OPERATION_SPECS as LAYER_SPECS

ACTIVATIONS = ACTIVATION_NAMES | {"residual"}
OPTIMIZERS = set(OPTIMIZER_CLASSES)
LOSS_FUNCTIONS = set(LOSS_CLASSES)


class SemanticAnalyzer:
    """
    Semantic analyzer for Aurane programs.

    Performs comprehensive semantic analysis including:
    - Layer validation
    - Configuration checking
    - Dependency analysis
    - Best practice suggestions
    """

    def __init__(self, program: AuraneProgram):
        self.program = program
        self.result = SemanticAnalysisResult()
        self.defined_models: Set[str] = set()
        self.defined_datasets: Set[str] = set()
        self.defined_experiments: Set[str] = set()

    def analyze(self) -> SemanticAnalysisResult:
        """Run all semantic analysis passes."""
        try:
            self.program = resolve_program(self.program)
        except ResolutionError as error:
            self._add_issue(
                IssueKind.ERROR, str(error), "constants", "E009", span=getattr(error, "span", None)
            )
            return self.result
        try:
            validate_program_configuration(self.program)
        except ValueError as error:
            self._add_issue(
                IssueKind.ERROR, str(error), "program", "E010", span=getattr(error, "span", None)
            )
            return self.result
        self._collect_definitions()
        self._analyze_imports()
        self._analyze_experiments()
        self._analyze_datasets()
        self._analyze_models()
        self._analyze_training()
        self._compute_dependencies()
        self._suggest_improvements()
        return self.result

    def _collect_definitions(self):
        """Collect all definitions."""
        for model in self.program.models:
            if model.name in self.defined_models:
                self._add_issue(
                    IssueKind.ERROR,
                    f"Duplicate model definition: {model.name}",
                    f"model {model.name}",
                    "E001",
                )
            self.defined_models.add(model.name)

        for dataset in self.program.datasets:
            if dataset.name in self.defined_datasets:
                self._add_issue(
                    IssueKind.ERROR,
                    f"Duplicate dataset definition: {dataset.name}",
                    f"dataset {dataset.name}",
                    "E002",
                )
            self.defined_datasets.add(dataset.name)

        for exp in self.program.experiments:
            if exp.name in self.defined_experiments:
                self._add_issue(
                    IssueKind.WARNING,
                    f"Duplicate experiment definition: {exp.name}",
                    f"experiment {exp.name}",
                    "W001",
                )
            self.defined_experiments.add(exp.name)

    def _analyze_imports(self):
        """Analyze import statements."""
        imported_modules = set()

        for use in self.program.uses:
            if use.module in imported_modules:
                self._add_issue(
                    IssueKind.WARNING,
                    f"Duplicate import: {use.module}",
                    f"use {use.module}",
                    "W002",
                )
            imported_modules.add(use.module)

    def _analyze_experiments(self):
        """Analyze experiment configurations."""
        for exp in self.program.experiments:
            # Check for required fields
            if "seed" not in exp.config:
                self._add_issue(
                    IssueKind.SUGGESTION,
                    "Consider setting a seed for reproducibility",
                    f"experiment {exp.name}",
                    "S001",
                    fix="Add: seed = 42",
                )

            # Check device configuration
            device = exp.config.get("device")
            if device and device not in ("auto", "cpu", "cuda", "mps"):
                self._add_issue(
                    IssueKind.WARNING, f"Unknown device: {device}", f"experiment {exp.name}", "W003"
                )

    def _analyze_datasets(self):
        """Analyze dataset definitions."""
        for dataset in self.program.datasets:
            # Check for source
            if not dataset.source:
                self._add_issue(
                    IssueKind.WARNING,
                    "Dataset has no source specified",
                    f"dataset {dataset.name}",
                    "W004",
                )

            # Check batch size
            batch = dataset.config.get("batch")
            if batch is not None:
                if isinstance(batch, int):
                    if batch <= 0:
                        self._add_issue(
                            IssueKind.ERROR,
                            "Batch size must be positive",
                            f"dataset {dataset.name}",
                            "E003",
                        )
                    elif batch > 1024:
                        self._add_issue(
                            IssueKind.SUGGESTION,
                            f"Large batch size ({batch}) may cause memory issues",
                            f"dataset {dataset.name}",
                            "S002",
                        )

    def _analyze_models(self):
        """Analyze model definitions."""
        for model in self.program.models:
            self._analyze_model(model)

    def _analyze_model(self, model: ModelNode):
        """Analyze a single model."""
        # Check for forward block
        if not model.forward_block:
            self._add_issue(
                IssueKind.ERROR, "Model has no forward block", f"model {model.name}", "E004"
            )
            return

        # Check for input_shape
        if "input_shape" not in model.config:
            self._add_issue(
                IssueKind.WARNING,
                "Model has no input_shape specified",
                f"model {model.name}",
                "W005",
                fix="Add: input_shape = (channels, height, width)",
            )

        # Analyze operations
        if isinstance(model.forward_block, ForwardGraphBlock):
            ops = [n.operation for n in model.forward_block.nodes if n.operation is not None]
        else:
            ops = model.forward_block.operations

        if not ops:
            self._add_issue(
                IssueKind.WARNING, "Model has empty forward block", f"model {model.name}", "W006"
            )
            return

        for idx, op in enumerate(ops):
            self._analyze_operation(op, model.name, idx)

        # Check for common patterns
        self._check_model_patterns(model)

    def _analyze_operation(self, op: LayerOperation, model_name: str, idx: int):
        """Analyze a single operation."""
        op_name = op.operation.lower()
        location = f"model {model_name}, layer {idx}"

        try:
            validate_operation(op)
        except ValueError as error:
            self._add_issue(
                IssueKind.ERROR,
                str(error),
                f"{location}, line {op.line}",
                "E005",
                span=SourceSpan.from_node(op),
            )

    def _check_model_patterns(self, model: ModelNode):
        """Check for common model patterns and best practices."""
        if not model.forward_block:
            return

        if isinstance(model.forward_block, ForwardGraphBlock):
            ops = [n.operation for n in model.forward_block.nodes if n.operation is not None]
        else:
            ops = model.forward_block.operations
        op_names = [op.operation.lower() for op in ops]

        # Check for missing batchnorm after conv
        for i, (op, name) in enumerate(zip(ops, op_names)):
            if name == "conv2d":
                if i + 1 < len(ops) and op_names[i + 1] not in ("batchnorm", "batch_norm"):
                    self._add_issue(
                        IssueKind.SUGGESTION,
                        "Consider adding batchnorm after conv2d",
                        f"model {model.name}, layer {i}",
                        "S004",
                    )

        # Check for dropout placement
        if "dropout" in op_names:
            last_dropout_idx = len(op_names) - 1 - op_names[::-1].index("dropout")
            if last_dropout_idx == len(ops) - 1:
                self._add_issue(
                    IssueKind.WARNING,
                    "Dropout at the end of the network has no effect",
                    f"model {model.name}",
                    "W009",
                )

        # Check for activation on output layer
        if ops and ops[-1].operation.lower() in ("dense", "linear"):
            if ops[-1].activation in ("relu", "gelu"):
                self._add_issue(
                    IssueKind.SUGGESTION,
                    "ReLU/GeLU on output layer may limit output range",
                    f"model {model.name}, output",
                    "S005",
                )

    def _analyze_training(self):
        """Analyze training configurations."""
        definitions = {"model": self.defined_models, "dataset": self.defined_datasets}
        for train in [*self.program.trains, *self.program.train_gans]:
            location = f"train at line {train.line}"
            try:
                validate_training_options(train)
            except ValueError as error:
                self._add_issue(
                    IssueKind.ERROR, str(error), location, "E010", span=getattr(error, "span", None)
                )
                continue
            for kind, name in training_references(train):
                if name not in definitions[kind]:
                    self._add_issue(
                        IssueKind.ERROR,
                        f"Undefined {kind}: {name}",
                        location,
                        "E006" if kind == "model" else "E007",
                        span=next(
                            (
                                train.config_spans[key]
                                for key in ("validate_on", "test_on")
                                if train.config.get(key) == name and key in train.config_spans
                            ),
                            SourceSpan.from_node(train),
                        ),
                    )
        for train in self.program.trains:
            epochs = train.config.get("epochs")
            if type(epochs) is int and epochs > 1000:
                self._add_issue(
                    IssueKind.SUGGESTION,
                    f"Many epochs ({epochs}), consider early stopping",
                    f"train {train.model_name}",
                    "S006",
                    span=train.config_spans.get("epochs"),
                )

    def _compute_dependencies(self):
        """Compute dependency graph."""
        for train in self.program.trains:
            deps = set()
            deps.add(train.model_name)
            deps.add(train.dataset_name)

            key = f"train:{train.model_name}:{train.dataset_name}"
            self.result.dependencies[key] = deps

        # Models can depend on other models (for ensemble, etc.)
        for model in self.program.models:
            self.result.dependencies[f"model:{model.name}"] = set()

    def _suggest_improvements(self):
        """Suggest general improvements."""
        # Check for missing experiments
        if not self.program.experiments:
            self._add_issue(
                IssueKind.SUGGESTION,
                "Consider adding an experiment block for configuration",
                "program",
                "S007",
            )

        # Check for empty program
        if not self.program.models and not self.program.datasets:
            self._add_issue(IssueKind.WARNING, "Program appears to be empty", "program", "W012")

    def _add_issue(
        self,
        kind: IssueKind,
        message: str,
        location: str,
        code: str,
        fix: Optional[str] = None,
        span: Optional[SourceSpan] = None,
    ):
        """Add an issue to results."""
        self.result.issues.append(
            SemanticIssue(
                kind=kind, message=message, location=location, code=code, fix=fix, span=span
            )
        )


def analyze_semantics(program: AuraneProgram) -> SemanticAnalysisResult:
    """
    Perform semantic analysis on an Aurane program.

    Args:
        program: The parsed Aurane program.

    Returns:
        SemanticAnalysisResult with issues and metadata.
    """
    analyzer = SemanticAnalyzer(program)
    return analyzer.analyze()


def format_semantic_issues(result: SemanticAnalysisResult) -> str:
    """Format semantic analysis results as a string."""
    lines = []

    errors = [i for i in result.issues if i.kind == IssueKind.ERROR]
    warnings = [i for i in result.issues if i.kind == IssueKind.WARNING]
    suggestions = [i for i in result.issues if i.kind == IssueKind.SUGGESTION]
    infos = [i for i in result.issues if i.kind == IssueKind.INFO]

    if errors:
        lines.append(f"[FAIL] {len(errors)} error(s):")
        for issue in errors:
            lines.append(f"  {issue.code}: {issue.location}")
            lines.append(f"    {issue.message}")

    if warnings:
        lines.append(f"[WARN] {len(warnings)} warning(s):")
        for issue in warnings:
            lines.append(f"  {issue.code}: {issue.location}")
            lines.append(f"    {issue.message}")

    if suggestions:
        lines.append(f"[SUGGEST] {len(suggestions)} suggestion(s):")
        for issue in suggestions:
            lines.append(f"  {issue.code}: {issue.message}")
            if issue.fix:
                lines.append(f"    Fix: {issue.fix}")

    if result.is_valid and not warnings:
        lines.append("[OK] No semantic issues found")

    return "\n".join(lines)
