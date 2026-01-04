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

from .ast import (
    AuraneProgram,
    ModelNode,
    DatasetNode,
    TrainNode,
    ExperimentNode,
    LayerOperation,
    ForwardBlock,
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


# Known layer operations and their expected arguments
LAYER_SPECS = {
    "conv2d": {
        "args": ["out_channels"],
        "kwargs": {"kernel": int, "stride": int, "padding": int, "bias": bool},
        "activations": ["relu", "gelu", "leaky_relu", "sigmoid", "tanh"],
    },
    "conv1d": {
        "args": ["out_channels"],
        "kwargs": {"kernel": int, "stride": int, "padding": int},
        "activations": ["relu", "gelu", "leaky_relu"],
    },
    "dense": {
        "args": ["out_features"],
        "kwargs": {"bias": bool},
        "activations": ["relu", "gelu", "sigmoid", "tanh", "softmax"],
    },
    "linear": {
        "args": ["out_features"],
        "kwargs": {"bias": bool},
        "activations": ["relu", "gelu", "sigmoid", "tanh", "softmax"],
    },
    "maxpool": {
        "args": ["kernel_size"],
        "kwargs": {"stride": int},
        "activations": [],
    },
    "avgpool": {
        "args": ["kernel_size"],
        "kwargs": {"stride": int},
        "activations": [],
    },
    "dropout": {
        "args": ["rate"],
        "kwargs": {},
        "activations": [],
    },
    "batchnorm": {
        "args": [],
        "kwargs": {"momentum": float, "eps": float},
        "activations": ["relu", "gelu"],
    },
    "layer_norm": {
        "args": [],
        "kwargs": {"eps": float},
        "activations": [],
    },
    "flatten": {
        "args": [],
        "kwargs": {},
        "activations": [],
    },
    "embedding": {
        "args": ["num_embeddings", "embedding_dim"],
        "kwargs": {"padding_idx": int},
        "activations": [],
    },
    "multihead_attention": {
        "args": [],
        "kwargs": {"heads": int, "dim": int, "dropout": float},
        "activations": [],
    },
    "positional_encoding": {
        "args": [],
        "kwargs": {"max_len": int},
        "activations": [],
    },
    "lstm": {
        "args": ["hidden_size"],
        "kwargs": {"num_layers": int, "bidirectional": bool, "dropout": float},
        "activations": [],
    },
    "gru": {
        "args": ["hidden_size"],
        "kwargs": {"num_layers": int, "bidirectional": bool, "dropout": float},
        "activations": [],
    },
    "global_avg_pool": {
        "args": [],
        "kwargs": {},
        "activations": [],
    },
    "upsample": {
        "args": [],
        "kwargs": {"scale_factor": int, "mode": str},
        "activations": [],
    },
    "concat": {
        "args": [],
        "kwargs": {"dim": int},
        "activations": [],
    },
    "add": {
        "args": [],
        "kwargs": {},
        "activations": ["relu"],
    },
}

# Known activation functions
ACTIVATIONS = {
    "relu",
    "gelu",
    "leaky_relu",
    "sigmoid",
    "tanh",
    "softmax",
    "silu",
    "mish",
    "elu",
    "selu",
    "swish",
    "hardswish",
    "residual",  # Special case for residual connections
}

# Known optimizers
OPTIMIZERS = {
    "adam",
    "sgd",
    "adamw",
    "rmsprop",
    "adagrad",
    "adadelta",
}

# Known loss functions
LOSS_FUNCTIONS = {
    "cross_entropy",
    "mse",
    "mae",
    "bce",
    "nll",
    "huber",
    "cross_entropy_loss",
    "mse_loss",
    "l1_loss",
    "bce_loss",
}


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

        # Check if operation is known
        if op_name not in LAYER_SPECS:
            self._add_issue(
                IssueKind.WARNING, f"Unknown operation: {op.operation}", location, "W007"
            )
            return

        spec = LAYER_SPECS[op_name]

        # Check activation
        if op.activation:
            activation = op.activation.lower()
            if activation not in ACTIVATIONS:
                self._add_issue(
                    IssueKind.WARNING, f"Unknown activation: {op.activation}", location, "W008"
                )
            elif (
                spec["activations"]
                and activation not in spec["activations"]
                and activation != "residual"
            ):
                self._add_issue(
                    IssueKind.INFO,
                    f"Unusual activation {op.activation} for {op_name}",
                    location,
                    "I001",
                )

        # Check dropout rate
        if op_name == "dropout" and op.args:
            rate = op.args[0]
            if isinstance(rate, (int, float)):
                if rate <= 0 or rate >= 1:
                    self._add_issue(
                        IssueKind.ERROR,
                        f"Dropout rate must be between 0 and 1, got {rate}",
                        location,
                        "E005",
                    )
                elif rate > 0.5:
                    self._add_issue(
                        IssueKind.SUGGESTION,
                        f"High dropout rate ({rate}) may hurt training",
                        location,
                        "S003",
                    )

    def _check_model_patterns(self, model: ModelNode):
        """Check for common model patterns and best practices."""
        if not model.forward_block:
            return

        ops = model.forward_block.operations
        op_names = [op.operation.lower() for op in ops]

        # Check for missing batchnorm after conv
        for i, (op, name) in enumerate(zip(ops, op_names)):
            if name == "conv2d":
                if i + 1 < len(ops) and op_names[i + 1] != "batchnorm":
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
        for train in self.program.trains:
            location = f"train {train.model_name}"

            # Check references
            if train.model_name not in self.defined_models:
                self._add_issue(
                    IssueKind.ERROR, f"Undefined model: {train.model_name}", location, "E006"
                )

            if train.dataset_name not in self.defined_datasets:
                self._add_issue(
                    IssueKind.ERROR, f"Undefined dataset: {train.dataset_name}", location, "E007"
                )

            # Check optimizer
            optimizer = train.config.get("optimizer")
            if optimizer:
                opt_name = (
                    optimizer.split("(")[0].lower() if "(" in optimizer else optimizer.lower()
                )
                if opt_name not in OPTIMIZERS:
                    self._add_issue(
                        IssueKind.WARNING, f"Unknown optimizer: {optimizer}", location, "W010"
                    )

            # Check loss function
            loss = train.config.get("loss")
            if loss:
                loss_name = loss.lower().replace("_", "")
                known = any(l.replace("_", "") in loss_name for l in LOSS_FUNCTIONS)
                if not known:
                    self._add_issue(
                        IssueKind.WARNING, f"Unknown loss function: {loss}", location, "W011"
                    )

            # Check epochs
            epochs = train.config.get("epochs")
            if epochs and isinstance(epochs, int):
                if epochs < 1:
                    self._add_issue(IssueKind.ERROR, "Epochs must be at least 1", location, "E008")
                elif epochs > 1000:
                    self._add_issue(
                        IssueKind.SUGGESTION,
                        f"Many epochs ({epochs}), consider early stopping",
                        location,
                        "S006",
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
        self, kind: IssueKind, message: str, location: str, code: str, fix: Optional[str] = None
    ):
        """Add an issue to results."""
        self.result.issues.append(
            SemanticIssue(kind=kind, message=message, location=location, code=code, fix=fix)
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
