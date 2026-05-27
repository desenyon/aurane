"""
AST node definitions for Aurane DSL.

This module defines the abstract syntax tree nodes used to represent
parsed Aurane code before code generation.
"""

from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Union


@dataclass(kw_only=True)
class ASTNode:
    """Base class for all AST nodes with position tracking."""

    line: int = 0
    column: int = 0


@dataclass
class UseStatement(ASTNode):
    """Represents a 'use' import statement."""

    module: str
    alias: Optional[str] = None


@dataclass
class Variable(ASTNode):
    """Represents a variable assignment."""

    name: str
    value: Any
    type_hint: Optional[str] = None


@dataclass
class HyperParameter(ASTNode):
    """Represents a hyperparameter with search space."""

    name: str
    default: Any
    range: Optional[tuple] = None  # (min, max) for numerical
    choices: Optional[List[Any]] = None  # For categorical


@dataclass
class ExperimentNode(ASTNode):
    """
    Represents an 'experiment' block.

    Example:
        experiment MnistBaseline:
            seed = 42
            device = "auto"
            backend = "torch"
    """

    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: List[HyperParameter] = field(default_factory=list)


@dataclass
class DatasetNode(ASTNode):
    """
    Represents a 'dataset' block.

    Example:
        dataset mnist_train:
            from torchvision.datasets.MNIST
            root = "./data"
            train = True
            batch = 128
    """

    name: str
    source: Optional[str] = None  # e.g., "torchvision.datasets.MNIST"
    config: Dict[str, Any] = field(default_factory=dict)
    transforms: List[str] = field(default_factory=list)


@dataclass
class LayerOperation(ASTNode):
    """
    Represents a single layer or operation in a model forward chain.

    Example: conv2d(32, kernel=3) with activation relu
    """

    operation: str  # e.g., "conv2d", "maxpool", "flatten", "dense"
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    activation: Optional[str] = None  # e.g., "relu", "sigmoid"


@dataclass
class CustomLayer(ASTNode):
    """Represents a custom layer definition."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    operations: List[LayerOperation] = field(default_factory=list)


@dataclass
class ForwardBlock(ASTNode):
    """
    Represents the forward pass definition in a model.

    Example:
        def forward(x):
            x -> conv2d(32, kernel=3).relu
              -> maxpool(2)
              -> flatten()
    """

    parameter: str = "x"  # Usually "x"
    operations: List[LayerOperation] = field(default_factory=list)


@dataclass
class GraphOp(ASTNode):
    """
    A single graph statement inside a graph-based forward definition.

    Example DSL:
        h1 = conv2d(x, 32, kernel=3).relu
    """

    target: str = ""
    inputs: List[str] = field(default_factory=list)  # variable names
    operation: Optional[LayerOperation] = field(default=None)  # set by the parser


@dataclass
class ForwardGraphBlock(ASTNode):
    """
    Graph-based forward definition with explicit wiring.

    Example:
        def forward(x):
            h1 = dense(x, 64).relu
            out = dense(h1, 10)
            return out
    """

    parameter: str = "x"
    nodes: List[GraphOp] = field(default_factory=list)
    output_var: Optional[str] = None


@dataclass
class ModelNode(ASTNode):
    """
    Represents a 'model' block.

    Example:
        model MnistNet:
            input_shape = (1, 28, 28)
            def forward(x):
                x -> conv2d(32, kernel=3).relu -> ...
    """

    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    forward_block: Optional[Union[ForwardBlock, ForwardGraphBlock]] = None
    custom_layers: List[CustomLayer] = field(default_factory=list)


@dataclass
class Callback(ASTNode):
    """Represents a training callback."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric(ASTNode):
    """Represents a training/evaluation metric."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LRScheduler(ASTNode):
    """Represents a learning rate scheduler."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainNode(ASTNode):
    """
    Represents a 'train' block.

    Example:
        train MnistNet on mnist_train:
            validate_on = mnist_test
            loss = cross_entropy
            optimizer = adam(lr=1e-3)
            epochs = 5
            metrics = [accuracy]
    """

    model_name: str
    dataset_name: str
    config: Dict[str, Any] = field(default_factory=dict)
    callbacks: List[Callback] = field(default_factory=list)
    metrics: List[Metric] = field(default_factory=list)
    scheduler: Optional[LRScheduler] = None


@dataclass
class TrainGANNode(ASTNode):
    """
    Represents a 'train_gan' block.

    Example:
        train_gan Generator and Discriminator on mnist_images:
            epochs = 200
            generator_optimizer = adam(lr=2e-4)
            ...
    """

    generator_name: str
    discriminator_name: str
    dataset_name: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuraneProgram:
    """
    Represents a complete Aurane program.

    Contains all top-level constructs: imports, experiments, datasets, models, and training blocks.
    """

    uses: List[UseStatement] = field(default_factory=list)
    variables: List[Variable] = field(default_factory=list)
    experiments: List[ExperimentNode] = field(default_factory=list)
    datasets: List[DatasetNode] = field(default_factory=list)
    models: List[ModelNode] = field(default_factory=list)
    trains: List[TrainNode] = field(default_factory=list)
    train_gans: List[TrainGANNode] = field(default_factory=list)
    custom_layers: List[CustomLayer] = field(default_factory=list)
