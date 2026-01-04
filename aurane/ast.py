"""
AST node definitions for Aurane DSL.

This module defines the abstract syntax tree nodes used to represent
parsed Aurane code before code generation.
"""

from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Union


@dataclass
class UseStatement:
    """Represents a 'use' import statement."""

    module: str
    alias: Optional[str] = None


@dataclass
class Variable:
    """Represents a variable assignment."""

    name: str
    value: Any
    type_hint: Optional[str] = None


@dataclass
class HyperParameter:
    """Represents a hyperparameter with search space."""

    name: str
    default: Any
    range: Optional[tuple] = None  # (min, max) for numerical
    choices: Optional[List[Any]] = None  # For categorical


@dataclass
class ExperimentNode:
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
class DatasetNode:
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
class LayerOperation:
    """
    Represents a single layer or operation in a model forward chain.

    Example: conv2d(32, kernel=3) with activation relu
    """

    operation: str  # e.g., "conv2d", "maxpool", "flatten", "dense"
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    activation: Optional[str] = None  # e.g., "relu", "sigmoid"


@dataclass
class CustomLayer:
    """Represents a custom layer definition."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    operations: List[LayerOperation] = field(default_factory=list)


@dataclass
class ForwardBlock:
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
class ModelNode:
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
    forward_block: Optional[ForwardBlock] = None
    custom_layers: List[CustomLayer] = field(default_factory=list)


@dataclass
class Callback:
    """Represents a training callback."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
    """Represents a training/evaluation metric."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LRScheduler:
    """Represents a learning rate scheduler."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainNode:
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
    custom_layers: List[CustomLayer] = field(default_factory=list)
