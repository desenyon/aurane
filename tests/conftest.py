"""
Pytest configuration and shared fixtures for Aurane tests.
"""

import pytest
import tempfile
import os
from pathlib import Path

from aurane.parser import parse_aurane
from aurane.ast import (
    AuraneProgram,
    UseStatement,
    ModelNode,
    ForwardBlock,
    LayerOperation,
)


# ============================================================================
# Sample Source Code Fixtures
# ============================================================================

@pytest.fixture
def simple_model_source():
    """Simple model source code."""
    return """model SimpleNet:
    def forward(x):
        x -> dense(10)
"""


@pytest.fixture
def cnn_model_source():
    """CNN model source code."""
    return """model CNN:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(32, kernel=3, padding=1).relu
          -> maxpool(2)
          -> conv2d(64, kernel=3, padding=1).relu
          -> maxpool(2)
          -> flatten()
          -> dense(256).relu
          -> dropout(0.5)
          -> dense(10)
"""


@pytest.fixture
def mnist_model_source():
    """MNIST model source code."""
    return """model MnistNet:
    input_shape = (1, 28, 28)
    def forward(x):
        x -> conv2d(32, kernel=3).relu
          -> maxpool(2)
          -> conv2d(64, kernel=3).relu
          -> maxpool(2)
          -> flatten()
          -> dense(128).relu
          -> dropout(0.5)
          -> dense(10)
"""


@pytest.fixture
def complete_program_source():
    """Complete Aurane program source code."""
    return """use torch

experiment MnistBaseline:
    seed = 42
    device = "auto"

dataset mnist_train:
    from torchvision.datasets.MNIST
    root = "./data"
    train = True
    batch = 128

model MnistNet:
    input_shape = (1, 28, 28)
    def forward(x):
        x -> conv2d(32, kernel=3).relu
          -> maxpool(2)
          -> conv2d(64, kernel=3).relu
          -> maxpool(2)
          -> flatten()
          -> dense(128).relu
          -> dropout(0.5)
          -> dense(10)

train MnistNet on mnist_train:
    epochs = 10
    lr = 0.001
    optimizer = "adam"
    loss = "cross_entropy"
"""


@pytest.fixture
def mlp_model_source():
    """Multi-layer perceptron source code."""
    return """model MLP:
    input_shape = (784,)
    def forward(x):
        x -> dense(512).relu
          -> dense(256).relu
          -> dense(128).relu
          -> dense(10)
"""


# ============================================================================
# Parsed Program Fixtures
# ============================================================================

@pytest.fixture
def simple_program(simple_model_source):
    """Parsed simple model program."""
    return parse_aurane(simple_model_source)


@pytest.fixture
def cnn_program(cnn_model_source):
    """Parsed CNN program."""
    return parse_aurane(cnn_model_source)


@pytest.fixture
def mnist_program(mnist_model_source):
    """Parsed MNIST program."""
    return parse_aurane(mnist_model_source)


@pytest.fixture
def complete_program(complete_program_source):
    """Parsed complete program."""
    return parse_aurane(complete_program_source)


@pytest.fixture
def mlp_program(mlp_model_source):
    """Parsed MLP program."""
    return parse_aurane(mlp_model_source)


@pytest.fixture
def empty_program():
    """Empty Aurane program."""
    return parse_aurane("")


# ============================================================================
# File Fixtures
# ============================================================================

@pytest.fixture
def temp_aurane_file(simple_model_source):
    """Create a temporary .aur file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.aur', delete=False) as f:
        f.write(simple_model_source)
        f.flush()
        yield f.name
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def temp_cnn_file(cnn_model_source):
    """Create a temporary CNN .aur file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.aur', delete=False) as f:
        f.write(cnn_model_source)
        f.flush()
        yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def temp_complete_file(complete_program_source):
    """Create a temporary complete program .aur file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.aur', delete=False) as f:
        f.write(complete_program_source)
        f.flush()
        yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# AST Node Fixtures
# ============================================================================

@pytest.fixture
def use_torch():
    """UseStatement for torch."""
    return UseStatement("torch", None)


@pytest.fixture
def use_numpy():
    """UseStatement for numpy."""
    return UseStatement("numpy", "np")


@pytest.fixture
def conv2d_op():
    """Conv2d LayerOperation."""
    return LayerOperation("conv2d", [32], {"kernel": 3, "padding": 1}, "relu")


@pytest.fixture
def dense_op():
    """Dense LayerOperation."""
    return LayerOperation("dense", [128], {}, "relu")


@pytest.fixture
def flatten_op():
    """Flatten LayerOperation."""
    return LayerOperation("flatten", [], {}, None)


@pytest.fixture
def dropout_op():
    """Dropout LayerOperation."""
    return LayerOperation("dropout", [0.5], {}, None)


@pytest.fixture
def maxpool_op():
    """MaxPool LayerOperation."""
    return LayerOperation("maxpool", [2], {}, None)


@pytest.fixture
def simple_forward_block(dense_op):
    """Simple ForwardBlock with one dense layer."""
    return ForwardBlock("x", [dense_op])


@pytest.fixture
def cnn_forward_block(conv2d_op, maxpool_op, flatten_op, dense_op):
    """CNN ForwardBlock."""
    output_dense = LayerOperation("dense", [10], {}, None)
    return ForwardBlock("x", [conv2d_op, maxpool_op, flatten_op, output_dense])


# ============================================================================
# Utility Functions
# ============================================================================

def create_temp_file(content: str, suffix: str = '.aur') -> str:
    """Create a temporary file with given content."""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(content)
        f.flush()
        return f.name


def cleanup_file(path: str):
    """Remove a file if it exists."""
    if os.path.exists(path):
        os.unlink(path)


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "cli: marks tests as CLI tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test file names
    for item in items:
        if "test_cli" in str(item.fspath):
            item.add_marker(pytest.mark.cli)
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
