"""
Tests for the Aurane parser module.
"""

import pytest
from aurane.parser import Parser, parse_aurane, ParseError
from aurane.ast import (
    AuraneProgram,
    UseStatement,
    ExperimentNode,
    DatasetNode,
    ModelNode,
    TrainNode,
    ForwardBlock,
    LayerOperation,
)


class TestParser:
    """Tests for the Parser class."""
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        source = "use torch"
        parser = Parser(source)
        assert parser.lines == ["use torch"]
        assert parser.current_line == 0
    
    def test_parser_empty_source(self):
        """Test parsing empty source."""
        parser = Parser("")
        program = parser.parse()
        assert isinstance(program, AuraneProgram)
        assert len(program.uses) == 0
        assert len(program.models) == 0
    
    def test_parser_comments_only(self):
        """Test parsing source with only comments."""
        source = """# This is a comment
# Another comment"""
        program = parse_aurane(source)
        assert len(program.uses) == 0


class TestUseStatements:
    """Tests for use statement parsing."""
    
    def test_simple_use(self):
        """Test parsing simple use statement."""
        source = "use torch"
        program = parse_aurane(source)
        assert len(program.uses) == 1
        assert program.uses[0].module == "torch"
        assert program.uses[0].alias is None
    
    def test_use_with_alias(self):
        """Test parsing use statement with alias."""
        source = "use torch as t"
        program = parse_aurane(source)
        assert len(program.uses) == 1
        assert program.uses[0].module == "torch"
        assert program.uses[0].alias == "t"
    
    def test_multiple_uses(self):
        """Test parsing multiple use statements."""
        source = """use torch
use numpy as np
use pandas"""
        program = parse_aurane(source)
        assert len(program.uses) == 3
        assert program.uses[0].module == "torch"
        assert program.uses[1].module == "numpy"
        assert program.uses[1].alias == "np"
        assert program.uses[2].module == "pandas"
    
    def test_use_with_dots(self):
        """Test parsing use statement with dotted module."""
        source = "use aurane.ml as aml"
        program = parse_aurane(source)
        assert program.uses[0].module == "aurane.ml"
        assert program.uses[0].alias == "aml"


class TestExperimentParsing:
    """Tests for experiment block parsing."""
    
    def test_simple_experiment(self):
        """Test parsing simple experiment."""
        source = """experiment MyExperiment:
    seed = 42
    device = "auto"
"""
        program = parse_aurane(source)
        assert len(program.experiments) == 1
        exp = program.experiments[0]
        assert exp.name == "MyExperiment"
        assert exp.config['seed'] == 42
        assert exp.config['device'] == "auto"
    
    def test_experiment_with_boolean(self):
        """Test parsing experiment with boolean values."""
        source = """experiment Test:
    debug = True
    verbose = False
"""
        program = parse_aurane(source)
        assert program.experiments[0].config['debug'] is True
        assert program.experiments[0].config['verbose'] is False
    
    def test_experiment_with_tuple(self):
        """Test parsing experiment with tuple values."""
        source = """experiment Test:
    input_shape = (3, 224, 224)
"""
        program = parse_aurane(source)
        assert program.experiments[0].config['input_shape'] == (3, 224, 224)


class TestDatasetParsing:
    """Tests for dataset block parsing."""
    
    def test_simple_dataset(self):
        """Test parsing simple dataset."""
        source = """dataset mnist_train:
    from torchvision.datasets.MNIST
    root = "./data"
    train = True
    batch = 128
"""
        program = parse_aurane(source)
        assert len(program.datasets) == 1
        ds = program.datasets[0]
        assert ds.name == "mnist_train"
        assert ds.source == "torchvision.datasets.MNIST"
        assert ds.config['root'] == "./data"
        assert ds.config['train'] is True
        assert ds.config['batch'] == 128
    
    def test_dataset_without_source(self):
        """Test parsing dataset without source."""
        source = """dataset custom:
    batch = 64
"""
        program = parse_aurane(source)
        assert program.datasets[0].source is None
        assert program.datasets[0].config['batch'] == 64
    
    def test_multiple_datasets(self):
        """Test parsing multiple datasets."""
        source = """dataset train:
    train = True
    batch = 32

dataset test:
    train = False
    batch = 64
"""
        program = parse_aurane(source)
        assert len(program.datasets) == 2
        assert program.datasets[0].name == "train"
        assert program.datasets[1].name == "test"


class TestModelParsing:
    """Tests for model block parsing."""
    
    def test_simple_model(self):
        """Test parsing simple model."""
        source = """model SimpleNet:
    input_shape = (1, 28, 28)
    def forward(x):
        x -> conv2d(32, kernel=3).relu
          -> flatten()
          -> dense(10)
"""
        program = parse_aurane(source)
        assert len(program.models) == 1
        model = program.models[0]
        assert model.name == "SimpleNet"
        assert model.config['input_shape'] == (1, 28, 28)
        assert model.forward_block is not None
        assert len(model.forward_block.operations) == 3
    
    def test_model_operations(self):
        """Test parsing model operations."""
        source = """model TestNet:
    def forward(x):
        x -> conv2d(64, kernel=3, padding=1).relu
          -> batchnorm()
          -> maxpool(2)
"""
        program = parse_aurane(source)
        assert program.models[0].forward_block is not None
        ops = program.models[0].forward_block.operations
        
        assert ops[0].operation == "conv2d"
        assert ops[0].args == [64]
        assert ops[0].kwargs == {'kernel': 3, 'padding': 1}
        assert ops[0].activation == "relu"
        
        assert ops[1].operation == "batchnorm"
        assert ops[1].args == []
        
        assert ops[2].operation == "maxpool"
        assert ops[2].args == [2]
    
    def test_model_without_forward(self):
        """Test parsing model without forward block."""
        source = """model EmptyModel:
    input_shape = (3, 32, 32)
"""
        program = parse_aurane(source)
        assert program.models[0].forward_block is None
    
    def test_model_activations(self):
        """Test parsing various activations."""
        source = """model ActivationTest:
    def forward(x):
        x -> dense(64).relu
          -> dense(32).gelu
          -> dense(16).sigmoid
          -> dense(10).softmax
"""
        program = parse_aurane(source)
        assert program.models[0].forward_block is not None
        ops = program.models[0].forward_block.operations
        
        assert ops[0].activation == "relu"
        assert ops[1].activation == "gelu"
        assert ops[2].activation == "sigmoid"
        assert ops[3].activation == "softmax"


class TestTrainParsing:
    """Tests for train block parsing."""
    
    def test_simple_train(self):
        """Test parsing simple train block."""
        source = """train MyModel on my_dataset:
    epochs = 10
    lr = 0.001
    optimizer = "adam"
    loss = "cross_entropy"
"""
        program = parse_aurane(source)
        assert len(program.trains) == 1
        train = program.trains[0]
        assert train.model_name == "MyModel"
        assert train.dataset_name == "my_dataset"
        assert train.config['epochs'] == 10
        assert train.config['lr'] == 0.001
        assert train.config['optimizer'] == "adam"
    
    def test_train_with_list(self):
        """Test parsing train block with list values."""
        source = """train Net on data:
    metrics = [accuracy, loss]
"""
        program = parse_aurane(source)
        assert program.trains[0].config['metrics'] == ['accuracy', 'loss']


class TestCompleteProgram:
    """Tests for parsing complete programs."""
    
    def test_mnist_program(self):
        """Test parsing a complete MNIST program."""
        source = """use torch

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
"""
        program = parse_aurane(source)
        
        assert len(program.uses) == 1
        assert len(program.experiments) == 1
        assert len(program.datasets) == 1
        assert len(program.models) == 1
        assert len(program.trains) == 1
        
        model = program.models[0]
        assert model.name == "MnistNet"
        assert model.forward_block is not None
        assert len(model.forward_block.operations) == 8


class TestValueParsing:
    """Tests for value parsing."""
    
    def test_integer_parsing(self):
        """Test parsing integer values."""
        source = """experiment Test:
    value = 42
"""
        program = parse_aurane(source)
        assert program.experiments[0].config['value'] == 42
    
    def test_float_parsing(self):
        """Test parsing float values."""
        source = """experiment Test:
    lr = 0.001
    momentum = 0.9
"""
        program = parse_aurane(source)
        assert program.experiments[0].config['lr'] == 0.001
        assert program.experiments[0].config['momentum'] == 0.9
    
    def test_string_parsing(self):
        """Test parsing string values."""
        source = """experiment Test:
    name = "test_experiment"
    path = './data'
"""
        program = parse_aurane(source)
        assert program.experiments[0].config['name'] == "test_experiment"
        assert program.experiments[0].config['path'] == "./data"
    
    def test_list_parsing(self):
        """Test parsing list values."""
        source = """experiment Test:
    layers = [64, 128, 256]
"""
        program = parse_aurane(source)
        assert program.experiments[0].config['layers'] == [64, 128, 256]


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_config_block(self):
        """Test parsing blocks with empty config."""
        source = """experiment Empty:
"""
        program = parse_aurane(source)
        assert len(program.experiments) == 1
        assert program.experiments[0].config == {}
    
    def test_whitespace_handling(self):
        """Test handling of various whitespace."""
        source = """use torch

experiment   Test:
    seed   =   42
"""
        program = parse_aurane(source)
        assert program.experiments[0].config['seed'] == 42
    
    def test_comments_in_blocks(self):
        """Test comments within blocks."""
        source = """experiment Test:
    # This is a comment
    seed = 42
    # Another comment
    device = "cpu"
"""
        program = parse_aurane(source)
        assert program.experiments[0].config['seed'] == 42
        assert program.experiments[0].config['device'] == "cpu"


class TestParseAuraneFunction:
    """Tests for the parse_aurane helper function."""
    
    def test_parse_aurane_returns_program(self):
        """Test that parse_aurane returns AuraneProgram."""
        result = parse_aurane("use torch")
        assert isinstance(result, AuraneProgram)
    
    def test_parse_aurane_empty(self):
        """Test parse_aurane with empty string."""
        result = parse_aurane("")
        assert isinstance(result, AuraneProgram)
        assert len(result.uses) == 0


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
