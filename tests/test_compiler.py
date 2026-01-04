"""
Tests for the Aurane compiler module.
"""

import pytest
import tempfile
import os
from pathlib import Path

from aurane.compiler import compile_file, compile_source, CompilationError


class TestCompileSource:
    """Tests for compile_source function."""
    
    def test_compile_simple_source(self):
        """Test compiling simple source code."""
        source = """use torch

model SimpleNet:
    input_shape = (1, 28, 28)
    def forward(x):
        x -> dense(10)
"""
        code = compile_source(source)
        assert isinstance(code, str)
        assert "SimpleNet" in code
    
    def test_compile_empty_source(self):
        """Test compiling empty source."""
        code = compile_source("")
        assert isinstance(code, str)
    
    def test_compile_use_only(self):
        """Test compiling source with only use statements."""
        source = """use torch"""
        code = compile_source(source)
        assert isinstance(code, str)
        assert "import torch" in code
    
    def test_compile_generates_class(self):
        """Test that compile generates PyTorch class."""
        source = """model TestModel:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(64, kernel=3).relu
          -> flatten()
          -> dense(10)
"""
        code = compile_source(source)
        assert isinstance(code, str)
        assert "class TestModel" in code
        assert "nn.Module" in code
        assert "def __init__" in code
        assert "def forward" in code
    
    def test_compile_conv2d_layer(self):
        """Test compiling conv2d layer."""
        source = """model ConvNet:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(32, kernel=3, padding=1).relu
"""
        code = compile_source(source)
        assert isinstance(code, str)
        assert "nn.Conv2d" in code
    
    def test_compile_dense_layer(self):
        """Test compiling dense layer."""
        source = """model DenseNet:
    input_shape = (1, 28, 28)
    def forward(x):
        x -> flatten()
          -> dense(128).relu
          -> dense(10)
"""
        code = compile_source(source)
        assert isinstance(code, str)
        assert "nn.Linear" in code
    
    def test_compile_multiple_models(self):
        """Test compiling multiple models."""
        source = """model ModelA:
    def forward(x):
        x -> dense(10)

model ModelB:
    def forward(x):
        x -> dense(20)
"""
        code = compile_source(source)
        assert isinstance(code, str)
        assert "class ModelA" in code
        assert "class ModelB" in code
    
    def test_compile_with_experiment(self):
        """Test compiling with experiment block."""
        source = """experiment TestExp:
    seed = 42
    device = "cuda"

model Net:
    def forward(x):
        x -> dense(10)
"""
        code = compile_source(source)
        assert isinstance(code, str)
        assert "class Net" in code
    
    def test_compile_with_dataset(self):
        """Test compiling with dataset block."""
        source = """dataset my_data:
    from torchvision.datasets.MNIST
    batch = 32

model Net:
    def forward(x):
        x -> dense(10)
"""
        code = compile_source(source)
        assert isinstance(code, str)
    
    def test_compile_with_train(self):
        """Test compiling with train block."""
        source = """model Net:
    def forward(x):
        x -> dense(10)

train Net on data:
    epochs = 10
    lr = 0.001
"""
        code = compile_source(source)
        assert isinstance(code, str)
    
    def test_compile_activations(self):
        """Test compiling various activations."""
        source = """model ActivationNet:
    def forward(x):
        x -> dense(64).relu
          -> dense(32).gelu
          -> dense(16).sigmoid
          -> dense(10).softmax
"""
        code = compile_source(source)
        assert isinstance(code, str)
    
    def test_compile_dropout(self):
        """Test compiling dropout layer."""
        source = """model DropoutNet:
    def forward(x):
        x -> dense(64).relu
          -> dropout(0.5)
          -> dense(10)
"""
        code = compile_source(source)
        assert isinstance(code, str)
        assert "Dropout" in code or "dropout" in code.lower()
    
    def test_compile_batchnorm(self):
        """Test compiling batchnorm layer."""
        source = """model BNNet:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(32, kernel=3).relu
          -> batchnorm()
"""
        code = compile_source(source)
        assert isinstance(code, str)
    
    def test_compile_pooling(self):
        """Test compiling pooling layers."""
        source = """model PoolNet:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(32, kernel=3).relu
          -> maxpool(2)
          -> conv2d(64, kernel=3).relu
          -> avgpool(2)
"""
        code = compile_source(source)
        assert isinstance(code, str)


class TestCompileFile:
    """Tests for compile_file function."""
    
    def test_compile_existing_file(self):
        """Test compiling an existing file."""
        source = """use torch

model FileNet:
    def forward(x):
        x -> dense(10)
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.aur', delete=False) as src_f:
            src_f.write(source)
            src_f.flush()
            
            output_path = src_f.name.replace('.aur', '.py')
            compile_file(src_f.name, output_path)
            
            assert os.path.exists(output_path)
            with open(output_path, 'r') as out_f:
                content = out_f.read()
                assert "FileNet" in content
            
            os.unlink(src_f.name)
            os.unlink(output_path)
    
    def test_compile_nonexistent_file(self):
        """Test compiling nonexistent file."""
        with pytest.raises(FileNotFoundError):
            compile_file("/nonexistent/path/file.aur", "/tmp/out.py")


class TestGeneratedCode:
    """Tests for the structure of generated code."""
    
    def test_generated_code_imports(self):
        """Test that generated code has necessary imports."""
        source = """model Net:
    def forward(x):
        x -> conv2d(32, kernel=3).relu
"""
        code = compile_source(source)
        assert "import torch" in code
        assert "torch.nn" in code or "import torch.nn as nn" in code
    
    def test_generated_code_is_valid_python(self):
        """Test that generated code is valid Python."""
        source = """model ValidNet:
    def forward(x):
        x -> dense(64).relu
          -> dense(10)
"""
        code = compile_source(source)
        
        # Try to compile the generated code as Python
        try:
            compile(code, '<string>', 'exec')
            is_valid = True
        except SyntaxError:
            is_valid = False
        
        assert is_valid, "Generated code has syntax errors"
    
    def test_generated_init_method(self):
        """Test that generated __init__ method is correct."""
        source = """model InitNet:
    def forward(x):
        x -> dense(128).relu
          -> dense(10)
"""
        code = compile_source(source)
        assert "def __init__(self)" in code
        assert "super()" in code
    
    def test_generated_forward_method(self):
        """Test that generated forward method is correct."""
        source = """model ForwardNet:
    def forward(x):
        x -> dense(10)
"""
        code = compile_source(source)
        assert "def forward(self, x)" in code


class TestComplexModels:
    """Tests for compiling complex models."""
    
    def test_compile_cnn(self):
        """Test compiling a CNN model."""
        source = """model CNN:
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
        code = compile_source(source)
        assert isinstance(code, str)
        assert "nn.Conv2d" in code
    
    def test_compile_sequential_layers(self):
        """Test compiling sequential layers."""
        source = """model SeqNet:
    def forward(x):
        x -> dense(512).relu
          -> dense(256).relu
          -> dense(128).relu
          -> dense(64).relu
          -> dense(10)
"""
        code = compile_source(source)
        assert isinstance(code, str)
        assert "nn.Linear" in code


class TestEdgeCases:
    """Tests for edge cases in compilation."""
    
    def test_compile_comments(self):
        """Test that comments are handled."""
        source = """# This is a comment
use torch
# Another comment
model Net:
    # Comment inside
    def forward(x):
        x -> dense(10)
"""
        code = compile_source(source)
        assert isinstance(code, str)
    
    def test_compile_extra_whitespace(self):
        """Test handling extra whitespace."""
        source = """

use torch


model Net:


    def forward(x):
        x -> dense(10)

"""
        code = compile_source(source)
        assert isinstance(code, str)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
