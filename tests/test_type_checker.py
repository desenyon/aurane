"""
Tests for the Aurane type checker module.
"""

import pytest
from aurane.parser import parse_aurane
from aurane.type_checker import (
    TypeChecker,
    check_types,
    TensorType,
    TypeCheckResult,
    format_type_errors,
)


class TestTensorType:
    """Tests for TensorType class."""
    
    def test_tensor_type_creation(self):
        """Test creating a TensorType."""
        t = TensorType(shape=(3, 224, 224), dtype="float32")
        assert t.shape == (3, 224, 224)
        assert t.dtype == "float32"
    
    def test_tensor_type_default_dtype(self):
        """Test TensorType with default dtype."""
        t = TensorType(shape=(10,))
        assert t.shape == (10,)
        assert t.dtype == "float32"  # default
    
    def test_tensor_type_str(self):
        """Test TensorType string representation."""
        t = TensorType(shape=(3, 32, 32), dtype="float32")
        str_repr = str(t)
        assert "3" in str_repr
        assert "32" in str_repr
    
    def test_tensor_type_compatibility(self):
        """Test TensorType compatibility check."""
        t1 = TensorType(shape=(64, 32), dtype="float32")
        t2 = TensorType(shape=(64, 32), dtype="float32")
        assert t1.is_compatible(t2)


class TestTypeChecker:
    """Tests for TypeChecker class."""
    
    def test_type_checker_initialization(self):
        """Test TypeChecker initialization."""
        source = """model Net:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        checker = TypeChecker(program)
        assert checker.program == program
    
    def test_check_returns_result(self):
        """Test that check returns TypeCheckResult."""
        source = """model Net:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        checker = TypeChecker(program)
        result = checker.check()
        assert isinstance(result, TypeCheckResult)
    
    def test_check_valid_model(self):
        """Test checking a valid model."""
        source = """model ValidNet:
    input_shape = (1, 28, 28)
    def forward(x):
        x -> conv2d(32, kernel=3).relu
          -> maxpool(2)
          -> flatten()
          -> dense(10)
"""
        program = parse_aurane(source)
        checker = TypeChecker(program)
        result = checker.check()
        assert result.is_valid


class TestCheckTypes:
    """Tests for check_types helper function."""
    
    def test_check_types_helper(self):
        """Test check_types helper function."""
        source = """model Net:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(64, kernel=3).relu
          -> flatten()
          -> dense(10)
"""
        program = parse_aurane(source)
        result = check_types(program)
        assert isinstance(result, TypeCheckResult)
    
    def test_check_types_empty_program(self):
        """Test check_types on empty program."""
        program = parse_aurane("")
        result = check_types(program)
        assert result.is_valid


class TestShapeInference:
    """Tests for shape inference in type checking."""
    
    def test_conv2d_shape_inference(self):
        """Test conv2d shape inference."""
        source = """model ConvNet:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(64, kernel=3, padding=1)
"""
        program = parse_aurane(source)
        result = check_types(program)
        assert result.is_valid
    
    def test_dense_shape_inference(self):
        """Test dense layer shape inference."""
        source = """model DenseNet:
    input_shape = (784,)
    def forward(x):
        x -> dense(256).relu
          -> dense(128).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        result = check_types(program)
        assert result.is_valid
    
    def test_flatten_shape_inference(self):
        """Test flatten shape inference."""
        source = """model FlattenNet:
    input_shape = (64, 4, 4)
    def forward(x):
        x -> flatten()
          -> dense(10)
"""
        program = parse_aurane(source)
        result = check_types(program)
        assert result.is_valid
    
    def test_pool_shape_inference(self):
        """Test pooling shape inference."""
        source = """model PoolNet:
    input_shape = (32, 16, 16)
    def forward(x):
        x -> maxpool(2)
          -> avgpool(2)
"""
        program = parse_aurane(source)
        result = check_types(program)
        assert result.is_valid


class TestTypeCheckResult:
    """Tests for TypeCheckResult class."""
    
    def test_successful_result(self):
        """Test successful type check result."""
        result = TypeCheckResult()
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_result_with_warnings(self):
        """Test result with warnings."""
        source = """model Net:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = check_types(program)
        # May have warnings but should still be valid
        assert isinstance(result.warnings, list)


class TestFormatTypeErrors:
    """Tests for format_type_errors function."""
    
    def test_format_empty_errors(self):
        """Test formatting empty error list."""
        source = """model Net:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = check_types(program)
        formatted = format_type_errors(result)
        assert isinstance(formatted, str)


class TestComplexModels:
    """Tests for type checking complex models."""
    
    def test_cnn_type_check(self):
        """Test type checking CNN model."""
        source = """model CNN:
    input_shape = (3, 224, 224)
    def forward(x):
        x -> conv2d(64, kernel=3, padding=1).relu
          -> conv2d(64, kernel=3, padding=1).relu
          -> maxpool(2)
          -> conv2d(128, kernel=3, padding=1).relu
          -> maxpool(2)
          -> flatten()
          -> dense(512).relu
          -> dropout(0.5)
          -> dense(1000)
"""
        program = parse_aurane(source)
        result = check_types(program)
        assert result.is_valid
    
    def test_multiple_models_type_check(self):
        """Test type checking multiple models."""
        source = """model ModelA:
    input_shape = (784,)
    def forward(x):
        x -> dense(128).relu
          -> dense(10)

model ModelB:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(32, kernel=3).relu
          -> flatten()
          -> dense(10)
"""
        program = parse_aurane(source)
        result = check_types(program)
        assert result.is_valid


class TestEdgeCases:
    """Tests for edge cases in type checking."""
    
    def test_empty_model(self):
        """Test type checking empty model."""
        source = """model EmptyModel:
    input_shape = (3, 32, 32)
"""
        program = parse_aurane(source)
        result = check_types(program)
        # Should have warning but no error
    
    def test_model_without_input_shape(self):
        """Test model without explicit input shape."""
        source = """model NoInputShape:
    def forward(x):
        x -> dense(64)
          -> dense(10)
"""
        program = parse_aurane(source)
        result = check_types(program)
        # Should use default input shape
    
    def test_single_layer_model(self):
        """Test single layer model."""
        source = """model SingleLayer:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = check_types(program)
        assert result.is_valid


class TestActivationTypes:
    """Tests for activation function type handling."""
    
    def test_relu_preserves_type(self):
        """Test that ReLU preserves tensor type."""
        source = """model ReluNet:
    input_shape = (64,)
    def forward(x):
        x -> dense(32).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        result = check_types(program)
        assert result.is_valid
    
    def test_softmax_preserves_type(self):
        """Test that softmax preserves tensor type."""
        source = """model SoftmaxNet:
    input_shape = (64,)
    def forward(x):
        x -> dense(10).softmax
"""
        program = parse_aurane(source)
        result = check_types(program)
        assert result.is_valid


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
