"""
Tests for the Aurane visualizer module.
"""

import pytest
from aurane.parser import parse_aurane
from aurane.visualizer import (
    calculate_output_shape,
    calculate_parameters,
    print_model_summary,
    visualize_model_architecture,
)
from aurane.ast import LayerOperation, ForwardBlock, ModelNode


class TestCalculateOutputShape:
    """Tests for calculate_output_shape function."""
    
    def test_conv2d_shape(self):
        """Test conv2d output shape calculation."""
        op = LayerOperation(operation="conv2d", args=[32], kwargs={"kernel": 3, "padding": 0, "stride": 1})
        input_shape = (3, 32, 32)
        output = calculate_output_shape(input_shape, op)
        assert output == (32, 30, 30)  # 32 - 3 + 1 = 30
    
    def test_conv2d_with_padding(self):
        """Test conv2d with padding."""
        op = LayerOperation(operation="conv2d", args=[64], kwargs={"kernel": 3, "padding": 1, "stride": 1})
        input_shape = (32, 16, 16)
        output = calculate_output_shape(input_shape, op)
        assert output == (64, 16, 16)  # Same padding
    
    def test_conv2d_with_stride(self):
        """Test conv2d with stride."""
        op = LayerOperation(operation="conv2d", args=[64], kwargs={"kernel": 3, "padding": 1, "stride": 2})
        input_shape = (32, 32, 32)
        output = calculate_output_shape(input_shape, op)
        assert output == (64, 16, 16)  # Halved by stride 2
    
    def test_maxpool_shape(self):
        """Test maxpool output shape calculation."""
        op = LayerOperation(operation="maxpool", args=[2])
        input_shape = (32, 16, 16)
        output = calculate_output_shape(input_shape, op)
        assert output == (32, 8, 8)  # Halved
    
    def test_avgpool_shape(self):
        """Test avgpool output shape calculation."""
        op = LayerOperation(operation="avgpool", args=[2])
        input_shape = (64, 8, 8)
        output = calculate_output_shape(input_shape, op)
        assert output == (64, 4, 4)
    
    def test_flatten_shape(self):
        """Test flatten output shape calculation."""
        op = LayerOperation(operation="flatten")
        input_shape = (64, 4, 4)
        output = calculate_output_shape(input_shape, op)
        assert output == (1024,)  # 64 * 4 * 4
    
    def test_dense_shape(self):
        """Test dense output shape calculation."""
        op = LayerOperation(operation="dense", args=[128])
        input_shape = (512,)
        output = calculate_output_shape(input_shape, op)
        assert output == (128,)
    
    def test_dropout_preserves_shape(self):
        """Test that dropout preserves shape."""
        op = LayerOperation(operation="dropout", args=[0.5])
        input_shape = (256,)
        output = calculate_output_shape(input_shape, op)
        assert output == (256,)
    
    def test_batchnorm_preserves_shape(self):
        """Test that batchnorm preserves shape."""
        op = LayerOperation(operation="batchnorm")
        input_shape = (64, 16, 16)
        output = calculate_output_shape(input_shape, op)
        assert output == (64, 16, 16)
    
    def test_global_avg_pool(self):
        """Test global average pooling."""
        op = LayerOperation(operation="global_avg_pool")
        input_shape = (256, 7, 7)
        output = calculate_output_shape(input_shape, op)
        assert output == (256,)


class TestCalculateParameters:
    """Tests for calculate_parameters function."""
    
    def test_conv2d_params(self):
        """Test conv2d parameter calculation."""
        op = LayerOperation(operation="conv2d", args=[32], kwargs={"kernel": 3})
        input_shape = (3, 32, 32)
        params = calculate_parameters(op, input_shape)
        # params = in_channels * out_channels * kernel * kernel + out_channels (bias)
        # 3 * 32 * 3 * 3 + 32 = 864 + 32 = 896
        assert params == 896
    
    def test_conv2d_params_larger(self):
        """Test conv2d params for larger kernel."""
        op = LayerOperation(operation="conv2d", args=[64], kwargs={"kernel": 5})
        input_shape = (32, 16, 16)
        params = calculate_parameters(op, input_shape)
        # 32 * 64 * 5 * 5 + 64 = 51200 + 64 = 51264
        assert params == 51264
    
    def test_dense_params(self):
        """Test dense parameter calculation."""
        op = LayerOperation(operation="dense", args=[128])
        input_shape = (512,)
        params = calculate_parameters(op, input_shape)
        # 512 * 128 + 128 = 65536 + 128 = 65664
        assert params == 65664
    
    def test_dense_params_small(self):
        """Test dense params for small layer."""
        op = LayerOperation(operation="dense", args=[10])
        input_shape = (64,)
        params = calculate_parameters(op, input_shape)
        # 64 * 10 + 10 = 640 + 10 = 650
        assert params == 650
    
    def test_dropout_no_params(self):
        """Test that dropout has no parameters."""
        op = LayerOperation(operation="dropout", args=[0.5])
        input_shape = (256,)
        params = calculate_parameters(op, input_shape)
        assert params == 0
    
    def test_flatten_no_params(self):
        """Test that flatten has no parameters."""
        op = LayerOperation(operation="flatten")
        input_shape = (64, 4, 4)
        params = calculate_parameters(op, input_shape)
        assert params == 0
    
    def test_maxpool_no_params(self):
        """Test that maxpool has no parameters."""
        op = LayerOperation(operation="maxpool", args=[2])
        input_shape = (32, 16, 16)
        params = calculate_parameters(op, input_shape)
        assert params == 0
    
    def test_batchnorm_params(self):
        """Test batchnorm parameter calculation."""
        op = LayerOperation(operation="batchnorm")
        input_shape = (64, 16, 16)
        params = calculate_parameters(op, input_shape)
        # 2 * channels (gamma and beta) = 2 * 64 = 128
        assert params == 128


class TestPrintModelSummary:
    """Tests for print_model_summary function."""
    
    def test_summary_runs(self):
        """Test that model summary runs without error."""
        source = """model TestNet:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(32, kernel=3).relu
          -> maxpool(2)
          -> flatten()
          -> dense(10)
"""
        program = parse_aurane(source)
        # Should not raise
        print_model_summary(program.models[0])
    
    def test_summary_no_forward(self):
        """Test summary for model without forward."""
        source = """model EmptyModel:
    input_shape = (3, 32, 32)
"""
        program = parse_aurane(source)
        # Should not raise even without forward block
        print_model_summary(program.models[0])


class TestVisualizeArchitecture:
    """Tests for visualize_model_architecture function."""
    
    def test_visualize_runs(self):
        """Test that visualization runs without error."""
        source = """model TestNet:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(32, kernel=3).relu
          -> flatten()
          -> dense(10)
"""
        program = parse_aurane(source)
        # Should not raise
        visualize_model_architecture(program.models[0])
    
    def test_visualize_simple_model(self):
        """Test visualizing simple model."""
        source = """model Net:
    def forward(x):
        x -> dense(64)
          -> dense(10)
"""
        program = parse_aurane(source)
        visualize_model_architecture(program.models[0])


class TestShapeInference:
    """Tests for shape inference across layers."""
    
    def test_cnn_shape_propagation(self):
        """Test shape propagation through CNN."""
        source = """model CNN:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(32, kernel=3, padding=1).relu
          -> maxpool(2)
          -> conv2d(64, kernel=3, padding=1).relu
          -> maxpool(2)
          -> flatten()
          -> dense(10)
"""
        program = parse_aurane(source)
        model = program.models[0]
        
        # Verify shape at each step
        assert model.forward_block is not None
        ops = model.forward_block.operations
        shape = model.config.get('input_shape', (3, 32, 32))
        
        # Conv2d 32: (3, 32, 32) -> (32, 32, 32)
        shape = calculate_output_shape(shape, ops[0])
        assert shape == (32, 32, 32)
        
        # Maxpool 2: (32, 32, 32) -> (32, 16, 16)
        shape = calculate_output_shape(shape, ops[1])
        assert shape == (32, 16, 16)
        
        # Conv2d 64: (32, 16, 16) -> (64, 16, 16)
        shape = calculate_output_shape(shape, ops[2])
        assert shape == (64, 16, 16)
        
        # Maxpool 2: (64, 16, 16) -> (64, 8, 8)
        shape = calculate_output_shape(shape, ops[3])
        assert shape == (64, 8, 8)
        
        # Flatten: (64, 8, 8) -> (4096,)
        shape = calculate_output_shape(shape, ops[4])
        assert shape == (4096,)
        
        # Dense 10: (4096,) -> (10,)
        shape = calculate_output_shape(shape, ops[5])
        assert shape == (10,)


class TestParameterCounting:
    """Tests for total parameter counting."""
    
    def test_total_params_cnn(self):
        """Test total parameters in CNN."""
        source = """model CNN:
    input_shape = (1, 28, 28)
    def forward(x):
        x -> conv2d(32, kernel=3).relu
          -> maxpool(2)
          -> flatten()
          -> dense(10)
"""
        program = parse_aurane(source)
        model = program.models[0]
        
        total = 0
        shape = model.config.get('input_shape', (1, 28, 28))
        
        assert model.forward_block is not None
        for op in model.forward_block.operations:
            params = calculate_parameters(op, shape)
            total += params
            shape = calculate_output_shape(shape, op)
        
        assert total > 0


class TestEdgeCases:
    """Tests for edge cases in visualization."""
    
    def test_model_no_input_shape(self):
        """Test model without explicit input shape."""
        source = """model Net:
    def forward(x):
        x -> dense(64)
          -> dense(10)
"""
        program = parse_aurane(source)
        model = program.models[0]
        
        # Should use default shape
        print_model_summary(model)
    
    def test_single_layer_model(self):
        """Test model with single layer."""
        source = """model SingleLayer:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        model = program.models[0]
        
        print_model_summary(model)
        visualize_model_architecture(model)
    
    def test_deep_model(self):
        """Test very deep model."""
        source = """model DeepNet:
    def forward(x):
        x -> dense(512)
          -> dense(256)
          -> dense(128)
          -> dense(64)
          -> dense(32)
          -> dense(16)
          -> dense(8)
          -> dense(4)
          -> dense(2)
          -> dense(1)
"""
        program = parse_aurane(source)
        model = program.models[0]
        
        print_model_summary(model)
        visualize_model_architecture(model)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
