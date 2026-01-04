"""
Tests for the Aurane profiler module.
"""

import pytest
from aurane.parser import parse_aurane
from aurane.profiler import (
    ModelProfiler,
    profile_model,
    ModelProfile,
    LayerProfile,
)


class TestModelProfiler:
    """Tests for ModelProfiler class."""
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        source = """model Net:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        profiler = ModelProfiler(program.models[0])
        assert profiler.model is not None
    
    def test_profile_returns_result(self):
        """Test that profile returns ModelProfile."""
        source = """model Net:
    input_shape = (784,)
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        profiler = ModelProfiler(program.models[0])
        result = profiler.profile_model()
        assert isinstance(result, ModelProfile)


class TestProfileModel:
    """Tests for profile_model helper function."""
    
    def test_profile_model_helper(self):
        """Test profile_model helper function."""
        source = """model Net:
    input_shape = (784,)
    def forward(x):
        x -> dense(256).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        assert isinstance(result, ModelProfile)
    
    def test_profile_with_input_shape(self):
        """Test profiling with explicit input shape."""
        source = """model Net:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(32, kernel=3).relu
          -> flatten()
          -> dense(10)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        assert result is not None


class TestModelProfile:
    """Tests for ModelProfile class."""
    
    def test_profile_has_model_name(self):
        """Test that profile has model name."""
        source = """model MyNetwork:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        assert result.model_name == "MyNetwork"
    
    def test_profile_has_total_params(self):
        """Test that profile has total parameters."""
        source = """model Net:
    input_shape = (784,)
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        assert hasattr(result, 'total_params')
        assert result.total_params >= 0
    
    def test_profile_has_total_flops(self):
        """Test that profile has total FLOPs."""
        source = """model Net:
    input_shape = (784,)
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        assert hasattr(result, 'total_flops')
        assert result.total_flops >= 0
    
    def test_profile_has_layers(self):
        """Test that profile has layer profiles."""
        source = """model Net:
    input_shape = (784,)
    def forward(x):
        x -> dense(64).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        assert hasattr(result, 'layers')


class TestLayerProfile:
    """Tests for LayerProfile class."""
    
    def test_layer_profile_has_name(self):
        """Test that layer profile has name."""
        source = """model Net:
    input_shape = (784,)
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        if result.layers:
            layer = result.layers[0]
            assert hasattr(layer, 'layer_name') or hasattr(layer, 'name')
    
    def test_layer_profile_has_params(self):
        """Test that layer profile has params."""
        source = """model Net:
    input_shape = (784,)
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        if result.layers:
            layer = result.layers[0]
            assert hasattr(layer, 'params') or hasattr(layer, 'parameters')


class TestEstimateFlops:
    """Tests for estimate_flops function."""
    
    def test_dense_flops(self):
        """Test FLOPs estimation for dense layer."""
        source = """model Net:
    input_shape = (784,)
    def forward(x):
        x -> dense(256)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        # Dense layer: 2 * in * out FLOPs (multiply-add)
        # 784 * 256 * 2 = 401408
        assert result.total_flops > 0
    
    def test_conv2d_flops(self):
        """Test FLOPs estimation for conv2d layer."""
        source = """model Net:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(32, kernel=3)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        # Conv2d has significant FLOPs
        assert result.total_flops > 0


class TestEstimateMemory:
    """Tests for estimate_memory function."""
    
    def test_memory_estimation(self):
        """Test memory estimation."""
        source = """model Net:
    input_shape = (3, 224, 224)
    def forward(x):
        x -> conv2d(64, kernel=3, padding=1).relu
          -> conv2d(64, kernel=3, padding=1).relu
          -> maxpool(2)
          -> flatten()
          -> dense(1000)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        # Should have memory estimate
        assert hasattr(result, 'memory_bytes') or result.total_params > 0


class TestSimpleModels:
    """Tests for profiling simple models."""
    
    def test_single_dense_layer(self):
        """Test profiling single dense layer."""
        source = """model SingleDense:
    input_shape = (10,)
    def forward(x):
        x -> dense(5)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        
        # params = 10 * 5 + 5 = 55
        assert result.total_params == 55
    
    def test_two_dense_layers(self):
        """Test profiling two dense layers."""
        source = """model TwoDense:
    input_shape = (10,)
    def forward(x):
        x -> dense(20)
          -> dense(5)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        
        # layer 1: 10 * 20 + 20 = 220
        # layer 2: 20 * 5 + 5 = 105
        # total: 325
        assert result.total_params == 325


class TestComplexModels:
    """Tests for profiling complex models."""
    
    def test_cnn_profile(self):
        """Test profiling CNN model."""
        source = """model CNN:
    input_shape = (1, 28, 28)
    def forward(x):
        x -> conv2d(32, kernel=3).relu
          -> maxpool(2)
          -> conv2d(64, kernel=3).relu
          -> maxpool(2)
          -> flatten()
          -> dense(128).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        
        assert result.model_name == "CNN"
        assert result.total_params > 0
        assert result.total_flops > 0
    
    def test_deep_network_profile(self):
        """Test profiling deep network."""
        source = """model DeepNet:
    input_shape = (100,)
    def forward(x):
        x -> dense(256).relu
          -> dense(256).relu
          -> dense(256).relu
          -> dense(256).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        
        assert result.total_params > 0
        assert len(result.layers) >= 5


class TestParameterCounting:
    """Tests for accurate parameter counting."""
    
    def test_conv2d_params(self):
        """Test conv2d parameter counting."""
        source = """model ConvNet:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(32, kernel=3)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        
        # params = in_channels * out_channels * kernel * kernel + out_channels
        # 3 * 32 * 3 * 3 + 32 = 864 + 32 = 896
        assert result.total_params == 896
    
    def test_no_params_layers(self):
        """Test layers with no parameters."""
        source = """model NoParamsNet:
    input_shape = (64, 16, 16)
    def forward(x):
        x -> maxpool(2)
          -> dropout(0.5)
          -> flatten()
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        
        # These layers have no trainable params
        assert result.total_params == 0


class TestFLOPsCounting:
    """Tests for accurate FLOPs counting."""
    
    def test_dense_flops_calculation(self):
        """Test dense FLOPs calculation."""
        source = """model DenseFlops:
    input_shape = (100,)
    def forward(x):
        x -> dense(50)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        
        # FLOPs = 2 * in * out = 2 * 100 * 50 = 10000
        assert result.total_flops == 10000
    
    def test_conv2d_flops_calculation(self):
        """Test conv2d FLOPs calculation."""
        source = """model ConvFlops:
    input_shape = (3, 8, 8)
    def forward(x):
        x -> conv2d(16, kernel=3, padding=1)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        
        # FLOPs significant for conv
        assert result.total_flops > 0


class TestEdgeCases:
    """Tests for edge cases in profiling."""
    
    def test_model_without_forward(self):
        """Test profiling model without forward block."""
        source = """model NoForward:
    input_shape = (3, 32, 32)
    num_classes = 10
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        
        assert result.total_params == 0
        assert result.total_flops == 0
    
    def test_model_without_input_shape(self):
        """Test profiling model without input shape."""
        source = """model NoInputShape:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        
        # Should use default input shape
        assert result is not None
    
    def test_empty_forward_block(self):
        """Test model with empty forward block."""
        source = """model EmptyForward:
    input_shape = (10,)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        
        assert result.total_params == 0


class TestProfileFormatting:
    """Tests for profile output formatting."""
    
    def test_profile_string_representation(self):
        """Test profile string representation."""
        source = """model Net:
    input_shape = (784,)
    def forward(x):
        x -> dense(256).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        
        # Should have string representation
        str_repr = str(result)
        assert 'Net' in str_repr or result.model_name in str_repr
    
    def test_profile_summary(self):
        """Test profile summary method if available."""
        source = """model Net:
    input_shape = (784,)
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = profile_model(program.models[0])
        
        if hasattr(result, 'summary'):
            summary = result.summary()
            # summary() returns a dict with model information
            assert isinstance(summary, dict)
            assert 'model' in summary


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
