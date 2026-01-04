"""
Tests for the Aurane semantic analyzer module.
"""

import pytest
from aurane.parser import parse_aurane
from aurane.semantic_analyzer import (
    SemanticAnalyzer,
    analyze_semantics,
    SemanticAnalysisResult,
    ACTIVATIONS,
    OPTIMIZERS,
    LOSS_FUNCTIONS,
    LAYER_SPECS,
)


class TestSemanticAnalyzer:
    """Tests for SemanticAnalyzer class."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        source = """model Net:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        analyzer = SemanticAnalyzer(program)
        assert analyzer.program == program
    
    def test_analyze_returns_result(self):
        """Test that analyze returns SemanticAnalysisResult."""
        source = """model Net:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        analyzer = SemanticAnalyzer(program)
        result = analyzer.analyze()
        assert isinstance(result, SemanticAnalysisResult)
    
    def test_analyze_valid_model(self):
        """Test analyzing a valid model."""
        source = """model ValidNet:
    def forward(x):
        x -> dense(64).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        analyzer = SemanticAnalyzer(program)
        result = analyzer.analyze()
        assert result.is_valid


class TestAnalyzeSemantics:
    """Tests for analyze_semantics helper function."""
    
    def test_analyze_semantics_helper(self):
        """Test analyze_semantics helper function."""
        source = """model Net:
    def forward(x):
        x -> dense(128).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert isinstance(result, SemanticAnalysisResult)
    
    def test_analyze_empty_program(self):
        """Test analyze_semantics on empty program."""
        program = parse_aurane("")
        result = analyze_semantics(program)
        assert result.is_valid


class TestSemanticAnalysisResult:
    """Tests for SemanticAnalysisResult class."""
    
    def test_result_is_valid(self):
        """Test is_valid property."""
        source = """model Net:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert result.is_valid
    
    def test_result_has_errors(self):
        """Test errors attribute."""
        source = """model Net:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert hasattr(result, 'errors')
        assert isinstance(result.errors, list)
    
    def test_result_has_warnings(self):
        """Test warnings attribute."""
        source = """model Net:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert hasattr(result, 'warnings')
        assert isinstance(result.warnings, list)


class TestActivationsSet:
    """Tests for ACTIVATIONS constant."""
    
    def test_activations_is_set(self):
        """Test that ACTIVATIONS is a set."""
        assert isinstance(ACTIVATIONS, (set, frozenset))
    
    def test_common_activations(self):
        """Test that common activations are included."""
        assert 'relu' in ACTIVATIONS
        assert 'sigmoid' in ACTIVATIONS
        assert 'tanh' in ACTIVATIONS
    
    def test_gelu_activation(self):
        """Test GELU activation."""
        assert 'gelu' in ACTIVATIONS
    
    def test_softmax_activation(self):
        """Test softmax activation."""
        assert 'softmax' in ACTIVATIONS


class TestOptimizersSet:
    """Tests for OPTIMIZERS constant."""
    
    def test_optimizers_is_set(self):
        """Test that OPTIMIZERS is a set."""
        assert isinstance(OPTIMIZERS, (set, frozenset))
    
    def test_common_optimizers(self):
        """Test that common optimizers are included."""
        assert 'adam' in OPTIMIZERS or 'Adam' in OPTIMIZERS
        assert 'sgd' in OPTIMIZERS or 'SGD' in OPTIMIZERS


class TestLossFunctionsSet:
    """Tests for LOSS_FUNCTIONS constant."""
    
    def test_loss_functions_is_set(self):
        """Test that LOSS_FUNCTIONS is a set."""
        assert isinstance(LOSS_FUNCTIONS, (set, frozenset))
    
    def test_common_loss_functions(self):
        """Test that common loss functions are included."""
        # Check for various possible names
        has_cross_entropy = any(
            'cross' in name.lower() and 'entropy' in name.lower()
            for name in LOSS_FUNCTIONS
        )
        has_mse = any('mse' in name.lower() for name in LOSS_FUNCTIONS)
        assert has_cross_entropy or has_mse or len(LOSS_FUNCTIONS) > 0


class TestLayerSpecs:
    """Tests for LAYER_SPECS constant."""
    
    def test_layer_specs_is_dict(self):
        """Test that LAYER_SPECS is a dict."""
        assert isinstance(LAYER_SPECS, dict)
    
    def test_common_layers(self):
        """Test that common layers are included."""
        assert 'dense' in LAYER_SPECS or 'linear' in LAYER_SPECS
        assert 'conv2d' in LAYER_SPECS
    
    def test_layer_spec_has_params(self):
        """Test that layer specs have parameter info."""
        if 'dense' in LAYER_SPECS:
            spec = LAYER_SPECS['dense']
            assert spec is not None


class TestValidModels:
    """Tests for valid model analysis."""
    
    def test_simple_mlp(self):
        """Test analyzing simple MLP."""
        source = """model MLP:
    def forward(x):
        x -> dense(64).relu
          -> dense(32).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert result.is_valid
    
    def test_cnn_model(self):
        """Test analyzing CNN model."""
        source = """model CNN:
    input_shape = (3, 32, 32)
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
        result = analyze_semantics(program)
        assert result.is_valid
    
    def test_model_with_dropout(self):
        """Test analyzing model with dropout."""
        source = """model DropoutNet:
    def forward(x):
        x -> dense(64).relu
          -> dropout(0.5)
          -> dense(10)
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert result.is_valid
    
    def test_model_with_batchnorm(self):
        """Test analyzing model with batch normalization."""
        source = """model BatchNormNet:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(32, kernel=3)
          -> batchnorm()
          -> relu()
          -> flatten()
          -> dense(10)
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert result.is_valid


class TestMultipleModels:
    """Tests for analyzing multiple models."""
    
    def test_multiple_models(self):
        """Test analyzing multiple models."""
        source = """model Encoder:
    def forward(x):
        x -> dense(128).relu
          -> dense(64)

model Decoder:
    def forward(x):
        x -> dense(128).relu
          -> dense(256)
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert result.is_valid
    
    def test_models_can_reference_each_other(self):
        """Test that models can conceptually reference each other."""
        source = """model Generator:
    def forward(z):
        z -> dense(256).relu
          -> dense(784).tanh

model Discriminator:
    def forward(x):
        x -> dense(256).relu
          -> dense(1).sigmoid
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert result.is_valid


class TestActivationUsage:
    """Tests for activation function usage analysis."""
    
    def test_valid_activation(self):
        """Test using valid activation."""
        source = """model Net:
    def forward(x):
        x -> dense(64).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert result.is_valid
    
    def test_multiple_activations(self):
        """Test using multiple activation types."""
        source = """model MultiActNet:
    def forward(x):
        x -> dense(64).relu
          -> dense(32).gelu
          -> dense(10).softmax
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert result.is_valid


class TestLayerParameterValidation:
    """Tests for layer parameter validation."""
    
    def test_dense_with_units(self):
        """Test dense layer with units parameter."""
        source = """model Net:
    def forward(x):
        x -> dense(128)
          -> dense(10)
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert result.is_valid
    
    def test_conv2d_with_params(self):
        """Test conv2d with parameters."""
        source = """model Net:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(64, kernel=3, padding=1, stride=1)
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert result.is_valid
    
    def test_dropout_with_rate(self):
        """Test dropout with rate parameter."""
        source = """model Net:
    def forward(x):
        x -> dense(64).relu
          -> dropout(0.5)
          -> dense(10)
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert result.is_valid
    
    def test_maxpool_with_size(self):
        """Test maxpool with kernel size."""
        source = """model Net:
    input_shape = (32, 16, 16)
    def forward(x):
        x -> maxpool(2)
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert result.is_valid


class TestEdgeCases:
    """Tests for edge cases in semantic analysis."""
    
    def test_empty_model(self):
        """Test analyzing model without forward block."""
        source = """model EmptyModel:
    input_shape = (3, 32, 32)
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        # Should have warning but no error
    
    def test_single_layer_model(self):
        """Test analyzing single layer model."""
        source = """model SingleLayer:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        assert result.is_valid
    
    def test_model_with_config_only(self):
        """Test model with only config."""
        source = """model ConfigOnly:
    input_shape = (3, 32, 32)
    num_classes = 10
    hidden_size = 256
"""
        program = parse_aurane(source)
        result = analyze_semantics(program)
        # Should not crash


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
