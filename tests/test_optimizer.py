"""
Tests for the Aurane optimizer module.
"""

import pytest
from aurane.parser import parse_aurane
from aurane.optimizer import (
    ASTOptimizer,
    optimize_ast,
    OptimizationResult,
)
from aurane.ast import LayerOperation, ForwardBlock, ModelNode


class TestASTOptimizer:
    """Tests for ASTOptimizer class."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        source = """model Net:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        optimizer = ASTOptimizer(program)
        assert optimizer.program == program
    
    def test_optimize_returns_result(self):
        """Test that optimize returns OptimizationResult."""
        source = """model Net:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        optimizer = ASTOptimizer(program)
        result = optimizer.optimize()
        assert isinstance(result, OptimizationResult)
    
    def test_optimized_program_returned(self):
        """Test that optimized program is returned."""
        source = """model Net:
    def forward(x):
        x -> dense(64).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        optimizer = ASTOptimizer(program)
        result = optimizer.optimize()
        assert result.program is not None


class TestOptimizeAST:
    """Tests for optimize_ast helper function."""
    
    def test_optimize_ast_helper(self):
        """Test optimize_ast helper function."""
        source = """model Net:
    def forward(x):
        x -> dense(128).relu
          -> dense(64).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        result = optimize_ast(program)
        assert isinstance(result, OptimizationResult)
        assert result.program is not None
    
    def test_optimize_empty_program(self):
        """Test optimize_ast on empty program."""
        program = parse_aurane("")
        result = optimize_ast(program)
        assert result.program is not None


class TestOptimizationResult:
    """Tests for OptimizationResult class."""
    
    def test_result_has_program(self):
        """Test that result has program attribute."""
        source = """model Net:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = optimize_ast(program)
        assert hasattr(result, 'program')
    
    def test_result_has_passes_applied(self):
        """Test that result has passes_applied attribute."""
        source = """model Net:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = optimize_ast(program)
        assert hasattr(result, 'applied_optimizations')


class TestOptimizationLevels:
    """Tests for optimization levels."""
    
    def test_level_zero(self):
        """Test optimization level 0 (no optimization)."""
        source = """model Net:
    def forward(x):
        x -> dense(64).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        optimizer = ASTOptimizer(program)
        result = optimizer.optimize(level=0)
        assert result.program is not None
    
    def test_level_one(self):
        """Test optimization level 1 (basic)."""
        source = """model Net:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(32, kernel=3)
          -> batchnorm()
          -> relu()
          -> dense(10)
"""
        program = parse_aurane(source)
        optimizer = ASTOptimizer(program)
        result = optimizer.optimize(level=1)
        assert result.program is not None
    
    def test_level_two(self):
        """Test optimization level 2 (aggressive)."""
        source = """model Net:
    def forward(x):
        x -> dense(64).relu
          -> dense(32).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        optimizer = ASTOptimizer(program)
        result = optimizer.optimize(level=2)
        assert result.program is not None


class TestFusionOptimization:
    """Tests for fusion optimization."""
    
    def test_conv_bn_fusion(self):
        """Test Conv + BatchNorm fusion."""
        source = """model Net:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(32, kernel=3)
          -> batchnorm()
          -> relu()
          -> dense(10)
"""
        program = parse_aurane(source)
        result = optimize_ast(program)
        assert result.program is not None
    
    def test_dense_activation_fusion(self):
        """Test Dense + Activation fusion."""
        source = """model Net:
    def forward(x):
        x -> dense(64).relu
          -> dense(32).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        result = optimize_ast(program)
        assert result.program is not None


class TestOptimizationStats:
    """Tests for optimization statistics."""
    
    def test_stats_available(self):
        """Test that stats are available."""
        source = """model Net:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = optimize_ast(program)
        assert hasattr(result, 'stats')
    
    def test_applied_optimizations(self):
        """Test applied optimizations list."""
        source = """model Net:
    def forward(x):
        x -> dense(64).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        result = optimize_ast(program)
        assert isinstance(result.applied_optimizations, list)


class TestComplexModels:
    """Tests for optimizing complex models."""
    
    def test_optimize_cnn(self):
        """Test optimizing CNN model."""
        source = """model CNN:
    input_shape = (3, 224, 224)
    def forward(x):
        x -> conv2d(64, kernel=3, padding=1).relu
          -> conv2d(64, kernel=3, padding=1).relu
          -> maxpool(2)
          -> conv2d(128, kernel=3, padding=1).relu
          -> conv2d(128, kernel=3, padding=1).relu
          -> maxpool(2)
          -> conv2d(256, kernel=3, padding=1).relu
          -> conv2d(256, kernel=3, padding=1).relu
          -> maxpool(2)
          -> flatten()
          -> dense(4096).relu
          -> dropout(0.5)
          -> dense(4096).relu
          -> dropout(0.5)
          -> dense(1000)
"""
        program = parse_aurane(source)
        result = optimize_ast(program)
        assert result.program is not None
        assert result.program.models[0].name == "CNN"
    
    def test_optimize_multiple_models(self):
        """Test optimizing multiple models."""
        source = """model Encoder:
    def forward(x):
        x -> dense(256).relu
          -> dense(128).relu
          -> dense(64)

model Decoder:
    def forward(x):
        x -> dense(128).relu
          -> dense(256).relu
          -> dense(784).sigmoid
"""
        program = parse_aurane(source)
        result = optimize_ast(program)
        assert result.program is not None
        assert len(result.program.models) == 2


class TestOptimizationPreservation:
    """Tests that optimization preserves semantics."""
    
    def test_model_name_preserved(self):
        """Test that model name is preserved after optimization."""
        source = """model MyNetwork:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = optimize_ast(program)
        assert result.program.models[0].name == "MyNetwork"
    
    def test_layer_operations_preserved(self):
        """Test that layer operations are preserved."""
        source = """model Net:
    def forward(x):
        x -> dense(64).relu
          -> dense(32).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        result = optimize_ast(program)
        
        # Should still have operations
        forward_block = result.program.models[0].forward_block
        assert forward_block is not None
        assert len(forward_block.operations) > 0
    
    def test_config_preserved(self):
        """Test that model config is preserved."""
        source = """model ConfiguredNet:
    input_shape = (3, 32, 32)
    num_classes = 10
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = optimize_ast(program)
        
        config = result.program.models[0].config
        assert 'input_shape' in config or config.get('input_shape') is not None


class TestEdgeCases:
    """Tests for edge cases in optimization."""
    
    def test_empty_model(self):
        """Test optimizing model without forward block."""
        source = """model EmptyModel:
    input_shape = (3, 32, 32)
"""
        program = parse_aurane(source)
        result = optimize_ast(program)
        assert result.program is not None
    
    def test_single_layer_model(self):
        """Test optimizing single layer model."""
        source = """model SingleLayer:
    def forward(x):
        x -> dense(10)
"""
        program = parse_aurane(source)
        result = optimize_ast(program)
        assert result.program is not None
    
    def test_no_activation_model(self):
        """Test optimizing model without activations."""
        source = """model NoActivation:
    def forward(x):
        x -> dense(64)
          -> dense(10)
"""
        program = parse_aurane(source)
        result = optimize_ast(program)
        assert result.program is not None
    
    def test_optimizer_idempotent(self):
        """Test that running optimizer twice gives same result."""
        source = """model Net:
    def forward(x):
        x -> dense(64).relu
          -> dense(10)
"""
        program = parse_aurane(source)
        result1 = optimize_ast(program)
        result2 = optimize_ast(result1.program)
        
        # Should have same structure
        assert len(result1.program.models) == len(result2.program.models)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
