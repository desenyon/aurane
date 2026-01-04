"""
Tests for the Aurane AST module.
"""

import pytest
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


class TestAuraneProgram:
    """Tests for AuraneProgram class."""
    
    def test_program_creation(self):
        """Test creating an AuraneProgram."""
        program = AuraneProgram()
        assert program.uses == []
        assert program.experiments == []
        assert program.datasets == []
        assert program.models == []
        assert program.trains == []
    
    def test_program_with_uses(self):
        """Test program with use statements."""
        program = AuraneProgram()
        program.uses.append(UseStatement("torch"))
        assert len(program.uses) == 1
        assert program.uses[0].module == "torch"
    
    def test_program_with_models(self):
        """Test program with models."""
        model = ModelNode(name="Net")
        program = AuraneProgram()
        program.models.append(model)
        assert len(program.models) == 1
        assert program.models[0].name == "Net"


class TestUseStatement:
    """Tests for UseStatement class."""
    
    def test_use_creation(self):
        """Test creating a UseStatement."""
        use = UseStatement("torch")
        assert use.module == "torch"
        assert use.alias is None
    
    def test_use_with_alias(self):
        """Test UseStatement with alias."""
        use = UseStatement("numpy", "np")
        assert use.module == "numpy"
        assert use.alias == "np"
    
    def test_use_repr(self):
        """Test UseStatement string representation."""
        use = UseStatement("torch")
        repr_str = repr(use)
        assert "torch" in repr_str


class TestExperimentNode:
    """Tests for ExperimentNode class."""
    
    def test_experiment_creation(self):
        """Test creating an ExperimentNode."""
        exp = ExperimentNode(
            name="MyExperiment",
            config={"seed": 42, "device": "auto"}
        )
        assert exp.name == "MyExperiment"
        assert exp.config["seed"] == 42
    
    def test_experiment_empty_config(self):
        """Test ExperimentNode with empty config."""
        exp = ExperimentNode(name="Empty")
        assert exp.name == "Empty"
        assert len(exp.config) == 0


class TestDatasetNode:
    """Tests for DatasetNode class."""
    
    def test_dataset_creation(self):
        """Test creating a DatasetNode."""
        dataset = DatasetNode(
            name="mnist",
            source="torchvision.datasets.MNIST",
            config={"batch": 32}
        )
        assert dataset.name == "mnist"
        assert dataset.source == "torchvision.datasets.MNIST"
        assert dataset.config["batch"] == 32
    
    def test_dataset_without_source(self):
        """Test DatasetNode without source."""
        dataset = DatasetNode(name="custom")
        dataset.config["batch"] = 64
        assert dataset.source is None


class TestModelNode:
    """Tests for ModelNode class."""
    
    def test_model_creation(self):
        """Test creating a ModelNode."""
        model = ModelNode(
            name="TestNet",
            config={"input_shape": (3, 32, 32)}
        )
        assert model.name == "TestNet"
        assert model.config["input_shape"] == (3, 32, 32)
    
    def test_model_with_forward_block(self):
        """Test ModelNode with forward block."""
        ops = [LayerOperation(operation="dense", args=[10])]
        forward = ForwardBlock(parameter="x", operations=ops)
        model = ModelNode(
            name="Net",
            forward_block=forward
        )
        assert model.forward_block is not None
        assert len(model.forward_block.operations) == 1


class TestTrainNode:
    """Tests for TrainNode class."""
    
    def test_train_creation(self):
        """Test creating a TrainNode."""
        train = TrainNode(
            model_name="Net",
            dataset_name="data",
            config={"epochs": 10, "lr": 0.001}
        )
        assert train.model_name == "Net"
        assert train.dataset_name == "data"
        assert train.config["epochs"] == 10
    
    def test_train_optimizer_config(self):
        """Test TrainNode with optimizer config."""
        train = TrainNode(
            model_name="Net",
            dataset_name="data",
            config={
                "optimizer": "adam",
                "lr": 0.001,
                "weight_decay": 0.0001
            }
        )
        assert train.config["optimizer"] == "adam"


class TestForwardBlock:
    """Tests for ForwardBlock class."""
    
    def test_forward_block_creation(self):
        """Test creating a ForwardBlock."""
        ops = [
            LayerOperation(operation="dense", args=[64], activation="relu"),
            LayerOperation(operation="dense", args=[10])
        ]
        forward = ForwardBlock(parameter="x", operations=ops)
        assert forward.parameter == "x"
        assert len(forward.operations) == 2
    
    def test_forward_block_empty(self):
        """Test ForwardBlock with no operations."""
        forward = ForwardBlock()
        assert len(forward.operations) == 0


class TestLayerOperation:
    """Tests for LayerOperation class."""
    
    def test_layer_operation_creation(self):
        """Test creating a LayerOperation."""
        op = LayerOperation(
            operation="conv2d",
            args=[32],
            kwargs={"kernel": 3, "padding": 1},
            activation="relu"
        )
        assert op.operation == "conv2d"
        assert op.args == [32]
        assert op.kwargs["kernel"] == 3
        assert op.activation == "relu"
    
    def test_layer_operation_no_activation(self):
        """Test LayerOperation without activation."""
        op = LayerOperation(operation="flatten")
        assert op.activation is None
    
    def test_layer_operation_dense(self):
        """Test dense LayerOperation."""
        op = LayerOperation(operation="dense", args=[128], activation="relu")
        assert op.operation == "dense"
        assert op.args[0] == 128
    
    def test_layer_operation_dropout(self):
        """Test dropout LayerOperation."""
        op = LayerOperation(operation="dropout", args=[0.5])
        assert op.operation == "dropout"
        assert op.args[0] == 0.5
    
    def test_layer_operation_repr(self):
        """Test LayerOperation string representation."""
        op = LayerOperation(operation="dense", args=[64], activation="relu")
        repr_str = repr(op)
        assert "dense" in repr_str or "LayerOperation" in repr_str


class TestASTEquality:
    """Tests for AST node equality."""
    
    def test_use_equality(self):
        """Test UseStatement equality."""
        use1 = UseStatement("torch")
        use2 = UseStatement("torch")
        assert use1.module == use2.module
    
    def test_layer_operation_equality(self):
        """Test LayerOperation equality."""
        op1 = LayerOperation(operation="dense", args=[64], activation="relu")
        op2 = LayerOperation(operation="dense", args=[64], activation="relu")
        assert op1.operation == op2.operation
        assert op1.args == op2.args
        assert op1.activation == op2.activation


class TestASTValidation:
    """Tests for AST validation."""
    
    def test_valid_layer_operation(self):
        """Test creating valid layer operations."""
        # All these should work without errors
        ops = [
            LayerOperation(operation="conv2d", args=[32], kwargs={"kernel": 3}, activation="relu"),
            LayerOperation(operation="dense", args=[128], activation="gelu"),
            LayerOperation(operation="dropout", args=[0.5]),
            LayerOperation(operation="flatten"),
            LayerOperation(operation="maxpool", args=[2]),
            LayerOperation(operation="batchnorm"),
        ]
        assert len(ops) == 6


class TestComplexAST:
    """Tests for complex AST structures."""
    
    def test_complete_program(self):
        """Test creating a complete program AST."""
        # Build a complete program
        program = AuraneProgram()
        
        program.uses.append(UseStatement("torch"))
        program.uses.append(UseStatement("numpy", "np"))
        
        program.experiments.append(ExperimentNode("Exp1", {"seed": 42}))
        
        program.datasets.append(
            DatasetNode("mnist", "torchvision.datasets.MNIST", {"batch": 32})
        )
        
        forward_ops = [
            LayerOperation(operation="conv2d", args=[32], kwargs={"kernel": 3}, activation="relu"),
            LayerOperation(operation="maxpool", args=[2]),
            LayerOperation(operation="flatten"),
            LayerOperation(operation="dense", args=[10])
        ]
        forward = ForwardBlock(parameter="x", operations=forward_ops)
        
        model = ModelNode(name="Net", config={"input_shape": (1, 28, 28)}, forward_block=forward)
        program.models.append(model)
        
        program.trains.append(
            TrainNode("Net", "mnist", {"epochs": 10, "lr": 0.001})
        )
        
        assert len(program.uses) == 2
        assert len(program.experiments) == 1
        assert len(program.datasets) == 1
        assert len(program.models) == 1
        assert len(program.trains) == 1
        assert program.models[0].forward_block is not None
        assert len(program.models[0].forward_block.operations) == 4


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
