"""
PyTorch code generator for Aurane DSL.

This module converts Aurane AST nodes into idiomatic PyTorch Python code.
"""

from typing import List, Dict, Any, Optional, Tuple
import re

from .ast import (
    AuraneProgram,
    ExperimentNode,
    DatasetNode,
    ModelNode,
    TrainNode,
    LayerOperation,
    ForwardBlock,
)


class TorchCodeGenerator:
    """Generates PyTorch code from Aurane AST."""

    def __init__(self, program: AuraneProgram):
        self.program = program
        self.indent_level = 0
        self.layer_counter = {}  # Track layer counts for naming
        self.layer_map = {}  # Map operation index to layer variable name

    def generate(self) -> str:
        """Generate complete Python code from the AST."""
        sections = []

        # Imports
        sections.append(self._generate_imports())

        # Experiment setup
        if self.program.experiments:
            sections.append(self._generate_experiment_setup(self.program.experiments[0]))

        # Dataset loaders
        for dataset in self.program.datasets:
            sections.append(self._generate_dataset(dataset))

        # Model definitions
        for model in self.program.models:
            sections.append(self._generate_model(model))

        # Training functions
        for train in self.program.trains:
            sections.append(self._generate_training(train))

        # Main execution
        if self.program.trains:
            sections.append(self._generate_main(self.program.trains[0]))

        return "\n\n".join(sections)

    def _generate_imports(self) -> str:
        """Generate import statements."""
        imports = [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "import torch.optim as optim",
            "from torch.utils.data import DataLoader",
            "import torchvision",
            "import torchvision.transforms as transforms",
        ]

        return "\n".join(imports)

    def _generate_experiment_setup(self, experiment: ExperimentNode) -> str:
        """Generate experiment configuration setup."""
        lines = [
            f"# Experiment: {experiment.name}",
        ]

        # Set seed
        if "seed" in experiment.config:
            seed = experiment.config["seed"]
            lines.extend(
                [
                    f"torch.manual_seed({seed})",
                    f"if torch.cuda.is_available():",
                    f"    torch.cuda.manual_seed({seed})",
                ]
            )

        # Set device
        device = experiment.config.get("device", "auto")
        if device == "auto":
            lines.append('device = torch.device("cuda" if torch.cuda.is_available() else "cpu")')
        else:
            lines.append(f'device = torch.device("{device}")')

        return "\n".join(lines)

    def _generate_dataset(self, dataset: DatasetNode) -> str:
        """Generate dataset and dataloader code."""
        lines = [f"# Dataset: {dataset.name}"]

        # Parse source
        if dataset.source:
            # Extract class name from source like "torchvision.datasets.MNIST"
            class_parts = dataset.source.split(".")
            class_name = class_parts[-1]

            # Build dataset instantiation
            args = []

            # Root directory
            if "root" in dataset.config:
                args.append(f"root={self._format_value(dataset.config['root'])}")

            # Train flag
            if "train" in dataset.config:
                args.append(f"train={dataset.config['train']}")

            # Transform
            args.append("transform=transforms.ToTensor()")

            # Download
            args.append("download=True")

            dataset_var = f"{dataset.name}_dataset"
            lines.append(f"{dataset_var} = torchvision.datasets.{class_name}({', '.join(args)})")

            # Create DataLoader
            batch_size = dataset.config.get("batch", 32)
            is_train = dataset.config.get("train", True)
            lines.append(
                f"{dataset.name} = DataLoader({dataset_var}, "
                f"batch_size={batch_size}, shuffle={is_train})"
            )

        return "\n".join(lines)

    def _generate_model(self, model: ModelNode) -> str:
        """Generate PyTorch model class."""
        lines = [
            f"# Model: {model.name}",
            f"class {model.name}(nn.Module):",
        ]

        # __init__ method
        init_lines = ["    def __init__(self):"]
        init_lines.append("        super().__init__()")

        # Parse forward block to determine layers
        if model.forward_block:
            # Infer input channels from input_shape if available
            input_shape = model.config.get("input_shape", (1, 28, 28))
            layer_defs = self._generate_layer_definitions(model.forward_block, input_shape)
            init_lines.extend([f"        {line}" for line in layer_defs])

        lines.extend(init_lines)
        lines.append("")

        # forward method
        if model.forward_block:
            forward_lines = self._generate_forward_method(model.forward_block)
            lines.extend([f"    {line}" if line else "" for line in forward_lines])

        return "\n".join(lines)

    def _generate_layer_definitions(
        self, forward_block: ForwardBlock, input_shape: tuple
    ) -> List[str]:
        """Generate layer definitions for __init__ method."""
        layers = []
        self.layer_counter = {}
        self.layer_map = {}  # Map operation index to layer variable name

        # Track shape through the network for proper layer instantiation
        # input_shape is (channels, height, width) for 2D or (features,) for 1D
        current_channels = input_shape[0] if len(input_shape) >= 1 else 1
        current_shape = input_shape

        for idx, op in enumerate(forward_block.operations):
            layer_def, current_channels, current_shape = self._operation_to_layer_def_with_shape(
                op, idx, current_channels, current_shape
            )
            if layer_def:
                layers.append(layer_def)

        return layers

    def _operation_to_layer_def_with_shape(
        self, op: LayerOperation, idx: int, in_channels: int, shape: tuple
    ) -> Tuple[Optional[str], int, tuple]:
        """Convert an operation to a layer definition, tracking shapes."""
        op_name = op.operation.lower()

        # Get unique layer name
        layer_var = self._get_layer_var_name(op_name)
        self.layer_map[idx] = layer_var

        if op_name == "conv2d":
            # conv2d(out_channels, kernel=3)
            out_channels = op.args[0] if op.args else 32
            kernel = op.kwargs.get("kernel", 3)

            layer_def = f"self.{layer_var} = nn.Conv2d({in_channels}, {out_channels}, {kernel})"

            # Update shape: conv reduces spatial dimensions by (kernel-1)
            if len(shape) == 3:
                c, h, w = shape
                new_h = h - kernel + 1
                new_w = w - kernel + 1
                new_shape = (out_channels, new_h, new_w)
            else:
                new_shape = (out_channels, 26, 26)  # Default assumption

            return layer_def, out_channels, new_shape

        elif op_name == "maxpool":
            # maxpool doesn't create a layer, but updates shape
            kernel = op.args[0] if op.args else 2
            if len(shape) == 3:
                c, h, w = shape
                new_shape = (c, h // kernel, w // kernel)
            else:
                new_shape = shape
            return None, in_channels, new_shape

        elif op_name == "flatten":
            # flatten converts (C, H, W) to (C*H*W,)
            if len(shape) == 3:
                c, h, w = shape
                flat_size = c * h * w
                new_shape = (flat_size,)
            else:
                flat_size = shape[0] if shape else 128
                new_shape = (flat_size,)
            return None, flat_size, new_shape

        elif op_name in ("dense", "linear"):
            # dense(out_features)
            out_features = op.args[0] if op.args else 128
            in_features = shape[0] if shape else 128

            layer_def = f"self.{layer_var} = nn.Linear({in_features}, {out_features})"
            new_shape = (out_features,)

            return layer_def, out_features, new_shape

        elif op_name == "dropout":
            p = op.args[0] if op.args else 0.5
            layer_def = f"self.{layer_var} = nn.Dropout({p})"
            return layer_def, in_channels, shape

        # For other operations, pass through
        return None, in_channels, shape

    def _generate_forward_method(self, forward_block: ForwardBlock) -> List[str]:
        """Generate the forward method."""
        lines = [
            f"def forward(self, {forward_block.parameter}):",
        ]

        x = forward_block.parameter

        for idx, op in enumerate(forward_block.operations):
            op_code = self._operation_to_forward_code(op, x, idx)
            lines.append(f"    {x} = {op_code}")

        lines.append(f"    return {x}")

        return lines

    def _operation_to_forward_code(self, op: LayerOperation, var: str, idx: int) -> str:
        """Convert an operation to forward pass code."""
        op_name = op.operation.lower()

        if op_name == "conv2d":
            layer_var = self.layer_map.get(idx, self._get_layer_var_name(op_name))
            code = f"self.{layer_var}({var})"
            if op.activation:
                code = self._apply_activation(code, op.activation)
            return code

        elif op_name == "dense" or op_name == "linear":
            layer_var = self.layer_map.get(idx, self._get_layer_var_name(op_name))
            code = f"self.{layer_var}({var})"
            if op.activation:
                code = self._apply_activation(code, op.activation)
            return code

        elif op_name == "dropout":
            layer_var = self.layer_map.get(idx, self._get_layer_var_name(op_name))
            return f"self.{layer_var}({var})"

        elif op_name == "maxpool":
            kernel = op.args[0] if op.args else 2
            return f"F.max_pool2d({var}, {kernel})"

        elif op_name == "avgpool":
            kernel = op.args[0] if op.args else 2
            return f"F.avg_pool2d({var}, {kernel})"

        elif op_name == "flatten":
            return f"torch.flatten({var}, 1)"

        elif op_name == "reshape":
            shape = op.args if op.args else [-1]
            return f"{var}.view{tuple(shape)}"

        elif op_name == "relu":
            return f"F.relu({var})"

        elif op_name == "leaky_relu":
            negative_slope = op.args[0] if op.args else 0.01
            return f"F.leaky_relu({var}, {negative_slope})"

        elif op_name == "gelu":
            return f"F.gelu({var})"

        elif op_name == "tanh":
            return f"torch.tanh({var})"

        elif op_name == "sigmoid":
            return f"torch.sigmoid({var})"

        elif op_name == "softmax":
            dim = op.args[0] if op.args else -1
            return f"F.softmax({var}, dim={dim})"

        elif op_name == "batch_norm" or op_name == "batchnorm":
            return f"self.{self.layer_map.get(idx, 'bn')}({var})"

        elif op_name == "layer_norm" or op_name == "layernorm":
            return f"self.{self.layer_map.get(idx, 'ln')}({var})"

        else:
            # Default: treat as function call
            args_str = ", ".join([str(a) for a in op.args])
            return f"{op_name}({var}, {args_str})" if args_str else f"{op_name}({var})"

    def _apply_activation(self, code: str, activation: str) -> str:
        """Apply activation function to code."""
        activation = activation.lower()

        if activation == "relu":
            return f"F.relu({code})"
        elif activation == "sigmoid":
            return f"torch.sigmoid({code})"
        elif activation == "tanh":
            return f"torch.tanh({code})"
        else:
            return code

    def _get_layer_var_name(self, base_name: str) -> str:
        """Get a unique variable name for a layer."""
        if base_name not in self.layer_counter:
            self.layer_counter[base_name] = 1
            return base_name + "1"
        else:
            self.layer_counter[base_name] += 1
            return base_name + str(self.layer_counter[base_name])

    def _generate_training(self, train: TrainNode) -> str:
        """Generate training function."""
        lines = [
            f"# Training: {train.model_name} on {train.dataset_name}",
            f"def train_{train.model_name.lower()}():",
        ]

        # Model instantiation
        lines.append(f"    model = {train.model_name}().to(device)")

        # Loss function
        loss_name = train.config.get("loss", "cross_entropy")
        loss_fn = self._get_loss_function(loss_name)
        lines.append(f"    criterion = {loss_fn}")

        # Optimizer
        optimizer_spec = train.config.get("optimizer", "adam(lr=1e-3)")
        optimizer_code = self._parse_optimizer(optimizer_spec)
        lines.append(f"    optimizer = {optimizer_code}")

        # Epochs
        epochs = train.config.get("epochs", 5)

        # Training loop
        lines.extend(
            [
                f"    ",
                f"    # Training loop",
                f"    for epoch in range({epochs}):",
                f"        model.train()",
                f"        running_loss = 0.0",
                f"        ",
                f"        for batch_idx, (data, target) in enumerate({train.dataset_name}):",
                f"            data, target = data.to(device), target.to(device)",
                f"            ",
                f"            optimizer.zero_grad()",
                f"            output = model(data)",
                f"            loss = criterion(output, target)",
                f"            loss.backward()",
                f"            optimizer.step()",
                f"            ",
                f"            running_loss += loss.item()",
                f"            ",
                f"            if batch_idx % 100 == 0:",
                f"                print(f'Epoch {{epoch+1}}/{epochs}, Batch {{batch_idx}}, Loss: {{loss.item():.4f}}')",
                f"        ",
                f"        avg_loss = running_loss / len({train.dataset_name})",
                f"        print(f'Epoch {{epoch+1}}/{epochs} completed. Average Loss: {{avg_loss:.4f}}')",
            ]
        )

        # Validation if specified
        if "validate_on" in train.config:
            val_dataset = train.config["validate_on"]
            lines.extend(
                [
                    f"        ",
                    f"        # Validation",
                    f"        model.eval()",
                    f"        correct = 0",
                    f"        total = 0",
                    f"        ",
                    f"        with torch.no_grad():",
                    f"            for data, target in {val_dataset}:",
                    f"                data, target = data.to(device), target.to(device)",
                    f"                output = model(data)",
                    f"                _, predicted = torch.max(output.data, 1)",
                    f"                total += target.size(0)",
                    f"                correct += (predicted == target).sum().item()",
                    f"        ",
                    f"        accuracy = 100 * correct / total",
                    f"        print(f'Validation Accuracy: {{accuracy:.2f}}%')",
                ]
            )

        lines.append("    ")
        lines.append("    return model")

        return "\n".join(lines)

    def _get_loss_function(self, loss_name: str) -> str:
        """Map loss name to PyTorch loss function."""
        loss_map = {
            "cross_entropy": "nn.CrossEntropyLoss()",
            "mse": "nn.MSELoss()",
            "bce": "nn.BCELoss()",
            "nll": "nn.NLLLoss()",
        }
        return loss_map.get(loss_name, "nn.CrossEntropyLoss()")

    def _parse_optimizer(self, optimizer_spec: str) -> str:
        """Parse optimizer specification into PyTorch code."""
        # Examples: "adam(lr=1e-3)", "sgd(lr=0.01, momentum=0.9)"
        if not isinstance(optimizer_spec, str):
            optimizer_spec = str(optimizer_spec)

        match = re.match(r"(\w+)\((.*)\)", optimizer_spec)
        if match:
            opt_name = match.group(1).lower()
            args_str = match.group(2)

            # Build optimizer
            opt_class = {
                "adam": "optim.Adam",
                "sgd": "optim.SGD",
                "adamw": "optim.AdamW",
                "rmsprop": "optim.RMSprop",
            }.get(opt_name, "optim.Adam")

            # Parse arguments
            if args_str:
                return f"{opt_class}(model.parameters(), {args_str})"
            else:
                return f"{opt_class}(model.parameters())"

        return "optim.Adam(model.parameters(), lr=1e-3)"

    def _generate_main(self, train: TrainNode) -> str:
        """Generate main execution block."""
        lines = [
            'if __name__ == "__main__":',
            f"    print('Starting training: {train.model_name} on {train.dataset_name}')",
            f"    model = train_{train.model_name.lower()}()",
            f"    print('Training completed!')",
        ]
        return "\n".join(lines)

    def _format_value(self, value: Any) -> str:
        """Format a value for code generation."""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, tuple):
            return str(value)
        elif isinstance(value, list):
            return str(value)
        else:
            return str(value)


def generate_torch_code(program: AuraneProgram) -> str:
    """
    Generate PyTorch Python code from an Aurane AST.

    Args:
        program: The parsed Aurane program AST.

    Returns:
        Python source code as a string.
    """
    generator = TorchCodeGenerator(program)
    return generator.generate()
