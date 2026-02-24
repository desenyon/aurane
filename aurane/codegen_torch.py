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
    TrainGANNode,
    LayerOperation,
    ForwardBlock,
    LRScheduler,
)
from .shapes import infer_output_shape


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

        # GAN Training functions
        for train_gan in self.program.train_gans:
            sections.append(self._generate_gan_training(train_gan))

        # Main execution
        if self.program.trains:
            sections.append(self._generate_main(self.program.trains[0]))
        elif self.program.train_gans:
            sections.append(self._generate_gan_training_main(self.program.train_gans[0]))

        return "\n\n".join(sections)

    def _generate_imports(self) -> str:
        """Generate import statements."""
        imports = [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "import torch.optim as optim",
            "from torch.utils.data import DataLoader",
        ]

        # Add imports from 'use' statements
        for use in self.program.uses:
            if use.alias:
                imports.append(f"import {use.module} as {use.alias}")
            else:
                imports.append(f"import {use.module}")

        # Always add torchvision if not already present, as it's common in examples
        # but ideally we should check if it's used. For now, keep it for compatibility.
        if not any("torchvision" in i for i in imports):
            imports.append("import torchvision")
            imports.append("import torchvision.transforms as transforms")

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

        new_shape = infer_output_shape(op, shape)

        if op_name == "conv2d":
            out_channels = op.args[0] if op.args else 32
            kernel = op.kwargs.get("kernel", 3)
            stride = op.kwargs.get("stride", 1)
            padding = op.kwargs.get("padding", 0)

            layer_def = f"self.{layer_var} = nn.Conv2d({in_channels}, {out_channels}, {kernel}, stride={stride}, padding={padding})"
            return layer_def, out_channels, new_shape

        elif op_name == "maxpool":
            return None, in_channels, new_shape

        elif op_name == "flatten":
            flat_size = new_shape[0] if new_shape else 128
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

        elif op_name in ("batch_norm", "batchnorm"):
            num_features = in_channels
            # Simple assumption: if shape is 3D, it's BatchNorm2d, else BatchNorm1d
            if len(shape) == 3:
                layer_def = f"self.{layer_var} = nn.BatchNorm2d({num_features})"
            else:
                layer_def = f"self.{layer_var} = nn.BatchNorm1d({num_features})"
            return layer_def, in_channels, shape

        elif op_name in ("layer_norm", "layernorm"):
            # LayerNorm needs normalized_shape
            normalized_shape = shape
            layer_def = f"self.{layer_var} = nn.LayerNorm({normalized_shape})"
            return layer_def, in_channels, shape

        elif op_name == "embedding":
            num_embeddings = op.args[0] if op.args else 1000
            embedding_dim = op.args[1] if len(op.args) > 1 else 128
            layer_def = f"self.{layer_var} = nn.Embedding({num_embeddings}, {embedding_dim})"
            # Input is (seq_len,), output is (seq_len, embedding_dim)
            seq_len = shape[0] if shape else 1
            new_shape = (seq_len, embedding_dim)
            return layer_def, embedding_dim, new_shape

        elif op_name == "multihead_attention":
            embed_dim = op.kwargs.get("dim", in_channels)
            num_heads = op.kwargs.get("heads", 8)
            dropout = op.kwargs.get("dropout", 0.0)
            layer_def = f"self.{layer_var} = nn.MultiheadAttention({embed_dim}, {num_heads}, dropout={dropout}, batch_first=True)"
            return layer_def, in_channels, shape

        elif op_name == "positional_encoding":
            max_len = op.kwargs.get("max_len", 5000)
            dim = in_channels
            # We'll generate a learned positional encoding for now
            layer_def = f"self.{layer_var} = nn.Parameter(torch.randn(1, {max_len}, {dim}))"
            return layer_def, in_channels, shape

        # For other operations, pass through
        return None, in_channels, shape

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
            stride = op.kwargs.get("stride", kernel)
            return f"F.max_pool2d({var}, {kernel}, stride={stride})"

        elif op_name == "avgpool":
            kernel = op.args[0] if op.args else 2
            stride = op.kwargs.get("stride", kernel)
            return f"F.avg_pool2d({var}, {kernel}, stride={stride})"

        elif op_name == "flatten":
            return f"torch.flatten({var}, 1)"

        elif op_name == "reshape":
            shape_args = list(op.args) if op.args else [-1]
            # Ensure batch dimension is preserved if not present
            if len(shape_args) > 0 and shape_args[0] != -1:
                shape_args = [-1] + shape_args
            return f"{var}.view{tuple(shape_args)}"

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

        elif op_name in ("batch_norm", "batchnorm"):
            layer_var = self.layer_map.get(idx, self._get_layer_var_name(op_name))
            return f"self.{layer_var}({var})"

        elif op_name in ("layer_norm", "layernorm"):
            layer_var = self.layer_map.get(idx, self._get_layer_var_name(op_name))
            return f"self.{layer_var}({var})"

        elif op_name == "embedding":
            layer_var = self.layer_map.get(idx, self._get_layer_var_name(op_name))
            return f"self.{layer_var}({var})"

        elif op_name == "multihead_attention":
            layer_var = self.layer_map.get(idx, self._get_layer_var_name(op_name))
            # nn.MultiheadAttention returns (output, weights)
            return f"self.{layer_var}({var}, {var}, {var})[0]"

        elif op_name == "positional_encoding":
            layer_var = self.layer_map.get(idx, self._get_layer_var_name(op_name))
            return f"{var} + self.{layer_var}[:, :{var}.size(1), :]"

        else:
            # Default: treat as function call
            args_str = ", ".join([str(a) for a in op.args])
            return f"{op_name}({var}, {args_str})" if args_str else f"{op_name}({var})"

    def _apply_activation(self, code: str, activation: str) -> str:
        """Apply activation function to code."""
        activation = activation.lower()

        if activation == "relu":
            return f"F.relu({code})"
        elif activation == "leaky_relu":
            return f"F.leaky_relu({code}, 0.01)"
        elif activation == "gelu":
            return f"F.gelu({code})"
        elif activation == "sigmoid":
            return f"torch.sigmoid({code})"
        elif activation == "tanh":
            return f"torch.tanh({code})"
        elif activation == "softmax":
            return f"F.softmax({code}, dim=-1)"
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

        # Mixed precision
        use_amp = train.config.get("mixed_precision", False)
        if use_amp:
            lines.append(f"    scaler = torch.cuda.amp.GradScaler()")

        # Scheduler
        if train.scheduler:
            scheduler_code = self._generate_scheduler(train.scheduler)
            lines.append(f"    scheduler = {scheduler_code}")

        # Gradient clipping
        grad_clip = train.config.get("gradient_clipping", None)

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
                f"            ",
            ]
        )

        if use_amp:
            lines.extend(
                [
                    f"            with torch.cuda.amp.autocast():",
                    f"                output = model(data)",
                    f"                loss = criterion(output, target)",
                    f"            ",
                    f"            scaler.scale(loss).backward()",
                ]
            )
            if grad_clip:
                lines.append(f"            scaler.unscale_(optimizer)")
                lines.append(
                    f"            torch.nn.utils.clip_grad_norm_(model.parameters(), {grad_clip})"
                )
            lines.append(f"            scaler.step(optimizer)")
            lines.append(f"            scaler.update()")
        else:
            lines.extend(
                [
                    f"            output = model(data)",
                    f"            loss = criterion(output, target)",
                    f"            loss.backward()",
                ]
            )
            if grad_clip:
                lines.append(
                    f"            torch.nn.utils.clip_grad_norm_(model.parameters(), {grad_clip})"
                )
            lines.append(f"            optimizer.step()")

        lines.extend(
            [
                f"            ",
                f"            running_loss += loss.item()",
                f"            ",
                f"            if batch_idx % 100 == 0:",
                f"                print(f'Epoch {{epoch+1}}/{epochs}, Batch {{batch_idx}}, Loss: {{loss.item():.4f}}')",
            ]
        )

        # Step scheduler
        if train.scheduler:
            if train.scheduler.name.lower() == "reduce_lr_on_plateau":
                lines.append(f"        scheduler.step(avg_loss)")
            else:
                lines.append(f"        scheduler.step()")

        lines.extend(
            [
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

    def _generate_gan_training(self, train: TrainGANNode) -> str:
        """Generate specialized GAN training function."""
        gen = train.generator_name
        disc = train.discriminator_name
        lines = [
            f"# GAN Training: {gen} and {disc} on {train.dataset_name}",
            f"def train_gan_{gen.lower()}_{disc.lower()}():",
            f"    # Models",
            f"    netG = {gen}().to(device)",
            f"    netD = {disc}().to(device)",
            f"    ",
            f"    # Binary Cross Entropy loss",
            f"    criterion = nn.BCELoss()",
            f"    ",
            f"    # Optimizers",
            f"    optimizerG = {self._parse_optimizer(train.config.get('generator_optimizer', 'adam(lr=2e-4, betas=(0.5, 0.999))')).replace('model.parameters()', 'netG.parameters()')}",
            f"    optimizerD = {self._parse_optimizer(train.config.get('discriminator_optimizer', 'adam(lr=2e-4, betas=(0.5, 0.999))')).replace('model.parameters()', 'netD.parameters()')}",
            f"    ",
            f"    epochs = {train.config.get('epochs', 100)}",
            f"    ",
            f"    for epoch in range(epochs):",
            f"        for i, (data, _) in enumerate({train.dataset_name}):",
            f"            # 1. Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))",
            f"            netD.zero_grad()",
            f"            real_cpu = data.to(device)",
            f"            batch_size = real_cpu.size(0)",
            f"            label = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)",
            f"            ",
            f"            output = netD(real_cpu).view(-1)",
            f"            errD_real = criterion(output, label)",
            f"            errD_real.backward()",
            f"            ",
            f"            noise = torch.randn(batch_size, {train.config.get('latent_dim', 100)}, 1, 1, device=device)",
            f"            # Handle 1D noise if needed",
            f"            if len(netG.config.get('input_shape', (100,))) == 1:",
            f"                noise = noise.view(batch_size, -1)",
            f"            ",
            f"            fake = netG(noise)",
            f"            label.fill_(0.0)",
            f"            output = netD(fake.detach()).view(-1)",
            f"            errD_fake = criterion(output, label)",
            f"            errD_fake.backward()",
            f"            optimizerD.step()",
            f"            ",
            f"            # 2. Update Generator: maximize log(D(G(z)))",
            f"            netG.zero_grad()",
            f"            label.fill_(1.0)",
            f"            output = netD(fake).view(-1)",
            f"            errG = criterion(output, label)",
            f"            errG.backward()",
            f"            optimizerG.step()",
            f"            ",
            f"            if i % 50 == 0:",
            f"                print(f'[{{epoch}}/{{epochs}}][{{i}}/{{len({train.dataset_name})}}] Loss_D: {{errD_real.item()+errD_fake.item():.4f}} Loss_G: {{errG.item():.4f}}')",
            f"    ",
            f"    return netG, netD",
        ]
        return "\n".join(lines)

    def _generate_scheduler(self, scheduler: LRScheduler) -> str:
        """Generate PyTorch LR scheduler code."""
        name = scheduler.name.lower()
        params = scheduler.params
        args = params.get("args", [])
        kwargs = params.get("kwargs", {})

        # Build kwargs string
        kwargs_str = ", ".join([f"{k}={self._format_value(v)}" for k, v in kwargs.items()])
        if args:
            args_str = ", ".join([self._format_value(a) for a in args])
            combined = f"{args_str}, {kwargs_str}" if kwargs_str else args_str
        else:
            combined = kwargs_str

        # Map name to PyTorch scheduler
        sched_class = {
            "step_lr": "optim.lr_scheduler.StepLR",
            "exponential_lr": "optim.lr_scheduler.ExponentialLR",
            "cosine_annealing": "optim.lr_scheduler.CosineAnnealingLR",
            "reduce_lr_on_plateau": "optim.lr_scheduler.ReduceLROnPlateau",
        }.get(name, "optim.lr_scheduler.StepLR")

        return f"{sched_class}(optimizer, {combined})"

    def _generate_gan_training_main(self, train: TrainGANNode) -> str:
        """Generate main execution block for GAN."""
        lines = [
            'if __name__ == "__main__":',
            f"    print('Starting GAN training...')",
            f"    netG, netD = train_gan_{train.generator_name.lower()}_{train.discriminator_name.lower()}()",
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
