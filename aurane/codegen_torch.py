"""
PyTorch code generator for Aurane DSL.

This module converts Aurane AST nodes into idiomatic PyTorch Python code.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import re
import hashlib
import math

from .ast import (
    AuraneProgram,
    ConfigCall,
    ExperimentNode,
    DatasetNode,
    ModelNode,
    TrainNode,
    TrainGANNode,
    LayerOperation,
    LRScheduler,
)
from .shapes import infer_output_shape, ACTIVATION_NAMES
from .symbols import resolve_program
from .ir import lower_model
from .dtypes import model_dtypes
from .configuration import (
    validate_training_options,
    validate_program_configuration,
    GAN_METRICS,
    configuration_call,
    loss_configuration,
    optimizer_configuration,
    scheduler_configuration,
    effective_training_config,
)
from .metrics import classification_metric_helpers
from .runtime_templates import ATTENTION_METHOD, DATA_STATE_HELPERS


class TorchCodeGenerator:
    """Generates PyTorch code from Aurane AST."""

    def __init__(self, program: AuraneProgram):
        self.program = resolve_program(program)
        self.indent_level = 0
        self.layer_counter: Dict[str, int] = {}  # Track layer counts for naming
        self.layer_map: Dict[int, str] = {}  # Map operation index to layer variable name

    def generate(self) -> str:
        """Generate complete Python code from the AST."""
        validate_program_configuration(self.program)
        sections = []

        # Imports
        sections.append(self._generate_imports())

        # Experiment setup
        if self.program.experiments:
            sections.append(self._generate_experiment_setup(self.program.experiments[0]))
        elif self.program.trains or self.program.train_gans:
            sections.append('device = torch.device("cpu")')

        # Dataset loaders
        for dataset in self.program.datasets:
            sections.append(self._generate_dataset(dataset))

        # Model definitions
        for model in self.program.models:
            sections.append(self._generate_model(model))

        # Preserve source order across standard and adversarial training blocks.
        jobs: List[Union[TrainNode, TrainGANNode]] = [
            *self.program.trains,
            *self.program.train_gans,
        ]
        jobs.sort(key=lambda job: job.line)
        used_names = {model.name for model in self.program.models}
        used_names.update(f"_make_{dataset.name}_loader" for dataset in self.program.datasets)
        used_names.update(use.alias or use.module.split(".")[0] for use in self.program.uses)
        main = ['if __name__ == "__main__":']
        for job in jobs:
            if isinstance(job, TrainNode):
                base = f"train_{job.model_name.lower()}"
            else:
                base = f"train_gan_{job.generator_name.lower()}_{job.discriminator_name.lower()}"
            name, suffix = base, 2
            while name in used_names:
                name = f"{base}_{suffix}"
                suffix += 1
            used_names.add(name)
            if isinstance(job, TrainNode):
                sections.append(self._generate_training(job, name))
            else:
                sections.append(self._generate_gan_training(job, name))
            main.extend([f"    print('Starting {name} on {job.dataset_name}')", f"    {name}()"])
        if jobs:
            main.append("    print('Training completed!')")
            sections.append("\n".join(main))

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
        """Create loaders on demand; pass only explicit dataset constructor options."""
        lines = [f"# Dataset: {dataset.name}", f"def _make_{dataset.name}_loader():"]
        if not dataset.source:
            lines.append(
                f"    raise ValueError('Dataset {dataset.name} requires a source or an injected loader')"
            )
            return "\n".join(lines)
        module, _, constructor = dataset.source.rpartition(".")
        if not module:
            raise ValueError(f"Dataset source must be a qualified constructor: {dataset.source}")
        loader_options = {"batch", "shuffle", "num_workers", "pin_memory", "drop_last"}
        kwargs = {key: value for key, value in dataset.config.items() if key not in loader_options}
        lines.extend(
            [
                "    from importlib import import_module",
                f"    constructor = getattr(import_module({module!r}), {constructor!r})",
            ]
        )
        if "transforms" in kwargs:
            if "transform" in kwargs or not isinstance(kwargs["transforms"], list):
                raise ValueError("Use either a transform call or a transforms list")
            kwargs["transform"] = ConfigCall("Compose()", [kwargs.pop("transforms")])
        arguments = []
        for key, value in kwargs.items():
            if key in ("transform", "target_transform") and value is not None:
                expression = self._transform_expression(value)
                if "    from torchvision import transforms" not in lines:
                    lines.append("    from torchvision import transforms")
            else:
                expression = self._format_value(value)
            arguments.append(f"{key}={expression}")
        if module.startswith("torchvision.datasets") and "transform" not in kwargs:
            lines.append("    from torchvision.transforms import ToTensor")
            arguments.append("transform=ToTensor()")
        lines.append(f"    dataset = constructor({', '.join(arguments)})")
        loader_kwargs = {
            "batch_size": dataset.config.get("batch", 32),
            "shuffle": dataset.config.get("shuffle", dataset.config.get("train", True)),
            "num_workers": dataset.config.get("num_workers", 0),
            "pin_memory": dataset.config.get("pin_memory", False),
            "drop_last": dataset.config.get("drop_last", False),
        }
        arguments = [f"{key}={value!r}" for key, value in loader_kwargs.items()]
        lines.append(f"    return DataLoader(dataset, {', '.join(arguments)})")
        return "\n".join(lines)

    def _transform_expression(self, value: Any) -> str:
        """Construct declared torchvision transforms without evaluating source strings."""
        supported = {
            "ToTensor",
            "PILToTensor",
            "Normalize",
            "Resize",
            "CenterCrop",
            "RandomCrop",
            "RandomHorizontalFlip",
            "RandomVerticalFlip",
            "RandomResizedCrop",
            "ColorJitter",
            "Grayscale",
            "Pad",
            "RandomRotation",
        }
        if not isinstance(value, ConfigCall):
            raise ValueError("A transform must be a named transform call")
        name = value.split("(", 1)[0]
        if name == "Compose":
            if len(value.args) == 1 and not value.kwargs:
                pipeline = value.args[0]
            elif not value.args and set(value.kwargs) == {"transforms"}:
                pipeline = value.kwargs["transforms"]
            else:
                raise ValueError("Compose transform requires one list of transforms")
            if not isinstance(pipeline, list):
                raise ValueError("Compose transform requires a list of transforms")
            return (
                "transforms.Compose(["
                + ", ".join(self._transform_expression(item) for item in pipeline)
                + "])"
            )
        if name not in supported:
            raise ValueError(f"Unsupported transform: {name}")
        if any(isinstance(item, ConfigCall) for item in [*value.args, *value.kwargs.values()]):
            raise ValueError(f"{name} transform arguments must be literals")
        arguments = [repr(item) for item in value.args]
        arguments.extend(f"{key}={item!r}" for key, item in value.kwargs.items())
        return f"transforms.{name}({', '.join(arguments)})"

    def _generate_model(self, model: ModelNode) -> str:
        """Generate PyTorch model class."""
        lines = [
            f"# Model: {model.name}",
            f"class {model.name}(nn.Module):",
        ]

        # __init__ method
        init_lines = ["    def __init__(self):"]
        init_lines.append("        super().__init__()")

        graph = None
        if model.forward_block:
            input_shape = tuple(model.config.get("input_shape", (1, 28, 28)))
            graph = lower_model(model)
            self.layer_counter = {}
            self.layer_map = {}
            for index, node in enumerate(graph.nodes):
                shape = node.inputs[0].shape
                assert shape is not None
                layer, _, _ = self._operation_to_layer_def_with_shape(
                    node.to_operation(), index, shape[0] if shape else 1, shape
                )
                if layer:
                    init_lines.append(f"        {layer}")
        declared_dtype, parameter_dtype = model_dtypes(model)
        init_lines.append(f"        self.to(dtype=torch.{parameter_dtype})")
        lines.extend(init_lines)
        lines.append("")
        if graph is not None:
            attention = any(node.op_name.lower() == "multihead_attention" for node in graph.nodes)
            if attention:
                lines.extend(ATTENTION_METHOD.strip("\n").splitlines())
            signature = ", padding_mask=None" if attention else ""
            parameter = graph.inputs[0].name
            lines.append(f"    def forward(self, {parameter}{signature}):")
            if declared_dtype is not None:
                lines.extend(
                    [
                        f"        if {parameter}.dtype != torch.{declared_dtype}:",
                        f"            raise ValueError('{model.name} requires input_dtype {declared_dtype}')",
                    ]
                )

            if "input_padding_idx" in model.config:
                if not attention or len(graph.inputs[0].shape or ()) != 1:
                    raise ValueError("input_padding_idx requires token input and attention")
                lines.extend(
                    [
                        "        if padding_mask is None:",
                        f"            padding_mask = {parameter} == {model.config['input_padding_idx']}",
                    ]
                )

            for index, node in enumerate(graph.nodes):
                assert node.output is not None
                operation = node.to_operation()
                operation.activation = None
                inputs = [value.name for value in node.inputs]
                if node.op_name.lower() == "add":
                    code = "(" + " + ".join(inputs) + ")"
                elif node.op_name.lower() == "concat":
                    code = f"torch.cat([{', '.join(inputs)}], dim={node.kwargs.get('dim', 1)})"
                else:
                    code = self._operation_to_forward_code(operation, inputs[0], index)
                if node.activation:
                    code = (
                        f"({code} + {inputs[0]})"
                        if node.activation == "residual"
                        else self._apply_activation(code, node.activation)
                    )
                lines.append(f"        {node.output.name} = {code}")
            lines.append(f"        return {graph.outputs[0].name}")
        return "\n".join(lines)

    def _operation_to_layer_def_with_shape(
        self, op: LayerOperation, idx: int, in_channels: int, shape: tuple
    ) -> Tuple[Optional[str], int, tuple]:
        """Convert an operation to a layer definition, tracking shapes."""
        op_name = op.operation.lower()

        # Get unique layer name
        layer_var = self._get_layer_var_name(op_name)
        self.layer_map[idx] = layer_var

        new_shape = infer_output_shape(op, shape)

        if op_name in ("conv1d", "conv2d"):
            out_channels = op.args[0] if op.args else 32
            kernel = op.kwargs.get("kernel", 3)
            stride = op.kwargs.get("stride", 1)
            padding = op.kwargs.get("padding", 0)

            bias = op.kwargs.get("bias", True)
            layer_def = f"self.{layer_var} = nn.{'Conv1d' if op_name == 'conv1d' else 'Conv2d'}({in_channels}, {out_channels}, {kernel}, stride={stride}, padding={padding}, dilation={op.kwargs.get('dilation', 1)}, groups={op.kwargs.get('groups', 1)}, bias={bias})"
            return layer_def, out_channels, new_shape

        elif op_name in ("lstm", "gru"):
            hidden = op.args[0] if op.args else 128
            options = ", ".join(f"{key}={value!r}" for key, value in op.kwargs.items())
            kind = "LSTM" if op_name == "lstm" else "GRU"
            return (
                f"self.{layer_var} = nn.{kind}({shape[-1]}, {hidden}, batch_first=True, {options})",
                new_shape[-1],
                new_shape,
            )

        elif op_name in ("maxpool", "avgpool"):
            return None, in_channels, new_shape

        elif op_name == "flatten":
            flat_size = new_shape[0] if new_shape else 128
            return None, flat_size, new_shape

        elif op_name in ("dense", "linear"):
            # dense(out_features)
            out_features = op.args[0] if op.args else 128
            in_features = shape[-1] if shape else 128

            bias = op.kwargs.get("bias", True)
            layer_def = f"self.{layer_var} = nn.Linear({in_features}, {out_features}, bias={bias})"

            return layer_def, out_features, new_shape

        elif op_name == "dropout":
            p = op.args[0] if op.args else 0.5
            layer_def = f"self.{layer_var} = nn.Dropout({p})"
            return layer_def, in_channels, shape

        elif op_name in ("batch_norm", "batchnorm"):
            num_features = in_channels
            # Simple assumption: if shape is 3D, it's BatchNorm2d, else BatchNorm1d
            kind = "BatchNorm2d" if len(shape) == 3 else "BatchNorm1d"
            options = ", ".join(f"{key}={value!r}" for key, value in op.kwargs.items())
            layer_def = f"self.{layer_var} = nn.{kind}({num_features}, {options})"
            return layer_def, in_channels, shape

        elif op_name in ("layer_norm", "layernorm"):
            # LayerNorm needs normalized_shape
            normalized_shape = (shape[-1],)
            options = ", ".join(f"{key}={value!r}" for key, value in op.kwargs.items())
            layer_def = f"self.{layer_var} = nn.LayerNorm({normalized_shape}, {options})"
            return layer_def, in_channels, shape

        elif op_name == "embedding":
            num_embeddings = op.args[0] if op.args else 1000
            embedding_dim = op.args[1] if len(op.args) > 1 else 128
            options = ", ".join(f"{key}={value!r}" for key, value in op.kwargs.items())
            layer_def = (
                f"self.{layer_var} = nn.Embedding({num_embeddings}, {embedding_dim}, {options})"
            )
            return layer_def, embedding_dim, new_shape

        elif op_name == "multihead_attention":
            embed_dim = op.kwargs.get("dim", shape[-1])
            num_heads = op.kwargs.get("heads", 8)
            dropout = op.kwargs.get("dropout", 0.0)
            layer_def = f"self.{layer_var} = nn.MultiheadAttention({embed_dim}, {num_heads}, dropout={dropout}, batch_first=True)"
            return layer_def, in_channels, shape

        elif op_name == "positional_encoding":
            max_len = op.kwargs.get("max_len", 5000)
            dim = shape[-1]
            # We'll generate a learned positional encoding for now
            layer_def = f"self.{layer_var} = nn.Parameter(torch.randn(1, {max_len}, {dim}))"
            return layer_def, in_channels, shape

        # For other operations, pass through
        return None, in_channels, shape

        # For other operations, pass through
        return None, in_channels, shape

    def _operation_to_forward_code(self, op: LayerOperation, var: str, idx: int) -> str:
        """Generate the base operation; the IR caller applies its suffix uniformly."""
        name = op.operation.lower()
        module_operations = {
            "conv1d",
            "conv2d",
            "dense",
            "linear",
            "dropout",
            "batchnorm",
            "batch_norm",
            "layernorm",
            "layer_norm",
            "embedding",
        }
        if name in module_operations:
            return f"self.{self.layer_map[idx]}({var})"
        if name in ("lstm", "gru"):
            return f"self.{self.layer_map[idx]}({var})[0]"
        if name == "upsample":
            options = ", ".join(f"{key}={value!r}" for key, value in op.kwargs.items())
            return f"F.interpolate({var}, {options})"
        if name in ("maxpool", "avgpool"):
            kernel = op.args[0] if op.args else 2
            stride = op.kwargs.get("stride", kernel)
            function = "max_pool2d" if name == "maxpool" else "avg_pool2d"
            return f"F.{function}({var}, {kernel}, stride={stride})"
        if name == "flatten":
            return f"torch.flatten({var}, 1)"
        if name == "reshape":
            dimensions = op.args if op.args else [-1]
            if len(dimensions) == 1 and isinstance(dimensions[0], (tuple, list)):
                dimensions = list(dimensions[0])
            return f"{var}.reshape({var}.size(0), {', '.join(str(dim) for dim in dimensions)})"
        if name == "global_avg_pool":
            return f"{var}.mean(dim=tuple(range(2, {var}.dim())))"
        if name in ACTIVATION_NAMES:
            if name == "leaky_relu":
                return f"F.leaky_relu({var}, {op.args[0] if op.args else 0.01})"
            if name in ("softmax", "log_softmax"):
                return f"F.{name}({var}, dim={op.args[0] if op.args else -1})"
            return self._apply_activation(var, name)
        if name == "multihead_attention":
            return f"self._self_attention(self.{self.layer_map[idx]}, {var}, padding_mask, {op.kwargs.get('causal', False)!r})"
        if name == "positional_encoding":
            return f"{var} + self.{self.layer_map[idx]}[:, :{var}.size(1), :]"
        raise ValueError(f"Unsupported operation: {op.operation}")

    def _apply_activation(self, code: str, activation: str) -> str:
        name = activation.lower()
        if name not in ACTIVATION_NAMES:
            raise ValueError(f"Unsupported activation: {activation}")
        if name in ("sigmoid", "tanh"):
            return f"torch.{name}({code})"
        if name in ("softmax", "log_softmax"):
            return f"F.{name}({code}, dim=-1)"
        return f"F.{'silu' if name == 'swish' else name}({code})"

    def _get_layer_var_name(self, base_name: str) -> str:
        """Get a unique variable name for a layer."""
        if base_name not in self.layer_counter:
            self.layer_counter[base_name] = 1
            return base_name + "1"
        else:
            self.layer_counter[base_name] += 1
            return base_name + str(self.layer_counter[base_name])

    def _generate_training(self, train: TrainNode, function_name: str) -> str:
        """Generate an importable training function with injectable model and loaders."""
        validate_training_options(train)
        train.config = effective_training_config(train)
        patience = train.config.get("patience", 10)
        if type(patience) is not int or patience <= 0:
            raise ValueError("patience must be a positive integer")
        min_delta = train.config.get("min_delta", 0.0)
        if not isinstance(min_delta, (int, float)) or min_delta < 0:
            raise ValueError("min_delta must be nonnegative")
        checkpoint_every = train.config.get("checkpoint_every")
        if checkpoint_every is not None and (
            type(checkpoint_every) is not int or checkpoint_every <= 0
        ):
            raise ValueError("checkpoint_every must be a positive integer")
        checkpoint_dir = train.config.get("checkpoint_dir")
        if checkpoint_dir is None and (checkpoint_every or train.config.get("save_best")):
            checkpoint_dir = f"checkpoints/{train.model_name}"
        resume_default = train.config.get("resume_from")
        loss_name = train.config.get("loss", "cross_entropy")
        loss_fn = self._get_loss_function(loss_name)
        classification = loss_fn.startswith(("nn.CrossEntropyLoss", "nn.NLLLoss"))
        binary = loss_fn.startswith(("nn.BCELoss", "nn.BCEWithLogitsLoss"))
        summary_metrics = []
        metric_expressions = {}
        perplexity = False
        for metric in train.metrics:
            name = metric.name.lower()
            top_k = re.fullmatch(r"top([1-9][0-9]*)_accuracy", name)
            if (classification or binary) and name in (
                "precision",
                "recall",
                "f1",
                "f1_score",
                "auc",
            ):
                summary_metrics.append(name)
            elif binary and name in ("accuracy", "binary_accuracy"):
                threshold = 0 if loss_fn.startswith("nn.BCEWithLogitsLoss") else 0.5
                metric_expressions[name] = (
                    f"((output.detach() >= {threshold}) == target).sum().item()"
                )
            elif classification and name == "perplexity":
                perplexity = True
            elif classification and (name == "accuracy" or top_k):
                k = int(top_k.group(1)) if top_k else 1
                metric_expressions[name] = (
                    f"(scores.topk(min({k}, scores.size(-1)), dim=-1).indices == labels[:, None]).any(-1).sum().item()"
                )
            elif not classification and name in ("mse", "mae"):
                transform = "square" if name == "mse" else "abs"
                metric_expressions[name] = f"(output.detach() - target).{transform}().sum().item()"
            else:
                raise ValueError(f"Unsupported or incompatible metric: {metric.name}")
        optimizer = self._parse_optimizer(
            train.config.get("optimizer", "adam"), default_lr=train.config.get("lr", 0.001)
        )
        epochs = train.config.get("epochs", 5)
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        grad_clip = train.config.get("gradient_clip", train.config.get("gradient_clipping"))
        if grad_clip is not None and (not isinstance(grad_clip, (int, float)) or grad_clip <= 0):
            raise ValueError("gradient_clip must be positive")
        lines = [
            f"# Training: {train.model_name} on {train.dataset_name}",
            f"def {function_name}(train_loader=None, validation_loader=None, model=None, resume_from={resume_default!r}, test_loader=None):",
            f"    model = globals()[{train.model_name!r}]() if model is None else model",
            "    model = model.to(device)",
            "    if train_loader is None:",
            f"        train_loader = _make_{train.dataset_name}_loader()",
        ]
        if "validate_on" in train.config:
            lines.extend(
                [
                    "    if validation_loader is None:",
                    f"        validation_loader = _make_{train.config['validate_on']}_loader()",
                ]
            )
        lines.extend(
            [
                f"    criterion = {loss_fn}.to(device=device, dtype=next(model.parameters()).dtype)",
                f"    optimizer = {optimizer}",
                "    model.training_history = []",
                "    model.test_metrics = None",
                "    scheduler = None",
                "    scaler = None",
                "    start_epoch = 0",
                "    best_loss = float('inf')",
                "    bad_epochs = 0",
            ]
        )
        if summary_metrics:
            if classification:
                mode = (
                    "multiclass_logprob"
                    if loss_fn.startswith("nn.NLLLoss")
                    else "multiclass_logits"
                )
            else:
                mode = (
                    "binary_logits"
                    if loss_fn.startswith("nn.BCEWithLogitsLoss")
                    else "binary_probability"
                )
            lines.extend(classification_metric_helpers(mode, summary_metrics))
        if metric_expressions:
            lines.append("    def measure(output, target):")
            if classification:
                lines.extend(
                    [
                        "        labels = target.reshape(-1)",
                        "        valid = labels != criterion.ignore_index",
                        "        scores = output.detach().reshape(-1, output.size(-1))[valid]",
                        "        labels = labels[valid]",
                    ]
                )
            expressions = ", ".join(
                f"{name!r}: {expression}" for name, expression in metric_expressions.items()
            )
            lines.append(f"        return {{{expressions}}}")
        else:
            lines.extend(["    def measure(output, target):", "        return {}"])
        step_scheduler = (
            train.scheduler is not None and train.scheduler.name.lower() == "warmup_cosine"
        )
        if perplexity or step_scheduler:
            lines.append("    import math")
        loss_expression = "compute_loss(output, target)"
        lines.append("    def compute_loss(output, target):")
        if classification:
            lines.extend(
                [
                    "        if output.shape[:-1] != target.shape:",
                    "            raise ValueError('Classification output and target shape mismatch')",
                    "        return criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))",
                    "    def batch_counts(target):",
                    "        labels = target[target != criterion.ignore_index]",
                    "        weight = labels.numel() if criterion.weight is None else criterion.weight[labels].sum().item()",
                    "        return weight, labels.numel()",
                ]
            )
        else:
            lines.extend(
                [
                    "        if output.shape != target.shape:",
                    "            raise ValueError('Output and target shape must match; implicit broadcasting is not supported')",
                    *(
                        [
                            "        if not torch.all((target == 0) | (target == 1)):",
                            "            raise ValueError('Binary classification metrics require targets of 0 or 1')",
                        ]
                        if binary
                        and (
                            summary_metrics
                            or any(
                                name in metric_expressions
                                for name in ("accuracy", "binary_accuracy")
                            )
                        )
                        else []
                    ),
                    "        return criterion(output, target)",
                    "    def batch_counts(target):",
                    "        return target.numel(), target.numel()",
                ]
            )
        lines.extend(
            [
                "    def evaluate(loader, label):",
                "        model.eval()",
                "        total_loss = count = metric_count = 0.0",
                f"        totals = dict.fromkeys({list(metric_expressions)!r}, 0.0)",
                *(
                    ["        summary = {'counts': None, 'scores': [], 'labels': []}"]
                    if summary_metrics
                    else []
                ),
                "        with torch.no_grad():",
                "            for data, target in loader:",
                "                data, target = data.to(device), target.to(device)",
                "                weight, examples = batch_counts(target)",
                "                if weight == 0:",
                "                    continue",
                "                output = model(data)",
                "                loss = compute_loss(output, target)",
                "                if not torch.isfinite(loss):",
                "                    raise ValueError(f'{label} loss is not finite')",
                *(
                    ["                update_summary(summary, output, target)"]
                    if summary_metrics
                    else []
                ),
                "                total_loss += loss.item() * weight",
                "                count += weight",
                "                metric_count += examples",
                "                for name, value in measure(output, target).items():",
                "                    totals[name] += value",
                "        if count == 0:",
                "            raise ValueError(f'{label} data is empty')",
                "        result = {'loss': total_loss / count}",
                "        result.update({name: value / metric_count for name, value in totals.items()})",
            ]
        )
        if perplexity:
            lines.append(
                "        result['perplexity'] = math.exp(result['loss']) if result['loss'] < 709 else float('inf')"
            )
        if summary_metrics:
            lines.append("        result.update(finish_summary(summary))")
        lines.append("        return result")
        use_amp = train.config.get("mixed_precision", False)
        if use_amp:
            lines.extend(
                [
                    "    amp_enabled = device.type == 'cuda'",
                    "    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)",
                ]
            )
        if train.scheduler:
            lines.append(f"    scheduler = {self._generate_scheduler(train.scheduler)}")
        contract = {
            "model": train.model_name,
            "model_code": hashlib.sha256(
                self._generate_model(
                    next(model for model in self.program.models if model.name == train.model_name)
                ).encode()
            ).hexdigest(),
            "optimizer": optimizer.split("(", 1)[0],
            "loss": loss_fn,
            "scheduler": self._generate_scheduler(train.scheduler) if train.scheduler else None,
            "mixed_precision": bool(use_amp),
        }
        lines.append("    state_loaders = {'train': train_loader, 'validation': validation_loader}")
        lines.extend(DATA_STATE_HELPERS.strip("\n").splitlines())
        lines.append(f"    if {checkpoint_dir is not None!r} or resume_from is not None:")
        lines.append("        validate_data_state()")
        lines.extend(
            [
                "    import random",
                f"    contract = {contract!r}",
                "    if resume_from is not None:",
                "        state = torch.load(resume_from, map_location='cpu', weights_only=True)",
                "        if state.get('format') != 1 or state.get('contract') != contract:",
                "            raise ValueError('Checkpoint is incompatible with this training configuration')",
                "        model.load_state_dict(state['model'])",
                "        if scheduler is not None:",
                "            scheduler.load_state_dict(state['scheduler'])",
                "        optimizer.load_state_dict(state['optimizer'])",
                "        if scaler is not None:",
                "            scaler.load_state_dict(state['scaler'])",
                "        model.training_history = state['history']",
                "        start_epoch = state['epoch']",
                "        best_loss = state['best_loss']",
                "        bad_epochs = state['bad_epochs']",
                "        torch.set_rng_state(state['torch_rng'])",
                "        random.setstate(state['python_rng'])",
                "        if torch.cuda.is_available() and state['cuda_rng'] is not None:",
                "            torch.cuda.set_rng_state_all(state['cuda_rng'])",
                "        if state['loader_rng'] is not None:",
                "            generator = getattr(train_loader, 'generator', None)",
                "            if generator is None:",
                "                raise ValueError('Checkpoint requires a loader with a generator')",
                "            generator.set_state(state['loader_rng'])",
                "        if state.get('data_state') is not None:",
                "            restore_data_state(state['data_state'])",
            ]
        )
        if checkpoint_dir is not None:
            lines.extend(self._artifact_writer())
            lines.append(f"    checkpoint_dir = Path({checkpoint_dir!r})")
        lines.extend(
            [
                f"    for epoch in range(start_epoch, {epochs}):",
                "        model.train()",
                *(
                    ["        summary = {'counts': None, 'scores': [], 'labels': []}"]
                    if summary_metrics
                    else []
                ),
                "        running_loss = 0.0",
                "        count = metric_count = 0",
                f"        totals = dict.fromkeys({list(metric_expressions)!r}, 0.0)",
                "        for data, target in train_loader:",
                "            data, target = data.to(device), target.to(device)",
                "            weight, examples = batch_counts(target)",
                "            if weight == 0:",
                "                continue",
                "            optimizer.zero_grad(set_to_none=True)",
            ]
        )
        if use_amp:
            lines.extend(
                [
                    "            with torch.autocast(device_type=device.type, enabled=amp_enabled):",
                    "                output = model(data)",
                    f"                loss = {loss_expression}",
                ]
            )
        else:
            lines.extend(
                [
                    "            output = model(data)",
                    f"            loss = {loss_expression}",
                ]
            )
        lines.extend(
            [
                "            if not torch.isfinite(loss):",
                "                raise ValueError('Training loss is not finite')",
                (
                    "            scaler.scale(loss).backward()"
                    if use_amp
                    else "            loss.backward()"
                ),
            ]
        )
        if grad_clip is not None:
            if use_amp:
                lines.append("            scaler.unscale_(optimizer)")
            lines.append(
                f"            torch.nn.utils.clip_grad_norm_(model.parameters(), {grad_clip})"
            )
        lines.extend(
            [
                *(["            previous_scale = scaler.get_scale()"] if use_amp else []),
                "            scaler.step(optimizer)" if use_amp else "            optimizer.step()",
            ]
        )
        if use_amp:
            lines.append("            scaler.update()")
        if step_scheduler:
            if use_amp:
                lines.extend(
                    [
                        "            if scaler.get_scale() >= previous_scale:",
                        "                scheduler.step()",
                    ]
                )
            else:
                lines.append("            scheduler.step()")
        lines.extend(
            [
                *(
                    ["            update_summary(summary, output, target)"]
                    if summary_metrics
                    else []
                ),
                "            running_loss += loss.item() * weight",
                "            count += weight",
                "            metric_count += examples",
                "            for name, value in measure(output, target).items():",
                "                totals[name] += value",
                "        if count == 0:",
                "            raise ValueError('Training data is empty')",
                "        avg_loss = running_loss / count",
                "        record = {'epoch': epoch + 1, 'loss': avg_loss}",
                "        record.update({name: value / metric_count for name, value in totals.items()})",
                *(["        record.update(finish_summary(summary))"] if summary_metrics else []),
                "        if validation_loader is not None:",
                "            validation = evaluate(validation_loader, 'Validation')",
                "            record.update({'val_' + name: value for name, value in validation.items()})",
            ]
        )
        if perplexity:
            lines.extend(
                [
                    "        record['perplexity'] = math.exp(avg_loss) if avg_loss < 709 else float('inf')",
                    "        if 'val_loss' in record:",
                    "            record['val_perplexity'] = math.exp(record['val_loss']) if record['val_loss'] < 709 else float('inf')",
                ]
            )
        if train.scheduler and not step_scheduler:
            if train.scheduler.name.lower() in ("reduce_lr_on_plateau", "reduce_on_plateau"):
                lines.append("        scheduler.step(record.get('val_loss', avg_loss))")
            else:
                lines.append("        scheduler.step()")
        lines.extend(
            [
                "        record['lr'] = optimizer.param_groups[0]['lr']",
                "        model.training_history.append(record)",
                f"        print(f'Epoch {{epoch + 1}}/{epochs}: loss={{avg_loss:.6f}}')",
            ]
        )
        lines.extend(
            [
                "        monitored_loss = record.get('val_loss', avg_loss)",
                f"        improved = monitored_loss < best_loss - {min_delta}",
                "        if improved:",
                "            best_loss = monitored_loss",
                "            bad_epochs = 0",
                "        else:",
                "            bad_epochs += 1",
            ]
        )
        if checkpoint_dir is not None:
            lines.extend(
                [
                    "        loader_generator = getattr(train_loader, 'generator', None)",
                    "        state = {",
                    "            'format': 1, 'contract': contract, 'epoch': epoch + 1,",
                    "            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),",
                    "            'scheduler': scheduler.state_dict() if scheduler is not None else None,",
                    "            'scaler': scaler.state_dict() if scaler is not None else None,",
                    "            'history': model.training_history, 'best_loss': best_loss, 'bad_epochs': bad_epochs,",
                    "            'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),",
                    "            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,",
                    "            'loader_rng': loader_generator.get_state() if loader_generator is not None else None,",
                    "            'data_state': capture_data_state(),",
                    "        }",
                    "        save_artifact(state, checkpoint_dir / 'last.pt')",
                ]
            )
            if train.config.get("save_best"):
                lines.extend(
                    [
                        "        if improved:",
                        "            save_artifact(state, checkpoint_dir / 'best.pt')",
                    ]
                )
            if checkpoint_every is not None:
                lines.extend(
                    [
                        f"        if (epoch + 1) % {checkpoint_every} == 0:",
                        "            save_artifact(state, checkpoint_dir / f'epoch_{epoch + 1:04d}.pt')",
                    ]
                )
        if train.config.get("early_stopping"):
            lines.extend([f"        if bad_epochs >= {patience}:", "            break"])
        if "test_on" in train.config:
            lines.extend(
                [
                    "    if test_loader is None:",
                    f"        test_loader = _make_{train.config['test_on']}_loader()",
                ]
            )
        lines.extend(
            [
                "    if test_loader is not None:",
                "        model.test_metrics = evaluate(test_loader, 'Test')",
                "        print({'test': model.test_metrics})",
            ]
        )
        lines.append("    return model")
        return "\n".join(lines)

    def _artifact_writer(self) -> List[str]:
        """Emit the shared atomic writer inside a generated training function."""
        return [
            "    from pathlib import Path",
            "    import os",
            "    import tempfile",
            "    def save_artifact(value, destination, writer=torch.save):",
            "        destination.parent.mkdir(parents=True, exist_ok=True)",
            "        temporary = None",
            "        try:",
            "            with tempfile.NamedTemporaryFile(mode='wb', dir=destination.parent, prefix='.' + destination.name, suffix='.tmp', delete=False) as stream:",
            "                temporary = Path(stream.name)",
            "                writer(value, stream)",
            "            os.replace(temporary, destination)",
            "        finally:",
            "            if temporary is not None:",
            "                temporary.unlink(missing_ok=True)",
        ]

    def _configuration_call(self, spec: str):
        """Accept named configuration calls with literal arguments, never Python code."""
        return configuration_call(spec)

    def _get_loss_function(self, loss_name: str) -> str:
        kind, kwargs = loss_configuration(loss_name)
        arguments = []
        for key, value in kwargs.items():
            if key in ("weight", "pos_weight") and value is not None:
                arguments.append(f"{key}=torch.tensor({value!r}, dtype=torch.float32)")
            else:
                arguments.append(f"{key}={value!r}")
        return f"nn.{kind}({', '.join(arguments)})"

    def _parse_optimizer(self, optimizer_spec: str, default_lr=0.001) -> str:
        kind, kwargs = optimizer_configuration(optimizer_spec, default_lr)
        arguments = [f"{key}={value!r}" for key, value in kwargs.items()]
        return f"optim.{kind}(model.parameters(), {', '.join(arguments)})"

    def _generate_gan_training(self, train: TrainGANNode, function_name: str) -> str:
        """Generate alternating GAN updates with separate losses and frozen phases."""
        validate_training_options(train)
        metrics = train.config.get("metrics", list(GAN_METRICS))
        gen, disc = train.generator_name, train.discriminator_name
        model = next((model for model in self.program.models if model.name == gen), None)
        if model is None or not any(model.name == disc for model in self.program.models):
            raise ValueError("GAN training requires defined generator and discriminator models")
        shape = tuple(model.config.get("input_shape", (1, 28, 28)))
        if "latent_dim" in train.config and shape != (train.config["latent_dim"],):
            raise ValueError("latent_dim must match the generator input_shape")
        epochs = train.config.get("epochs", 100)
        d_steps = train.config.get("discriminator_steps", 1)
        g_steps = train.config.get("generator_steps", 1)
        sample_every = train.config.get("generate_samples_every")
        num_samples = train.config.get("num_samples", 64)
        for name, value in (
            ("epochs", epochs),
            ("discriminator_steps", d_steps),
            ("generator_steps", g_steps),
            ("num_samples", num_samples),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if sample_every is not None and (type(sample_every) is not int or sample_every <= 0):
            raise ValueError("generate_samples_every must be a positive integer")
        checkpoint_every = train.config.get("checkpoint_every")
        if checkpoint_every is not None and (
            type(checkpoint_every) is not int or checkpoint_every <= 0
        ):
            raise ValueError("checkpoint_every must be a positive integer")
        checkpoint_dir = train.config.get("checkpoint_dir")
        if checkpoint_dir is None and checkpoint_every is not None:
            checkpoint_dir = f"checkpoints/{gen}_{disc}"
        sample_format = train.config.get("sample_format", "tensor")
        if sample_format not in ("tensor", "png", "both"):
            raise ValueError("sample_format must be tensor, png or both")
        if sample_format != "tensor":
            output_shape = lower_model(model).outputs[0].shape
            if output_shape is None or len(output_shape) != 3 or output_shape[0] not in (1, 3):
                raise ValueError(
                    "PNG samples require generator output (1 or 3 channels, height, width)"
                )
        sample_range = train.config.get("sample_range", (-1.0, 1.0))
        if (
            not isinstance(sample_range, (tuple, list))
            or len(sample_range) != 2
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in sample_range
            )
            or sample_range[0] >= sample_range[1]
        ):
            raise ValueError("sample_range requires two finite increasing numbers")
        resume_default = train.config.get("resume_from")
        loss_g = self._get_loss_function(train.config.get("generator_loss", "binary_cross_entropy"))
        loss_d = self._get_loss_function(
            train.config.get("discriminator_loss", "binary_cross_entropy")
        )
        if not all(
            loss.startswith(("nn.BCELoss", "nn.BCEWithLogitsLoss", "nn.MSELoss"))
            for loss in (loss_g, loss_d)
        ):
            raise ValueError("GAN losses must be binary_cross_entropy, bce_with_logits or mse")
        opt_g = self._parse_optimizer(
            train.config.get("generator_optimizer", "adam(lr=0.0002, betas=(0.5, 0.999))")
        ).replace("model.parameters()", "netG.parameters()")
        opt_d = self._parse_optimizer(
            train.config.get("discriminator_optimizer", "adam(lr=0.0002, betas=(0.5, 0.999))")
        ).replace("model.parameters()", "netD.parameters()")
        lines = [
            f"# GAN Training: {gen} and {disc} on {train.dataset_name}",
            f"def {function_name}(train_loader=None, generator=None, discriminator=None, resume_from={resume_default!r}):",
            f"    netG = (globals()[{gen!r}]() if generator is None else generator).to(device)",
            f"    netD = (globals()[{disc!r}]() if discriminator is None else discriminator).to(device)",
            "    if train_loader is None:",
            f"        train_loader = _make_{train.dataset_name}_loader()",
            f"    criterionG = {loss_g}.to(device=device, dtype=next(netD.parameters()).dtype)",
            f"    criterionD = {loss_d}.to(device=device, dtype=next(netD.parameters()).dtype)",
            f"    optimizerG = {opt_g}",
            f"    optimizerD = {opt_d}",
            "    history = []",
            "    start_epoch = 0",
        ]
        if checkpoint_dir is not None or sample_every is not None:
            lines.extend(self._artifact_writer())
        if checkpoint_dir is not None:
            lines.append(f"    checkpoint_dir = Path({checkpoint_dir!r})")
        contract = {
            "generator": hashlib.sha256(self._generate_model(model).encode()).hexdigest(),
            "discriminator": hashlib.sha256(
                self._generate_model(
                    next(item for item in self.program.models if item.name == disc)
                ).encode()
            ).hexdigest(),
            "optimizer_g": opt_g.split("(", 1)[0],
            "optimizer_d": opt_d.split("(", 1)[0],
            "loss_g": loss_g,
            "loss_d": loss_d,
            "g_steps": g_steps,
            "d_steps": d_steps,
            "metrics": metrics,
        }
        lines.append("    state_loaders = {'train': train_loader}")
        lines.extend(DATA_STATE_HELPERS.strip("\n").splitlines())
        lines.append(f"    if {checkpoint_dir is not None!r} or resume_from is not None:")
        lines.append("        validate_data_state()")
        lines.extend(
            [
                "    import random",
                f"    contract = {contract!r}",
                "    if resume_from is not None:",
                "        state = torch.load(resume_from, map_location='cpu', weights_only=True)",
                "        if state.get('format') != 'gan-1' or state.get('contract') != contract:",
                "            raise ValueError('GAN checkpoint is incompatible with this training configuration')",
                "        netG.load_state_dict(state['generator'])",
                "        netD.load_state_dict(state['discriminator'])",
                "        optimizerG.load_state_dict(state['optimizer_g'])",
                "        optimizerD.load_state_dict(state['optimizer_d'])",
                "        history = state['history']",
                "        start_epoch = state['epoch']",
                "        torch.set_rng_state(state['torch_rng'])",
                "        random.setstate(state['python_rng'])",
                "        if torch.cuda.is_available() and state['cuda_rng'] is not None:",
                "            torch.cuda.set_rng_state_all(state['cuda_rng'])",
                "        if state['loader_rng'] is not None:",
                "            loader_generator = getattr(train_loader, 'generator', None)",
                "            if loader_generator is None:",
                "                raise ValueError('Checkpoint requires a loader with a generator')",
                "            loader_generator.set_state(state['loader_rng'])",
                "        if state.get('data_state') is not None:",
                "            restore_data_state(state['data_state'])",
                "    netG.training_history = netD.training_history = history",
            ]
        )
        if sample_every is not None:
            lines.extend(
                [
                    "    from pathlib import Path",
                    f"    sample_dir = Path({train.config.get('save_dir', './generated_samples')!r})",
                    "    sample_dir.mkdir(parents=True, exist_ok=True)",
                    f"    fixed_noise = torch.randn(({num_samples}, *{shape!r}), device=device, dtype=next(netG.parameters()).dtype, generator=torch.Generator(device=device).manual_seed(0))",
                ]
            )
        lines.extend(
            [
                f"    for epoch in range(start_epoch, {epochs}):",
                "        netD.train()",
                "        g_loss = d_loss = real_score = fake_score = 0.0",
                "        count = 0",
                "        for data, _ in train_loader:",
                "            real = data.to(device)",
                "            batch = real.size(0)",
                "            if batch == 0:",
                "                continue",
                "            optimizerG.zero_grad(set_to_none=True)",
                "            netG.eval()",
                f"            for _ in range({d_steps}):",
                "                optimizerD.zero_grad(set_to_none=True)",
                "                with torch.no_grad():",
                f"                    fake = netG(torch.randn((batch, *{shape!r}), device=device, dtype=next(netG.parameters()).dtype))",
                "                real_output, fake_output = netD(real), netD(fake.detach())",
                "                lossD = criterionD(real_output, torch.ones_like(real_output)) + criterionD(fake_output, torch.zeros_like(fake_output))",
                "                if not torch.isfinite(lossD):",
                "                    raise ValueError('Discriminator loss is not finite')",
                "                lossD.backward()",
                "                optimizerD.step()",
                f"                d_loss += lossD.item() * batch / {d_steps}",
                f"                real_score += real_output.detach().mean().item() * batch / {d_steps}",
                f"                fake_score += fake_output.detach().mean().item() * batch / {d_steps}",
                "            flags = [parameter.requires_grad for parameter in netD.parameters()]",
                "            netD.requires_grad_(False)",
                "            netD.eval()",
                "            netG.train()",
                "            try:",
                f"                for _ in range({g_steps}):",
                "                    optimizerG.zero_grad(set_to_none=True)",
                f"                    fake = netG(torch.randn((batch, *{shape!r}), device=device, dtype=next(netG.parameters()).dtype))",
                "                    output = netD(fake)",
                "                    lossG = criterionG(output, torch.ones_like(output))",
                "                    if not torch.isfinite(lossG):",
                "                        raise ValueError('Generator loss is not finite')",
                "                    lossG.backward()",
                "                    optimizerG.step()",
                f"                    g_loss += lossG.item() * batch / {g_steps}",
                "            finally:",
                "                for parameter, flag in zip(netD.parameters(), flags):",
                "                    parameter.requires_grad_(flag)",
                "                netD.train()",
                "            count += batch",
                "        if count == 0:",
                "            raise ValueError('GAN training data is empty')",
                "        record = {'epoch': epoch + 1, 'generator_loss': g_loss / count, 'discriminator_loss': d_loss / count, 'real_score': real_score / count, 'fake_score': fake_score / count}",
                f"        record = {{key: value for key, value in record.items() if key == 'epoch' or key in {metrics!r}}}",
                "        history.append(record)",
                "        print(record)",
            ]
        )
        if sample_every is not None:
            lines.extend(
                [
                    f"        if (epoch + 1) % {sample_every} == 0:",
                    "            netG.eval()",
                    "            with torch.no_grad():",
                    "                samples = netG(fixed_noise).cpu()",
                ]
            )
            if sample_format in ("tensor", "both"):
                lines.append(
                    "            save_artifact(samples, sample_dir / f'samples_{epoch + 1:04d}.pt')"
                )
            if sample_format in ("png", "both"):
                lines.extend(
                    [
                        "            from torchvision.utils import save_image",
                        f"            save_artifact(samples, sample_dir / f'samples_{{epoch + 1:04d}}.png', lambda value, stream: save_image(value, stream, format='PNG', normalize=True, value_range={tuple(sample_range)!r}))",
                    ]
                )
            lines.append("            netG.train()")
        if checkpoint_dir is not None:
            lines.extend(
                [
                    "        loader_generator = getattr(train_loader, 'generator', None)",
                    "        state = {",
                    "            'format': 'gan-1', 'contract': contract, 'epoch': epoch + 1,",
                    "            'generator': netG.state_dict(), 'discriminator': netD.state_dict(),",
                    "            'optimizer_g': optimizerG.state_dict(), 'optimizer_d': optimizerD.state_dict(),",
                    "            'history': history, 'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),",
                    "            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,",
                    "            'loader_rng': loader_generator.get_state() if loader_generator is not None else None,",
                    "            'data_state': capture_data_state(),",
                    "        }",
                    "        save_artifact(state, checkpoint_dir / 'last.pt')",
                ]
            )
            if checkpoint_every is not None:
                lines.extend(
                    [
                        f"        if (epoch + 1) % {checkpoint_every} == 0:",
                        "            save_artifact(state, checkpoint_dir / f'epoch_{epoch + 1:04d}.pt')",
                    ]
                )
        lines.append("    return netG, netD")
        return "\n".join(lines)

    def _generate_scheduler(self, scheduler: LRScheduler) -> str:
        """Generate PyTorch LR scheduler code."""
        name = scheduler.name.lower()
        kind, kwargs = scheduler_configuration(scheduler)
        if name == "warmup_cosine":
            warmup = kwargs.get("warmup_steps", 0)
            maximum = kwargs.get("max_steps")
            decay = (
                f"0.5 * (1 + math.cos(math.pi * min(1.0, (step - {warmup}) / {maximum - warmup})))"
            )
            factor = f"step / {warmup} if step < {warmup} else {decay}" if warmup else decay
            return f"optim.lr_scheduler.LambdaLR(optimizer, lambda step: {factor})"

        options = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
        return f"optim.lr_scheduler.{kind}(optimizer, {options})"

    def _format_value(self, value: Any) -> str:
        """Format a value for code generation."""
        if isinstance(value, str):
            return repr(value)
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
