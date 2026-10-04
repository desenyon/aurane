"""Supported training fields shared by analysis and code generation."""

import ast as python_ast
import builtins
import math
import re

from .ast import ConfigCall, TrainGANNode, TrainNode, ForwardGraphBlock
from .diagnostics import at_config
from .dtypes import model_dtypes, FLOAT_DTYPES
from .ir import lower_model, lower_forward_block

TRAIN_OPTIONS = {
    "epochs",
    "optimizer",
    "lr",
    "loss",
    "validate_on",
    "test_on",
    "gradient_clip",
    "gradient_clipping",
    "mixed_precision",
    "early_stopping",
    "patience",
    "min_delta",
    "checkpoint_dir",
    "checkpoint_every",
    "save_best",
    "resume_from",
}
GAN_OPTIONS = {
    "epochs",
    "latent_dim",
    "generator_optimizer",
    "discriminator_optimizer",
    "generator_loss",
    "discriminator_loss",
    "discriminator_steps",
    "generator_steps",
    "generate_samples_every",
    "num_samples",
    "save_dir",
    "metrics",
    "checkpoint_dir",
    "checkpoint_every",
    "resume_from",
    "sample_format",
    "sample_range",
}
GAN_METRICS = ("generator_loss", "discriminator_loss", "real_score", "fake_score")


def validate_training_options(train: TrainNode | TrainGANNode) -> None:
    with at_config(train):
        _validate_training_options(train)


def _validate_training_options(train: TrainNode | TrainGANNode) -> None:
    allowed = TRAIN_OPTIONS if isinstance(train, TrainNode) else GAN_OPTIONS
    unknown = set(train.config) - allowed
    if unknown:
        with at_config(train, sorted(unknown)[0]):
            raise ValueError(f"Unsupported training options: {', '.join(sorted(unknown))}")
    for key in ("mixed_precision", "early_stopping", "save_best"):
        if key in train.config and type(train.config[key]) is not bool:
            with at_config(train, key):
                raise ValueError(f"{key} must be a boolean")
    for key in ("validate_on", "test_on"):
        if key in train.config and (
            not isinstance(train.config[key], str) or not train.config[key].isidentifier()
        ):
            with at_config(train, key):
                raise ValueError(f"{key} must name a dataset")
    if isinstance(train, TrainGANNode):
        metrics = train.config.get("metrics", list(GAN_METRICS))
        if not isinstance(metrics, list) or any(metric not in GAN_METRICS for metric in metrics):
            with at_config(train, "metrics"):
                raise ValueError(f"Unsupported GAN metrics: {metrics}")

    config = effective_training_config(train)
    if isinstance(train, TrainNode):
        with at_config(train, "loss"):
            loss, _ = loss_configuration(config.get("loss", "cross_entropy"))
        with at_config(train, "optimizer"):
            optimizer_configuration(config.get("optimizer", "adam"), config.get("lr", 0.001))
        if train.scheduler:
            with at_config(train, "scheduler"):
                scheduler_configuration(train.scheduler)
        for metric in train.metrics:
            name = metric.name.lower()
            multiclass = loss in ("CrossEntropyLoss", "NLLLoss")
            binary = loss in ("BCELoss", "BCEWithLogitsLoss")
            valid = (
                (
                    name in {"precision", "recall", "f1", "f1_score", "auc", "accuracy"}
                    and (multiclass or binary)
                )
                or (
                    multiclass
                    and (name == "perplexity" or re.fullmatch(r"top[1-9][0-9]*_accuracy", name))
                )
                or (binary and name == "binary_accuracy")
                or (not multiclass and name in {"mse", "mae"})
            )
            if not valid:
                with at_config(train, "metrics"):
                    raise ValueError(f"Unsupported or incompatible metric: {metric.name}")
    else:
        for role in ("generator", "discriminator"):
            with at_config(train, role + "_loss"):
                loss, _ = loss_configuration(config.get(role + "_loss", "binary_cross_entropy"))
            if loss not in ("BCELoss", "BCEWithLogitsLoss", "MSELoss"):
                with at_config(train, role + "_loss"):
                    raise ValueError(
                        "GAN losses must be binary_cross_entropy, bce_with_logits or mse"
                    )
            with at_config(train, role + "_optimizer"):
                optimizer_configuration(config.get(role + "_optimizer", "adam"))
        if config.get("sample_format", "tensor") not in ("tensor", "png", "both"):
            with at_config(train, "sample_format"):
                raise ValueError("sample_format must be tensor, png or both")
        sample_range = config.get("sample_range", (-1, 1))
        if (
            not isinstance(sample_range, (tuple, list))
            or len(sample_range) != 2
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in sample_range
            )
            or sample_range[0] >= sample_range[1]
        ):
            with at_config(train, "sample_range"):
                raise ValueError("sample_range requires two finite increasing numbers")


def training_references(train: TrainNode | TrainGANNode) -> list[tuple[str, str]]:
    references = [("dataset", train.dataset_name)]
    if isinstance(train, TrainNode):
        references.append(("model", train.model_name))
        references.extend(
            ("dataset", train.config[key])
            for key in ("validate_on", "test_on")
            if key in train.config
        )
    else:
        references.extend([("model", train.generator_name), ("model", train.discriminator_name)])
    return references


LOSS_CLASSES = {
    "cross_entropy": "CrossEntropyLoss",
    "cross_entropy_loss": "CrossEntropyLoss",
    "cross_entropy_with_label_smoothing": "CrossEntropyLoss",
    "nll": "NLLLoss",
    "mse": "MSELoss",
    "mse_loss": "MSELoss",
    "mae": "L1Loss",
    "l1_loss": "L1Loss",
    "huber": "HuberLoss",
    "bce": "BCELoss",
    "bce_loss": "BCELoss",
    "binary_cross_entropy": "BCELoss",
    "bce_with_logits": "BCEWithLogitsLoss",
}
OPTIMIZER_CLASSES = {
    "adam": "Adam",
    "adamw": "AdamW",
    "sgd": "SGD",
    "rmsprop": "RMSprop",
    "adagrad": "Adagrad",
    "adadelta": "Adadelta",
}
SCHEDULER_CLASSES = {
    "step_lr": "StepLR",
    "exponential_lr": "ExponentialLR",
    "cosine_annealing": "CosineAnnealingLR",
    "reduce_lr_on_plateau": "ReduceLROnPlateau",
    "reduce_on_plateau": "ReduceLROnPlateau",
    "warmup_cosine": "LambdaLR",
}


def configuration_call(spec):
    """Parse legacy string calls without execution; preserve structured call values."""
    if not isinstance(spec, str):
        raise ValueError(f"Expected a named configuration, got {spec!r}")
    if isinstance(spec, ConfigCall):
        return spec.split("(", 1)[0].lower(), list(spec.args), dict(spec.kwargs)
    node = python_ast.parse(spec, mode="eval").body
    if isinstance(node, python_ast.Name):
        return node.id.lower(), [], {}
    if isinstance(node, python_ast.Call) and isinstance(node.func, python_ast.Name):
        if any(item.arg is None for item in node.keywords):
            raise ValueError("Expanded configuration arguments are unsupported")
        return (
            node.func.id.lower(),
            [python_ast.literal_eval(value) for value in node.args],
            {item.arg: python_ast.literal_eval(item.value) for item in node.keywords},
        )
    raise ValueError(f"Unsupported configuration: {spec}")


def number(value, name, *, minimum=0, maximum=None, strict_min=False, strict_max=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (value <= minimum if strict_min else value < minimum)
        or (maximum is not None and (value >= maximum if strict_max else value > maximum))
    ):
        raise ValueError(f"{name} must be a finite number in its supported range")


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def keywords(name, values, allowed):
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"Unsupported {name} options: {', '.join(sorted(unknown))}")


def loss_configuration(spec):
    name, args, values = configuration_call(spec)
    if name not in LOSS_CLASSES:
        raise ValueError(f"Unsupported loss: {name}")
    if args:
        raise ValueError("Loss settings must use named arguments")
    kind = LOSS_CLASSES[name]
    if name == "cross_entropy_with_label_smoothing":
        if "smoothing" in values and "label_smoothing" in values:
            raise ValueError("Use either smoothing or label_smoothing")
        values.setdefault("label_smoothing", values.pop("smoothing", 0.1))
    options = {
        "CrossEntropyLoss": {"weight", "ignore_index", "label_smoothing"},
        "NLLLoss": {"weight", "ignore_index"},
        "MSELoss": set(),
        "L1Loss": set(),
        "HuberLoss": {"delta"},
        "BCELoss": {"weight"},
        "BCEWithLogitsLoss": {"weight", "pos_weight"},
    }
    keywords(name, values, options[kind] | {"reduction"})
    if values.get("reduction", "mean") != "mean":
        raise ValueError("Training requires mean loss reduction")
    for key, value in values.items():
        if key in ("weight", "pos_weight") and value is not None:
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"{key} requires a nonempty list of positive weights")
            for item in value:
                number(item, key, strict_min=True)
        elif key == "ignore_index" and type(value) is not int:
            raise ValueError("ignore_index must be an integer")
        elif key == "label_smoothing":
            number(value, key, maximum=1)
        elif key == "delta":
            number(value, key, strict_min=True)
    return kind, values


def optimizer_configuration(spec, default_lr=0.001):
    name, args, values = configuration_call(spec)
    if name not in OPTIMIZER_CLASSES:
        raise ValueError(f"Unsupported optimizer: {name}")
    if args:
        raise ValueError("Optimizer settings must use named arguments")
    options = {
        "adam": {"betas", "eps", "amsgrad", "capturable", "fused"},
        "adamw": {"betas", "eps", "amsgrad", "capturable", "fused"},
        "sgd": {"momentum", "dampening", "nesterov", "fused"},
        "rmsprop": {"alpha", "eps", "momentum", "centered", "capturable"},
        "adagrad": {"lr_decay", "initial_accumulator_value", "eps", "fused"},
        "adadelta": {"rho", "eps", "capturable"},
    }
    keywords(
        name,
        values,
        options[name] | {"lr", "weight_decay", "foreach", "maximize", "differentiable"},
    )
    values.setdefault("lr", default_lr)
    for key, value in values.items():
        if key in ("foreach", "fused") and value is None:
            continue
        if key in (
            "amsgrad",
            "nesterov",
            "centered",
            "foreach",
            "maximize",
            "capturable",
            "differentiable",
            "fused",
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} {key} must be boolean")
            if key in ("capturable", "differentiable") and value:
                raise ValueError(f"{key}=True is not supported by generated training")
        elif key == "betas":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("betas requires two values")
            for beta in value:
                number(beta, "beta", maximum=1, strict_max=True)
        else:
            number(value, f"{name} {key}", maximum=1 if key in ("rho", "alpha") else None)
    if values.get("nesterov") and (
        values.get("momentum", 0) <= 0 or values.get("dampening", 0) != 0
    ):
        raise ValueError("Nesterov requires positive momentum and zero dampening")
    if values.get("fused") and values.get("foreach"):
        raise ValueError("fused and foreach cannot both be True")
    return OPTIMIZER_CLASSES[name], values


def scheduler_configuration(scheduler):
    name = scheduler.name.lower()
    if name not in SCHEDULER_CLASSES:
        raise ValueError(f"Unsupported scheduler: {name}")
    orders = {
        "StepLR": ("step_size", "gamma", "last_epoch"),
        "ExponentialLR": ("gamma", "last_epoch"),
        "CosineAnnealingLR": ("T_max", "eta_min", "last_epoch"),
        "ReduceLROnPlateau": (
            "mode",
            "factor",
            "patience",
            "threshold",
            "threshold_mode",
            "cooldown",
            "min_lr",
            "eps",
        ),
        "LambdaLR": ("warmup_steps", "max_steps"),
    }
    kind = SCHEDULER_CLASSES[name]
    args = scheduler.params.get("args", [])
    values = dict(scheduler.params.get("kwargs", {}))
    order = orders[kind]
    if len(args) > len(order) or (kind == "LambdaLR" and args):
        raise ValueError(f"Unsupported positional arguments for {name}")
    for key, value in zip(order, args):
        if key in values:
            raise ValueError(f"Duplicate scheduler argument: {key}")
        values[key] = value
    keywords(name, values, order)
    required = {
        "StepLR": "step_size",
        "ExponentialLR": "gamma",
        "CosineAnnealingLR": "T_max",
        "LambdaLR": "max_steps",
    }.get(kind)
    if required and required not in values:
        raise ValueError(f"{name} requires {required}")
    for key, value in values.items():
        if key in ("step_size", "T_max", "max_steps"):
            integer(value, key)
        elif key in ("patience", "cooldown", "warmup_steps"):
            integer(value, key, 0)
        elif key == "last_epoch":
            if type(value) is not int or value != -1:
                raise ValueError("Use resume_from instead of scheduler last_epoch")
        elif key in ("mode", "threshold_mode"):
            if value not in (("min", "max") if key == "mode" else ("rel", "abs")):
                raise ValueError(f"Invalid scheduler {key}: {value!r}")
        elif key == "min_lr" and isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError("min_lr requires one value for the single parameter group")
            number(value[0], key)
        else:
            number(value, key, maximum=1 if key == "factor" else None, strict_max=key == "factor")
    if kind == "LambdaLR" and values.get("warmup_steps", 0) >= values["max_steps"]:
        raise ValueError("warmup_cosine requires 0 <= warmup_steps < max_steps")
    return kind, values


def effective_training_config(train):
    config = dict(train.config)
    if isinstance(train, TrainNode):
        with at_config(train, "callbacks"):
            for callback in train.callbacks:
                name, args, values = configuration_call(callback.name)
                allowed = {
                    "early_stopping": {"patience", "min_delta"},
                    "checkpoint": {"checkpoint_dir", "checkpoint_every", "save_best"},
                }
                if name not in allowed or args:
                    raise ValueError(f"Unsupported callback: {callback.name}")
                keywords(name, values, allowed[name])
                # Validate supplied callback values even when explicit fields override them.
                _validate_training_scalars(values)
                for key, value in values.items():
                    config.setdefault(key, value)
                config.setdefault(
                    "early_stopping" if name == "early_stopping" else "checkpoint_every",
                    True if name == "early_stopping" else 1,
                )
    for key, value in config.items():
        with at_config(train, key if key in train.config else "callbacks"):
            _validate_training_scalars({key: value})
    return config


def _validate_training_scalars(config):
    for key in (
        "epochs",
        "patience",
        "checkpoint_every",
        "latent_dim",
        "generator_steps",
        "discriminator_steps",
        "generate_samples_every",
        "num_samples",
    ):
        if key in config:
            integer(config[key], key)
    for key in ("lr", "min_delta", "gradient_clip", "gradient_clipping"):
        if key in config:
            number(config[key], key, strict_min=key.startswith("gradient"))
    for key in ("checkpoint_dir", "resume_from", "save_dir"):
        if key in config and (
            not isinstance(config[key], str) or not config[key] or "\x00" in config[key]
        ):
            raise ValueError(f"{key} must be a nonempty path string")
    for key in ("mixed_precision", "early_stopping", "save_best"):
        if key in config and type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")


def validate_program_configuration(program):
    """Reject definition ambiguity before optional analysis and code emission."""
    if len(program.experiments) > 1:
        with at_config(program.experiments[1]):
            raise ValueError("Only one experiment block is supported per program")
    reserved = set(dir(builtins)) | {
        "torch",
        "nn",
        "F",
        "optim",
        "DataLoader",
        "device",
        "__name__",
        "__file__",
    }
    used = set()
    runtime_imports = {
        "torch": "torch",
        "nn": "torch.nn",
        "F": "torch.nn.functional",
        "optim": "torch.optim",
    }
    for use in program.uses:
        with at_config(use):
            binding = use.alias or use.module.split(".")[0]
            imported = use.module if use.alias else use.module.split(".")[0]
            if binding in reserved and runtime_imports.get(binding) != imported:
                raise ValueError(
                    f"Import {binding!r} conflicts with a runtime binding at line {use.line}"
                )
    for node in [*program.datasets, *program.models]:
        with at_config(node):
            key = (type(node).__name__, node.name)
            if key in used:
                raise ValueError(f"Duplicate definition {node.name!r} at line {node.line}")
            used.add(key)
    bindings = reserved | {use.alias or use.module.split(".")[0] for use in program.uses}
    bindings.update(f"_make_{dataset.name}_loader" for dataset in program.datasets)
    for model in program.models:
        with at_config(model):
            model_dtypes(model)
            with at_config(model, "input_shape"):
                shape = model.config.get("input_shape", (1, 28, 28))
                if (
                    not isinstance(shape, (tuple, list))
                    or not shape
                    or any(type(dim) is not int or dim == 0 or dim < -1 for dim in shape)
                ):
                    raise ValueError(
                        "Invalid input_shape; expected nonzero integer dimensions, optionally -1 for unknown"
                    )
            if model.name in bindings:
                raise ValueError(
                    f"Model name {model.name!r} conflicts with a runtime binding at line {model.line}"
                )
            if model.forward_block and model.forward_block.parameter in reserved | {
                "self",
                "padding_mask",
            }:
                raise ValueError(
                    f"forward parameter conflicts with a runtime binding at line {model.forward_block.line}"
                )
            if "input_padding_idx" in model.config:
                with at_config(model, "input_padding_idx"):
                    integer(model.config["input_padding_idx"], "input_padding_idx", 0)
                    shape = model.config.get("input_shape", (1, 28, 28))
                    if not isinstance(shape, (tuple, list)) or len(shape) != 1:
                        raise ValueError("input_padding_idx requires one-dimensional token input")
                    block = model.forward_block
                    operations = (
                        [node.operation for node in block.nodes if node.operation]
                        if isinstance(block, ForwardGraphBlock)
                        else block.operations if block else []
                    )
                    if not any(
                        operation.operation.lower() == "multihead_attention"
                        for operation in operations
                    ):
                        raise ValueError("input_padding_idx requires attention")

    model_names = {model.name for model in program.models}
    for train in program.trains:
        with at_config(train):
            if train.model_name not in model_names:
                raise ValueError(f"Undefined model {train.model_name!r} at line {train.line}")
    for train in program.train_gans:
        with at_config(train):
            models = {model.name: model for model in program.models}
            if train.generator_name not in models or train.discriminator_name not in models:
                missing = [
                    name
                    for name in (train.generator_name, train.discriminator_name)
                    if name not in models
                ]
                raise ValueError(
                    f"Undefined GAN model(s): {', '.join(missing)} at line {train.line}"
                )
            generator, discriminator = (
                models[train.generator_name],
                models[train.discriminator_name],
            )
            if generator.forward_block is None or discriminator.forward_block is None:
                raise ValueError("GAN models require forward blocks")
            input_dtype, parameter_dtype = model_dtypes(generator)
            if input_dtype is not None and input_dtype not in FLOAT_DTYPES:
                raise ValueError("GAN noise requires a floating input_dtype")
            output = lower_model(generator).outputs[0]
            shape = tuple(generator.config.get("input_shape", (1, 28, 28)))
            if any(type(dim) is not int or dim <= 0 for dim in shape):
                raise ValueError("GAN noise requires a concrete positive input_shape")
            if "latent_dim" in train.config and shape != (train.config["latent_dim"],):
                with at_config(train, "latent_dim"):
                    raise ValueError("latent_dim must match the generator input_shape")
            expected = tuple(discriminator.config.get("input_shape", (1, 28, 28)))
            if (
                output.shape is None
                or len(output.shape) != len(expected)
                or any(
                    left != right and -1 not in (left, right)
                    for left, right in zip(output.shape, expected)
                )
            ):
                raise ValueError("GAN generator output shape must match discriminator input_shape")
            actual_dtype = output.type_hint or parameter_dtype
            declared, parameter_dtype = model_dtypes(discriminator)
            if declared is not None and declared != actual_dtype:
                raise ValueError("GAN generator output dtype must match discriminator input_dtype")
            lower_forward_block(
                discriminator.forward_block,
                expected,
                input_dtype=actual_dtype,
                parameter_dtype=parameter_dtype,
            )
    for experiment in program.experiments:
        with at_config(experiment):
            config = experiment.config
            unknown = sorted(set(config) - {"seed", "device", "backend"})
            with at_config(experiment, unknown[0] if unknown else None):
                keywords("experiment", config, {"seed", "device", "backend"})
            if "seed" in config:
                with at_config(experiment, "seed"):
                    integer(config["seed"], "seed", 0)
                if config["seed"] >= 2**64:
                    with at_config(experiment, "seed"):
                        raise ValueError("seed must be below 2**64")
            if not isinstance(config.get("device", "auto"), str) or not re.fullmatch(
                r"auto|(?:cpu|cuda|mps)(?::[0-9]+)?", config.get("device", "auto")
            ):
                with at_config(experiment, "device"):
                    raise ValueError("Unsupported experiment device")
            if config.get("backend", "torch") != "torch":
                with at_config(experiment, "backend"):
                    raise ValueError(
                        "Experiment backend must be torch; select plugins through the compiler backend argument"
                    )
    for dataset in program.datasets:
        with at_config(dataset):
            for key in ("batch", "num_workers"):
                if key in dataset.config:
                    with at_config(dataset, key):
                        integer(dataset.config[key], key, 1 if key == "batch" else 0)
            for key in ("shuffle", "pin_memory", "drop_last"):
                if key in dataset.config and type(dataset.config[key]) is not bool:
                    with at_config(dataset, key):
                        raise ValueError(f"Dataset {key} must be boolean")
