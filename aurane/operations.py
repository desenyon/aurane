"""Supported operation arguments, shared by semantic analysis and shape lowering."""

import math
from typing import Dict, Optional, Tuple, Set

from .ast import LayerOperation

ACTIVATION_NAMES = {
    "relu",
    "gelu",
    "leaky_relu",
    "sigmoid",
    "tanh",
    "softmax",
    "log_softmax",
    "silu",
    "swish",
    "mish",
    "elu",
    "selu",
    "hardswish",
}

# Maximum positional argument count (None means variadic), accepted keyword names.
OPERATION_SPECS: Dict[str, Tuple[Optional[int], Set[str]]] = {
    name: (0, set()) for name in ACTIVATION_NAMES
}
OPERATION_SPECS.update(
    {
        "conv2d": (1, {"kernel", "stride", "padding", "dilation", "groups", "bias"}),
        "dense": (1, {"bias"}),
        "maxpool": (1, {"stride"}),
        "avgpool": (1, {"stride"}),
        "dropout": (1, set()),
        "flatten": (0, set()),
        "reshape": (None, set()),
        "global_avg_pool": (0, set()),
        "batchnorm": (0, {"eps", "momentum", "affine", "track_running_stats"}),
        "layer_norm": (0, {"eps", "elementwise_affine", "bias"}),
        "embedding": (2, {"padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse"}),
        "multihead_attention": (0, {"heads", "dim", "dropout", "causal"}),
        "positional_encoding": (0, {"max_len"}),
        "lstm": (1, {"num_layers", "bidirectional", "dropout", "bias"}),
        "upsample": (0, {"size", "scale_factor", "mode", "align_corners"}),
        "concat": (0, {"dim"}),
        "add": (0, set()),
        "leaky_relu": (1, set()),
        "softmax": (1, set()),
        "log_softmax": (1, set()),
    }
)
for alias, canonical in {
    "conv1d": "conv2d",
    "linear": "dense",
    "batch_norm": "batchnorm",
    "layernorm": "layer_norm",
    "gru": "lstm",
}.items():
    OPERATION_SPECS[alias] = OPERATION_SPECS[canonical]


def validate_operation(operation: LayerOperation) -> None:
    """Reject options that cannot be honored before emitting executable code."""
    name = operation.operation.lower()
    if name not in OPERATION_SPECS:
        raise ValueError(f"Unsupported operation: {operation.operation}")
    count, keywords = OPERATION_SPECS[name]
    if count is not None and len(operation.args) > count:
        raise ValueError(f"{name} accepts at most {count} positional arguments")
    unknown = set(operation.kwargs) - keywords
    if unknown:
        raise ValueError(f"Unsupported {name} options: {', '.join(sorted(unknown))}")
    if operation.activation and operation.activation.lower() not in ACTIVATION_NAMES | {"residual"}:
        raise ValueError(f"Unsupported activation: {operation.activation}")
    for key in {
        "bias",
        "affine",
        "elementwise_affine",
        "track_running_stats",
        "causal",
        "scale_grad_by_freq",
        "sparse",
        "bidirectional",
        "align_corners",
    }:
        if key in operation.kwargs and type(operation.kwargs[key]) is not bool:
            raise ValueError(f"{name} {key} must be a boolean")
    for key in {"eps", "max_norm", "norm_type"}:
        value = operation.kwargs.get(key)
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        ):
            raise ValueError(f"{name} {key} must be finite and positive")
    if "momentum" in operation.kwargs:
        value = operation.kwargs["momentum"]
        if value is not None and (type(value) not in (int, float) or not 0 <= value <= 1):
            raise ValueError(f"{name} momentum must be between 0 and 1 or None")
    probability = (
        operation.args[0]
        if name == "dropout" and operation.args
        else operation.kwargs.get("dropout", 0)
    )
    if type(probability) not in (int, float) or not 0 <= probability <= 1:
        raise ValueError(f"{name} dropout probability must be between 0 and 1")
