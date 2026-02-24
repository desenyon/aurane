"""
Unified shape inference and parameter calculation engine for Aurane.
Consolidates logic from type_checker, visualizer, and profiler.
"""

from typing import Tuple, List, Any, Optional, Union
from .ast import LayerOperation


def to_int(val: Any, default: int = 0) -> int:
    """Safely convert a value to int."""
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            return default
    return default


def infer_output_shape(operation: LayerOperation, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Infer the output shape of a layer given its operation and input shape.

    Args:
        operation: The LayerOperation AST node.
        input_shape: The shape of the input tensor (e.g., (C, H, W) or (Features,)).

    Returns:
        The inferred output shape.
    """
    op_name = operation.operation.lower()

    if op_name == "conv2d":
        if len(input_shape) != 3:
            # Fallback or error
            return input_shape

        c, h, w = input_shape
        out_channels = to_int(operation.args[0] if operation.args else 32, 32)
        kernel = to_int(operation.kwargs.get("kernel", 3), 3)
        stride = to_int(operation.kwargs.get("stride", 1), 1)
        padding = to_int(operation.kwargs.get("padding", 0), 0)

        h_out = (h + 2 * padding - kernel) // stride + 1
        w_out = (w + 2 * padding - kernel) // stride + 1
        return (out_channels, h_out, w_out)

    elif op_name in ("maxpool", "avgpool"):
        if len(input_shape) != 3:
            return input_shape

        c, h, w = input_shape
        kernel = to_int(operation.args[0] if operation.args else 2, 2)
        stride = to_int(operation.kwargs.get("stride", kernel), kernel)
        return (c, h // stride, w // stride)

    elif op_name == "flatten":
        size = 1
        for s in input_shape:
            size *= s
        return (size,)

    elif op_name in ("dense", "linear"):
        out_features = to_int(operation.args[0] if operation.args else 128, 128)
        return (out_features,)

    elif op_name == "reshape":
        # Handle reshape(1, 28, 28) -> (1, 28, 28)
        # Often used for (latent_dim,) -> (1, 28, 28)
        if operation.args:
            return tuple(to_int(a) for a in operation.args)
        return input_shape

    elif op_name == "embedding":
        # embedding(vocab_size, embed_dim)
        # Input is usually (seq_len,), output (seq_len, embed_dim)
        embed_dim = to_int(operation.args[1] if len(operation.args) > 1 else 128, 128)
        seq_len = input_shape[0] if input_shape else 1
        return (seq_len, embed_dim)

    elif op_name == "global_avg_pool":
        if len(input_shape) == 3:
            return (input_shape[0],)
        return input_shape

    # Identity operations (shape-preserving)
    if op_name in (
        "dropout",
        "batchnorm",
        "batch_norm",
        "layer_norm",
        "layernorm",
        "relu",
        "leaky_relu",
        "gelu",
        "sigmoid",
        "tanh",
        "softmax",
        "positional_encoding",
        "multihead_attention",
        "residual",
    ):
        return input_shape

    return input_shape


def calculate_params(operation: LayerOperation, input_shape: Tuple[int, ...]) -> int:
    """
    Calculate the number of trainable parameters in a layer.

    Args:
        operation: The LayerOperation AST node.
        input_shape: The shape of the input tensor.

    Returns:
        Number of parameters.
    """
    op_name = operation.operation.lower()

    if op_name == "conv2d":
        if len(input_shape) != 3:
            return 0
        in_channels = input_shape[0]
        out_channels = to_int(operation.args[0] if operation.args else 32, 32)
        kernel = to_int(operation.kwargs.get("kernel", 3), 3)
        # (kernel * kernel * in_channels + 1) * out_channels
        return (kernel * kernel * in_channels + 1) * out_channels

    elif op_name in ("dense", "linear"):
        in_features = input_shape[0] if input_shape else 0
        out_features = to_int(operation.args[0] if operation.args else 128, 128)
        # (in_features + 1) * out_features
        return (in_features + 1) * out_features

    elif op_name == "embedding":
        vocab_size = to_int(operation.args[0] if operation.args else 1000, 1000)
        embed_dim = to_int(operation.args[1] if len(operation.args) > 1 else 128, 128)
        return vocab_size * embed_dim

    elif op_name in ("batchnorm", "batch_norm"):
        # gamma and beta per channel
        return 2 * input_shape[0] if input_shape else 0

    elif op_name in ("layernorm", "layer_norm"):
        # gamma and beta per normalized element
        size = 1
        for s in input_shape:
            size *= s
        return 2 * size

    elif op_name == "multihead_attention":
        embed_dim = to_int(
            operation.kwargs.get("dim", input_shape[-1] if input_shape else 512), 512
        )
        # Q, K, V projections + output projection (each is dim*dim + dim)
        return 4 * (embed_dim * embed_dim + embed_dim)

    return 0
