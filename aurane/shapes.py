"""
Unified shape inference and parameter calculation engine for Aurane.
Consolidates logic from type_checker, visualizer, and profiler.
"""

from typing import Tuple, List, Any, Optional, Union
import math
from .ast import LayerOperation

from .operations import ACTIVATION_NAMES, validate_operation


def _positive_integer(value, name, *, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return value


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
    validate_operation(operation)
    if not input_shape or any(type(dim) is not int or dim == 0 or dim < -1 for dim in input_shape):
        raise ValueError(f"Invalid input shape: {input_shape}")

    if op_name in ("conv1d", "conv2d"):
        spatial_rank = 1 if op_name == "conv1d" else 2
        if len(input_shape) != spatial_rank + 1:
            raise ValueError(f"{op_name} requires channels and {spatial_rank} spatial dimensions")
        channels, *spatial = input_shape
        out_channels = _positive_integer(
            operation.args[0] if operation.args else 32, "out_channels"
        )
        kernel = _positive_integer(operation.kwargs.get("kernel", 3), "kernel")
        stride = _positive_integer(operation.kwargs.get("stride", 1), "stride")
        padding = _positive_integer(operation.kwargs.get("padding", 0), "padding", allow_zero=True)
        dilation = _positive_integer(operation.kwargs.get("dilation", 1), "dilation")
        groups = _positive_integer(operation.kwargs.get("groups", 1), "groups")
        if channels <= 0 or channels % groups or out_channels % groups:
            raise ValueError(
                f"{op_name} input and output channels must be positive and divisible by groups"
            )
        result = tuple(
            -1 if size == -1 else (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
            for size in spatial
        )
        if any(original != -1 and output <= 0 for original, output in zip(spatial, result)):
            raise ValueError(f"{op_name} kernel produces an empty output shape")
        return (out_channels, *result)

    elif op_name in ("lstm", "gru"):
        if len(input_shape) != 2 or input_shape[-1] <= 0:
            raise ValueError(f"{op_name} requires (sequence, positive features)")
        hidden = _positive_integer(operation.args[0] if operation.args else 128, "hidden_size")
        layers = _positive_integer(operation.kwargs.get("num_layers", 1), "num_layers")
        if layers == 1 and operation.kwargs.get("dropout", 0):
            raise ValueError(f"{op_name} dropout requires num_layers greater than 1")
        return (input_shape[0], hidden * (2 if operation.kwargs.get("bidirectional", False) else 1))

    elif op_name == "upsample":
        rank = len(input_shape) - 1
        mode = operation.kwargs.get("mode", "nearest")
        modes = {
            "nearest": {1, 2, 3},
            "nearest-exact": {1, 2, 3},
            "linear": {1},
            "bilinear": {2},
            "bicubic": {2},
            "trilinear": {3},
        }
        if mode not in modes or rank not in modes[mode]:
            raise ValueError(
                f"upsample mode {mode!r} is incompatible with {rank} spatial dimensions"
            )
        if mode in ("nearest", "nearest-exact") and "align_corners" in operation.kwargs:
            raise ValueError("upsample align_corners requires a linear interpolation mode")
        size, scale = operation.kwargs.get("size"), operation.kwargs.get("scale_factor")
        if (size is None) == (scale is None):
            raise ValueError("upsample requires exactly one of size or scale_factor")
        value: Any = size if size is not None else scale
        values = tuple(value) if isinstance(value, (tuple, list)) else (value,) * rank
        if len(values) != rank:
            raise ValueError("upsample requires one size or scale per spatial dimension")
        if size is not None:
            result = tuple(_positive_integer(value, "size") for value in values)
        else:
            if any(
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for value in values
            ):
                raise ValueError("upsample scale_factor must be finite and positive")
            result = tuple(
                -1 if dim == -1 else math.floor(dim * factor)
                for dim, factor in zip(input_shape[1:], values)
            )
            if 0 in result:
                raise ValueError("upsample produces an empty output shape")
        return (input_shape[0], *result)

    elif op_name in ("maxpool", "avgpool"):
        if len(input_shape) != 3:
            raise ValueError(f"{op_name} requires input_shape (channels, height, width)")

        c, h, w = input_shape
        kernel = _positive_integer(operation.args[0] if operation.args else 2, "kernel")
        stride = _positive_integer(operation.kwargs.get("stride", kernel), "stride")
        if h < kernel or w < kernel:
            raise ValueError(f"{op_name} kernel produces an empty output shape")
        return (c, (h - kernel) // stride + 1, (w - kernel) // stride + 1)

    elif op_name == "flatten":
        size = -1 if -1 in input_shape else math.prod(input_shape)
        return (size,)

    elif op_name in ("dense", "linear"):
        if input_shape[-1] <= 0:
            raise ValueError("dense requires a known positive input feature dimension")
        out_features = _positive_integer(
            operation.args[0] if operation.args else 128, "out_features"
        )
        return (*input_shape[:-1], out_features)

    elif op_name == "reshape":
        # Handle reshape(1, 28, 28) -> (1, 28, 28)
        # Often used for (latent_dim,) -> (1, 28, 28)
        shape = tuple(operation.args) if operation.args else (-1,)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if (
            not shape
            or any(type(dim) is not int or dim == 0 or dim < -1 for dim in shape)
            or shape.count(-1) > 1
        ):
            raise ValueError("reshape requires positive dimensions and at most one -1")
        size = -1 if -1 in input_shape else math.prod(input_shape)
        known = math.prod(dim for dim in shape if dim != -1)
        if -1 in shape:
            if size > 0 and size % known:
                raise ValueError("reshape size is not divisible by the requested shape")
            return tuple(size // known if dim == -1 and size > 0 else dim for dim in shape)
        if size > 0 and math.prod(shape) != size:
            raise ValueError("reshape must preserve the number of elements per sample")
        return shape

    elif op_name == "embedding":
        # embedding(vocab_size, embed_dim)
        # Input is usually (seq_len,), output (seq_len, embed_dim)
        vocabulary = _positive_integer(
            operation.args[0] if operation.args else 1000, "num_embeddings"
        )
        padding = operation.kwargs.get("padding_idx")
        if padding is not None and (
            type(padding) is not int or not -vocabulary <= padding < vocabulary
        ):
            raise ValueError("embedding padding_idx must be an integer within the vocabulary")
        embed_dim = _positive_integer(
            operation.args[1] if len(operation.args) > 1 else 128, "embedding_dim"
        )
        return (*input_shape, embed_dim)

    elif op_name == "global_avg_pool":
        if len(input_shape) >= 2:
            return (input_shape[0],)
        raise ValueError("global_avg_pool requires spatial dimensions")

    elif op_name == "multihead_attention":
        if len(input_shape) != 2:
            raise ValueError("multihead_attention requires (sequence, features)")
        dim = _positive_integer(operation.kwargs.get("dim", input_shape[-1]), "attention dim")
        heads = _positive_integer(operation.kwargs.get("heads", 8), "attention heads")
        if dim != input_shape[-1] or dim % heads:
            raise ValueError("attention dim must match input features and be divisible by heads")
    elif op_name == "positional_encoding":
        maximum = _positive_integer(operation.kwargs.get("max_len", 5000), "max_len")
        if len(input_shape) != 2 or input_shape[0] > maximum:
            raise ValueError("positional_encoding requires (sequence, features) within max_len")
    elif op_name in ("batchnorm", "batch_norm"):
        if len(input_shape) not in (1, 2, 3) or input_shape[0] <= 0:
            raise ValueError("batchnorm requires known channels and at most two spatial dimensions")
    elif op_name in ("layernorm", "layer_norm"):
        if input_shape[-1] <= 0:
            raise ValueError("layer_norm requires a known positive feature dimension")
    elif op_name in ("softmax", "log_softmax"):
        dim = operation.args[0] if operation.args else -1
        rank = len(input_shape) + 1
        if type(dim) is not int or not -rank <= dim < rank:
            raise ValueError(f"{op_name} dim is invalid for rank {rank}")
    elif op_name == "leaky_relu":
        slope = operation.args[0] if operation.args else 0.01
        if type(slope) not in (int, float) or not math.isfinite(slope):
            raise ValueError("leaky_relu slope must be a finite number")

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

    if op_name in ("conv1d", "conv2d"):
        kernel = operation.kwargs.get("kernel", 3)
        groups = operation.kwargs.get("groups", 1)
        out_channels = operation.args[0] if operation.args else 32
        return int(
            (
                kernel ** (len(input_shape) - 1) * (input_shape[0] // groups)
                + int(operation.kwargs.get("bias", True))
            )
            * out_channels
        )

    elif op_name in ("lstm", "gru"):
        hidden = operation.args[0] if operation.args else 128
        directions = 2 if operation.kwargs.get("bidirectional", False) else 1
        gates = 4 if op_name == "lstm" else 3
        bias = 2 if operation.kwargs.get("bias", True) else 0
        layers = operation.kwargs.get("num_layers", 1)
        return sum(
            directions
            * gates
            * hidden
            * ((input_shape[-1] if index == 0 else directions * hidden) + hidden + bias)
            for index in range(layers)
        )

    elif op_name in ("dense", "linear"):
        in_features = input_shape[-1] if input_shape else 0
        out_features = to_int(operation.args[0] if operation.args else 128, 128)
        # (in_features + 1) * out_features
        return (in_features + int(operation.kwargs.get("bias", True))) * out_features

    elif op_name == "embedding":
        vocab_size = to_int(operation.args[0] if operation.args else 1000, 1000)
        embed_dim = to_int(operation.args[1] if len(operation.args) > 1 else 128, 128)
        return vocab_size * embed_dim

    elif op_name in ("batchnorm", "batch_norm"):
        # gamma and beta per channel
        return 2 * input_shape[0] if input_shape and operation.kwargs.get("affine", True) else 0

    elif op_name in ("layernorm", "layer_norm"):
        # gamma and beta per normalized element
        return (
            (1 + int(operation.kwargs.get("bias", True))) * input_shape[-1]
            if operation.kwargs.get("elementwise_affine", True)
            else 0
        )

    elif op_name == "multihead_attention":
        embed_dim = to_int(
            operation.kwargs.get("dim", input_shape[-1] if input_shape else 512), 512
        )
        # Q, K, V projections + output projection (each is dim*dim + dim)
        return 4 * (embed_dim * embed_dim + embed_dim)

    elif op_name == "positional_encoding":
        return to_int(operation.kwargs.get("max_len", 5000), 5000) * input_shape[-1]

    return 0


def infer_graph_output_shape(
    operation: LayerOperation, inputs: List[Tuple[int, ...]]
) -> Tuple[int, ...]:
    """Infer a graph operation using runtime axes on shapes without batch.

    Graph merges preserve batch size: add requires matching tensor shapes and
    concat accepts any non-batch axis, including negative PyTorch-style axes.
    """
    name = operation.operation.lower()
    validate_operation(operation)
    if not inputs:
        raise ValueError(f"{name} requires an input tensor")
    first = inputs[0]
    if name in ("add", "concat"):
        if len(inputs) < 2:
            raise ValueError(f"{name} requires at least two input tensors")
        if any(len(shape) != len(first) for shape in inputs):
            raise ValueError(f"{name} shape rank mismatch: {inputs}")
        axis = None
        if name == "concat":
            dim = operation.kwargs.get("dim", 1)
            rank = len(first) + 1
            if type(dim) is not int or not -rank <= dim < rank:
                raise ValueError(f"concat dim {dim!r} is invalid for rank {rank}")
            runtime_axis = dim % rank
            if runtime_axis == 0:
                raise ValueError("concat on the batch axis is not supported")
            axis = runtime_axis - 1
        for shape in inputs[1:]:
            for index, (left, right) in enumerate(zip(first, shape)):
                if index != axis and left != -1 and right != -1 and left != right:
                    raise ValueError(f"{name} shape mismatch: {inputs}")
        output = list(first)
        if axis is not None:
            output[axis] = (
                -1
                if any(shape[axis] == -1 for shape in inputs)
                else sum(shape[axis] for shape in inputs)
            )
        result = tuple(output)
    else:
        if len(inputs) != 1:
            raise ValueError(f"{name} requires one input tensor")
        result = infer_output_shape(operation, first)
    if operation.activation == "residual" and result != first:
        raise ValueError(f"residual shape mismatch: {first} vs {result}")
    return result
