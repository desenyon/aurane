"""
Profiler for Aurane models.

Provides model profiling capabilities including:
- FLOPs calculation
- Memory estimation
- Layer-wise timing
- Bottleneck detection
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import math

from .ast import (
    AuraneProgram,
    ModelNode,
    LayerOperation,
    ForwardBlock,
)


@dataclass
class LayerProfile:
    """Profile information for a single layer."""

    name: str
    operation: str
    input_shape: tuple
    output_shape: tuple
    flops: int
    params: int
    memory_bytes: int
    percentage_flops: float = 0.0
    percentage_params: float = 0.0


@dataclass
class ModelProfile:
    """Complete profile for a model."""

    model_name: str
    layers: List[LayerProfile] = field(default_factory=list)
    total_flops: int = 0
    total_params: int = 0
    total_memory_bytes: int = 0
    bottleneck_layer: Optional[str] = None
    input_shape: tuple = ()
    output_shape: tuple = ()

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "model": self.model_name,
            "total_flops": self.total_flops,
            "total_flops_readable": self._format_flops(self.total_flops),
            "total_params": self.total_params,
            "total_params_readable": self._format_params(self.total_params),
            "total_memory_mb": self.total_memory_bytes / (1024 * 1024),
            "num_layers": len(self.layers),
            "bottleneck": self.bottleneck_layer,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }

    @staticmethod
    def _format_flops(flops: int) -> str:
        """Format FLOPs in human-readable form."""
        if flops >= 1e12:
            return f"{flops / 1e12:.2f} TFLOPs"
        elif flops >= 1e9:
            return f"{flops / 1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            return f"{flops / 1e6:.2f} MFLOPs"
        elif flops >= 1e3:
            return f"{flops / 1e3:.2f} KFLOPs"
        return f"{flops} FLOPs"

    @staticmethod
    def _format_params(params: int) -> str:
        """Format parameters in human-readable form."""
        if params >= 1e9:
            return f"{params / 1e9:.2f}B"
        elif params >= 1e6:
            return f"{params / 1e6:.2f}M"
        elif params >= 1e3:
            return f"{params / 1e3:.2f}K"
        return str(params)


class ModelProfiler:
    """
    Profiler for Aurane models.

    Calculates:
    - FLOPs (floating point operations)
    - Parameter counts
    - Memory requirements
    - Bottleneck detection
    """

    def __init__(self, model: ModelNode):
        self.model = model
        self.profile = ModelProfile(model_name=model.name)

    def profile_model(self, batch_size: int = 1) -> ModelProfile:
        """
        Profile the model.

        Args:
            batch_size: Batch size for memory calculations.

        Returns:
            ModelProfile with detailed profiling information.
        """
        if not self.model.forward_block:
            return self.profile

        input_shape = self.model.config.get("input_shape", (1, 28, 28))
        if isinstance(input_shape, list):
            input_shape = tuple(input_shape)

        self.profile.input_shape = input_shape
        current_shape = input_shape

        for idx, op in enumerate(self.model.forward_block.operations):
            # Calculate layer metrics
            output_shape = self._calculate_output_shape(op, current_shape)
            flops = self._calculate_flops(op, current_shape, output_shape)
            params = self._calculate_params(op, current_shape)
            memory = self._calculate_memory(output_shape, batch_size)

            layer_profile = LayerProfile(
                name=f"layer_{idx}",
                operation=op.operation,
                input_shape=current_shape,
                output_shape=output_shape,
                flops=flops,
                params=params,
                memory_bytes=memory,
            )

            self.profile.layers.append(layer_profile)
            self.profile.total_flops += flops
            self.profile.total_params += params
            self.profile.total_memory_bytes += memory

            current_shape = output_shape

        self.profile.output_shape = current_shape

        # Calculate percentages and find bottleneck
        self._calculate_percentages()
        self._find_bottleneck()

        return self.profile

    def _calculate_output_shape(self, op: LayerOperation, input_shape: tuple) -> tuple:
        """Calculate output shape for an operation."""
        op_name = op.operation.lower()

        def to_int(val, default: int) -> int:
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

        if op_name == "conv2d":
            if len(input_shape) == 3:
                c, h, w = input_shape
                out_channels = to_int(op.args[0], 32) if op.args else 32
                kernel = to_int(op.kwargs.get("kernel", 3), 3)
                stride = to_int(op.kwargs.get("stride", 1), 1)
                padding = to_int(op.kwargs.get("padding", 0), 0)

                h_out = (h + 2 * padding - kernel) // stride + 1
                w_out = (w + 2 * padding - kernel) // stride + 1
                return (out_channels, h_out, w_out)
            return input_shape

        elif op_name in ("maxpool", "avgpool"):
            if len(input_shape) == 3:
                c, h, w = input_shape
                kernel = to_int(op.args[0], 2) if op.args else 2
                stride = to_int(op.kwargs.get("stride", kernel), kernel)
                return (c, max(1, h // stride), max(1, w // stride))
            return input_shape

        elif op_name == "flatten":
            if len(input_shape) == 3:
                c, h, w = input_shape
                return (c * h * w,)
            return input_shape

        elif op_name in ("dense", "linear"):
            out_features = to_int(op.args[0], 128) if op.args else 128
            return (out_features,)

        elif op_name in ("dropout", "batchnorm", "layer_norm"):
            return input_shape

        elif op_name == "embedding":
            if len(op.args) >= 2:
                embed_dim = to_int(op.args[1], 256)
                seq_len = to_int(input_shape[0], 1) if input_shape else 1
                return (seq_len, embed_dim)
            return input_shape

        elif op_name == "global_avg_pool":
            if len(input_shape) == 3:
                c, h, w = input_shape
                return (c,)
            return input_shape

        return input_shape

    def _calculate_flops(self, op: LayerOperation, input_shape: tuple, output_shape: tuple) -> int:
        """Calculate FLOPs for an operation."""
        op_name = op.operation.lower()

        def to_int(val, default: int) -> int:
            if isinstance(val, int):
                return val
            if isinstance(val, float):
                return int(val)
            return default

        if op_name == "conv2d":
            if len(input_shape) == 3 and len(output_shape) == 3:
                in_channels = input_shape[0]
                out_channels, h_out, w_out = output_shape
                kernel = to_int(op.kwargs.get("kernel", 3), 3)

                # FLOPs = 2 * K^2 * Cin * Cout * Hout * Wout
                flops = 2 * kernel * kernel * in_channels * out_channels * h_out * w_out
                return flops
            return 0

        elif op_name in ("dense", "linear"):
            in_features = input_shape[0] if input_shape else 128
            out_features = output_shape[0] if output_shape else 128
            # FLOPs = 2 * in * out (multiply + add)
            return 2 * in_features * out_features

        elif op_name in ("maxpool", "avgpool"):
            if len(output_shape) == 3:
                c, h, w = output_shape
                kernel = to_int(op.args[0], 2) if op.args else 2
                return c * h * w * kernel * kernel
            return 0

        elif op_name == "batchnorm":
            # Roughly 4 ops per element (normalize + scale + shift)
            if len(input_shape) == 3:
                c, h, w = input_shape
                return 4 * c * h * w
            elif len(input_shape) == 1:
                return 4 * input_shape[0]
            return 0

        elif op_name == "multihead_attention":
            # Attention: O(n^2 * d) for Q, K, V
            dim = to_int(op.kwargs.get("dim", 512), 512)
            seq_len = input_shape[0] if input_shape else 128
            heads = to_int(op.kwargs.get("heads", 8), 8)

            # QKV projection + attention + output projection
            return 4 * seq_len * dim * dim + 2 * seq_len * seq_len * dim

        elif op_name == "embedding":
            # Lookup is essentially free (no FLOPs)
            return 0

        return 0

    def _calculate_params(self, op: LayerOperation, input_shape: tuple) -> int:
        """Calculate parameters for an operation."""
        op_name = op.operation.lower()

        def to_int(val, default: int) -> int:
            if isinstance(val, int):
                return val
            if isinstance(val, float):
                return int(val)
            return default

        if op_name == "conv2d":
            if len(input_shape) == 3:
                in_channels = input_shape[0]
                out_channels = to_int(op.args[0], 32) if op.args else 32
                kernel = to_int(op.kwargs.get("kernel", 3), 3)
                return kernel * kernel * in_channels * out_channels + out_channels
            return 0

        elif op_name in ("dense", "linear"):
            in_features = input_shape[0] if input_shape else 128
            out_features = to_int(op.args[0], 128) if op.args else 128
            return in_features * out_features + out_features

        elif op_name == "batchnorm":
            if len(input_shape) >= 1:
                return 2 * input_shape[0]  # gamma and beta
            return 0

        elif op_name == "layer_norm":
            if len(input_shape) >= 1:
                return 2 * input_shape[-1]
            return 0

        elif op_name == "embedding":
            if len(op.args) >= 2:
                vocab_size = to_int(op.args[0], 0)
                embed_dim = to_int(op.args[1], 0)
                return vocab_size * embed_dim
            return 0

        elif op_name == "multihead_attention":
            dim = to_int(op.kwargs.get("dim", 512), 512)
            # Q, K, V projections + output projection
            return 4 * dim * dim + 4 * dim

        return 0

    def _calculate_memory(self, output_shape: tuple, batch_size: int) -> int:
        """Calculate memory for activations in bytes."""
        if not output_shape:
            return 0

        # Calculate number of elements
        num_elements = batch_size
        for dim in output_shape:
            num_elements *= dim

        # Assume float32 (4 bytes)
        return num_elements * 4

    def _calculate_percentages(self):
        """Calculate percentage of total for each layer."""
        if self.profile.total_flops > 0:
            for layer in self.profile.layers:
                layer.percentage_flops = (layer.flops / self.profile.total_flops) * 100

        if self.profile.total_params > 0:
            for layer in self.profile.layers:
                layer.percentage_params = (layer.params / self.profile.total_params) * 100

    def _find_bottleneck(self):
        """Find the bottleneck layer (highest FLOPs)."""
        if not self.profile.layers:
            return

        max_flops = 0
        bottleneck = None

        for layer in self.profile.layers:
            if layer.flops > max_flops:
                max_flops = layer.flops
                bottleneck = layer.name

        self.profile.bottleneck_layer = bottleneck


def profile_model(model: ModelNode, batch_size: int = 1) -> ModelProfile:
    """
    Profile an Aurane model.

    Args:
        model: The model to profile.
        batch_size: Batch size for memory calculations.

    Returns:
        ModelProfile with detailed profiling information.
    """
    profiler = ModelProfiler(model)
    return profiler.profile_model(batch_size)


def profile_program(program: AuraneProgram, batch_size: int = 1) -> Dict[str, ModelProfile]:
    """
    Profile all models in a program.

    Args:
        program: The Aurane program.
        batch_size: Batch size for memory calculations.

    Returns:
        Dictionary mapping model names to profiles.
    """
    profiles = {}
    for model in program.models:
        profiles[model.name] = profile_model(model, batch_size)
    return profiles


def format_profile(profile: ModelProfile, detailed: bool = False) -> str:
    """Format a model profile as a string."""
    lines = []
    summary = profile.summary()

    lines.append(f"Model: {profile.model_name}")
    lines.append(f"  Input Shape: {profile.input_shape}")
    lines.append(f"  Output Shape: {profile.output_shape}")
    lines.append(f"  Total Parameters: {summary['total_params_readable']}")
    lines.append(f"  Total FLOPs: {summary['total_flops_readable']}")
    lines.append(f"  Memory (batch=1): {summary['total_memory_mb']:.2f} MB")
    lines.append(f"  Bottleneck: {summary['bottleneck']}")

    if detailed and profile.layers:
        lines.append("\n  Layer Details:")
        lines.append("  " + "-" * 70)
        lines.append(
            f"  {'Layer':<12} {'Operation':<15} {'Output Shape':<18} {'FLOPs':<12} {'%':<6}"
        )
        lines.append("  " + "-" * 70)

        for layer in profile.layers:
            lines.append(
                f"  {layer.name:<12} {layer.operation:<15} "
                f"{str(layer.output_shape):<18} "
                f"{ModelProfile._format_flops(layer.flops):<12} "
                f"{layer.percentage_flops:.1f}%"
            )

    return "\n".join(lines)
