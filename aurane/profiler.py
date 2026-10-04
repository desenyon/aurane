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
)
from .shapes import infer_output_shape, calculate_params
from .symbols import resolve_program, resolve_model
from .ir import lower_model
from .dtypes import DTYPE_BYTES


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
    batch_size: int = 1

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
            "batch_size": self.batch_size,
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
        self.model = resolve_model(model)
        self.profile = ModelProfile(model_name=model.name)

    def profile_model(self, batch_size: int = 1) -> ModelProfile:
        """
        Profile the model.

        Args:
            batch_size: Batch size for memory calculations.

        Returns:
            ModelProfile with detailed profiling information.
        """
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.profile = ModelProfile(model_name=self.model.name, batch_size=batch_size)
        if not self.model.forward_block:
            return self.profile

        input_shape = self.model.config.get("input_shape", (1, 28, 28))
        if isinstance(input_shape, list):
            input_shape = tuple(input_shape)

        self.profile.input_shape = input_shape
        graph = lower_model(self.model)
        for index, node in enumerate(graph.nodes):
            assert node.output is not None
            operation = node.to_operation()
            in_shape = node.inputs[0].shape
            output_shape = node.output.shape
            assert in_shape is not None and output_shape is not None
            flops = self._calculate_flops(operation, in_shape, output_shape)
            params = calculate_params(operation, in_shape)
            memory = self._calculate_memory(output_shape, batch_size, node.output.type_hint)
            self.profile.layers.append(
                LayerProfile(
                    name=f"layer_{index}",
                    operation=operation.operation,
                    input_shape=in_shape,
                    output_shape=output_shape,
                    flops=flops,
                    params=params,
                    memory_bytes=memory,
                )
            )
            self.profile.total_flops += flops
            self.profile.total_params += params
            self.profile.total_memory_bytes += memory
        assert graph.outputs[0].shape is not None
        self.profile.output_shape = graph.outputs[0].shape

        # Calculate percentages and find bottleneck
        self._calculate_percentages()
        self._find_bottleneck()

        return self.profile

    def _calculate_output_shape(self, op: LayerOperation, input_shape: tuple) -> tuple:
        """Infer output shape for an operation."""
        return infer_output_shape(op, input_shape)

    def _calculate_flops(self, op: LayerOperation, input_shape: tuple, output_shape: tuple) -> int:
        """Calculate FLOPs for an operation."""
        op_name = op.operation.lower()

        def to_int(val, default: int) -> int:
            if isinstance(val, int):
                return val
            if isinstance(val, float):
                return int(val)
            return default

        if op_name in ("conv1d", "conv2d"):
            kernel = op.kwargs.get("kernel", 3)
            groups = op.kwargs.get("groups", 1)
            return int(
                2
                * kernel ** (len(input_shape) - 1)
                * (input_shape[0] // groups)
                * math.prod(output_shape)
            )

        elif op_name in ("lstm", "gru"):
            # Matrix multiply-add estimate; excludes pointwise gates and bias additions.
            hidden = op.args[0] if op.args else 128
            directions = 2 if op.kwargs.get("bidirectional", False) else 1
            gates = 4 if op_name == "lstm" else 3
            layers = op.kwargs.get("num_layers", 1)
            return int(
                2
                * input_shape[0]
                * sum(
                    directions
                    * gates
                    * hidden
                    * ((input_shape[-1] if index == 0 else directions * hidden) + hidden)
                    for index in range(layers)
                )
            )

        elif op_name in ("dense", "linear"):
            in_features = input_shape[-1] if input_shape else 128
            out_features = output_shape[-1] if output_shape else 128
            # FLOPs = 2 * in * out (multiply + add)
            return int(2 * math.prod(input_shape[:-1]) * in_features * out_features)

        elif op_name in ("maxpool", "avgpool"):
            if len(output_shape) == 3:
                c, h, w = output_shape
                kernel = to_int(op.args[0], 2) if op.args else 2
                return int(c * h * w * kernel * kernel)
            return 0

        elif op_name == "batchnorm":
            # Roughly 4 ops per element (normalize + scale + shift)
            if len(input_shape) == 3:
                c, h, w = input_shape
                return int(4 * c * h * w)
            elif len(input_shape) == 1:
                return int(4 * input_shape[0])
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
        return calculate_params(op, input_shape)

    def _calculate_memory(
        self, output_shape: tuple, batch_size: int, dtype: Optional[str] = None
    ) -> int:
        """Calculate memory for activations in bytes."""
        if not output_shape:
            return 0

        # Calculate number of elements
        num_elements = batch_size
        for dim in output_shape:
            num_elements *= dim

        # Unknown input-only paths retain the documented float32 estimate.
        return num_elements * DTYPE_BYTES.get(dtype or "float32", 4)

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
    for model in resolve_program(program).models:
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
    lines.append(
        f"  Activation memory (batch={profile.batch_size}, inferred dtype; unknown=4 bytes): {summary['total_memory_mb']:.2f} MB"
    )
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
