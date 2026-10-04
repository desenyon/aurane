"""Dtype rules for the compiler's supported tensor operations."""

from .diagnostics import at_config

FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}
INTEGER_DTYPES = {"int32", "int64"}
DTYPE_BYTES = {
    "bool": 1,
    "int32": 4,
    "int64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}
PARAMETER_OPERATIONS = {
    "conv1d",
    "conv2d",
    "dense",
    "linear",
    "batchnorm",
    "batch_norm",
    "layer_norm",
    "layernorm",
    "lstm",
    "gru",
    "multihead_attention",
    "positional_encoding",
}


def model_dtypes(model):
    declared = model.config.get("input_dtype")
    if "input_dtype" in model.config and (
        not isinstance(declared, str) or declared not in DTYPE_BYTES
    ):
        with at_config(model, "input_dtype"):
            raise ValueError(f"Unsupported input_dtype: {declared!r}")
    return declared, declared if declared in FLOAT_DTYPES else "float32"


def infer_dtype(operation, inputs, parameter_dtype):
    name = operation.operation.lower()
    known = {dtype for dtype in inputs if dtype is not None}
    if len(known) > 1:
        raise ValueError(f"{name} dtype mismatch: {inputs}")
    incoming = next(iter(known), None)
    if name == "embedding":
        if incoming is not None and incoming not in INTEGER_DTYPES:
            raise ValueError(f"embedding requires integer dtype, got {incoming}")
        result = parameter_dtype
    elif name in PARAMETER_OPERATIONS:
        if incoming is not None and incoming != parameter_dtype:
            raise ValueError(f"{name} requires dtype {parameter_dtype}, got {incoming}")
        result = parameter_dtype
    elif name in ("flatten", "reshape", "add", "concat"):
        result = incoming
    else:
        if incoming is not None and incoming not in FLOAT_DTYPES:
            raise ValueError(f"{name} requires floating dtype, got {incoming}")
        result = incoming
    if operation.activation == "residual":
        if incoming is not None and incoming != result:
            raise ValueError(f"residual dtype mismatch: {incoming} vs {result}")
    elif operation.activation and result is not None and result not in FLOAT_DTYPES:
        raise ValueError(f"{operation.activation} requires floating dtype, got {result}")
    return result
