"""Declared and inferred dtypes agree between lowering, checks and execution."""

import json
import pytest

from aurane.compiler import CompilationError, compile_source
from aurane.parser import parse_aurane
from aurane.type_checker import check_types, TensorType
from tests.test_cli_contracts import cli


def source(dtype, chain="dense(3)", shape=(4,)):
    return f"model Net:\n    input_shape = {shape!r}\n    input_dtype = '{dtype}'\n    def forward(x):\n        x -> {chain}\n"


@pytest.mark.parametrize(
    "dtype,chain,shape",
    [
        ("float32", "embedding(10, 3)", (4,)),
        ("int64", "dense(3)", (4,)),
        ("float64", "embedding(10, 3)", (4,)),
        ("unknown_type", "dense(3)", (4,)),
        ("int64", "reshape(2, 2).softmax", (4,)),
        ("int64", "embedding(10, 4) -> embedding(10, 2)", (4,)),
    ],
)
def test_invalid_dtype_paths_fail_statically(dtype, chain, shape):
    code = source(dtype, chain, shape)
    result = check_types(parse_aurane(code))
    assert result.has_errors
    assert any("dtype" in item.message.lower() for item in result.errors)
    with pytest.raises(CompilationError, match="(?i)dtype"):
        compile_source(code, disable_cache=True)


def test_ir_json_and_type_report_preserve_dtype_transitions(tmp_path):
    code = source("int64", "embedding(10, 4) -> dense(3)")
    path = tmp_path / "tokens.aur"
    path.write_text(code)
    result = cli("ir", path, "--format", "json")
    graph = json.loads(result.stdout)["models"][0]["ir"]
    assert graph["inputs"][0]["type_hint"] == "int64"
    assert graph["nodes"][0]["output"]["type_hint"] == "float32"
    assert graph["outputs"][0]["type_hint"] == "float32"
    types = check_types(parse_aurane(code)).inferred_types["Net"]
    assert types["input"].dtype == "int64"
    assert types["output"].dtype == "float32"


def test_tensor_compatibility_checks_known_dtypes():
    assert not TensorType(shape=(4,), dtype="int64").is_compatible(
        TensorType(shape=(4,), dtype="float32")
    )


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_float_dtype_controls_parameters_output_and_profile_memory(dtype):
    torch = pytest.importorskip("torch")
    from tests.test_runtime import load_model
    from aurane.profiler import profile_model

    code = source(dtype)
    model = load_model(code)
    inputs = torch.randn(2, 4, dtype=getattr(torch, dtype), requires_grad=True)
    output = model(inputs)
    assert output.dtype == getattr(torch, dtype)
    assert all(parameter.dtype == output.dtype for parameter in model.parameters())
    output.sum().backward()
    assert inputs.grad.dtype == inputs.dtype
    report = profile_model(parse_aurane(code).models[0], batch_size=2)
    assert report.total_memory_bytes == output.numel() * output.element_size()


def test_declared_integer_dtype_is_checked_at_model_entry():
    torch = pytest.importorskip("torch")
    from tests.test_runtime import load_model

    model = load_model(source("int64", "embedding(10, 3)"))
    assert model(torch.zeros(2, 4, dtype=torch.int64)).dtype == torch.float32
    with pytest.raises(ValueError, match="input_dtype"):
        model(torch.zeros(2, 4, dtype=torch.int32))


def test_float64_weighted_training_and_gan_noise():
    torch = pytest.importorskip("torch")
    from tests.test_training_runtime import SOURCE, module
    from tests.test_gan_runtime import SOURCE as GAN_SOURCE

    code = SOURCE.replace("    input_shape", "    input_dtype = 'float64'\n    input_shape")
    code = code.replace("loss = cross_entropy", "loss = cross_entropy(weight=[1, 2, 3])")
    trained = module(code)["train_net"](
        train_loader=[(torch.randn(3, 1, 4, 4, dtype=torch.float64), torch.tensor([0, 1, 2]))]
    )
    assert next(trained.parameters()).dtype == torch.float64
    code = GAN_SOURCE.replace("    input_shape", "    input_dtype = 'float64'\n    input_shape")
    generator, discriminator = module(code)["train_gan_generator_discriminator"](
        train_loader=[(torch.rand(3, 1, 2, 2, dtype=torch.float64), torch.zeros(3))]
    )
    assert (
        next(generator.parameters()).dtype
        == next(discriminator.parameters()).dtype
        == torch.float64
    )
