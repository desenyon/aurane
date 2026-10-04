"""Check and compile must agree about unsupported or invalid configuration."""

import pytest

from aurane.compiler import CompilationError, compile_source
from aurane.parser import parse_aurane
from aurane.semantic_analyzer import analyze_semantics
from aurane.type_checker import check_types

MODEL = """dataset data:

model Net:
    input_shape = (3,)
    def forward(x):
        x -> dense(3)
"""


@pytest.mark.parametrize(
    "setting",
    [
        "loss = unknown",
        "loss = cross_entropy(label_smoothing=2)",
        "loss = cross_entropy(ignore_index=0.5)",
        "loss = huber(delta=0)",
        "optimizer = adam(learning_rate=0.1)",
        "optimizer = adam(betas=(0.9, 1))",
        "optimizer = sgd(momentum=-1)",
        "optimizer = sgd(nesterov=True)",
        "optimizer = adam(lr=-0.1)",
        "optimizer = rmsprop(alpha=2)",
        "scheduler = step_lr",
        "scheduler = step_lr(step_size=0)",
        "scheduler = exponential_lr(gamma=-0.1)",
        "scheduler = cosine_annealing(T_max=0)",
        "scheduler = reduce_lr_on_plateau(factor=1.2)",
        "scheduler = step_lr(step_size=2, bogus=True)",
        "scheduler = warmup_cosine(warmup_steps=4, max_steps=2)",
        "metrics = [unknown]",
        "loss = mse\n    metrics = [accuracy]",
        "epochs = False",
        "gradient_clip = False",
        "checkpoint_dir = 3",
        "callbacks = [unknown]",
        "callbacks = [early_stopping(patience=0)]",
    ],
)
def test_invalid_training_settings_fail_all_entry_points(setting):
    code = MODEL + "train Net on data:\n    " + setting + "\n"
    program = parse_aurane(code)
    assert analyze_semantics(program).has_errors, setting
    assert check_types(program).has_errors, setting
    with pytest.raises(CompilationError):
        compile_source(code, disable_cache=True)


@pytest.mark.parametrize(
    "prefix",
    [
        "experiment A:\n    seed = 1\nexperiment B:\n    seed = 2\n",
        "experiment A:\n    sed = 1\n",
        "experiment A:\n    seed = True\n",
        "experiment A:\n    device = 'nonsense'\n",
        "experiment A:\n    backend = 'jax'\n",
        "model nn:\n    input_shape = (3,)\n    def forward(x):\n        x -> dense(3)\n",
        "model range:\n    input_shape = (3,)\n    def forward(x):\n        x -> dense(3)\n",
        "dataset data:\n\n",
        "model Net:\n    input_shape = (3,)\n",
    ],
)
def test_ambiguous_or_ignored_program_configuration_is_rejected(prefix):
    code = prefix + MODEL
    assert analyze_semantics(parse_aurane(code)).has_errors
    with pytest.raises(CompilationError):
        compile_source(code, disable_cache=True)


@pytest.mark.parametrize(
    "prefix", ["use math as torch\n", "use math as nn\n", "use math as range\n"]
)
def test_imports_cannot_replace_generated_runtime_bindings(prefix):
    code = prefix + MODEL
    assert analyze_semantics(parse_aurane(code)).has_errors
    assert check_types(parse_aurane(code)).has_errors
    with pytest.raises(CompilationError, match="binding"):
        compile_source(code, disable_cache=True)


def test_padding_without_attention_fails_all_entry_points():
    code = MODEL.replace("    input_shape", "    input_padding_idx = 0\n    input_shape")
    assert analyze_semantics(parse_aurane(code)).has_errors
    assert check_types(parse_aurane(code)).has_errors
    with pytest.raises(CompilationError, match="attention"):
        compile_source(code, disable_cache=True)


@pytest.mark.parametrize("change", ["shape", "dtype", "noise", "latent"])
def test_gan_model_boundary_is_checked_before_execution(change):
    code = """dataset data:
model Generator:
    input_shape = (4,)
    def forward(x):
        x -> dense(3)
model Discriminator:
    input_shape = (3,)
    def forward(x):
        x -> dense(1).sigmoid
train_gan Generator and Discriminator on data:
    epochs = 1
"""
    if change == "shape":
        code = code.replace("dense(3)", "dense(2)")
    elif change == "dtype":
        code = code.replace("model Generator:", "model Generator:\n    input_dtype = 'float64'")
    elif change == "noise":
        code = code.replace(
            "model Generator:", "model Generator:\n    input_dtype = 'int64'"
        ).replace("dense(3)", "embedding(10, 3)")
    else:
        code += "    latent_dim = 5\n"
    assert analyze_semantics(parse_aurane(code)).has_errors
    assert check_types(parse_aurane(code)).has_errors
    with pytest.raises(CompilationError):
        compile_source(code, disable_cache=True)
