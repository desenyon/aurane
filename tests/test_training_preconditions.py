"""Fail before data IO for untrainable models; document experiment seed scope."""

import random
import sys

import pytest

torch = pytest.importorskip("torch")

from aurane.compiler import compile_source

STANDARD = """dataset injected:
model Net:
    input_shape = (4,)
    def forward(x):
        x -> dense(2)
train Net on injected:
    epochs = 1
"""


def module(source):
    namespace = {"__name__": "generated_test"}
    exec(compile_source(source, disable_cache=True), namespace)
    return namespace


@pytest.mark.parametrize("frozen", [False, True])
def test_training_rejects_untrainable_model_before_loader_construction(frozen):
    namespace = module(STANDARD)
    model = torch.nn.Linear(4, 2).requires_grad_(False) if frozen else torch.nn.Identity()
    with pytest.raises(ValueError, match="Net.*trainable parameters"):
        namespace["train_net"](model=model)


@pytest.mark.parametrize("role", ["generator", "discriminator"])
@pytest.mark.parametrize("frozen", [False, True])
def test_gan_rejects_each_untrainable_role_before_loader_construction(role, frozen):
    from tests.test_gan_runtime import SOURCE

    namespace = module(SOURCE)
    model = torch.nn.Linear(4, 4).requires_grad_(False) if frozen else torch.nn.Identity()
    with pytest.raises(ValueError, match=f"{role}.*trainable parameters"):
        namespace["train_gan_generator_discriminator"](**{role: model})


@pytest.mark.parametrize("seed", [37, 2**64 - 1])
def test_experiment_seed_covers_python_numpy_and_torch(seed):
    np = pytest.importorskip("numpy")
    source = f'experiment E:\n    seed = {seed}\n    device = "cpu"\n'
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    try:
        module(source)
        first = random.random(), float(np.random.rand()), torch.rand(3)
        module(source)
        second = random.random(), float(np.random.rand()), torch.rand(3)
        assert first[:2] == second[:2]
        assert torch.equal(first[2], second[2])
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def test_seeded_program_does_not_require_numpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)
    namespace = module('experiment E:\n    seed = 37\n    device = "cpu"\n')
    assert namespace["device"].type == "cpu"
