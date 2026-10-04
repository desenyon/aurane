"""Tiny GAN execution checks for phase isolation, update ratios and samples."""

import pytest

torch = pytest.importorskip("torch")

from aurane.compiler import CompilationError, compile_source

SOURCE = """model Generator:
    input_shape = (4,)
    def forward(z):
        z -> dense(4).tanh -> reshape(1, 2, 2)
model Discriminator:
    input_shape = (1, 2, 2)
    def forward(x):
        x -> flatten() -> dense(1).sigmoid
train_gan Generator and Discriminator on injected:
    epochs = 1
    generator_optimizer = sgd(lr=0.1)
    discriminator_optimizer = sgd(lr=0.1)
    generator_loss = binary_cross_entropy
    discriminator_loss = binary_cross_entropy
    discriminator_steps = 2
    generator_steps = 1
"""


def module(source):
    namespace = {"__name__": "gan_test"}
    exec(compile_source(source, disable_cache=True), namespace)
    return namespace


@pytest.mark.parametrize("loss", ["binary_cross_entropy", "mse"])
def test_gan_runs_both_phases_and_honors_update_ratios(monkeypatch, loss):
    namespace = module(SOURCE.replace("binary_cross_entropy", loss))
    generator, discriminator = namespace["Generator"](), namespace["Discriminator"]()
    initial_g = [parameter.clone() for parameter in generator.parameters()]
    initial_d = [parameter.clone() for parameter in discriminator.parameters()]
    updates = []
    original = torch.optim.SGD.step

    def step(optimizer, *args, **kwargs):
        parameters = optimizer.param_groups[0]["params"]
        is_generator = parameters[0] is next(generator.parameters())
        updates.append("G" if is_generator else "D")
        if not is_generator:
            assert all(parameter.grad is None for parameter in generator.parameters())
        else:
            assert all(not parameter.requires_grad for parameter in discriminator.parameters())
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", step)
    result = namespace["train_gan_generator_discriminator"](
        train_loader=[(torch.randn(3, 1, 2, 2), torch.zeros(3))],
        generator=generator,
        discriminator=discriminator,
    )
    assert result == (generator, discriminator)
    assert updates == ["D", "D", "G"]
    assert any(not torch.equal(old, new) for old, new in zip(initial_g, generator.parameters()))
    assert any(not torch.equal(old, new) for old, new in zip(initial_d, discriminator.parameters()))
    assert all(parameter.requires_grad for parameter in discriminator.parameters())
    assert generator.training_history[0]["generator_loss"] >= 0


def test_gan_saves_requested_samples(tmp_path):
    source = (
        SOURCE
        + f"    generate_samples_every = 1\n    num_samples = 5\n    save_dir = {str(tmp_path)!r}\n"
    )
    namespace = module(source)
    namespace["train_gan_generator_discriminator"](
        train_loader=[(torch.randn(2, 1, 2, 2), torch.zeros(2))]
    )
    samples = torch.load(tmp_path / "samples_0001.pt", weights_only=True)
    assert samples.shape == (5, 1, 2, 2)
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("setting", ["discriminator_steps = 0", "generator_loss = unknown_loss"])
def test_invalid_gan_settings_fail_before_execution(setting):
    key = setting.split("=", 1)[0].strip()
    source = "\n".join(
        line for line in SOURCE.splitlines() if not line.strip().startswith(key + " =")
    )
    with pytest.raises(CompilationError, match="(?i)steps|loss"):
        compile_source(source + "\n    " + setting + "\n", disable_cache=True)
