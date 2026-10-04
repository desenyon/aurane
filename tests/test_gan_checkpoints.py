"""GAN persistence must reproduce both adversaries, optimizer state and samples."""

import pytest

torch = pytest.importorskip("torch")

from aurane.compiler import CompilationError, compile_source
from tests.test_gan_runtime import SOURCE, module


def training_source(directory, epochs=1):
    return (
        SOURCE.replace("epochs = 1", f"epochs = {epochs}")
        .replace("sgd(lr=0.1)", "adam(lr=0.01)")
        .replace("flatten() -> dense(1)", "flatten() -> dropout(0.3) -> dense(1)")
        + f"    checkpoint_dir = {str(directory / 'checkpoints')!r}\n"
        + f"    save_dir = {str(directory / 'samples')!r}\n"
        + "    checkpoint_every = 1\n    generate_samples_every = 1\n    num_samples = 3\n"
    )


def loader():
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.arange(20, dtype=torch.float32).reshape(5, 1, 2, 2) / 20, torch.zeros(5)
        ),
        batch_size=2,
        shuffle=True,
        generator=torch.Generator().manual_seed(90),
    )


def test_gan_resume_matches_uninterrupted_models_history_and_samples(tmp_path):
    torch.manual_seed(42)
    full = module(training_source(tmp_path / "full", 3))["train_gan_generator_discriminator"](
        train_loader=loader()
    )
    torch.manual_seed(42)
    module(training_source(tmp_path / "split"))["train_gan_generator_discriminator"](
        train_loader=loader()
    )
    path = tmp_path / "split/checkpoints/last.pt"
    saved = torch.load(path, weights_only=True)
    assert saved["epoch"] == 1
    assert saved["optimizer_g"]["state"] and saved["optimizer_d"]["state"]
    torch.manual_seed(777)
    resumed = module(training_source(tmp_path / "split", 3))["train_gan_generator_discriminator"](
        train_loader=loader(), resume_from=path
    )
    for actual, expected in zip(resumed, full):
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
        assert actual.training_history == expected.training_history
    assert (tmp_path / "split/checkpoints/epoch_0003.pt").exists()
    torch.testing.assert_close(
        torch.load(tmp_path / "split/samples/samples_0003.pt", weights_only=True),
        torch.load(tmp_path / "full/samples/samples_0003.pt", weights_only=True),
        rtol=0,
        atol=0,
    )


def test_gan_checkpoint_failure_preserves_previous_file(tmp_path, monkeypatch):
    namespace = module(training_source(tmp_path).replace("    generate_samples_every = 1\n", ""))
    namespace["train_gan_generator_discriminator"](train_loader=loader())
    path = tmp_path / "checkpoints/last.pt"
    before = path.read_bytes()

    def fail(value, stream, *args, **kwargs):
        stream.write(b"partial")
        raise OSError("disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        namespace["train_gan_generator_discriminator"](train_loader=loader())
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".*.tmp"))


def test_gan_rejects_changed_resume_contract(tmp_path):
    code = training_source(tmp_path)
    module(code)["train_gan_generator_discriminator"](train_loader=loader())
    changed = code.replace("generator_steps = 1", "generator_steps = 2")
    with pytest.raises(ValueError, match="incompatible"):
        module(changed)["train_gan_generator_discriminator"](
            train_loader=loader(), resume_from=tmp_path / "checkpoints/last.pt"
        )


def test_gan_exports_decodable_image_grid_and_tensor(tmp_path):
    pytest.importorskip("torchvision")
    from PIL import Image

    namespace = module(training_source(tmp_path) + "    sample_format = 'both'\n")
    namespace["train_gan_generator_discriminator"](train_loader=loader())
    path = tmp_path / "samples/samples_0001.png"
    with Image.open(path) as grid:
        grid.load()
        assert grid.format == "PNG"
        assert grid.width > 2 and grid.height > 2
    assert (tmp_path / "samples/samples_0001.pt").exists()


def test_gan_metric_selection_is_honored():
    namespace = module(SOURCE + "    metrics = [generator_loss]\n")
    generator, _ = namespace["train_gan_generator_discriminator"](train_loader=loader())
    assert set(generator.training_history[0]) == {"epoch", "generator_loss"}


@pytest.mark.parametrize(
    "setting",
    ["checkpoint_every = 0", "sample_format = 'jpg'", "metrics = [accuracy]", "epohs = 4"],
)
def test_invalid_gan_configuration_is_rejected(setting):
    with pytest.raises(CompilationError):
        compile_source(SOURCE + "    " + setting + "\n", disable_cache=True)
