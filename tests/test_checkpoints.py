"""Resume must reproduce uninterrupted offline training, not just load weights."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tests.test_training_runtime import SOURCE, module
from aurane.compiler import CompilationError, compile_source


def training_source(directory, epochs):
    return (
        SOURCE.replace("epochs = 1", f"epochs = {epochs}")
        .replace("sgd(lr=0.01)", "adam(lr=0.01)")
        .replace("-> dense(3)", "-> dropout(0.25) -> dense(3)")
        .replace("shuffle = False", "shuffle = True")
        + f"    checkpoint_dir = {str(directory)!r}\n"
        + "    checkpoint_every = 1\n    save_best = True\n    scheduler = step_lr(step_size=1, gamma=0.5)\n"
    )


def test_resume_restores_optimizer_scheduler_rng_and_history(tmp_path):
    torch.manual_seed(123)
    uninterrupted = module(training_source(tmp_path / "full", 3))["train_net"]()
    torch.manual_seed(123)
    first = module(training_source(tmp_path / "split", 1))["train_net"]()
    checkpoint = tmp_path / "split/last.pt"
    assert checkpoint.exists()
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["epoch"] == 1
    assert saved["optimizer"]["state"]
    torch.manual_seed(999)  # Resume must override this state after model construction.
    resumed = module(training_source(tmp_path / "split", 3))["train_net"](resume_from=checkpoint)
    for name, expected in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)
    assert resumed.training_history == uninterrupted.training_history
    assert first.training_history == uninterrupted.training_history[:1]
    assert (tmp_path / "split/best.pt").exists()
    assert (tmp_path / "split/epoch_0003.pt").exists()


def test_early_stopping_stops_at_patience_and_keeps_first_best(tmp_path):
    source = SOURCE.replace("epochs = 1", "epochs = 8").replace("lr=0.01", "lr=0.0")
    source += f"    early_stopping = True\n    patience = 2\n    save_best = True\n    checkpoint_dir = {str(tmp_path)!r}\n"
    model = module(source)["train_net"]()
    assert len(model.training_history) == 3
    assert torch.load(tmp_path / "best.pt", weights_only=True)["epoch"] == 1
    assert torch.load(tmp_path / "last.pt", weights_only=True)["epoch"] == 3


def test_interrupted_checkpoint_write_preserves_last_complete_file(tmp_path, monkeypatch):
    namespace = module(training_source(tmp_path, 1))
    namespace["train_net"]()
    checkpoint = tmp_path / "last.pt"
    previous = checkpoint.read_bytes()

    def fail_save(value, stream, *args, **kwargs):
        stream.write(b"partial")
        raise OSError("disk failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk failure"):
        namespace["train_net"]()
    assert checkpoint.read_bytes() == previous
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "setting", ["patience = 0", "checkpoint_every = 0", "callbacks = [unknown_callback]"]
)
def test_invalid_checkpoint_and_callback_options_fail(setting):
    with pytest.raises(CompilationError, match="(?i)patience|checkpoint|callback"):
        compile_source(
            SOURCE + "    early_stopping = True\n" + f"    {setting}\n", disable_cache=True
        )


def test_early_stopping_callback_uses_its_parameters():
    source = SOURCE.replace("epochs = 1", "epochs = 6").replace("lr=0.01", "lr=0.0")
    source += "    callbacks = [early_stopping(patience=1)]\n"
    model = module(source)["train_net"]()
    assert len(model.training_history) == 2


def test_resume_rejects_changed_model_behavior_with_identical_weight_shapes(tmp_path):
    source = training_source(tmp_path, 1)
    module(source)["train_net"]()
    changed = source.replace("-> dense(3)", "-> dense(3).tanh")
    with pytest.raises(ValueError, match="incompatible"):
        module(changed)["train_net"](resume_from=tmp_path / "last.pt")


def test_callback_arguments_resolve_global_constants():
    source = "stop_after = 1\n" + SOURCE.replace("epochs = 1", "epochs = 6").replace(
        "lr=0.01", "lr=0.0"
    )
    source += "    callbacks = [early_stopping(patience=stop_after)]\n"
    assert len(module(source)["train_net"]().training_history) == 2


def test_checkpoint_callback_honors_interval_and_explicit_override(tmp_path):
    for override in (False, True):
        directory = tmp_path / str(override)
        source = SOURCE.replace("epochs = 1", "epochs = 3")
        source += (
            f"    callbacks = [checkpoint(checkpoint_dir={str(directory)!r}, checkpoint_every=2)]\n"
        )
        if override:
            source += "    checkpoint_every = 3\n"
        module(source)["train_net"]()
        expected = "epoch_0003.pt" if override else "epoch_0002.pt"
        assert [path.name for path in directory.glob("epoch_*.pt")] == [expected]
