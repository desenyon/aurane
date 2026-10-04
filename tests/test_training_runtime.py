"""Execute generated training on tiny offline data, including failure paths."""

import builtins
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from aurane.compiler import CompilationError, compile_source

SOURCE = """dataset samples:
    from torchvision.datasets.FakeData
    size = 9
    image_size = (1, 4, 4)
    num_classes = 3
    batch = 4
    shuffle = False

model Net:
    input_shape = (1, 4, 4)
    def forward(x):
        x -> flatten()
          -> dense(3)

train Net on samples:
    epochs = 1
    optimizer = sgd(lr=0.01)
    loss = cross_entropy
"""


def module(source=SOURCE):
    namespace = {"__name__": "generated_test"}
    exec(compile_source(source, disable_cache=True), namespace)
    return namespace


def test_model_only_import_does_not_require_torchvision(monkeypatch):
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name.startswith("torchvision"):
            raise ImportError("torchvision is intentionally unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    namespace = module(SOURCE[SOURCE.index("model Net:") : SOURCE.index("train Net")])
    assert namespace["Net"]()(torch.zeros(2, 1, 4, 4)).shape == (2, 3)


def test_import_does_not_construct_datasets(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dataset was constructed while importing model definitions")

    monkeypatch.setattr(torchvision.datasets, "FakeData", forbidden)
    assert "Net" in module()


@pytest.mark.parametrize("amp", [False, True])
def test_training_without_experiment_updates_weights_and_clips_gradients(monkeypatch, amp):
    source = SOURCE + f"    gradient_clip = 0.1\n    mixed_precision = {amp}\n"
    namespace = module(source)
    model = namespace["Net"]()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    norms = []
    original = torch.nn.utils.clip_grad_norm_

    def clip(parameters, max_norm, *args, **kwargs):
        norms.append(max_norm)
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    trained = namespace["train_net"](model=model)
    assert trained is model
    assert any(not torch.equal(initial[name], value) for name, value in model.state_dict().items())
    assert norms == [0.1, 0.1, 0.1]
    assert math.isfinite(model.training_history[0]["loss"])


def test_epoch_loss_is_sample_weighted_and_plateau_scheduler_receives_it():
    namespace = module(
        SOURCE.replace("lr=0.01", "lr=0.0") + "    scheduler = reduce_lr_on_plateau(patience=1)\n"
    )
    model = namespace["Net"]()
    loader = namespace["_make_samples_loader"]()
    expected = (
        sum(
            torch.nn.functional.cross_entropy(model(x), y, reduction="sum").item()
            for x, y in loader
        )
        / 9
    )
    trained = namespace["train_net"](model=model, train_loader=loader)
    assert trained.training_history[0]["loss"] == pytest.approx(expected, rel=1e-6)


def test_empty_training_data_has_explicit_error():
    namespace = module()
    with pytest.raises(ValueError, match="(?i)empty.*train|train.*empty"):
        namespace["train_net"](train_loader=[])


@pytest.mark.parametrize(
    "setting",
    ["loss = not_a_loss", "optimizer = not_an_optimizer(lr=0.1)", "scheduler = not_a_scheduler()"],
)
def test_unknown_training_implementations_fail_compilation(setting):
    source = (
        SOURCE.replace("    loss = cross_entropy\n", "").replace(
            "    optimizer = sgd(lr=0.01)\n", ""
        )
        + f"    {setting}\n"
    )
    with pytest.raises(CompilationError, match="(?i)unsupported|unknown"):
        compile_source(source, disable_cache=True)


def test_init_starter_completes_check_compile_and_offline_training(tmp_path):
    from tests.test_cli_contracts import cli

    project = tmp_path / "starter"
    initialized = cli("init", project)
    assert initialized.returncode == 0, initialized.stdout + initialized.stderr
    source = project / "src/main.aur"
    assert cli("check", source).returncode == 0
    namespace = module(source.read_text())
    trained = namespace["train_mymodel"]()
    assert math.isfinite(trained.training_history[-1]["loss"])


def test_classification_metrics_and_validation_match_native_calculations():
    namespace = module(
        SOURCE.replace("lr=0.01", "lr=0.0")
        + "    metrics = [accuracy, top2_accuracy, perplexity]\n"
    )
    model = namespace["Net"]()
    loader = namespace["_make_samples_loader"]()
    batches = list(loader)
    scores = torch.cat([model(x).detach() for x, _ in batches])
    labels = torch.cat([y for _, y in batches])
    expected_loss = torch.nn.functional.cross_entropy(scores, labels).item()
    trained = namespace["train_net"](model=model, train_loader=batches, validation_loader=batches)
    record = trained.training_history[0]
    for prefix in ("", "val_"):
        assert record[prefix + "accuracy"] == pytest.approx(
            (scores.argmax(-1) == labels).float().mean().item()
        )
        assert record[prefix + "top2_accuracy"] == pytest.approx(
            (scores.topk(2).indices == labels[:, None]).any(-1).float().mean().item()
        )
        assert record[prefix + "perplexity"] == pytest.approx(math.exp(expected_loss), rel=1e-6)
        assert record[prefix + "loss"] == pytest.approx(expected_loss, rel=1e-6)


def test_regression_metrics_do_not_assume_classification_targets():
    source = (
        SOURCE.replace("cross_entropy", "mse").replace("lr=0.01", "lr=0.0")
        + "    metrics = [mse, mae]\n"
    )
    namespace = module(source)
    model = namespace["Net"]()
    x, y = torch.randn(5, 1, 4, 4), torch.randn(5, 3)
    error = model(x).detach() - y
    loader = [(x[:4], y[:4]), (x[4:], y[4:])]
    record = namespace["train_net"](
        model=model, train_loader=loader, validation_loader=loader
    ).training_history[0]
    for prefix in ("", "val_"):
        assert record[prefix + "mse"] == pytest.approx(error.square().mean().item())
        assert record[prefix + "mae"] == pytest.approx(error.abs().mean().item())


def test_token_loss_and_accuracy_ignore_padding_and_use_last_class_axis():
    source = """model Net:
    input_shape = (4,)
    def forward(x):
        x -> embedding(7, 5) -> dense(7)
train Net on injected:
    epochs = 1
    optimizer = sgd(lr=0.0)
    loss = cross_entropy(ignore_index=-100)
    metrics = [accuracy]
"""
    namespace = module(source)
    model = namespace["Net"]()
    x = torch.tensor([[1, 2, 3, 4], [5, 4, 3, 2]])
    y = torch.tensor([[2, 3, -100, -100], [4, 3, 2, 1]])
    output = model(x).detach()
    valid = y != -100
    loss = torch.nn.functional.cross_entropy(output.reshape(-1, 7), y.reshape(-1)).item()
    loader = [(x[:1], y[:1]), (x[1:], y[1:])]
    record = namespace["train_net"](model=model, train_loader=loader).training_history[0]
    assert record["loss"] == pytest.approx(loss)
    assert record["accuracy"] == pytest.approx(
        (output.argmax(-1)[valid] == y[valid]).float().mean().item()
    )


def test_empty_validation_data_has_explicit_error():
    namespace = module()
    with pytest.raises(ValueError, match="(?i)validation.*empty"):
        namespace["train_net"](validation_loader=[])


@pytest.mark.parametrize("metrics", ["[unknown_metric]", "[mae]", '"accuracy"'])
def test_unsupported_or_incompatible_metrics_fail(metrics):
    with pytest.raises(CompilationError, match="(?i)metric"):
        compile_source(SOURCE + f"    metrics = {metrics}\n", disable_cache=True)


def test_warmup_cosine_steps_per_update_and_reaches_decay(monkeypatch):
    namespace = module(
        SOURCE.replace("lr=0.01", "lr=0.1")
        + "    scheduler = warmup_cosine(warmup_steps=2, max_steps=4)\n"
    )
    learning_rates = []
    original = torch.optim.SGD.step

    def step(optimizer, *args, **kwargs):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", step)
    trained = namespace["train_net"]()
    assert learning_rates == pytest.approx([0.0, 0.05, 0.1])
    assert trained.training_history[0]["lr"] == pytest.approx(0.05)
