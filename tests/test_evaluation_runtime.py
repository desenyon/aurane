"""Held-out evaluation and weighted objectives use dataset-level denominators."""

import pytest

torch = pytest.importorskip("torch")

from aurane.compiler import CompilationError, compile_source

SOURCE = """model Net:
    input_shape = (3,)
    def forward(x):
        x -> dense(3)
train Net on injected:
    epochs = 2
    loss = cross_entropy(weight=[1.0, 4.0, 2.0], ignore_index=-100)
    optimizer = sgd(lr=0.0)
    metrics = [accuracy, perplexity]
"""


def module(source=SOURCE):
    namespace = {"__name__": "evaluation_test"}
    exec(compile_source(source, disable_cache=True), namespace)
    return namespace


def test_weighted_classification_and_test_metrics_match_whole_dataset(monkeypatch):
    namespace = module()
    model = namespace["Net"]()
    x = torch.randn(7, 3)
    y = torch.tensor([0, 1, -100, 2, 1, 0, 2])
    with torch.no_grad():
        output = model(x)
        expected_loss = torch.nn.functional.cross_entropy(
            output, y, weight=torch.tensor([1.0, 4.0, 2.0]), ignore_index=-100
        ).item()
        valid = y != -100
        expected_accuracy = (output.argmax(-1)[valid] == y[valid]).float().mean().item()
    batches = [(x[:2], y[:2]), (x[2:5], y[2:5]), (x[5:], y[5:])]
    visits = []

    class HeldOut:
        def __iter__(self):
            visits.append(len(model.training_history))
            assert not model.training
            assert not torch.is_grad_enabled()
            return iter(batches)

    before = {name: value.clone() for name, value in model.state_dict().items()}
    trained = namespace["train_net"](
        model=model, train_loader=batches, validation_loader=batches, test_loader=HeldOut()
    )
    assert visits == [2]
    for record in trained.training_history:
        assert record["loss"] == pytest.approx(expected_loss)
        assert record["val_loss"] == pytest.approx(expected_loss)
        assert record["accuracy"] == pytest.approx(expected_accuracy)
        assert record["val_accuracy"] == pytest.approx(expected_accuracy)
    assert trained.test_metrics["loss"] == pytest.approx(expected_loss)
    assert trained.test_metrics["accuracy"] == pytest.approx(expected_accuracy)
    assert trained.test_metrics["perplexity"] == pytest.approx(
        torch.exp(torch.tensor(expected_loss)).item()
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_test_on_uses_declared_factory_once_after_early_stopping():
    source = SOURCE.replace(
        "    loss = cross_entropy(weight=[1.0, 4.0, 2.0], ignore_index=-100)",
        "    loss = cross_entropy",
    )
    source += "    test_on = held_out\n    early_stopping = True\n    patience = 1\n"
    namespace = module(source)
    calls = []
    batch = [(torch.randn(2, 3), torch.tensor([0, 1]))]
    namespace["_make_held_out_loader"] = lambda: calls.append("test") or batch
    model = namespace["train_net"](train_loader=batch)
    assert calls == ["test"]
    assert len(model.training_history) == 2
    assert set(model.test_metrics) == {"loss", "accuracy", "perplexity"}


def test_empty_test_loader_fails_explicitly():
    namespace = module(SOURCE.replace("weight=[1.0, 4.0, 2.0], ", ""))
    with pytest.raises(ValueError, match="Test data is empty"):
        namespace["train_net"](
            train_loader=[(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long))], test_loader=[]
        )


def test_regression_rejects_accidental_target_broadcasting():
    code = SOURCE.replace(
        "cross_entropy(weight=[1.0, 4.0, 2.0], ignore_index=-100)", "mse"
    ).replace("metrics = [accuracy, perplexity]", "metrics = [mse]")
    namespace = module(code)
    with pytest.raises(ValueError, match="shape"):
        namespace["train_net"](train_loader=[(torch.zeros(2, 3), torch.zeros(2, 1))])


@pytest.mark.parametrize(
    "setting", ["epohs = 2", "mixed_precision = 'yes'", "scheduler = nonsense", "scheduler = 9"]
)
def test_unhandled_training_settings_fail_compilation(setting):
    with pytest.raises(CompilationError):
        compile_source(SOURCE + "    " + setting + "\n", disable_cache=True)
