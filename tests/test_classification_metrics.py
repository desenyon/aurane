"""Non-additive metrics are aggregated over the entire dataset, including ties."""

import pytest

torch = pytest.importorskip("torch")

from aurane.compiler import compile_source


def train(scores, labels, loss, metrics):
    classes = scores.size(-1)
    code = f"""model Net:
    input_shape = ({classes},)
    def forward(x):
        x -> dense({classes}, bias=False)
train Net on injected:
    epochs = 1
    optimizer = sgd(lr=0.0)
    loss = {loss}
    metrics = {metrics}
"""
    namespace = {"__name__": "metrics_test"}
    exec(compile_source(code, disable_cache=True), namespace)
    model = namespace["Net"]()
    with torch.no_grad():
        model.dense1.weight.copy_(torch.eye(classes))
    batches = [(scores[:2], labels[:2]), (scores[2:], labels[2:])]
    return namespace["train_net"](
        model=model, train_loader=batches, validation_loader=batches, test_loader=batches
    )


def pairwise_auc(scores, positive):
    pairs = scores[positive][:, None] - scores[~positive][None, :]
    return ((pairs > 0).float() + (pairs == 0).float() / 2).mean().item()


def test_multiclass_macro_metrics_use_dataset_counts_and_ovr_auc():
    scores = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 3.0],
            [2.0, 0.0, 1.0],
            [1.0, 2.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 1, 1, 2, 2, -100])
    model = train(scores, labels, "cross_entropy", "[precision, recall, f1, auc]")
    expected = {
        "precision": (1 / 3 + 1 + 1) / 3,
        "recall": (1 + 1 / 2 + 1 / 2) / 3,
        "f1": (1 / 2 + 2 / 3 + 2 / 3) / 3,
    }
    probabilities = scores[:5].softmax(-1)
    expected["auc"] = (
        sum(pairwise_auc(probabilities[:, index], labels[:5] == index) for index in range(3)) / 3
    )
    for name, value in expected.items():
        assert model.training_history[0][name] == pytest.approx(value)
        assert model.training_history[0]["val_" + name] == pytest.approx(value)
        assert model.test_metrics[name] == pytest.approx(value)


@pytest.mark.parametrize("loss", ["bce_with_logits", "binary_cross_entropy"])
def test_binary_metrics_threshold_and_tied_auc(loss):
    scores = torch.tensor([[-2.0], [-0.5], [0.0], [0.0], [2.0]])
    if loss == "binary_cross_entropy":
        scores = scores.sigmoid()
    labels = torch.tensor([[0.0], [1.0], [1.0], [0.0], [1.0]])
    model = train(scores, labels, loss, "[accuracy, precision, recall, f1, auc]")
    expected = {
        "accuracy": 3 / 5,
        "precision": 2 / 3,
        "recall": 2 / 3,
        "f1": 2 / 3,
        "auc": pairwise_auc(scores.flatten(), labels.flatten().bool()),
    }
    for name, value in expected.items():
        assert model.test_metrics[name] == pytest.approx(value)


def test_auc_is_explicitly_undefined_when_a_class_is_absent():
    model = train(
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.ones(3, 1),
        "bce_with_logits",
        "[auc, precision, recall]",
    )
    assert model.test_metrics["auc"] is None
    assert model.test_metrics["precision"] == model.test_metrics["recall"] == 1.0
