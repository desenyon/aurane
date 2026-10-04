"""Run every bundled example with small offline fixtures; never download data."""

from pathlib import Path
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from aurane.codegen_torch import generate_torch_code
from aurane.symbols import parse_resolved
from aurane.ast import ForwardGraphBlock

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.aur")), ids=lambda path: path.stem)
def test_example_training_executes_offline_with_scaled_data(path, tmp_path):
    program = parse_resolved(path.read_text())
    for experiment in program.experiments:
        experiment.config["device"] = "cpu"
    for dataset in program.datasets:
        if dataset.source in ("torchvision.datasets.MNIST", "torchvision.datasets.CIFAR10"):
            shape = (1, 28, 28) if dataset.source.endswith("MNIST") else (3, 32, 32)
            dataset.source = "torchvision.datasets.FakeData"
            dataset.config = {
                key: value
                for key, value in dataset.config.items()
                if key not in ("root", "train", "download")
            }
            dataset.config.update(image_size=shape, num_classes=10)
        dataset.config.update(size=4, batch=2, shuffle=False)
    for train in [*program.trains, *program.train_gans]:
        train.config["epochs"] = 1
        train.config["checkpoint_dir"] = str(tmp_path / "checkpoints")
        if hasattr(train, "generator_name"):
            train.config.update(
                generate_samples_every=1, num_samples=2, save_dir=str(tmp_path / "samples")
            )
    namespace = {"__name__": "example_test"}
    exec(generate_torch_code(program), namespace)
    for train in program.trains:
        model = namespace[f"train_{train.model_name.lower()}"]()
        assert math.isfinite(model.training_history[-1]["loss"])
    for train in program.train_gans:
        generator, discriminator = namespace[
            f"train_gan_{train.generator_name.lower()}_{train.discriminator_name.lower()}"
        ]()
        assert math.isfinite(generator.training_history[-1]["generator_loss"])
        assert math.isfinite(discriminator.training_history[-1]["discriminator_loss"])


def test_resnet_example_contains_real_skip_connections():
    program = parse_resolved((EXAMPLES / "resnet.aur").read_text())
    for model in program.models:
        assert isinstance(model.forward_block, ForwardGraphBlock)
        assert any(node.operation.operation == "add" for node in model.forward_block.nodes)


def test_synthetic_token_data_is_deterministic_and_predicts_next_token():
    from aurane.data import SyntheticTokens

    torch.manual_seed(7)
    before = torch.get_rng_state().clone()
    dataset = SyntheticTokens(size=4, sequence_length=5, vocab_size=13, seed=10)
    assert torch.equal(before, torch.get_rng_state())
    repeated = SyntheticTokens(size=4, sequence_length=5, vocab_size=13, seed=10)
    assert len(dataset) == 4
    tokens, targets = dataset[0]
    assert tokens.shape == targets.shape == (5,)
    torch.testing.assert_close(tokens[1:], targets[:-1])
    torch.testing.assert_close(tokens, repeated[0][0])
    assert tokens.dtype == torch.long
