"""Declarative transform calls are constructed lazily and preserve nesting."""

import pytest

torch = pytest.importorskip("torch")
vision = pytest.importorskip("torchvision")

from aurane.compiler import CompilationError, compile_source
from tests.test_training_runtime import SOURCE, module


def test_nested_transforms_resolve_constants_and_execute_lazily(monkeypatch):
    calls = []
    original = vision.transforms.Normalize

    def normalize(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(vision.transforms, "Normalize", normalize)
    source = "center = (0.5,)\nspread = (0.5,)\n" + SOURCE.replace(
        "    shuffle = False",
        "    shuffle = False\n    transform = Compose([ToTensor(), Normalize(center, spread)])",
    )
    namespace = module(source)
    assert calls == []
    loader = namespace["_make_samples_loader"]()
    assert calls == [(((0.5,), (0.5,)), {})]
    actual = loader.dataset[0][0]
    baseline = vision.datasets.FakeData(
        size=9, image_size=(1, 4, 4), num_classes=3, transform=vision.transforms.ToTensor()
    )[0][0]
    torch.testing.assert_close(actual, (baseline - 0.5) / 0.5)
    assert namespace["train_net"](train_loader=loader).training_history


def test_transform_list_is_composed_without_passing_it_to_dataset_constructor():
    namespace = module(
        SOURCE.replace(
            "    shuffle = False",
            "    shuffle = False\n    transforms = [Resize((6, 6)), ToTensor()]",
        )
    )
    data, _ = next(iter(namespace["_make_samples_loader"]()))
    assert data.shape == (4, 1, 6, 6)


@pytest.mark.parametrize(
    "setting",
    [
        "transform = 'ToTensor()'",
        "transform = Unknown()",
        "transforms = ToTensor()",
        "transform = Compose([ToTensor(), 3])",
        "transform = ToTensor()\n    transforms = [ToTensor()]",
    ],
)
def test_unsupported_transforms_are_rejected(setting):
    source = SOURCE.replace("    shuffle = False", "    shuffle = False\n    " + setting)
    with pytest.raises(CompilationError, match="(?i)transform"):
        compile_source(source, disable_cache=True)
