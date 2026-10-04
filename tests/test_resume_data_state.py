"""Checkpoint data state covers stochastic train and validation pipelines."""

import random
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from tests.test_checkpoints import training_source
from tests.test_training_runtime import module


class StatefulData(torch.utils.data.Dataset):
    def __init__(self):
        self.visits = 0

    def __len__(self):
        return 6

    def __getitem__(self, index):
        self.visits += 1
        value = np.random.random() + random.random() + self.visits / 100 + index / 10
        return torch.full((1, 4, 4), value), index % 3

    def state_dict(self):
        return {"visits": self.visits}

    def load_state_dict(self, state):
        self.visits = state["visits"]


def loader(seed):
    dataset = StatefulData()
    sampler = torch.utils.data.RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))
    return torch.utils.data.DataLoader(
        dataset, batch_size=2, sampler=sampler, generator=torch.Generator().manual_seed(seed + 1)
    )


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def test_resume_restores_validation_numpy_dataset_and_sampler_state(tmp_path):
    seed_all(123)
    full = module(training_source(tmp_path / "full", 3))["train_net"](
        train_loader=loader(4), validation_loader=loader(8)
    )
    seed_all(123)
    module(training_source(tmp_path / "split", 1))["train_net"](
        train_loader=loader(4), validation_loader=loader(8)
    )
    seed_all(987)
    resumed = module(training_source(tmp_path / "split", 3))["train_net"](
        train_loader=loader(70),
        validation_loader=loader(90),
        resume_from=tmp_path / "split/last.pt",
    )
    assert resumed.training_history == full.training_history
    for name, expected in full.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)


def test_checkpoint_requires_compatible_validation_loader(tmp_path):
    namespace = module(training_source(tmp_path, 1))
    namespace["train_net"](train_loader=loader(4), validation_loader=loader(8))
    with pytest.raises(ValueError, match="validation"):
        namespace["train_net"](train_loader=loader(4), resume_from=tmp_path / "last.pt")


@pytest.mark.parametrize("gan", [False, True])
def test_checkpoint_rejects_persistent_workers_before_iteration(tmp_path, gan):
    if gan:
        from tests.test_gan_checkpoints import training_source as gan_source

        namespace = module(gan_source(tmp_path))
        function = namespace["train_gan_generator_discriminator"]
    else:
        function = module(training_source(tmp_path, 1))["train_net"]
    # Workers start only upon iteration, after the boundary check.
    data = torch.utils.data.DataLoader(StatefulData(), num_workers=1, persistent_workers=True)
    with pytest.raises(ValueError, match="persistent_workers"):
        function(train_loader=data)
    assert data.dataset.visits == 0
