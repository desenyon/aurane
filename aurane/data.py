"""Small deterministic offline data for runnable examples, not model evaluation."""


class SyntheticTokens:
    """Random token sequences paired with their next tokens.

    Uses a private generator so construction does not perturb model/dropout RNG.
    PyTorch is imported only when the dataset is instantiated.
    """

    def __init__(self, size=128, sequence_length=16, vocab_size=64, seed=42):
        for name, value in (
            ("size", size),
            ("sequence_length", sequence_length),
            ("vocab_size", vocab_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        import torch

        self.tokens = torch.randint(
            vocab_size, (size, sequence_length + 1), generator=torch.Generator().manual_seed(seed)
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        row = self.tokens[index]
        return row[:-1], row[1:]
