# Examples and execution evidence

| File | Purpose | Data and side effects |
| --- | --- | --- |
| [simple.aur](../examples/simple.aur) | Small CNN, two epochs | 64 FakeData images; no network |
| [transformer.aur](../examples/transformer.aur) | Causal token attention, normalization, warmup/cosine, validation | Synthetic token pairs; no network |
| [mnist.aur](../examples/mnist.aur) | Image classification with validation | Explicit MNIST download when training starts |
| [resnet.aur](../examples/resnet.aur) | Explicit residual graph with early stopping and checkpoints | Explicit CIFAR10 download; checkpoint files |
| [gan.aur](../examples/gan.aur) | Alternating generator/discriminator training | Explicit MNIST download; normalized images; sample tensors/PNG grids and checkpoints |

Start with an offline example from the repository root:

```bash
aurane check examples/simple.aur --json
aurane run examples/simple.aur
aurane run examples/transformer.aur
```

The Transformer example demonstrates compiler/runtime wiring on random token
sequences. Its metrics are not evidence of useful language modeling.

`tests/test_examples_runtime.py` executes all five training paths using four
samples, one epoch and CPU. MNIST/CIFAR constructors are replaced with FakeData;
network downloads and long training are deliberately outside the regression
suite. GAN sample exports and checkpoints go into temporary directories.
Separate tests compare generated numerical results and gradients with PyTorch,
validate metrics, and compare resumed versus uninterrupted training exactly.

For real datasets, review the declared epochs and download paths before running.
Importing generated modules constructs neither datasets nor training jobs.

```bash
aurane compile examples/mnist.aur mnist.py --analyze --validate
python mnist.py
```

For custom data, use a qualified dataset constructor or inject a loader into the
generated training function. Each loader yields `(input, target)`. See the
[language reference](language-reference.md#datasets).
