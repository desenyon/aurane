# Getting started

Aurane compiles an indentation-based model language into readable PyTorch.
The compiler can run without PyTorch; executing generated models requires it.
These instructions describe Aurane 3.0.0. See the [changelog](../CHANGELOG.md)
for migration notes from earlier versions.

## Install from this checkout

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[torch,dev]'
aurane --version
```

The runtime extra requires PyTorch 2.4+ and torchvision 0.19+. Use a compatible
pair for your platform. Model-only generated modules do not implicitly import
torchvision. The compiler alone can be installed with `pip install -e .`.

## Run the offline starter

```bash
aurane init demo
cd demo
aurane check src/main.aur --json
aurane compile src/main.aur outputs/main.py --validate
python outputs/main.py
```

The starter trains for one epoch on 32 synthetic images. It downloads nothing.
It demonstrates execution, not useful classification accuracy.
`aurane run src/main.aur` combines compilation and execution, cleaning up its
temporary Python file afterward.

## Define and inspect a model

Save this as `model.aur`:

```aur
width = 8

model Classifier:
    input_shape = (4,)
    def forward(x):
        x -> dense(width).relu
          -> dense(3)
```

`input_shape` excludes batch. This model accepts `(batch, 4)` tensors and returns
`(batch, 3)` logits. Constants resolve before generation; undefined or cyclic
constants fail compilation.

```bash
aurane check model.aur --json
aurane compile model.aur model.py
aurane ir model.aur --format json
aurane profile model.aur --batch-size 8
aurane visualize model.aur --format mermaid
```

Import the generated class like any PyTorch model:

```python
import torch
from model import Classifier

net = Classifier()
logits = net(torch.randn(2, 4))
assert logits.shape == (2, 3)
logits.square().mean().backward()
```

## Training and evaluation

The [simple example](../examples/simple.aur) contains an offline dataset and
training block. Generated modules define loader factories and training
functions; importing them does not construct datasets or start training.
Calling `train_tinynet()` constructs its loaders and returns the trained model.

You can supply your own PyTorch objects:

```python
trained = train_tinynet(train_loader=my_loader, validation_loader=my_validation)
print(trained.training_history)
```

Each loader yields `(input, target)`. Without an experiment block, training uses
CPU. With `device = "auto"`, it selects CUDA when available, otherwise CPU.
`mixed_precision = True` enables CUDA AMP; CPU uses full precision.

## Checkpoints

Add these fields to a training block:

```aur
    checkpoint_dir = "./checkpoints/demo"
    checkpoint_every = 2
    save_best = True
    early_stopping = True
    patience = 3
```

The directory receives `last.pt`, improving `best.pt` snapshots, and numbered
snapshots every two epochs. Files are published atomically. Best loss and early
stopping use validation loss when configured, otherwise training loss.

```python
trained = train_tinynet(resume_from="./checkpoints/demo/last.pt")
```

`epochs` is the total target, including completed epochs. Resume restores model,
optimizer, scheduler, scaler, training history, early-stopping state, Python and
PyTorch RNG, NumPy's global RNG when installed, and train/validation loader and
sampler generators. A changed generated model or incompatible training configuration
is rejected. Exact split/uninterrupted execution is tested with stochastic offline
training and validation data, separate sampler generators and dataset state.

Injected datasets, samplers and batch samplers may implement paired `state_dict()`
and `load_state_dict(state)` methods. State must contain tensors and primitive
containers compatible with `torch.load(weights_only=True)`. Dataset hooks require
`num_workers=0`; a parent dataset cannot capture state in worker copies. A saved
loader, generator or state hook must be present and compatible when resuming.
Old checkpoints without the new data state retain their older restoration scope.

Checkpointed/resumed runs reject `persistent_workers=True` before iteration.
Other private RNGs (including `numpy.random.Generator`) or external data cursors
must be included in the object's state hooks. Checkpoints are epoch boundaries,
not mid-batch snapshots. Custom worker initialization and external data sources
remain the caller's responsibility; no universal external-state replay is claimed.

See the [language reference](language-reference.md), [CLI guide](cli-commands.md),
and [audit/backlog](rework-plan.md) for contracts and remaining limitations.
