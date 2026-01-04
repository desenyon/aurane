# Aurane

**A Modern Domain-Specific Language for Machine Learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

Aurane is a domain-specific language designed for machine learning development. It enables developers to write expressive, clean `.aur` files that compile into production-ready, idiomatic PyTorch code. The language abstracts away boilerplate while maintaining full control over model architecture and training configuration.

## Installation

```bash
# Clone repository
git clone https://github.com/desenyon/aurane.git
cd aurane

# Install with all features
pip install -e ".[all]"
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (for ML features)
- Rich 13.0+ (for CLI interface)

## Quick Start

### Example Model Definition

Create `mnist.aur`:

```aur
use torch
use torchvision

experiment MnistBaseline:
    seed = 42
    device = "auto"

dataset mnist_train:
    from torchvision.datasets.MNIST
    root = "./data"
    train = True
    batch = 128

model MnistNet:
    input_shape = (1, 28, 28)
    def forward(x):
        x -> conv2d(32, kernel=3).relu
          -> maxpool(2)
          -> conv2d(64, kernel=3).relu
          -> maxpool(2)
          -> flatten()
          -> dense(128).relu
          -> dropout(0.5)
          -> dense(10)

train MnistNet on mnist_train:
    loss = cross_entropy
    optimizer = adam(lr=1e-3)
    epochs = 5
```

### Compilation

```bash
# Basic compilation
aurane compile mnist.aur mnist.py

# With analysis and visualization
aurane compile mnist.aur mnist.py --analyze --show-ast

# Auto-recompile on changes
aurane watch mnist.aur mnist.py
```

### Generated Output

Clean, idiomatic PyTorch code:

```python
class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(1, 32, 3)
        self.conv2d2 = nn.Conv2d(32, 64, 3)
        self.dense1 = nn.Linear(1600, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv2d1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2d2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = self.dropout1(x)
        x = self.dense2(x)
        return x
```

## Features

### CLI

Aurane includes a feature-rich command-line interface with:

```bash
# Compile with beautiful progress bars
aurane compile model.aur output.py --analyze

# Inspect model architecture
aurane inspect model.aur --verbose --stats

# Live development with auto-reload
aurane watch model.aur output.py

# Interactive REPL for experimentation
aurane interactive

# Format your Aurane code
aurane format examples/

# Lint for potential issues
aurane lint model.aur

# Benchmark compilation performance
aurane benchmark model.aur
```

### Model Inspection

Obtain detailed insights into model architecture:

```bash
$ aurane inspect examples/resnet.aur --verbose

## Quick Start
```

### Watch Mode

Automatic recompilation on file changes:

```bash
$ aurane watch model.aur output.py
[watching] model.aur
[success] Compiled successfully (1.2s)

[changed] File modified, recompiling...
[success] Compiled successfully (0.8s)
```

### Interactive REPL

Live coding environment for rapid prototyping:

```bash
$ aurane interactive

aurane> model SimpleNet:
....... input_shape = (1, 28, 28)
....... def forward(x):
.......     x -> conv2d(32).relu -> flatten() -> dense(10)

aurane> .compile
[success] Compilation successful
```

## Advanced Examples

### ResNet-Style Architecture

```aur
model ResNetClassifier:
    input_shape = (3, 224, 224)
    def forward(x):
        x -> conv2d(64, kernel=7, stride=2, padding=3).relu
          -> maxpool(3, stride=2)
          -> conv2d(64, kernel=3, padding=1).relu
          -> conv2d(64, kernel=3, padding=1).relu
          -> conv2d(128, kernel=3, stride=2, padding=1).relu
          -> avgpool(7)
          -> flatten()
          -> dense(1000)

train ResNetClassifier on imagenet:
    loss = cross_entropy
    optimizer = adam(lr=1e-3, weight_decay=1e-4)
    scheduler = cosine_annealing(T_max=50)
    epochs = 50
```

### Transformer Model

```aur
model LanguageModel:
    input_shape = (128,)
    vocab_size = 50000
  
    def forward(x):
        x -> embedding(vocab_size, 512)
          -> positional_encoding(max_len=128)
          -> multihead_attention(heads=8)
          -> layer_norm()
          -> dense(2048).gelu
          -> dense(512)
          -> dropout(0.1)
          -> dense(vocab_size)
```

### GAN Architecture

```aur
model Generator:
    input_shape = (100,)
    def forward(z):
        z -> dense(256).relu
          -> batch_norm()
          -> dense(512).relu
          -> dense(784).tanh
          -> reshape(1, 28, 28)

model Discriminator:
    input_shape = (1, 28, 28)
    def forward(x):
        x -> flatten()
          -> dense(512).leaky_relu(0.2)
          -> dropout(0.3)
          -> dense(1).sigmoid
```

More examples in the [`examples/`](examples/) directory.

## CLI Commands

### Core Commands

| Command         | Description                     |
| --------------- | ------------------------------- |
| `compile`     | Compile `.aur` file to Python |
| `inspect`     | Analyze model architecture      |
| `watch`       | Auto-recompile on changes       |
| `interactive` | Start REPL mode                 |
| `format`      | Format Aurane source files      |
| `lint`        | Check for style and errors      |
| `benchmark`   | Measure compilation performance |
| `run`         | Compile and execute             |

See [CLI Reference](docs/cli-commands.md) for detailed usage.

## Language Reference

### Supported Layers

**Convolution**: `conv1d`, `conv2d`, `conv3d`
**Pooling**: `maxpool`, `avgpool`, `adaptive_avgpool`
**Linear**: `dense`/`linear`, `embedding`
**Normalization**: `batch_norm`, `layer_norm`, `group_norm`
**Activation**: `.relu`, `.gelu`, `.leaky_relu`, `.tanh`, `.sigmoid`, `.softmax`
**Regularization**: `dropout`
**Reshaping**: `flatten`, `reshape`

### Configuration Blocks

```aur
# Experiments
experiment MyExperiment:
    seed = 42
    device = "cuda"
    mixed_precision = true

# Datasets
dataset training_data:
    from torchvision.datasets.CIFAR10
    root = "./data"
    train = True
    batch = 256

# Training
train MyModel on training_data:
    validate_on = validation_data
    loss = cross_entropy
    optimizer = adam(lr=1e-3)
    scheduler = cosine_annealing(T_max=50)
    epochs = 100
    early_stopping = true
```

See [Language Reference](docs/language-reference.md) for complete syntax.

## Shape Inference

Aurane automatically infers tensor shapes through your network:

```bash
$ aurane inspect model.aur --verbose

Layer              Output Shape    Parameters
--------------------------------------------
Conv2D(32)         (32, 26, 26)        320
MaxPool(2)         (32, 13, 13)          0
Conv2D(64)         (64, 11, 11)     18,496
Flatten()          (7,744)               0
Dense(128)         (128)           991,360
Dense(10)          (10)              1,290
--------------------------------------------
Total Parameters: 1,011,466
```

## Roadmap

### v0.1.0 (Current)

- [X] Complete PyTorch backend
- [X] Enhanced CLI with 8+ commands
- [X] Model inspection and visualization
- [X] Interactive REPL
- [X] Format and lint tools

### v0.1.5 (Planned)

- [ ] TensorFlow/Keras backend
- [ ] Custom layer definitions
- [ ] Model composition
- [ ] VS Code extension

### v0.2.0 (Future)

- [ ] JAX/Flax backend
- [ ] Distributed training
- [ ] Hyperparameter search
- [ ] Model optimization

## Documentation

Comprehensive documentation is available in the [docs/](docs/) directory:

- [Getting Started](docs/getting-started.md)
- [Language Reference](docs/language-reference.md)
- [CLI Commands](docs/cli-commands.md)
- [Examples](docs/examples.md)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
