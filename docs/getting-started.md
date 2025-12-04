# Getting Started with Aurane

Welcome to Aurane! This guide will help you get up and running quickly.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Install Aurane

```bash
# Clone the repository
git clone https://github.com/yourusername/aurane.git
cd aurane

# Install with all features (recommended)
pip install -e ".[all]"

# Or minimal installation (compiler only)
pip install -e .
```

### Verify Installation

```bash
# Check version and commands
aurane --help

# Should show all available commands
```

## Your First Aurane Program

### Step 1: Create a Simple Model

Create a file named `simple.aur`:

```aur
use torch

model SimpleNet:
    input_shape = (1, 28, 28)
    def forward(x):
        x -> flatten()
          -> dense(128).relu
          -> dense(10)
```

### Step 2: Compile It

```bash
aurane compile simple.aur simple.py
```

You should see:

```
Compiling: simple.aur
[████████████████████] 100% Complete!

╭─────── Compilation Complete ───────╮
│ ✓ Status      Success              │
│ Input         simple.aur            │
│ Output        simple.py             │
│ Output Size   1,234 lines, 987 bytes│
╰────────────────────────────────────╯
```

### Step 3: Inspect the Generated Code

```bash
cat simple.py
```

You'll see clean PyTorch code:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(784, 128)
        self.dense2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x
```

## Building a Complete Training Pipeline

### Step 1: Define Everything

Create `mnist_complete.aur`:

```aur
use torch
use torchvision

experiment MnistBaseline:
    seed = 42
    device = "auto"
    log_interval = 100

dataset mnist_train:
    from torchvision.datasets.MNIST
    root = "./data"
    train = True
    download = True
    batch = 128

dataset mnist_test:
    from torchvision.datasets.MNIST
    root = "./data"
    train = False
    download = True
    batch = 128

model MnistCNN:
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

train MnistCNN on mnist_train:
    validate_on = mnist_test
    loss = cross_entropy
    optimizer = adam(lr=1e-3)
    epochs = 5
    metrics = [accuracy]
```

### Step 2: Compile and Inspect

```bash
# Compile with analysis
aurane compile mnist_complete.aur mnist.py --analyze --show-ast

# Inspect model structure
aurane inspect mnist_complete.aur --verbose
```

### Step 3: Run It

```bash
# Compile and execute
aurane run mnist_complete.aur
```

## Interactive Development

Aurane includes an interactive REPL for experimentation:

```bash
$ aurane interactive

aurane> model TestNet:
....... input_shape = (3, 32, 32)
....... def forward(x):
.......     x -> conv2d(64).relu -> flatten() -> dense(10)

aurane> .compile
✓ Compilation successful!

[Generated PyTorch code displayed]

aurane> .show
[Shows your code buffer]

aurane> .help
[Shows all REPL commands]

aurane> .exit
Goodbye!
```

### REPL Commands

- `.compile` / `.c` - Compile buffer
- `.show` / `.s` - Show buffer
- `.clear` / `.clr` - Clear buffer
- `.save <file>` - Save to file
- `.load <file>` - Load from file
- `.validate` / `.check` - Validate syntax
- `.history` / `.hist` - Show history
- `.help` / `.h` / `.?` - Show help
- `.exit` / `.quit` / `.q` - Exit

## Watch Mode for Live Development

Auto-recompile when your file changes:

```bash
$ aurane watch model.aur output.py

👁  Watching: model.aur
✓ Compiled successfully (1.2s)

[Edit model.aur in another window]

⟳ File changed, recompiling...
✓ Compiled successfully (0.8s)
```

## Inspecting Models

Get detailed architecture analysis:

```bash
# Basic inspection
aurane inspect model.aur

# Detailed with stats
aurane inspect model.aur --verbose --stats

# Export AST to JSON
aurane inspect model.aur --export model.json
```

Example output:

```
Inspecting: model.aur
234 lines • 5,678 bytes

Aurane Program
├── Imports
│   ├── torch
│   └── torchvision
├── Experiments
│   └── MnistBaseline
├── Datasets
│   ├── mnist_train
│   └── mnist_test
└── Models
    └── MnistCNN
        └── forward
            ├── conv2d(32).relu
            ├── maxpool(2)
            ├── conv2d(64).relu
            ├── flatten()
            ├── dense(128).relu
            └── dense(10)

━━━ Model Details ━━━

Layer              Output Shape    Parameters
─────────────────────────────────────────────
Input              (1, 28, 28)              0
Conv2D(32)         (32, 26, 26)           320
MaxPool(2)         (32, 13, 13)             0
Conv2D(64)         (64, 11, 11)        18,496
MaxPool(2)         (64, 5, 5)               0
Flatten()          (1,600)                  0
Dense(128)         (128)              204,928
Dense(10)          (10)                 1,290
─────────────────────────────────────────────
Total Parameters: 225,034
```

## Code Quality Tools

### Format

Auto-format your Aurane code:

```bash
# Format a file
aurane format model.aur

# Format a directory
aurane format examples/

# Check without modifying
aurane format model.aur --check
```

### Lint

Check for potential issues:

```bash
aurane lint model.aur

# Output:
# ✗ 2 error(s):
#   • Model 'MyModel' has no forward operations
#   • Training references undefined dataset 'data'
#
# ⚠ 3 warning(s):
#   • Line 45: Line too long (120 > 100)
#   • Line 67: Inconsistent indentation (6 spaces)
```

## Benchmarking

Measure compilation performance:

```bash
$ aurane benchmark model.aur --iterations 20

Benchmarking: model.aur
Running 20 iterations...

Benchmark Results
─────────────────────────────────────────────
Phase      Mean     Median   Std Dev  Min      Max
─────────────────────────────────────────────
Parse      12.45ms  12.30ms  0.85ms   11.20ms  14.10ms
Compile    45.67ms  45.20ms  2.10ms   42.80ms  50.30ms
Total      58.12ms  57.50ms  2.45ms   54.00ms  64.40ms

File: 234 lines, 5,678 bytes
Throughput: 4,025 lines/sec
```

## Next Steps

Now that you've learned the basics:

1. **Explore Examples** - Check the `examples/` directory for more complex models
2. **Read Language Reference** - Learn all supported layers and syntax in [language-reference.md](language-reference.md)
3. **Master the CLI** - Explore all commands in [cli-commands.md](cli-commands.md)
4. **Build Real Projects** - Try implementing your own models

## Common Patterns

### Multiple Models

```aur
model Encoder:
    input_shape = (3, 224, 224)
    def forward(x):
        x -> conv2d(64).relu -> maxpool(2)

model Decoder:
    input_shape = (64, 56, 56)
    def forward(x):
        x -> conv2d(32).relu -> flatten() -> dense(1000)
```

### Shared Configuration

```aur
experiment CommonSettings:
    seed = 42
    device = "cuda"
    mixed_precision = true

# All training blocks inherit these settings
train Model1 on data1:
    epochs = 10
    
train Model2 on data2:
    epochs = 20
```

### Validation and Testing

```aur
train MyModel on train_data:
    validate_on = val_data
    test_on = test_data
    metrics = [accuracy, precision, recall]
    early_stopping = true
    patience = 10
```

## Tips and Best Practices

1. **Use Meaningful Names** - Make your models self-documenting
2. **Start Simple** - Begin with basic models and add complexity gradually
3. **Inspect Often** - Use `aurane inspect` to verify architecture
4. **Watch Mode** - Use during development for instant feedback
5. **Validate Early** - Run `aurane lint` before compiling
6. **Format Consistently** - Use `aurane format` to maintain style

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'aurane'`

**Solution**: Make sure you installed with `pip install -e .`

---

**Issue**: `Command not found: aurane`

**Solution**: Check your Python scripts directory is in PATH, or use `python -m aurane.cli`

---

**Issue**: Compilation fails with parse error

**Solution**: Run `aurane lint model.aur` to identify syntax issues

---

**Issue**: Generated code has wrong shapes

**Solution**: Specify `input_shape` explicitly in your model

## Getting Help

- **Documentation**: Check other docs in `docs/`
- **Examples**: Browse `examples/` directory
- **CLI Help**: Run `aurane <command> --help`
- **Issues**: Report bugs on GitHub

## What's Next?

- [Language Reference](language-reference.md) - Complete syntax guide
- [CLI Commands](cli-commands.md) - All command details
- [Examples Guide](examples.md) - Walkthrough of example models

Happy coding with Aurane! 🚀
