# Aurane Language Reference

Complete syntax and semantics reference for the Aurane ML DSL.

## Table of Contents

- [Program Structure](#program-structure)
- [Imports](#imports)
- [Experiments](#experiments)
- [Datasets](#datasets)
- [Models](#models)
- [Training](#training)
- [Layers](#layers)
- [Activations](#activations)
- [Data Types](#data-types)
- [Comments](#comments)

---

## Program Structure

An Aurane program consists of top-level declarations in any order:

```aur
use <module>

experiment <name>:
    <config>

dataset <name>:
    <config>

model <name>:
    <config>
    def forward(x):
        <operations>

train <model> on <dataset>:
    <config>
```

---

## Imports

Import external Python modules for use in generated code.

### Syntax

```aur
use <module>
use <module> as <alias>
```

### Examples

```aur
use torch
use torchvision
use torch.nn as nn
use torch.nn.functional as F
```

### Common Imports

- `torch` - PyTorch core
- `torchvision` - Vision datasets and transforms
- `torch.nn` - Neural network modules
- `torch.optim` - Optimizers

---

## Experiments

Configure experimental settings like seeds, devices, and logging.

### Syntax

```aur
experiment <name>:
    <key> = <value>
    ...
```

### Supported Keys

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `seed` | int | Random seed | `42` |
| `device` | str | Device to use | `"cuda"`, `"cpu"`, `"auto"` |
| `mixed_precision` | bool | Enable AMP | `true`, `false` |
| `log_interval` | int | Logging frequency | `100` |
| `checkpoint_dir` | str | Checkpoint location | `"./checkpoints"` |

### Example

```aur
experiment MnistBaseline:
    seed = 42
    device = "auto"
    mixed_precision = true
    log_interval = 100
    checkpoint_dir = "./checkpoints"
```

---

## Datasets

Configure datasets for training and evaluation.

### Syntax

```aur
dataset <name>:
    from <source>
    <key> = <value>
    ...
```

### Common Keys

| Key | Type | Description |
|-----|------|-------------|
| `root` | str | Data directory |
| `train` | bool | Training mode |
| `download` | bool | Auto-download |
| `batch` | int | Batch size |
| `transform` | str | Transformation pipeline |

### Examples

**MNIST:**
```aur
dataset mnist_train:
    from torchvision.datasets.MNIST
    root = "./data"
    train = True
    download = True
    batch = 128
```

**CIFAR-10:**
```aur
dataset cifar_train:
    from torchvision.datasets.CIFAR10
    root = "./data"
    train = True
    download = True
    batch = 256
    transform = "standard"
```

**Custom Dataset:**
```aur
dataset custom_data:
    from mymodule.CustomDataset
    path = "./custom_data"
    split = "train"
    batch = 64
```

---

## Models

Define neural network architectures.

### Syntax

```aur
model <name>:
    <attribute> = <value>
    def forward(<param>):
        <operations>
```

### Attributes

| Attribute | Type | Description | Required |
|-----------|------|-------------|----------|
| `input_shape` | tuple | Input dimensions | Yes |
| `vocab_size` | int | Vocabulary size (NLP) | No |
| `embedding_dim` | int | Embedding size (NLP) | No |
| `num_classes` | int | Output classes | No |

### Forward Block

The `forward` block defines data flow through the model:

```aur
def forward(x):
    x -> <operation>
      -> <operation>.<activation>
      -> <operation>
```

### Operation Syntax

```
<variable> -> <layer>(<args>).<activation>
```

**Components:**
- `<variable>` - Input variable (usually `x`)
- `<layer>` - Layer type (conv2d, dense, etc.)
- `<args>` - Layer arguments
- `<activation>` - Optional activation function

### Example

```aur
model CNN:
    input_shape = (3, 32, 32)
    num_classes = 10
    
    def forward(x):
        x -> conv2d(64, kernel=3, padding=1).relu
          -> maxpool(2)
          -> conv2d(128, kernel=3).relu
          -> avgpool(4)
          -> flatten()
          -> dense(256).relu
          -> dropout(0.5)
          -> dense(10)
```

---

## Training

Configure training loops.

### Syntax

```aur
train <model_name> on <dataset_name>:
    <key> = <value>
    ...
```

### Required Keys

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `loss` | str | Loss function | `cross_entropy`, `mse`, `bce` |
| `optimizer` | str | Optimizer | `adam(lr=1e-3)`, `sgd(lr=0.01)` |
| `epochs` | int | Training epochs | `10`, `100` |

### Optional Keys

| Key | Type | Description |
|-----|------|-------------|
| `validate_on` | str | Validation dataset |
| `test_on` | str | Test dataset |
| `metrics` | list | Metrics to track |
| `scheduler` | str | LR scheduler |
| `gradient_clip` | float | Gradient clipping |
| `early_stopping` | bool | Enable early stopping |
| `patience` | int | Early stopping patience |

### Loss Functions

- `cross_entropy` - Cross-entropy loss
- `mse` - Mean squared error
- `bce` - Binary cross-entropy
- `nll` - Negative log likelihood
- `l1` - L1 loss
- `smooth_l1` - Smooth L1 loss

### Optimizers

```aur
adam(lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
sgd(lr=0.01, momentum=0.9, weight_decay=1e-4)
rmsprop(lr=1e-3, alpha=0.99)
adagrad(lr=0.01)
adamw(lr=1e-3, weight_decay=0.01)
```

### Schedulers

```aur
step_lr(step_size=30, gamma=0.1)
exponential_lr(gamma=0.95)
cosine_annealing(T_max=50, eta_min=0)
reduce_on_plateau(patience=10, factor=0.1)
```

### Metrics

```aur
metrics = [accuracy, precision, recall, f1, auc]
```

### Example

```aur
train ResNet on imagenet_train:
    validate_on = imagenet_val
    test_on = imagenet_test
    loss = cross_entropy
    optimizer = adam(lr=1e-3, weight_decay=1e-4)
    scheduler = cosine_annealing(T_max=50)
    epochs = 100
    metrics = [accuracy, top5_accuracy]
    gradient_clip = 1.0
    early_stopping = true
    patience = 10
```

---

## Layers

### Convolution Layers

**Conv1D:**
```aur
conv1d(out_channels, kernel=3, stride=1, padding=0)
```

**Conv2D:**
```aur
conv2d(out_channels, kernel=3, stride=1, padding=0, dilation=1)
```

**Conv3D:**
```aur
conv3d(out_channels, kernel=3, stride=1, padding=0)
```

**Examples:**
```aur
x -> conv2d(64, kernel=7, stride=2, padding=3)
x -> conv2d(128, kernel=3, padding=1)
```

### Pooling Layers

**MaxPool:**
```aur
maxpool(kernel_size, stride=None, padding=0)
```

**AvgPool:**
```aur
avgpool(kernel_size, stride=None, padding=0)
```

**AdaptiveAvgPool:**
```aur
adaptive_avgpool(output_size)
```

**Examples:**
```aur
x -> maxpool(2)
x -> avgpool(2, stride=2)
x -> adaptive_avgpool(1)
```

### Linear Layers

**Dense/Linear:**
```aur
dense(out_features)
linear(out_features)
```

**Embedding:**
```aur
embedding(num_embeddings, embedding_dim)
```

**Examples:**
```aur
x -> dense(512)
x -> linear(1000)
x -> embedding(50000, 512)
```

### Normalization Layers

**Batch Normalization:**
```aur
batch_norm()
batchnorm()
```

**Layer Normalization:**
```aur
layer_norm()
layernorm()
```

**Group Normalization:**
```aur
group_norm(num_groups)
```

**Examples:**
```aur
x -> conv2d(64).relu -> batch_norm()
x -> dense(512) -> layer_norm()
```

### Regularization

**Dropout:**
```aur
dropout(p)
```

**Examples:**
```aur
x -> dropout(0.5)
x -> dropout(0.3)
```

### Reshaping

**Flatten:**
```aur
flatten(start_dim=1)
```

**Reshape:**
```aur
reshape(*shape)
```

**Examples:**
```aur
x -> flatten()
x -> reshape(batch, -1)
x -> reshape(1, 28, 28)
```

### Advanced Layers

**Attention:**
```aur
multihead_attention(heads=8, dim=512)
self_attention(dim=512)
```

**Positional Encoding:**
```aur
positional_encoding(max_len=512)
```

---

## Activations

Activations are applied as method calls on operations:

```aur
x -> <layer>.<activation>
```

### Supported Activations

| Activation | Syntax | Description |
|------------|--------|-------------|
| ReLU | `.relu` | Rectified Linear Unit |
| GELU | `.gelu` | Gaussian Error Linear Unit |
| Leaky ReLU | `.leaky_relu(slope)` | Leaky ReLU with negative slope |
| Tanh | `.tanh` | Hyperbolic tangent |
| Sigmoid | `.sigmoid` | Sigmoid function |
| Softmax | `.softmax(dim)` | Softmax over dimension |
| LogSoftmax | `.log_softmax(dim)` | Log softmax |
| ELU | `.elu(alpha)` | Exponential Linear Unit |
| SELU | `.selu` | Scaled ELU |

### Examples

```aur
x -> conv2d(64).relu
x -> dense(512).gelu
x -> dense(256).leaky_relu(0.2)
x -> dense(10).softmax(dim=1)
```

### Chaining

Activations can be chained with batch norm and dropout:

```aur
x -> conv2d(64).relu
  -> batch_norm()
  -> dropout(0.3)
```

---

## Data Types

### Numbers

**Integers:**
```aur
epochs = 100
batch = 256
seed = 42
```

**Floats:**
```aur
lr = 1e-3
dropout_rate = 0.5
weight_decay = 1e-4
```

**Scientific Notation:**
```aur
lr = 1e-3  # 0.001
epsilon = 1e-8  # 0.00000001
```

### Strings

```aur
device = "cuda"
root = "./data"
checkpoint_dir = "/path/to/checkpoints"
```

### Booleans

```aur
train = True
download = False
mixed_precision = true
early_stopping = false
```

Note: Both `True/False` and `true/false` are supported.

### Tuples

```aur
input_shape = (1, 28, 28)
kernel_size = (3, 3)
stride = (2, 2)
```

### Lists

```aur
metrics = [accuracy, precision, recall]
layers = [64, 128, 256]
```

### None/null

```aur
pretrained = None
bias = null
```

---

## Comments

### Single-line Comments

```aur
# This is a comment
use torch  # Import PyTorch

# Model definition
model MyNet:
    input_shape = (3, 32, 32)  # CIFAR-10 input
```

### Multi-line Documentation

While Aurane doesn't have official multi-line comments, you can use multiple single-line comments:

```aur
# ResNet-style architecture
# Based on "Deep Residual Learning for Image Recognition"
# He et al., 2015
model ResNet:
    input_shape = (3, 224, 224)
```

---

## Complete Example

```aur
# Complete MNIST classification pipeline
use torch
use torchvision

# Experimental configuration
experiment MnistBaseline:
    seed = 42
    device = "auto"
    mixed_precision = true
    log_interval = 100

# Training dataset
dataset mnist_train:
    from torchvision.datasets.MNIST
    root = "./data"
    train = True
    download = True
    batch = 128

# Validation dataset
dataset mnist_test:
    from torchvision.datasets.MNIST
    root = "./data"
    train = False
    batch = 128

# Model architecture
model MnistCNN:
    input_shape = (1, 28, 28)
    num_classes = 10
    
    def forward(x):
        # First conv block
        x -> conv2d(32, kernel=3).relu
          -> batch_norm()
          -> maxpool(2)
          
        # Second conv block
          -> conv2d(64, kernel=3).relu
          -> batch_norm()
          -> maxpool(2)
          
        # Classifier
          -> flatten()
          -> dense(128).relu
          -> dropout(0.5)
          -> dense(10)

# Training configuration
train MnistCNN on mnist_train:
    validate_on = mnist_test
    loss = cross_entropy
    optimizer = adam(lr=1e-3)
    scheduler = step_lr(step_size=10, gamma=0.1)
    epochs = 20
    metrics = [accuracy]
    early_stopping = true
    patience = 5
```

---

## Best Practices

### Naming Conventions

- **Models**: PascalCase (e.g., `MnistCNN`, `ResNet50`)
- **Datasets**: snake_case (e.g., `mnist_train`, `cifar_val`)
- **Experiments**: PascalCase (e.g., `BaselineExperiment`)

### Model Organization

```aur
model ComplexNet:
    # Attributes first
    input_shape = (3, 32, 32)
    num_classes = 10
    
    # Forward definition
    def forward(x):
        # Group related operations
        # Feature extraction
        x -> conv2d(64).relu
          -> maxpool(2)
          
        # More features
          -> conv2d(128).relu
          -> maxpool(2)
          
        # Classification
          -> flatten()
          -> dense(10)
```

### Configuration

Keep related configurations together:

```aur
experiment Production:
    # Reproducibility
    seed = 42
    
    # Performance
    device = "cuda"
    mixed_precision = true
    
    # Logging
    log_interval = 100
    checkpoint_dir = "./checkpoints"
```

---

## Error Handling

### Common Errors

**Undefined Model:**
```aur
train NonExistentModel on dataset:  # Error: model not defined
```

**Missing Forward:**
```aur
model MyNet:
    input_shape = (1, 28, 28)
    # Error: no forward definition
```

**Invalid Operation:**
```aur
x -> invalid_layer(64)  # Error: unknown operation
```

### Validation

Use `aurane lint` to catch errors early:

```bash
aurane lint model.aur
```

---

## See Also

- [Getting Started](getting-started.md)
- [CLI Commands](cli-commands.md)
- [Examples](examples.md)

---

For questions or issues, please visit the GitHub repository.
