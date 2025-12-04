# Examples Guide

Walkthrough of example Aurane models with explanations.

## Available Examples

All examples are in the [`examples/`](../examples/) directory:

1. [Simple Network](#simple-network) - `simple.aur`
2. [MNIST CNN](#mnist-cnn) - `mnist.aur`
3. [ResNet Architecture](#resnet-architecture) - `resnet.aur`
4. [Transformer Model](#transformer-model) - `transformer.aur`
5. [GAN](#generative-adversarial-network) - `gan.aur`

---

## Simple Network

**File:** `examples/simple.aur`

The simplest possible neural network - great for getting started.

```aur
use torch

model SimpleNet:
    input_shape = (1, 28, 28)
    def forward(x):
        x -> flatten()
          -> dense(128).relu
          -> dense(10)
```

### What it does:
1. Flattens 28x28 images into vectors
2. Applies a hidden layer with 128 neurons
3. Outputs 10 class scores

### Compile and inspect:

```bash
aurane compile examples/simple.aur build/simple.py
aurane inspect examples/simple.aur
```

### Parameters: ~101,000

---

## MNIST CNN

**File:** `examples/mnist.aur`

Complete MNIST classification with CNN and training configuration.

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

### Architecture breakdown:

**Layer 1:** Conv(32 filters) + ReLU
- Input: (1, 28, 28)
- Output: (32, 26, 26)
- Parameters: 320

**Layer 2:** MaxPool
- Output: (32, 13, 13)

**Layer 3:** Conv(64 filters) + ReLU
- Output: (64, 11, 11)
- Parameters: 18,496

**Layer 4:** MaxPool
- Output: (64, 5, 5)

**Layer 5:** Flatten + Dense(128) + ReLU
- Output: (128)
- Parameters: 204,928

**Layer 6:** Dropout(0.5)

**Layer 7:** Dense(10)
- Output: (10)
- Parameters: 1,290

**Total Parameters:** ~225,000

### Key features:
- Two convolutional blocks
- Max pooling for downsampling
- Dropout for regularization
- Complete training pipeline

### Run it:

```bash
aurane compile examples/mnist.aur build/mnist.py --analyze
python build/mnist.py
```

---

## ResNet Architecture

**File:** `examples/resnet.aur`

ResNet-style architecture with modern components.

```aur
use torch
use torchvision

model ResNetClassifier:
    input_shape = (3, 32, 32)
    def forward(x):
        x -> conv2d(64, kernel=7, stride=2, padding=3).relu
          -> maxpool(3, stride=2)
          -> conv2d(64, kernel=3, padding=1).relu
          -> conv2d(64, kernel=3, padding=1).relu
          -> conv2d(128, kernel=3, stride=2, padding=1).relu
          -> conv2d(128, kernel=3, padding=1).relu
          -> avgpool(4)
          -> flatten()
          -> dense(512).relu
          -> dropout(0.5)
          -> dense(10)

train ResNetClassifier on cifar_train:
    validate_on = cifar_val
    loss = cross_entropy
    optimizer = adam(lr=1e-3, weight_decay=1e-4)
    scheduler = cosine_annealing(T_max=50)
    epochs = 50
    early_stopping = true
    patience = 10
```

### Key components:

**Initial Block:**
- 7x7 conv with stride 2 (downsampling)
- 3x3 max pooling

**Residual Blocks:**
- Multiple 3x3 convolutions
- Strided convolutions for downsampling
- Padding to maintain spatial dimensions

**Classifier:**
- Average pooling instead of max
- Dense layers with dropout

**Advanced training:**
- Learning rate scheduling
- Weight decay regularization
- Early stopping
- Validation during training

### Parameters: ~460,000

### Inspect architecture:

```bash
aurane inspect examples/resnet.aur --verbose --stats
```

---

## Transformer Model

**File:** `examples/transformer.aur`

Transformer-based language model.

```aur
use torch

model LanguageModel:
    input_shape = (128,)
    vocab_size = 50000
    embedding_dim = 512
    
    def forward(x):
        x -> embedding(vocab_size, embedding_dim)
          -> positional_encoding(max_len=128)
          -> multihead_attention(heads=8, dim=512)
          -> layer_norm()
          -> dense(2048).gelu
          -> dense(512)
          -> dropout(0.1)
          -> dense(vocab_size)
```

### Architecture breakdown:

**Embedding Layer:**
- Maps token IDs to dense vectors
- Vocab: 50,000 tokens
- Dimension: 512

**Positional Encoding:**
- Adds position information
- Max sequence length: 128

**Multi-Head Attention:**
- 8 attention heads
- Allows model to attend to different positions

**Feed-Forward Network:**
- Expansion to 2048 dimensions
- GELU activation (Transformer standard)
- Project back to 512

**Output Projection:**
- Maps to vocabulary size
- Used for next-token prediction

### Key features:
- Modern NLP architecture
- Self-attention mechanism
- Layer normalization
- GELU activations (better than ReLU for transformers)

### Parameters: ~26 million

### Use cases:
- Language modeling
- Text generation
- Sequence-to-sequence tasks

---

## Generative Adversarial Network

**File:** `examples/gan.aur`

GAN with generator and discriminator.

```aur
use torch

model Generator:
    input_shape = (100,)
    def forward(z):
        z -> dense(256).relu
          -> batch_norm()
          -> dense(512).relu
          -> batch_norm()
          -> dense(1024).relu
          -> batch_norm()
          -> dense(784).tanh
          -> reshape(1, 28, 28)

model Discriminator:
    input_shape = (1, 28, 28)
    def forward(x):
        x -> flatten()
          -> dense(1024).leaky_relu(0.2)
          -> dropout(0.3)
          -> dense(512).leaky_relu(0.2)
          -> dropout(0.3)
          -> dense(256).leaky_relu(0.2)
          -> dense(1).sigmoid
```

### Generator:

**Purpose:** Create fake images from random noise

**Architecture:**
1. Takes 100-dim noise vector
2. Progressively upsamples through dense layers
3. Batch normalization for stability
4. Final tanh activation (outputs in [-1, 1])
5. Reshapes to 28x28 image

**Parameters:** ~1.5 million

### Discriminator:

**Purpose:** Distinguish real from fake images

**Architecture:**
1. Flattens input image
2. Dense layers with decreasing size
3. Leaky ReLU (helps with vanishing gradients)
4. Dropout for regularization
5. Sigmoid output (probability real/fake)

**Parameters:** ~1.7 million

### Key features:
- Two competing networks
- Batch normalization in generator
- Leaky ReLU in discriminator
- Dropout for regularization
- Appropriate activations (tanh, sigmoid)

### Training GANs:

GANs require special training procedures (alternating updates). The compiled code provides the model definitions - you'll need to implement the GAN training loop.

---

## Comparing Examples

| Example | Complexity | Parameters | Use Case |
|---------|-----------|------------|----------|
| Simple | ⭐ | ~100K | Learning basics |
| MNIST | ⭐⭐ | ~225K | Image classification |
| ResNet | ⭐⭐⭐⭐ | ~460K | Advanced vision |
| Transformer | ⭐⭐⭐⭐⭐ | ~26M | NLP tasks |
| GAN | ⭐⭐⭐⭐ | ~3M | Image generation |

---

## Running Examples

### Compile all examples:

```bash
# Compile each one
for file in examples/*.aur; do
    aurane compile "$file" "build/$(basename "$file" .aur).py"
done
```

### Inspect an example:

```bash
aurane inspect examples/resnet.aur --verbose --stats
```

### Benchmark compilation:

```bash
aurane benchmark examples/transformer.aur
```

### Watch mode during development:

```bash
aurane watch examples/mnist.aur build/mnist.py
```

---

## Customizing Examples

### Modify MNIST for CIFAR-10:

```aur
model CifarNet:
    input_shape = (3, 32, 32)  # 3 channels, 32x32
    def forward(x):
        x -> conv2d(64, kernel=3).relu  # More filters
          -> batch_norm()  # Add normalization
          -> maxpool(2)
          -> conv2d(128, kernel=3).relu
          -> batch_norm()
          -> maxpool(2)
          -> flatten()
          -> dense(512).relu  # Larger hidden layer
          -> dropout(0.5)
          -> dense(10)
```

### Add validation to any model:

```aur
dataset train_data:
    from torchvision.datasets.MNIST
    root = "./data"
    train = True
    batch = 128

dataset val_data:
    from torchvision.datasets.MNIST
    root = "./data"
    train = False
    batch = 128

train MyModel on train_data:
    validate_on = val_data  # Add this
    loss = cross_entropy
    optimizer = adam(lr=1e-3)
    epochs = 10
```

### Use learning rate scheduling:

```aur
train MyModel on data:
    loss = cross_entropy
    optimizer = adam(lr=1e-3)
    scheduler = cosine_annealing(T_max=50)  # Add this
    epochs = 50
```

---

## Best Practices from Examples

### 1. Start Simple
- Begin with `simple.aur`
- Add complexity gradually
- Test each addition

### 2. Use Appropriate Activations
- ReLU for general use
- GELU for transformers
- Leaky ReLU for GANs
- Tanh for outputs in [-1, 1]

### 3. Add Regularization
- Dropout after dense layers
- Batch norm after convolutions
- Weight decay in optimizer

### 4. Configure Training Properly
- Set reasonable learning rates
- Use validation sets
- Add early stopping
- Track metrics

### 5. Match Architecture to Task
- CNNs for images (MNIST, ResNet)
- Transformers for sequences (Language Model)
- Specialized structures for generation (GAN)

---

## Creating Your Own Examples

### Template:

```aur
use torch
use torchvision

experiment MyExperiment:
    seed = 42
    device = "auto"

dataset my_data:
    from mymodule.MyDataset
    root = "./data"
    batch = 128

model MyModel:
    input_shape = (...)
    def forward(x):
        x -> # your architecture
          -> # here

train MyModel on my_data:
    loss = # appropriate loss
    optimizer = # appropriate optimizer
    epochs = # reasonable number
```

### Tips:
1. Define input shape correctly
2. Ensure dimensions match between layers
3. Use `aurane inspect` to verify architecture
4. Use `aurane lint` to catch errors
5. Test with small epochs first

---

## Next Steps

- Modify examples for your datasets
- Combine techniques from multiple examples
- Create custom architectures
- Share your examples with the community!

---

## Additional Resources

- [Getting Started](getting-started.md)
- [Language Reference](language-reference.md)
- [CLI Commands](cli-commands.md)

For more examples, check the [GitHub repository](https://github.com/yourusername/aurane).
