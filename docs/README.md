# Aurane Documentation

Welcome to the Aurane documentation! This directory contains comprehensive guides and references.

## 📚 Documentation Index

### Getting Started
- **[Getting Started Guide](getting-started.md)** - Quick start tutorial and first steps
  - Installation instructions
  - Your first model
  - Interactive development
  - Common patterns

### Reference Materials
- **[Language Reference](language-reference.md)** - Complete syntax and semantics
  - Program structure
  - All supported layers
  - Activation functions
  - Training configuration
  - Complete examples

- **[CLI Commands](cli-commands.md)** - Command-line interface reference
  - All commands explained
  - Options and flags
  - Usage examples
  - Tips and tricks

### Tutorials
- **[Examples Guide](examples.md)** - Walkthrough of example models
  - Simple network
  - MNIST CNN
  - ResNet architecture
  - Transformer model
  - GAN

## 🚀 Quick Links

### For Beginners
1. Start with [Getting Started](getting-started.md)
2. Try the examples in [Examples Guide](examples.md)
3. Reference [Language Reference](language-reference.md) as needed

### For Advanced Users
1. Master all [CLI Commands](cli-commands.md)
2. Study advanced patterns in [Examples Guide](examples.md)
3. Deep dive into [Language Reference](language-reference.md)

## 📖 Topics by Task

### Writing Models
- [Language Reference - Models](language-reference.md#models)
- [Language Reference - Layers](language-reference.md#layers)
- [Examples - MNIST CNN](examples.md#mnist-cnn)

### Training Configuration
- [Language Reference - Training](language-reference.md#training)
- [Examples - Complete Pipeline](examples.md#mnist-cnn)

### CLI Usage
- [CLI Commands - Compile](cli-commands.md#compile)
- [CLI Commands - Inspect](cli-commands.md#inspect)
- [CLI Commands - Watch](cli-commands.md#watch)

### Code Quality
- [CLI Commands - Format](cli-commands.md#format)
- [CLI Commands - Lint](cli-commands.md#lint)
- [Getting Started - Best Practices](getting-started.md#tips-and-best-practices)

### Performance
- [CLI Commands - Benchmark](cli-commands.md#benchmark)
- [Getting Started - Watch Mode](getting-started.md#watch-mode-for-live-development)

## 🎯 Common Tasks

### Compile a Model
```bash
aurane compile model.aur output.py
```
See: [CLI Commands - Compile](cli-commands.md#compile)

### Inspect Architecture
```bash
aurane inspect model.aur --verbose
```
See: [CLI Commands - Inspect](cli-commands.md#inspect)

### Live Development
```bash
aurane watch model.aur output.py
```
See: [CLI Commands - Watch](cli-commands.md#watch)

### Interactive Coding
```bash
aurane interactive
```
See: [Getting Started - Interactive Development](getting-started.md#interactive-development)

## 🔍 Finding Information

### By Language Feature

**Layers:**
- Convolution: [Language Reference - Layers](language-reference.md#convolution-layers)
- Pooling: [Language Reference - Layers](language-reference.md#pooling-layers)
- Linear: [Language Reference - Layers](language-reference.md#linear-layers)
- Normalization: [Language Reference - Layers](language-reference.md#normalization-layers)

**Configuration:**
- Experiments: [Language Reference - Experiments](language-reference.md#experiments)
- Datasets: [Language Reference - Datasets](language-reference.md#datasets)
- Training: [Language Reference - Training](language-reference.md#training)

### By Architecture Type

**Vision Models:**
- [Examples - MNIST CNN](examples.md#mnist-cnn)
- [Examples - ResNet](examples.md#resnet-architecture)

**NLP Models:**
- [Examples - Transformer](examples.md#transformer-model)

**Generative Models:**
- [Examples - GAN](examples.md#generative-adversarial-network)

## 💡 Learning Path

### Beginner (Days 1-3)
1. Read [Getting Started](getting-started.md)
2. Try [Simple Network Example](examples.md#simple-network)
3. Experiment with [Interactive Mode](getting-started.md#interactive-development)
4. Build your first MNIST model

### Intermediate (Week 2)
1. Study [Language Reference - Models](language-reference.md#models)
2. Build a custom CNN
3. Master [CLI Commands](cli-commands.md)
4. Try [ResNet Example](examples.md#resnet-architecture)

### Advanced (Month 1+)
1. Implement [Transformer](examples.md#transformer-model)
2. Create custom architectures
3. Use all CLI tools (format, lint, benchmark)
4. Contribute examples back to community

## 🛠️ Tools Overview

### Development Tools
- `compile` - Convert .aur to Python
- `watch` - Auto-recompile on changes
- `interactive` - REPL for experimentation

### Analysis Tools
- `inspect` - View model architecture
- `benchmark` - Measure performance
- `lint` - Check for issues

### Quality Tools
- `format` - Auto-format code
- `validate` - Check syntax

See [CLI Commands](cli-commands.md) for complete details.

## 📊 Cheat Sheet

### Basic Workflow
```bash
# 1. Write model
vim model.aur

# 2. Check it
aurane lint model.aur

# 3. Compile it
aurane compile model.aur output.py --analyze

# 4. Run it
python output.py
```

### Development Workflow
```bash
# Terminal 1: Watch mode
aurane watch model.aur output.py

# Terminal 2: Edit
vim model.aur

# Terminal 3: Test
python output.py
```

### Quality Workflow
```bash
# Format
aurane format model.aur

# Lint
aurane lint model.aur

# Compile with validation
aurane compile model.aur output.py --validate --format
```

## 🔗 External Resources

- [Main README](../README.md) - Project overview
- [Examples Directory](../examples/) - Sample .aur files
- [GitHub Repository](https://github.com/yourusername/aurane) - Source code
- [GitHub Issues](https://github.com/yourusername/aurane/issues) - Bug reports

## 🤝 Contributing

Want to improve the documentation?

1. Fix typos or unclear sections
2. Add more examples
3. Create tutorials for specific use cases
4. Translate documentation

See main [README](../README.md) for contribution guidelines.

## 📝 Documentation Standards

All documentation follows:
- Clear, concise language
- Code examples for every feature
- Progressive complexity (simple → advanced)
- Cross-references between documents
- Tested code samples

## 🆘 Getting Help

**Documentation unclear?**
- Check other sections in this directory
- Look at [Examples](examples.md)
- Try [Interactive Mode](getting-started.md#interactive-development)

**Found a bug?**
- Report on [GitHub Issues](https://github.com/yourusername/aurane/issues)

**Have a question?**
- Start a [GitHub Discussion](https://github.com/yourusername/aurane/discussions)

## 📅 Last Updated

Documentation version: 0.2.0  
Last updated: December 2025

---

**Happy learning with Aurane!** 🚀
