# Aurane Documentation

Welcome to the Aurane documentation! This directory contains comprehensive guides and references.

## 📚 Documentation Index

### Getting Started
- **[Getting Started Guide](getting-started.md)** - Quick start tutorial and first steps
  - Installation instructions
  - Your first model
  - Offline training and evaluation
  - Checkpoint restoration

### Reference Materials
- **[Language Reference](language-reference.md)** - Complete syntax and semantics
  - Program structure
  - All supported layers
  - Activation functions
  - Training configuration
  - Compiler behavior and migration

- **[CLI Commands](cli-commands.md)** - Command-line interface reference
  - All commands explained
  - Options and flags
  - Usage examples
  - Tips and tricks

### Tutorials
- **[Examples Guide](examples.md)** - Example purposes, execution and QA scope
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
- [Language Reference - Models](language-reference.md#models-and-tensor-graphs)
- [Language Reference - Layers](language-reference.md#implemented-operations)
- [Examples - MNIST CNN](../examples/mnist.aur)

### Training Configuration
- [Language Reference - Training](language-reference.md#standard-training)
- [Examples - Complete Pipeline](../examples/mnist.aur)

### CLI Usage
- [CLI Commands - Compile](cli-commands.md#compile-and-check)
- [CLI Commands - Inspect](cli-commands.md#inspect-ir-profile-and-visualize)
- [CLI Commands - Watch](cli-commands.md#run-and-watch)

### Code Quality
- [CLI Commands - Format](cli-commands.md#format-and-lint)
- [CLI Commands - Lint](cli-commands.md#format-and-lint)
- [QA evidence and limits](qa-report.md)

### Performance
- [CLI Commands - Benchmark](cli-commands.md#benchmark)
- [CLI Commands - Watch Mode](cli-commands.md#run-and-watch)

## 🎯 Common Tasks

### Compile a Model
```bash
aurane compile model.aur output.py
```
See: [CLI Commands - Compile](cli-commands.md#compile-and-check)

### Inspect Architecture
```bash
aurane inspect model.aur --verbose
```
See: [CLI Commands - Inspect](cli-commands.md#inspect-ir-profile-and-visualize)

### Live Development
```bash
aurane watch model.aur output.py
```
See: [CLI Commands - Watch](cli-commands.md#run-and-watch)

### Interactive Coding
```bash
aurane interactive
```
See: [CLI Commands - Interactive Development](cli-commands.md#scaffold-interactive-mode-and-cleanup)

## 🔍 Finding Information

### By Language Feature

**Layers:**
- Convolution: [Language Reference - Layers](language-reference.md#implemented-operations)
- Pooling: [Language Reference - Layers](language-reference.md#implemented-operations)
- Linear: [Language Reference - Layers](language-reference.md#implemented-operations)
- Normalization: [Language Reference - Layers](language-reference.md#implemented-operations)

**Configuration:**
- Experiments: [Language Reference - Experiments](language-reference.md#experiment-configuration)
- Datasets: [Language Reference - Datasets](language-reference.md#datasets)
- Training: [Language Reference - Training](language-reference.md#standard-training)

### By Architecture Type

**Vision Models:**
- [Examples - MNIST CNN](../examples/mnist.aur)
- [Examples - ResNet](../examples/resnet.aur)

**NLP Models:**
- [Examples - Transformer](../examples/transformer.aur)

**Generative Models:**
- [Examples - GAN](../examples/gan.aur)

## 💡 Learning Path

### Beginner (Days 1-3)
1. Read [Getting Started](getting-started.md)
2. Try [Simple Network Example](../examples/simple.aur)
3. Experiment with [Interactive Mode](cli-commands.md#scaffold-interactive-mode-and-cleanup)
4. Build your first MNIST model

### Intermediate (Week 2)
1. Study [Language Reference - Models](language-reference.md#models-and-tensor-graphs)
2. Build a custom CNN
3. Master [CLI Commands](cli-commands.md)
4. Try [ResNet Example](../examples/resnet.aur)

### Advanced (Month 1+)
1. Implement [Transformer](../examples/transformer.aur)
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
- `benchmark` - Measure compiler parse, cold compile and cache-read timing
- `lint` - Check for issues

### Quality Tools
- `format` - Auto-format code
- `check` - Check semantic, shape and dtype contracts

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
- [GitHub Repository](https://github.com/desenyon/aurane) - Source code
- [GitHub Issues](https://github.com/desenyon/aurane/issues) - Bug reports

## 🤝 Contributing

Want to improve the documentation?

1. Fix typos or unclear sections
2. Add more examples
3. Create tutorials for specific use cases
4. Translate documentation

See the main [README development section](../README.md#development) for verification commands.

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
- Try [Interactive Mode](cli-commands.md#scaffold-interactive-mode-and-cleanup)

**Found a bug?**
- Report on [GitHub Issues](https://github.com/desenyon/aurane/issues)

**Have a question?**
- Start a [GitHub Discussion](https://github.com/desenyon/aurane/discussions)

## 📅 Last Updated

Documentation version: 3.0.0

Last updated: October 3, 2026

---

**Happy learning with Aurane!** 🚀
