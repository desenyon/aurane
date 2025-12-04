# CLI Commands Reference

Complete reference for all Aurane command-line tools.

## Overview

Aurane provides a rich set of CLI commands for compiling, inspecting, and working with Aurane code:

```bash
aurane <command> [options]
```

## Commands

### `compile`

Compile an Aurane file to Python.

```bash
aurane compile <input.aur> <output.py> [options]
```

**Arguments:**
- `input` - Input `.aur` file path (required)
- `output` - Output `.py` file path (required)

**Options:**
- `--backend {torch}` - Code generation backend (default: torch)
- `--analyze` - Show analysis after compilation
- `--show-ast` - Display abstract syntax tree
- `--validate` - Validate AST before compiling
- `--format` - Format output with black (if available)
- `--diff` - Show side-by-side comparison
- `--verbose, -v` - Verbose error messages

**Examples:**

```bash
# Basic compilation
aurane compile model.aur output.py

# With analysis and AST display
aurane compile model.aur output.py --analyze --show-ast

# Validate and format
aurane compile model.aur output.py --validate --format

# Verbose mode with diff
aurane compile model.aur output.py --verbose --diff
```

**Output:**

```
Compiling: model.aur
[████████████████████] 100% Complete!

╭──────── Compilation Complete ────────╮
│ ✓ Status        Success              │
│ Input           model.aur            │
│ Output          output.py            │
│ Input Size      234 lines, 5,678 bytes│
│ Output Size     456 lines, 12,345 bytes│
│ Backend         torch                │
│ Compression     2.2x                 │
│                                      │
│ Analysis                             │
│ Models          2                    │
│ Datasets        1                    │
│ Training Configs 1                   │
│ Total Layers    8                    │
╰──────────────────────────────────────╯
```

---

### `inspect`

Inspect Aurane file structure and architecture.

```bash
aurane inspect <input.aur> [options]
```

**Arguments:**
- `input` - Input `.aur` file path (required)

**Options:**
- `--verbose, -v` - Show detailed model information
- `--stats` - Display comprehensive statistics
- `--export FILE` - Export AST to JSON file

**Examples:**

```bash
# Basic inspection
aurane inspect model.aur

# Detailed with statistics
aurane inspect model.aur --verbose --stats

# Export AST to JSON
aurane inspect model.aur --export ast.json
```

**Output:**

```
Inspecting: model.aur
234 lines • 5,678 bytes

Aurane Program
├── Imports
│   ├── torch
│   └── torchvision
├── Experiments
│   └── MnistBaseline
│       ├── seed = 42
│       └── device = auto
├── Datasets
│   └── mnist_train
│       ├── from torchvision.datasets.MNIST
│       ├── root = ./data
│       ├── train = True
│       └── batch = 128
└── Models
    └── MnistNet
        └── forward
            ├── conv2d(32, kernel=3).relu
            ├── maxpool(2)
            ├── flatten()
            └── dense(10)

━━━ Model Details ━━━

MnistNet
Input Shape: (1, 28, 28)

Layer              Output Shape    Parameters
─────────────────────────────────────────────
Conv2D(32)         (32, 26, 26)           320
MaxPool(2)         (32, 13, 13)             0
Flatten()          (10,816)                 0
Dense(10)          (10)              108,170
─────────────────────────────────────────────
Total: 108,490 parameters
```

---

### `watch`

Watch file and automatically recompile on changes.

```bash
aurane watch <input.aur> <output.py> [options]
```

**Arguments:**
- `input` - Input `.aur` file path (required)
- `output` - Output `.py` file path (required)

**Options:**
- `--backend {torch}` - Code generation backend (default: torch)
- `--analyze` - Show analysis on each compile

**Examples:**

```bash
# Basic watch mode
aurane watch model.aur output.py

# With analysis
aurane watch model.aur output.py --analyze
```

**Output:**

```
👁  Watching: model.aur
Press Ctrl+C to stop

Compiling: model.aur
✓ Compiled successfully (1.2s)

[Edit detected]

⟳ File changed, recompiling...
✓ Compiled successfully (0.8s)

[Edit detected]

⟳ File changed, recompiling...
✓ Compiled successfully (0.7s)
```

**Usage Tips:**
- Keep watch running in a separate terminal
- Compilation is debounced (0.5s) to avoid rapid recompiles
- Press `Ctrl+C` to stop watching

---

### `interactive`

Start interactive REPL mode.

```bash
aurane interactive
```

**No arguments required.**

**REPL Commands:**

| Command | Alias | Description |
|---------|-------|-------------|
| `.help` | `.h`, `.?` | Show help |
| `.compile` | `.c` | Compile buffer |
| `.show` | `.s` | Show buffer contents |
| `.clear` | `.clr` | Clear buffer |
| `.save <file>` | - | Save buffer to file |
| `.load <file>` | - | Load file into buffer |
| `.validate` | `.check` | Validate buffer |
| `.history` | `.hist` | Show command history |
| `.exit` | `.quit`, `.q` | Exit REPL |

**Examples:**

```bash
$ aurane interactive

    █████╗ ██╗   ██╗██████╗  █████╗ ███╗   ██╗███████╗
   ██╔══██╗██║   ██║██╔══██╗██╔══██╗████╗  ██║██╔════╝
   ███████║██║   ██║██████╔╝███████║██╔██╗ ██║█████╗  
   ██╔══██║██║   ██║██╔══██╗██╔══██║██║╚██╗██║██╔══╝  
   ██║  ██║╚██████╔╝██║  ██║██║  ██║██║ ╚████║███████╗
   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝
   
   ML-oriented DSL that transpiles to idiomatic Python
   Version 0.2.0 • PyTorch Backend • MIT License

Interactive Mode - Type .help for commands

aurane> model TestNet:
....... input_shape = (1, 28, 28)
....... def forward(x):
.......     x -> conv2d(32).relu -> flatten() -> dense(10)

aurane> .compile
✓ Compilation successful!

[Python code displayed with syntax highlighting]

aurane> .save test.aur
✓ Saved to test.aur

aurane> .exit
Goodbye!
```

---

### `format`

Format Aurane source files with consistent style.

```bash
aurane format <path> [options]
```

**Arguments:**
- `path` - File or directory to format (required)

**Options:**
- `--check` - Check formatting without modifying files
- `--verbose, -v` - Show all files processed

**Examples:**

```bash
# Format a single file
aurane format model.aur

# Format all files in a directory
aurane format examples/

# Check formatting without changes
aurane format examples/ --check

# Verbose output
aurane format . --verbose
```

**Output:**

```
Formatting 5 file(s)...

✓ Formatted examples/mnist.aur
✓ Formatted examples/resnet.aur
  examples/simple.aur (no changes)
✓ Formatted examples/transformer.aur
  examples/gan.aur (no changes)

Formatted 3 file(s)
```

**Formatting Rules:**
- Consistent indentation (4 spaces)
- Normalized whitespace
- Proper spacing around operators
- Aligned block structures

---

### `lint`

Check Aurane files for potential issues.

```bash
aurane lint <input.aur> [options]
```

**Arguments:**
- `input` - Input `.aur` file path (required)

**Options:**
- `--verbose, -v` - Show all issue types including info

**Examples:**

```bash
# Basic linting
aurane lint model.aur

# Verbose (shows info messages)
aurane lint model.aur --verbose
```

**Output:**

```
Linting: model.aur

✗ 2 error(s):
  • Model 'MnistNet' has no forward operations
  • Training references undefined dataset 'mnist_val'

⚠ 3 warning(s):
  • Line 45: Line too long (120 > 100)
  • Line 67: Inconsistent indentation (6 spaces)
  • Line 89: Training block missing 'loss' configuration

ℹ 2 info(s):
  • Line 23: Trailing whitespace
  • Line 56: Consider adding validation dataset
```

**Issue Levels:**
- **Error** - Must fix, will cause compilation failure
- **Warning** - Should fix, potential problems
- **Info** - Optional suggestions for better code

---

### `benchmark`

Measure compilation performance.

```bash
aurane benchmark <input.aur> [options]
```

**Arguments:**
- `input` - Input `.aur` file path (required)

**Options:**
- `--iterations, -n INT` - Number of iterations (default: 10)

**Examples:**

```bash
# Default benchmark (10 iterations)
aurane benchmark model.aur

# Custom iterations
aurane benchmark model.aur --iterations 50
```

**Output:**

```
Benchmarking: model.aur
Running 50 iterations...

Benchmark Results
─────────────────────────────────────────────────────────
Phase      Mean     Median   Std Dev  Min      Max
─────────────────────────────────────────────────────────
Parse      12.45ms  12.30ms  0.85ms   11.20ms  14.10ms
Compile    45.67ms  45.20ms  2.10ms   42.80ms  50.30ms
Total      58.12ms  57.50ms  2.45ms   54.00ms  64.40ms

File: 234 lines, 5,678 bytes
Throughput: 4,025 lines/sec
```

**Metrics:**
- **Mean** - Average time across all iterations
- **Median** - Middle value (less affected by outliers)
- **Std Dev** - Standard deviation (consistency measure)
- **Min/Max** - Best and worst case times

---

### `run`

Compile and execute an Aurane file.

```bash
aurane run <input.aur> [options]
```

**Arguments:**
- `input` - Input `.aur` file path (required)

**Options:**
- `--backend {torch}` - Code generation backend (default: torch)
- `--keep-temp` - Keep temporary compiled file

**Examples:**

```bash
# Compile and run
aurane run model.aur

# Keep temporary file
aurane run model.aur --keep-temp
```

**How it works:**
1. Compiles `.aur` file to temporary `.py` file
2. Executes the Python file
3. Deletes temporary file (unless `--keep-temp`)

---

## Global Options

These options work with all commands:

- `--help, -h` - Show command help
- `--version` - Show Aurane version

**Examples:**

```bash
# Get help for a command
aurane compile --help

# Check version
aurane --version
```

---

## Exit Codes

Aurane uses standard exit codes:

- `0` - Success
- `1` - General error (compilation, parsing, etc.)
- `2` - Command-line usage error

These can be used in scripts:

```bash
if aurane compile model.aur output.py; then
    echo "Compilation successful"
    python output.py
else
    echo "Compilation failed"
    exit 1
fi
```

---

## Configuration

Currently, Aurane uses sensible defaults and doesn't require configuration files.

Future versions may support:
- `.auranerc` - Project-level configuration
- `aurane.toml` - Package-level settings

---

## Tips and Tricks

### Chaining Commands

```bash
# Compile, lint, and run
aurane lint model.aur && \
aurane compile model.aur output.py && \
python output.py
```

### Using with Make

```makefile
.PHONY: compile watch clean

compile:
	aurane compile model.aur output.py --analyze

watch:
	aurane watch model.aur output.py

clean:
	rm -f output.py
```

### Integration with Pre-commit

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: aurane-lint
        name: Lint Aurane files
        entry: aurane lint
        language: system
        files: \.aur$
      - id: aurane-format
        name: Format Aurane files
        entry: aurane format
        language: system
        files: \.aur$
```

### Shell Aliases

```bash
# .bashrc or .zshrc
alias ac='aurane compile'
alias ai='aurane inspect'
alias aw='aurane watch'
alias af='aurane format'
alias al='aurane lint'
```

---

## Troubleshooting

### Command Not Found

If `aurane` command is not found:

```bash
# Use module syntax
python -m aurane.cli <command> [options]

# Or add to PATH
export PATH="$HOME/.local/bin:$PATH"
```

### Rich Not Available

Some features require the `rich` library:

```bash
pip install rich

# Or install with all features
pip install -e ".[all]"
```

### Watch Mode Not Working

Watch mode requires `watchdog`:

```bash
pip install watchdog
```

---

## See Also

- [Getting Started](getting-started.md) - Quick start guide
- [Language Reference](language-reference.md) - Complete syntax
- [Examples](examples.md) - Example models

---

For more help, run any command with `--help` or visit the documentation.
