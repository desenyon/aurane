# Rework QA evidence

Date: 2026-10-03. Release 3.0.0 verification, based on `5941e3e`.

## Environment and results

- macOS, Python 3.13.12, PyTorch 2.14.1, torchvision 0.29.1; CPU runtime tests.
- `python -m pytest -q -o addopts='' --tb=short`: **643 passed**.
- Repeated the complete 643-test suite in a fresh Python 3.10.20 environment with
  PyTorch 2.4.0, torchvision 0.19.0 and NumPy 1.26.4: **643 passed**.
- Compiler-only Python 3.10 environment without torch/torchvision:
  **480 passed, 20 skipped** (runtime-dependent cases skip explicitly).
- `python -m mypy aurane --ignore-missing-imports`: **42 source files passed**.
  Compiler checks exclude optional PyTorch dependency stubs; runtime behavior is
  tested separately.
- `python -m black --check aurane tests scripts`: **80 files passed**.
- `git diff --check`: passed.

## Behavior exercised

- Malformed syntax/indentation, nested/escaped literals, inline comments, symbol
  scopes, cycles, undefined/duplicate names and located errors.
- Shared configuration validation across checking/generation: invalid loss,
  optimizer and scheduler arguments, callbacks, experiment settings, import/name
  collisions and GAN model boundaries. Callback interval/override runtime proof.
- Explicit/inferred dtypes in checks, generated models, JSON IR and profiling;
  float16/bfloat16/float32/float64 dense forward/backward, float64 weighted training
  and GAN noise. Native attention parity and padding masks in train/eval.
- Exact operation spans, repeated calls, continuation lines and Unicode-aware
  symbol references, with compile errors identifying the failing expression.
  Structured JSON spans cover training fields, experiment/dataset settings,
  input dtypes, operations, undefined references and parse errors.
- Shared graph shapes, undefined uses/returns, value reassignment, branch diagrams,
  all-input addition and concat axes. Runtime output and gradient comparisons.
- Optimizer equivalence in train/eval, parameter and buffer preservation.
- Pooling/dense shapes, learned positional parameters, variable-length LayerNorm,
  padding gradients, causal attention, activation suffixes and batch-preserving
  reshape. Invalid graph/shape operations fail before execution.
- Import-safe dataset factories; offline weight updates; CPU AMP fallback,
  clipping, loss weighting, validation, token padding and scheduler order.
- Accuracy/top-k/perplexity and regression metrics checked against PyTorch.
- Exact resumed/uninterrupted dropout training, optimizer/scheduler/RNG/history
  restoration, early stopping, incompatible checkpoints and interrupted saves.
  Stochastic train/validation resume also restores NumPy RNG, separate sampler
  generators and explicit dataset state hooks. Persistent workers fail before
  iteration when checkpointing/resuming.
- Ordered mixed standard/GAN jobs, repeated names and per-job epoch settings.
- Shared operation keyword/arity validation; Conv1d, grouped/dilated convolutions,
  LSTM/GRU and upsampling shapes, parameter counts, numerical outputs and gradients.
- Held-out evaluation runs exactly once after training; weighted CE uses class
  weights for loss and unweighted metric counts. Target broadcasting is rejected.
- Binary/multiclass precision, recall, F1 and ROC AUC, with tied ranks, missing
  classes, ignored labels and whole-dataset aggregation across uneven batches.
- Lazy nested torchvision transform construction, constants, numerical normalization,
  resize output shapes and invalid transform declarations.
- GAN phase isolation, update ratios, BCE/MSE losses, selected metrics, sample
  tensors and decoded PNG grids. GAN split/uninterrupted resume matches both
  models, histories and samples exactly; failed writes preserve previous files.
- CLI stdout/JSON contracts, file preservation, cache corruption/concurrency,
  cleanup boundaries, lint/format preservation, interactive indentation/recovery,
  atomic watcher saves, rapid consecutive edits, run cleanup and isolated benchmarks.
- Every bundled example's training path on small offline fixtures. MNIST/CIFAR
  downloads are replaced with FakeData in this suite.

## Package and end-to-end checks

Built wheel and source distribution with `uv build`; installed the wheel into a
fresh virtual environment without PyTorch. Ran
`python -I scripts/verify_distribution.py DIST_DIRECTORY` against that installation,
checking import provenance, version, JSON check, Python compilation output, IR,
benchmark and required source-archive contents. Artifacts remained outside the
repository and were not published.

All five examples pass semantic/type checks and generated-Python syntax checks
at optimization levels 0, 1 and 2. The actual commands
`aurane run examples/simple.aur` and `aurane run examples/transformer.aur` completed
their two configured epochs with finite loss and removed their temporary scripts.

CI now declares compiler tests for Python 3.10–3.13, required runtime tests,
minimum/current PyTorch jobs running the full suite, required formatting/type
checks and a built-distribution check. Publishing depends on that reusable quality
workflow and verifies its exact artifacts before upload. YAML parsing and the
release dependency chain were checked locally. Hosted jobs have
not been executed in this session.

## Limits

No CUDA hardware run was performed. No real dataset
download, long training run, model-quality evaluation, publication, or deployment
was performed. Deterministic resume evidence covers single-process offline
loaders with documented RNG/state hooks. Private worker state and arbitrary external
sources are not captured automatically; persistent workers are rejected for
checkpointed runs. The remaining
feature and release gates are listed in the [rework plan](rework-plan.md).
