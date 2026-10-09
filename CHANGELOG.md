# Changelog

## Unreleased

- Share an invocation-local resolved AST and lazy graphs across compiler checks and
  the Torch backend; invalidate graphs after optimization and preserve AST plugins.
- Preserve compilation stages, source spans and diagnostic lists through file/API
  boundaries; add JSON compilation failures on stderr.
- Unify compile/run/watch analysis, optimization and cache options. Support explicit
  cache directories and `AURANE_CACHE_DIR`, plus polling watch mode.
- Publish formatted source atomically and isolate benchmark caches without changing
  the process working directory.
- Initialize Python and optional NumPy RNGs from experiment seeds, and reject
  parameter-free/frozen training models before constructing data loaders.
- Expand regression and installed-wheel checks and document architecture, migration,
  reproducibility boundaries and verification. Package version remains 3.0.0.

## 3.0.0 - 2026-10-03

### Added

- Train and evaluate offline models with classification, token, binary and regression metrics; held-out testing runs once after training.
- Resume standard and GAN training from atomic checkpoints, including optimizer, scheduler, scaler, RNG, loader and supported dataset state.
- Compile Conv1d, grouped/dilated convolutions, recurrent layers, upsampling, causal attention and padding masks through a shared graph representation.
- Declare input dtypes and inspect inferred shapes, dtypes and precise source spans in JSON diagnostics and IR.
- Construct declarative torchvision transform pipelines lazily and export GAN tensor samples or PNG grids.

### Changed

- Parsing, reference resolution and configuration validation reject unsupported or ambiguous declarations instead of silently ignoring them.
- LayerNorm normalizes the feature axis; dense preserves leading dimensions; reshape preserves the batch axis; graph assignments retain distinct tensor identities.
- Optimization preserves training behavior and parameter state; unsafe dropout removal and layer fusions are disabled.
- Imports no longer construct datasets or start training. Generated training functions accept injected models and loaders; multiple jobs run in source order.
- Profiling and diagrams share the compiler graph; activation memory estimates use inferred dtypes.

### Fixed

- CLI JSON/stdout contracts, atomic compilation and cache writes, editor save watching, formatting, linting, temporary-file cleanup and interactive input.
- Unknown dimensions remain unknown through flatten and reshape instead of generating invalid fixed-width layers.
- Weighted loss aggregation, GAN phase isolation, scheduler ordering, checkpoint callback intervals and generated-name collisions.
- Release publication now runs once per version tag, after the required compiler/runtime/package checks.

### Migration

Recompile `.aur` files and regenerate models before training. Old files that depended on ignored settings or invalid shapes now fail explicitly. Feature-axis LayerNorm and batch-preserving reshape can change parameter shapes or outputs; retrain or migrate weights deliberately. Checkpoints validate generated model behavior and can reject incompatible older definitions. Python 3.10+ and PyTorch 2.4+ are required; model-only compilation does not require PyTorch. See `docs/language-reference.md` and `docs/getting-started.md` for supported contracts.

### Verification

643 tests passed locally on both PyTorch 2.4.0 / Python 3.10 and PyTorch 2.14.1 / Python 3.13. Compiler-only checks, typing, formatting and clean distribution installation also passed. CUDA hardware and model learning quality are not covered by these CPU checks.
