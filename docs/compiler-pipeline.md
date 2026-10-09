# Compiler pipeline reliability design

This upgrade builds on the 3.0 graph compiler. It keeps `.aur` syntax, generated
standalone Python, the `compile_source` return type, and AST backend callbacks
compatible. The goal is consistent, diagnosable compilation from the API and CLI.

## Preparation ownership

One compilation owns a resolved copy of its parsed AST. Semantic analysis, type
checking and the built-in backend share this prepared program. Model graphs are
lowered lazily once per model identity and reused for checking, class emission
and checkpoint fingerprints. Optimization creates a new prepared program with
an empty graph cache, so pre-optimization graphs cannot leak into generated code.
No process-global AST/IR cache or mutable marker is attached to public ASTs.

Standalone analysis/generation calls still prepare their own copies. Registered
backends continue receiving an `AuraneProgram`; an optional prepared callback
lets a backend opt into shared preparation. Re-registering a backend replaces
both callbacks and its cache version together.

## Error and IO contracts

`CompilationError` retains readable text and adds stage, source span and a list
of structured diagnostics. File compilation adds the source filename without
discarding the original exception or its diagnostics. CLI compilation failures
can emit one JSON record to stderr, preserving stdout for generated Python.

Cache location follows explicit argument, `AURANE_CACHE_DIR`, then the existing
working-directory `.aurane_cache` default. Disabled caching performs no cache IO.
Directory selection does not change artifact identity. Benchmarking uses its own
temporary cache without changing the process working directory.

Compile, run and watch share semantic/type checks, optimizer settings and cache
options. Source formatting publishes an adjacent temporary file with atomic
replacement; failed replacement leaves the prior file and permissions intact.

## Verification

Regressions will count resolution/lowering calls, compare generated output for
all examples and optimization levels, exercise independent concurrent
compilations and backend replacement, validate stages/spans and JSON IO, and
simulate failed publication. Existing offline training, checkpoint and runtime
suites remain the behavioral compatibility gate. Parameter-free or fully frozen
injected training models receive actionable errors before dataset construction.

Full pytest, Black, mypy, wheel/sdist builds, isolated wheel verification, example
CLI smoke checks and exact-commit hosted CI are required. No timing improvement
or model-quality claim follows from eliminating redundant work.
