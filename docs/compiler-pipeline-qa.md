# Compiler pipeline upgrade verification

Upgrade baseline: `d2e3ceb1aa0da333555b7cf476ae6d1bfb32bdd4` (3.0.0).
This records local evidence for the unreleased pipeline reliability changes.
Hosted results belong to the improvement PR and its exact head commit; this
document does not assert that an unrun workflow passed or that a release shipped.

## Local checks

| Check | Environment | Result |
| --- | --- | --- |
| Full offline test suite | macOS arm64, Python 3.13.12, Torch 2.14.1, torchvision 0.29.1, NumPy 2.5.3 | 686 passed |
| Compiler-only suite | Fresh Python 3.10.20 environment, no torch/torchvision | 514 passed, 21 explicit skip entries |
| Black | `python -m black --check aurane tests scripts` | 86 files passed |
| mypy | `python -m mypy aurane --ignore-missing-imports` | 45 source files passed |
| Whitespace | `git diff --check` | Passed |
| Distribution | `python -m build --outdir dist` | Wheel and sdist built |
| Installed wheel | Python 3.10, `python -I scripts/verify_distribution.py dist` | Passed |
| Example CLI checks | All five bundled examples, optimization levels 0/1/2 | Semantic/type checks and generated-Python syntax passed |
| Actual CLI training | `simple.aur` and `transformer.aur`, analyze/validate/optimize, no cache | Both completed configured two-epoch runs offline |

Test command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -o addopts='' --tb=short`.
Runtime work ran with one CPU thread on the shared machine. The initial baseline
under the filesystem sandbox had 642 passes and a native watcher failure; the
unchanged watcher test passed when macOS filesystem events were accessible.
The final full suites ran with that access, including native and polling watchers.

## Added regression evidence

- Count resolution and graph lowering calls for checked compilation; verify
  optimization invalidates graphs and leaves caller ASTs unchanged.
- Compare prepared and standalone generation for every example at all supported
  optimization levels; preserve legacy resolved-AST backend callbacks and clear
  prepared callbacks when a plugin is replaced.
- Preserve parse/resolution/semantic/type/codegen stages and exact spans across
  `compile_file`; retain source paths and structured error lists.
- Exercise explicit/environment/default cache precedence, zero cache IO when
  disabled, temporary-file options and cleanup, CLI JSON stderr/stdout separation,
  watch option forwarding, and benchmark isolation from cwd/user caches.
- Interrupt formatter replacement and verify original bytes/mode survive with no
  temporary leftovers; successful formatting preserves file symlinks and modes.
- Run native and polling watchers through atomic saves, repeated edits, deletion,
  invalid source and subsequent recovery with validation and optimization enabled.
- Reject parameter-free and completely frozen standard/GAN models before loader
  construction; seed Python/NumPy/Torch reproducibly, including a 64-bit seed and
  an environment where NumPy cannot be imported.
- Existing offline training, optimizer equivalence, evaluation, metrics, checkpoint
  resume, gradients, transforms, GAN phase isolation and example suites still pass.

Installed-wheel checks assert site-packages provenance, version agreement, explicit
cache options, structured compilation errors, JSON checks/IR, benchmark behavior,
and the presence of new pipeline modules in the source archive.

## Limits

No dataset downloads, CUDA/MPS execution, long training, model-quality evaluation,
release publication or deployment were performed. Runtime regressions use small
offline CPU fixtures. Python 3.10 local verification is compiler-only; minimum
PyTorch compatibility is covered by the hosted workflow, whose outcome must be
read from the PR checks. No compiler timing/speedup claim is made. Seeding global
RNGs does not guarantee deterministic external data sources or device kernels.
