# Aurane rework: audit, delivery plan, and acceptance gates

Status: local implementation and CPU verification complete; release gates open. Baseline: `5941e3e` (2.0.0), inspected 2026-10-02.

## Product and implementation

Aurane is an indentation-based ML language implemented in Python. The parser
builds dataclass AST nodes for imports, configuration, datasets, models, and
training. Models support sequential chains and explicit tensor graphs. The
compiler optionally runs semantic checks, shape checks, and AST optimization,
then calls a registered backend. The bundled PyTorch backend now lowers each
model through shared graph IR annotated with shapes, dtypes and operation spans. Type checking, profiling,
visualization and the `ir` command use that same graph contract. Backend plugins
still receive an AST; changing the plugin API is a separate compatibility migration.

Working product assumption: keep `.aur` concise and generate readable, runnable
PyTorch. Completing existing behavior takes priority over new backends or a
hosted experiment platform. “Every feature” means the language constructs,
operations, training options, commands, and workflows already exposed in this
repository. Expansion requires a separately specified contract.

## Baseline evidence

- 265 tests pass when the virtual environment is on PATH. Invoking its Python
  directly initially produced 24 failures because CLI tests hard-code `python`.
- Existing optimizer tests mostly assert that a result exists; they do not
  compare numerical outputs, gradients, or module state.
- CI installs compiler/dev dependencies only, so generated programs never run
  against PyTorch. Type checking is explicitly allowed to fail. Only Python 3.11
  is tested despite the declared Python 3.10+ support.
- The graph architecture request eventually reported Aurane was not indexed.
  Initial inspection used local source; a fresh index has been requested.

## Full feature inventory and backlog

The table records baseline gaps; the execution log below records repairs.
Unfinished acceptance criteria remain open even when a subset is repaired.
Source-level findings are separate from runtime proof. Each wave is completed
and reviewed before the next.

| ID | Area and source | Verified gap / required work | Acceptance and QA |
| --- | --- | --- | --- |
| F01 | Parser: `parser.py` | Indented unknown statements can be skipped; regex calls accept trailing text; blank/comment lines influence block detection | Valid corpus parses; malformed syntax, indentation, incomplete calls and stray statements produce located errors; no silent omission |
| F02 | Values and symbols: `parser.py`, `codegen_torch.py` | Globals/model config values are parsed but symbolic layer arguments are not resolved; nested/string values use ad hoc splitting | Nested literals, escaped strings, symbolic dimensions, unresolved and cyclic names have deterministic tests |
| F03 | AST diagnostics: `ast.py` | Location fields exist but are inconsistently populated; expressions have no explicit reference type | Every diagnostic identifies source span; distinguish literal strings and symbol references |
| F04 | Semantic analysis: `semantic_analyzer.py` | Operation/activation catalogs exceed backend support; unknown operations only warn; GAN and validation references incompletely checked | Every accepted operation executes or produces a clear unsupported error; references, duplicate names, argument types/ranges and unknown options tested |
| F05 | Shapes: `shapes.py`, `type_checker.py` | Dense collapses leading axes; pooling ignores kernel in output size; invalid ranks/dimensions often pass through | Compare shapes with real PyTorch; invalid stride/rank/reshape/attention/residual cases fail before generation |
| F06 | Graph wiring: type checker, generator, profiler, visualizer | Concat uses runtime dimension on shapes without batch; add ignores inputs beyond the first two; merge validation differs across consumers | Branch/add/concat outputs, gradients, shapes and reports agree; negative/out-of-range axes and mismatches tested |
| F07 | Optimizer: `optimizer.py` | Deletes dropout; removes non-equivalent activations; emits unsupported conv2d_bn; pooling leaves ignored skip metadata | Compare unoptimized/optimized train and eval outputs, gradients, buffers, parameter topology; only proven rewrites enabled |
| F08 | IR and backend registry | IR is an inspection side path; mutable graph values lack SSA identity on reassignment | Make typed, source-located IR the shared execution contract; validate uses/definitions; keep backend API migration explicit |
| F09 | Compiler/cache: `compiler.py` | Validation opt-in; cached code unversioned relative to compiler changes; corrupt reads fail; outputs not atomic | Cache version/options/backend isolation; corruption/read-only/concurrency tests; deterministic output; failed compile preserves prior artifact |
| F10 | Core PyTorch layers | Dense uses first axis; several bias/normalization options ignored; unknown ops emit undefined functions; activation suffixes inconsistently applied | Per-operation forward/backward tests against native modules, aliases and keyword coverage |
| F11 | Sequence models | LayerNorm includes sequence length; positional parameter counts omitted; embedding symbols unresolved | Variable-length token batches, attention masks, residual shapes and sequence loss tested on a small offline model |
| F12 | Datasets/imports | Generator forces torchvision and assumes every dataset accepts root/train/download; `transforms` missing when torchvision explicitly imported; construction occurs on import | Model-only modules require only torch; dataset-specific arguments; no downloads on import; deterministic offline fixtures and train/validation separation |
| F13 | Standard training | No default device without experiment; unknown loss/optimizer silently replaced; plateau scheduler reads avg_loss before assignment; gradient_clip alias ignored | Offline training changes weights, finite loss, correct device/loss/optimizer/scheduler/gradient clipping; empty datasets handled explicitly |
| F14 | Metrics/evaluation | Parsed metrics ignored; validation always computes classification accuracy | Accuracy/top-k/perplexity and regression metric contracts; weighted aggregates and task-compatible loss/shape handling |
| F15 | Callbacks/checkpointing | Parsed callbacks, early stopping and checkpoint settings are ignored | Best/last checkpoints, restore model/optimizer/scheduler/RNG, deterministic resume, patience and interrupted-write tests |
| F16 | Mixed precision | CUDA-only legacy generated AMP path; device behavior unspecified | CPU fallback, supported CUDA AMP, unscale-before-clipping; optional hardware tests clearly separated |
| F17 | GAN training | Generated code reads nonexistent netG.config; losses/step ratios/sample saving ignored | One offline G/D update, latent shape, detached D phase, configured ratios/losses, samples and checkpoints |
| F18 | Profiler | Dense sequence FLOPs and positional parameters wrong; graph inference duplicated; repeated calls accumulate; memory label always batch=1 | Params equal sum of runtime parameters; FLOP convention stated; batch/dtype/activation-memory limits explicit; stable repeated reports |
| F19 | Visualization | Separate graph shape inference; rendering/escaping and graph edge coverage need verification | Snapshot plus structural tests for branches, graph returns, aliases, labels, shapes, params, Mermaid and DOT escaping |
| F20 | compile CLI | stdout mixed with progress; formatting can mistake Black NothingChanged for failure; output handling duplicated | stdout parses as Python without --quiet; diagnostics on stderr; diff/format/quiet/error exit codes and preserved output tested |
| F21 | check CLI | --json parse/IO failures output prose | JSON parses for success, syntax/type/semantic failures and missing paths; stable codes and truthful selected-pass summary |
| F22 | inspect / ir / profile / visualize CLI | Coverage largely smoke/string assertions | Every output format parsed and checked against known models; file paths, empty source, invalid graphs and failures covered |
| F23 | run CLI | Temporary cleanup not guaranteed on interruption; configuration/validation options limited | Child exit/signal propagation, cleanup and --keep-temp; run offline starter in subprocess |
| F24 | watch CLI | Only modified events; editor atomic saves can be missed; time debounce can discard last update | Atomic replace/create/delete-recreate, burst updates, recoverable invalid input, shutdown and final artifact tests |
| F25 | format CLI | Resets def forward to top level, changing program meaning | Preserve parse tree; idempotence; graph/train_gan/comments/strings; --check never writes |
| F26 | lint CLI | Limited rules and fixes | Document rule IDs/severity; fixes preserve semantics, are idempotent and never rewrite literals |
| F27 | init / interactive CLI | Starter has sourceless dataset; interactive behavior needs end-to-end proof | init -> check -> compile -> offline run succeeds; interactive multiline/error/recovery/exit transcript tests |
| F28 | benchmark CLI | Measures cached compile and double-counts parse; no positive iteration check | Separate cold/warm measurement, validated iterations, no leftovers, machine-readable result |
| F29 | clean CLI | Recursively targets .so/.pyd and *_temp.py, including dependencies/user files | Delete only documented compiler artifacts; prune .git/venvs; symlink/dry-run tests with sentinel files |
| F30 | Examples | Simple uses invalid FakeData arguments; MNIST imports nonexistent aurane.ml; ResNet has no real skip links; Transformer invalid WikiText2 source; GAN runtime broken | All five examples parse/check/generate; offline scaled runtime fixtures; examples named for their actual architecture and dependencies |
| F31 | Documentation | Stale 0.2 guide; claims exceed execution evidence; placeholder GitHub links | Rewrite tutorial/reference/how-to around tested features; executable snippets; explicit limitations and upgrade notes |
| F32 | Packaging and CI | One Python version, optional typing failure, no runtime/package-install QA, publish pipeline separate from tests | Required compiler and runtime jobs, supported Python matrix, wheel install in clean env, published artifacts gated by passing checks |

## Delivery sequence

1. **Correctness foundation (F05/F07/F10/F18/F32):** reproduce shape and optimizer
   failures with runtime tests, repair them, make CLI tests interpreter-safe,
   and add a required PyTorch test job. Gate: full suite plus numerical parity.
2. **Language contract (F01-F06/F08/F09):** strict parsing, symbol resolution,
   typed graph analysis, common IR and compiler diagnostics. Gate: valid/invalid
   corpus, graph parity, cache/atomic-write tests and migration review.
3. **Runnable product (F11-F17/F27/F30):** dataset construction, standard and GAN
   training, metrics, callbacks, checkpoints, sequence models and offline
   starters. Gate: train/evaluate/save/resume workflows with finite results.
4. **Developer tools (F18-F29):** shared analysis and reliable command IO/file
   behavior. Gate: subprocess acceptance matrix including failure paths,
   formatting semantic preservation, watcher events and cleanup sentinels.
5. **Release quality (F31/F32):** rebuild documentation and examples, enforce
   typing/package gates and test supported environments. Gate: clean-install
   tutorial replay, complete feature matrix, no silent unsupported behavior.

## Definition of done

Each item needs implementation, positive and negative tests, executed QA evidence,
documentation, and a reviewed diff. Passing old tests or generated Python syntax
alone is insufficient. Runtime tests use tiny offline tensors/datasets; real
downloads and long training are separate opt-in workflows. No training-quality
or performance improvement claim is made without a controlled measurement.

## Execution log

- Baseline: 265 existing tests passed with the venv on PATH. Runtime dependencies
  absent initially; installed locally for numerical verification.
- Correctness foundation repair completed: dense leading dimensions, exact
  pooling sizes, average-pool propagation, bias flags, positional feature axis
  and parameter counts, sequence dense FLOPs, interpreter-safe CLI tests,
  cache schema invalidation, and removal of unsafe optimizer rewrites.
  The initial 34 runtime cases reproduced 25 failures before repair; all pass
  after repair. The max-pooling fixture was then strengthened to use even
  spatial dimensions so that the previous shape formula cannot pass by accident.
- Urgent CLI preservation repairs completed after review of the first wave:
  source formatting preserves block structure; cleanup preserves user files,
  environments and symlinks; compile stdout is Python; check/IR JSON handles
  failure paths; compile failures use stderr and preserve existing output.
  The new 19-case subprocess/preservation suite reproduced 16 failures before
  repair, then passed in full.
- Four additional checks cover optimizer input immutability/idempotence,
  invalid optimization levels, graph/literal formatting, and repeated Black
  formatting.
- Local verification: 322 tests passed on Python 3.13.12 / PyTorch 2.14.1;
  mypy passed all 34 source files; Black passed all 49 source/test files;
  `git diff --check` passed. CI is configured for compiler tests on Python
  3.10–3.13, a separate CPU PyTorch runtime job, and mandatory type checking.
  Those hosted jobs have not been executed in this session.
- Built the wheel and source distribution, installed the wheel into a fresh
  environment outside the repository, and verified version, JSON check, Python
  stdout compilation, and JSON IR commands. All five bundled examples passed
  semantic/type checks and generated-Python syntax checks at optimization levels
  0, 1, and 2. This is not evidence that their training workflows execute.
- Language repair: strict block parsing, balanced/nested literals and calls,
  comment/string handling, preserved activation chains, located syntax errors,
  separate symbol references, global/model constant resolution with shadowing,
  and undefined/duplicate/cyclic constant diagnostics. Analysis commands use
  resolved values. The 25-case parser suite initially reproduced 24 failures;
  symbol, escaped-literal, scheduler and CLI regressions were also reproduced
  before repair. Verification at this boundary: 364 tests and mypy passed.
- Graph repair: shared shape-annotated IR now preserves distinct values through
  reassignment, validates tensor definitions/returns/merges, interprets concat
  dimensions with the runtime batch axis, and generates all operands of add.
  Diagrams reflect actual branches and selected returns. Rich SVG export and
  shape plots use the same graph. Runtime tests compare values and gradients.
  Full dtype propagation and exact expression spans remain open.
- Compiler IO repair: JSON cache records have content integrity checks and
  versioned SHA-256 keys; unreadable/corrupt entries are misses. Custom backends
  opt into caching with `cache_version`. Cache and compiled-file writes publish
  atomically; the CLI/API reject overwriting source. Tests reproduced truncated
  results under concurrent compilation and verified preserved output on failed
  publication. Verification at this boundary: 395 tests and mypy passed.
- Offline training foundation: lazy dataset factories pass explicit constructor
  settings, model-only modules no longer import torchvision implicitly, and
  training accepts injected models/loaders. No experiment is required. Added
  sample-weighted loss, clipping aliases, plateau ordering, nonfinite/empty data
  diagnostics, modern AMP with CPU full-precision fallback, and errors for unknown
  losses/optimizers/schedulers. The starter completes offline training. All ten
  new runtime tests failed before repair, then passed; full suite: 405 tests.
- Metrics: accuracy/top-k/perplexity and MSE/MAE aggregate across uneven batches;
  validation uses the same task-specific loss and metrics. Token classification
  uses the last class axis and excludes padding from loss weights and accuracy.
  Unknown/incompatible metric requests fail explicitly. Six additional failures
  were reproduced and repaired; the 17-case training suite passes locally.
- Still open: complete operation/rank/argument validation, advanced sequence
  behavior, GAN checkpoint/resume, wider metric coverage, multi-training-block
  execution, full dtype/source-span diagnostics, and release acceptance gates.
  These repairs do not constitute completion of the full rework.
- Standard checkpoints/callbacks: atomic best/last/periodic snapshots, patience
  and minimum improvement, restored optimizer/scheduler/scaler/history/RNG state,
  and generated-model compatibility checks. Exact split/uninterrupted dropout
  training matches in weights and history. Nine checkpoint tests pass, including
  interrupted saves and symbolic callback arguments. Full suite: 421 tests.
- Training completion work: per-update warmup/cosine scheduling, configured GAN
  losses and update ratios, separated generator/discriminator gradient phases,
  fixed-noise sample tensor exports, feature-axis LayerNorm, embedding padding,
  causal attention, uniform activation suffixes, and batch-preserving reshape.
  Invalid ranks, empty spatial outputs, reshape sizes and unknown operation names
  now fail before execution. Fourteen sequence/layer failures were reproduced.
- Tooling repairs: profiler calls no longer accumulate and label the requested
  batch; run cleans temporary files on launch failure and interruption; watch
  handles atomic saves and recompiles the last edit in a burst; benchmark separates
  parse/cold/warm timings with JSON and an isolated temporary cache. Lint inserts
  colons before comments and publishes only parseable fixes. Interactive input
  now preserves indentation. Subprocess tests cover both repairs and recovery.
- Examples and guides: all five training paths execute on small offline fixtures.
  Removed the nonexistent MNIST import, made downloads explicit, replaced the
  nonexistent WikiText source with deterministic synthetic tokens, and implemented
  actual ResNet skip connections. Rewrote getting-started, language and example
  guides around tested contracts and explicit limitations.
- Latest full local verification before package QA: **459 tests passed**, Black
  checked the source/tests, and mypy passed 36 source files. Optional PyTorch
  dependency stubs are excluded from compiler typing; generated PyTorch behavior
  is covered by runtime tests. CI runtime coverage includes examples and resume.
- Package/command QA found and repaired missing examples/guides in the source
  distribution and silent multi-model diagram-file overwrites. Added explicit
  model selection and a preservation regression test. Final suite: **460 passed**;
  mypy: 36 source files; Black: 61 source/test/script files; diff check passed.
  Built wheel/sdist, installed the wheel into a fresh environment, and verified
  import provenance, version/check/compile/IR/benchmark commands and archive
  contents. Added a corresponding CI package job. Ran both unmodified offline
  examples through `aurane run`, completing two epochs each.

- Continuation, 2026-10-03: all standard and GAN jobs now run in source order;
  repeated/case-colliding training names receive unique functions. Three failing
  regressions were reproduced before repair.
- Shared operation argument catalog now rejects unknown keywords, extra positional
  arguments and invalid boolean flags through both analysis and generation.
  Conv1d, grouped/dilated convolutions, LSTM/GRU and upsampling now execute with
  shape, parameter-count, numerical and gradient proof. Profiler multiply-add
  estimates cover convolution groups and recurrent matrices. Additional range/rank
  failures and missing GAN/evaluation references were reproduced and repaired.
- Standard training now evaluates held-out test data once after training/early
  stopping; results never influence selection. Weighted CE/NLL denominators are
  distinct from unweighted metrics. Loss weights become device-local tensors;
  accidental target broadcasting and unknown training fields fail explicitly.
  Bare schedulers are preserved rather than discarded by the parser.
- GAN checkpoint/resume restores both models/optimizers, history and RNG, including
  an explicit shuffled-loader generator. Split/uninterrupted runs match exactly
  in weights, histories and samples. Sample tensors and optional PNG grids publish
  atomically. Metric selection is honored; invalid configuration fails.
- Declarative torchvision transforms now construct lazily, including nested Compose
  and constants. The GAN example normalizes real images to its generator's range.
  Binary/multiclass precision, recall, F1 and ROC AUC aggregate across the full
  dataset; tests include uneven batches, ties, ignored targets and undefined AUC.
- Local verification: **540 tests passed on both Python 3.13.12 / PyTorch 2.14.1
  and Python 3.10.20 / PyTorch 2.4.0**, with torchvision 0.29.1 and 0.19.0
  respectively. Mypy passed 39 source files; Black checked 71 files. Release
  configuration now depends on the reusable quality workflow and verifies the
  exact wheel before uploading artifacts. Hosted execution remains unverified.

- Further continuation, 2026-10-03: shared configuration validation now covers
  loss/optimizer/scheduler keywords and ranges, callbacks, experiment settings,
  duplicate definitions and generated binding collisions. Checkpoint callbacks
  honor their configured interval and explicit overrides. GAN model boundaries
  reject shape/dtype/noise mismatches before execution.
- Dtype-aware model lowering now serves generation, checks, IR, profiling and
  diagrams. Explicit floating input types set parameter types; token embeddings
  transition integer inputs to float outputs. Runtime tests cover all four floating
  dense dtypes, float64 weighted training and GAN noise. Profiling uses dtype bytes.
- Attention supports explicit boolean padding masks and automatic token masks.
  Native attention parity, causal/noncausal behavior, left/right/full padding,
  masked-value invariance and finite gradients are verified.
- Checkpoints additionally save NumPy global RNG, train/validation loader and sampler
  generators, and paired state hooks. Exact resume matches stochastic training and
  validation. Persistent workers and stateful datasets with workers fail explicitly.
- Parser/IR now preserve exact operation start/end columns, repeated calls, suffix
  spans and Unicode-aware symbol columns. Six source-location failures were
  reproduced and repaired. Configuration fields retain spans through resolution;
  analysis and JSON CLI errors now expose structured positions for configuration,
  model operations, references and syntax failures. Shared validation replaced
  the analyzer's duplicated optimizer/loss warning catalogs.
- Final local verification: **643 tests passed** on both current and minimum
  PyTorch environments; compiler-only environment: **480 passed, 20 skipped**.
  Mypy passes 42 source files, Black checks 80 files, and diff whitespace checks
  pass. Fresh distribution installation verifies dtype/span JSON and failure paths
  as well as the existing compiler command contract.

## Remaining acceptance gates

The supported local implementation is covered by the suites listed in the QA report.
Release readiness is still subject to these gates:

1. Execute the hosted Python 3.10–3.13 matrix and release workflow, plus CUDA
   hardware QA. Local minimum/current CPU runtime tests are recorded separately.
   Publication remains separate from this unreleased working tree.
2. Review release compatibility and versioning for the accumulated language and
   generated-code changes before publishing. Backend plugins still consume AST;
   a typed-IR plugin API would require its own migration contract.

External streams, persistent-worker private state, cross-attention, packed recurrent
sequences and automatic mixed-dtype promotion are not exposed contracts. Supported
data-state hooks and explicit rejection of unreplayable worker settings replace the
previous ambiguous resume guarantee. No general third-party state replay is claimed.

See [QA evidence](qa-report.md) for the exact local scope. Passing these tests does
not establish learning quality, GPU correctness, or release readiness.
