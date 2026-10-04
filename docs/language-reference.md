# Language reference

This reference describes the current working tree. Aurane is an indentation-based
language; blank lines and comments do not change block nesting. Unknown statements
are errors. Values support numbers, booleans, `None`, escaped strings, nested
lists/tuples, named constants, and named configuration calls. Arbitrary Python
expressions are not evaluated in configuration.

## Constants and imports

```aur
use torch
width = 32
model Net:
    width = 16
    input_shape = (8,)
    def forward(x):
        x -> dense(width)
```

Model constants shadow globals and do not leak into other models. Forward
references are resolved; duplicate, undefined and cyclic constants fail. Quoted
strings remain literal strings. `use package.module as alias` generates a Python
import, so the imported package must exist when the generated module is imported.

## Models and tensor graphs

`input_shape` excludes the batch axis. The legacy default is `(1, 28, 28)`;
specify it explicitly for predictable analysis.

Sequential syntax:

```aur
model Net:
    input_shape = (3, 16, 16)
    def forward(x):
        x -> conv2d(8, kernel=3).relu
          -> maxpool(2)
          -> flatten()
          -> dense(4)
```

Graph syntax:

```aur
model Branched:
    input_shape = (8,)
    def forward(x):
        left = dense(x, 4)
        right = dense(x, 2)
        joined = concat(left, right, dim=1)
        result = dense(joined, 8)
        result = add(x, result).relu
        return result
```

Graph inputs must already be defined. Reassigning a name creates a distinct IR
value; earlier uses keep their original meaning. `return` selects the output;
without it, the last assignment is returned. `add` requires at least two equal
shapes and consumes every input. `concat` requires compatible non-concatenated
axes. Its dimension includes runtime batch: `dim=1` joins vector features and
`dim=-1` joins the last axis. Concatenating the batch axis is unsupported.

Every model consumer uses the shared graph with inferred shapes and dtypes.
Operations have one-based start positions and exclusive end positions in the IR;
columns count source characters, including when strings contain Unicode. Parsed
configuration fields retain their spans through resolution and analysis; JSON
checks expose structured spans alongside diagnostic messages.

`input_dtype` can declare `float16`, `bfloat16`, `float32`, `float64`, `int32`,
`int64`, or `bool`. A declared floating dtype also sets floating model parameters;
integer token inputs use float32 embedding parameters. The generated forward
method checks declared input dtypes without silently casting them. With no
`input_dtype`, the input dtype is unknown statically and floating parameters use
float32. Typed operations propagate known output dtypes and reject incompatible
paths, such as floating-point embedding indices or integer dense inputs. There is
no implicit mixed-dtype promotion in graph merges. Device/operator support for
low precision remains a PyTorch runtime constraint, separate from static checking.

## Implemented operations

| Operation | Contract |
| --- | --- |
| `conv1d(channels, ...)`, `conv2d(channels, ...)` | Input `(channels, length)` or `(channels, height, width)`; integer `kernel=3`, `stride=1`, `padding=0`, `dilation=1`, `groups=1`, `bias=True`; groups must divide input/output channels |
| `lstm(hidden_size, ...)`, `gru(hidden_size, ...)` | Input `(sequence, features)`; output full sequence, with zero initial state each call; `num_layers=1`, `bidirectional=False`, `dropout=0`, `bias=True`; dropout requires multiple layers |
| `upsample(size=..., ...)`, `upsample(scale_factor=..., ...)` | Exactly one size/scale; scalar or one value per spatial axis. Nearest/nearest-exact, linear (1D), bilinear/bicubic (2D), trilinear (3D); linear modes accept `align_corners` |
| `dense(features, bias=True)`, `linear(...)` | Transform the last axis, preserving leading sample axes |
| `maxpool(kernel, stride=kernel)`, `avgpool(...)` | 2D spatial pooling; exact kernel/stride output shape |
| `flatten()` | Flatten sample dimensions, preserve batch |
| `reshape(d1, d2, ...)` | Preserve batch and element count; one inferred `-1` allowed |
| `global_avg_pool()` | Average all spatial axes; preserve channels |
| `dropout(probability)` | PyTorch training/evaluation behavior |
| `batchnorm(...)`, `batch_norm(...)` | Channel normalization; `eps`, `momentum`, `affine`, `track_running_stats` pass to PyTorch |
| `layer_norm(...)`, `layernorm(...)` | Normalize the last feature axis; `eps`, `elementwise_affine`, `bias` pass to PyTorch |
| `embedding(vocabulary, width, padding_idx=...)` | Append embedding features to integer input; embedding options pass to PyTorch |
| `positional_encoding(max_len=...)` | Learned positional parameters, not sinusoidal encoding; input `(sequence, features)` |
| `multihead_attention(heads=..., dim=..., dropout=0, causal=False)` | Batch-first self-attention; feature width must match and divide heads |
| `add(...)`, `concat(..., dim=1)` | Graph merges as described above |

Activations work as calls and suffixes: `relu`, `gelu`, `leaky_relu`, `sigmoid`,
`tanh`, `softmax`, `log_softmax`, `silu`/`swish`, `mish`, `elu`, `selu`, and
`hardswish`. `leaky_relu(slope)` accepts a slope; softmax calls accept a dimension.
`.residual` adds an operation's input to its result, requiring equal shapes.
Suffixes apply consistently to every supported operation.

LayerNorm now normalizes features rather than the whole declared sample shape.
This intentionally changes old sequence-model parameter shapes. Positional
encoding still requires runtime sequence lengths no greater than `max_len`.
Unknown operation/activation names, extra positional arguments and unsupported
keywords fail through a shared catalog used by semantic analysis and generation.
Boolean flags must be actual booleans. Shape-dependent dimensions and ranks are
checked during lowering. Cross-attention is not exposed.
Recurrent operations return sequence outputs only; hidden-state passing and packed
sequences are not exposed. Profiler recurrent FLOPs count matrix multiply-adds,
excluding pointwise gates and bias additions.

## Sequence padding

Models containing attention accept `model(inputs, padding_mask=mask)`, where
`mask` is boolean with shape `(batch, sequence)` and True marks padding. The mask
blocks padded keys and combines with `causal=True`. Attention emits zero at padded
query positions, including fully padded rows, with finite outputs and gradients.
Later layers and residual additions can produce nonzero values there; exclude
padded targets from loss with `ignore_index`.

For token models, set `input_padding_idx = 0` to derive the mask from the original
input tokens. This requires one-dimensional token input and an attention layer.
An explicit mask overrides the derived one. Set the embedding's `padding_idx`
separately if that embedding row should remain fixed. Each attention operation
uses the same input mask, so sequence-changing graph operations must preserve its
alignment. Runtime length may vary within the positional encoding limit.

## Experiment configuration

At most one `experiment` block is allowed. Its fields are `seed` (a nonnegative
64-bit PyTorch seed), `device` (`cpu`, `cuda`, `mps`, or `auto`, with optional device
indices), and `backend = "torch"`. Auto selects CUDA when available, otherwise CPU.
Without an experiment, generated training uses CPU. Set Python/NumPy or custom
dataset seeds externally when those sources also need reproducible initial runs.
Backend plugins are selected through the compiler API/CLI, not the experiment.
Duplicate definitions and imports that replace generated runtime bindings fail.

## Datasets

```aur
dataset images:
    from torchvision.datasets.FakeData
    size = 64
    image_size = (1, 28, 28)
    num_classes = 10
    batch = 16
    shuffle = False
```

Dataset sources are qualified Python constructors. Loader fields are `batch`,
`shuffle`, `num_workers`, `pin_memory`, and `drop_last`; other fields are passed
to the constructor. `shuffle` defaults to `train` when supplied, otherwise True.
Torchvision datasets receive `ToTensor()` unless `transform` is explicitly given.
Declare `transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])` or
`transforms = [ToTensor(), Normalize((0.5,), (0.5,))]`. The latter is shorthand
for Compose; do not specify both. `target_transform` accepts a transform call too.
Supported torchvision transforms are Compose, ToTensor, PILToTensor, Normalize,
Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip,
RandomResizedCrop, ColorJitter, Grayscale, Pad and RandomRotation. Calls may use
resolved constants and literal constructor arguments. Compose can nest. Quoted
call strings and unknown transform names fail. No implicit ToTensor is inserted
into an explicit pipeline. Construction remains lazy; torchvision validates the
individual transform's arguments when the loader is constructed.

Constructor-specific options
must be valid for that dataset: FakeData does not accept MNIST's root/train fields.

Downloads require explicit `download = True`; there is no implicit download.
Dataset construction occurs when training starts or its `_make_NAME_loader()`
factory is called, never on import. A sourceless dataset requires an injected
loader when calling the generated training function.

`aurane.data.SyntheticTokens(size, sequence_length, vocab_size, seed)` supplies
deterministic next-token pairs for offline wiring tests. It does not represent
natural language or establish language-model quality.

## Standard training

```aur
train Net on images:
    validate_on = held_out
    epochs = 5
    loss = cross_entropy
    optimizer = adam(lr=0.001)
    metrics = [accuracy, top5_accuracy]
    scheduler = step_lr(step_size=2, gamma=0.5)
    gradient_clip = 1.0
```

The generated `train_net(train_loader=None, validation_loader=None, model=None,
resume_from=None, test_loader=None)` returns the model and records epochs in
`model.training_history`. Every declared standard/GAN job executes in source order.
Each job constructs a fresh model unless one is injected; repeated names receive
`_2`, `_3`, etc. suffixes to keep distinct callables. Imported modules do not train.
Use separate explicit output directories when jobs should retain separate artifacts.

`test_on = dataset_name` or an injected `test_loader` evaluates once after training
or early stopping, in eval mode with gradients disabled. Results live in
`model.test_metrics` (None if no test loader); test results never drive scheduling
or early stopping. Evaluation uses the final model, not an automatically reloaded
best checkpoint.

Losses are averaged over target elements (non-padding tokens for classification),
not over batches. Weighted CE/NLL uses the sum of target-class weights as the
loss denominator; accuracy and other metrics remain unweighted. Classification logits put classes on the last axis. Empty
training/validation data and nonfinite losses fail explicitly.

Supported losses: `cross_entropy`, `cross_entropy_loss`,
`cross_entropy_with_label_smoothing(smoothing=...)`, `mse`/`mse_loss`,
`mae`/`l1_loss`, `huber`, `bce`/`bce_loss`/`binary_cross_entropy`,
`bce_with_logits`, and `nll`. Named loss options pass to PyTorch; mean reduction
is required. `weight=[...]` and BCE-with-logits `pos_weight=[...]` become device-local
tensors; weights must be finite and positive. Use `ignore_index` for excluded
classification targets. Regression/binary output and target shapes must match;
implicit broadcasting is rejected. Unknown loss options fail at compilation.

Optimizers: `adam`, `adamw`, `sgd`, `rmsprop`, `adagrad`, and `adadelta`, with
literal named options. Bare optimizer names use `lr` from the training block
(default 0.001). Shared validation checks keyword names, booleans, finite numeric
ranges, betas, and Nesterov constraints. `capturable=True` and
`differentiable=True` are not supported by generated training. Hardware-specific
options such as fused optimizers still require native runtime support.

Metrics: multiclass/token `accuracy`, `topN_accuracy` for positive N, and
`perplexity`; regression `mse` and `mae`. Accuracy is a fraction in `[0, 1]`.
Top-k caps k at the available classes. Perplexity is `exp(mean loss)`.
Validation keys begin with `val_`. Binary BCE losses support `accuracy` (also
`binary_accuracy`), with a probability threshold of 0.5 (zero logits).
Both binary and multiclass losses support `precision`, `recall`, `f1`/`f1_score`
and `auc`. Binary targets must be 0 or 1. Multiclass precision/recall/F1 are macro
averages across all output classes; undefined per-class ratios contribute zero.
Binary metrics report the positive class. ROC AUC averages tied ranks and uses
macro one-vs-rest for multiclass; it is None if any required class lacks positives
or negatives. AUC retains epoch predictions on CPU, requiring memory proportional
to samples times classes. Other summary metrics retain only per-class counts.
Unsupported/incompatible metrics fail explicitly.

Schedulers: `step_lr`, `exponential_lr`, `cosine_annealing`,
`reduce_lr_on_plateau`/`reduce_on_plateau`, and
`warmup_cosine(warmup_steps=..., max_steps=...)`. The first four step after each
epoch; plateau receives validation loss when available. Warmup/cosine steps per
successful optimizer update, linearly from zero to the base LR, then decays to
zero at `max_steps`. `0 <= warmup_steps < max_steps` is required.

Unknown training fields are compile errors. Bare scheduler names remain scheduler
requests; they are not discarded. Required arguments and ranges are checked before
generation. Scheduler `last_epoch` must remain -1; use checkpoints to resume.

Both `gradient_clip` and legacy `gradient_clipping` are supported. CUDA AMP
unscales before clipping. CPU AMP requests use full precision. Local QA currently
covers CPU execution, including this fallback; CUDA hardware QA remains open.

## Early stopping and checkpoints

`early_stopping`, `patience`, `min_delta`, `checkpoint_dir`, `checkpoint_every`,
`save_best`, and `resume_from` configure standard training persistence.
Supported callbacks are `early_stopping(patience=..., min_delta=...)` and
`checkpoint(checkpoint_dir=..., checkpoint_every=..., save_best=...)`.
Explicit training fields take precedence over callback options.

Patience counts completed epochs without improvement. Checkpoint directories
receive atomic `last.pt` saves each epoch, optional improving `best.pt`, and
periodic `epoch_NNNN.pt` snapshots. `epochs` is the total target after resume.
See the [resume guide](getting-started.md#checkpoints) for saved state and limits.

## GAN training

```aur
train_gan Generator and Discriminator on images:
    epochs = 2
    generator_optimizer = adam(lr=0.0002)
    discriminator_optimizer = adam(lr=0.0002)
    generator_loss = binary_cross_entropy
    discriminator_loss = binary_cross_entropy
    discriminator_steps = 1
    generator_steps = 1
    generate_samples_every = 1
    num_samples = 16
    save_dir = "./samples"
```

Generator noise follows its declared `input_shape` and floating dtype. Generator
output shape/dtype must be compatible with the discriminator; these boundaries
are checked before generation. The discriminator phase uses
detached generator output. During the generator phase discriminator parameters
are frozen and its running statistics do not update. Each phase honors its step
count. Losses can be BCE, BCE-with-logits, or MSE; discriminator output activation
must match the selected loss. The function returns `(generator, discriminator)`
with shared history containing losses and real/fake mean scores.

Samples use fixed noise and publish atomically. The default `sample_format =
"tensor"` writes CPU tensors to `samples_NNNN.pt`; `"png"` writes image grids and
`"both"` writes both files. PNG output requires one or three image channels and
uses `sample_range = (-1, 1)` for normalization; set `(0, 1)` for sigmoid images.
Image exports use [torchvision save_image](https://docs.pytorch.org/vision/stable/generated/torchvision.utils.save_image.html).

`checkpoint_dir`, optional `checkpoint_every`, and `resume_from` enable GAN resume.
Each completed epoch atomically saves `last.pt`; intervals additionally save
`epoch_NNNN.pt`. The function accepts `resume_from=path`, restoring both models,
both optimizers, history, PyTorch/Python/CUDA and NumPy RNG, loader/sampler
generators, and supported dataset state hooks.
Model definitions, loss choices, optimizer classes, update ratios and metric
selection must match. `epochs` remains the total target. There is no automatic
"best GAN" criterion. `metrics` selects history fields from generator_loss,
discriminator_loss, real_score and fake_score; the default records all four.
The same data-state requirements and worker restrictions as standard checkpoints
apply; see the [resume guide](getting-started.md#checkpoints).

## Compiler behavior and migration

`compile_source` resolves constants and validates model graphs by default. Full
semantic analysis and additional type checks remain optional. Optimization levels
1 and 2 currently perform the same proven duplicate-ReLU elimination; training
layers and parameter state are retained. Level 0 performs no rewrites.

Cache records are versioned and integrity checked; corruption is a cache miss.
Compiled outputs and cache records publish atomically. A custom registered backend
is uncached unless it declares `cache_version`; bump that version when its behavior
changes. Backend callables still receive the resolved AST.

Previously ignored malformed syntax and unsupported operations now fail. Dataset
construction moved to factories. LayerNorm's default feature axis and reshape's
batch preservation intentionally change previously broken behavior. Legacy cache
entries are not reused. Use the audit for remaining unsupported options rather
than assuming that accepting a configuration key proves it is implemented.
