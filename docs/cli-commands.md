# CLI reference

Use `aurane COMMAND --help` for the installed version's exact flags.
The commands below describe Aurane 3.0.0.

## Compile and check

```bash
aurane compile model.aur model.py --analyze --validate
aurane compile model.aur > model.py
aurane check model.aur --json
```

`compile` accepts a positional output or `-o/--output`. Without an output path,
stdout is Python and diagnostics go to stderr. `--quiet` reduces progress;
`--show-ast` prints the parsed AST; `--diff` shows changes against an existing
output. `--format` requires Black. `--analyze` adds semantic analysis and
`--validate` adds type checks; constant resolution and graph lowering always run.
`--optimize --opt-level 0|1|2` controls the conservative AST optimizer.
The bundled backend is `--backend torch`.

Compilation failures preserve existing output. Successful file publication is
atomic. Using the source file itself as output is rejected.

`check` runs both semantic and type passes by default. `--semantic` or `--types`
selects only that pass; combining them runs both. `--json` emits machine-readable
success and failure payloads, including missing files and parse errors. Located
diagnostics carry `span` with one-based `line`, `column`, `end_line`, and exclusive
`end_column`; errors without source locations (such as missing files) use null. Exit 0
means no selected-pass errors, not proof that all runtime options are implemented.

## Inspect, IR, profile and visualize

```bash
aurane inspect model.aur --verbose --stats --export ast.json
aurane ir model.aur --model Net --format json
aurane profile model.aur --batch-size 8 --detailed
aurane visualize model.aur --format mermaid --model Net --output net.mmd
```

`inspect` exposes the parsed AST; verbose summaries resolve constants. `ir`
returns graph nodes with inferred shapes/dtypes, operation source spans and distinct
value identities. Unknown input dtypes are null in JSON.
Its formats are `json` (default) and `text`.

`profile` reports parameter counts, estimated FLOPs per sample, and the sum of
layer-output storage using each inferred dtype and the requested positive batch
size. Unknown dtypes use a documented four-byte estimate. It does
not measure latency or peak training memory and excludes optimizer state and
allocator behavior. Repeated profiling does not accumulate prior results.

`visualize` formats are `rich`, `mermaid`, and `dot`. Rich `--output` exports SVG.
Graphs show actual branch edges and the selected return tensor. When exporting a
file from a source containing multiple models, choose `--model`; the command fails
without overwriting the file if the selection is ambiguous. Select one model for
a standalone Mermaid/DOT document on stdout as well.

## Run and watch

```bash
aurane run examples/simple.aur
aurane run model.aur --keep-temp
aurane watch model.aur model.py --analyze
```

`run` compiles to a temporary Python file, executes it with the CLI's interpreter
and the source directory as working directory, and removes it afterward.
`--keep-temp` preserves the script. Child failures propagate as nonzero status;
interruption returns 130 and still cleans up the temporary file.

`watch` performs an initial compile and handles modification, creation, deletion
and atomic replacement events. It waits briefly for a burst to settle and compiles
its final contents. Invalid edits preserve the previous artifact; a later valid
save recovers. Ctrl+C shuts down the observer.

## Format and lint

```bash
aurane format examples/ --check
aurane format model.aur
aurane lint model.aur --auto-fix
```

`format` trims trailing whitespace and ensures a final newline, preserving block
indentation, strings and comments. It accepts one file or a directory.
`--check` does not write and exits 1 if formatting would change a file.

`lint` reports these rules:

| Rule | Level | Behavior |
| --- | --- | --- |
| W001 | Info | Trailing whitespace; safely auto-fixable |
| W002 | Warning | Missing block colon; inserted before an inline comment |
| W003 | Warning | Lines longer than 100 characters; no automatic wrapping |
| E001 | Error | Source does not parse |

Fixes are published only if the resulting source parses. Repeated fixes are
idempotent. Lint returns 1 for errors; warnings alone return 0. Use `check` for
semantic and shape diagnostics.

## Benchmark

```bash
aurane benchmark model.aur --iterations 20
aurane benchmark model.aur -i 5 --json
```

Iterations must be positive (default 10). Parsing, complete cold compilation,
and warm cache reads are measured separately. Cold compilation already includes
parsing, so these timings must not be summed. Timing excludes source/output file
IO. The warm cache lives in an isolated temporary directory that is removed.
Text output reports mean, median, standard deviation, min and max; JSON returns
raw durations in seconds under `parse`, `cold_compile`, and `warm_compile`.
This measures compiler behavior, not model execution speed.

## Scaffold, interactive mode and cleanup

```bash
aurane init demo
aurane interactive
aurane clean . --dry-run
```

`init` creates an offline FakeData starter and README; it refuses a nonempty target
directory. `interactive` preserves typed indentation. Commands are `.help`,
`.show`, `.compile`, `.clear`, and `.exit`; a compilation error does not exit the
session. Interactive compilation displays Python without executing it.

`clean` removes recognized cache directories and Python bytecode. It preserves
user source, generated Python, native libraries, `*_temp.py`, virtual environments,
version-control directories and symlink targets. Preview with `--dry-run`.
