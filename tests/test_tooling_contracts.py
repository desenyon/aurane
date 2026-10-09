"""Developer commands must preserve final output and report truthful measurements."""

from argparse import Namespace
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from aurane.parser import parse_aurane
from aurane.profiler import ModelProfiler, format_profile
from tests.test_cli_contracts import ROOT, SOURCE, cli


def test_profiler_is_repeatable_and_labels_requested_batch():
    profiler = ModelProfiler(parse_aurane(SOURCE).models[0])
    first = profiler.profile_model(batch_size=2)
    second = profiler.profile_model(batch_size=8)
    assert first.total_params == second.total_params == 10
    assert len(first.layers) == len(second.layers) == 1
    assert second.total_memory_bytes == 4 * first.total_memory_bytes
    assert "batch=8" in format_profile(second)


@pytest.mark.parametrize("batch", [0, -1])
def test_profiler_rejects_nonpositive_batches(batch):
    with pytest.raises(ValueError, match="batch"):
        ModelProfiler(parse_aurane(SOURCE).models[0]).profile_model(batch_size=batch)


@pytest.mark.parametrize("failure", [RuntimeError("launch failed"), KeyboardInterrupt()])
def test_run_cleans_temporary_file_on_launch_failure_or_interruption(
    tmp_path, monkeypatch, failure
):
    from aurane.cli.commands import run

    source = tmp_path / "model.aur"
    source.write_text(SOURCE)
    temporary = tmp_path / "generated.py"
    temporary.write_text("pass")
    monkeypatch.setattr(run, "compile_to_temp", lambda *args, **kwargs: temporary)

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(run.subprocess, "run", fail)
    result = run.cmd_run(Namespace(input=str(source), backend="torch", keep_temp=False))
    assert result == (130 if isinstance(failure, KeyboardInterrupt) else 1)
    assert not temporary.exists()


def wait_for_output(path, fragment, process):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if path.exists() and fragment in path.read_text():
            return
        assert process.poll() is None, "watch exited before compiling the change"
        time.sleep(0.05)
    pytest.fail(f"watch did not produce {fragment!r}")


@pytest.mark.parametrize("poll", [False, True])
def test_watch_handles_atomic_saves_bursts_and_invalid_then_valid_source(tmp_path, poll):
    source, output = tmp_path / "model.aur", tmp_path / "model.py"
    source.write_text(SOURCE)
    with (tmp_path / "watch.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "aurane.cli",
                "watch",
                str(source),
                str(output),
                "--validate",
                "--optimize",
                "--no-cache",
                *(["--poll"] if poll else []),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=log,
        )
        try:
            wait_for_output(output, "Linear(4, 2", process)
            replacement = tmp_path / "replacement"
            replacement.write_text(SOURCE.replace("dense(2)", "dense(3)"))
            replacement.replace(source)
            wait_for_output(output, "Linear(4, 3", process)
            source.write_text(SOURCE.replace("dense(2)", "dense(4)"))
            wait_for_output(output, "Linear(4, 4", process)
            source.write_text("invalid source")
            time.sleep(0.6)
            assert "Linear(4, 4" in output.read_text()
            source.unlink()
            source.write_text(SOURCE.replace("dense(2)", "dense(4)"))
            source.write_text(SOURCE.replace("dense(2)", "dense(5)"))
            wait_for_output(output, "Linear(4, 5", process)
        finally:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=5)
        assert process.returncode == 0


def test_benchmark_measures_cold_and_warm_compile_without_leftovers(tmp_path, monkeypatch):
    from aurane.cli.commands import benchmark

    source = tmp_path / "model.aur"
    source.write_text(SOURCE)
    monkeypatch.chdir(tmp_path)
    results = []
    monkeypatch.setattr(
        benchmark, "show_benchmark_results", lambda timings, _: results.append(timings)
    )
    assert benchmark.cmd_benchmark(Namespace(input=str(source), iterations=2)) == 0
    assert set(results[0]) == {"parse", "cold_compile", "warm_compile"}
    assert all(len(values) == 2 for values in results[0].values())
    assert sorted(path.name for path in tmp_path.iterdir()) == ["model.aur"]


@pytest.mark.parametrize("iterations", [0, 1])
def test_benchmark_json_reports_validated_outcomes(tmp_path, iterations):
    import json

    source = tmp_path / "model.aur"
    source.write_text(SOURCE)
    result = cli("benchmark", source, "--iterations", iterations, "--json")
    payload = json.loads(result.stdout)
    assert payload["ok"] == bool(iterations)
    assert result.returncode == (0 if iterations else 1)
    if iterations:
        assert set(payload["seconds"]) == {"parse", "cold_compile", "warm_compile"}
    else:
        assert "positive" in payload["error"]


def test_lint_autofix_preserves_comments_and_only_publishes_parseable_fixes(tmp_path):
    source = tmp_path / "model.aur"
    source.write_text(SOURCE.replace("model Net:", "model Net # header"))
    assert cli("lint", source, "--auto-fix").returncode == 0
    assert source.read_text().startswith("model Net: # header\n")
    repaired = source.read_text()
    assert cli("lint", source, "--auto-fix").returncode == 0
    assert source.read_text() == repaired
    invalid = "model Net:   \n    invalid statement\n"
    source.write_text(invalid)
    assert cli("lint", source, "--auto-fix").returncode == 1
    assert source.read_text() == invalid


def test_interactive_multiline_error_recovery_and_exit():
    transcript = "invalid source\n.compile\n.clear\n" + SOURCE + ".compile\n.exit\n"
    result = subprocess.run(
        [sys.executable, "-m", "aurane.cli", "interactive"],
        input=transcript,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=15,
    )
    assert result.returncode == 0
    assert "Error" in result.stdout and "Buffer cleared" in result.stdout
    assert "class Net" in result.stdout and "Goodbye" in result.stdout
