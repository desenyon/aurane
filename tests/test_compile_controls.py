"""Compiler controls behave identically across API and CLI entry points."""

from argparse import Namespace
import json
from pathlib import Path

import pytest

from aurane import compiler
from tests.test_cli_contracts import SOURCE, cli


def test_cache_directory_precedence_and_disabled_io(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env_cache, explicit = tmp_path / "env", tmp_path / "explicit"
    monkeypatch.setenv("AURANE_CACHE_DIR", str(env_cache))
    expected = compiler.compile_source(SOURCE, cache_dir=explicit)
    assert len(list(explicit.glob("*.json"))) == 1
    assert not env_cache.exists()
    assert compiler.compile_source(SOURCE) == expected
    assert len(list(env_cache.glob("*.json"))) == 1
    assert not (tmp_path / ".aurane_cache").exists()

    def fail(*args, **kwargs):
        pytest.fail("Disabled caching must not read or write files")

    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(compiler, "_atomic_write", fail)
    assert compiler.compile_source(SOURCE, disable_cache=True, cache_dir=explicit) == expected


def test_temp_compilation_uses_all_options_and_does_not_leave_failed_files(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    source = SOURCE.replace("dense(2)", "dense(2).relu -> relu()")
    cache = tmp_path / "cache"
    temporary = compiler.compile_to_temp(
        source, analyze=True, validate=True, optimize=True, opt_level=2, cache_dir=cache
    )
    try:
        assert temporary.read_text().count("F.relu") == 1
        assert len(list(cache.glob("*.json"))) == 1
    finally:
        temporary.unlink()
    with pytest.raises(compiler.CompilationError):
        compiler.compile_to_temp(
            source.replace("dense(2)", "reshape(3)"), validate=True, disable_cache=True
        )
    assert list(tmp_path.glob("*.py")) == []


@pytest.mark.parametrize("command", ["compile", "run"])
def test_cli_json_errors_preserve_stdout_and_stage(tmp_path, command):
    path = tmp_path / "model.aur"
    path.write_text(SOURCE.replace("dense(2)", "reshape(3)"))
    result = cli(command, path, "--validate", "--no-cache", "--diagnostics-format", "json")
    assert result.returncode == 1
    assert result.stdout == ""
    error = json.loads(result.stderr)["error"]
    assert error["stage"] == "type"
    assert error["source"] == str(path)
    assert error["diagnostics"][0]["span"]["line"] == 4


def test_compile_cli_uses_explicit_cache_and_optimization(tmp_path):
    path = tmp_path / "model.aur"
    path.write_text(SOURCE.replace("dense(2)", "dense(2).relu -> relu()"))
    cache = tmp_path / "cache"
    result = cli(
        "compile",
        path,
        "--analyze",
        "--validate",
        "--optimize",
        "--opt-level",
        "2",
        "--cache-dir",
        cache,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("F.relu") == 1
    assert len(list(cache.glob("*.json"))) == 1


def test_watch_forwards_compiler_controls():
    from aurane.cli.commands.watch import _compile_args_from_watch_args
    from aurane.cli.compilation import compiler_options

    args = Namespace(
        input="in.aur",
        output="out.py",
        backend="torch",
        analyze=True,
        validate=True,
        optimize=True,
        opt_level=2,
        disable_cache=True,
        cache_dir="somewhere",
        diagnostics_format="json",
        format=True,
    )
    forwarded = _compile_args_from_watch_args(args)
    assert compiler_options(forwarded) == compiler_options(args)
    assert forwarded.diagnostics_format == "json"
    assert forwarded.format is True


def test_benchmark_ignores_user_cache_and_never_changes_cwd(tmp_path, monkeypatch):
    from aurane.cli.commands import benchmark
    import os

    path = tmp_path / "model.aur"
    path.write_text(SOURCE)
    cache = tmp_path / "user-cache"
    monkeypatch.setenv("AURANE_CACHE_DIR", str(cache))
    original = Path.cwd()
    monkeypatch.setattr(os, "chdir", lambda *args: pytest.fail("benchmark changed process cwd"))
    assert benchmark.cmd_benchmark(Namespace(input=str(path), iterations=1, json=True)) == 0
    assert Path.cwd() == original
    assert not cache.exists()


def test_formatter_failed_replacement_preserves_bytes_mode_and_cleans_temp(tmp_path, monkeypatch):
    from aurane.cli.commands.format import cmd_format
    import os

    path = tmp_path / "model.aur"
    original = SOURCE.replace("dense(2)", "dense(2)   ")
    path.write_text(original)
    path.chmod(0o640)

    def fail(*args):
        raise OSError("interrupted")

    monkeypatch.setattr(os, "replace", fail)
    assert cmd_format(Namespace(path=str(path), check=False, verbose=False)) == 1
    assert path.read_text() == original
    assert path.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.iterdir()) == [path]


def test_formatter_preserves_symlink_and_target_mode(tmp_path):
    path, link = tmp_path / "model.aur", tmp_path / "link.aur"
    path.write_text(SOURCE.replace("dense(2)", "dense(2)   "))
    path.chmod(0o640)
    link.symlink_to(path)
    result = cli("format", link)
    assert result.returncode == 0, result.stderr
    assert link.is_symlink()
    assert path.read_text() == SOURCE
    assert path.stat().st_mode & 0o777 == 0o640
