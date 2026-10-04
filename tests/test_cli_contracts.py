"""Regression checks for command IO and preservation of user files."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from aurane.cli.commands.format import format_aurane_code
from aurane.parser import parse_aurane

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "model Net:\n    input_shape = (4,)\n    def forward(x):\n        x -> dense(2)\n"


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "aurane.cli", *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("extra", [[], ["--format"], ["--show-ast"], ["--quiet"]])
def test_compile_stdout_is_python(tmp_path, extra):
    path = tmp_path / "model.aur"
    path.write_text(SOURCE)
    result = cli("compile", path, *extra)
    assert result.returncode == 0, result.stderr
    ast.parse(result.stdout)
    assert "class Net" in result.stdout


@pytest.mark.parametrize("content", [None, "invalid source", SOURCE])
def test_check_json_on_every_outcome(tmp_path, content):
    path = tmp_path / "model.aur"
    if content is not None:
        path.write_text(content)
    result = cli("check", path, "--json")
    payload = json.loads(result.stdout)
    assert payload["ok"] == (content == SOURCE)
    assert result.returncode == (0 if content == SOURCE else 1)


@pytest.mark.parametrize("content", [None, "invalid source", SOURCE])
def test_ir_json_on_every_outcome(tmp_path, content):
    path = tmp_path / "model.aur"
    if content is not None:
        path.write_text(content)
    result = cli("ir", path, "--format", "json")
    payload = json.loads(result.stdout)
    assert result.returncode == (0 if content == SOURCE else 1)
    if content == SOURCE:
        assert payload["models"][0]["name"] == "Net"
    else:
        assert payload["ok"] is False


@pytest.mark.parametrize("example", sorted((ROOT / "examples").glob("*.aur")))
def test_formatter_preserves_example_structure(example):
    source = example.read_text()
    formatted = format_aurane_code(source)
    assert parse_aurane(formatted) == parse_aurane(source)
    assert format_aurane_code(formatted) == formatted


def test_format_check_never_writes(tmp_path):
    path = tmp_path / "model.aur"
    original = SOURCE.replace("dense(2)", "dense(2)   ")
    path.write_text(original)
    result = cli("format", path, "--check")
    assert result.returncode == 1
    assert path.read_text() == original
    assert cli("format", path).returncode == 0
    assert cli("format", path, "--check").returncode == 0


def test_formatter_preserves_graph_and_literal_whitespace():
    source = """label = "  value  "
model Net:
    input_shape = (4,)
    def forward(x):
        a = dense(x, 4)
        b = add(a, x)
        return b
"""
    assert parse_aurane(format_aurane_code(source)) == parse_aurane(source)


def test_black_formatting_is_idempotent():
    from aurane.cli.commands.compile import _maybe_black_format

    once = _maybe_black_format("x=1\n", None)
    assert _maybe_black_format(once, None) == once


def test_compile_cli_preserves_source_if_output_is_same_path(tmp_path):
    path = tmp_path / "model.aur"
    path.write_text(SOURCE)
    result = cli("compile", path, path, "--quiet")
    assert result.returncode == 1
    assert "source" in result.stderr.lower()
    assert path.read_text() == SOURCE


def test_visualize_file_output_requires_unambiguous_model(tmp_path):
    source, output = tmp_path / "models.aur", tmp_path / "model.dot"
    source.write_text(SOURCE + SOURCE.replace("Net", "Other").replace("dense(2)", "dense(3)"))
    output.write_text("existing output")
    result = cli("visualize", source, "--format", "dot", "--output", output)
    assert result.returncode == 1
    assert output.read_text() == "existing output"
    result = cli("visualize", source, "--format", "dot", "--output", output, "--model", "Other")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Output: (3,)" in output.read_text()


@pytest.mark.parametrize("command", ["profile", "visualize", "ir", "inspect"])
def test_analysis_commands_resolve_global_and_model_constants(tmp_path, command):
    path = tmp_path / "constants.aur"
    path.write_text("""width = 8
model Net:
    classes = 3
    input_shape = (5, width)
    def forward(x):
        x -> dense(classes)
""")
    options = {
        "profile": [],
        "visualize": ["--format", "mermaid"],
        "ir": ["--format", "json"],
        "inspect": ["--verbose"],
    }
    result = cli(command, path, *options[command])
    assert result.returncode == 0, result.stdout + result.stderr
    if command == "ir":
        assert json.loads(result.stdout)["models"][0]["ir"]["nodes"][0]["args"] == [3]
        assert json.loads(result.stdout)["models"][0]["ir"]["nodes"][0]["output"]["shape"] == [5, 3]
    else:
        assert "(5, 3)" in result.stdout


def test_clean_preserves_user_files_dependencies_and_symlinks(tmp_path):
    protected = [
        "extension.so",
        "extension.pyd",
        "user_temp.py",
        "model.aur.py",
        "src/model.aur",
        ".venv/lib/cache.pyc",
        "custom-env/lib/cache.pyc",
        ".git/cache.pyc",
    ]
    for name in protected:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep")
    (tmp_path / "custom-env/pyvenv.cfg").write_text("home = /python")
    cache = tmp_path / "src/.aurane_cache/torch/code.py"
    cache.parent.mkdir(parents=True)
    cache.write_text("remove")
    (tmp_path / "src/__pycache__").mkdir()
    (tmp_path / "src/__pycache__/module.pyc").write_bytes(b"remove")
    (tmp_path / "external").mkdir()
    (tmp_path / "external/keep.py").write_text("keep")
    (tmp_path / ".aurane_cache").symlink_to(tmp_path / "external", target_is_directory=True)
    dry = cli("clean", tmp_path, "--dry-run")
    assert dry.returncode == 0, dry.stdout
    assert cache.exists()
    result = cli("clean", tmp_path)
    assert result.returncode == 0, result.stdout
    assert not cache.exists()
    assert not (tmp_path / "src/__pycache__").exists()
    for name in protected:
        assert (tmp_path / name).read_text() == "keep", name
    assert (tmp_path / "external/keep.py").read_text() == "keep"
    assert (tmp_path / ".aurane_cache").is_symlink()


def test_clean_rejects_symlink_root(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    cache = target / ".aurane_cache"
    cache.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    assert cli("clean", link).returncode == 1
    assert cache.exists()


def test_compile_failure_preserves_output_and_uses_stderr(tmp_path):
    path = tmp_path / "model.aur"
    path.write_text("invalid source")
    output = tmp_path / "model.py"
    output.write_text("keep")
    result = cli("compile", path, output, "--quiet")
    assert result.returncode == 1
    assert result.stderr
    assert result.stdout == ""
    assert output.read_text() == "keep"
