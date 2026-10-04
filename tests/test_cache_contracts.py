"""A cache is optional; corrupt or interrupted IO must not change compilation."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from aurane import compiler
from aurane.backends import register_backend_generator

SOURCE = "model Net:\n    input_shape = (4,)\n    def forward(x):\n        x -> dense(3)\n"


@pytest.mark.parametrize("payload", [b"", b"not python!", b"\xff", b"print('modified')\n"])
def test_corrupt_cache_is_rebuilt(tmp_path, monkeypatch, payload):
    monkeypatch.chdir(tmp_path)
    expected = compiler.compile_source(SOURCE)
    files = list((tmp_path / ".aurane_cache").rglob("*.*"))
    assert len(files) == 1
    files[0].write_bytes(payload)
    assert compiler.compile_source(SOURCE) == expected


def test_unreadable_cache_is_nonfatal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = compiler.compile_source(SOURCE)
    original = Path.read_text

    def read(path, *args, **kwargs):
        if ".aurane_cache" in path.parts:
            raise PermissionError("cache unreadable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert compiler.compile_source(SOURCE) == expected


def test_replacing_registered_backend_does_not_return_old_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    register_backend_generator("changing", lambda program: "value = 1\n")
    assert compiler.compile_source(SOURCE, backend="changing") == "value = 1\n"
    register_backend_generator("changing", lambda program: "value = 2\n")
    assert compiler.compile_source(SOURCE, backend="changing") == "value = 2\n"


def test_backend_name_cannot_escape_cache_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    register_backend_generator("../outside", lambda program: "value = 1\n")
    compiler.compile_source(SOURCE, backend="../outside")
    assert not (tmp_path / "outside").exists()


def test_failed_publication_preserves_output(tmp_path, monkeypatch):
    import os

    source = tmp_path / "model.aur"
    output = tmp_path / "model.py"
    source.write_text(SOURCE)
    output.write_text("previous complete artifact\n")

    def fail_replace(*args, **kwargs):
        raise OSError("publication interrupted")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(compiler.CompilationError, match="publication interrupted"):
        compiler.compile_file(str(source), str(output), disable_cache=True)
    assert output.read_text() == "previous complete artifact\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["model.aur", "model.py"]


def test_compilation_cannot_overwrite_its_source(tmp_path):
    source = tmp_path / "model.aur"
    source.write_text(SOURCE)
    with pytest.raises(compiler.CompilationError, match="source"):
        compiler.compile_file(str(source), str(source), disable_cache=True)
    assert source.read_text() == SOURCE


def test_concurrent_compilations_leave_one_complete_cache_entry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = compiler.compile_source(SOURCE, disable_cache=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: compiler.compile_source(SOURCE), range(24)))
    assert results == [expected] * 24
    assert compiler.compile_source(SOURCE) == expected
    assert len(list((tmp_path / ".aurane_cache").rglob("*.*"))) == 1


def test_cache_write_failure_and_disable_cache_are_nonfatal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = compiler.compile_source(SOURCE, disable_cache=True)
    assert not (tmp_path / ".aurane_cache").exists()
    (tmp_path / ".aurane_cache").write_text("not a directory")
    assert compiler.compile_source(SOURCE) == expected


def test_cache_hits_and_invalidation_contract(tmp_path, monkeypatch):
    import json

    monkeypatch.chdir(tmp_path)
    calls = []

    def generate(program):
        calls.append(program)
        return f"value = {len(calls)}\n"

    register_backend_generator("versioned", generate, cache_version="1")
    assert compiler.compile_source(SOURCE, backend="versioned") == "value = 1\n"
    assert compiler.compile_source(SOURCE, backend=" VERSIONED ") == "value = 1\n"
    cached = next((tmp_path / ".aurane_cache").glob("*.json"))
    record = json.loads(cached.read_text())
    record["code"] = "value = 999\n"
    cached.write_text(json.dumps(record))
    assert compiler.compile_source(SOURCE, backend="versioned") == "value = 2\n"
    assert compiler.compile_source(SOURCE, backend="versioned", validate=True) == "value = 3\n"
    monkeypatch.setattr(compiler, "CACHE_SCHEMA", compiler.CACHE_SCHEMA + 1)
    assert compiler.compile_source(SOURCE, backend="versioned") == "value = 4\n"
    register_backend_generator("versioned", generate, cache_version="2")
    assert compiler.compile_source(SOURCE, backend="versioned") == "value = 5\n"
