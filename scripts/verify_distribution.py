"""Check a clean wheel installation and the contents of its source archive."""

import argparse
import ast
import json
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import aurane


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("distribution_directory", type=Path)
    args = parser.parse_args()
    assert (
        "site-packages" in Path(aurane.__file__).parts
    ), "Run with the installed wheel, not an editable checkout"
    assert aurane.__version__ == version("aurane"), "Runtime and package versions disagree"
    archives = list(args.distribution_directory.glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0]) as archive:
        names = archive.getnames()
    for required in (
        "CHANGELOG.md",
        "examples/simple.aur",
        "examples/transformer.aur",
        "docs/getting-started.md",
        "aurane/data.py",
        "aurane/symbols.py",
        "aurane/operations.py",
        "aurane/configuration.py",
        "aurane/metrics.py",
        "aurane/dtypes.py",
        "aurane/diagnostics.py",
        "aurane/runtime_templates.py",
    ):
        assert any(name.endswith("/" + required) for name in names), f"sdist missing {required}"
    with tempfile.TemporaryDirectory(prefix="aurane-wheel-") as directory:
        path = Path(directory) / "model.aur"
        path.write_text(
            "width = 3\nmodel Net:\n    input_shape = (4,)\n    input_dtype = 'float64'\n    def forward(x):\n        x -> dense(width)\n"
        )

        def cli(*arguments):
            result = subprocess.run(
                [sys.executable, "-m", "aurane.cli", *map(str, arguments)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            return result.stdout

        assert "Aurane" in cli("--version")
        assert json.loads(cli("check", path, "--json"))["ok"]
        ast.parse(cli("compile", path))
        graph = json.loads(cli("ir", path, "--format", "json"))["models"][0]["ir"]
        assert graph["outputs"][0]["shape"] == [3]
        assert graph["inputs"][0]["type_hint"] == "float64"
        assert graph["outputs"][0]["type_hint"] == "float64"
        assert graph["nodes"][0]["end_column"] > graph["nodes"][0]["column"]
        assert json.loads(cli("benchmark", path, "-i", 1, "--json"))["ok"]
        path.write_text("experiment E:\n    seed = -1\n")
        invalid = subprocess.run(
            [sys.executable, "-m", "aurane.cli", "check", str(path), "--json"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert invalid.returncode == 1
        issue = json.loads(invalid.stdout)["types"]["errors"][0]
        assert issue["span"]["line"] == 2 and issue["span"]["column"] == 5
    print("Wheel import, version, check, compile, IR, benchmark and sdist contents verified.")


if __name__ == "__main__":
    main()
