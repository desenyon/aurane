"""Preparation is local to a compile, reusable within it, and plugin compatible."""

from copy import deepcopy
from pathlib import Path

import pytest

from aurane import compiler
from aurane.backends import register_backend_generator
from aurane.codegen_torch import generate_torch_code
from aurane.parser import parse_aurane

SOURCE = "width = 4\nmodel Net:\n    input_shape = (width,)\n    def forward(x):\n        x -> dense(2).relu -> relu()\n"


def test_compile_resolves_once_and_lowers_each_model_once(monkeypatch):
    from aurane import preparation

    calls = {"resolve": 0, "lower": 0}
    resolve, lower = preparation.resolve_program, preparation.lower_model

    def resolve_count(program):
        calls["resolve"] += 1
        return resolve(program)

    def lower_count(model):
        calls["lower"] += 1
        return lower(model)

    monkeypatch.setattr(preparation, "resolve_program", resolve_count)
    monkeypatch.setattr(preparation, "lower_model", lower_count)
    compiler.compile_source(SOURCE, analyze=True, validate=True, disable_cache=True)
    assert calls == {"resolve": 1, "lower": 1}


def test_optimization_invalidates_prepared_graphs():
    from aurane.preparation import prepare_program

    parsed = parse_aurane(SOURCE)
    before = deepcopy(parsed)
    prepared = prepare_program(parsed)
    old_graph = prepared.graph_for(prepared.program.models[0])
    optimized = prepared.optimized(1)
    new_graph = optimized.graph_for(optimized.program.models[0])
    assert len(old_graph.nodes) == 2
    assert len(new_graph.nodes) == 1
    assert parsed == before
    assert prepared.graph_for(prepared.program.models[0]) is old_graph
    with pytest.raises(ValueError, match="belong"):
        optimized.graph_for(prepared.program.models[0])


@pytest.mark.parametrize("example", sorted((Path(__file__).parents[1] / "examples").glob("*.aur")))
@pytest.mark.parametrize("level", [0, 1, 2])
def test_prepared_pipeline_preserves_standalone_generation(example, level):
    from aurane.optimizer import optimize_ast
    from aurane.symbols import resolve_program

    source = example.read_text()
    program = optimize_ast(resolve_program(parse_aurane(source)), level=level).program
    expected = generate_torch_code(program)
    actual = compiler.compile_source(
        source, analyze=True, validate=True, optimize=True, opt_level=level, disable_cache=True
    )
    assert actual == expected
    compile(actual, str(example), "exec")


def test_legacy_backend_receives_resolved_ast_and_replacement_clears_prepared_callback(tmp_path):
    from aurane.ast import AuraneProgram

    received = []

    def legacy(program):
        assert isinstance(program, AuraneProgram)
        assert program.models[0].config["input_shape"] == (4,)
        received.append(program)
        return "legacy = True\n"

    register_backend_generator(
        "pipeline-test", legacy, prepared_generator=lambda program: "prepared = True\n"
    )
    assert compiler.compile_source(SOURCE, backend="pipeline-test") == "prepared = True\n"
    register_backend_generator("pipeline-test", legacy)
    assert compiler.compile_source(SOURCE, backend="pipeline-test") == "legacy = True\n"
    assert len(received) == 1


@pytest.mark.parametrize(
    "source,options,stage,line",
    [
        ("model Net:\n    invalid\n", {}, "parse", 2),
        (SOURCE.replace("dense(2)", "dense(missing)"), {}, "resolve", 5),
        (SOURCE.replace("dense(2)", "reshape(3)"), {"validate": True}, "type", 5),
        (SOURCE.replace("dense(2)", "reshape(3)"), {}, "codegen", 5),
        ('experiment E:\n    device = "bad"\n', {"analyze": True}, "semantic", 2),
    ],
)
def test_compilation_errors_preserve_stage_span_and_file(tmp_path, source, options, stage, line):
    path = tmp_path / "invalid.aur"
    path.write_text(source)
    with pytest.raises(compiler.CompilationError) as caught:
        compiler.compile_file(str(path), str(tmp_path / "out.py"), disable_cache=True, **options)
    error = caught.value
    assert error.stage == stage
    assert error.span.line == line
    payload = error.to_dict()
    assert payload["source"] == str(path)
    assert payload["diagnostics"][0]["stage"] == stage
    assert payload["diagnostics"][0]["span"]["line"] == line
    assert not (tmp_path / "out.py").exists()


def test_backend_and_optimization_failures_have_stages():
    with pytest.raises(compiler.CompilationError) as caught:
        compiler.compile_source(SOURCE, backend="does-not-exist", disable_cache=True)
    assert caught.value.stage == "backend"
    with pytest.raises(compiler.CompilationError) as caught:
        compiler.compile_source(SOURCE, optimize=True, opt_level=99, disable_cache=True)
    assert caught.value.stage == "optimize"
