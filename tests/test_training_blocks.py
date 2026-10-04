"""Every declared job gets a distinct callable and runs in source order."""

import ast

import pytest

pytest.importorskip("torch")

from aurane.compiler import compile_source

MODELS = """model Net:
    input_shape = (2,)
    def forward(x):
        x -> dense(1).sigmoid
model net:
    input_shape = (2,)
    def forward(x):
        x -> dense(1).sigmoid
model Generator:
    input_shape = (2,)
    def forward(x):
        x -> dense(2)
"""


def test_all_jobs_execute_in_source_order_with_unique_functions():
    source = MODELS + """train Net on first:
    epochs = 1
train_gan Generator and Net on second:
    epochs = 1
train net on third:
    epochs = 2
train Net on fourth:
    epochs = 3
train_gan Generator and Net on fifth:
    epochs = 2
"""
    tree = ast.parse(compile_source(source, disable_cache=True))
    main = tree.body.pop()
    namespace = {"__name__": "__main__"}
    exec(compile(tree, "jobs.py", "exec"), namespace)
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert len(functions) == len(set(functions)) == 5
    assert functions == [
        "train_net",
        "train_gan_generator_net",
        "train_net_2",
        "train_net_3",
        "train_gan_generator_net_2",
    ]
    calls = []
    for name in functions:
        namespace[name] = lambda name=name: calls.append(name)
    exec(compile(ast.Module(body=[main], type_ignores=[]), "main.py", "exec"), namespace)
    assert calls == functions


def test_repeated_jobs_keep_their_own_epoch_settings():
    import torch

    source = MODELS + """train Net on injected:
    epochs = 1
    loss = mse
train Net on injected:
    epochs = 2
    loss = mse
"""
    namespace = {"__name__": "jobs_test"}
    exec(compile_source(source, disable_cache=True), namespace)
    loader = [(torch.zeros(2, 2), torch.ones(2, 1))]
    first = namespace["train_net"](train_loader=loader)
    second = namespace["train_net_2"](train_loader=loader)
    assert len(first.training_history) == 1
    assert len(second.training_history) == 2
    assert first is not second


def test_training_function_does_not_shadow_model_class():
    source = (
        """model train_net:
    input_shape = (2,)
    def forward(x):
        x -> dense(1)
"""
        + MODELS
        + """train Net on injected:
    epochs = 1
"""
    )
    namespace = {"__name__": "jobs_test"}
    exec(compile_source(source, disable_cache=True), namespace)
    assert isinstance(namespace["train_net"], type)
    assert callable(namespace["train_net_2"])


@pytest.mark.parametrize("name", ["random", "model", "train_loader", "Path"])
def test_model_names_do_not_collide_with_training_locals(name):
    import torch

    source = MODELS.split("model net:")[0].replace("Net", name)
    source += f"train {name} on injected:\n    epochs = 1\n    loss = mse\n"
    namespace = {"__name__": "jobs_test"}
    exec(compile_source(source, disable_cache=True), namespace)
    trained = namespace["train_" + name.lower()](
        train_loader=[(torch.zeros(2, 2), torch.ones(2, 1))]
    )
    assert len(trained.training_history) == 1
