"""The parser must never turn malformed source into a different valid model."""

import pytest

from aurane.parser import ParseError, parse_aurane
from aurane.ast import ForwardGraphBlock


@pytest.mark.parametrize(
    "source,line",
    [
        ("    stray\n", 1),
        ("model Net: trailing\n    input_shape = (4,)\n", 1),
        ("model Net:\n    garbage\n", 2),
        ("model Net:\n    input_shape = (4,)\n      ignored = 2\n", 3),
        ("model Net:\n    def forward(x):\n        x -> dense(2) trailing\n", 3),
        ("model Net:\n    def forward(x):\n        x -> dense(2\n", 3),
        ("model Net:\n    def forward(x):\n        x -> dense(2))\n", 3),
        ("model Net:\n    def forward(x):\n        x -> dense(2) ->\n", 3),
        ("model Net:\n    def forward(x):\n        x -> mystery\n", 3),
        ("model Net:\n    def forward(x):\n        y -> dense(2)\n", 3),
        ("model Net:\n    def forward(x):\n        x -> dense(2, bias=True, bias=False)\n", 3),
        ("model Net:\n    def forward(x):\n        x -> dense(size=2, 3)\n", 3),
        ("model Net:\n    def forward(x):\n        a = dense(x, 2)\n        typo\n", 4),
        ("model Net:\n    def forward(x):\n        return x\n        a = dense(x, 2)\n", 4),
        ("model Net:\n    def forward(x):\n        a = dense(123, 2)\n", 3),
        (
            "model Net:\n    def forward(x):\n        a = dense(x, 2)\n          b = dense(a, 3)\n",
            4,
        ),
        ("experiment E:\n    seed =\n", 2),
        ("experiment E:\n    seed = 1\n    seed = 2\n", 3),
        ("use torch as t as other\n", 1),
        ("experiment E:\n    name = 'unterminated\n", 2),
        ("experiment E:\n    value = 1 + 2\n", 2),
    ],
)
def test_malformed_source_has_located_error(source, line):
    with pytest.raises(ParseError, match=rf"line {line}\b"):
        parse_aurane(source)


def test_blank_and_comment_lines_do_not_hide_graph_or_following_blocks():
    source = """model Net:

# comment with no indentation
    input_shape = (4,)
    def forward(x):

        # another comment
        a = dense(x, 3)
        return a

model Other:
    input_shape = (3,)
    def forward(y):
        y -> dense(2)
"""
    program = parse_aurane(source)
    assert [model.name for model in program.models] == ["Net", "Other"]
    block = program.models[0].forward_block
    assert isinstance(block, ForwardGraphBlock)
    assert block.nodes[0].operation.args == [3]
    assert block.nodes[0].line == 8
    assert block.nodes[0].operation.line == 8
    assert block.output_var == "a"


def test_nested_literals_and_inline_comments_preserve_string_contents():
    program = parse_aurane(r"""experiment E: # header comment
    values = [(1, 2), [3, 4], "a,b", "a#b", "a->b"] # outside
    escaped = "say \"hello\""
    empty = None

model Net:
    input_shape = (4,)
    def forward(x):
        x -> custom("a->b", label="a,b#c", shape=(1, (2, 3))) # outside
""")
    config = program.experiments[0].config
    assert config["values"] == [(1, 2), [3, 4], "a,b", "a#b", "a->b"]
    assert config["escaped"] == 'say "hello"'
    assert config["empty"] is None
    op = program.models[0].forward_block.operations[0]
    assert op.args == ["a->b"]
    assert op.kwargs == {"label": "a,b#c", "shape": (1, (2, 3))}


def test_empty_block_does_not_consume_next_definition():
    program = parse_aurane("experiment Empty:\n\nmodel Net:\n    input_shape = (4,)\n")
    assert program.experiments[0].config == {}
    assert len(program.models) == 1


def test_chained_activations_are_not_silently_overwritten():
    program = parse_aurane("model Net:\n    def forward(x):\n        x -> dense(2).relu.sigmoid\n")
    ops = program.models[0].forward_block.operations
    assert ops[0].activation == "relu"
    assert ops[1].operation == "sigmoid"


def test_scheduler_preserves_nested_call_arguments():
    program = parse_aurane("train Net on data:\n    scheduler = custom(bounds=(1, 2))\n")
    assert program.trains[0].scheduler.params["kwargs"] == {"bounds": (1, 2)}


def test_graph_rejects_quoted_tensor_name():
    with pytest.raises(ParseError, match="tensor"):
        parse_aurane('model Net:\n    def forward(x):\n        a = dense("x", 2)\n')
