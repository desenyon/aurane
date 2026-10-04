"""
Parser for Aurane DSL.

This module implements a simple indentation-based parser that converts
.aur source code into an AST (Abstract Syntax Tree).
"""

import re
import ast as python_ast
import keyword
from typing import List, Tuple, Any, Optional, Dict, Union

from .diagnostics import SourceSpan

from .ast import (
    AuraneProgram,
    UseStatement,
    ExperimentNode,
    DatasetNode,
    ModelNode,
    TrainNode,
    TrainGANNode,
    ForwardBlock,
    ForwardGraphBlock,
    GraphOp,
    LayerOperation,
    Variable,
    Metric,
    Callback,
    LRScheduler,
    SymbolReference,
    ConfigCall,
)


class ParseError(Exception):
    """Exception raised when parsing fails."""

    def __init__(self, message, span=None):
        self.span = span
        super().__init__(message)


class Parser:
    """
    Simple line-based parser for Aurane DSL.

    Handles indentation-based blocks similar to Python.
    """

    def __init__(self, source: str):
        self.lines = source.split("\n")
        self.current_line = 0
        self.errors: List[Tuple[int, str]] = []
        self.error_span: Optional[SourceSpan] = None
        self._value_source = ""
        self._value_column = 1

    def parse(self) -> AuraneProgram:
        """Parse every non-comment statement, recording located syntax errors."""
        program = AuraneProgram()
        handlers: Dict[str, Any] = {
            "experiment": (self._parse_experiment, program.experiments),
            "dataset": (self._parse_dataset, program.datasets),
            "model": (self._parse_model, program.models),
            "train": (self._parse_train, program.trains),
            "train_gan": (self._parse_train_gan, program.train_gans),
        }
        while self.current_line < len(self.lines):
            try:
                self._skip_blank_lines()
                if self.current_line >= len(self.lines):
                    break
                line = self._line()
                if self._get_indent_level(self.current_line):
                    raise ParseError("Unexpected indentation")
                word = line.split()[0]
                if word == "use":
                    program.uses.append(self._parse_use(line))
                elif word in handlers:
                    parse_block, destination = handlers[word]
                    destination.append(parse_block())
                elif "=" in line:
                    program.variables.append(self._parse_global_variable())
                else:
                    raise ParseError(f"Unknown top-level statement: {line}")
            except (ParseError, SyntaxError, ValueError) as error:
                self.errors.append((self.current_line + 1, str(error)))
                self.error_span = getattr(error, "span", None) or SourceSpan(
                    self.current_line + 1,
                    self._get_indent_level(self.current_line) + 1,
                    self.current_line + 1,
                    (
                        len(self.lines[self.current_line]) + 1
                        if self.current_line < len(self.lines)
                        else 1
                    ),
                )
                # Stop rather than publishing a partial interpretation of a block.
                break
        for node in [
            *program.uses,
            *program.variables,
            *program.models,
            *program.datasets,
            *program.experiments,
            *program.trains,
            *program.train_gans,
        ]:
            node.column = node.column or 1
            node.end_line = node.line
            node.end_column = len(self.lines[node.line - 1]) + 1
        return program

    def _current_span(self) -> SourceSpan:
        return SourceSpan(
            self.current_line + 1,
            self._get_indent_level(self.current_line) + 1,
            self.current_line + 1,
            len(self._line()) + 1,
        )

    def _line(self) -> str:
        """Strip inline comments while preserving quoted text and indentation."""
        line = self.lines[self.current_line]
        quote = None
        escaped = False
        for index, char in enumerate(line):
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in ("'", '"'):
                quote = char
            elif char == "#":
                return line[:index].rstrip()
        if quote:
            raise ParseError("Unterminated string")
        return line.rstrip()

    def _skip_blank_lines(self) -> None:
        while self.current_line < len(self.lines) and not self._line().strip():
            self.current_line += 1

    def _body_indent(self, parent_indent: int = 0) -> Optional[int]:
        self._skip_blank_lines()
        if self.current_line == len(self.lines):
            return None
        indent = self._get_indent_level(self.current_line)
        return indent if indent > parent_indent else None

    def _identifier(self, value: str) -> str:
        if not value.isidentifier() or keyword.iskeyword(value) or value == "self":
            raise ParseError(f"Invalid identifier: {value!r}")
        return value

    def _split_expression(self, text: str, separator: str) -> List[str]:
        """Split at unquoted separators outside balanced parentheses/brackets."""
        parts = []
        start = 0
        stack = []
        quote = None
        escaped = False
        index = 0
        pairs = {")": "(", "]": "[", "}": "{"}
        while index < len(text):
            char = text[index]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in ("'", '"'):
                quote = char
            elif char in "([{":
                stack.append(char)
            elif char in ")]}":
                if not stack or stack.pop() != pairs[char]:
                    raise ParseError("Unbalanced delimiters")
            elif not stack and text.startswith(separator, index):
                parts.append(text[start:index].strip())
                index += len(separator)
                start = index
                continue
            index += 1
        if quote or stack:
            raise ParseError("Unterminated string or delimiter")
        parts.append(text[start:].strip())
        if any(not part for part in parts):
            raise ParseError(f"Missing expression around {separator!r}")
        return parts

    def _parse_use(self, line: str) -> UseStatement:
        match = re.fullmatch(r"use\s+([\w.]+)(?:\s+as\s+(\w+))?", line)
        if not match:
            raise ParseError("Invalid use statement")
        module, alias = match.groups()
        for part in module.split("."):
            self._identifier(part)
        if alias:
            self._identifier(alias)
        node = UseStatement(module=module, alias=alias, line=self.current_line + 1, column=1)
        self.current_line += 1
        return node

    def _parse_experiment(self) -> ExperimentNode:
        """Parse an 'experiment' block."""
        line = self._line()
        # experiment MnistBaseline:
        match = re.fullmatch(r"experiment\s+(\w+):", line)
        if not match:
            raise ParseError(f"Invalid experiment syntax at line {self.current_line + 1}")

        name = self._identifier(match.group(1))
        start_line = self.current_line + 1
        self.current_line += 1

        config, config_spans = self._parse_config_block()

        return ExperimentNode(name=name, config=config, line=start_line, config_spans=config_spans)

    def _parse_dataset(self) -> DatasetNode:
        """Parse a 'dataset' block."""
        line = self._line()
        # dataset mnist_train:
        match = re.fullmatch(r"dataset\s+(\w+):", line)
        if not match:
            raise ParseError(f"Invalid dataset syntax at line {self.current_line + 1}")

        name = self._identifier(match.group(1))
        start_line = self.current_line + 1
        self.current_line += 1

        config, config_spans = self._parse_config_block()

        # Extract 'from' clause if present
        source = config.pop("from", None)

        return DatasetNode(
            name=name, source=source, config=config, line=start_line, config_spans=config_spans
        )

    def _parse_model(self) -> ModelNode:
        match = re.fullmatch(r"model\s+(\w+):", self._line())
        if not match:
            raise ParseError("Invalid model definition")
        name = self._identifier(match.group(1))
        start_line = self.current_line + 1
        self.current_line += 1
        config: Dict[str, Any] = {}
        config_spans: Dict[str, SourceSpan] = {}
        forward_block = None
        indent = self._body_indent()
        while indent is not None and self.current_line < len(self.lines):
            self._skip_blank_lines()
            if self.current_line >= len(self.lines):
                break
            current_indent = self._get_indent_level(self.current_line)
            if current_indent < indent:
                break
            if current_indent != indent:
                raise ParseError("Unexpected indentation in model")
            line = self._line().strip()
            if line.startswith("def "):
                if forward_block is not None:
                    raise ParseError("Duplicate forward definition")
                forward_block = self._parse_forward_block()
            else:
                key, value = self._parse_assignment(line)
                if key in config:
                    raise ParseError(f"Duplicate configuration key: {key}")
                config[key] = value
                config_spans[key] = self._current_span()
                self.current_line += 1
        return ModelNode(
            name=name,
            config=config,
            forward_block=forward_block,
            line=start_line,
            column=1,
            config_spans=config_spans,
        )

    def _parse_forward_block(self) -> Union[ForwardBlock, ForwardGraphBlock]:
        match = re.fullmatch(r"\s*def\s+forward\((\w+)\):", self._line())
        if not match:
            raise ParseError("Invalid forward definition")
        parameter = self._identifier(match.group(1))
        start_line = self.current_line + 1
        parent_indent = self._get_indent_level(self.current_line)
        self.current_line += 1
        indent = self._body_indent(parent_indent)
        if indent is None:
            return ForwardBlock(parameter=parameter, line=start_line, column=parent_indent + 1)
        first = self._line().strip()
        block: Union[ForwardBlock, ForwardGraphBlock]
        if first.startswith("return ") or re.match(r"^\w+\s*=", first):
            block = self._parse_graph_forward_block(parameter)
        else:
            block = ForwardBlock(parameter=parameter, operations=self._parse_layer_chain(parameter))
        block.line = start_line
        block.column = parent_indent + 1
        block.end_line = start_line
        block.end_column = len(self.lines[start_line - 1]) + 1
        return block

    def _split_dot_outside_parens(self, text: str) -> List[str]:
        return self._split_expression(text, ".")

    def _parse_graph_forward_block(self, parameter: str) -> ForwardGraphBlock:
        indent = self._get_indent_level(self.current_line)
        nodes: List[GraphOp] = []
        output_var: Optional[str] = None
        while self.current_line < len(self.lines):
            self._skip_blank_lines()
            if self.current_line >= len(self.lines):
                break
            current_indent = self._get_indent_level(self.current_line)
            if current_indent < indent:
                break
            if current_indent != indent:
                raise ParseError("Unexpected indentation in graph")
            if output_var is not None:
                raise ParseError("Statement after return")
            stripped = self._line().strip()
            if stripped.startswith("return "):
                value = stripped[7:].strip()
                output_var = SymbolReference(
                    self._identifier(value),
                    self.current_line + 1,
                    self._line().index(value, indent + 7) + 1,
                )
            else:
                match = re.fullmatch(r"(\w+)\s*=\s*(.+)", stripped)
                if not match:
                    raise ParseError("Expected graph assignment or return")
                target = self._identifier(match.group(1))
                op, inputs = self._parse_graph_expr(match.group(2), indent + match.start(2) + 1)
                nodes.append(
                    GraphOp(
                        target=target,
                        inputs=inputs,
                        operation=op,
                        line=self.current_line + 1,
                        column=indent + 1,
                    )
                )
            self.current_line += 1
        if output_var is None and nodes:
            output_var = nodes[-1].target
        return ForwardGraphBlock(parameter=parameter, nodes=nodes, output_var=output_var)

    def _parse_graph_expr(self, expr: str, column: int) -> Tuple[LayerOperation, List[str]]:
        operations = self._parse_operation(expr, self.current_line + 1, column)
        if len(operations) != 1:
            raise ParseError("Use separate graph assignments for chained operation calls")
        op = operations[0]
        input_count = len(op.args) if op.operation.lower() in ("add", "concat") else 1
        if not op.args:
            raise ParseError(f"Graph op '{op.operation}' requires tensor inputs")
        inputs = op.args[:input_count]
        if any(not isinstance(value, SymbolReference) for value in inputs):
            raise ParseError("Graph inputs must be tensor variable names")
        op.args = op.args[input_count:]
        return op, inputs

    def _parse_layer_chain(self, parameter: str = "x") -> List[LayerOperation]:
        operations: List[LayerOperation] = []
        indent = self._get_indent_level(self.current_line)
        while self.current_line < len(self.lines):
            self._skip_blank_lines()
            if self.current_line >= len(self.lines):
                break
            current_indent = self._get_indent_level(self.current_line)
            if current_indent < indent:
                break
            text = self._line().strip()
            column = current_indent + 1
            if text.startswith("->"):
                if not operations:
                    raise ParseError("A chain must start with its forward parameter")
                column += 2
                text = text[2:]
            else:
                match = re.match(r"(\w+)\s*->\s*", text)
                if not match or match.group(1) != parameter:
                    raise ParseError(f"Expected '{parameter} ->' or a continuation '->'")
                if current_indent != indent:
                    raise ParseError("Unexpected indentation in forward chain")
                text = text[match.end() :]
                column += match.end()
            cursor = 0
            for part in self._split_expression(text, "->"):
                offset = text.index(part, cursor)
                operations.extend(
                    self._parse_operation(part, self.current_line + 1, column + offset)
                )
                cursor = offset + len(part)
            self.current_line += 1
        return operations

    def _parse_operation(self, text: str, line_num: int, column: int) -> List[LayerOperation]:
        operations: List[LayerOperation] = []
        cursor = 0
        for segment in self._split_dot_outside_parens(text):
            offset = text.index(segment, cursor)
            cursor = offset + len(segment)
            segment_column = column + offset
            end_column = segment_column + len(segment)
            if "(" not in segment:
                self._identifier(segment)
                if not operations:
                    raise ParseError("Expected an operation call")
                if operations[-1].activation is None:
                    operations[-1].activation = segment
                    operations[-1].end_column = end_column
                else:
                    operations.append(
                        LayerOperation(
                            operation=segment,
                            line=line_num,
                            column=segment_column,
                            end_line=line_num,
                            end_column=end_column,
                        )
                    )
                continue
            match = re.fullmatch(r"(\w+)\((.*)\)", segment)
            if not match:
                raise ParseError(f"Invalid operation: {segment}")
            op_name = self._identifier(match.group(1))
            args, kwargs = self._parse_arguments(match.group(2), segment_column + match.start(2))
            operations.append(
                LayerOperation(
                    operation=op_name,
                    args=args,
                    kwargs=kwargs,
                    line=line_num,
                    column=segment_column,
                    end_line=line_num,
                    end_column=end_column,
                )
            )
        return operations

    def _parse_train(self) -> TrainNode:
        """Parse a 'train' block."""
        line = self._line()
        # train MnistNet on mnist_train:
        match = re.fullmatch(r"train\s+(\w+)\s+on\s+(\w+):", line)
        if not match:
            raise ParseError(f"Invalid train syntax at line {self.current_line + 1}")

        model_name = self._identifier(match.group(1))
        dataset_name = match.group(2)
        start_line = self.current_line + 1
        self.current_line += 1

        config, config_spans = self._parse_config_block()

        # Post-process config to extract specific AST nodes
        metrics = []
        if "metrics" in config:
            metric_names = config.pop("metrics")
            if not isinstance(metric_names, list):
                raise ParseError("metrics must be a list")
            metrics = [Metric(name=str(m), params={}, line=start_line) for m in metric_names]

        callbacks = []
        if "callbacks" in config:
            callback_names = config.pop("callbacks")
            if not isinstance(callback_names, list) or not all(
                isinstance(c, str) for c in callback_names
            ):
                raise ParseError("callbacks must be a list of named callbacks")
            callbacks = [Callback(name=c, params={}, line=start_line) for c in callback_names]

        scheduler = None
        if "scheduler" in config:
            sched_val = config.pop("scheduler")
            if isinstance(sched_val, str) and "(" in sched_val:
                name = sched_val.split("(", 1)[0].strip()
                if isinstance(sched_val, ConfigCall):
                    args, kwargs = sched_val.args, sched_val.kwargs
                else:
                    arg_str = sched_val.split("(", 1)[1][:-1]
                    args, kwargs = self._parse_arguments(arg_str)
                # Keep both positional and keyword arguments for scheduler
                params = {"args": args, "kwargs": kwargs}
                scheduler = LRScheduler(name=name, params=params, line=start_line)
            elif isinstance(sched_val, str):
                scheduler = LRScheduler(name=sched_val, params={}, line=start_line)
            else:
                raise ParseError(f"scheduler must be a named scheduler at line {start_line}")

        return TrainNode(
            model_name=model_name,
            dataset_name=dataset_name,
            config=config,
            config_spans=config_spans,
            metrics=metrics,
            callbacks=callbacks,
            scheduler=scheduler,
            line=start_line,
        )

    def _parse_train_gan(self) -> TrainGANNode:
        """Parse a 'train_gan' block."""
        line = self._line()
        # train_gan Generator and Discriminator on mnist_images:
        match = re.fullmatch(r"train_gan\s+(\w+)\s+and\s+(\w+)\s+on\s+(\w+):", line)
        if not match:
            raise ParseError(f"Invalid train_gan syntax at line {self.current_line + 1}")

        gen_name = self._identifier(match.group(1))
        disc_name = match.group(2)
        dataset_name = match.group(3)
        start_line = self.current_line + 1
        self.current_line += 1

        config, config_spans = self._parse_config_block()

        return TrainGANNode(
            generator_name=gen_name,
            discriminator_name=disc_name,
            dataset_name=dataset_name,
            config=config,
            config_spans=config_spans,
            line=start_line,
        )

    def _parse_config_block(self) -> Tuple[Dict[str, Any], Dict[str, SourceSpan]]:
        config: Dict[str, Any] = {}
        config_spans: Dict[str, SourceSpan] = {}
        indent = self._body_indent()
        while indent is not None and self.current_line < len(self.lines):
            self._skip_blank_lines()
            if self.current_line >= len(self.lines):
                break
            current_indent = self._get_indent_level(self.current_line)
            if current_indent < indent:
                break
            if current_indent != indent:
                raise ParseError("Unexpected indentation in configuration")
            text = self._line().strip()
            if text.startswith("from "):
                key, value = "from", text[5:].strip()
                for part in value.split("."):
                    self._identifier(part)
            else:
                key, value = self._parse_assignment(text)
            if key in config:
                raise ParseError(f"Duplicate configuration key: {key}")
            config[key] = value
            config_spans[key] = self._current_span()
            self.current_line += 1
        return config, config_spans

    def _parse_assignment(self, line: str) -> Tuple[str, Any]:
        if "=" not in line:
            raise ParseError("Expected a key = value assignment")
        key, value = line.strip().split("=", 1)
        source = self._line()
        offset = source.index("=") + 1
        column = offset + len(source[offset:]) - len(source[offset:].lstrip()) + 1
        return self._identifier(key.strip()), self._parse_value(value.strip(), column)

    def _parse_global_variable(self) -> Variable:
        """Parse a top-level variable assignment."""
        line = self._line()
        key, value = self._parse_assignment(line)
        if not key:
            raise ParseError(f"Invalid variable assignment at line {self.current_line + 1}")
        self.current_line += 1
        return Variable(name=key, value=value, line=self.current_line)

    def _parse_value(self, value_str: str, column: int = 1) -> Any:
        self._value_source = value_str.strip()
        self._value_column = column
        try:
            node = python_ast.parse(value_str.strip(), mode="eval").body
        except SyntaxError as error:
            raise ParseError(f"Invalid value: {error.msg}") from error
        return self._value_from_node(node)

    def _value_from_node(self, node) -> Any:
        if isinstance(node, python_ast.Constant) and isinstance(
            node.value, (str, int, float, bool, type(None))
        ):
            return node.value
        if isinstance(node, python_ast.Name):
            name = node.id
            if name.lower() in ("true", "false", "none", "null"):
                return {"true": True, "false": False, "none": None, "null": None}[name.lower()]
            return SymbolReference(
                self._identifier(name),
                self.current_line + 1,
                self._value_column
                + len(self._value_source.encode("utf-8")[: node.col_offset].decode("utf-8")),
            )
        if isinstance(node, (python_ast.Tuple, python_ast.List)):
            values = [self._value_from_node(value) for value in node.elts]
            return tuple(values) if isinstance(node, python_ast.Tuple) else values
        if isinstance(node, python_ast.UnaryOp) and isinstance(
            node.op, (python_ast.USub, python_ast.UAdd)
        ):
            value = self._value_from_node(node.operand)
            if type(value) in (int, float):
                return -value if isinstance(node.op, python_ast.USub) else value
        if isinstance(node, python_ast.Call) and isinstance(node.func, python_ast.Name):
            args, kwargs = self._call_arguments(node)
            return ConfigCall(python_ast.unparse(node), args, kwargs)
        raise ParseError("Expected a literal, symbol, or named configuration call")

    def _call_arguments(self, node) -> Tuple[List[Any], Dict[str, Any]]:
        args = [self._value_from_node(arg) for arg in node.args]
        kwargs: Dict[str, Any] = {}
        for item in node.keywords:
            if item.arg is None:
                raise ParseError("Expanded keyword arguments are not supported")
            if item.arg in kwargs:
                raise ParseError(f"Duplicate argument: {item.arg}")
            kwargs[item.arg] = self._value_from_node(item.value)
        return args, kwargs

    def _parse_arguments(self, args_str: str, column: int = 1) -> Tuple[List[Any], Dict[str, Any]]:
        self._value_source = f"_args({args_str})"
        self._value_column = column - len("_args(")
        try:
            node = python_ast.parse(f"_args({args_str})", mode="eval").body
        except SyntaxError as error:
            raise ParseError(f"Invalid arguments: {error.msg}") from error
        return self._call_arguments(node)

    def _get_indent_level(self, line_num: int) -> int:
        """Get the indentation level of a line."""
        if line_num >= len(self.lines):
            return 0
        line = self.lines[line_num]
        return len(line) - len(line.lstrip())


def parse_aurane(source: str) -> AuraneProgram:
    """
    Parse Aurane source code and return an AST.

    Args:
        source: The Aurane source code as a string.

    Returns:
        An AuraneProgram AST node.

    Raises:
        ParseError: If the source code contains syntax errors.
    """
    parser = Parser(source)
    program = parser.parse()
    if parser.errors:
        formatted = "\n".join(f"line {line}: {message}" for line, message in parser.errors)
        raise ParseError(formatted, parser.error_span)
    return program
