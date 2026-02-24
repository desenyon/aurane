"""
Parser for Aurane DSL.

This module implements a simple indentation-based parser that converts
.aur source code into an AST (Abstract Syntax Tree).
"""

import re
from typing import List, Tuple, Any, Optional, Dict

from .ast import (
    AuraneProgram,
    UseStatement,
    ExperimentNode,
    DatasetNode,
    ModelNode,
    TrainNode,
    TrainGANNode,
    ForwardBlock,
    LayerOperation,
    Metric,
    Callback,
    LRScheduler,
)


class ParseError(Exception):
    """Exception raised when parsing fails."""

    pass


class Parser:
    """
    Simple line-based parser for Aurane DSL.

    Handles indentation-based blocks similar to Python.
    """

    def __init__(self, source: str):
        self.lines = source.split("\n")
        self.current_line = 0
        self.errors = []

    def parse(self) -> AuraneProgram:
        """Parse the entire Aurane program and return an AST."""
        program = AuraneProgram()

        while self.current_line < len(self.lines):
            line = self.lines[self.current_line].rstrip()

            # Skip empty lines and comments
            if not line or line.lstrip().startswith("#"):
                self.current_line += 1
                continue

            # Check for no indentation (top-level construct)
            if line[0] not in (" ", "\t"):
                try:
                    if line.startswith("use "):
                        program.uses.append(self._parse_use(line))
                    elif line.startswith("experiment "):
                        program.experiments.append(self._parse_experiment())
                    elif line.startswith("dataset "):
                        program.datasets.append(self._parse_dataset())
                    elif line.startswith("model "):
                        program.models.append(self._parse_model())
                    elif line.startswith("train "):
                        program.trains.append(self._parse_train())
                    elif line.startswith("train_gan "):
                        program.train_gans.append(self._parse_train_gan())
                    elif "=" in line:
                        program.variables.append(self._parse_global_variable())
                    else:
                        self.errors.append((self.current_line + 1, f"Unknown top-level statement: {line}"))
                        self.current_line += 1
                except Exception as e:
                    self.errors.append((self.current_line + 1, str(e)))
                    # Try to skip to next top-level block
                    self.current_line += 1
                    while self.current_line < len(self.lines):
                        next_line = self.lines[self.current_line].rstrip()
                        if next_line and next_line[0] not in (" ", "\t"):
                            break
                        self.current_line += 1
            else:
                self.current_line += 1

        return program

    def _parse_use(self, line: str) -> UseStatement:
        """Parse a 'use' statement."""
        # use torch
        # use aurane.ml as aml
        self.current_line += 1

        parts = line[4:].strip().split(" as ")
        module = parts[0].strip()
        alias = parts[1].strip() if len(parts) > 1 else None

        return UseStatement(module=module, alias=alias, line=self.current_line)

    def _parse_experiment(self) -> ExperimentNode:
        """Parse an 'experiment' block."""
        line = self.lines[self.current_line]
        # experiment MnistBaseline:
        match = re.match(r"experiment\s+(\w+):", line)
        if not match:
            raise ParseError(f"Invalid experiment syntax at line {self.current_line + 1}")

        name = match.group(1)
        start_line = self.current_line + 1
        self.current_line += 1

        config = self._parse_config_block()

        return ExperimentNode(name=name, config=config, line=start_line)

    def _parse_dataset(self) -> DatasetNode:
        """Parse a 'dataset' block."""
        line = self.lines[self.current_line]
        # dataset mnist_train:
        match = re.match(r"dataset\s+(\w+):", line)
        if not match:
            raise ParseError(f"Invalid dataset syntax at line {self.current_line + 1}")

        name = match.group(1)
        start_line = self.current_line + 1
        self.current_line += 1

        config = self._parse_config_block()

        # Extract 'from' clause if present
        source = config.pop("from", None)

        return DatasetNode(name=name, source=source, config=config, line=start_line)

    def _parse_model(self) -> ModelNode:
        """Parse a 'model' block."""
        line = self.lines[self.current_line]
        # model MnistNet:
        match = re.match(r"model\s+(\w+):", line)
        if not match:
            raise ParseError(f"Invalid model syntax at line {self.current_line + 1}")

        name = match.group(1)
        start_line = self.current_line + 1
        self.current_line += 1

        config = {}
        forward_block = None

        # Parse model body
        indent = self._get_indent_level(self.current_line)
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line].rstrip()

            if not line or line.lstrip().startswith("#"):
                self.current_line += 1
                continue

            current_indent = self._get_indent_level(self.current_line)
            if current_indent < indent:
                break

            if current_indent == indent:
                if line.strip().startswith("def forward("):
                    forward_line = self.current_line + 1
                    forward_block = self._parse_forward_block()
                    forward_block.line = forward_line
                    # After parsing forward block, we're done
                    break
                else:
                    # Regular config line
                    key, value = self._parse_assignment(line)
                    if key and value is not None:
                        config[key] = value
                        self.current_line += 1
                    else:
                        # Not a valid assignment, exit the model block
                        break
            else:
                self.current_line += 1

        return ModelNode(name=name, config=config, forward_block=forward_block, line=start_line)

    def _parse_forward_block(self) -> ForwardBlock:
        """Parse a 'def forward(x):' block."""
        line = self.lines[self.current_line]
        # def forward(x):
        match = re.match(r"\s*def\s+forward\((\w+)\):", line)
        if not match:
            raise ParseError(f"Invalid forward definition at line {self.current_line + 1}")

        param = match.group(1)
        self.current_line += 1

        operations = self._parse_layer_chain()

        return ForwardBlock(parameter=param, operations=operations)

    def _parse_layer_chain(self) -> List[LayerOperation]:
        """Parse the layer chain (x -> conv2d(...).relu -> ...)."""
        operations = []
        indent = self._get_indent_level(self.current_line)

        # Get the forward parameter name from the previous model parsing state if possible
        # but for now we just look for common "x ->" or whatever is at the start
        
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line].rstrip()

            if not line or line.lstrip().startswith("#"):
                self.current_line += 1
                continue

            current_indent = self._get_indent_level(self.current_line)
            if current_indent < indent:
                break

            stripped = line.strip()
            
            # Remove any leading 'param ->' if present on this line
            # We look for a word followed by ->
            stripped = re.sub(r"^\w+\s*->\s*", "", stripped)
            
            # Split by -> and parse each operation
            # Note: we need to be careful with -> inside strings if we ever support them, 
            # but for now simple split is better than before
            parts = [p.strip() for p in stripped.split("->") if p.strip()]
            for part in parts:
                operations.extend(self._parse_operation(part, self.current_line + 1))
            
            self.current_line += 1

        return operations

    def _parse_operation(self, text: str, line_num: int) -> List[LayerOperation]:
        """
        Parse a single operation or chained operations with activations.

        Examples:
            conv2d(32, kernel=3).relu
            maxpool(2)
            flatten()
            dropout(0.5)
        """
        operations = []

        # Split by dots that are not inside parentheses
        segments = []
        current = ""
        paren_depth = 0

        for char in text:
            if char == "(":
                paren_depth += 1
                current += char
            elif char == ")":
                paren_depth -= 1
                current += char
            elif char == "." and paren_depth == 0:
                if current:
                    segments.append(current)
                current = ""
            else:
                current += char

        if current:
            segments.append(current)

        for i, segment in enumerate(segments):
            segment = segment.strip()
            if not segment:
                continue

            # Check if this is just an activation (no parentheses)
            if "(" not in segment:
                # This is an activation on the previous operation
                if operations:
                    operations[-1].activation = segment
                continue

            # Parse function call: name(args...)
            match = re.match(r"(\w+)\((.*)\)", segment)
            if match:
                op_name = match.group(1)
                args_str = match.group(2)

                args, kwargs = self._parse_arguments(args_str)

                operations.append(LayerOperation(operation=op_name, args=args, kwargs=kwargs, line=line_num))

        return operations

    def _parse_train(self) -> TrainNode:
        """Parse a 'train' block."""
        line = self.lines[self.current_line]
        # train MnistNet on mnist_train:
        match = re.match(r"train\s+(\w+)\s+on\s+(\w+):", line)
        if not match:
            raise ParseError(f"Invalid train syntax at line {self.current_line + 1}")

        model_name = match.group(1)
        dataset_name = match.group(2)
        start_line = self.current_line + 1
        self.current_line += 1

        config = self._parse_config_block()

        # Post-process config to extract specific AST nodes
        metrics = []
        if "metrics" in config:
            metric_names = config.pop("metrics")
            if isinstance(metric_names, list):
                metrics = [Metric(name=str(m), params={}, line=start_line) for m in metric_names]

        callbacks = []
        if "callbacks" in config:
            callback_names = config.pop("callbacks")
            if isinstance(callback_names, list):
                callbacks = [Callback(name=str(c), params={}, line=start_line) for c in callback_names]

        scheduler = None
        if "scheduler" in config:
            sched_val = config.pop("scheduler")
            if isinstance(sched_val, str) and "(" in sched_val:
                name = sched_val.split("(", 1)[0].strip()
                arg_str = sched_val.split("(", 1)[1].rstrip(")")
                args, kwargs = self._parse_arguments(arg_str)
                # Keep both positional and keyword arguments for scheduler
                params = {"args": args, "kwargs": kwargs}
                scheduler = LRScheduler(name=name, params=params, line=start_line)

        return TrainNode(
            model_name=model_name,
            dataset_name=dataset_name,
            config=config,
            metrics=metrics,
            callbacks=callbacks,
            scheduler=scheduler,
            line=start_line,
        )

    def _parse_train_gan(self) -> TrainGANNode:
        """Parse a 'train_gan' block."""
        line = self.lines[self.current_line]
        # train_gan Generator and Discriminator on mnist_images:
        match = re.match(r"train_gan\s+(\w+)\s+and\s+(\w+)\s+on\s+(\w+):", line)
        if not match:
            raise ParseError(f"Invalid train_gan syntax at line {self.current_line + 1}")

        gen_name = match.group(1)
        disc_name = match.group(2)
        dataset_name = match.group(3)
        start_line = self.current_line + 1
        self.current_line += 1

        config = self._parse_config_block()

        return TrainGANNode(
            generator_name=gen_name,
            discriminator_name=disc_name,
            dataset_name=dataset_name,
            config=config,
            line=start_line,
        )

    def _parse_config_block(self) -> Dict[str, Any]:
        """Parse a configuration block (key = value pairs)."""
        config = {}
        indent = self._get_indent_level(self.current_line)

        while self.current_line < len(self.lines):
            line = self.lines[self.current_line].rstrip()

            if not line or line.lstrip().startswith("#"):
                self.current_line += 1
                continue

            current_indent = self._get_indent_level(self.current_line)
            if current_indent < indent:
                break

            if current_indent == indent:
                stripped = line.strip()

                # Handle special 'from' clause (e.g., "from torchvision.datasets.MNIST")
                if stripped.startswith("from "):
                    config["from"] = stripped[5:].strip()
                    self.current_line += 1
                else:
                    key, value = self._parse_assignment(line)
                    if key and value is not None:
                        config[key] = value
                        self.current_line += 1
                    else:
                        # Not an assignment, likely a different construct
                        break
            else:
                self.current_line += 1

        return config

    def _parse_assignment(self, line: str) -> Tuple[Optional[str], Optional[Any]]:
        """Parse a key = value assignment."""
        line = line.strip()

        if "=" not in line:
            return None, None

        key, value_str = line.split("=", 1)
        key = key.strip()
        value_str = value_str.strip()

        value = self._parse_value(value_str)

        return key, value

    def _parse_value(self, value_str: str) -> Any:
        """Parse a value from a string."""
        value_str = value_str.strip()

        # Boolean
        if value_str == "True":
            return True
        if value_str == "False":
            return False

        # None
        if value_str == "None":
            return None

        # Integer
        try:
            return int(value_str)
        except ValueError:
            pass

        # Float
        try:
            return float(value_str)
        except ValueError:
            pass

        # Tuple (1, 28, 28)
        if value_str.startswith("(") and value_str.endswith(")"):
            inner = value_str[1:-1]
            elements = [self._parse_value(v.strip()) for v in inner.split(",") if v.strip()]
            return tuple(elements)

        # List [accuracy, loss]
        if value_str.startswith("[") and value_str.endswith("]"):
            inner = value_str[1:-1]
            elements = [self._parse_value(v.strip()) for v in inner.split(",") if v.strip()]
            return elements

        # Function call (e.g., adam(lr=1e-3))
        if "(" in value_str and value_str.endswith(")"):
            return value_str  # Keep as string for now

        # String (remove quotes if present)
        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            return value_str[1:-1]

        # Default: return as string
        return value_str

    def _parse_arguments(self, args_str: str) -> Tuple[List[Any], Dict[str, Any]]:
        """Parse function arguments into positional args and kwargs."""
        args = []
        kwargs = {}

        if not args_str.strip():
            return args, kwargs

        # Robust parsing: split by comma but respect nested parentheses/brackets
        parts = []
        current = ""
        paren_depth = 0
        bracket_depth = 0
        
        for char in args_str:
            if char == "(":
                paren_depth += 1
                current += char
            elif char == ")":
                paren_depth -= 1
                current += char
            elif char == "[":
                bracket_depth += 1
                current += char
            elif char == "]":
                bracket_depth -= 1
                current += char
            elif char == "," and paren_depth == 0 and bracket_depth == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += char
        
        if current:
            parts.append(current.strip())

        for part in parts:
            if not part:
                continue
            if "=" in part and "(" not in part.split("=")[0]: # Simple check for kwarg
                key, value_str = part.split("=", 1)
                kwargs[key.strip()] = self._parse_value(value_str.strip())
            else:
                args.append(self._parse_value(part))

        return args, kwargs

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
    return parser.parse()
