"""Source spans carried through analysis and machine-readable diagnostics."""

from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpan:
    line: int
    column: int
    end_line: int
    end_column: int

    @classmethod
    def from_node(cls, node):
        return cls(
            node.line,
            node.column or 1,
            node.end_line or node.line,
            node.end_column or (node.column or 1) + 1,
        )


class LocatedError(ValueError):
    def __init__(self, message, span):
        self.span = span
        super().__init__(f"{message} at line {span.line}, column {span.column}")


@contextmanager
def at_config(node, key=None):
    """Add the relevant setting location without replacing more precise errors."""
    try:
        yield
    except ValueError as error:
        if isinstance(error, LocatedError):
            raise
        span = node.config_spans.get(key, SourceSpan.from_node(node))
        raise LocatedError(str(error), span) from error
