"""Source spans carried through analysis and machine-readable diagnostics."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Optional


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


@dataclass(frozen=True)
class CompilationDiagnostic:
    """A stable, serializable error across compiler API and CLI boundaries."""

    stage: str
    message: str
    span: Optional[SourceSpan] = None
    code: Optional[str] = None
    location: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


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
