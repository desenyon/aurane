"""
Backend registry utilities.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from ..ast import AuraneProgram

BackendGenerator = Callable[[AuraneProgram], str]

_backend_generators: Dict[str, BackendGenerator] = {}
_backend_cache_versions: Dict[str, Optional[str]] = {}


def register_backend_generator(
    name: str, generator: BackendGenerator, *, cache_version: Optional[str] = None
) -> None:
    """Register a generator; opt into caching with an explicit implementation version."""
    normalized = name.lower().strip()
    if not normalized:
        raise ValueError("Backend name must be non-empty")
    _backend_generators[normalized] = generator
    _backend_cache_versions[normalized] = cache_version


def get_backend_cache_version(name: str) -> Optional[str]:
    return _backend_cache_versions.get(name.lower().strip())


def get_backend_generator(name: str) -> BackendGenerator:
    """Get a backend generator by name."""
    normalized = name.lower().strip()
    if normalized not in _backend_generators:
        supported = ", ".join(sorted(_backend_generators.keys()))
        raise KeyError(f"Unsupported backend '{name}'. Supported: {supported}")
    return _backend_generators[normalized]


def _register_default_backends() -> None:
    # Local import to avoid import cycles.
    from .torch_backend import generate_torch_code_backend

    register_backend_generator("torch", generate_torch_code_backend, cache_version="2")


_register_default_backends()
