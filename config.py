"""Small configuration primitives shared by pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunPaths:
    """Resolved root directory used for one pipeline execution."""

    root: Path

    @classmethod
    def from_root(cls, root: Path) -> "RunPaths":
        """Create paths anchored at an absolute, normalized run root."""
        return cls(root=Path(root).expanduser().resolve())


class DotDict(dict[str, Any]):
    """Dictionary with attribute access for notebook-compatible settings."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def set_random_seed(seed: int) -> None:
    """Seed standard numerical libraries when they are installed.

    The pipeline can be imported in lightweight environments, so PyTorch is
    intentionally optional at this configuration seam.
    """
    import random

    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
