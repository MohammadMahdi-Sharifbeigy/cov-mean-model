"""Training and cross-validation interfaces."""

from .comparison import run_model_comparisons
from .lightning import LitModel

__all__ = ["LitModel", "run_model_comparisons"]
