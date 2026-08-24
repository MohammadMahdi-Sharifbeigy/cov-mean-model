"""YAML experiment configuration loading, validation, and snapshots."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .config import DotDict


def to_dot_dict(value: Any) -> Any:
    """Recursively convert YAML mappings for notebook-compatible attribute access."""
    if isinstance(value, Mapping):
        return DotDict({key: to_dot_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_dot_dict(item) for item in value]
    return value


def _require(payload: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current or current[key] is None:
            raise ValueError(f"config.yaml must define {dotted_key}")
        current = current[key]
    return current


_REQUIRED_FIELDS = (
    "run_id", "seed", "session_idx", "device", "paths.data", "paths.logs", "paths.results",
    "training.batch_size", "training.max_epoch", "training.min_delta", "training.patience", "training.n_seeds",
    "model_type.name", "model_type.mean.n_hidden", "model_type.mean.n_heads", "model_type.mean.n_layers", "model_type.mean.nonlinearity", "model_type.mean.dropout",
    "model_type.cov.n_latent", "model_type.cov.n_hidden", "model_type.cov.n_heads", "model_type.cov.n_layers", "model_type.cov.nonlinearity", "model_type.cov.dropout",
    "optimization.optimizer_type.name", "optimization.scheduler_type.name",
    "data_preparation.spike_scale", "data_preparation.dense_indices", "data_preparation.sparse_indices", "data_preparation.log_task_var_names", "data_preparation.max_log_task_value", "data_preparation.unit_names_to_remove", "data_preparation.train_fraction",
)


def validate_config(payload: Mapping[str, Any], *, complete: bool = True) -> None:
    """Raise actionable errors before data loading or model training starts."""
    _require(payload, "paths.data")
    if not complete:
        return
    for field in _REQUIRED_FIELDS:
        _require(payload, field)
    optimizer = _require(payload, "optimization.optimizer_type")
    scheduler = _require(payload, "optimization.scheduler_type")
    optimizer_name, scheduler_name = str(optimizer["name"]), str(scheduler["name"])
    if optimizer_name not in {"Adam", "AdamW"}:
        raise ValueError("config.yaml optimization.optimizer_type.name must be Adam or AdamW")
    if scheduler_name not in {"Reduce", "Cosine"}:
        raise ValueError("config.yaml optimization.scheduler_type.name must be Reduce or Cosine")
    for field in ("lr", "weight_decay"):
        _require(optimizer, f"{optimizer_name}.{field}")
    for field in (("factor", "patience", "min_lr") if scheduler_name == "Reduce" else ("T_mult", "eta_min", "last_epoch")):
        _require(scheduler, f"{scheduler_name}.{field}")
    if not 0 < float(payload["data_preparation"]["train_fraction"]) < 1:
        raise ValueError("config.yaml data_preparation.train_fraction must be between zero and one")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return str(value) if isinstance(value, Path) else value


def write_resolved_config(conf: Mapping[str, Any], artifact_dir: Path) -> Path:
    """Persist the exact resolved configuration alongside run artifacts."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    destination = artifact_dir / "resolved-config.yaml"
    destination.write_text(yaml.safe_dump(_plain(conf), sort_keys=False), encoding="utf-8")
    return destination


def load_config(config_path: Path, run_dir: Path, *, validate_complete: bool = True) -> DotDict:
    """Load one YAML file, resolve path fields relative to it, then snapshot it."""
    source = Path(config_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {source}")
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("config.yaml must contain a mapping at its root")
    validate_config(payload, complete=validate_complete)
    conf = to_dot_dict(payload)
    for key in ("data", "logs", "results"):
        if key in conf.paths:
            value = Path(conf.paths[key]).expanduser()
            conf.paths[key] = str((value if value.is_absolute() else source.parent / value).resolve())
    write_resolved_config(conf, Path(run_dir).expanduser().resolve() / "artifacts")
    return conf
