"""Faithful, callable extraction of the notebook's real-data preparation cells."""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.io as sio
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from mean_cov_pipeline.config import DotDict
from mean_cov_pipeline.data.datasets import Anscombe, NeuralDataset
from mean_cov_pipeline.data.splits import deterministic_trial_split, standardize_from_train


def _setting(conf: Any, name: str) -> Any:
    source = getattr(conf, "data_preparation", None)
    if source is None:
        raise ValueError("config.yaml must define data_preparation")
    try:
        return source[name] if isinstance(source, dict) else getattr(source, name)
    except (KeyError, AttributeError) as error:
        raise ValueError(f"config.yaml must define data_preparation.{name}") from error


def _exact_column(names: np.ndarray, requested: str) -> int:
    matches = np.flatnonzero(names == requested)
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one task variable named {requested!r}; found {len(matches)}")
    return int(matches[0])


def _set_metadata(conf: Any, spikes: np.ndarray, position: np.ndarray, dense: list[int], sparse: list[int], bin_size: Any) -> None:
    data = DotDict()
    data.bin_size, data.n_bins = bin_size, spikes.shape[2]
    data.n_position_vars, data.n_dense_vars, data.n_sparse_vars = position.shape[1], len(dense), len(sparse)
    data.n_vars, data.n_units = data.n_position_vars + data.n_dense_vars + data.n_sparse_vars, spikes.shape[1]
    conf.data = data


def load_and_clean_session(conf: Any, *, loadmat: Any = sio.loadmat) -> dict[str, Any]:
    """Load and clean a session in the exact order used by the notebook.

    Every editable choice comes from ``config.yaml`` under
    ``data_preparation``; no data-preparation defaults are hidden in code.
    """
    raw = loadmat(conf.paths.data, squeeze_me=True, struct_as_record=False)
    session_idx = int(conf.session_idx)
    session_name = raw["session_names"][session_idx]
    spikes = np.transpose(raw["spikes"][session_idx], (1, 0, 2)) * _setting(conf, "spike_scale")
    position = np.transpose(raw["position_vars"][session_idx], (1, 0, 2))
    task = np.transpose(raw["task_vars"][session_idx], (1, 0))
    position_names, task_names = np.asarray(raw["position_var_names"]), np.asarray(raw["task_var_names"])
    units = np.asarray(raw["unit_names"][session_idx])

    remove_mask = np.isin(units, _setting(conf, "unit_names_to_remove"))
    units, spikes = units[~remove_mask], spikes[:, ~remove_mask]
    finite = np.isfinite(spikes).all(axis=(1, 2)) & np.isfinite(position).all(axis=(1, 2)) & np.isfinite(task).all(axis=1)
    zero_position = (position == 0).all(axis=(1, 2))
    invalid_task = np.zeros(task.shape[0], dtype=bool)
    log_names = tuple(_setting(conf, "log_task_var_names"))
    for name in log_names:
        values = task[:, _exact_column(task_names, name)]
        invalid_task |= (values > _setting(conf, "max_log_task_value")) | (values <= 0)
    keep = finite & ~zero_position & ~invalid_task
    spikes, position, task = spikes[keep], position[keep], task[keep]
    for name in log_names:
        task[:, _exact_column(task_names, name)] = np.log(task[:, _exact_column(task_names, name)])
    task = MinMaxScaler(feature_range=(0, 1)).fit_transform(task)

    dense, sparse = list(_setting(conf, "dense_indices")), list(_setting(conf, "sparse_indices"))
    if set(dense) & set(sparse) or set(dense + sparse).difference(range(task.shape[1])):
        raise ValueError("dense_indices and sparse_indices must be disjoint valid task columns")
    _set_metadata(conf, spikes, position, dense, sparse, raw["bin_size"])
    return {
        "session_name": session_name, "spikes": spikes, "position_vars": position, "task_vars": task,
        "position_var_names": position_names, "task_var_names": task_names,
        "variable_names": np.concatenate((position_names, task_names)), "unit_names": units,
        "events": raw["events"][session_idx], "event_names": raw["event_names"],
        "channel_names": raw["channel_names"][session_idx], "unit_types": raw["unit_types"][session_idx],
        "bin_size": raw["bin_size"], "bin_times": raw["bin_times"], "trials_mask_keep": keep,
        "dense_indices": dense, "sparse_indices": sparse, "X_position": position,
        "X_dense_task": task[:, dense], "X_sparse_task": task[:, sparse], "X_full_task": task,
        "Y": spikes, "counts": np.round(spikes / _setting(conf, "spike_scale")).astype(int),
        "spike_scale": _setting(conf, "spike_scale"), "n_units": spikes.shape[1],
    }


def create_train_valid_loaders(prep: Mapping[str, Any], conf: Any, device: Any | None = None) -> dict[str, Any]:
    """Create notebook-equivalent split arrays, datasets, and seeded loaders."""
    y, position, dense, sparse = prep["Y"], prep["X_position"], prep["X_dense_task"], prep["X_sparse_task"]
    train_idx, valid_idx = deterministic_trial_split(len(y), int(conf.seed), _setting(conf, "train_fraction"))
    result: dict[str, Any] = {"train_idx": train_idx, "valid_idx": valid_idx}
    for name, values in {"position": position, "dense": dense, "sparse": sparse, "Y": y}.items():
        result[f"{name}_train"], result[f"{name}_valid"] = values[train_idx], values[valid_idx]
    result["position_train"], result["position_valid"], result["position_mean"], result["position_std"] = standardize_from_train(result["position_train"], result["position_valid"])
    result["dense_train"], result["dense_valid"], result["dense_mean"], result["dense_std"] = standardize_from_train(result["dense_train"], result["dense_valid"])
    target_device = device if device is not None else getattr(conf, "device", "cpu")
    train_tensor = torch.tensor(result["Y_train"], dtype=torch.float32, device=target_device).transpose(1, 2)
    y_mean = Anscombe().forward(train_tensor).mean(dim=(0, 1), keepdim=True)
    train_ds = NeuralDataset(result["position_train"], result["dense_train"], result["sparse_train"], result["Y_train"], y_mean)
    valid_ds = NeuralDataset(result["position_valid"], result["dense_valid"], result["sparse_valid"], result["Y_valid"], y_mean)
    train_generator, valid_generator = torch.Generator(), torch.Generator()
    train_generator.manual_seed(int(conf.seed)); valid_generator.manual_seed(int(conf.seed))
    args = {"batch_size": int(conf.training.batch_size), "num_workers": 0, "pin_memory": True, "persistent_workers": False}
    result.update(train_dataset=train_ds, valid_dataset=valid_ds, Y_train_mean=y_mean,
                  train_loader=DataLoader(train_ds, shuffle=True, generator=train_generator, **args),
                  valid_loader=DataLoader(valid_ds, shuffle=False, generator=valid_generator, **args),
                  loader_generators=(train_generator, valid_generator))
    return result
