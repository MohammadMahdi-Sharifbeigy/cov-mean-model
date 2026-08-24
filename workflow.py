"""Explicit, resumable orchestration for notebook Steps 1--7.

The notebook remains the historical record.  This module makes its active
data-generation sequence executable one step at a time: each step writes a
small named artifact below ``run_dir/artifacts`` and reconstructs prerequisites
when invoked directly.  ``all`` additionally runs the notebook comparison and
summary plots.
"""
from __future__ import annotations

import copy
import pickle
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DotDict, RunPaths, set_random_seed
from .configuration import load_config
from .data.preparation import create_train_valid_loaders, load_and_clean_session
from .data.selection import select_task_variables
from .data.datasets import Anscombe, NeuralDataset
from .synthetic.glm import fit_glm_per_unit_bin, predict_glm_means
from .synthetic.steps import (generate_step1, generate_step2_poisson_binwise,
    generate_step3_glm_poisson, generate_step5_shared_noise, generate_step6a_data,
    generate_step7_synthetic_pattern_cov)
from .training.comparison import run_model_comparisons
from .training.lightning import LitModel, predict_loader
from .training.seeded import run_seeded_training

STEP3_VARIABLES = ("tslp", "rew")
STEP4_VARIABLES = ("tslp", "rew", "tunp")
ALLOWED_TRAINING_VARIABLES = {name: list(STEP3_VARIABLES) for name in ("step3", "step4", "step5", "step6a", "step7")}


def load_run_config(paths: RunPaths, config_path: Path | None = None) -> DotDict:
    """Load this run's required YAML configuration and choose its torch device."""
    source = config_path or paths.root / "config.yaml"
    conf = load_config(source, paths.root)
    conf.device = torch.device(conf.device)
    return conf


def _artifact(paths: RunPaths, name: str) -> Path:
    directory = paths.root / "artifacts"; directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def _save_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as stream: pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as stream: return pickle.load(stream)


def _prepare(paths: RunPaths, conf: DotDict) -> dict[str, Any]:
    target = _artifact(paths, "prepared.pkl")
    if target.exists():
        prep = _load_pickle(target)
        # Metadata is deliberately reconstructed, not trusted from a pickle.
        conf.data = DotDict(prep["data_metadata"])
        return prep
    prep = load_and_clean_session(conf)
    prep["data_metadata"] = dict(conf.data)
    _save_pickle(target, prep)
    return prep


def _save_array(paths: RunPaths, name: str, value: np.ndarray) -> np.ndarray:
    np.save(_artifact(paths, name), value)
    return value


def _array(paths: RunPaths, name: str) -> np.ndarray | None:
    path = _artifact(paths, name)
    return np.load(path, allow_pickle=False) if path.exists() else None


def _glm(paths: RunPaths, prep: dict[str, Any], variable_names: tuple[str, ...], label: str) -> tuple[np.ndarray, np.ndarray]:
    betas_file, mean_file = f"betas_{label}.npy", f"lambda_hat_{label}.npy"
    betas, means = _array(paths, betas_file), _array(paths, mean_file)
    if betas is None or means is None:
        dense, sparse = select_task_variables(prep, variable_names)
        betas = fit_glm_per_unit_bin(prep["counts"], dense, sparse)
        means = predict_glm_means(betas, dense, sparse)
        _save_array(paths, betas_file, betas); _save_array(paths, mean_file, means)
    return betas, means


def run_step1(paths: RunPaths, conf: DotDict | None = None) -> np.ndarray:
    conf = conf or load_run_config(paths); set_random_seed(conf.seed)
    prep = _prepare(paths, conf); cached = _array(paths, "step1.npy")
    return cached if cached is not None else _save_array(paths, "step1.npy", generate_step1(prep, seed=conf.seed))


def run_step2(paths: RunPaths, conf: DotDict | None = None) -> np.ndarray:
    cached = _array(paths, "step2.npy")
    if cached is not None: return cached
    conf = conf or load_run_config(paths); run_step1(paths, conf); prep = _prepare(paths, conf)
    return cached if cached is not None else _save_array(paths, "step2.npy", generate_step2_poisson_binwise(prep, seed=conf.seed))


def run_step3(paths: RunPaths, conf: DotDict | None = None) -> np.ndarray:
    cached = _array(paths, "step3.npy")
    if cached is not None: return cached
    conf = conf or load_run_config(paths); run_step2(paths, conf); prep = _prepare(paths, conf); betas, _ = _glm(paths, prep, STEP3_VARIABLES, "step3")
    dense, sparse = select_task_variables(prep, STEP3_VARIABLES)
    return _save_array(paths, "step3.npy", generate_step3_glm_poisson(prep, betas, seed=conf.seed, X_dense=dense, X_sparse=sparse))


def run_step4(paths: RunPaths, conf: DotDict | None = None) -> np.ndarray:
    cached = _array(paths, "step4.npy")
    if cached is not None: return cached
    conf = conf or load_run_config(paths); run_step3(paths, conf); prep = _prepare(paths, conf); betas, _ = _glm(paths, prep, STEP4_VARIABLES, "step4")
    dense, sparse = select_task_variables(prep, STEP4_VARIABLES)
    return _save_array(paths, "step4.npy", generate_step3_glm_poisson(prep, betas, seed=conf.seed, X_dense=dense, X_sparse=sparse))


def run_step5(paths: RunPaths, conf: DotDict | None = None) -> np.ndarray:
    cached = _array(paths, "step5.npy")
    if cached is not None: return cached
    conf = conf or load_run_config(paths); run_step4(paths, conf); prep = _prepare(paths, conf); _, mean = _glm(paths, prep, STEP4_VARIABLES, "step4")
    return cached if cached is not None else _save_array(paths, "step5.npy", generate_step5_shared_noise(prep, mean, cov_type="structured", seed=conf.seed))


def _deep_empirical_covariance(paths: RunPaths, prep: dict[str, Any], conf: DotDict) -> tuple[np.ndarray, np.ndarray]:
    saved = _artifact(paths, "step6_deep_covariance.npz")
    if saved.exists():
        data = np.load(saved); return data["empirical_covs"], data["residuals_real"]
    # This is the active notebook Step-6a extraction: conditional mean + identity covariance on real data.
    split = create_train_valid_loaders(prep, conf)
    histories, model, _ = run_seeded_training(conf, lambda: LitModel(conf, "conditional", "identity").to(conf.device), split["train_loader"], split["valid_loader"], n_seeds=1, checkpoint_prefix=_artifact(paths, "step6_deep_mean"))
    pos = prep["X_position"]; dense = prep["X_dense_task"]; sparse = prep["X_sparse_task"]
    train = split["train_idx"]
    pos = (pos - pos[train].mean(0, keepdims=True)) / (pos[train].std(0, keepdims=True) + 1e-8)
    dense = (dense - dense[train].mean(0, keepdims=True)) / (dense[train].std(0, keepdims=True) + 1e-8)
    mean = Anscombe()(torch.as_tensor(prep["Y"][train], dtype=torch.float32, device=conf.device).transpose(1, 2)).mean((0, 1), keepdim=True)
    dataset = NeuralDataset(pos, dense, sparse, prep["Y"], mean)
    _, predicted = predict_loader(conf, model, DataLoader(dataset, batch_size=conf.training.batch_size), mean, "mean")
    # predict_loader returns (trial, bin, unit); notebook residuals are (trial, unit, bin).
    residuals = prep["counts"] - np.transpose(predicted / prep["spike_scale"], (0, 2, 1))
    empirical = np.stack([np.cov(residuals[:, :, bin_index], rowvar=False) for bin_index in range(residuals.shape[2])])
    np.savez(saved, empirical_covs=empirical, residuals_real=residuals)
    return empirical, residuals


def run_step6(paths: RunPaths, conf: DotDict | None = None) -> np.ndarray:
    cached = _array(paths, "step6a.npy")
    if cached is not None: return cached
    conf = conf or load_run_config(paths); run_step5(paths, conf); prep = _prepare(paths, conf); _, means = _glm(paths, prep, STEP3_VARIABLES, "step3")
    # Preserve the notebook's RNG isolation around deep-model training.
    torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    empirical, residuals = _deep_empirical_covariance(paths, prep, conf)
    torch.set_rng_state(torch_state); np.random.set_state(numpy_state); random.setstate(python_state)
    if cuda_state is not None: torch.cuda.set_rng_state_all(cuda_state)
    return _save_array(paths, "step6a.npy", generate_step6a_data(prep, empirical, residuals, means, seed=conf.seed))


def run_step7(paths: RunPaths, conf: DotDict | None = None) -> np.ndarray:
    cached = _array(paths, "step7.npy")
    if cached is not None: return cached
    conf = conf or load_run_config(paths); run_step6(paths, conf); prep = _prepare(paths, conf); _, means = _glm(paths, prep, STEP3_VARIABLES, "step3")
    return cached if cached is not None else _save_array(paths, "step7.npy", generate_step7_synthetic_pattern_cov(prep, means, noise_scale=1.0, ridge=1e-6, seed=conf.seed))


def run_all(paths: RunPaths, conf: DotDict | None = None) -> tuple[Any, Any, Any]:
    conf = conf or load_run_config(paths); run_step7(paths, conf); prep = _prepare(paths, conf)
    datasets = {"real": prep["Y"], "step1": _array(paths, "step1.npy"), "step2": _array(paths, "step2.npy"), "step3": _array(paths, "step3.npy"), "step4": _array(paths, "step4.npy"), "step5": _array(paths, "step5.npy"), "step6a": _array(paths, "step6a.npy"), "step7": _array(paths, "step7.npy")}
    return run_model_comparisons(datasets, prep["X_position"], prep, conf, conf.device, n_splits=10, force_retrain=False, use_multi_seed=False, allowed_vars_dict=ALLOWED_TRAINING_VARIABLES, checkpoint_root=paths.root, results_root=Path(conf.paths.results))


_DISPATCH: dict[str, Callable[[RunPaths, DotDict | None], Any]] = {"step1": run_step1, "step2": run_step2, "step3": run_step3, "step4": run_step4, "step5": run_step5, "step6": run_step6, "step7": run_step7, "all": run_all}


def dispatch(command: str, paths: RunPaths, conf: DotDict | None = None) -> Any:
    """Run one named notebook step.  Imports and dispatch are side-effect free."""
    try: return _DISPATCH[command](paths, conf)
    except KeyError as error: raise ValueError(f"Unknown workflow command: {command}") from error
