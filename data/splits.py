"""Deterministic train/validation splitting and training-only normalization."""
from __future__ import annotations
import numpy as np


def deterministic_trial_split(n_trials: int, seed: int, train_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(n_trials, size=int(n_trials * train_fraction), replace=False)
    return train_idx, np.setdiff1d(np.arange(n_trials), train_idx)


def standardize_from_train(train: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean, std = train.mean(axis=0, keepdims=True), train.std(axis=0, keepdims=True)
    return (train - mean) / (std + 1e-8), (valid - mean) / (std + 1e-8), mean, std
