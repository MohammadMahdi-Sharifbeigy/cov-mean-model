"""Notebook-compatible generators for synthetic Steps 1--7.

Step 4 intentionally uses :func:`generate_step3_glm_poisson` with a GLM fit
on its selected (including hidden) variables.  The exported notebook did not
call a separate Step-4 generator.
"""

from __future__ import annotations

import warnings

import numpy as np

from .glm import DEFAULT_SEED, predict_glm_means

DEFAULT_STEP1_NOISE = "poisson"


def generate_step1(prep: dict, noise: str = DEFAULT_STEP1_NOISE, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Generate an iid per-neuron baseline while preserving marginal means."""
    rng = np.random.default_rng(seed + 202)
    counts = prep["counts"]
    mean = counts.mean(axis=(0, 2), keepdims=True)
    if noise == "poisson":
        synthetic = rng.poisson(np.broadcast_to(mean, counts.shape)).astype(float)
    elif noise == "gaussian":
        std = counts.std(axis=(0, 2), keepdims=True)
        synthetic = np.clip(rng.normal(mean, std, size=counts.shape), 0, None)
    else:
        raise ValueError(f"unknown STEP1_NOISE: {noise}")
    return synthetic * prep["spike_scale"]


def generate_step2_poisson_binwise(prep: dict, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Generate iid Poisson counts preserving each neuron/bin PSTH mean."""
    rng = np.random.default_rng(seed + 303)
    counts = prep["counts"]
    n_trials, n_units, n_bins = counts.shape
    mean_per_unit_bin = counts.mean(axis=0, keepdims=True)
    zero_cells = int((mean_per_unit_bin == 0).sum())
    if zero_cells:
        warnings.warn(
            f"generate_step2_poisson_binwise: {zero_cells} (neuron,bin) cells with μ=0; "
            "those entries will be 0 in the synthetic tensor.",
            RuntimeWarning,
            stacklevel=2,
        )
    synthetic = rng.poisson(np.broadcast_to(mean_per_unit_bin, (n_trials, n_units, n_bins))).astype(float)
    assert synthetic.shape == counts.shape
    assert synthetic.dtype == float
    assert np.all(synthetic >= 0)
    return synthetic * prep["spike_scale"]


def generate_step3_glm_poisson(
    prep: dict, betas: np.ndarray, seed: int = DEFAULT_SEED, *,
    X_dense: np.ndarray | None = None, X_sparse: np.ndarray | None = None,
) -> np.ndarray:
    """Generate the Step 3/4 Poisson null from explicitly selected GLM inputs."""
    rng = np.random.default_rng(seed + 505)
    counts = prep["counts"]
    dense = prep["X_dense_task"] if X_dense is None else X_dense
    sparse = prep["X_sparse_task"] if X_sparse is None else X_sparse
    rates = predict_glm_means(betas, dense, sparse)
    large_cells = int((rates > 1000).sum())
    if large_cells:
        warnings.warn(f"generate_step3a: {large_cells} cells with lambda>1000 - check GLM fit.", RuntimeWarning, stacklevel=2)
    synthetic = rng.poisson(rates).astype(float)
    assert synthetic.shape == counts.shape
    assert np.all(synthetic >= 0)
    return synthetic * prep["spike_scale"]


def generate_step5_shared_noise(
    prep: dict, lambda_hat: np.ndarray, cov_type: str = "global", n_blocks: int = 4,
    seed: int = DEFAULT_SEED, noise_scale: float = 1.0, **_: object,
) -> np.ndarray:
    """Add residual covariance sampled from the real counts to GLM means."""
    rng = np.random.default_rng(seed)
    true_counts = prep["counts"]
    n_trials, n_units, n_bins = true_counts.shape
    eps = 1e-8
    residuals = (true_counts - lambda_hat) / np.sqrt(lambda_hat + eps)
    synthetic_noise = np.zeros_like(residuals)
    if cov_type == "global":
        covariance = np.cov(residuals.transpose(0, 2, 1).reshape(n_trials * n_bins, n_units), rowvar=False)
        flat_noise = rng.multivariate_normal(np.zeros(n_units), covariance * noise_scale, size=n_trials * n_bins)
        synthetic_noise = flat_noise.reshape(n_trials, n_bins, n_units).transpose(0, 2, 1)
    elif cov_type == "binwise":
        for bin_index in range(n_bins):
            covariance = np.cov(residuals[:, :, bin_index], rowvar=False)
            synthetic_noise[:, :, bin_index] = rng.multivariate_normal(np.zeros(n_units), covariance * noise_scale, size=n_trials)
    elif cov_type == "structured":
        covariance = np.cov(residuals.transpose(0, 2, 1).reshape(n_trials * n_bins, n_units), rowvar=False)
        structured = np.zeros_like(covariance)
        block_size = int(np.ceil(n_units / n_blocks))
        for block in range(n_blocks):
            start, end = block * block_size, min((block + 1) * block_size, n_units)
            if start < n_units:
                structured[start:end, start:end] = covariance[start:end, start:end]
        eigenvalues, eigenvectors = np.linalg.eigh(structured)
        structured = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-8, None)) @ eigenvectors.T
        flat_noise = rng.multivariate_normal(np.zeros(n_units), structured * noise_scale, size=n_trials * n_bins)
        synthetic_noise = flat_noise.reshape(n_trials, n_bins, n_units).transpose(0, 2, 1)
    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")
    rates = np.clip(lambda_hat + synthetic_noise * np.sqrt(lambda_hat + eps), 1e-8, 1000.0)
    return rng.poisson(rates).astype(float) * prep.get("spike_scale", 1.0)


def generate_step6a_data(
    prep: dict, empirical_covs: np.ndarray, residuals_real: np.ndarray,
    lambda_hat_step3: np.ndarray, seed: int = 42,
) -> np.ndarray:
    """Inject deep-model empirical covariance into Step 3 GLM means."""
    rng = np.random.default_rng(seed)
    n_trials, n_units, n_bins = lambda_hat_step3.shape
    noise = np.zeros_like(lambda_hat_step3)
    for bin_index in range(n_bins):
        noise[:, :, bin_index] = rng.multivariate_normal(np.zeros(n_units), empirical_covs[bin_index], size=n_trials)
    scale_factor = np.std(residuals_real) / (np.std(noise) + 1e-8)
    print(f"-> [Step 6a] Variance Control Scale Factor: {scale_factor:.4f}")
    rates = np.clip(lambda_hat_step3 + noise * scale_factor * np.sqrt(lambda_hat_step3 + 1e-8), 1e-8, 1000.0)
    return rng.poisson(rates).astype(float) * prep.get("spike_scale", 1.0)


def generate_step7_synthetic_pattern_cov(
    prep: dict, lambda_hat: np.ndarray, noise_scale: float = 0.5, ridge: float = 1e-6,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Generate the notebook's active, purely synthetic pattern-covariance null."""
    rng = np.random.default_rng(seed)
    n_trials, n_units, n_bins = lambda_hat.shape
    pattern = np.eye(n_units)
    pattern[0:11, 11:22] = pattern[11:22, 0:11] = -0.90
    pattern[0:11, 0:11] = pattern[11:22, 11:22] = 0.10
    pattern[16:36, 16:36] = 0.12
    pattern[30:50, 30:50] = 0.85
    pattern[31:41, 6:16] = pattern[6:16, 31:41] = 0.85
    np.fill_diagonal(pattern, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(pattern)
    covariance = (eigenvectors @ np.diag(np.clip(eigenvalues, ridge, None)) @ eigenvectors.T) * noise_scale
    flat_noise = rng.multivariate_normal(np.zeros(n_units), covariance, size=n_trials * n_bins)
    noise = flat_noise.reshape(n_trials, n_bins, n_units).transpose(0, 2, 1)
    rates = np.clip(lambda_hat + noise * np.sqrt(lambda_hat + 1e-8), 1e-8, 1000.0)
    synthetic = rng.poisson(rates).astype(float)
    assert synthetic.shape == (n_trials, n_units, n_bins)
    assert np.all(synthetic >= 0)
    return synthetic * prep.get("spike_scale", 1.0)
