"""Poisson GLM fitting and prediction used by synthetic Steps 3--7."""

from __future__ import annotations

import warnings

import numpy as np

DEFAULT_SEED = 42


def _build_design_matrix(X_dense: np.ndarray, X_sparse: np.ndarray) -> np.ndarray:
    """Concatenate task variables and prepend the GLM intercept."""
    import statsmodels.api as sm

    return sm.add_constant(np.concatenate([X_dense, X_sparse], axis=1), prepend=True)


def _fit_one_glm(y_nb: np.ndarray, X_design: np.ndarray) -> np.ndarray:
    """Fit one Poisson log-linear GLM, with a stable intercept-only fallback."""
    n_coefficients = X_design.shape[1]
    mean_count = float(y_nb.mean())
    if mean_count == 0.0:
        beta = np.zeros(n_coefficients)
        beta[0] = np.log(1e-8)
        return beta
    try:
        import statsmodels.api as sm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sm.GLM(
                y_nb,
                X_design,
                family=sm.families.Poisson(link=sm.families.links.Log()),
            ).fit(maxiter=200, disp=False)
        return np.asarray(result.params, dtype=float)
    except Exception:
        beta = np.zeros(n_coefficients)
        beta[0] = np.log(max(mean_count, 1e-8))
        return beta


def fit_glm_per_unit_bin(
    counts: np.ndarray, X_dense: np.ndarray, X_sparse: np.ndarray, n_jobs: int = -1,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Fit one Poisson GLM per neuron/bin using all trials.

    ``seed`` remains part of the notebook-compatible interface.  Statsmodels'
    deterministic fit does not otherwise consume it.
    """
    del seed
    from joblib import Parallel, delayed

    _, n_units, n_bins = counts.shape
    design = _build_design_matrix(X_dense, X_sparse)
    tasks = [(counts[:, unit, bin_index], design) for unit in range(n_units) for bin_index in range(n_bins)]
    print(f"Fitting {n_units * n_bins} Poisson GLMs  ({n_units} units x {n_bins} bins) ...")
    flat_betas = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(_fit_one_glm)(target, design_matrix) for target, design_matrix in tasks
    )
    betas = np.asarray(flat_betas, dtype=float).reshape(n_units, n_bins, design.shape[1])
    print(f"  betas shape: {betas.shape}  (neurons, bins, V+1)")
    return betas


def predict_glm_means(betas: np.ndarray, X_dense: np.ndarray, X_sparse: np.ndarray) -> np.ndarray:
    """Return ``exp(X @ beta)`` rates with the notebook's overflow clipping."""
    design = _build_design_matrix(X_dense, X_sparse)
    eta = (design[:, np.newaxis, np.newaxis, :] * betas[np.newaxis, :, :, :]).sum(axis=-1)
    return np.exp(np.clip(eta, -30, 30))
