"""Synthetic-data recovery metrics from the experiment notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_step6a_recovery(df_history, empirical_covs, K, N):
    print("Evaluating Step 6a Covariance Recovery...")
    
    # We plot the empirical covariance for Bin 5 as an example
    bin_idx = min(6, K - 1)
    true_cov = empirical_covs[bin_idx]
    
    # In a full implementation, you would load the trained SharedCovModel checkpoint
    # for condition='step6a' and extract its lambda_matrix parameter.
    # Since run_model_comparisons is a black box here, we can visualize the True Covariance
    # that we injected, and print instructions for model extraction.
    
    fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    im = axes.imshow(true_cov, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-np.max(np.abs(true_cov)), vmax=np.max(np.abs(true_cov)))
    axes.set_title(f"True Empirical Covariance (Bin {bin_idx})")
    axes.set_xlabel("Unit")
    axes.set_ylabel("Unit")
    fig.colorbar(im, ax=axes)
    plt.show()
