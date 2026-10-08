from __future__ import annotations

import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from scipy.stats import wilcoxon
from sklearn.decomposition import PCA

import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import math
from typing import Tuple, Optional
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import LinearSegmentedColormap

_MODEL_COLORS = {
'Zero / Identity': '#A9A9A9', # Gray
'Baseline / Identity': '#4878CF', # Blue
'Baseline / Shared': '#9B59B6', # Amethyst - great alternative for better differentiation
'Conditional / Identity': '#E87B2C', # Dark Orange
'Conditional / Shared': '#6ACC65', # Green
}

_COND_HUE = {
"real": "#0F8B8D", # teal
"step0": "#B5179E", # magenta
"step1": "#3A0CA3", # indigo
"step2": "#F77F00", # amber
"step2b": "#D62828", # crimson
"step3": "#8AC926", # apple green
"step3b": "#FFCA3A", # warm gold
"step4": "#4361EE", # royal blue (changed to contrast with teal and pink)
"step5": "#EF476F", # rose pink
"step6": "#7209B7", # Deep Violet - very distinctive and fits the theme
"step7": "#06D6A0", # Emerald green - beta-factor synthetic covariance
}

def save_figure(fig, file_name, file_path, ext=".png", transparent=False, **savefig_kwargs):
    file_path = Path(file_path)
    file_path.mkdir(parents=True, exist_ok=True)
    dpi = savefig_kwargs.pop("dpi", 100)

    fig.savefig(
        file_path / f"{file_name}{ext}",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.25,
        transparent=transparent,
        **savefig_kwargs,
    )

    plt.close(fig)

    return file_path


def save_shap_plot(make_plot, title, file_name, file_path, figsize=(10, 6)):
    plt.close("all")

    plot_result = make_plot()

    if hasattr(plot_result, "figure"):
        fig = plot_result.figure
        ax = plot_result
    else:
        fig = plt.gcf()
        ax = plt.gca()

    ax.set_title(title, pad=20)
    fig.canvas.draw()

    return save_figure(fig, file_name, file_path)

def p_to_stars(p: float) -> str:
    """Convert a p-value to significance stars."""
    if p is None or p != p:
        return "n.s."
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _hue_cmap(hex_color: str):
    """White -> hue LinearSegmentedColormap for a single-hue raster panel."""
    return LinearSegmentedColormap.from_list("_h", ["#FFFFFF", hex_color])

def set_pub_style():
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
    })

def aggregate(X, axis, method="mean"):
    X = np.asarray(X)

    dist_names = {"normal", "uniform", "exponential", "gamma", "beta", "poisson", "nb"}

    if method in dist_names:
        def _sim_1d(v):
            v = np.asarray(v).reshape(-1)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                return np.nan
            d = Distribution.fit(v, method)
            return d.similarity(v, method="js")

        if X.ndim == 1:
            return _sim_1d(X)
        return np.apply_along_axis(_sim_1d, axis, X)

    if method == "mean":
        return np.nanmean(X, axis=axis)

    if method == "median":
        return np.nanmedian(X, axis=axis)

    if method == "std":
        return np.nanstd(X, axis=axis)

    if method == "mad":
        med = np.nanmedian(X, axis=axis, keepdims=True)
        return np.nanmedian(np.abs(X - med), axis=axis)

    if method == "max":
        return np.nanmax(X, axis=axis)

    if method == "min":
        return np.nanmin(X, axis=axis)

    if method == "rms":
        return np.sqrt(np.nanmean(X ** 2, axis=axis))

    if method == "kurtosis":
        return stats.kurtosis(X, axis=axis, nan_policy="omit", fisher=False)

    if method == "max_abs":
        idx = np.nanargmax(np.abs(X), axis=axis)
        return np.take_along_axis(X, idx[..., None], axis=axis).squeeze(axis=axis)

    if method == "fisher":
        eps = 1e-10
        chi2 = -2 * np.nansum(np.log(X + 1e-10), axis=axis)
        df = 2 * X.shape[axis]
        return stats.chi2.sf(chi2, df)

    raise ValueError(f"Unknown method: {method}")

def run_pairwise_wilcoxon(a, b, magnitude_threshold=None):
    """
    Runs a Wilcoxon signed-rank test on two related samples a and b.
    If magnitude_threshold is provided, forces p_val to 1.0 if the 
    absolute mean difference is less than the threshold.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    # Filter out NaNs
    mask = ~np.isnan(a) & ~np.isnan(b)
    a_clean, b_clean = a[mask], b[mask]
    
    # Wilcoxon test requires at least a few samples
    if len(a_clean) < 5:
        return {'p_val': np.nan, 'statistic': np.nan}
        
    # Effect size thresholding (ignore tiny differences)
    if magnitude_threshold is not None:
        mean_diff = np.abs(np.mean(a_clean) - np.mean(b_clean))
        if mean_diff < magnitude_threshold:
            return {'p_val': 1.0, 'statistic': 0.0}
            
    try:
        # zero_method='zsplit' handles ties and zeros safely
        stat, p_val = wilcoxon(a_clean, b_clean)
        return {'p_val': p_val, 'statistic': stat}
    except ValueError:
        # Catches errors like "all differences are zero"
        return {'p_val': 1.0, 'statistic': 0.0}

def filter_task_vars(prep, selected_vars=None):
    """Filter task variables by name for GLMs."""
    if selected_vars is None:
        return prep['X_dense_task'], prep['X_sparse_task']
        
    dense_cols = []
    sparse_cols = []
    task_names = list(prep['task_var_names_full'])
    
    for var in selected_vars:
        if var in task_names:
            idx = task_names.index(var)
            if idx in prep['dense_indices_full']:
                dense_cols.append(idx)
            elif idx in prep['sparse_indices_full']:
                sparse_cols.append(idx)
        else:
            print(f"Warning: '{var}' not found in task_var_names")
            
    if dense_cols:
        X_dense = prep['X_full_task'][:, dense_cols]
    else:
        X_dense = np.zeros((prep['X_full_task'].shape[0], 0))
        
    if sparse_cols:
        X_sparse = prep['X_full_task'][:, sparse_cols]
    else:
        X_sparse = np.zeros((prep['X_full_task'].shape[0], 0))
        
    return X_dense, X_sparse

def plot_firing_rate_histogram(Y_real, Y_test, test_label='step3a (Poisson)', save_path=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(np.asarray(Y_real).ravel(), bins=40, alpha=0.6, label='real', density=True)
    ax.hist(np.asarray(Y_test).ravel(), bins=40, alpha=0.6, label=test_label, density=True)
    ax.set_xlabel('Firing rate')
    ax.set_ylabel('Density')
    ax.set_title(f'Real vs {test_label}')
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    return fig

def plot_mean_variance_comparison(Y_ref: np.ndarray, Y_test: np.ndarray, prep: dict,
                                  label_ref: str = 'Real', label_test: str = 'Synthetic',
                                  save_path: str = None):
    """Compares the mean firing rate and variance per unit between two datasets."""
    set_pub_style()
    
    # Convert rates back to raw counts for accurate variance calculation
    counts_ref = Y_ref / prep['spike_scale']
    counts_test = Y_test / prep['spike_scale']
    
    # Calculate means and variances in count space
    mu_ref_counts = counts_ref.mean(axis=(0, 2))
    var_ref_counts = counts_ref.var(axis=(0, 2))
    
    mu_test_counts = counts_test.mean(axis=(0, 2))
    var_test_counts = counts_test.var(axis=(0, 2))
    
    # Convert mean counts back to Hz for the first panel
    rate_ref = mu_ref_counts / prep['spike_scale']
    rate_test = mu_test_counts / prep['spike_scale']
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # ── 1. Mean Firing Rate ──
    axes[0].scatter(rate_ref, rate_test, alpha=0.8, edgecolors='w', linewidth=0.5, s=60)
    max_rate = max(rate_ref.max(), rate_test.max())
    axes[0].plot([0, max_rate * 1.05], [0, max_rate * 1.05], 'k--', lw=1.2, alpha=0.6)
    axes[0].set_xlabel(f'{label_ref} Mean Firing Rate (Hz)', fontweight='bold')
    axes[0].set_ylabel(f'{label_test} Mean Firing Rate (Hz)', fontweight='bold')
    axes[0].set_title('Mean Firing Rate per Unit', pad=10)
    
    # ── 2. Variance of Counts ──
    axes[1].scatter(var_ref_counts, var_test_counts, alpha=0.8, edgecolors='w', linewidth=0.5, s=60, color='C1')
    max_var = max(var_ref_counts.max(), var_test_counts.max())
    axes[1].plot([0, max_var * 1.05], [0, max_var * 1.05], 'k--', lw=1.2, alpha=0.6)
    axes[1].set_xlabel(f'{label_ref} Variance (Spike Counts)', fontweight='bold')
    axes[1].set_ylabel(f'{label_test} Variance (Spike Counts)', fontweight='bold')
    axes[1].set_title('Variance of Spike Counts per Unit', pad=10)
    
    sns.despine()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
    plt.show()
    return fig

def plot_population_raster(
    conditions: dict,
    prep: dict,
    n_trials_show: int = 40,
    save_path: Optional[str] = None,
    seed: int = 1,
    title: str = "Population raster (sub-sampled trials per neuron)",
):
    """Population raster for ANY condition dict.
    
    Neurons are stacked vertically with white gaps for crisp separation.
    """
    set_pub_style()
    t = prep.get("bin_centers", np.linspace(-1, 2, 100))
    t0, t1 = float(t[0]), float(t[-1])
    labels = list(conditions.keys())
    n_units = prep.get("n_units", 63)

    rng = np.random.default_rng(seed)
    counts0 = np.asarray(conditions[labels[0]])
    n_trials = counts0.shape[0]
    n_show = min(n_trials_show, n_trials)
    sel = np.sort(rng.choice(n_trials, n_show, replace=False))

    vmax = max(np.percentile(np.asarray(conditions[l])[sel], 99) for l in labels)
    vmax = max(vmax, 1e-9)

    gap = max(1, round(n_show * 0.18))
    block_h = n_show + gap
    total_h = n_units * block_h

    fig, axes = plt.subplots(1, len(labels), figsize=(3.6 * len(labels), 5.0),
                              squeeze=False, dpi=150)

    for ax, lab in zip(axes[0], labels):
        Y = np.asarray(conditions[lab])[sel]
        cmap = _hue_cmap(_COND_HUE.get(lab, "#333333"))
        cmap.set_bad("white")

        n_bins = Y.shape[-1]
        img = np.full((total_h, n_bins), np.nan)
        for u in range(n_units):
            r0 = u * block_h
            img[r0:r0 + n_show, :] = Y[:, u, :]

        ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                  extent=[t0, t1, total_h, 0], interpolation="nearest")

        ax.axvline(0.0, color="0.3", ls="--", lw=0.9, zorder=3)

        ax.set_yticks([n_show * 0.5, total_h - block_h + n_show * 0.5])
        ax.set_yticklabels(["1", str(n_units)])
        ax.tick_params(axis="y", length=0)
        ax.set_xticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(lab, fontweight="bold", pad=8)

        strip_pad = total_h * 0.05
        y_bar = total_h + strip_pad
        ax.set_ylim(total_h + strip_pad * 3.2, -total_h * 0.015)

        event_len = min(0.5, t1)
        ax.plot([0.0, event_len], [y_bar, y_bar], color="k", lw=2.5,
                solid_capstyle="butt", clip_on=False)
        ax.text(event_len / 2, y_bar + strip_pad * 0.9, "event", ha="center",
                va="top", fontsize=8.5)

        scale_s = 1.0
        x_bar0 = t0
        ax.plot([x_bar0, x_bar0 + scale_s], [y_bar, y_bar], color="k", lw=2.5,
                solid_capstyle="butt", clip_on=False)
        ax.text(x_bar0 + scale_s / 2, y_bar + strip_pad * 0.9, f"{scale_s:.0f} s",
                ha="center", va="top", fontsize=8.5)

    axes[0][0].set_ylabel("Neurons")
    fig.suptitle(title, y=1.02, fontweight="bold")
    fig.subplots_adjust(top=0.86, bottom=0.08, wspace=0.25)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.show()
    return fig

def neuron_correlation(Y_rate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pairwise neuron-neuron Pearson correlation across (trial,bin) samples."""
    M = np.asarray(Y_rate).transpose(1, 0, 2)
    M = M.reshape(M.shape[0], -1)
    n_units = M.shape[0]
    R = np.corrcoef(M)
    n = M.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        tval = R * np.sqrt((n - 2) / (1 - R ** 2))
        P = 2 * stats.t.sf(np.abs(tval), df=n - 2)
    np.fill_diagonal(P, 0.0)
    return R, P

def plot_correlation_matrices(conditions: dict, prep: dict, alpha: float = 0.05,
                              save_path: Optional[str] = None):
    """Neuron-neuron correlation heatmaps with FDR-corrected significance."""
    set_pub_style()
    labels = list(conditions.keys())
    fig, axes = plt.subplots(1, len(labels) + 1, figsize=(4.0 * (len(labels) + 1), 3.8),
                              squeeze=False)
    offdiag_means = {}
    for ax, lab in zip(axes[0], labels):
        R, P = neuron_correlation(conditions[lab])
        n = R.shape[0]
        im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
        
        upper_tri_indices = np.triu_indices(n, k=1)
        p_vals_upper = P[upper_tri_indices]
        
        reject, pvals_corrected, _, _ = multipletests(p_vals_upper, alpha=alpha, method='fdr_bh')
        
        sig_matrix = np.zeros((n, n), dtype=bool)
        sig_matrix[upper_tri_indices] = reject
        sig_matrix = sig_matrix | sig_matrix.T
        
        sig_i, sig_j = np.where(sig_matrix)
        if len(sig_i) > 0:
            ax.scatter(sig_j, sig_i, marker="*", color="k", s=3.0, alpha=0.7)
            
        offmask = ~np.eye(n, dtype=bool)
        offdiag_means[lab] = float(np.nanmean(np.abs(R[offmask])))
        ax.set_title(f"{lab}\n(mean |off-diag r|={offdiag_means[lab]:.3f})", fontsize=10)
        ax.set_xlabel("unit"); ax.set_ylabel("unit")
        
        ax.set_xticks([0, n - 1])
        ax.set_yticks([0, n - 1])
        ax.set_xticklabels(["1", str(n)])
        ax.set_yticklabels(["1", str(n)])
        
        fig.colorbar(im, ax=ax, shrink=0.7)
        
    axb = axes[0][-1]
    axb.bar(range(len(labels)), [offdiag_means[l] for l in labels],
            color=sns.color_palette("muted", len(labels)))
    axb.set_xticks(range(len(labels))); axb.set_xticklabels(labels, rotation=20, ha="right")
    axb.set_ylabel("mean |off-diagonal r|"); axb.set_title("Co-fluctuation summary", fontsize=10)
    sns.despine(ax=axb)
    
    fig.suptitle("Neuron-neuron correlation (* = FDR < %.2g)" % alpha, y=1.04, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        
    plt.show()
    return fig, offdiag_means

def _psth(Y_rate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Trial-averaged firing rate (PSTH) per neuron."""
    mean = Y_rate.mean(axis=0)
    sem = Y_rate.std(axis=0) / math.sqrt(max(Y_rate.shape[0], 1))
    return mean, sem

def plot_psth_traces(
    conditions: dict,
    prep: dict,
    n_show: int = 4,
    save_path: Optional[str] = None,
    title: str = "Trial-averaged firing rate (PSTH)",
):
    """PSTH overlay for ANY conditions dict."""
    set_pub_style()
    plt.rcParams["figure.autolayout"] = False
    t = prep.get("bin_centers", np.linspace(-1, 2, 100))
    labels = list(conditions.keys())
    n_units = prep.get("n_units", 63)
    show = list(range(min(n_show, n_units)))
    
    psth = {lab: _psth(np.asarray(conditions[lab])) for lab in labels}
    ymax = max(psth[l][0][show].max() for l in labels)

    cols = min(2, len(show))
    rows = math.ceil(len(show) / cols) if len(show) > 0 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.8 * rows),
                              sharey=True, squeeze=False)
    
    ax_flat = axes.flatten()
    for ax, u in zip(ax_flat, show):
        for lab in labels:
            mean, sem = psth[lab]
            lw = 2.2 if lab == "real" else 1.4
            ax.plot(t, mean[u], color=_COND_HUE.get(lab, "#555"), lw=lw, label=lab)
            ax.fill_between(t, mean[u] - sem[u], mean[u] + sem[u],
                            color=_COND_HUE.get(lab, "#555"), alpha=0.15)
        ax.axvline(0.0, color="0.4", ls="--", lw=0.8)
        ax.set_title(f"unit {u}", fontsize=10)
        ax.set_xlabel("time (s)")
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        sns.despine(ax=ax, left=True)

    # Hide unused axes
    for ax in ax_flat[len(show):]:
        ax.set_visible(False)

    ax0 = ax_flat[0]
    step = max(1.0, round(ymax / 3))
    x0 = t[0]
    ax0.plot([x0, x0], [0, step], color="k", lw=2.5, clip_on=False)
    ax0.text(x0 - (t[-1] - t[0]) * 0.02, step / 2, f"{step:.0f} Hz",
             rotation=90, ha="right", va="center", fontsize=8)
    
    if len(show) > 0:
        ax_flat[len(show) - 1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(title, y=1.03, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    return fig

def plot_population_psth(
    conditions: dict,
    prep: dict,
    save_path: Optional[str] = None,
    title: str = "Population-averaged PSTH",
):
    """Population-average PSTH for ANY conditions dict."""
    set_pub_style()
    plt.rcParams["figure.autolayout"] = False
    t = prep.get("bin_centers", np.linspace(-1, 2, 100))
    labels = list(conditions.keys())
    
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    
    pop_psth = {}
    for lab in labels:
        Y = np.asarray(conditions[lab])
        unit_mean, _ = _psth(Y)
        mean = unit_mean.mean(axis=0)
        sem = unit_mean.std(axis=0) / math.sqrt(max(unit_mean.shape[0], 1))
        pop_psth[lab] = (mean, sem)
        
    ymax = max(pop_psth[l][0].max() for l in labels)
    
    for lab in labels:
        mean, sem = pop_psth[lab]
        lw = 2.2 if lab == "real" else 1.4
        ax.plot(t, mean, color=_COND_HUE.get(lab, "#555"), lw=lw, label=lab)
        ax.fill_between(t, mean - sem, mean + sem,
                        color=_COND_HUE.get(lab, "#555"), alpha=0.15)
                        
    ax.axvline(0.0, color="0.4", ls="--", lw=0.8)
    ax.set_title("Population Average", fontsize=10)
    ax.set_xlabel("time (s)")
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    sns.despine(ax=ax, left=True)

    step = max(1.0, round(ymax / 3))
    x0 = t[0]
    ax.plot([x0, x0], [0, step], color="k", lw=2.5, clip_on=False)
    ax.text(x0 - (t[-1] - t[0]) * 0.02, step / 2, f"{step:.0f} Hz",
            rotation=90, ha="right", va="center", fontsize=8)

    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(title, y=1.03, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    return fig

def plot_glm_cv_heatmap(lambda_hat, save_path=None):
    """Heatmap of the coefficient of variation (CV) of lambda_hat."""
    mu = lambda_hat.mean(axis=0)
    std = lambda_hat.std(axis=0)
    cv = np.zeros_like(mu)
    mask = mu > 0
    cv[mask] = std[mask] / mu[mask]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(cv, aspect='auto', cmap='hot_r', vmin=0)
    fig.colorbar(im, ax=ax, label='CV of lambda_hat across trials')
    ax.set_xlabel('Time bin', fontweight='bold')
    ax.set_ylabel('Neuron',   fontweight='bold')
    ax.set_title('GLM modulation depth: CV(lambda_hat) per (unit, bin)', pad=10)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    return fig

def plot_step_boxplot(df_step: pd.DataFrame, step_name: str = 'Step', metric: str = 'pearson', save_path: str = None, show: bool = False):
    """
    Boxplot + stripplot of performance per model with a Broken Y-Axis if needed.
    """
    set_pub_style()
    preferred = ['Zero / Identity', 'Baseline / Identity', 'Conditional / Identity', 'Conditional / Shared', 'Baseline / Shared']
    models = [ml for ml in preferred if ml in df_step['model_label'].unique()]
    for m in df_step['model_label'].unique():
        if m not in models:
            models.append(m)
    palette_dict = {ml: _MODEL_COLORS.get(ml, '#333333') for ml in models}
    
    # 1. Determine if we need a broken axis by checking the "real" models
    real_models = df_step[~df_step['model_label'].str.contains('Zero', case=False, na=False)]
    
    if not real_models.empty:
        y_min = real_models[metric].min()
        y_max = real_models[metric].max()
        margin = (y_max - y_min) * 0.2
        top_min = y_min - margin
        top_max = y_max + margin
    else:
        top_min, top_max = 0, 1
        
    metric_display = metric.upper() if metric in ['nll', 'mse', 'r2'] else metric.capitalize()
    
    # 2. If there is a massive gap between 0 and the real models, split the axis!
    if top_min > 0.05 and metric in ['pearson', 'r2']:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 5.5), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot the exact same thing on BOTH axes
        for ax in [ax1, ax2]:
            sns.boxplot(data=df_step, x='model_label', y=metric, hue='model_label', order=models,
                        palette=palette_dict, width=0.4, boxprops={'alpha': 0.5}, ax=ax,
                        showfliers=False, legend=False)
            sns.stripplot(data=df_step, x='model_label', y=metric, hue='model_label', order=models,
                          palette=palette_dict, size=7, alpha=0.8, jitter=True, edgecolor='white',
                          linewidth=0.6, ax=ax, legend=False)
            if metric in ['pearson', 'r2']:
                ax.axhline(0, color='k', ls='--', lw=1.0, alpha=0.5)
                
        # Zoom ax1 (top) to the real models, and ax2 (bottom) to the zero baseline
        ax1.set_ylim(top_min, top_max)
        ax2.set_ylim(-0.02, 0.02)
        
        # Hide the spines between ax1 and ax2 to make it look like one plot
        sns.despine(ax=ax1, bottom=True)
        sns.despine(ax=ax2, top=True)
        ax1.tick_params(labeltop=False, bottom=False)
        ax2.xaxis.tick_bottom()
        
        # Add the diagonal cut marks (//) to the axis
        d = .015 
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        
        # Formatting
        ax1.set_xlabel('')
        ax2.set_xlabel('')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([ml.replace(' / ', '\n/ ') for ml in models])
        
        # Shared Y-label and Title
        fig.text(0.02, 0.5, f'{step_name} {metric_display}', va='center', rotation='vertical')
        ax1.set_ylabel('')
        ax2.set_ylabel('')
        ax1.set_title(f'{step_name} Performance Summary ({metric_display})', fontweight='bold', pad=12)
        
    else:
        # Standard plot (no break needed)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(data=df_step, x='model_label', y=metric, hue='model_label', order=models,
                    palette=palette_dict, width=0.4, boxprops={'alpha': 0.5}, ax=ax,
                    showfliers=False, legend=False)
        sns.stripplot(data=df_step, x='model_label', y=metric, hue='model_label', order=models,
                      palette=palette_dict, size=7, alpha=0.8, jitter=True, edgecolor='white',
                      linewidth=0.6, ax=ax, legend=False)
        if metric in ['pearson', 'r2']:
            ax.axhline(0, color='k', ls='--', lw=1.0, alpha=0.5)
        ax.set_xlabel('')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([ml.replace(' / ', '\n/ ') for ml in models])
        ax.set_ylabel(f'{step_name} {metric_display}')
        ax.set_title(f'{step_name} Performance Summary ({metric_display})', fontweight='bold', pad=12)
        sns.despine(ax=ax)

    # Adjust layout
    plt.tight_layout()
    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def plot_statistics_heatmap(
    df,
    filter_col: str,
    filter_val: str,
    compare_col: str = 'model_label',
    metric: str = 'pearson',
    unit_col: str = 'unit',
    fold_col: str = 'fold',
    alpha: float = 0.05,
    correction: str = 'fdr_bh',      # 'bonferroni' | 'fdr_bh' | None
    magnitude_threshold: float = 'auto', # 'auto' | float | None
    save_path: str = None,
):
    """
    Pairwise statistical comparison between models with exact unit pairing,
    directional effect size (Cliff's delta), and practical magnitude thresholding.
    """
    set_pub_style()

    sub = df[df[filter_col] == filter_val].copy()
    if sub.empty:
        print(f"plot_statistics_heatmap: no data for {filter_col}={filter_val!r}")
        return None

    # Aggregate to unit level (mean across folds per unit & model)
    group_cols = [compare_col, unit_col]
    if fold_col in sub.columns:
        agg = sub.groupby(group_cols + [fold_col], as_index=False)[metric].mean()
        unit_agg = agg.groupby(group_cols, as_index=False)[metric].mean()
    else:
        unit_agg = sub.groupby(group_cols, as_index=False)[metric].mean()

    # Pivot to ensure exact neuron-to-neuron alignment
    pivot = unit_agg.pivot(index=unit_col, columns=compare_col, values=metric)
    preferred = ['Zero / Identity', 'Baseline / Identity',
                 'Conditional / Identity', 'Conditional / Shared', 'Baseline / Shared']
    all_items = list(pivot.columns)
    items = [m for m in preferred if m in all_items] + [m for m in all_items if m not in preferred]
    n = len(items)

    if n < 2:
        print("plot_statistics_heatmap: need at least 2 models to compare.")
        return None

    higher_is_better = {'pearson': True, 'r2': True, 'mse': False, 'nll': False}.get(metric, True)

    if magnitude_threshold == 'auto':
        THRESHOLDS = {'pearson': 0.015, 'r2': 0.01, 'mse': 0.01, 'nll': 0.01}
        thresh_val = THRESHOLDS.get(metric, 0.01)
    else:
        thresh_val = magnitude_threshold

    raw_p   = np.full((n, n), np.nan)
    eff     = np.zeros((n, n))
    n_obs   = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i == j:
                eff[i, j] = 0.0
                raw_p[i, j] = np.nan
                continue

            mod_a, mod_b = items[i], items[j]
            pair_df = pivot[[mod_a, mod_b]].dropna()
            a = pair_df[mod_a].values
            b = pair_df[mod_b].values
            min_len = len(a)
            n_obs[i, j] = min_len

            if min_len < 5:
                continue

            delta_mean = np.mean(a) - np.mean(b)

            # Magnitude thresholding for practical equivalence
            if thresh_val is not None and abs(delta_mean) < thresh_val:
                eff[i, j] = 0.0
                if i > j:
                    raw_p[i, j] = np.nan
                continue

            # Directional Cliff's delta
            eff[i, j] = delta_mean

            if i > j:  # lower triangle Wilcoxon
                diffs = a - b
                if np.all(diffs == 0):
                    p = 1.0
                else:
                    try:
                        _, p = stats.wilcoxon(a, b, alternative='two-sided')
                    except Exception:
                        p = np.nan
                raw_p[i, j] = p

    # FDR correction on lower triangle p-values
    lower_mask = np.tril(np.ones((n, n), bool), k=-1)
    raw_vec = raw_p[lower_mask]
    valid   = ~np.isnan(raw_vec)

    adj_p_full = np.full((n, n), np.nan)
    if correction and valid.any():
        _, adj_vec, _, _ = multipletests(raw_vec[valid], alpha=alpha, method=correction)
        tmp = raw_vec.copy()
        tmp[valid] = adj_vec
        adj_p_full[lower_mask] = tmp
    else:
        adj_p_full[lower_mask] = raw_vec

    # Mirror adjusted p-values to upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            adj_p_full[i, j] = adj_p_full[j, i]

    # Plotting
    # The figure is kept compact (about 9 x 4.5 in for 5 models) so that the
    # labels stay legible when it is scaled down onto a slide; resolution
    # comes from the save dpi, not from a large canvas.
    tick_labels = [str(it).replace(' / ', '\n/ ') for it in items]
    metric_name = {'pearson': 'Pearson r', 'r2': r'$R^2$', 'mse': 'MSE', 'nll': 'NLL'}.get(metric, metric)
    cell_sz     = 0.75 if n <= 6 else max(0.45, 4.5 / n)
    panel_sz    = cell_sz * n
    fs_cell     = 13 if n <= 6 else max(7.0, 66.0 / n)
    fs_tick, fs_label, fs_title, fs_sub = 9.5, 11.5, 12.5, 9.5

    fig, axes = plt.subplots(
        1, 2, figsize=(2 * panel_sz + 2.0, panel_sz + 1.25), layout="constrained"
    )

    def _style_axis(ax, show_ylabel):
        ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels, fontsize=fs_tick)
        # row labels only on the left panel: both panels share the same order
        ax.set_yticks(range(n))
        ax.set_yticklabels(tick_labels if show_ylabel else [], fontsize=fs_tick)
        ax.tick_params(length=0, pad=3)
        ax.set_xlabel("Model B (column)", fontsize=fs_label)
        if show_ylabel:
            ax.set_ylabel("Model A (row)", fontsize=fs_label)
        # thin white grid between cells
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="minor", length=0)

    def _panel_title(ax, title, subtitle):
        ax.set_title(subtitle, fontsize=fs_sub, color="#374151", pad=5)
        # bold headline sits just above the (one- or two-line) subtitle
        n_sub = subtitle.count("\n") + 1
        ax.annotate(title, xy=(0.5, 1.0), xycoords="axes fraction",
                    xytext=(0, n_sub * fs_sub * 1.3 + 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fs_title, fontweight="bold")

    # Panel 0: Significance Matrix (Red = Significant via RdYlGn)
    disp_comb = np.where(np.eye(n, dtype=bool), np.nan, raw_p)
    for i in range(n):
        for j in range(i + 1, n):
            disp_comb[i, j] = adj_p_full[i, j]

    im0 = axes[0].imshow(disp_comb, cmap="RdYlGn", vmin=0, vmax=alpha * 2, aspect="equal")

    for i in range(n):
        for j in range(n):
            if i == j:
                axes[0].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#E5E7EB", zorder=2))
                axes[0].text(j, i, "Ref", ha="center", va="center", fontsize=fs_cell,
                             color="black", zorder=3, fontweight="bold")
            elif np.isnan(disp_comb[i, j]):
                axes[0].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#F3F4F6", zorder=2))
                axes[0].text(j, i, "n.d.", ha="center", va="center", fontsize=fs_cell - 1,
                             color="#6B7280", zorder=3)
            else:
                p_val = disp_comb[i, j]
                stars = p_to_stars(p_val)
                p_txt = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
                col = "white" if (p_val < alpha / 2 or p_val > alpha * 1.6) else "black"
                axes[0].text(j, i, f"{p_txt}\n{stars}", ha="center", va="center",
                             fontsize=fs_cell, color=col, zorder=3, linespacing=0.95)

    _style_axis(axes[0], show_ylabel=True)
    thr_txt = "" if thresh_val is None else rf"   n.d.: $|\Delta|$ < {thresh_val}"
    _panel_title(
        axes[0],
        f"Significance: {metric_name} ({filter_val})",
        f"lower: raw Wilcoxon p   |   upper: {correction or 'uncorrected'} adjusted\n"
        f"* p<0.05   ** p<0.01   *** p<0.001{thr_txt}",
    )
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.85, pad=0.02)
    cb0.set_label("p-value (red = significant)", fontsize=fs_label - 1)
    cb0.ax.tick_params(labelsize=fs_tick - 1)
    cb0.ax.axhline(alpha, color="red", lw=1.8, ls="--")

    # Panel 1: paired mean difference (row - column).
    # Numbers are the raw difference; colour encodes who is better
    # (blue = row model better), so it reads the same for error metrics
    # (MSE, NLL: lower is better) and score metrics (Pearson, R2).
    sign    = 1.0 if higher_is_better else -1.0
    eff_max = float(np.nanmax(np.abs(eff))) if np.isfinite(eff).any() else 0.0
    vlim    = eff_max if eff_max > 0 else 1.0
    im1 = axes[1].imshow(sign * eff, cmap="RdBu", vmin=-vlim, vmax=vlim, aspect="equal")
    for i in range(n):
        for j in range(n):
            if i == j:
                axes[1].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#E5E7EB", zorder=2))
                axes[1].text(j, i, "Ref", ha="center", va="center", fontsize=fs_cell,
                             color="black", zorder=3, fontweight="bold")
            else:
                v = eff[i, j]
                txt = "0" if v == 0 else (f"{v:+.2f}" if abs(v) >= 0.095 else f"{v:+.3f}")
                col = "white" if abs(v) > 0.6 * vlim else "black"
                axes[1].text(j, i, txt, ha="center", va="center", fontsize=fs_cell, color=col, zorder=3)

    _style_axis(axes[1], show_ylabel=False)
    better = "higher" if higher_is_better else "lower"
    _panel_title(
        axes[1],
        rf"Effect size: $\Delta$ {metric_name} (row $-$ column)",
        f"paired mean difference (n = {int(n_obs.max())} pairs per comparison)\n{better} {metric_name} is better   |   blue = row model better",
    )
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.85, pad=0.02)
    cb1.set_label("advantage of row model", fontsize=fs_label - 1)
    cb1.ax.tick_params(labelsize=fs_tick - 1)

    for ax in axes:
        for sp in ax.spines.values():
            sp.set_visible(False)

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved -> {save_path}")
    plt.close(fig)
    return fig


def plot_unit_spikes(
    spikes,
    unit_idx,
    title,
    file_name,
    file_path,
    bin_times,
    bin_size,
    figsize=(4, 6),
    show=False,
):
    unit_spikes = spikes[:, unit_idx, :] / bin_size

    mean_by_bin = np.nanmean(unit_spikes, axis=0)
    sem_by_bin = np.nanstd(unit_spikes, axis=0) / np.sqrt(unit_spikes.shape[0])
    ci_low = mean_by_bin - 1.96 * sem_by_bin
    ci_high = mean_by_bin + 1.96 * sem_by_bin

    xticks = [
        bin_times[0],
        0,
        bin_times[-1],
    ]

    fig = plt.figure(figsize=figsize, layout="constrained")

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[4, 1],
        hspace=0.05,
    )

    ax_heatmap = fig.add_subplot(gs[0])

    ax_psth = fig.add_subplot(
        gs[1],
        sharex=ax_heatmap,
    )

    im = ax_heatmap.imshow(
        unit_spikes,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap='Greys',
        extent=[
            bin_times[0],
            bin_times[-1],
            0,
            unit_spikes.shape[0],
        ],
    )

    ax_heatmap.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax_heatmap.set_ylabel(
        "Trial",
    )

    ax_heatmap.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )

    ax_heatmap.spines[["top", "right"]].set_visible(False)

    ax_psth.fill_between(
        bin_times,
        ci_low,
        ci_high,
        color="black",
        alpha=0.25,
    )

    ax_psth.plot(
        bin_times,
        mean_by_bin,
        color="black",
        linewidth=2,
    )

    ax_psth.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax_psth.set(
        xlabel="Time from press onset (s)",
        ylabel="Mean\nspikes per s",
        xticks=xticks,
        xticklabels=[
            f"{bin_times[0]:g}",
            "0",
            f"{bin_times[-1]:g}",
        ],
    )

    ax_psth.spines[["top", "right"]].set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=ax_heatmap,
        fraction=0.025,
        pad=0.02,
    )

    cbar.set_label(
        "Spikes per s",
        fontsize=8,
    )

    cbar.ax.tick_params(
        labelsize=7,
    )

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(
        fig,
        file_name,
        file_path,
    )


def plot_spike_summary(
    spikes,
    file_name,
    file_path,
    title=None,
    metric="mean",
    k=4.0,
    unit_threshold=None,
    trial_threshold=None,
    unit_names=None,
    trial_names=None,
    cmap="viridis",
    figsize=(8, 5),
    show=False,
):
    to_numpy = lambda x: (
        x.detach().cpu().numpy()
        if torch.is_tensor(x)
        else np.asarray(x)
    )

    metric_functions = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "max": np.nanmax,
        "min": np.nanmin,
        "std": np.nanstd,
        "var": np.nanvar,
        "sum": np.nansum,
    }

    if metric not in [*metric_functions.keys(), "outlier_rate"]:
        raise ValueError(
            "metric must be one of: "
            f"{[*metric_functions.keys(), 'outlier_rate']}."
        )

    def parse_threshold(threshold):
        if threshold is None:
            return None, None

        if np.isscalar(threshold):
            return None, float(threshold)

        threshold = tuple(threshold)

        if len(threshold) != 2:
            raise ValueError(
                "threshold must be None, a scalar, or (low, high)."
            )

        low, high = threshold

        low = None if low is None else float(low)
        high = None if high is None else float(high)

        return low, high

    spikes = to_numpy(spikes).astype(float)

    if spikes.ndim != 3:
        raise ValueError(
            "spikes must have shape (n_trials, n_units, n_bins). "
            f"Received shape {spikes.shape}."
        )

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    n_trials, n_units, n_bins = spikes.shape

    if unit_names is None:
        unit_names = np.arange(n_units)

    if trial_names is None:
        trial_names = np.arange(n_trials)

    unit_names = np.asarray(unit_names)
    trial_names = np.asarray(trial_names)

    if unit_names.size != n_units:
        raise ValueError("unit_names must contain one name per unit.")

    if trial_names.size != n_trials:
        raise ValueError("trial_names must contain one name per trial.")

    if metric == "outlier_rate":
        median = np.nanmedian(
            spikes,
            axis=0,
            keepdims=True,
        )

        mad = np.nanmedian(
            np.abs(spikes - median),
            axis=0,
            keepdims=True,
        )

        robust_scale = np.maximum(
            1.4826 * mad,
            1.0,
        )

        flagged = spikes > median + k * robust_scale

        summary = np.nanmean(
            flagged,
            axis=2,
        )

        unit_score = np.nanmean(
            flagged,
            axis=(0, 2),
        )

        trial_score = np.nanmean(
            flagged,
            axis=(1, 2),
        )

        metric_name = f"Outlier rate"
        colorbar_label = "Proportion of flagged bins"

    else:
        metric_function = metric_functions[metric]

        summary = metric_function(
            spikes,
            axis=2,
        )

        unit_score = metric_function(
            spikes,
            axis=(0, 2),
        )

        trial_score = metric_function(
            spikes,
            axis=(1, 2),
        )

        metric_name = metric.title()
        colorbar_label = f"{metric.title()} spike value"

    unit_low, unit_high = parse_threshold(
        unit_threshold,
    )

    trial_low, trial_high = parse_threshold(
        trial_threshold,
    )

    bad_units = np.zeros(
        n_units,
        dtype=bool,
    )

    bad_trials = np.zeros(
        n_trials,
        dtype=bool,
    )

    if unit_low is not None:
        bad_units |= unit_score < unit_low

    if unit_high is not None:
        bad_units |= unit_score > unit_high

    if trial_low is not None:
        bad_trials |= trial_score < trial_low

    if trial_high is not None:
        bad_trials |= trial_score > trial_high

    units_to_remove = np.where(
        bad_units,
    )[0]

    trials_to_remove = np.where(
        bad_trials,
    )[0]

    unit_tick_idx = np.linspace(
        0,
        n_units - 1,
        min(10, n_units),
        dtype=int,
    )

    trial_tick_idx = np.linspace(
        0,
        n_trials - 1,
        min(10, n_trials),
        dtype=int,
    )

    fig = plt.figure(
        figsize=figsize,
        layout="constrained",
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.25, 1],
        height_ratios=[1, 0.25],
        wspace=0.03,
        hspace=0.03,
    )

    ax_main = fig.add_subplot(gs[0, 1])

    ax_left = fig.add_subplot(
        gs[0, 0],
        sharey=ax_main,
    )

    ax_bottom = fig.add_subplot(
        gs[1, 1],
        sharex=ax_main,
    )

    ax_corner = fig.add_subplot(gs[1, 0])
    ax_corner.axis("off")

    im = ax_main.imshow(
        summary.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
    )

    ax_main.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )

    ax_main.tick_params(
        axis="y",
        left=False,
        labelleft=False,
    )

    ax_main.spines[["top", "right"]].set_visible(False)

    unit_idx = np.arange(n_units)
    trial_idx = np.arange(n_trials)

    ax_left.scatter(
        unit_score,
        unit_idx,
        color="C0",
        s=8,
        alpha=0.7,
    )

    if unit_low is not None:
        ax_left.axvline(
            unit_low,
            color="tab:red",
            linewidth=1,
        )

    if unit_high is not None:
        ax_left.axvline(
            unit_high,
            color="tab:red",
            linewidth=1,
        )

    ax_left.scatter(
        unit_score[bad_units],
        unit_idx[bad_units],
        color="tab:red",
        s=18,
        zorder=3,
    )

    ax_left.set(
        xlabel=metric_name,
        ylabel="Unit",
        yticks=unit_tick_idx,
        yticklabels=unit_names[unit_tick_idx],
    )

    ax_left.tick_params(
        axis="x",
        labelsize=7,
    )

    ax_left.tick_params(
        axis="y",
        labelsize=7,
    )

    ax_left.spines[["top", "right"]].set_visible(False)

    ax_bottom.scatter(
        trial_idx,
        trial_score,
        color="C0",
        s=8,
        alpha=0.7,
    )

    if trial_low is not None:
        ax_bottom.axhline(
            trial_low,
            color="tab:red",
            linewidth=1,
        )

    if trial_high is not None:
        ax_bottom.axhline(
            trial_high,
            color="tab:red",
            linewidth=1,
        )

    ax_bottom.scatter(
        trial_idx[bad_trials],
        trial_score[bad_trials],
        color="tab:red",
        s=18,
        zorder=3,
    )

    ax_bottom.set(
        xlabel="Trial",
        ylabel=metric_name,
        xticks=trial_tick_idx,
        xticklabels=trial_names[trial_tick_idx],
    )

    ax_bottom.tick_params(
        axis="x",
        labelrotation=90,
        labelsize=7,
    )

    ax_bottom.tick_params(
        axis="y",
        labelsize=7,
    )

    ax_bottom.spines[["top", "right"]].set_visible(False)

    for unit_idx in units_to_remove:
        ax_main.axhline(
            unit_idx - 0.5,
            color="tab:red",
            linewidth=0.6,
        )

        ax_main.axhline(
            unit_idx + 0.5,
            color="tab:red",
            linewidth=0.6,
        )

    for trial_idx in trials_to_remove:
        ax_main.axvline(
            trial_idx - 0.5,
            color="tab:red",
            linewidth=0.6,
        )

        ax_main.axvline(
            trial_idx + 0.5,
            color="tab:red",
            linewidth=0.6,
        )

    cbar = fig.colorbar(
        im,
        ax=ax_main,
        fraction=0.025,
        pad=0.02,
    )

    cbar.set_label(
        colorbar_label,
        fontsize=8,
    )

    cbar.ax.tick_params(
        labelsize=7,
    )

    fig.suptitle(title)

    if show:
        plt.show()

    save_figure(
        fig,
        file_name,
        file_path,
    )

    return trials_to_remove, units_to_remove


def plot_training_history(
    trainer,
    title,
    file_name,
    file_path,
    show=False,
):

    h = {}
    if hasattr(trainer, 'history'):
        if hasattr(trainer.history, 'history'):
            h = trainer.history.history
        elif isinstance(trainer.history, dict):
            h = trainer.history
    elif hasattr(trainer, 'callbacks') and len(trainer.callbacks) > 0:
        cb = trainer.callbacks[0]
        if hasattr(cb, 'history') and isinstance(cb.history, dict):
            h = cb.history
        else:
            h = {
                'train_loss_epoch': getattr(cb, 'train_loss', []),
                'valid_loss_epoch': getattr(cb, 'valid_loss', []),
                'learning_rate': getattr(cb, 'lr', []),
            }

    train = np.asarray(h.get('train_loss_epoch', h.get('train_loss', [])))
    valid = np.asarray(h.get('valid_loss_epoch', h.get('valid_loss', [])))
    lr = np.asarray(h.get('learning_rate', h.get('lr', [])))

    if len(valid) == 0 and len(train) == 0:
        print(f"Skipping plot '{title}': no training history found.")
        return None

    trend = lambda x: np.diff(x) / np.maximum(np.abs(x[:-1]), 1e-8) if len(x) > 1 else np.array([])

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")

    # 1. Loss plot
    if len(valid) > 0:
        ax[0].plot(np.arange(1, len(valid) + 1), valid, color="tab:orange", label="Validation")
    if len(train) > 0:
        ax[0].plot(np.arange(1, len(train) + 1), train, color="tab:blue", label="Train")
    ax[0].set(title="Loss", xlabel="Epoch", ylabel="NLL loss")
    if len(valid) > 0 or len(train) > 0:
        ax[0].legend(frameon=False)

    # 2. Loss trend plot
    ax[1].axhline(0, color="black", ls="--", lw=1)
    if len(valid) > 1:
        ax[1].plot(np.arange(2, len(valid) + 1), trend(valid), color="tab:orange", label="Validation")
    if len(train) > 1:
        ax[1].plot(np.arange(2, len(train) + 1), trend(train), color="tab:blue", label="Train")
    ax[1].set(title="Loss trend", xlabel="Epoch", ylabel="Relative change")
    if len(valid) > 1 or len(train) > 1:
        ax[1].legend(frameon=False)

    # 3. Learning rate plot
    valid_lr = len(lr) > 0 and not np.all(np.isnan(lr))
    if valid_lr:
        ax[2].plot(np.arange(1, len(lr) + 1), lr, color="tab:green")
        ax[2].set_yscale("log")
    ax[2].set(title="Learning rate", xlabel="Epoch", ylabel="LR")

    fig.text(
        -0.05,
        0.5,
        title,
        rotation="vertical",
        va="center",
        ha="left",
        fontsize=12,
    )

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)


def get_cov_loading_matrices(cov_model):
    if hasattr(cov_model, "lambda_matrix"):
        return {
            "shared": cov_model.lambda_matrix
        }

    if hasattr(cov_model, "component_loadings"):
        return {
            component_name: cov_model.full_loading_matrix(component_name)
            for component_name in cov_model.component_loadings
        }

    raise TypeError(
        f"Unsupported covariance model: {type(cov_model).__name__}."
    )

def get_cov_length_scales(cov_model):
    if hasattr(cov_model, "length_scales"):
        return {
            "shared": F.softplus(cov_model.length_scales)
        }

    if hasattr(cov_model, "component_length_scales"):
        return {
            component_name: F.softplus(length_scales)
            for component_name, length_scales
            in cov_model.component_length_scales.items()
        }

    raise TypeError(
        f"Unsupported covariance model: {type(cov_model).__name__}."
    )

def get_covariance_matrix(cov_model):
    if hasattr(cov_model, "covariance_matrix"):
        covariance = cov_model.covariance_matrix()

    elif hasattr(cov_model, "build_component_covariance"):
        covariance = cov_model.build_component_covariance(
            cov_model.lambda_matrix,
            cov_model.length_scales,
        )

    elif hasattr(cov_model, "build_covariance_matrix"):
        lambda_matrix = cov_model.lambda_matrix.unsqueeze(0)

        cov_tensor = cov_model.build_covariance_matrix(
            lambda_matrix,
        )

        covariance = (
            cov_tensor @ cov_tensor.transpose(-1, -2)
        ).squeeze(0)

        return covariance

    else:
        raise TypeError(
            f"Unsupported covariance model: "
            f"{type(cov_model).__name__}."
        )

    covariance = covariance + cov_model.I

    noise_diag = torch.sigmoid(cov_model.noise.flatten())
    covariance.diagonal().add_(noise_diag)

    return covariance

def plot_cov_loading_matrix(
    mean_cov_lit_model,
    title,
    file_name,
    file_path,
    bin_times=None,
    vmax=None,
    show=False,
):
    cov_model = mean_cov_lit_model.full_model.cov_model
    loading_matrices = get_cov_loading_matrices(cov_model)

    if bin_times is None:
        bin_times = np.arange(cov_model.n_bins)

    bin_times = np.asarray(bin_times)

    loading_matrix = np.concatenate(
        [
            lambda_matrix.detach().cpu().numpy().transpose(2, 0, 1).reshape(
                lambda_matrix.size(-1),
                -1,
            )
            for lambda_matrix in loading_matrices.values()
        ],
        axis=0,
    )

    component_names = [
        component_name
        for component_name, lambda_matrix in loading_matrices.items()
        for _ in range(lambda_matrix.size(-1))
    ]

    component_boundaries = np.cumsum(
        [
            lambda_matrix.size(-1)
            for lambda_matrix in loading_matrices.values()
        ]
    )[:-1]

    if vmax is None:
        vmax = np.nanmax(np.abs(loading_matrix))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    vmin = -vmax

    fig, ax = plt.subplots(
        figsize=(
            max(4, cov_model.n_bins * 0.4),
            max(2, loading_matrix.shape[0] * 0.175),
        ),
        layout="constrained",
    )

    im = ax.imshow(
        loading_matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    for bin_idx in range(1, cov_model.n_bins):
        ax.axvline(
            bin_idx * cov_model.n_units - 0.5,
            color="black",
            linewidth=0.7,
            alpha=0.7,
        )

    for boundary in component_boundaries:
        ax.axhline(
            boundary - 0.5,
            color="black",
            linewidth=1.0,
        )

    bin_centers = (
        np.arange(cov_model.n_bins) * cov_model.n_units
        + (cov_model.n_units - 1) / 2
    )

    ax.set(
        xlabel="Time bin",
        ylabel="Latent",
        xticks=bin_centers,
        xticklabels=np.round(bin_times, 2),
        yticks=np.arange(loading_matrix.shape[0]),
        yticklabels=component_names,
    )

    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.015,
        pad=0.01,
    )
    cbar.set_label(r"$\Lambda$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)


def plot_cov_noise(
    mean_cov_lit_model,
    title,
    file_name,
    file_path,
    bin_times=None,
    unit_names=None,
    vmax=None,
    show=False,
):
    noise = mean_cov_lit_model.full_model.cov_model.noise
    noise = torch.sigmoid(noise).detach().cpu().numpy()

    n_bins, n_units = noise.shape

    if bin_times is None:
        bin_times = np.arange(n_bins)

    if unit_names is None:
        unit_names = np.arange(n_units)

    if vmax is None:
        vmax = np.max(noise)

    xticks = [
        0,
        np.where(bin_times == 0)[0][0],
        n_bins - 1,
    ]

    fig, ax = plt.subplots(
        figsize=(5, 4),
        layout="constrained",
    )

    im = ax.imshow(
        noise.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="Reds",
        vmin=0,
        vmax=vmax,
    )

    ax.axvline(
        np.where(bin_times == 0)[0][0],
        color="black",
        linestyle="--",
        linewidth=1,
    )

    max_yticks = 12

    ytick_idx = np.unique(
        np.linspace(
            0,
            n_units - 1,
            min(max_yticks, n_units),
            dtype=int,
        )
    )

    ax.set(
        xlabel="Time",
        ylabel="Unit",
        xticks=xticks,
        xticklabels=[
            f"{bin_times[0]:g}",
            "0",
            f"{bin_times[-1]:g}",
        ],
        yticks=ytick_idx,
        yticklabels=np.asarray(unit_names)[ytick_idx],
    )

    ax.tick_params(
        axis="x",
        labelsize=7,
    )

    ax.tick_params(
        axis="y",
        labelsize=7,
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )

    cbar.set_label(
        "Noise variance",
        fontsize=8,
    )

    cbar.ax.tick_params(
        labelsize=7,
    )

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(
        fig,
        file_name,
        file_path,
    )


def plot_cov_length_scales(
    mean_cov_lit_model,
    title,
    file_name,
    file_path,
    ymax=None,
    show=False,
):
    cov_model = mean_cov_lit_model.full_model.cov_model
    length_scales_dict = get_cov_length_scales(cov_model)

    length_scales = np.concatenate(
        [
            length_scale.detach().cpu().numpy()
            for length_scale in length_scales_dict.values()
        ]
    )

    component_names = [
        component_name
        for component_name, length_scale in length_scales_dict.items()
        for _ in range(length_scale.numel())
    ]

    component_boundaries = np.cumsum(
        [
            length_scale.numel()
            for length_scale in length_scales_dict.values()
        ]
    )[:-1]

    if ymax is None:
        ymax = np.max(length_scales)

    fig, ax = plt.subplots(
        figsize=(
            max(4, len(length_scales) * 0.225),
            3.5,
        ),
        layout="constrained",
    )

    latent_idx = np.arange(len(length_scales))

    ax.bar(
        latent_idx,
        length_scales,
        color="C0",
    )

    for boundary in component_boundaries:
        ax.axvline(
            boundary - 0.5,
            color="black",
            linewidth=1.0,
        )

    ax.set(
        xlabel="Latent component",
        ylabel="Length scale",
        xticks=latent_idx,
        xticklabels=component_names,
        ylim=(0, 1.05 * ymax),
    )

    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)


def plot_covariance_matrix(
    mean_cov_lit_model,
    title,
    file_name,
    file_path,
    bin_times=None,
    time_indices=None,
    vmax=None,
    linthresh=None,
    show=False,
):
    cov_model = mean_cov_lit_model.full_model.cov_model

    covariance_matrix = get_covariance_matrix(cov_model)
    covariance_matrix = covariance_matrix.detach().cpu().numpy()

    n_bins = cov_model.n_bins
    n_units = cov_model.n_units

    if bin_times is None:
        bin_times = np.arange(n_bins)

    if time_indices is None:
        time_indices = np.arange(n_bins)
    else:
        time_indices = np.atleast_1d(time_indices)

    feature_indices = np.concatenate(
        [
            np.arange(
                time_idx * n_units,
                (time_idx + 1) * n_units,
            )
            for time_idx in time_indices
        ]
    )

    covariance_matrix = covariance_matrix[
        np.ix_(feature_indices, feature_indices)
    ]

    bin_times = np.asarray(bin_times)[time_indices]
    n_bins = len(time_indices)

    if vmax is None:
        vmax = np.nanmax(np.abs(covariance_matrix))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    vmin = -vmax

    if linthresh is None:
        linthresh = max(vmax * 0.01, 1e-8)

    norm = mcolors.SymLogNorm(
        linthresh=linthresh,
        vmin=vmin,
        vmax=vmax,
        base=10,
    )

    fig, ax = plt.subplots(
        figsize=(
            max(5, n_bins * 0.25),
            max(5, n_bins * 0.25),
        ),
        layout="constrained",
    )

    im = ax.imshow(
        covariance_matrix,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=norm,
    )

    for bin_idx in range(1, n_bins):
        line_pos = bin_idx * n_units - 0.5

        ax.axvline(
            line_pos,
            color="black",
            linewidth=0.5,
            alpha=0.5,
        )

        ax.axhline(
            line_pos,
            color="black",
            linewidth=0.5,
            alpha=0.5,
        )

    bin_centers = (
        np.arange(n_bins) * n_units
        + (n_units - 1) / 2
    )

    tick_labels = np.round(bin_times, 2)

    ax.set(
        xticks=bin_centers,
        xticklabels=tick_labels,
        yticks=bin_centers,
        yticklabels=tick_labels,
    )

    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label(
        "Covariance (SymLog Scale)",
        fontsize=8,
    )
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)


def plot_unit_covariance(
    mean_cov_lit_model,
    unit_idx,
    title,
    file_name,
    file_path,
    bin_times=None,
    unit_names=None,
    cmap="RdBu_r",
    vmax=None,
    ymax=None,
    figsize=(5, 3),
    show=False,
):
    cov_model = mean_cov_lit_model.full_model.cov_model
    covariance = get_covariance_matrix(cov_model)
    covariance = covariance.detach().cpu().numpy()

    n_bins = cov_model.n_bins
    n_units = cov_model.n_units

    if bin_times is None:
        bin_times = np.arange(n_bins)

    if unit_names is None:
        unit_names = np.arange(n_units)

    bin_times = np.asarray(bin_times)
    unit_names = np.asarray(unit_names)

    if unit_idx < 0 or unit_idx >= n_units:
        raise ValueError(
            f"unit_idx must be in [0, {n_units - 1}], got {unit_idx}."
        )

    if bin_times.size != n_bins:
        raise ValueError("bin_times must contain one value per bin.")

    if unit_names.size != n_units:
        raise ValueError("unit_names must contain one name per unit.")

    other_unit_idx = np.delete(np.arange(n_units), unit_idx)

    unit_cov = np.empty((other_unit_idx.size, n_bins), dtype=float)

    for bin_idx in range(n_bins):
        feature_idx = bin_idx * n_units + unit_idx
        block_start = bin_idx * n_units
        block_end = (bin_idx + 1) * n_units

        diag_block_values = covariance[
            feature_idx,
            block_start:block_end,
        ]
        unit_cov[:, bin_idx] = diag_block_values[other_unit_idx]

    mean_by_bin = np.nanmean(unit_cov, axis=0)

    if vmax is None:
        vmax = np.nanmax(np.abs(unit_cov))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    vmin = -vmax

    if ymax is None:
        ymax = np.nanmax(np.abs(mean_by_bin))

    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0

    xticks = [
        bin_times[0],
        0,
        bin_times[-1],
    ]

    fig = plt.figure(
        figsize=figsize,
        layout="constrained",
    )

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[4, 1],
        hspace=0.05,
    )

    ax_heatmap = fig.add_subplot(gs[0])
    ax_mean = fig.add_subplot(
        gs[1],
        sharex=ax_heatmap,
    )

    im = ax_heatmap.imshow(
        unit_cov,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[
            bin_times[0],
            bin_times[-1],
            0,
            unit_cov.shape[0],
        ],
    )

    ax_heatmap.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    max_yticks = 12

    ytick_pos = np.unique(
        np.linspace(
            0,
            unit_cov.shape[0] - 1,
            min(max_yticks, unit_cov.shape[0]),
            dtype=int,
        )
    )

    ax_heatmap.set(
        ylabel="Other unit",
        yticks=ytick_pos + 0.5,
        yticklabels=unit_names[other_unit_idx][ytick_pos],
    )

    ax_heatmap.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )

    ax_heatmap.tick_params(
        axis="y",
        labelsize=7,
    )

    ax_heatmap.spines[["top", "right"]].set_visible(False)

    ax_mean.plot(
        bin_times,
        mean_by_bin,
        color="tab:blue",
        linewidth=2,
    )

    ax_mean.axhline(
        0,
        color="black",
        linestyle=":",
        linewidth=1,
    )

    ax_mean.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax_mean.set(
        xlabel="Time from press onset (s)",
        ylabel="Mean\ncovariance",
        xticks=xticks,
        xticklabels=[
            f"{bin_times[0]:g}",
            "0",
            f"{bin_times[-1]:g}",
        ],
        ylim=(-ymax, ymax),
    )

    ax_mean.spines[["top", "right"]].set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=ax_heatmap,
        fraction=0.025,
        pad=0.02,
    )

    cbar.set_label(
        "Covariance",
        fontsize=8,
    )

    cbar.ax.tick_params(
        labelsize=7,
    )

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(
        fig,
        file_name,
        file_path,
    )


def plot_prediction(
    Y,
    Y_hat,
    trial_idx,
    unit_idx,
    bin_times,
    title,
    filename,
    filepath,
    figsize=(6.5, 2.6),
    show=False,
    target_label="Synthetic data (generated)",
    prediction_label="Model prediction",
    target_color="tab:green",
    prediction_color="tab:purple",
    Y_reference=None,
    reference_label="Real data",
):
    """
    One unit in one trial: the target the model was trained on vs. its prediction.

    ``Y`` is the target returned by ``predict_loader`` -- for the synthetic
    steps this is the data generated by that step (not the recorded spikes),
    so it is labelled as such. ``Y_reference`` optionally adds a third, thin
    grey trace (e.g. the recorded data of the same trial) for context.
    """
    to_numpy = lambda x: (
        x.detach().cpu().numpy()
        if torch.is_tensor(x)
        else np.asarray(x)
    )

    y = to_numpy(Y[trial_idx])[:, unit_idx]
    y_hat = to_numpy(Y_hat[trial_idx])[:, unit_idx]
    bin_times = np.asarray(bin_times)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    if Y_reference is not None:
        ax.plot(
            bin_times,
            to_numpy(Y_reference[trial_idx])[:, unit_idx],
            color="0.6",
            linewidth=1.2,
            label=reference_label,
        )
    ax.plot(
        bin_times,
        y,
        color=target_color,
        linewidth=2,
        marker="o",
        markersize=3,
        label=target_label,
    )
    ax.plot(
        bin_times,
        y_hat,
        color=prediction_color,
        linewidth=2,
        label=prediction_label,
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Time from press onset (s)", fontsize=11)
    ax.set_ylabel("Neural activity", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(frameon=False, fontsize=10, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=12)

    if show:
        plt.show()

    return save_figure(fig, filename, filepath, dpi=200)


def plot_train_valid_metrics_comparison(
    correlations_train,
    correlations_valid,
    r2s_train,
    r2s_valid,
    mses_train,
    mses_valid,
    title,
    filename,
    filepath,
    figsize=(7, 4),
    show=False,
):
    metrics = [
        (
            "Correlation",
            np.asarray(correlations_train, dtype=float),
            np.asarray(correlations_valid, dtype=float),
            "tab:brown",
        ),
        (
            r"$R^2$",
            np.asarray(r2s_train, dtype=float),
            np.asarray(r2s_valid, dtype=float),
            "tab:brown",
        ),
        (
            "MSE",
            np.asarray(mses_train, dtype=float),
            np.asarray(mses_valid, dtype=float),
            "tab:brown",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")

    for idx, (metric_name, train, valid, color) in enumerate(metrics):
        finite = np.isfinite(train) & np.isfinite(valid)
        train = train[finite]
        valid = valid[finite]

        ax = axes[idx]

        box = ax.boxplot(
            [train, valid],
            tick_labels=["Train", "Validation"],
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
            boxprops=dict(linewidth=1.0, color="black"),
            flierprops=dict(
                marker="o",
                markersize=4,
                markerfacecolor=color,
                markeredgecolor="none",
                alpha=0.35,
            ),
        )

        box["boxes"][0].set_facecolor("tab:blue")
        box["boxes"][0].set_alpha(0.45)
        box["boxes"][1].set_facecolor("tab:orange")
        box["boxes"][1].set_alpha(0.45)

        ax.set_title(metric_name, x=0.15, y=0.98, ha="left")
        ax.set_ylabel(metric_name)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(fig, filename, filepath)


# def plot_model_metrics_comparison(
#     correlations_x_train,
#     correlations_x_valid,
#     r2s_x_train,
#     r2s_x_valid,
#     mses_x_train,
#     mses_x_valid,
#     correlations_y_train,
#     correlations_y_valid,
#     r2s_y_train,
#     r2s_y_valid,
#     mses_y_train,
#     mses_y_valid,
#     x_label,
#     y_label,
#     file_name,
#     file_path,
#     title=None,
#     bins=100,
#     figsize=(14, 7),
#     show=False,
# ):
#     metrics = [
#         (
#             "Correlation",
#             correlations_x_train,
#             correlations_x_valid,
#             correlations_y_train,
#             correlations_y_valid,
#         ),
#         (
#             r"$R^2$",
#             r2s_x_train,
#             r2s_x_valid,
#             r2s_y_train,
#             r2s_y_valid,
#         ),
#         (
#             "MSE",
#             mses_x_train,
#             mses_x_valid,
#             mses_y_train,
#             mses_y_valid,
#         ),
#     ]

#     fig, axes = plt.subplots(2, 3, figsize=figsize, layout="constrained")

#     for idx, (metric_name, x_train, x_valid, y_train, y_valid) in enumerate(metrics):
#         x_train = np.asarray(x_train, dtype=float)
#         x_valid = np.asarray(x_valid, dtype=float)
#         y_train = np.asarray(y_train, dtype=float)
#         y_valid = np.asarray(y_valid, dtype=float)

#         finite_train = np.isfinite(x_train) & np.isfinite(y_train)
#         finite_valid = np.isfinite(x_valid) & np.isfinite(y_valid)

#         x_train, y_train = x_train[finite_train], y_train[finite_train]
#         x_valid, y_valid = x_valid[finite_valid], y_valid[finite_valid]

#         lower = min(x_train.min(), x_valid.min(), y_train.min(), y_valid.min())
#         upper = max(x_train.max(), x_valid.max(), y_train.max(), y_valid.max())
#         span = max(upper - lower, 1e-8)
#         xlim = (lower - 0.1 * span, upper + 0.4 * span)
#         ylim = (lower - 0.1 * span, upper + 0.4 * span)

#         ax = axes[0, idx]

#         ax.scatter(
#             x_train,
#             y_train,
#             s=32,
#             color="tab:blue",
#             edgecolor="none",
#             label="Training",
#             zorder=1,
#         )
#         ax.scatter(
#             x_valid,
#             y_valid,
#             s=32,
#             color="tab:orange",
#             edgecolor="none",
#             label="Validation",
#             zorder=2,
#         )
#         ax.plot(xlim, ylim, color="gray", linestyle="--", linewidth=1.5)
#         ax.set_xlim(xlim)
#         ax.set_ylim(ylim)
#         ax.set_aspect("equal")
#         ax.set_xlabel(f"{x_label} {metric_name}")
#         ax.set_ylabel(f"{y_label} {metric_name}")
#         ax.set_title(metric_name, x=0.15, y=0.98, ha="left")
#         ax.legend(frameon=False, loc="upper left")
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)

#         difference_train = x_train - y_train
#         difference_valid = x_valid - y_valid
#         limit = max(
#             np.max(np.abs(difference_train)),
#             np.max(np.abs(difference_valid)),
#             1e-8,
#         )
#         hist_limit = 4 * limit
#         edges = np.linspace(-hist_limit, hist_limit, bins + 1)

#         ax = axes[1, idx]

#         counts_train, _, _ = ax.hist(
#             difference_train,
#             bins=edges,
#             color="tab:blue",
#             alpha=0.5,
#             edgecolor="none",
#             label="Training",
#         )
#         counts_valid, _, _ = ax.hist(
#             difference_valid,
#             bins=edges,
#             color="tab:orange",
#             alpha=0.5,
#             edgecolor="none",
#             label="Validation",
#         )
#         ymax = max(np.max(counts_train), np.max(counts_valid), 1.0)
#         ax.set_xlim(-hist_limit, hist_limit)
#         ax.set_ylim(0, 2 * ymax)
#         ax.set_ylabel("")
#         ax.set_xlabel("")
#         ax.set_yticks([])
#         ax.set_title(metric_name, x=0.15, y=0.98, ha="left")
#         ax.legend(frameon=False, loc="upper left")
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.spines["left"].set_visible(False)

#     fig.suptitle(title)

#     if show:
#         plt.show()

#     return save_figure(fig, file_name, file_path, transparent=True)

def plot_model_metrics_comparison(
    correlations_x_train,
    correlations_x_valid,
    r2s_x_train,
    r2s_x_valid,
    mses_x_train,
    mses_x_valid,
    correlations_y_train,
    correlations_y_valid,
    r2s_y_train,
    r2s_y_valid,
    mses_y_train,
    mses_y_valid,
    x_label,
    y_label,
    file_name,
    file_path,
    title=None,
    bins=30,
    figsize=(14, 7),
    show=False,
):
    metrics = [
        ("Correlation", correlations_x_train, correlations_x_valid, correlations_y_train, correlations_y_valid),
        (r"$R^2$", r2s_x_train, r2s_x_valid, r2s_y_train, r2s_y_valid),
        ("MSE", mses_x_train, mses_x_valid, mses_y_train, mses_y_valid),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize, layout="constrained")
    
    if title is None:
        title = f"{x_label} vs {y_label} Model Comparison"
    fig.suptitle(title, fontweight="bold", fontsize=13, y=1.05)

    for idx, (metric_name, x_train, x_valid, y_train, y_valid) in enumerate(metrics):
        x_train = np.nan_to_num(np.asarray(x_train, dtype=float), nan=0.0)
        x_valid = np.nan_to_num(np.asarray(x_valid, dtype=float), nan=0.0)
        y_train = np.nan_to_num(np.asarray(y_train, dtype=float), nan=0.0)
        y_valid = np.nan_to_num(np.asarray(y_valid, dtype=float), nan=0.0)

        lower = min(x_train.min(), x_valid.min(), y_train.min(), y_valid.min())
        upper = max(x_train.max(), x_valid.max(), y_train.max(), y_valid.max())
        span = max(upper - lower, 1e-8)
        
        # Tighter limits for scatter
        xlim = (lower - 0.05 * span, upper + 0.05 * span)
        ylim = (lower - 0.05 * span, upper + 0.05 * span)

        ax = axes[0, idx]
        ax.scatter(x_train, y_train, s=24, color="tab:blue", alpha=0.7, edgecolor="none", label="Training", zorder=2)
        ax.scatter(x_valid, y_valid, s=24, color="tab:orange", alpha=0.7, edgecolor="none", label="Validation", zorder=3)
        ax.plot(xlim, ylim, color="gray", linestyle="--", linewidth=1.5, zorder=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{x_label} {metric_name}")
        ax.set_ylabel(f"{y_label} {metric_name}")
        ax.set_title(metric_name)
        ax.legend(frameon=False, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        difference_train = x_train - y_train
        difference_valid = x_valid - y_valid
        limit = max(np.max(np.abs(difference_train)), np.max(np.abs(difference_valid)), 1e-8)
        
        # Tighter limits for histograms (target image doesn't squish data into the middle)
        hist_limit = 1.1 * limit
        edges = np.linspace(-hist_limit, hist_limit, bins + 1)

        ax = axes[1, idx]
        counts_train, _, _ = ax.hist(difference_train, bins=edges, color="tab:blue", alpha=0.6, edgecolor="white", linewidth=0.5, label="Training")
        counts_valid, _, _ = ax.hist(difference_valid, bins=edges, color="tab:orange", alpha=0.6, edgecolor="white", linewidth=0.5, label="Validation")
        ymax = max(np.max(counts_train), np.max(counts_valid), 1.0)
        
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_xlim(-hist_limit, hist_limit)
        ax.set_ylim(0, 1.1 * ymax)
        ax.set_ylabel("Number of units")
        ax.set_xlabel(f"{x_label} - {y_label} {metric_name}")
        ax.legend(frameon=False, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)

def plot_model_metric_improvement(
    mse_trial_baseline,
    mse_trial_model,
    corr_trial_baseline,
    corr_trial_model,
    r2_trial_baseline,
    r2_trial_model,
    title,
    file_name,
    file_path,
    alpha=0.05,
    show=False,
):
    n_units = len(mse_trial_baseline)
    p_values_mse, p_values_corr, p_values_r2 = [], [], []
    better_mse, better_corr, better_r2 = [], [], []

    for unit_idx in range(n_units):
        mse_b = np.asarray(mse_trial_baseline[unit_idx], dtype=float)
        mse_m = np.asarray(mse_trial_model[unit_idx], dtype=float)
        corr_b = np.nan_to_num(np.asarray(corr_trial_baseline[unit_idx], dtype=float), nan=0.0)
        corr_m = np.nan_to_num(np.asarray(corr_trial_model[unit_idx], dtype=float), nan=0.0)
        r2_b = np.nan_to_num(np.asarray(r2_trial_baseline[unit_idx], dtype=float), nan=0.0)
        r2_m = np.nan_to_num(np.asarray(r2_trial_model[unit_idx], dtype=float), nan=0.0)

        try:
            _, p_corr = wilcoxon(corr_b, corr_m)
        except ValueError:
            p_corr = 1.0

        try:
            _, p_r2 = wilcoxon(r2_b, r2_m)
        except ValueError:
            p_r2 = 1.0

        try:
            _, p_mse = wilcoxon(mse_b, mse_m)
        except ValueError:
            p_mse = 1.0

        p_values_corr.append(p_corr)
        p_values_r2.append(p_r2)
        p_values_mse.append(p_mse)

        better_corr.append(np.nanmean(corr_m) > np.nanmean(corr_b))
        better_r2.append(np.nanmean(r2_m) > np.nanmean(r2_b))
        better_mse.append(np.nanmean(mse_m) < np.nanmean(mse_b))

    def get_proportions(pvals, is_better):
        sig = sum(1 for p, better in zip(pvals, is_better) if p < alpha and better)
        total = len(pvals)
        return sig / total, (total - sig) / total

    sig_mse, nonsig_mse = get_proportions(p_values_mse, better_mse)
    sig_corr, nonsig_corr = get_proportions(p_values_corr, better_corr)
    sig_r2, nonsig_r2 = get_proportions(p_values_r2, better_r2)

    metrics = ["MSE", "Correlation", r"$R^2$"]
    significant = [sig_mse, sig_corr, sig_r2]
    no_significant_improvement = [nonsig_mse, nonsig_corr, nonsig_r2]

    fig, ax = plt.subplots(figsize=(6, 6))

    p1 = ax.bar(
        metrics,
        significant,
        color="#4558C4",
        label="Significant improvement",
    )

    p2 = ax.bar(
        metrics,
        no_significant_improvement,
        bottom=significant,
        color="#C42A2F",
        label="No significant improvement",
    )

    ax.bar_label(p1, label_type="center", fmt="%.2f", fontsize=12)
    ax.bar_label(p2, label_type="center", fmt="%.2f", fontsize=12)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of units", fontsize=14)

    ax.tick_params(axis="both", labelsize=14)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="grey", linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=False, fontsize=14)

    fig.suptitle(
        title,
        x=0.5,
        y=1.02,
        fontsize=18,
    )
    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)


def plot_shap(
    shap_values,
    title,
    file_name, 
    file_path,
    bin_times=None,
    unit_names=None,
    cmap="RdBu_r",
    center_zero=False,
    figsize=(5, 4),
    show=False,
):
    shap_values = np.asarray(shap_values, dtype=float)


    if shap_values.ndim != 2:
        raise ValueError(
            "shap_values must have shape (n_bins, n_units). "
            f"Received shape {shap_values.shape}."
        )


    n_bins, n_units = shap_values.shape


    if bin_times is None:
        bin_times = np.arange(n_bins)
    if unit_names is None:
        unit_names = np.arange(n_units)


    bin_times = np.asarray(bin_times)
    unit_names = np.asarray(unit_names)


    if bin_times.size != n_bins:
        raise ValueError("bin_times must contain one value per bin.")
    if unit_names.size != n_units:
        raise ValueError("unit_names must contain one name per unit.")


    if center_zero:
        vmin = -np.nanmax(np.abs(shap_values))
        vmax = np.nanmax(np.abs(shap_values))
    else:
        vmin = 0
        vmax = np.nanmax(shap_values)


    mean_by_unit = np.nanmean(shap_values, axis=0)
    mean_by_bin = np.nanmean(shap_values, axis=1)


    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.25, 1],
        height_ratios=[1, 0.25],
        wspace=0.03,
        hspace=0.03,
    )


    ax_main = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_bottom = fig.add_subplot(gs[1, 1])
    ax_corner = fig.add_subplot(gs[1, 0])
    ax_corner.axis("off")


    im = ax_main.imshow(
        shap_values.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


    ax_main.set_title(title, y=1.02)
    ax_main.set_xticks([])
    max_yticks = 12
    ytick_idx = np.unique(
        np.linspace(0, n_units - 1, min(max_yticks, n_units), dtype=int)
    )


    ax_main.set_yticks(ytick_idx)
    ax_main.tick_params(axis="y", left=False, labelleft=False)
    ax_main.spines[["top", "right"]].set_visible(False)


    unit_idx = np.arange(n_units)


    ax_left.axvline(0, color="black", linewidth=0.8, zorder=0)


    ax_left.hlines(
        y=unit_idx,
        xmin=0,
        xmax=mean_by_unit,
        color="black",
        linewidth=1.1,
        zorder=1,
    )


    ax_left.scatter(
        mean_by_unit,
        unit_idx,
        color="black",
        s=14,
        zorder=2,
    )


    left_limit = max(np.nanmax(np.abs(mean_by_unit)), np.finfo(float).eps)
    ax_left.set_xlim(-1.1 * left_limit, 1.1 * left_limit)


    ax_left.set_ylabel("Unit")
    ax_left.set_yticks(ytick_idx, unit_names[ytick_idx], fontsize=7)
    ax_left.set_xlabel("Mean SHAP", fontsize=8)
    ax_left.tick_params(axis="x", labelsize=7)
    ax_left.tick_params(axis="y", labelsize=7)
    ax_left.spines[["top", "right"]].set_visible(False)


    ax_bottom.axhline(0, color="black", linewidth=0.8, zorder=0)


    ax_bottom.plot(
        bin_times,
        mean_by_bin,
        color="black",
        linewidth=1.5,
    )


    ax_bottom.fill_between(
        bin_times,
        0,
        mean_by_bin,
        color="black",
        alpha=0.20,
    )


    ax_bottom.set_xlabel("Time")
    ax_bottom.set_ylabel("Mean\nSHAP", fontsize=8)
    ax_bottom.tick_params(axis="both", labelsize=7)
    ax_bottom.spines[["top", "right"]].set_visible(False)


    cbar = fig.colorbar(
        im,
        ax=ax_main,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("SHAP value", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


    if show:
        plt.show()


    return save_figure(fig, file_name, file_path)


def plot_shap_hist(
    unit_bin_shap_shuffles,
    unit_bin_shap_permutations,
    variable_name,
    unit_name,
    bin_time,
    color,
    title,
    filename,
    filepath,
    bins=50,
    show=False,
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")
    shuffle_values = unit_bin_shap_shuffles[np.isfinite(unit_bin_shap_shuffles)]
    permutation_values = unit_bin_shap_permutations[np.isfinite(unit_bin_shap_permutations)]
    values = np.concatenate([shuffle_values, permutation_values])
    bin_edges = np.linspace(values.min(), values.max(), bins + 1)
    ax.hist(shuffle_values, bins=bin_edges, density=True, color="gray", edgecolor=None, alpha=0.5, label="Shuffle null")
    ax.hist(permutation_values, bins=bin_edges, density=True, color=color, edgecolor=None, alpha=0.5, label="Permutation")
    ax.axvline(shuffle_values.mean(), color="gray", linestyle="--", linewidth=2, label="Shuffle mean")
    ax.axvline(permutation_values.mean(), color=color, linestyle="--", linewidth=2, label="Permutation mean")
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    if show:
        plt.show()
    return save_figure(fig, filename, filepath)


def plot_variable_selectivity_hist(
    unit_selectivity_shuffles,
    unit_selectivity_permutations,
    variable_name,
    unit_name,
    color,
    title,
    filename,
    filepath,
    bins=50,
    show=False,
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")
    shuffle_values = unit_selectivity_shuffles[np.isfinite(unit_selectivity_shuffles)]
    permutation_values = unit_selectivity_permutations[np.isfinite(unit_selectivity_permutations)]
    values = np.concatenate([shuffle_values, permutation_values])
    bin_edges = np.linspace(values.min(), values.max(), bins + 1)
    ax.hist(shuffle_values, bins=bin_edges, density=True, color="gray", edgecolor=None, alpha=0.5, label="Shuffle null")
    ax.hist(permutation_values, bins=bin_edges, density=True, color=color, edgecolor=None, alpha=0.5, label="Permutation")
    ax.axvline(shuffle_values.mean(), color="gray", linestyle="--", linewidth=2, label="Shuffle mean")
    ax.axvline(permutation_values.mean(), color=color, linestyle="--", linewidth=2, label="Permutation mean")
    ax.set_xlabel("Mean absolute SHAP")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    if show:
        plt.show()
    return save_figure(fig, filename, filepath)


def plot_variable_selectivity_curves(
    variable_selectivity,
    variable_name,
    significant,
    color,
    title,
    filename,
    filepath,
    show=False,
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")
    sort_idx = np.argsort(variable_selectivity)
    sorted_selectivity = variable_selectivity[sort_idx]
    sorted_significant = significant[sort_idx]
    x = np.arange(1, len(sorted_selectivity) + 1)
    split_idx = np.argmax(sorted_significant) if sorted_significant.any() else len(sorted_significant) - 1
    ax.plot(x[: split_idx + 1], sorted_selectivity[: split_idx + 1], color=color, linewidth=2.5, alpha=0.3)
    ax.plot(x[split_idx:], sorted_selectivity[split_idx:], color=color, linewidth=2.5, alpha=1.0)
    ax.set_xlabel("Units")
    ax.set_ylabel("Effect size")
    ax.set_xlim(1, len(sorted_selectivity))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    if show:
        plt.show()
    return save_figure(fig, filename, filepath)


def plot_variable_selectivity_pca(
    variable_selectivities,
    variable_selectivity,
    variable_name,
    significant,
    color,
    title,
    filename,
    filepath,
    show=False,
):
    unit_features = variable_selectivities.T
    unit_coordinates = PCA(n_components=2).fit_transform(unit_features)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")
    norm = mcolors.Normalize(vmin=np.nanmin(variable_selectivity), vmax=np.nanmax(variable_selectivity))
    base_rgb = mcolors.to_rgb(color)
    alphas = np.clip(norm(variable_selectivity), 0.05, 1.0)
    colors = [(*base_rgb, alpha) for alpha in alphas]
    edgecolors = ["black" if sig else "none" for sig in significant]
    ax.scatter(
        unit_coordinates[:, 0],
        unit_coordinates[:, 1],
        color=colors,
        s=70,
        edgecolors=edgecolors,
        linewidths=1.2,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    if show:
        plt.show()
    return save_figure(fig, filename, filepath)


def plot_class_selectivity_pca(
    variable_selectivity,
    class_name,
    significant,
    color,
    title,
    filename,
    filepath,
    show=False,
):
    unit_features = variable_selectivity.T
    unit_coordinates = PCA(n_components=2).fit_transform(unit_features)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")
    colors = [color if sig else mcolors.to_rgba(color, alpha=0.15) for sig in significant]
    edgecolors = ["black" if sig else "lightgray" for sig in significant]
    ax.scatter(
        unit_coordinates[:, 0],
        unit_coordinates[:, 1],
        c=colors,
        s=70,
        edgecolors=edgecolors,
        linewidths=1.2,
    )
    ax.set_title(class_name.replace("_", " ").title())
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    if show:
        plt.show()
    return save_figure(fig, filename, filepath)


def plot_variable_selectivity_matrix(
    variable_selectivity,
    variable_names,
    unit_names,
    significant,
    variable_colors,
    title,
    filename,
    filepath,
    show=False,
):
    unit_scores = np.nanmean(variable_selectivity, axis=0)
    sort_idx = np.argsort(-unit_scores)
    sorted_matrix = variable_selectivity[:, sort_idx]
    sorted_significant = significant[:, sort_idx]
    sorted_unit_names = unit_names[sort_idx]
    vmin = np.nanmin(sorted_matrix)
    vmax = np.nanmax(sorted_matrix)
    n_vars, n_units = sorted_matrix.shape
    fig, ax = plt.subplots(1, 1, figsize=(max(8, 0.15 * n_units), 4), layout="constrained")
    row_height = 0.7
    row_gap = 0.3
    for row_idx, color in enumerate(variable_colors):
        row_values = sorted_matrix[row_idx : row_idx + 1]
        cmap = mcolors.LinearSegmentedColormap.from_list(f"var_{row_idx}", ["white", color], N=256)
        center = row_idx * (row_height + row_gap)
        ax.imshow(
            row_values,
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            extent=[-0.5, n_units - 0.5, center - row_height / 2.0, center + row_height / 2.0],
        )
        selected_units = np.where(sorted_significant[row_idx])[0]
        for unit_idx in selected_units:
            ax.add_patch(
                plt.Rectangle(
                    (unit_idx - 0.5, center - row_height / 2.0),
                    1.0,
                    row_height,
                    fill=False,
                    edgecolor="black",
                    linewidth=0.8,
                )
            )
    row_positions = [i * (row_height + row_gap) for i in range(n_vars)]
    ax.set_xticks(np.arange(n_units))
    ax.set_xticklabels(sorted_unit_names, rotation=90, fontsize=7)
    ax.set_yticks(row_positions)
    ax.set_yticklabels(variable_names, fontsize=9)
    ax.set_xlim(-0.5, n_units - 0.5)
    ax.set_ylim(-row_gap, (n_vars - 1) * (row_height + row_gap) + row_height + row_gap)
    ax.set_xlabel("Units")
    ax.set_ylabel("Variables")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    if show:
        plt.show()
    return save_figure(fig, filename, filepath)


def plot_class_selectivity_matrix(
    class_selectivity,
    class_names,
    unit_names,
    significant,
    class_colors,
    title,
    filename,
    filepath,
    show=False,
):
    unit_scores = np.nanmean(class_selectivity, axis=0)
    sort_idx = np.argsort(-unit_scores)
    sorted_matrix = class_selectivity[:, sort_idx]
    sorted_significant = significant[:, sort_idx]
    sorted_unit_names = unit_names[sort_idx]
    n_classes, n_units = sorted_matrix.shape
    fig, ax = plt.subplots(1, 1, figsize=(max(8, 0.15 * n_units), 4), layout="constrained")
    row_height = 0.7
    row_gap = 0.3
    for row_idx, color in enumerate(class_colors):
        center = row_idx * (row_height + row_gap)
        row_colors = [color if sig else mcolors.to_rgba(color, alpha=0.15) for sig in sorted_significant[row_idx]]
        for unit_idx, c in enumerate(row_colors):
            ax.add_patch(
                plt.Rectangle(
                    (unit_idx - 0.5, center - row_height / 2.0),
                    1.0,
                    row_height,
                    facecolor=c,
                    edgecolor="black" if sorted_significant[row_idx, unit_idx] else "none",
                    linewidth=0.8,
                )
            )
    row_positions = [i * (row_height + row_gap) for i in range(n_classes)]
    ax.set_xticks(np.arange(n_units))
    ax.set_xticklabels(sorted_unit_names, rotation=90, fontsize=7)
    ax.set_yticks(row_positions)
    ax.set_yticklabels([c.replace("_", " ").title() for c in class_names], fontsize=9)
    ax.set_xlim(-0.5, n_units - 0.5)
    ax.set_ylim(-row_gap, (n_classes - 1) * (row_height + row_gap) + row_height + row_gap)
    ax.set_xlabel("Units")
    ax.set_ylabel("Classes")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    if show:
        plt.show()
    return save_figure(fig, filename, filepath)


def plot_group_influence_matrix(
    correlation_matrix,
    r2_matrix,
    mse_matrix,
    group_names,
    title,
    file_name,
    file_path,
    show=False,
):
    matrices = [
        ("Correlation", correlation_matrix),
        (r"$R^2$", r2_matrix),
        ("MSE", mse_matrix),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
        layout="constrained",
    )

    for ax, (metric_name, matrix) in zip(axes, matrices):
        im = ax.imshow(
            matrix,
            origin="upper",
            cmap="magma",
            vmin=0,
            vmax=1,
        )

        ax.set_title(metric_name)
        ax.set_xlabel("Conditioning source group")
        ax.set_ylabel("Predicted target group")
        ax.set_xticks(
            np.arange(len(group_names)),
            group_names,
            rotation=45,
            ha="right",
        )
        ax.set_yticks(np.arange(len(group_names)), group_names)

        for row_idx in range(matrix.shape[0]):
            for column_idx in range(matrix.shape[1]):
                value = matrix[row_idx, column_idx]
                label = "n/a" if np.isnan(value) else f"{value:.2f}"
                color = (
                    "white"
                    if not np.isnan(value) and value < 0.55
                    else "black"
                )

                ax.text(
                    column_idx,
                    row_idx,
                    label,
                    ha="center",
                    va="center",
                    color=color,
                )

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="Fraction of target units",
        )

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)


def plot_side_by_side_heatmaps(
    R_left,
    R_right,
    title_left="R_true",
    title_right="R_hat",
    title="Correlation Matrix Comparison",
    file_name="side_by_side_heatmaps",
    file_path="./plot/",
    vmin=-1.0,
    vmax=1.0,
    cmap="RdBu_r",
    show=False,
):
    """Plot two square matrices side by side as heatmaps.

    Intended for comparing R_true (ground-truth correlation) vs R_hat
    (model-estimated correlation) in the positive-control factor-analytic
    identifiability check (Section B). Uses project-standard styling.

    Parameters
    ----------
    R_left  : ndarray (D, D)  left heatmap matrix (e.g. R_true)
    R_right : ndarray (D, D)  right heatmap matrix (e.g. R_hat)
    title_left  : str   subtitle for the left panel
    title_right : str   subtitle for the right panel
    title   : str   figure suptitle
    file_name, file_path : str  passed to save_figure
    vmin, vmax : float  shared colour scale limits
    cmap : str  matplotlib colormap name
    show : bool  call plt.show() if True
    """
    set_pub_style()

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 4.5),
        layout="constrained",
    )

    for ax, matrix, subtitle in zip(axes, [R_left, R_right], [title_left, title_right]):
        im = ax.imshow(
            matrix,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Feature index")
        n = matrix.shape[0]
        ax.set_xticks([0, n - 1])
        ax.set_yticks([0, n - 1])
        fig.colorbar(im, ax=ax, shrink=0.75, label="Correlation")

    fig.suptitle(title, fontweight="bold")

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)

def bin_averaged_unit_covariance(covariance_matrix, n_bins, n_units):
    """
    Collapse a (bin, unit)-ordered ``(K*N, K*N)`` covariance into an ``N x N``
    unit-by-unit matrix: for each pair of units, the same-bin covariance
    ``Cov[(b, i), (b, j)]`` averaged over all bins ``b``.
    """
    cov = np.asarray(covariance_matrix).reshape(n_bins, n_units, n_bins, n_units)
    return np.einsum("bibj->ij", cov) / n_bins


def plot_covariance_matrix_by_unit(
    mean_cov_lit_model,
    title,
    file_name,
    file_path,
    unit_names=None,
    unit_indices=None,
    vmax=None,
    linthresh=None,
    show=False,
    aggregate="mean",
    scale=None,
):
    """
    Unit-by-unit covariance.

    aggregate="mean" (default): one cell per unit pair, the same-bin covariance
        averaged over all time bins (an N x N matrix). This removes the
        bin-by-bin sampling noise and shows the cross-neuron block structure.
    aggregate=None: the full (unit, bin) x (unit, bin) matrix, one K x K block
        per unit pair (the previous behaviour).

    scale: "linear" or "symlog"; defaults to linear for the averaged matrix and
        symlog for the full one. When ``vmax`` is not given the colour range is
        set from the largest off-diagonal value, so the variances on the
        diagonal do not wash out the structure.
    """
    cov_model = mean_cov_lit_model.full_model.cov_model

    covariance_matrix = get_covariance_matrix(cov_model)
    covariance_matrix = covariance_matrix.detach().cpu().numpy()

    n_bins = cov_model.n_bins
    n_units = cov_model.n_units

    if unit_names is None:
        unit_names = np.arange(n_units)

    if unit_indices is None:
        unit_indices = np.arange(n_units)
    else:
        unit_indices = np.atleast_1d(unit_indices)

    unit_names = np.asarray(unit_names)[unit_indices]
    n_units_plot = len(unit_indices)

    if aggregate == "mean":
        covariance_matrix = bin_averaged_unit_covariance(covariance_matrix, n_bins, n_units)
        covariance_matrix = covariance_matrix[np.ix_(unit_indices, unit_indices)]
        block = 1
    elif aggregate is None:
        # Reorder the matrix from (bin, unit) to (unit, bin)
        covariance_matrix = covariance_matrix.reshape(n_bins, n_units, n_bins, n_units)
        covariance_matrix = covariance_matrix.transpose(1, 0, 3, 2)
        covariance_matrix = covariance_matrix.reshape(n_units * n_bins, n_units * n_bins)

        feature_indices = np.concatenate(
            [np.arange(unit_idx * n_bins, (unit_idx + 1) * n_bins) for unit_idx in unit_indices]
        )
        covariance_matrix = covariance_matrix[np.ix_(feature_indices, feature_indices)]
        block = n_bins
    else:
        raise ValueError(f"aggregate must be 'mean' or None, got {aggregate!r}")

    if scale is None:
        scale = "linear" if aggregate == "mean" else "symlog"

    if vmax is None:
        off_diag = covariance_matrix[~np.eye(len(covariance_matrix), dtype=bool)]
        vmax = np.nanmax(np.abs(off_diag)) if off_diag.size else np.nan
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.nanmax(np.abs(covariance_matrix))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    if scale == "symlog":
        if linthresh is None:
            linthresh = max(vmax * 0.01, 1e-8)
        norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
    else:
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    if aggregate == "mean":
        fig_side = 6.0
        fs_tick, fs_label, fs_title = 9, 12, 13
        tick_step = max(1, int(np.ceil(n_units_plot / 20)))
    else:
        fig_side = max(5, n_units_plot * 0.25)
        fs_tick, fs_label, fs_title = 7, 10, 12
        tick_step = 1

    fig, ax = plt.subplots(figsize=(fig_side + 1.0, fig_side), layout="constrained")

    im = ax.imshow(
        covariance_matrix,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=norm,
    )

    if aggregate is None:
        for unit_idx in range(1, n_units_plot):
            line_pos = unit_idx * n_bins - 0.5
            ax.axvline(line_pos, color="black", linewidth=0.5, alpha=0.5)
            ax.axhline(line_pos, color="black", linewidth=0.5, alpha=0.5)

    unit_centers = np.arange(n_units_plot) * block + (block - 1) / 2
    shown = np.arange(0, n_units_plot, tick_step)

    ax.set(
        xticks=unit_centers[shown],
        xticklabels=unit_names[shown],
        yticks=unit_centers[shown],
        yticklabels=unit_names[shown],
    )
    ax.tick_params(axis="x", labelrotation=90, labelsize=fs_tick)
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.set_xlabel("Unit", fontsize=fs_label)
    ax.set_ylabel("Unit", fontsize=fs_label)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar_label = "Covariance, mean over bins" if aggregate == "mean" else "Covariance"
    if scale == "symlog":
        cbar_label += " (symlog scale)"
    cbar.set_label(cbar_label, fontsize=fs_label - 1)
    cbar.ax.tick_params(labelsize=fs_tick)

    fig.suptitle(title, fontsize=fs_title, fontweight="bold")

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path, dpi=300 if aggregate == "mean" else 100)


def task_tuning_map(Y, x):
    """
    Tuning of every (unit, bin) to one task variable.

    Y : (T, N, K) activity, x : (T,) task variable (binary or continuous).
    Returns the (N, K) Pearson correlation across trials between x and the
    activity in that unit and bin (point-biserial for a binary variable).
    Cells with no variance across trials are NaN.
    """
    Y = np.asarray(Y, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1)
    xc = x - np.nanmean(x)
    Yc = Y - np.nanmean(Y, axis=0, keepdims=True)
    num = np.einsum("t,tnk->nk", xc, Yc)
    den = np.sqrt(np.sum(xc ** 2)) * np.sqrt(np.sum(Yc ** 2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    r[~np.isfinite(r)] = np.nan
    return r


def plot_task_tuning_maps(
    conditions: dict,
    task_vars,
    task_var_names,
    variables=("tslp", "rew"),
    bin_times=None,
    labels: Optional[dict] = None,
    variable_labels: Optional[dict] = None,
    title: Optional[str] = None,
    vmax: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Unit x time-bin tuning maps, one group of panels per task variable.

    For each variable, every condition in ``conditions`` (label -> (T, N, K)
    array; the first one is the reference, normally the real data) gets an
    imshow of the per-(unit, bin) correlation between the variable and the
    activity across trials. Within a variable all panels share the unit order
    (sorted by the reference's mean tuning) and the colour scale, so the
    panels can be compared directly. The pattern correlation between each
    condition's map and the reference map is printed in the panel title.

    task_vars : (T, V) array, task_var_names : names of its columns.
    """
    set_pub_style()

    cond_names = list(conditions.keys())
    task_var_names = list(task_var_names)
    task_vars = np.asarray(task_vars, dtype=float)
    variables = [v for v in variables if v in task_var_names]
    if not variables:
        raise ValueError(f"None of the requested variables are in {task_var_names}")

    default_labels = {
        c: ("Real data" if c == "real" else f"Synthetic ({c})") for c in cond_names
    }
    labels = {**default_labels, **(labels or {})}
    default_var_labels = {"rew": "Reward (rew)", "tslp": "Time since last press (tslp)",
                          "choice": "Choice"}
    variable_labels = {**default_var_labels, **(variable_labels or {})}

    first = np.asarray(conditions[cond_names[0]])
    n_units, n_bins = first.shape[1], first.shape[2]
    if bin_times is None:
        bin_times = np.arange(n_bins)
    bin_times = np.asarray(bin_times, dtype=float)
    dt = np.median(np.diff(bin_times)) if n_bins > 1 else 1.0
    extent = [bin_times[0] - dt / 2, bin_times[-1] + dt / 2, -0.5, n_units - 0.5]

    n_cond, n_var = len(cond_names), len(variables)
    fs_tick, fs_label, fs_title, fs_group = 9.5, 11, 11, 12.5

    fig = plt.figure(figsize=(2.55 * n_cond * n_var + 0.9 * n_var, 4.3), layout="constrained")
    subfigs = np.atleast_1d(fig.subfigures(1, n_var, wspace=0.04))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E5E7EB")

    for sf, var in zip(subfigs, variables):
        x = task_vars[:, task_var_names.index(var)]
        maps = {c: task_tuning_map(conditions[c], x) for c in cond_names}

        ref = maps[cond_names[0]]
        order = np.argsort(np.nan_to_num(np.nanmean(ref, axis=1)))   # most negative at the bottom
        v = vmax
        if v is None:
            pooled = np.concatenate([np.abs(m[np.isfinite(m)]).ravel() for m in maps.values()])
            v = float(np.percentile(pooled, 99)) if pooled.size else 1.0
            v = max(v, 1e-3)

        axes = np.atleast_1d(sf.subplots(1, n_cond, sharey=True))
        for k, (ax, c) in enumerate(zip(axes, cond_names)):
            m = maps[c][order]
            im = ax.imshow(np.ma.masked_invalid(m), origin="lower", aspect="auto",
                           interpolation="nearest", cmap=cmap, vmin=-v, vmax=v, extent=extent)
            if bin_times[0] < 0 < bin_times[-1]:
                ax.axvline(0, color="black", lw=1.0, ls="--")
            panel_title = labels[c]
            if k > 0:
                ok = np.isfinite(ref) & np.isfinite(maps[c])
                if ok.sum() > 2 and np.std(ref[ok]) > 0 and np.std(maps[c][ok]) > 0:
                    sim = np.corrcoef(ref[ok], maps[c][ok])[0, 1]
                    panel_title += f"\nmap similarity r = {sim:.2f}"
            elif n_cond > 1:
                panel_title += "\n"
            ax.set_title(panel_title, fontsize=fs_title)
            ax.set_xlabel("Time from press (s)", fontsize=fs_label)
            ax.tick_params(labelsize=fs_tick)
            ax.set_yticks([0, n_units - 1])
            ax.set_yticklabels(["1", str(n_units)])
            if k == 0:
                ax.set_ylabel("Units (sorted by real-data tuning)" if cond_names[0] == "real"
                              else "Units (sorted)", fontsize=fs_label)
            for sp in ax.spines.values():
                sp.set_linewidth(0.8)

        cbar = sf.colorbar(im, ax=axes, shrink=0.85, pad=0.02, aspect=28)
        cbar.set_label("correlation with variable", fontsize=fs_label - 1)
        cbar.ax.tick_params(labelsize=fs_tick - 0.5)
        sf.suptitle(variable_labels.get(var, var), fontsize=fs_group, fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=fs_group + 0.5, fontweight="bold")

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
