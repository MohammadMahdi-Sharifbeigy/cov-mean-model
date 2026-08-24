"""Plotting helpers faithfully extracted from the experiment notebook.

Imports are deliberately lightweight at module load; optional SHAP support is loaded lazily.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib import colors
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon, norm, ttest_rel
from sklearn.metrics import r2_score
SEED = 42
try:
    import colormaps as cmaps
except ImportError:  # optional notebook styling dependency
    cmaps = None

from .core import save_figure, save_shap_plot, p_to_stars
def plot_shap(shap_values, title, file_name, file_path,
    bin_times=None, unit_names=None, cmap="RdBu_r", figsize=(10, 7), show=True,):
    """
    Plots a 2D Spatiotemporal SHAP Heatmap (Time Bins x Units) with marginal 
    mean projections on the left and bottom axes.
    """
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim != 2:
        raise ValueError(f"shap_values must have shape (n_bins, n_units). Received shape {shap_values.shape}.")

    n_bins, n_units = shap_values.shape

    if bin_times is None:
        bin_times = np.arange(n_bins)
    if unit_names is None:
        unit_names = np.arange(n_units)

    bin_times = np.asarray(bin_times)
    unit_names = np.asarray(unit_names)

    if cmap == "RdBu_r":
        vmin = -np.nanmax(np.abs(shap_values))
        vmax = np.nanmax(np.abs(shap_values))
    else:
        vmin = 0
        vmax = np.nanmax(shap_values)

    mean_by_unit = np.nanmean(shap_values, axis=0)
    mean_by_bin = np.nanmean(shap_values, axis=1)

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[0.25, 1], height_ratios=[1, 0.25], wspace=0.03, hspace=0.03)

    ax_main = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_bottom = fig.add_subplot(gs[1, 1])
    ax_corner = fig.add_subplot(gs[1, 0])
    ax_corner.axis("off")

    im = ax_main.imshow(shap_values.T, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_main.set_title(title, fontweight="bold")
    ax_main.set_ylabel("Unit")
    ax_main.set_xticks([])
    ax_main.set_yticks(np.arange(n_units))
    ax_main.set_yticklabels(unit_names, fontsize=7)
    ax_main.spines[["top", "right"]].set_visible(False)

    unit_idx = np.arange(n_units)
    ax_left.axvline(0, color="black", linewidth=0.8, zorder=0)
    ax_left.hlines(y=unit_idx, xmin=0, xmax=mean_by_unit, color="C0", linewidth=1.1, zorder=1)
    ax_left.scatter(mean_by_unit, unit_idx, color="C0", s=14, zorder=2)
    left_limit = max(np.nanmax(np.abs(mean_by_unit)), np.finfo(float).eps)
    ax_left.set_xlim(-1.1 * left_limit, 1.1 * left_limit)
    ax_left.set_xlabel("Mean SHAP", fontsize=8)
    ax_left.tick_params(axis="x", labelsize=7)
    ax_left.tick_params(axis="y", left=False, labelleft=False)
    ax_left.spines[["top", "right", "left"]].set_visible(False)

    ax_bottom.axhline(0, color="black", linewidth=0.8, zorder=0)
    ax_bottom.plot(bin_times, mean_by_bin, color="C0", linewidth=1.5)
    ax_bottom.fill_between(bin_times, 0, mean_by_bin, color="C0", alpha=0.20)
    ax_bottom.set_xlabel("Time")
    ax_bottom.set_ylabel("Mean\nSHAP", fontsize=8)
    ax_bottom.tick_params(axis="both", labelsize=7)
    ax_bottom.spines[["top", "right"]].set_visible(False)

    cbar = fig.colorbar(im, ax=ax_main, fraction=0.025, pad=0.02)
    cbar.set_label("SHAP value", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)

def plot_shap_results(Conf, condition='step3', model_label='Conditional / Identity',
    n_splits=20, variable_names=None, scaling_factor=1.0, target_vars=None):
    matplotlib.use('Agg')
    print(f"\n{'='*40}\nPlotting Spatiotemporal SHAP Analysis for {condition} - {model_label}\n{'='*40}")
    
    all_shap_values = []
    all_base_values = []
    all_explain_data = []
    
    active_var_names = None
    for fold_idx in range(n_splits):
        shap_save_path = f"{RESULTS}shap_values_{condition}_{model_label.replace(' / ', '_')}_fold{fold_idx}.npz"
        if os.path.exists(shap_save_path):
            data = np.load(shap_save_path)
            all_shap_values.append(data['shap_values'])
            all_base_values.append(data['base_values'])
            all_explain_data.append({
                'pos': data['explain_pos'],
                'den': data['explain_den'],
                'spa': data['explain_spa']
            })
            if 'active_var_names' in data:
                active_var_names = data['active_var_names'].tolist()
            
    if not all_shap_values:
        print("No SHAP values found to plot.")
        return
        
    shap_values = np.concatenate(all_shap_values, axis=0) * scaling_factor
    base_values = np.concatenate(all_base_values, axis=0) * scaling_factor
    print(f"Loaded SHAP tensors (shape: {shap_values.shape}, scaling: {scaling_factor}x)")
    
    ex_pos = np.concatenate([d['pos'] for d in all_explain_data], axis=0)
    ex_den = np.concatenate([d['den'] for d in all_explain_data], axis=0)
    ex_spa = np.concatenate([d['spa'] for d in all_explain_data], axis=0)
    
    n_trials, n_vars, n_bins, n_units = shap_values.shape
    
    save_dir = f"{FIGURES}SHAP_Plots/{condition}/{model_label.replace(' / ', '_')}/"
    os.makedirs(save_dir, exist_ok=True)
    
    feat_names = []
    if active_var_names is not None:
        feat_names = [str(v).strip("['\"] ") for v in active_var_names]
    elif variable_names is not None:
        for v in variable_names:
            if hasattr(v, 'item'):
                try: v = v.item()
                except: pass
            if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                v = v[0]
            feat_names.append(str(v).strip("['\"] "))
    if len(feat_names) < n_vars:
        missing = n_vars - len(feat_names)
        feat_names += (['Density (den)', 'Sparsity (spa)'] if missing == 2
                       else [f"Feature {i}" for i in range(len(feat_names), n_vars)])
    elif len(feat_names) > n_vars:
        feat_names = feat_names[:n_vars]
        
    bt = bin_times if 'bin_times' in globals() else np.arange(n_bins)
    un = unit_names if 'unit_names' in globals() else np.arange(n_units)

    # ---> Resolve target_indices from caller-supplied target_vars
    if target_vars is not None:
        target_indices = [i for i, name in enumerate(feat_names)
                          if any(tv == name for tv in target_vars)]
        if not target_indices:
            print(f"Warning: None of {target_vars} matched feat_names. Plotting all.")
            target_indices = list(range(n_vars))
        else:
            print(f"Filtering 2D plots to: {[feat_names[i] for i in target_indices]}")
    else:
        target_indices = list(range(n_vars))

    # =========================================================================
    # 1. POPULATION-WIDE SUMMARY PLOTS (Beeswarm, Bar, & Mean Waterfall)
    # =========================================================================
    print("Generating population-wide overall feature importance (Beeswarm, Bar, Waterfall)...")
    explanation_global = None
    try:
        sv_global = shap_values.sum(axis=(2, 3))
        bv_global = base_values.sum(axis=(1, 2))
        ex_pos_mean = ex_pos.mean(axis=1)
        bin_data_global = np.concatenate([ex_pos_mean, ex_den, (ex_spa > 0).astype(int)], axis=1)
        bin_data_global = np.array(bin_data_global.astype(float), copy=True)
        explanation_global = shap.Explanation(
            values=sv_global, base_values=bv_global,
            data=bin_data_global, feature_names=feat_names
        )
        summary_dir = os.path.join(save_dir, "population_summary")
        os.makedirs(summary_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(explanation_global, max_display=len(feat_names), show=False)
        plt.title(f"Population-Wide Feature Importance (Beeswarm) - {condition}")
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, "population_beeswarm.png"), bbox_inches='tight', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        shap.plots.bar(explanation_global, max_display=len(feat_names), show=False)
        plt.title(f"Population-Wide Feature Importance (Bar) - {condition}")
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, "population_bar.png"), bbox_inches='tight', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation_global.mean(0), max_display=len(feat_names), show=False)
        plt.title(f"Global Average Feature Contributions (Waterfall) - {condition}")
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, "population_mean_waterfall.png"), bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate population-wide summary plots: {e}")

    # =========================================================================
    # 2. GLOBAL SPATIOTEMPORAL HEATMAPS
    # =========================================================================
    global_dir = os.path.join(save_dir, "spatiotemporal_global_mean")
    print(f"Generating Global 2D Spatiotemporal maps -> {global_dir}")
    for var_idx in target_indices:
        var_name = feat_names[var_idx]
        mean_abs_shap = np.abs(shap_values[:, var_idx, :, :]).mean(axis=0)
        plot_shap(
            shap_values=mean_abs_shap,
            title=f"Global Mean Absolute SHAP: {var_name}",
            file_name=f"global_mean_abs_{var_name.replace(' ', '_')}",
            file_path=global_dir, bin_times=bt, unit_names=un, cmap="magma", show=False
        )

    # =========================================================================
    # 3. PER-TRIAL SPATIOTEMPORAL HEATMAPS & TRIAL WATERFALLS
    # =========================================================================
    sample_trials = np.random.choice(n_trials, size=min(5, n_trials), replace=False)
    trials_dir = os.path.join(save_dir, "spatiotemporal_trials")
    print(f"Generating Per-trial maps & Waterfalls for trials {sample_trials} -> {trials_dir}")
    for t_idx in sample_trials:
        cur_trial_dir = os.path.join(trials_dir, f"trial_{t_idx}")
        os.makedirs(cur_trial_dir, exist_ok=True)

        if explanation_global is not None:
            try:
                plt.figure(figsize=(10, 6))
                shap.plots.waterfall(explanation_global[t_idx], max_display=len(feat_names), show=False)
                plt.title(f"Trial {t_idx} Population Contributions (Waterfall)")
                plt.tight_layout()
                plt.savefig(os.path.join(cur_trial_dir, f"waterfall_trial_{t_idx}.png"), bbox_inches='tight', dpi=300)
                plt.close()
            except Exception:
                pass

        for var_idx in target_indices:
            var_name = feat_names[var_idx]
            plot_shap(
                shap_values=shap_values[t_idx, var_idx, :, :],
                title=f"Trial {t_idx} SHAP Modulation: {var_name}",
                file_name=f"trial_{t_idx}_{var_name.replace(' ', '_')}",
                file_path=cur_trial_dir, bin_times=bt, unit_names=un, cmap="RdBu_r", show=False
            )

    print(f"SHAP analysis complete! All figures saved in: {save_dir}")

def plot_shap_pearson(
    r_matrix,          # (n_units, n_bins) — Pearson r values
    sig_mask,          # (n_units, n_bins) bool — True = significant (p <= 0.05)
    title,
    file_name,
    file_path,
    bin_times=None,
    unit_names=None,
    figsize=(11, 7),
    show=False,
    ):
    """
    Plot a 2D Pearson r heatmap (Units x Bins) with marginal projections.
    Cells where p > 0.05 are grayed out.
    Layout matches plot_shap: left=mean SHAP per unit, bottom=mean SHAP per bin.
    """
    r_matrix = np.asarray(r_matrix, dtype=float)
    sig_mask  = np.asarray(sig_mask,  dtype=bool)
    n_units, n_bins = r_matrix.shape

    if bin_times is None:
        bin_times = np.arange(n_bins)
    if unit_names is None:
        unit_names = np.arange(n_units)

    # Only compute means over significant cells to avoid noise pulling the signal
    r_sig = np.where(sig_mask, r_matrix, np.nan)
    mean_by_unit = np.nanmean(r_sig, axis=1)   # (n_units,)
    mean_by_bin  = np.nanmean(r_sig, axis=0)   # (n_bins,)

    # ---> BUGFIX 1: Safely handle NaN limits when no cells are significant
    if np.all(np.isnan(mean_by_unit)) or np.all(np.isnan(r_matrix)):
        xlim = 1.0
        vmin, vmax = -1.0, 1.0
    else:
        vabs = max(np.nanmax(np.abs(r_matrix)), 1e-6)
        vmin, vmax = -vabs, vabs
        xlim = max(np.nanmax(np.abs(mean_by_unit)), 1e-6) * 1.15

    # Replace NaNs with 0.0 for clean plotting of empty/non-significant projections
    mean_by_unit = np.nan_to_num(mean_by_unit, nan=0.0)
    mean_by_bin  = np.nan_to_num(mean_by_bin, nan=0.0)

    # ── Layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs  = fig.add_gridspec(2, 2, width_ratios=[0.25, 1],
                           height_ratios=[1, 0.25], wspace=0.03, hspace=0.03)
    ax_main   = fig.add_subplot(gs[0, 1])
    ax_left   = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_bottom = fig.add_subplot(gs[1, 1])
    ax_corner = fig.add_subplot(gs[1, 0]); ax_corner.axis("off")

    # ── Main heatmap — masked cells shown in gray ─────────────────────────
    cmap_base = matplotlib.colormaps["RdBu_r"].copy()
    cmap_base.set_bad(color="lightgray")   # NaN = not significant → gray
    display = np.where(sig_mask, r_matrix, np.nan)
    
    # ---> BUGFIX 2: Removed .T from display! Now height=n_units, width=n_bins (fixes missing top half)
    im = ax_main.imshow(display, origin="lower", aspect="auto",
                        interpolation="nearest", cmap=cmap_base,
                        vmin=vmin, vmax=vmax)
    ax_main.set_title(title, fontweight="bold")
    ax_main.set_ylabel("Unit")
    ax_main.set_xticks([])
    ax_main.set_yticks(np.arange(n_units))
    ax_main.set_yticklabels(unit_names, fontsize=7)
    ax_main.spines[["top", "right"]].set_visible(False)

    # ── Left panel — mean Pearson r per unit ──────────────────────────────
    unit_idx = np.arange(n_units)
    ax_left.axvline(0, color="black", linewidth=0.8, zorder=0)
    colors = ["C3" if v >= 0 else "C0" for v in mean_by_unit]
    ax_left.hlines(y=unit_idx, xmin=0, xmax=mean_by_unit,
                   color=colors, linewidth=1.1, zorder=1)
    ax_left.scatter(mean_by_unit, unit_idx, color=colors, s=14, zorder=2)
    ax_left.set_xlim(-xlim, xlim)
    ax_left.set_xlabel("Mean\nPearson r", fontsize=8)
    ax_left.tick_params(axis="x", labelsize=7)
    ax_left.tick_params(axis="y", left=False, labelleft=False)
    ax_left.spines[["top", "right", "left"]].set_visible(False)

    # ── Bottom panel — mean Pearson r per bin ────────────────────────────
    ax_bottom.axhline(0, color="black", linewidth=0.8, zorder=0)
    ax_bottom.plot(bin_times, mean_by_bin, color="C0", linewidth=1.5)
    ax_bottom.fill_between(bin_times, 0, mean_by_bin, color="C0", alpha=0.20)
    ax_bottom.set_xlabel("Time")
    ax_bottom.set_ylabel("Mean\nPearson r", fontsize=8)
    ax_bottom.tick_params(axis="both", labelsize=7)
    ax_bottom.spines[["top", "right"]].set_visible(False)

    cbar = fig.colorbar(im, ax=ax_main, fraction=0.025, pad=0.02)
    cbar.set_label("Pearson r  (SHAP vs. feature)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    os.makedirs(file_path, exist_ok=True)
    full_path = os.path.join(file_path, f"{file_name}.png")
    fig.savefig(full_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    if show:
        plt.show()
    plt.close(fig)
    return full_path

def _compute_pearson_r_matrix(shap_values, feature_vals, var_idx, p_threshold=0.05):
    """
    Compute Pearson r between feature values and SHAP values at each (unit, bin).

    Parameters
    ----------
    shap_values  : (n_trials, n_vars, n_bins, n_units)
    feature_vals : (n_trials,)  — actual feature values for var_idx across trials
    var_idx      : int          — which variable column to evaluate
    p_threshold  : float        — significance cutoff (default 0.05)

    Returns
    -------
    r_matrix : (n_units, n_bins)
    sig_mask : (n_units, n_bins) bool
    """
    n_trials, n_vars, n_bins, n_units = shap_values.shape
    r_matrix = np.zeros((n_units, n_bins), dtype=float)
    sig_mask  = np.zeros((n_units, n_bins), dtype=bool)

    x = np.asarray(feature_vals, dtype=float)
    if np.std(x) < 1e-8:
        # Feature has no variance — Pearson r undefined everywhere
        return r_matrix, sig_mask

    for u in range(n_units):
        for b in range(n_bins):
            s = shap_values[:, var_idx, b, u]
            if np.std(s) > 1e-8:
                r, p = _pearsonr(x, s)
                r_matrix[u, b] = r
                sig_mask[u, b]  = p <= p_threshold

    return r_matrix, sig_mask

def _get_feature_vals_from_explain(all_explain_data, var_idx, n_pos, n_den):
    """
    Reconstruct per-trial feature values for a given var_idx from loaded .npz data.
    Variable ordering in SHAP tensor:
        0 … n_pos-1        → position vars   (explain_pos[:, bin, pos_col])
        n_pos … n_pos+n_den-1 → dense task vars (explain_den[:, den_col])
        n_pos+n_den …      → sparse task vars (explain_spa[:, spa_col])
    Position vars are per-bin; we average across bins as the trial-level value.
    """
    all_vals = []
    for d in all_explain_data:
        pos = d['pos']   # (n_trials, n_bins, n_pos)
        den = d['den']   # (n_trials, n_den)
        spa = d['spa']   # (n_trials, n_spa)

        if var_idx < n_pos:
            # Position variable — average across time bins to get one value per trial
            vals = pos[:, :, var_idx].mean(axis=1)          # (n_trials,)
        elif var_idx < n_pos + n_den:
            col = var_idx - n_pos
            vals = den[:, col]                               # (n_trials,)
        else:
            col = var_idx - n_pos - n_den
            vals = (spa[:, col] > 0).astype(float)          # (n_trials,)
        all_vals.append(vals)

    return np.concatenate(all_vals, axis=0)

def _get_glm_weight_matrix(betas, var_local_col, n_units, n_bins):
    """
    Extract GLM weight for a specific variable from betas (N, K, V+1).
    col 0 = intercept, col 1 = first dense var, etc.

    Parameters
    ----------
    betas         : (n_units, n_bins, V+1)
    var_local_col : int — 1-based column index in design matrix (1 = first dense)

    Returns
    -------
    weight_matrix : (n_units, n_bins)
    """
    return betas[:, :, var_local_col]

def _get_fa_weight_matrix(fa_components, n_units, n_bins):
    """
    Compute per-unit FA loading magnitude from FactorAnalysis.components_ (n_components, n_units).
    Broadcast across bins (Step 5 has no bin-specific FA weights).

    Returns
    -------
    weight_matrix : (n_units, n_bins)
    """
    unit_loading = np.linalg.norm(fa_components, axis=0)   # (n_units,)
    return np.tile(unit_loading[:, np.newaxis], (1, n_bins))

def plot_pearson_shap_results(
    Conf,
    condition='step3',
    model_label='Conditional / Identity',
    n_splits=20,
    variable_names=None,
    betas_step3=None,
    betas_step4=None,
    fa_components=None,
    p_threshold=0.05,
    scaling_factor=1.0,
    target_vars=None,
    Y_full=None,
    ground_truth_betas=None,
    ground_truth_var_names=None,
    ):
    """
    For each generative variable in the given condition, compute and save:
    1. Pearson r 2D heatmap (Feature vs SHAP)
    2. Spatiotemporal Global Mean SHAP
    3. Ground Truth (GLM Betas for synthetic, Empirical Correlation (Feature vs Firing Rate) for real)
    """
    import matplotlib
    matplotlib.use('Agg')
    print(f"\n{'='*40}\nPearson SHAP, Global Mean, and Ground Truth Matrix for {condition} - {model_label}\n{'='*40}")

    all_shap_values = []
    all_explain_data = []
    all_y_true = []

    # Fetch Y_full globally if not passed
    if Y_full is None and 'conditions' in globals():
        Y_full = globals()['conditions'].get(condition, None)

    # Determine default Ground Truth betas if not explicitly passed
    gt_betas = ground_truth_betas
    gt_vars = ground_truth_var_names
    
    if gt_betas is None:
        if condition in ['step3', 'step5'] and betas_step3 is not None:
            gt_betas = betas_step3
            gt_vars = ['tslp', 'rew'] if gt_vars is None else gt_vars
        elif condition == 'step4' and betas_step4 is not None:
            gt_betas = betas_step4
            gt_vars = ['tslp', 'rew', 'tunp'] if gt_vars is None else gt_vars

    from sklearn.model_selection import KFold
    import numpy as np
    import os
    kf = KFold(n_splits=n_splits, shuffle=False)
    y_splits = list(kf.split(np.arange(Y_full.shape[0]))) if Y_full is not None else [None]*n_splits

    RESULTS = globals().get('RESULTS', './results/')
    FIGURES = globals().get('FIGURES', './figures/')
    
    active_var_names = None
    for fold_idx in range(n_splits):
        path = f"{RESULTS}shap_values_{condition}_{model_label.replace(' / ', '_')}_fold{fold_idx}.npz"
        if os.path.exists(path):
            data = np.load(path)
            all_shap_values.append(data['shap_values'])
            all_explain_data.append({
                'pos': data['explain_pos'],
                'den': data['explain_den'],
                'spa': data['explain_spa']
            })
            if 'active_var_names' in data:
                active_var_names = data['active_var_names'].tolist()
            if Y_full is not None:
                train_idx, valid_idx = y_splits[fold_idx]
                all_y_true.append(Y_full[valid_idx])

    if not all_shap_values:
        print("No SHAP files found."); return

    shap_values = np.concatenate(all_shap_values, axis=0) * scaling_factor
    n_trials, n_vars, n_bins, n_units = shap_values.shape
    
    Y_true_concat = np.concatenate(all_y_true, axis=0) if all_y_true else None

    if all_explain_data:
        n_pos = all_explain_data[0]['pos'].shape[-1]
        n_den = all_explain_data[0]['den'].shape[-1]
    else:
        n_pos = Conf.data.n_position_vars
        n_den = Conf.data.n_dense_vars

    feat_names = []
    if active_var_names is not None:
        feat_names = [str(v).strip("['\"] ") for v in active_var_names]
    elif variable_names is not None:
        for v in variable_names:
            if hasattr(v, 'item'):
                try: v = v.item()
                except: pass
            if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                v = v[0]
            feat_names.append(str(v).strip("['\"] "))
    if len(feat_names) < n_vars:
        missing = n_vars - len(feat_names)
        feat_names += (['Density', 'Sparsity'] if missing == 2
                       else [f"Feature {i}" for i in range(len(feat_names), n_vars)])
    elif len(feat_names) > n_vars:
        feat_names = feat_names[:n_vars]

    if target_vars is not None:
        target_indices = [i for i, nm in enumerate(feat_names)
                          if any(tv == nm for tv in target_vars)]
        if not target_indices:
            print(f"Warning: None of {target_vars} matched feat_names. Using all.")
            target_indices = list(range(n_vars))
    else:
        target_indices = list(range(n_vars))

    bt = globals().get('bin_times', np.arange(n_bins))
    un = globals().get('unit_names', np.arange(n_units))

    save_dir_base = os.path.join(FIGURES, "SHAP_Plots", condition, model_label.replace(' / ', '_'))
    dir_pearson = os.path.join(save_dir_base, "pearson_corr")
    dir_mean_shap = os.path.join(save_dir_base, "mean_shap")
    dir_reference = os.path.join(save_dir_base, "ground_truth_reference")
    
    for d in [dir_pearson, dir_mean_shap, dir_reference]:
        os.makedirs(d, exist_ok=True)

    for var_idx in target_indices:
        var_name = feat_names[var_idx]
        print(f"  Computing metrics for '{var_name}' (var_idx={var_idx})...")

        feature_vals = _get_feature_vals_from_explain(all_explain_data, var_idx, n_pos, n_den)
        r_matrix, sig_mask = _compute_pearson_r_matrix(shap_values, feature_vals, var_idx, p_threshold)

        plot_shap_pearson(
            r_matrix=r_matrix, sig_mask=sig_mask,
            title=f"Pearson r (Feature vs SHAP): {var_name}",
            file_name=f"pearson_r_{var_name.replace(' ', '_')}",
            file_path=dir_pearson, bin_times=bt, unit_names=un, show=False,
        )

        shap_mean = np.nanmean(shap_values[:, var_idx, :, :], axis=0).T 
        all_true_mask = np.ones_like(shap_mean, dtype=bool)
        
        plot_shap_pearson(
            r_matrix=shap_mean, sig_mask=all_true_mask,
            title=f"Spatiotemporal Mean SHAP: {var_name}",
            file_name=f"mean_shap_{var_name.replace(' ', '_')}",
            file_path=dir_mean_shap, bin_times=bt, unit_names=un, show=False,
        )

        if gt_betas is not None and gt_vars is not None:
            # We have ground truth GLM weights
            gt_val = np.zeros((n_units, n_bins))
            
            # Find which index in gt_betas this variable corresponds to
            gt_idx = None
            for i, gt_var in enumerate(gt_vars):
                if gt_var == var_name:
                    gt_idx = i
                    break
            
            if gt_idx is not None:
                # Shape is (N, K, V+1). Transpose to (N, K)
                if gt_idx + 1 < gt_betas.shape[-1]:
                    gt_val = gt_betas[:, :, gt_idx + 1]
                else:
                    print(f"Warning: GT beta shape {gt_betas.shape} doesn't match expected index {gt_idx}")
            
            plot_shap_pearson(
                r_matrix=gt_val, sig_mask=np.ones_like(gt_val, dtype=bool),
                title=f"Ground Truth GLM Betas: {var_name}",
                file_name=f"ground_truth_{var_name.replace(' ', '_')}",
                file_path=dir_reference, bin_times=bt, unit_names=un, show=False,
            )

        elif Y_true_concat is not None:
            # We don't have synthetic ground truth, so compute Empirical Correlation of Counts vs Features
            from scipy.stats import pearsonr as _pearsonr
            r_empirical = np.zeros((n_units, n_bins))
            
            x_feat = np.asarray(feature_vals, dtype=float)
            if np.std(x_feat) > 1e-8:
                for n in range(n_units):
                    for b in range(n_bins):
                        if Y_true_concat.shape[1] == n_units and Y_true_concat.shape[2] == n_bins:
                            y_nb = Y_true_concat[:, n, b]
                        else:
                            y_nb = Y_true_concat[:, b, n]
                        
                        if np.std(y_nb) > 1e-8:
                            r_val, _ = _pearsonr(x_feat, y_nb)
                            r_empirical[n, b] = r_val
            
            # UNMASKED - do not threshold by p-value, so it looks like the GLM betas heatmap
            plot_shap_pearson(
                r_matrix=r_empirical, sig_mask=np.ones_like(r_empirical, dtype=bool),
                title=f"Empirical r (Feature vs Real Firing Rate): {var_name}",
                file_name=f"ground_truth_{var_name.replace(' ', '_')}",
                file_path=dir_reference, bin_times=bt, unit_names=un, show=False,
            )

    print(f"\nAll plots saved to: {save_dir_base}")
