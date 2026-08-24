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

def plot_training_history(
    df_hist,
    title,
    mode="mean",  # Can be "mean" or "best"
    file_name=None,
    file_path=None,
    show=False,
):
    if df_hist is None or df_hist.empty:
        return

    if mode == "best":
        # Find the fold with the lowest final validation loss
        best_fold = None
        best_val = float('inf')
        for fold, group in df_hist.groupby('fold'):
            final_val = group['valid_loss_epoch'].dropna().iloc[-1] if not group['valid_loss_epoch'].dropna().empty else float('inf')
            if final_val < best_val:
                best_val = final_val
                best_fold = fold
        
        # Filter dataframe to only the best fold
        df_agg = df_hist[df_hist['fold'] == best_fold].copy()
        df_agg = df_agg.sort_values('epoch')
        if 'learning_rate_step' in df_agg.columns and 'learning_rate' not in df_agg.columns:
            df_agg['learning_rate'] = df_agg['learning_rate_step']
            
    else:
        # Default: Average across folds and seeds
        df_agg = df_hist.groupby('epoch', as_index=False)[['train_loss_epoch', 'valid_loss_epoch']].mean()
        if 'learning_rate' in df_hist.columns:
            df_lr = df_hist.groupby('epoch', as_index=False)[['learning_rate']].mean()
            df_agg['learning_rate'] = df_lr['learning_rate']
        elif 'learning_rate_step' in df_hist.columns:
            df_lr = df_hist.groupby('epoch', as_index=False)[['learning_rate_step']].mean()
            df_agg['learning_rate'] = df_lr['learning_rate_step']
        
    train = df_agg['train_loss_epoch'].values
    valid = df_agg['valid_loss_epoch'].values
    lr = df_agg['learning_rate'].values if 'learning_rate' in df_agg.columns else np.zeros_like(train)
    epochs = df_agg['epoch'].values

    trend = lambda x: np.diff(x) / np.maximum(np.abs(x[:-1]), 1e-8)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")

    ax[0].plot(epochs, valid, color="tab:orange", label="Validation")
    ax[0].plot(epochs, train, color="tab:blue", label="Train")
    ax[0].set(title="Loss", xlabel="Epoch", ylabel="NLL loss")
    ax[0].legend(frameon=False)

    ax[1].axhline(0, color="black", ls="--", lw=1)
    if len(epochs) > 1:
        ax[1].plot(epochs[1:], trend(valid), color="tab:orange", label="Validation")
        ax[1].plot(epochs[1:], trend(train), color="tab:blue", label="Train")
    ax[1].set(title="Loss trend", xlabel="Epoch", ylabel="Relative change")
    ax[1].legend(frameon=False)

    ax[2].plot(epochs, lr, color="tab:green")
    ax[2].set(title="Learning rate", xlabel="Epoch", ylabel="LR")
    ax[2].set_yscale("log")

    fig.suptitle(title, fontweight="bold")

    if show:
        plt.show()

    if file_name and file_path:
        return save_figure(fig, file_name, file_path)
    return fig

def plot_cov_loading_matrix(
    mean_cov_lit_model,
    title,
    file_name,
    file_path,
    bin_times=None,
    vmin=None,
    vmax=None,
    show=False,
):

    lambda_matrix = mean_cov_lit_model.full_model.cov_model.lambda_matrix
    lambda_matrix = lambda_matrix.detach().cpu().numpy()

    n_bins, n_units, n_latent = lambda_matrix.shape

    if bin_times is None:
        bin_times = np.arange(n_bins)

    loading_matrix = lambda_matrix.transpose(2, 0, 1).reshape(n_latent, -1)

    if vmin is None and vmax is None:
        vmax = np.max(np.abs(loading_matrix))
        vmin = -vmax
    elif vmin is None:
        vmin = -vmax
    elif vmax is None:
        vmax = -vmin

    fig, ax = plt.subplots(
        figsize=(min(20, max(12, n_bins * 0.8)), min(10, max(4, n_latent * 0.35))),
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

    for bin_idx in range(1, n_bins):
        ax.axvline(
            bin_idx * n_units - 0.5,
            color="black",
            linewidth=0.7,
            alpha=0.7,
        )

    bin_centers = np.arange(n_bins) * n_units + (n_units - 1) / 2

    ax.set(
        xlabel="Time bin",
        ylabel="Latent",
        xticks=bin_centers,
        xticklabels=np.round(bin_times, 2),
        yticks=np.arange(n_latent),
        yticklabels=np.arange(n_latent),
    )

    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label(r"$\lambda$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(title, fontweight="bold")

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

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    im = ax.imshow(
        noise.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0,
        vmax=vmax,
    )

    ax.set(
        xlabel="Time",
        ylabel="Unit",
        xticks=np.arange(n_bins),
        xticklabels=np.round(bin_times, 2),
        yticks=np.arange(n_units),
        yticklabels=unit_names,
    )

    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Noise variance", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(title, fontweight="bold")

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)

def plot_cov_length_scales(
    mean_cov_lit_model,
    title,
    file_name,
    file_path,
    ymax=None,
    show=False,
):

    length_scales = mean_cov_lit_model.full_model.cov_model.length_scales
    length_scales = F.softplus(length_scales).detach().cpu().numpy()

    if ymax is None:
        ymax = np.max(length_scales)

    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")

    ax.bar(np.arange(len(length_scales)), length_scales, color="C0")

    ax.set(
        xlabel="Latent",
        ylabel="Length scale",
        xticks=np.arange(len(length_scales)),
        ylim=(0, ymax),
    )

    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontweight="bold")

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
    vmin=None,
    vmax=None,
    linthresh=None,
    show=False,
):
    lambda_matrix = mean_cov_lit_model.full_model.cov_model.lambda_matrix.unsqueeze(0)
    cov_tensor = mean_cov_lit_model.full_model.cov_model.build_covariance_matrix(lambda_matrix)
    cov_matrix = (cov_tensor @ cov_tensor.transpose(-1, -2)).squeeze(0).detach().cpu().numpy()
    
    n_bins, n_units = mean_cov_lit_model.full_model.cov_model.n_bins, mean_cov_lit_model.full_model.cov_model.n_units
    
    if bin_times is None:
        bin_times = np.arange(n_bins)

    if time_indices is None:
        time_indices = np.arange(n_bins)
    else:
        time_indices = np.atleast_1d(time_indices)

    feature_indices = np.concatenate([
        np.arange(time_idx * n_units, (time_idx + 1) * n_units)
        for time_idx in time_indices
    ])

    cov_matrix = cov_matrix[np.ix_(feature_indices, feature_indices)]
    bin_times = np.asarray(bin_times)[time_indices]
    n_bins = len(time_indices)
        
    if vmin is None and vmax is None:
        vmax = np.max(np.abs(cov_matrix))
        vmin = -vmax
    elif vmin is None:
        vmin = -vmax
    elif vmax is None:
        vmax = -vmin

    if linthresh is None:
        linthresh = max(vmax * 0.01, 1e-8)

    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)

    fig, ax = plt.subplots(figsize=(max(10, n_bins * 0.5), max(10, n_bins * 0.5)), layout="constrained")

    im = ax.imshow(
        cov_matrix,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=norm,
    )

    for bin_idx in range(1, n_bins):
        line_pos = bin_idx * n_units - 0.5
        ax.axvline(line_pos, color="black", linewidth=0.5, alpha=0.5)
        ax.axhline(line_pos, color="black", linewidth=0.5, alpha=0.5)

    bin_centers = np.arange(n_bins) * n_units + (n_units - 1) / 2
    tick_labels = np.round(bin_times, 2)

    ax.set(
        xticks=bin_centers,
        xticklabels=tick_labels,
        yticks=bin_centers,
        yticklabels=tick_labels,
    )
    
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Covariance (SymLog Scale)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(title, fontweight="bold")

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)

def plot_prediction(
    Y,
    Y_hat,
    trial_idx,
    unit_idx,
    bin_times,
    title,
    filename,
    filepath,
    figsize=(10, 2),
    show=False,
):
    to_numpy = lambda x: (
        x.detach().cpu().numpy()
        if torch.is_tensor(x)
        else np.asarray(x)
    )

    y = to_numpy(Y[trial_idx])[:, unit_idx]
    y_hat = to_numpy(Y_hat[trial_idx])[:, unit_idx]
    bin_times = np.asarray(bin_times)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    ax.plot(
        bin_times,
        y,
        color="tab:green",
        linewidth=2,
        label="Ground truth",
    )
    ax.plot(
        bin_times,
        y_hat,
        color="tab:purple",
        linewidth=2,
        label="GRU prediction",
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Time from press onset (ms)",
        ylabel="Neural activity",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontweight="bold")

    if show:
        plt.show()

    return save_figure(fig, filename, filepath)

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
    bins=20,
    figsize=(14, 7),
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

    fig, axes = plt.subplots(2, 3, figsize=figsize, layout="constrained")

    for idx, (metric_name, train, valid, color) in enumerate(metrics):
        finite = np.isfinite(train) & np.isfinite(valid)
        train = train[finite]
        valid = valid[finite]

        lower = min(train.min(), valid.min())
        upper = max(train.max(), valid.max())
        padding = max(0.05 * (upper - lower), 1e-8)
        limits = (lower - padding, upper + padding)

        ax = axes[0, idx]

        ax.scatter(
            train,
            valid,
            s=32,
            alpha=0.7,
            color=color,
            edgecolor="none",
        )
        ax.plot(limits, limits, color="black", linestyle="--", linewidth=1)
        ax.set(
            title=metric_name,
            xlabel=f"Train {metric_name}",
            ylabel=f"Validation {metric_name}",
            xlim=limits,
            ylim=limits,
            aspect="equal",
        )
        ax.spines[["top", "right"]].set_visible(False)

        difference = train - valid
        limit = max(np.max(np.abs(difference)), 1e-8)
        edges = np.linspace(-limit, limit, bins + 1)

        ax = axes[1, idx]

        ax.hist(
            difference,
            bins=edges,
            color=color,
            alpha=0.7,
            edgecolor="white",
        )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set(
            xlabel=f"Train - validation {metric_name}",
            ylabel="Number of units",
            xlim=(-limit, limit),
        )
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontweight="bold")

    if show:
        plt.show()

    return save_figure(fig, filename, filepath)

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
    title,
    file_name,
    file_path,
    bins=20,
    figsize=(14, 7),
    show=False,
):
    metrics = [
        (
            "Correlation",
            correlations_x_train,
            correlations_x_valid,
            correlations_y_train,
            correlations_y_valid,
        ),
        (
            r"$R^2$",
            r2s_x_train,
            r2s_x_valid,
            r2s_y_train,
            r2s_y_valid,
        ),
        (
            "MSE",
            mses_x_train,
            mses_x_valid,
            mses_y_train,
            mses_y_valid,
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize, layout="constrained")

    for idx, (metric_name, x_train, x_valid, y_train, y_valid) in enumerate(metrics):
        x_train = np.asarray(x_train, dtype=float)
        x_valid = np.asarray(x_valid, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        y_valid = np.asarray(y_valid, dtype=float)

        finite_train = np.isfinite(x_train) & np.isfinite(y_train)
        finite_valid = np.isfinite(x_valid) & np.isfinite(y_valid)

        x_train, y_train = x_train[finite_train], y_train[finite_train]
        x_valid, y_valid = x_valid[finite_valid], y_valid[finite_valid]

        lower = min(x_train.min(), x_valid.min(), y_train.min(), y_valid.min())
        upper = max(x_train.max(), x_valid.max(), y_train.max(), y_valid.max())
        padding = max(0.05 * (upper - lower), 1e-8)
        limits = (lower - padding, upper + padding)

        ax = axes[0, idx]

        ax.scatter(
            x_train,
            y_train,
            s=32,
            alpha=0.7,
            color="tab:blue",
            edgecolor="none",
            label="Training",
        )
        ax.scatter(
            x_valid,
            y_valid,
            s=32,
            alpha=0.7,
            color="tab:orange",
            edgecolor="none",
            label="Validation",
        )
        ax.plot(limits, limits, color="gray", linestyle="--", linewidth=1.5)
        ax.set(
            title=metric_name,
            xlabel=f"{x_label} {metric_name}",
            ylabel=f"{y_label} {metric_name}",
            xlim=limits,
            ylim=limits,
            aspect="equal",
        )
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

        difference_train = x_train - y_train
        difference_valid = x_valid - y_valid
        limit = max(
            np.max(np.abs(difference_train)),
            np.max(np.abs(difference_valid)),
            1e-8,
        )
        edges = np.linspace(-limit, limit, bins + 1)

        ax = axes[1, idx]

        ax.hist(
            difference_train,
            bins=edges,
            color="tab:blue",
            alpha=0.6,
            edgecolor="white",
            label="Training",
        )
        ax.hist(
            difference_valid,
            bins=edges,
            color="tab:orange",
            alpha=0.6,
            edgecolor="white",
            label="Validation",
        )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.5)
        ax.set(
            xlabel=f"{x_label} - {y_label} {metric_name}",
            ylabel="Number of units",
            xlim=(-limit, limit),
        )
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontweight="bold")

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
        show=False
):

    
    n_units = len(mse_trial_baseline)
    
    p_values_mse, p_values_corr, p_values_r2 = [], [], []
    better_mse, better_corr, better_r2 = [], [], []

    for unit_idx in range(n_units):
        mse_b = mse_trial_baseline[unit_idx]
        mse_m = mse_trial_model[unit_idx]
        corr_b = corr_trial_baseline[unit_idx]
        corr_m = corr_trial_model[unit_idx]
        r2_b = r2_trial_baseline[unit_idx]
        r2_m = r2_trial_model[unit_idx]
        
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

    def get_proportions(p_vals, is_better):
        sig = sum(1 for p, better in zip(p_vals, is_better) if p < alpha and better)
        total = len(p_vals)
        return sig / total, (total - sig) / total

    sig_mse, nonsig_mse = get_proportions(p_values_mse, better_mse)
    sig_corr, nonsig_corr = get_proportions(p_values_corr, better_corr)
    sig_r2, nonsig_r2 = get_proportions(p_values_r2, better_r2)
    
    metrics = ['Loss', 'Correlation', 'R²']
    significant = [sig_mse, sig_corr, sig_r2]
    non_significant = [nonsig_mse, nonsig_corr, nonsig_r2]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    p1 = ax.bar(metrics, significant, color='#4558C4', label='Significant')
    p2 = ax.bar(metrics, non_significant, bottom=significant, color='#C42A2F', label='Non-significant')
    
    ax.bar_label(p1, label_type='center', fmt='%.2f', fontsize=12)
    ax.bar_label(p2, label_type='center', fmt='%.2f', fontsize=12)
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Percentage', fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='grey', linewidth=1)
    ax.set_axisbelow(True)
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, frameon=False, fontsize=14)    
    fig.suptitle(title, fontweight='bold')
    
    if show:
        plt.show()
        
    return save_figure(fig, file_name, file_path)

def plot_unit_selectivity_curves(
    unit_selectivities,
    class_names,
    title,
    filename,
    file_path,
    show=False
):
    colors = ["gray", "green", "red", "blue"]

    fig, axes = plt.subplots(
        1,
        unit_selectivities.shape[0],
        figsize=(4 * unit_selectivities.shape[0], 4),
        layout="constrained"
    )

    axes = np.atleast_1d(axes)

    for ax, color, unit_selectivity, class_name in zip(
        axes,
        colors,
        unit_selectivities,
        class_names
    ):
        sorted_unit_selectivity = np.sort(unit_selectivity)

        ax.plot(
            np.arange(1, len(sorted_unit_selectivity) + 1),
            sorted_unit_selectivity,
            color=color,
            linewidth=2.5
        )

        ax.set_title(class_name.replace("_", " ").title())
        ax.set_xlabel("Units")
        ax.set_xlim(1, len(sorted_unit_selectivity))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Mean absolute SHAP")
    fig.suptitle(title, fontweight="bold")

    if show:
        plt.show()

    return save_figure(fig, filename, file_path)

def plot_unit_class_selectivity_matrix(
    unit_selectivities,
    class_names,
    unit_names,
    title,
    filename,
    file_path,
    show=False
):

    base_colors = ["gray", "green", "red", "blue"][:unit_selectivities.shape[0]]

    unit_scores = np.nanmean(unit_selectivities, axis=0)
    sort_idx = np.argsort(-unit_scores)

    sorted_matrix = unit_selectivities[:, sort_idx]
    sorted_unit_names = np.array(list(unit_names))[sort_idx]

    vmin = np.nanmin(sorted_matrix)
    vmax = np.nanmax(sorted_matrix)

    n_classes, n_units = sorted_matrix.shape

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(max(8, 0.15 * n_units), 4),
        layout="constrained"
    )

    for row_idx, color in enumerate(base_colors):
        row_values = sorted_matrix[row_idx : row_idx + 1, :]

        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"class_{row_idx}",
            ["white", color],
            N=256
        )

        ax.imshow(
            row_values,
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            extent=(
                -0.5,
                n_units - 0.5,
                row_idx - 0.5,
                row_idx + 0.5
            )
        )

    ax.set_xticks(np.arange(n_units))
    ax.set_xticklabels(sorted_unit_names, rotation=90, fontsize=7)

    ax.set_yticks(np.arange(n_classes))
    ax.set_yticklabels(class_names, fontsize=9)

    ax.set_xlim(-0.5, n_units - 0.5)
    ax.set_ylim(-0.5, n_classes - 0.5)

    ax.set_xlabel("Units")
    ax.set_ylabel("Classes")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontweight="bold")

    if show:
        plt.show()

    return save_figure(fig, filename, file_path)

def plot_empirical_covariance_matrix(data, title, mode='unit', units=None, file_name=None, file_path=None, show=True):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    import os
    
    T, N, K = data.shape
    
    # Filter units if specified
    if units is not None:
        if isinstance(units, int):
            units = [units]
        data = data[:, units, :]
        N = data.shape[1]
        
    if mode == 'time':
        flattened_data = data.transpose(0, 1, 2).reshape(T * N, K)
        cov_matrix = np.cov(flattened_data, rowvar=False)
        label_str = "Bin Time"
        ticks = range(K)
    elif mode == 'unit':
        flattened_data = data.transpose(0, 2, 1).reshape(T * K, N)
        cov_matrix = np.cov(flattened_data, rowvar=False)
        label_str = "Unit"
        ticks = units if units is not None else range(N)
    else:
        raise ValueError("mode must be 'time' or 'unit'")
        
    vmax = np.max(np.abs(cov_matrix))
    vmin = -vmax
    linthresh = max(vmax * 0.01, 1e-8)
    
    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)
    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    
    im = ax.imshow(
        cov_matrix,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=norm,
    )
    
    # Only draw ticks if not too many
    if len(ticks) <= 50:
        ax.set_xticks(range(len(ticks)))
        ax.set_yticks(range(len(ticks)))
        ax.set_xticklabels(ticks, rotation=90 if mode=='unit' else 0, fontsize=8)
        ax.set_yticklabels(ticks, fontsize=8)
        
        for i in range(1, len(ticks)):
            ax.axhline(i - 0.5, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
            ax.axvline(i - 0.5, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
            
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Covariance (SymLog Scale)")
    ax.set_xlabel(label_str)
    ax.set_ylabel(label_str)
    fig.suptitle(title, fontweight="bold")
    
    if show:
        plt.show()
    
    if file_name and file_path:
        os.makedirs(file_path, exist_ok=True)
        full_path = os.path.join(file_path, f"{file_name}.png")
        fig.savefig(full_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        return full_path
    return fig

def plot_residual_covariance(Y_synth, lambda_hat, title, mode='unit', units=None, time_bin=None, bin_times=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    T, N, K = Y_synth.shape
    eps = 1e-8
    residuals = (Y_synth - lambda_hat) / np.sqrt(lambda_hat + eps)
    
    if units is not None:
        if isinstance(units, int):
            units = [units]
        residuals = residuals[:, units, :]
        N = residuals.shape[1]

    if mode == 'time':
        residuals_flat = residuals.transpose(0, 1, 2).reshape(T * N, K)
        cov_matrix = np.cov(residuals_flat, rowvar=False)
        label_str = "Bin Time"
        ticks = range(K)
        tick_labels = ticks
        
    elif mode == 'unit':
        if time_bin is not None:
            residuals_flat = residuals[:, :, time_bin] # Shape: (T, N)
            label_str = "Unit"
            ticks = [N // 2]
            tick_labels = [f"{bin_times[time_bin]:.2f}"] if bin_times is not None else [f"Bin {time_bin}"]
        else:
            residuals_flat = residuals.transpose(0, 2, 1).reshape(T * K, N) # Shape: (T*K, N)
            label_str = "Unit"
            ticks = units if units is not None else range(N)
            tick_labels = ticks
            
        cov_matrix = np.cov(residuals_flat, rowvar=False)
    else:
        raise ValueError("mode must be 'time' or 'unit'")
    
    vmax = np.max(np.abs(cov_matrix))
    vmin = -vmax
    linthresh = max(vmax * 0.01, 1e-8)
    
    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)
    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    im = ax.imshow(cov_matrix, origin="lower", aspect="equal", interpolation="nearest", cmap="RdBu_r", norm=norm)
    
    if len(ticks) <= 50:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=90 if mode=='time' else 0, fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)
        
        # Grid lines for better visualization
        ax.set_xticks(np.arange(-0.5, cov_matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, cov_matrix.shape[0], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.2, alpha=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlabel(label_str, fontsize=12)
    ax.set_ylabel(label_str, fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Residual Covariance (SymLog Scale)', fontsize=12)
    
    plt.show()
