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

def plot_learning_curve(
    metric_history,
    step_history,
    initial_train_loss: float = None,
    initial_valid_loss: float = None,
    title: str = None,
    smooth_window: int = 5,
    save_path: str = None,
):
    """Two-panel learning curve for one model/fold.

    Top panel  : per-STEP train loss (raw + smoothed).
    Bottom panel: per-EPOCH train + valid loss, with initial random-weight
                  loss shown as horizontal dashed lines.

    Parameters
    ----------
    metric_history : dict  (metric_cb.history)
    step_history   : StepHistory callback instance (step_cb)
    initial_train_loss, initial_valid_loss : float, from compute_initial_loss()
    """
    import pandas as _pd

    # ── epoch data ──────────────────────────────────────────────────────────
    if hasattr(metric_history, "to_dict"):
        metric_history = metric_history.sort_values("epoch").to_dict("list")

    epochs      = np.asarray(metric_history.get("epoch", []), dtype=float)
    train_ep    = np.asarray(metric_history.get("train_loss_epoch", []), dtype=float)
    valid_ep    = np.asarray(metric_history.get("valid_loss_epoch", []), dtype=float)

    # ── step data ───────────────────────────────────────────────────────────
    steps       = np.asarray(step_history.global_steps, dtype=float)
    train_steps = np.asarray(step_history.train_loss_step, dtype=float)

    def _smooth(y, w):
        if w is None or w <= 1 or len(y) < 2:
            return y
        return _pd.Series(y).rolling(window=w, center=True, min_periods=1).mean().to_numpy()

    train_steps_s = _smooth(train_steps, smooth_window)

    TRAIN = "#15803d"   # green
    VALID = "#dc2626"   # red
    INIT  = "#6b21a8"   # purple — initial random-weight level

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained")

    # ── Top: per-step train loss ─────────────────────────────────────────────
    ax = axes[0]
    if len(steps) > 0:
        ax.plot(steps, train_steps, color=TRAIN, alpha=0.18, linewidth=0.8)
        ax.plot(steps, train_steps_s, color=TRAIN, linewidth=2.0, label="Train loss (per step)")
    if initial_train_loss is not None:
        ax.axhline(initial_train_loss, color=INIT, linewidth=1.4, linestyle="--",
                   label=f"Initial (random weights) = {initial_train_loss:.4f}")
    ax.set_xlabel("Global step")
    ax.set_ylabel("NLL loss")
    ax.set_title("Per-step train loss")
    ax.legend(frameon=False, fontsize=9)
    sns.despine(ax=ax)

    # ── Bottom: per-epoch train + valid ──────────────────────────────────────
    ax = axes[1]
    if len(epochs) > 0:
        ax.plot(epochs, train_ep, color=TRAIN, linewidth=2.0, marker=".",
                markersize=4, label="Train loss (per epoch)")
        ax.plot(epochs, valid_ep, color=VALID, linewidth=2.0, marker=".",
                markersize=4, label="Valid loss (per epoch)")
    if initial_train_loss is not None:
        ax.axhline(initial_train_loss, color=INIT, linewidth=1.2, linestyle="--", alpha=0.7,
                   label=f"Initial train = {initial_train_loss:.4f}")
    if initial_valid_loss is not None:
        ax.axhline(initial_valid_loss, color=INIT, linewidth=1.2, linestyle=":", alpha=0.7,
                   label=f"Initial valid = {initial_valid_loss:.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL loss")
    ax.set_title("Per-epoch loss")
    ax.legend(frameon=False, fontsize=9)
    sns.despine(ax=ax)

    if title:
        fig.suptitle(title, fontweight="bold", fontsize=13)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved → {save_path}")
    plt.show()
    return fig

def plot_model_comparison_scatter_by_fold(
    df_units, ref_model, metric='pearson', condition='', save_path=None,
):
    """Per-FOLD scatter: one dot per CV fold (metric mean across units)."""
    MODEL_ORDER = ['Zero / Identity', 'Baseline / Identity',
                   'Conditional / Identity', 'Conditional / Shared']
    metric_label = {'pearson': 'Correlation', 'r2': 'R2', 'nll': 'NLL'}.get(metric, metric)
    per_fold = (
        df_units.groupby(['model_label', 'fold'])
        [[metric, 'train_' + metric]].mean().reset_index()
    )
    ref_va = per_fold[per_fold['model_label'] == ref_model].sort_values('fold')
    others = [m for m in MODEL_ORDER
              if m in per_fold['model_label'].unique() and m != ref_model]
    if not others:
        print('No comparison models found.'); return None
    fig, axes = plt.subplots(1, len(others), figsize=(5.5 * len(others), 5.0), squeeze=False)
    cond_label = ' [' + condition.upper() + ']' if condition else ''
    fig.suptitle(ref_model + ' vs Others  (per fold)' + cond_label,
                 fontsize=14, fontweight='bold')
    for ax, other in zip(axes[0], others):
        other_va = per_fold[per_fold['model_label'] == other].sort_values('fold')
        xv_tr = ref_va['train_' + metric].values
        yv_tr = other_va['train_' + metric].values
        xv_va = ref_va[metric].values
        yv_va = other_va[metric].values
        folds_arr = ref_va['fold'].values
        tr_ok = np.isfinite(xv_tr) & np.isfinite(yv_tr)
        va_ok = np.isfinite(xv_va) & np.isfinite(yv_va)
        all_v = np.concatenate([xv_tr[tr_ok], yv_tr[tr_ok], xv_va[va_ok], yv_va[va_ok]])
        if all_v.size == 0:
            ax.set_title(other + '\n(no finite values)'); continue
        ax.scatter(xv_tr[tr_ok], yv_tr[tr_ok], color='#1f77b4', s=70, alpha=0.85,
                   edgecolor='white', linewidth=0.5, label='Training', zorder=3)
        ax.scatter(xv_va[va_ok], yv_va[va_ok], color='#ff7f0e', s=70, alpha=0.85,
                   edgecolor='white', linewidth=0.5, label='Validation', zorder=2)
        for xi, yi, fi in zip(xv_va[va_ok], yv_va[va_ok], folds_arr[va_ok]):
            ax.annotate(str(int(fi)), (xi, yi), fontsize=7, ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points', color='#555')
        lo, hi = np.nanmin(all_v), np.nanmax(all_v)
        pad = max((hi - lo) * 0.12, 1e-4)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color='gray', linestyle='--', linewidth=1.3, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(ref_model + '\n' + metric_label, fontsize=10)
        ax.set_ylabel(other + '\n' + metric_label, fontsize=10)
        ax.set_title(other, fontsize=11, pad=10)
        ax.legend(loc='lower right', frameon=False, fontsize=8)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print('Saved -> ' + save_path)
    plt.show()
    return fig

def plot_model_comparison_scatter_by_unit(
    df_units, ref_model, metric='pearson', condition='', save_path=None,
):
    """Per-UNIT scatter: one dot per neuron (metric averaged across folds)."""
    MODEL_ORDER = ['Zero / Identity', 'Baseline / Identity',
                   'Conditional / Identity', 'Conditional / Shared']
    metric_label = {'pearson': 'Correlation', 'r2': 'R2', 'nll': 'NLL'}.get(metric, metric)
    
    per_unit = (
        df_units.groupby(['model_label', 'unit'])
        [[metric, 'train_' + metric]].mean().reset_index()
    )
    ref_vals = per_unit[per_unit['model_label'] == ref_model].sort_values('unit')
    others   = [m for m in MODEL_ORDER
                if m in per_unit['model_label'].unique() and m != ref_model]
    
    if not others:
        print('No comparison models found.'); return None
        
    fig, axes = plt.subplots(1, len(others), figsize=(6.0 * len(others), 5.5), squeeze=False)
    cond_label = ' [' + condition.upper() + ']' if condition else ''
    fig.suptitle(ref_model + ' vs Others  (per unit)' + cond_label,
                 fontsize=14, fontweight='bold')
                 
    for col_i, (ax, other) in enumerate(zip(axes[0], others)):
        other_vals = per_unit[per_unit['model_label'] == other].sort_values('unit')
        
        xv_tr = ref_vals['train_' + metric].values
        yv_tr = other_vals['train_' + metric].values
        xv_va = ref_vals[metric].values
        yv_va = other_vals[metric].values
        
        tr_ok = np.isfinite(xv_tr) & np.isfinite(yv_tr)
        va_ok = np.isfinite(xv_va) & np.isfinite(yv_va)
        all_v = np.concatenate([xv_tr[tr_ok], yv_tr[tr_ok], xv_va[va_ok], yv_va[va_ok]])
        
        if all_v.size == 0:
            ax.set_title(other + '\n(no finite values -- skipped)', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            continue
            
        ax.scatter(xv_tr[tr_ok], yv_tr[tr_ok], color='#1f77b4', s=18, alpha=0.85,
                   edgecolors='none', label='Training', zorder=2)
        ax.scatter(xv_va[va_ok], yv_va[va_ok], color='#ff7f0e', s=18, alpha=0.85,
                   edgecolors='none', label='Validation', zorder=3)
                   
        if va_ok.sum() >= 3:
            r_va = float(np.corrcoef(xv_va[va_ok], yv_va[va_ok])[0, 1])
            ax.text(0.05, 0.95, 'r = {:.2f}'.format(r_va), transform=ax.transAxes,
                    fontsize=10, va='top', color='#ff7f0e')
        if tr_ok.sum() >= 3:
            r_tr = float(np.corrcoef(xv_tr[tr_ok], yv_tr[tr_ok])[0, 1])
            ax.text(0.05, 0.88, 'r = {:.2f}'.format(r_tr), transform=ax.transAxes,
                    fontsize=10, va='top', color='#1f77b4')
                    
        lo, hi = np.nanmin(all_v), np.nanmax(all_v)
        pad = max((hi - lo) * 0.12, 1e-4)
        lo, hi = lo - pad, hi + pad
        
        ax.plot([lo, hi], [lo, hi], color='darkgray', linestyle='--', linewidth=1.5, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(ref_model, fontsize=11)
        ax.set_ylabel('Conditioned on ' + other, fontsize=11)
        ax.set_title(metric_label, fontsize=12, pad=12)
        
        n_drop = int((~tr_ok).sum() + (~va_ok).sum())
        if n_drop > 0:
            ax.text(0.02, 0.02, '{} units dropped (NaN)'.format(n_drop),
                    transform=ax.transAxes, fontsize=8, color='gray', va='bottom')
        if col_i == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.83), frameon=True, fontsize=9)
            
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        
        diff_va = np.abs(yv_va[va_ok] - xv_va[va_ok])
        diff_tr = np.abs(yv_tr[tr_ok] - xv_tr[tr_ok])
        all_d = np.concatenate([diff_tr, diff_va])
        all_d = all_d[np.isfinite(all_d)]
                
        if all_d.size >= 1:
            R = hi - lo
            max_abs_d = np.nanmax(np.abs(all_d))
            if max_abs_d < 1e-9:
                max_abs_d = 0.05 * R  
                
            d_bound = max_abs_d * 1.05
            edges = np.linspace(0, d_bound, 25)
            
            counts_va, _ = np.histogram(diff_va, bins=edges, density=True) if diff_va.size > 0 else (np.zeros(len(edges)-1), edges)
            counts_tr, _ = np.histogram(diff_tr, bins=edges, density=True) if diff_tr.size > 0 else (np.zeros(len(edges)-1), edges)
            
            max_density = max(counts_va.max() if counts_va.size else 0,
                              counts_tr.max() if counts_tr.size else 0)
            
            if max_density > 0:
                x_b = hi - 0.12 * R
                y_b = hi - 0.12 * R
                scale = (0.15 * R) / max_density 
                ext = 0.15 * d_bound
                px_start, py_start = x_b - (d_bound+ext)/2, y_b + (d_bound+ext)/2
                px_end, py_end = x_b - (-d_bound-ext)/2, y_b + (-d_bound-ext)/2
                
                ax.plot([px_start, px_end], [py_start, py_end], 
                        color='black', linewidth=1.0, zorder=4, clip_on=False)
                
                def draw_hist_bars(counts, direction, color):
                    for i in range(len(edges)-1):
                        c = counts[i]
                        if c > 0:
                            h = c * scale
                            d1, d2 = edges[i], edges[i+1]
                            
                            bx1, by1 = x_b - d1/2, y_b + d1/2
                            bx2, by2 = x_b - d2/2, y_b + d2/2
                            
                            tx1 = bx1 + direction * h / np.sqrt(2)
                            ty1 = by1 + direction * h / np.sqrt(2)
                            tx2 = bx2 + direction * h / np.sqrt(2)
                            ty2 = by2 + direction * h / np.sqrt(2)
                            
                            ax.fill([bx1, bx2, tx2, tx1], [by1, by2, ty2, ty1],
                                    color=color, alpha=0.55, edgecolor='none', zorder=3, clip_on=False)
                
                draw_hist_bars(counts_tr, 1, '#1f77b4')  
                draw_hist_bars(counts_va, 1, '#ff7f0e') 
                
                if max_abs_d < 1e-9:
                    tick_vals = [0.0]
                else:
                    mag = 10**np.floor(np.log10(max_abs_d))
                    tv = np.round(max_abs_d / mag) * mag
                    tick_vals = [0.0, tv]
                    
                tick_len = 0.012 * R
                tv_x, tv_y = tick_len / np.sqrt(2), tick_len / np.sqrt(2)
                
                for tv_val in tick_vals:
                    if abs(tv_val) > d_bound + 1e-9: continue
                    bx, by = x_b - tv_val/2, y_b + tv_val/2
                    ax.plot([bx - tv_x/2, bx + tv_x/2], [by - tv_y/2, by + tv_y/2], 
                            color='black', lw=1.0, zorder=4, clip_on=False)
                    
                    lbl = '0' if abs(tv_val) < 1e-9 else f'{tv_val:g}'
                    if tv_val > 0:   
                        ax.text(bx - tv_x*0.6, by + tv_y*0.6, lbl, fontsize=9,
                                ha='right', va='bottom', zorder=5, clip_on=False)
                    elif tv_val < 0: 
                        ax.text(bx + tv_x*0.6, by - tv_y*0.6, lbl, fontsize=9,
                                ha='left', va='top', zorder=5, clip_on=False)
                    else:            
                        ax.text(bx + tv_x*1.2, by - tv_y*0.5, lbl, fontsize=9,
                                ha='left', va='center', zorder=5, clip_on=False)
        
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print('Saved -> ' + save_path)
    plt.show()
    return fig

def plot_loss_curves_per_epoch(
    df_history: pd.DataFrame,
    condition: str,
    folds: list = None,
    max_folds_show: int = 10,
    save_path: str = None,
):
    """Train + validation loss over epochs, one panel per trainable model."""
    set_pub_style()
    df = df_history[df_history["condition"] == condition].copy()
    trainable = df["model"].unique()

    all_folds = sorted(df["fold"].unique())
    if folds is not None:
        show_folds = [f for f in folds if f in all_folds]
    else:
        step = max(1, len(all_folds) // max_folds_show)
        show_folds = all_folds[::step][:max_folds_show]

    n_models = len(trainable)
    if n_models == 0:
        return None
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4), squeeze=False)

    TRAIN_COL = "#15803d"
    VALID_COL = "#dc2626"
    FOLD_ALPHA = 0.18

    for ax, key in zip(axes[0], trainable):
        sub = df[df["model"] == key]

        for fold in show_folds:
            fold_df = sub[sub["fold"] == fold].sort_values("epoch")
            if fold_df.empty:
                continue
            ax.plot(fold_df["epoch"], fold_df["train_loss_epoch"],
                    color=TRAIN_COL, alpha=FOLD_ALPHA, linewidth=0.9)
            ax.plot(fold_df["epoch"], fold_df["valid_loss_epoch"],
                    color=VALID_COL, alpha=FOLD_ALPHA, linewidth=0.9)

        grouped = sub.groupby("epoch")
        med_train = grouped["train_loss_epoch"].median()
        med_valid = grouped["valid_loss_epoch"].median()

        ax.plot(med_train.index, med_train.values,
                color=TRAIN_COL, linewidth=2.2, label="Train (median)")
        ax.plot(med_valid.index, med_valid.values,
                color=VALID_COL, linewidth=2.2, label="Valid (median)")

        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("NLL loss" if ax == axes[0][0] else "")
        ax.set_title(key, fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        sns.despine(ax=ax)

    fig.suptitle(f"[{condition}] Loss curves per epoch", y=1.02,
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved → {save_path}")
    plt.show()
    return fig

def plot_mse_curves_per_epoch(
    df_history: pd.DataFrame,
    condition: str,
    folds: list = None,
    max_folds_show: int = 10,
    save_path: str = None,
):
    """Train + validation MSE over epochs, one panel per trainable model."""
    if "train_mse_epoch" not in df_history.columns:
        print("plot_mse_curves_per_epoch: no MSE columns found in df_history — skipped.")
        return None

    set_pub_style()
    df = df_history[df_history["condition"] == condition].copy()
    trainable = df["model"].unique()

    all_folds = sorted(df["fold"].unique())
    if folds is not None:
        show_folds = [f for f in folds if f in all_folds]
    else:
        step = max(1, len(all_folds) // max_folds_show)
        show_folds = all_folds[::step][:max_folds_show]

    n_models = len(trainable)
    if n_models == 0:
        return None
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4), squeeze=False)

    TRAIN_COL = "#15803d"
    VALID_COL = "#dc2626"
    FOLD_ALPHA = 0.18

    for ax, key in zip(axes[0], trainable):
        sub = df[df["model"] == key]

        for fold in show_folds:
            fold_df = sub[sub["fold"] == fold].sort_values("epoch")
            if fold_df.empty:
                continue
            ax.plot(fold_df["epoch"], fold_df["train_mse_epoch"],
                    color=TRAIN_COL, alpha=FOLD_ALPHA, linewidth=0.9)
            ax.plot(fold_df["epoch"], fold_df["valid_mse_epoch"],
                    color=VALID_COL, alpha=FOLD_ALPHA, linewidth=0.9)

        grouped = sub.groupby("epoch")
        med_train = grouped["train_mse_epoch"].median()
        med_valid = grouped["valid_mse_epoch"].median()

        ax.plot(med_train.index, med_train.values,
                color=TRAIN_COL, linewidth=2.2, label="Train MSE (median)")
        ax.plot(med_valid.index, med_valid.values,
                color=VALID_COL, linewidth=2.2, label="Valid MSE (median)")

        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("MSE" if ax == axes[0][0] else "")
        ax.set_title(key, fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        sns.despine(ax=ax)

    fig.suptitle(f"[{condition}] MSE curves per epoch", y=1.02,
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved → {save_path}")
    plt.show()
    return fig

def plot_step_boxplot(df_step: pd.DataFrame, step_name: str = 'Step', metric: str = 'pearson', save_path: str = None):
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
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_step_scatter_separated(df_step: pd.DataFrame, step_name: str = 'Step', metric: str = 'pearson', save_path: str = None):
    """
    One panel per model, showing a swarm plot of the given step's folds.
    """
    set_pub_style()
    preferred = ['Zero / Identity', 'Baseline / Identity', 'Conditional / Identity', 'Conditional / Shared', 'Baseline / Shared']
    models = [ml for ml in preferred if ml in df_step['model_label'].unique()]
    for m in df_step['model_label'].unique():
        if m not in models:
            models.append(m)
            
    fig, axes = plt.subplots(1, len(models), figsize=(2.5 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]
        
    for ax, ml in zip(axes, models):
        sub = df_step[df_step['model_label'] == ml]
        col = _MODEL_COLORS.get(ml, '#333333')
        
        sns.swarmplot(y=sub[metric], color=col, size=8, ax=ax, alpha=0.8, edgecolor='white', linewidth=0.6)
        
        ax.axhline(0, color='k', ls='--', lw=1, alpha=0.5)
        ax.set_title(ml.replace(' / ', '\n/ '), fontsize=10, fontweight='bold', color=col)
        ax.set_xlabel('Folds')
        if ax == axes[0]:
            ax.set_ylabel(f'{step_name} {metric.capitalize()}')
        else:
            ax.set_ylabel('')
        ax.set_xticks([])
        sns.despine(ax=ax, bottom=True)
        
    fig.suptitle(f'{step_name} Folds Separated ({metric.capitalize()})', fontweight='bold', y=1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

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
    tick_labels = [str(it).replace(' / ', '\n/ ') for it in items]
    cell_sz     = max(1.6, 6.0 / n)
    fig_sz      = cell_sz * n + 2.5

    fig, axes = plt.subplots(1, 2, figsize=(fig_sz * 2.1, fig_sz), layout="constrained")

    # Panel 0: Significance Matrix (Red = Significant via RdYlGn)
    disp_comb = np.where(np.eye(n, dtype=bool), np.nan, raw_p)
    for i in range(n):
        for j in range(i + 1, n):
            disp_comb[i, j] = adj_p_full[i, j]

    im0 = axes[0].imshow(disp_comb, cmap="RdYlGn", vmin=0, vmax=alpha * 2, aspect="auto")

    for i in range(n):
        for j in range(n):
            if i == j:
                axes[0].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#EBEBEB", zorder=2))
                axes[0].text(j, i, "Ref", ha="center", va="center", fontsize=8.5, color="black", zorder=3, fontweight="bold")
            elif np.isnan(disp_comb[i, j]):
                axes[0].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#F3F4F6", zorder=2))
                axes[0].text(j, i, "n.d.", ha="center", va="center", fontsize=8.0, color="#6B7280", zorder=3)
            else:
                p_val = disp_comb[i, j]
                stars = p_to_stars(p_val)
                txt = f"{p_val:.2e}\n{stars}" if p_val < 0.001 else f"{p_val:.3f}\n{stars}"
                col = "white" if p_val < alpha / 2 else "black"
                axes[0].text(j, i, txt, ha="center", va="center", fontsize=7.5, color=col, zorder=3)

    axes[0].set_xticks(range(n)); axes[0].set_xticklabels(tick_labels, fontsize=8)
    axes[0].set_yticks(range(n)); axes[0].set_yticklabels(tick_labels, fontsize=8)
    axes[0].set_title(
        f"Statistical Significance ({metric.upper()}, {filter_val})\n"
        f"Lower tri = raw Wilcoxon  |  Upper tri = {correction or 'uncorrected'}\n"
        f"(* p<0.05, ** p<0.01, *** p<0.001 | Ignored: |Δ| < {thresh_val})",
        fontsize=9.5, fontweight="bold"
    )
    axes[0].set_xlabel("Model B (column)")
    axes[0].set_ylabel("Model A (row)")
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.6, label="p-value (Red = Significant)")
    cb0.ax.axhline(alpha, color="red", lw=1.5, ls="--")

    # Panel 1: Cliff's Delta Effect-Size Matrix
    im1 = axes[1].imshow(eff, cmap="RdBu", vmin=-1.0, vmax=1.0, aspect="auto")
    for i in range(n):
        for j in range(n):
            if i == j:
                txt = "—"
                col = "black"
            else:
                txt = f"{eff[i,j]:.2f}"
                col = "white" if abs(eff[i, j]) > 0.5 else "black"
            axes[1].text(j, i, txt, ha="center", va="center", fontsize=8, color=col)

    axes[1].set_xticks(range(n)); axes[1].set_xticklabels(tick_labels, fontsize=8)
    axes[1].set_yticks(range(n)); axes[1].set_yticklabels(tick_labels, fontsize=8)
    axes[1].set_title(
        f"Effect Size ({metric.upper()})\n"
        f"Paired Mean Difference (Δ {metric.upper()})\n"
        f"(>0 = Row better | <0 = Col better)",
        fontsize=9.5, fontweight="bold"
    )
    axes[1].set_xlabel("Model B (column)")
    fig.colorbar(im1, ax=axes[1], shrink=0.6, label="mean_diff")

    fig.suptitle(
        f"Pairwise Statistical Comparison [Wilcoxon] — {filter_col}={filter_val!r}",
        fontsize=12, fontweight="bold"
    )
    sns.despine(fig=fig, left=True, bottom=True)

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved -> {save_path}")
    plt.close(fig)
    return fig

def plot_step_model_summary(
    df:            pd.DataFrame,
    step_name:     str  = 'Step',
    metrics:       list = None,
    metric_labels: dict = None,
    condition:     str  = None,   # if None, uses all rows
    unit_col:      str  = 'unit',
    fold_col:      str  = 'fold',
    alpha:         float = 0.05,
    test_type:     str  = 'permutation', # 'permutation' | 'wilcoxon' | 'bootstrap'
    save_path:     str  = None,
):
    """
    Traffic-light model summary table with correct statistical aggregation.

    Aggregation pipeline
    --------------------
    raw df (unit x fold x model) -> mean per unit across folds
                                  -> mean per model (= reported value)
    Statistical annotation
    ----------------------
    Each adjacent model pair is compared with aligned paired statistical test
    on unit-level metric distributions. FDR (BH) correction applied across all pairs.

    Layout
    ------
    One row per model (ordered Zero -> Baseline -> Cond/Id -> Cond/Sh).
    Columns: model description | metric_1 | metric_2 | ...
    Cell colour: traffic-light (green=better, red=worse within column).
    Stars between adjacent rows show significance of improvement.
    """
    set_pub_style()

    if metrics is None:
        metrics = [m for m in ['pearson', 'r2', 'mse'] if m in df.columns]
    if metric_labels is None:
        metric_labels = {
            'pearson': f'{step_name}\nPearson r',
            'r2'     : f'{step_name}\nR²',
            'nll'    : f'{step_name}\nNLL',
            'mse'    : f'{step_name}\nMSE',
        }

    MODEL_DESC = {
        'Zero / Identity'      : 'Null\n(mean spike rate)',
        'Baseline / Identity'  : 'Baseline\n(mean rate per bin)',
        'Baseline / Shared'    : 'Baseline\n+ shared covariance',
        'Conditional / Identity': 'Conditional mean\n(Transformer)',
        'Conditional / Shared' : 'Conditional mean\n+ shared covariance',
    }
    preferred_order = ['Zero / Identity', 'Baseline / Identity',
                       'Conditional / Identity', 'Conditional / Shared']

    # ── filter condition ──────────────────────────────────────────────────────
    work = df.copy()
    if condition is not None:
        work = work[work['condition'] == condition]
    if work.empty:
        print(f"plot_step_model_summary: empty dataframe after filtering.")
        return None

    all_models = work['model_label'].unique().tolist()
    rows_order  = [m for m in preferred_order if m in all_models] + \
                  [m for m in all_models if m not in preferred_order]

    # ── aggregate: unit -> fold mean -> model mean ────────────────────────────
    grp_cols = ['model_label', unit_col]
    if fold_col in work.columns:
        unit_vals = work.groupby(grp_cols + [fold_col], as_index=False)[metrics].mean()
        unit_vals = unit_vals.groupby(grp_cols, as_index=False)[metrics].mean()
    else:
        unit_vals = work.groupby(grp_cols, as_index=False)[metrics].mean()

    model_means = unit_vals.groupby('model_label')[metrics].mean()

    # ── pairwise statistical tests on aligned unit distributions (adjacent pairs) ──
    higher_is_better = {'pearson': True, 'r2': True, 'nll': False, 'mse': False}

    # Collect all p-values for FDR correction
    pair_tests = []   # (i, j, metric, raw_p)
    for mi, metric in enumerate(metrics):
        pivot_m = unit_vals.pivot(index=unit_col, columns='model_label', values=metric)
        for k in range(len(rows_order) - 1):
            mod_a = rows_order[k]
            mod_b = rows_order[k + 1]
            if mod_a not in pivot_m.columns or mod_b not in pivot_m.columns:
                pair_tests.append((k, k+1, metric, np.nan))
                continue
            pair_df = pivot_m[[mod_a, mod_b]].dropna()
            a = pair_df[mod_a].values
            b = pair_df[mod_b].values
            min_len = len(a)
            if min_len < 5:
                pair_tests.append((k, k+1, metric, np.nan))
                continue
            res = run_pairwise_wilcoxon(b, a)
            p = res['p_val']
            pair_tests.append((k, k+1, metric, p))

    # FDR correction
    raw_ps  = np.array([t[3] for t in pair_tests])
    valid_m = ~np.isnan(raw_ps)
    adj_ps  = raw_ps.copy()
    if valid_m.any():
        _, adj_vec, _, _ = multipletests(raw_ps[valid_m], alpha=alpha, method='fdr_bh')
        adj_ps[valid_m] = adj_vec

    # Build lookup: (row_i, row_j, metric) -> (adj_p, raw_p)
    sig_lookup = {}
    for idx_t, (ri, rj, met, rp) in enumerate(pair_tests):
        sig_lookup[(ri, rj, met)] = (adj_ps[idx_t], rp)

    # ── Layout ────────────────────────────────────────────────────────────────
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import FancyBboxPatch

    traffic_cmap = LinearSegmentedColormap.from_list('traffic',
                    ['#d62728', '#ffdd57', '#2ca02c'], N=256)

    n_rows  = len(rows_order)
    n_mcols = len(metrics)
    n_cols  = n_mcols + 1     # description col + metric cols

    ROW_H   = 0.9             # inches per model row
    DESC_W  = 3.2             # inches for description column
    CELL_W  = 1.8             # inches per metric cell
    STAR_H  = 0.35            # inches between rows for significance stars

    fig_h = n_rows * ROW_H + (n_rows - 1) * STAR_H + 1.2
    fig_w = DESC_W + n_mcols * CELL_W + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    total_content_h = n_rows * ROW_H + (n_rows - 1) * STAR_H
    cell_h_frac  = ROW_H  / (total_content_h + 1.0)   # fraction of axes height
    star_h_frac  = STAR_H / (total_content_h + 1.0)
    cell_w_frac  = [DESC_W / fig_w] + [CELL_W / fig_w] * n_mcols

    # cumulative x positions
    x_edges = [0.0]
    for cw in cell_w_frac:
        x_edges.append(x_edges[-1] + cw)

    # Column headers
    header_y = 1.0 - 0.5 * cell_h_frac
    ax.text((x_edges[0] + x_edges[1]) / 2, header_y + cell_h_frac * 0.55,
            'Model', ha='center', va='center', fontsize=10, fontweight='bold',
            transform=ax.transAxes)
    for j, m in enumerate(metrics):
        cx = (x_edges[j+1] + x_edges[j+2]) / 2
        ax.text(cx, header_y + cell_h_frac * 0.55,
                metric_labels.get(m, m),
                ha='center', va='center', fontsize=9, fontweight='bold',
                transform=ax.transAxes)

    # Rows
    for i, model_name in enumerate(rows_order):
        y_top = 1.0 - cell_h_frac * 0.5 - i * (cell_h_frac + star_h_frac)
        y0    = y_top - cell_h_frac / 2
        h     = cell_h_frac * 0.92

        # Description cell (white)
        desc = MODEL_DESC.get(model_name, model_name)
        x0   = x_edges[0] + 0.003
        w    = (x_edges[1] - x_edges[0]) - 0.006
        rect = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.005",
                              transform=ax.transAxes,
                              facecolor='#F5F5F5', edgecolor='#555', linewidth=1.0,
                              clip_on=False)
        ax.add_patch(rect)
        ax.text(x0 + w / 2, y0 + h / 2, desc,
                ha='center', va='center', fontsize=8, color='black',
                transform=ax.transAxes, linespacing=1.3)

        # Metric cells
        for j, m in enumerate(metrics):
            col_vals = model_means[m].reindex(rows_order).values.astype(float)
            vmin_c, vmax_c = np.nanmin(col_vals), np.nanmax(col_vals)
            if (vmax_c - vmin_c) < 0.01:
                vmin_c -= 0.05; vmax_c += 0.05

            norm = Normalize(vmin=vmin_c, vmax=vmax_c)
            val  = model_means.loc[model_name, m] if model_name in model_means.index else np.nan

            flip = not higher_is_better.get(m, True)
            nval = norm(val)
            color = traffic_cmap(1.0 - nval if flip else nval)

            x0c = x_edges[j+1] + 0.003
            wc  = (x_edges[j+2] - x_edges[j+1]) - 0.006
            rect = FancyBboxPatch((x0c, y0), wc, h, boxstyle="round,pad=0.005",
                                  transform=ax.transAxes,
                                  facecolor=color, edgecolor='#555', linewidth=1.0,
                                  clip_on=False)
            ax.add_patch(rect)

            lum = color[0]*0.299 + color[1]*0.587 + color[2]*0.114
            txt_c = 'white' if lum < 0.55 else 'black'
            label = f"{val:.4f}" if not np.isnan(val) else "—"
            ax.text(x0c + wc / 2, y0 + h / 2, label,
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color=txt_c, transform=ax.transAxes)

        # Significance stars between this row and the NEXT row
        if i < n_rows - 1:
            star_y = y0 - star_h_frac / 2
            for j, m in enumerate(metrics):
                adj_p, rp = sig_lookup.get((i, i+1, m), (np.nan, np.nan))
                stars = p_to_stars(adj_p)
                cx = (x_edges[j+1] + x_edges[j+2]) / 2
                star_color = '#2ca02c' if (not np.isnan(adj_p) and adj_p < alpha) else '#999999'
                ax.text(cx, star_y, stars, ha='center', va='center',
                        fontsize=10, color=star_color, fontweight='bold',
                        transform=ax.transAxes)

    # Colorbar
    from matplotlib import cm as _cm
    sm = _cm.ScalarMappable(cmap=traffic_cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=20, location='right')
    cbar.set_label('Relative performance (column-wise)', fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Worst', 'Mid', 'Best'], fontsize=7)

    cond_str = f" — {condition}" if condition else ""
    fig.suptitle(f'{step_name} model summary{cond_str}\n'
                 f'Stars = FDR-corrected {test_type.capitalize()} (unit-level), '
                 f'green=significant',
                 fontsize=11, fontweight='bold', y=1.02)

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=180)
        print(f"Saved -> {save_path}")
    plt.show()
    return fig

def plot_metric_heatmap(
    df: pd.DataFrame,
    metric: str = "pearson",
    model_col: str = "model_label",
    condition_col: str = "condition",
    unit_col: str = "unit",
    fold_col: str = "fold",
    alpha: float = 0.05,
    test_type: str = "permutation",
    magnitude_threshold: float = 'auto',
    cmap: str = None,
    save_path: str = None,
    title: str = None,
):
    """
    Model x Condition heatmap with:
      - Correct aggregation: unit -> fold-mean -> model-mean
      - Per-cell annotation: mean value + significance star vs. baseline model
      - Colourmap direction: green=better regardless of metric direction
      - Model rows in preferred order
      - Condition columns sorted, 'real' first if present

    Statistical annotation
    ----------------------
    Each cell (model M, condition C) is compared against the *baseline*
    model (first row) in the same condition using aligned paired tests
    on unit-level metric distributions. FDR (BH) corrected.
    """
    set_pub_style()
    from matplotlib.colors import Normalize

    higher_is_better = {'pearson': True, 'r2': True, 'nll': False, 'mse': False}
    hib = higher_is_better.get(metric, True)

    if cmap is None:
        cmap = "RdYlGn" if hib else "RdYlGn_r"

    # ── model order ───────────────────────────────────────────────────────────
    preferred_models = ['Zero / Identity', 'Baseline / Identity',
                        'Conditional / Identity', 'Conditional / Shared', 'Baseline / Shared']
    all_models = df[model_col].unique().tolist()
    order_models = [m for m in preferred_models if m in all_models] + \
                   [m for m in all_models if m not in preferred_models]

    # ── condition order ('real' first) ────────────────────────────────────────
    all_conds = df[condition_col].unique().tolist()
    cond_order = (["real"] if "real" in all_conds else []) + \
                 sorted(c for c in all_conds if c != "real")

    n_models = len(order_models)
    n_conds  = len(cond_order)

    # ── aggregate to unit level ───────────────────────────────────────────────
    grp = [model_col, condition_col, unit_col]
    if fold_col in df.columns:
        unit_agg = df.groupby(grp + [fold_col], as_index=False)[metric].mean()
        unit_agg = unit_agg.groupby(grp, as_index=False)[metric].mean()
    else:
        unit_agg = df.groupby(grp, as_index=False)[metric].mean()

    model_means = unit_agg.groupby([model_col, condition_col])[metric].mean().unstack(condition_col)
    model_means = model_means.reindex(index=order_models, columns=cond_order)

    # ── pairwise stats: each model vs Zero / Identity baseline ────────────────
    baseline = order_models[0]
    p_mat    = np.full((n_models, n_conds), np.nan)
    raw_ps_all = []
    pairs_idx  = []

    for i, mod in enumerate(order_models[1:], 1):   # skip baseline
        for j, cond in enumerate(cond_order):
            cond_sub = unit_agg[unit_agg[condition_col] == cond]
            pivot_cond = cond_sub.pivot(index=unit_col, columns=model_col, values=metric)
            if baseline not in pivot_cond.columns or mod not in pivot_cond.columns:
                raw_ps_all.append(np.nan)
                pairs_idx.append((i, j))
                continue
            pair_df = pivot_cond[[baseline, mod]].dropna()
            a = pair_df[baseline].values
            b = pair_df[mod].values
            min_n = len(a)
            if min_n < 5:
                raw_ps_all.append(np.nan)
            else:
                res = run_pairwise_wilcoxon(b, a, magnitude_threshold=(None if magnitude_threshold is None else (0.015 if metric=='pearson' else 0.01)))
                p = res['p_val']
                raw_ps_all.append(p)
            pairs_idx.append((i, j))

    raw_arr = np.array(raw_ps_all)
    valid   = ~np.isnan(raw_arr)
    adj_arr = raw_arr.copy()
    if valid.any():
        _, adj_vec, _, _ = multipletests(raw_arr[valid], alpha=alpha, method='fdr_bh')
        adj_arr[valid] = adj_vec
    for idx_t, (ri, ci) in enumerate(pairs_idx):
        p_mat[ri, ci] = adj_arr[idx_t]

    # ── build display matrix ───────────────────────────────────────────────────
    grid = model_means.values.astype(float)

    fig_w = max(5.0, 1.8 * n_conds + 3.0)
    fig_h = max(3.5, 0.75 * n_models + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmin = np.nanmin(grid)
    vmax = np.nanmax(grid)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    for i in range(n_models):
        for j in range(n_conds):
            val = grid[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9)
                continue

            norm_val = (val - vmin) / (vmax - vmin + 1e-12)
            if not hib:
                norm_val = 1.0 - norm_val
            txt_c = "white" if norm_val > 0.6 or norm_val < 0.15 else "black"

            stars = ""
            if i > 0 and not np.isnan(p_mat[i, j]):
                stars = "\n" + p_to_stars(p_mat[i, j])

            ax.text(j, i, f"{val:.3f}{stars}",
                    ha="center", va="center", fontsize=8,
                    color=txt_c, linespacing=1.2)

    ax.set_xticks(range(n_conds))
    ax.set_xticklabels(cond_order, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(order_models, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(f"mean {metric} ({'higher' if hib else 'lower'} = better)",
                   fontsize=8)

    direction = "higher=better" if hib else "lower=better"
    ax.set_title(
        title or (f"Model x Condition: {metric}  ({direction})\n"
                  f"Stars = FDR-corrected {test_type.capitalize()} vs. {baseline}"),
        fontsize=10, fontweight="bold", pad=10
    )
    ax.set_xlabel(f"Condition", fontsize=10)
    ax.set_ylabel(f"Model", fontsize=10)
    sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()
    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved -> {save_path}")
    plt.show()
    return fig

def _arr(history, k):
    return _np.asarray(history.get(k, []), dtype=float)

def _smooth(y, w):
    if w is None or w <= 1 or len(y) < 2:
        return _np.asarray(y, dtype=float)
    return _pd.Series(y).rolling(window=w, center=True, min_periods=1).mean().to_numpy()

def _get_lr(history):
    for key in ("learning_rate", "learning_rate_epoch", "learning_rate_step"):
        v = _arr(history, key)
        if v.size > 0 and _np.any(_np.isfinite(v)):
            return v
    return _np.array([])

def _lr_drops(epochs, lr_vals):
    drops = []
    for i in range(1, len(lr_vals)):
        if _np.isfinite(lr_vals[i]) and _np.isfinite(lr_vals[i-1]):
            if lr_vals[i] < lr_vals[i-1] * 0.95:
                drops.append(epochs[i])
    return drops

def plot_training_diagnostics(history, title=None, smooth_window=5, save_path=None):
    """5-panel diagnostic plot for a single training run.

    Parameters
    ----------
    history : dict | pd.DataFrame
        MetricHistory.history dict OR a filtered df_history row-group
        (auto-converted via to_dict).
    title : str, optional
    smooth_window : int
    save_path : str, optional
    """
    if hasattr(history, "to_dict"):
        history = history.sort_values("epoch").to_dict("list")

    epochs   = _arr(history, "epoch")
    tr_loss  = _arr(history, "train_loss_epoch")
    val_loss = _arr(history, "valid_loss_epoch")
    tr_mse   = _arr(history, "train_mse_epoch")
    val_mse  = _arr(history, "valid_mse_epoch")
    lr_vals  = _get_lr(history)

    if len(epochs) == 0:
        print("plot_training_diagnostics: empty history.")
        return None

    has_mse = tr_mse.size > 0 and _np.any(_np.isfinite(tr_mse))
    has_lr  = lr_vals.size > 0

    tr_s  = _smooth(tr_loss,  smooth_window)
    val_s = _smooth(val_loss, smooth_window)
    gap   = tr_s - val_s
    w     = max(3, smooth_window)
    val_jitter = _pd.Series(val_loss).rolling(window=w, min_periods=2).std().to_numpy()

    drops = _lr_drops(epochs, lr_vals) if has_lr else []

    def _vlines(ax):
        for ep in drops:
            ax.axvline(ep, color="gray", ls=":", lw=0.8, alpha=0.5)

    n_rows = 3 + int(has_mse) + int(has_lr)
    fig, axes = _plt.subplots(n_rows, 1, figsize=(10, 3*n_rows+1),
                              sharex=True, layout="constrained")

    # 0: loss
    ax = axes[0]
    ax.plot(epochs, tr_loss,  color="tab:blue",   alpha=0.22, lw=0.8)
    ax.plot(epochs, val_loss, color="tab:orange",  alpha=0.22, lw=0.8)
    ax.plot(epochs, tr_s,     color="tab:blue",   lw=2.0, label="train")
    ax.plot(epochs, val_s,    color="tab:orange",  lw=2.0, label="valid")
    _vlines(ax)
    ax.set_ylabel("NLL loss"); ax.set_title("Loss (raw + smoothed)")
    ax.legend(frameon=False, fontsize=9); _sns.despine(ax=ax)

    # 1: generalisation gap
    ax = axes[1]
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.fill_between(epochs, gap, 0, where=(gap>0), color="tab:green", alpha=0.35, label="underfit (train > valid)")
    ax.fill_between(epochs, gap, 0, where=(gap<0), color="tab:red",   alpha=0.35, label="overfit (train < valid)")
    ax.plot(epochs, gap, color="black", lw=1.2)
    _vlines(ax)
    ax.set_ylabel("train - valid"); ax.set_title("Generalisation gap")
    ax.legend(frameon=False, fontsize=9); _sns.despine(ax=ax)

    # 2: jitter
    ax = axes[2]
    ax.plot(epochs, val_jitter, color="tab:purple", lw=1.5)
    _vlines(ax)
    ax.set_ylabel("Rolling std"); ax.set_title(f"Valid-loss jitter (window={w})")
    _sns.despine(ax=ax)

    r = 3
    if has_mse:
        ax = axes[r]
        ax.plot(epochs, tr_mse,  color="tab:blue",   alpha=0.22, lw=0.8)
        ax.plot(epochs, val_mse, color="tab:orange",  alpha=0.22, lw=0.8)
        ax.plot(epochs, _smooth(tr_mse, smooth_window),  color="tab:blue",   lw=2.0, label="train")
        ax.plot(epochs, _smooth(val_mse, smooth_window), color="tab:orange", lw=2.0, label="valid")
        _vlines(ax)
        ax.set_ylabel("MSE"); ax.set_title("MSE (Anscombe space)")
        ax.legend(frameon=False, fontsize=9); _sns.despine(ax=ax)
        r += 1

    if has_lr:
        ax = axes[r]
        ax.plot(epochs, lr_vals, color="tab:green", lw=1.8)
        for ep in drops:
            ax.axvline(ep, color="red", ls=":", lw=1.0, alpha=0.7)
        ax.set_ylabel("LR"); ax.set_yscale("log"); ax.set_title("Learning rate")
        _sns.despine(ax=ax)

    axes[-1].set_xlabel("Epoch")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved -> {save_path}")
    _plt.show()
    return fig

def plot_seed_stability(
    all_histories,
    seed_summary,
    title=None,
    smooth_window=5,
    save_path=None,
):
    """2-panel multi-seed stability summary.

    Panels
    ------
    0  Final loss — strip + violin of final valid loss per seed; colour = jitter
    1  Jitter     — heatmap of rolling-std valid loss (epochs x seeds)

    Parameters
    ----------
    all_histories : list[dict]   one per seed (from run_seeded_training)
    seed_summary  : pd.DataFrame (from run_seeded_training)
    """
    n_seeds = len(all_histories)
    seeds   = seed_summary["seed"].tolist()

    # Local helpers to safely extract and smooth data
    def _arr(h, key):
        return _np.asarray(h.get(key, []), dtype=float)

    def _smooth(y, w):
        if w is None or w <= 1 or len(y) < 2:
            return y
        return _pd.Series(y).rolling(window=w, center=True, min_periods=1).mean().to_numpy()

    # Pre-compute smoothed valid loss aligned on a common epoch axis
    max_ep = max(len(_arr(h, "epoch")) for h in all_histories)
    val_mat  = _np.full((n_seeds, max_ep), _np.nan)
    jit_mat  = _np.full((n_seeds, max_ep), _np.nan)
    
    for i, h in enumerate(all_histories):
        ep  = _arr(h, "epoch")
        vl  = _arr(h, "valid_loss_epoch")
        n   = len(ep)
        w   = max(3, smooth_window)
        # Calculate standard deviation rolling window (Jitter)
        jit = _pd.Series(vl).rolling(window=w, min_periods=2).std().to_numpy()
        jit_mat[i, :n] = jit

    # Create a 2-panel figure with better vertical proportions
    fig, axes = _plt.subplots(2, 1, figsize=(10, 7.5), layout="constrained")

    # ── Jitter calculations for color mapping ──
    jitter_vals = seed_summary["mean_jitter"].to_numpy()
    if jitter_vals.max() == jitter_vals.min():
        jitter_norm = _np.zeros_like(jitter_vals)
    else:
        jitter_norm = (jitter_vals - jitter_vals.min()) / (jitter_vals.max() - jitter_vals.min())
    
    cmap_jit = _plt.get_cmap("RdYlGn_r") # Green = stable, Red = chaotic

    # ── Panel 0: Final valid loss distribution ───────────────────────────────
    ax = axes[0]
    fl = seed_summary["final_valid_loss"].to_numpy()
    
    # Modern violin plot (removed ugly internal median lines for a cleaner look)
    violin = ax.violinplot(fl, positions=[0], widths=0.5, showmedians=False, showextrema=False)
    for pc in violin["bodies"]:
        pc.set_facecolor("slategray")
        pc.set_alpha(0.15)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)

    # Clean, distinct horizontal lines for summary stats
    ax.axhline(fl.min(),  color="forestgreen", ls="--", lw=1.5, alpha=0.8, label=f"Best: {fl.min():.4f}")
    ax.axhline(fl.mean(), color="royalblue",   ls="-.", lw=1.5, alpha=0.8, label=f"Mean: {fl.mean():.4f}")
    ax.axhline(fl.max(),  color="crimson",     ls=":",  lw=1.5, alpha=0.8, label=f"Worst: {fl.max():.4f}")

    # Scatter points with aesthetic jitter, larger sizes, and white outlines
    rng = _np.random.default_rng(42) # fixed seed for visual consistency
    jitter_x = rng.uniform(-0.15, 0.15, n_seeds)
    sc = ax.scatter(jitter_x, fl, c=jitter_norm, cmap=cmap_jit, s=120, zorder=5, edgecolors="white", lw=1.2, alpha=0.9)
    
    # Add seed text labels neatly next to the scatter points
    for j, (xj, yj, s) in enumerate(zip(jitter_x, fl, seeds)):
        ax.text(xj + 0.025, yj, f"s{s}", fontsize=9, va="center", fontweight="medium", color="#333333")

    ax.set_xticks([]) # Remove x-axis tick completely for cleaner look
    ax.set_ylabel("Final Valid NLL Loss", fontweight="bold")
    ax.set_title("Final Loss Distribution Across Seeds", pad=12, fontweight="bold", fontsize=11)
    ax.legend(frameon=True, fontsize=9, loc="upper right", edgecolor="#e0e0e0")
    _sns.despine(ax=ax, bottom=True) # Remove x-axis spine

    # ── Panel 1: Jitter heatmap ──────────────────────────────────────────────
    ax = axes[1]
    
    # Clip to common range for readability
    vmax = _np.nanpercentile(jit_mat, 95)
    
    # Swapped YlOrRd for magma_r which looks much more premium and high-contrast
    im = ax.imshow(jit_mat, aspect="auto", cmap="magma_r",
                   origin="upper", vmin=0, vmax=vmax,
                   extent=[0, max_ep, n_seeds - 0.5, -0.5])
                   
    ax.set_yticks(_np.arange(n_seeds))
    ax.set_yticklabels([f"seed {s}" for s in seeds], fontsize=9)
    ax.set_xlabel("Training Epoch", fontweight="bold")
    ax.set_ylabel("Seed", fontweight="bold")
    ax.set_title("Loss Jitter Heatmap (Rolling Std)", pad=12, fontweight="bold", fontsize=11)
    
    # Refined colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Rolling Std", rotation=270, labelpad=15, fontweight="bold", fontsize=9)
    cbar.outline.set_visible(False)
    
    _sns.despine(ax=ax)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="heavy", y=1.03)
        
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200, facecolor='white')
        print(f"Saved -> {save_path}")
        
    _plt.show()
    return fig

def _safe_diag(trainer_or_history, title="", smooth_window=7):
    """Call plot_training_diagnostics safely: accepts trainer or history dict."""
    if hasattr(trainer_or_history, "metric_history_callback"):
        h = trainer_or_history.metric_history_callback.history
    elif isinstance(trainer_or_history, dict):
        h = trainer_or_history
    else:
        print(f"_safe_diag: unrecognised input type {type(trainer_or_history)}")
        return
    plot_training_diagnostics(h, title=title, smooth_window=smooth_window)

def plot_step5_shared_factors(prep, lambda_hat_4, n_components=5, seed=SEED):
    '''Visualizes the latent factors generated by Step 5.'''
    from sklearn.decomposition import FactorAnalysis
    import matplotlib.pyplot as plt
    
    Y_true = prep['counts']
    T, N, K = Y_true.shape
    
    Y_flat = Y_true.transpose(0, 2, 1).reshape(T * K, N)
    lam_flat = lambda_hat_4.transpose(0, 2, 1).reshape(T * K, N)
    
    eps = 1e-8
    residuals = (Y_flat - lam_flat) / np.sqrt(lam_flat + eps)
    
    fa = FactorAnalysis(n_components=n_components, random_state=seed)
    fa.fit(residuals)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot Loadings
    im = axes[0].imshow(fa.components_, aspect='auto', cmap='RdBu_r', vmin=-np.max(np.abs(fa.components_)), vmax=np.max(np.abs(fa.components_)))
    axes[0].set_title("Factor Loadings across Units")
    axes[0].set_xlabel("Unit Index")
    axes[0].set_ylabel("Factor Index")
    fig.colorbar(im, ax=axes[0])
    
    # Plot first 100 timesteps of Factor 1
    Z = fa.transform(residuals)
    axes[1].plot(Z[:100, 0], label="Factor 1", color="tab:blue")
    axes[1].plot(Z[:100, 1], label="Factor 2", color="tab:orange")
    axes[1].set_title("Latent Factors (First 100 bins)")
    axes[1].set_xlabel("Time Bin")
    axes[1].set_ylabel("Factor Activation")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
