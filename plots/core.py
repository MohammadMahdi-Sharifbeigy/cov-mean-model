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

def save_figure(fig, file_name, file_path, ext=".png", **savefig_kwargs):
    os.makedirs(file_path, exist_ok=True)
    full_path = os.path.join(file_path, f"{file_name}{ext}")
    fig.savefig(full_path, dpi=150, bbox_inches="tight", pad_inches=0.25, **savefig_kwargs)
    plt.close(fig)
    return full_path

def save_shap_plot(make_plot, title, file_name, file_path, figsize=(10, 6)):
    plt.close("all")
    plot_result = make_plot()
    if hasattr(plot_result, "figure"):
        fig = plot_result.figure
        ax = plot_result
    else:
        fig = plt.gcf()
        ax = plt.gca()

    fig.set_size_inches(figsize)
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

def plot_imshow(
    data,
    sig=None,
    *,
    fig=None,
    subspec=None,
    ax=None,
    title=None,
    xlabel=None,
    ylabel=None,
    xthicks=None,
    ythicks=None,
    vmin=None,
    vmax=None,
    norm_mode="normal",
    cmap=None,
    colorbar=True,
    cax=None,
    return_im=False,
    left_method=None,
    bottom_method=None,
    left_mode="line",
    bottom_mode="line",
    left_width=0.22,
    bottom_height=0.22,
    wspace=0.02,
    hspace=0.02,
    figsize=(7.5, 5.5),
    save_path=None,
    **plot_kwargs,
):

    if sig is None:
        sig = np.ones_like(data, dtype=bool)
    sig = np.asarray(sig).astype(bool)

    n_y, n_x = data.shape
    disp = data * sig

    n_xticks = min(25, n_x)
    xticks_idx = np.linspace(0, n_x - 1, n_xticks, dtype=int)
    n_yticks = min(25, n_y)
    yticks_idx = np.linspace(0, n_y - 1, n_yticks, dtype=int)

    vmax = np.nanmax(data) if vmax is None else vmax
    vmin = np.nanmin(data) if vmin is None else vmin

    if norm_mode == "normal":
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    elif norm_mode == "symlog":
        norm = colors.SymLogNorm(
            linthresh=0.5 * (vmax - vmin) / 2,
            linscale=1.0,
            vmin=vmin,
            vmax=vmax,
            base=10,
        )

    want_left = left_method is not None
    want_bottom = bottom_method is not None

    standalone = (fig is None or subspec is None) and ax is None

    if ax is not None:
        fig = ax.figure
        ax_main = ax
        ax_left = None
        ax_bottom = None
        want_left = False
        want_bottom = False
    else:
        if fig is None or subspec is None:
            fig = plt.figure(figsize=figsize, layout="constrained")
            subspec = fig.add_gridspec(1, 1)[0, 0]

        nrows = 2 if want_bottom else 1
        ncols = 2 if want_left else 1
        width_ratios = ([left_width, 1.0] if want_left else [1.0])
        height_ratios = ([1.0, bottom_height] if want_bottom else [1.0])

        gs = subspec.subgridspec(
            nrows,
            ncols,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            wspace=wspace,
            hspace=hspace,
        )

        r_main, c_main = 0, (1 if want_left else 0)
        ax_main = fig.add_subplot(gs[r_main, c_main])

        ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main) if want_left else None
        ax_bottom = fig.add_subplot(gs[1, c_main], sharex=ax_main) if want_bottom else None

        if want_left and want_bottom:
            ax_corner = fig.add_subplot(gs[1, 0])
            ax_corner.axis("off")

    cmap = cmaps.redshift if cmap is None else cmap

    im = ax_main.imshow(
        disp,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        **plot_kwargs,
    )

    ax_main.set_title(title if title is not None else None)

    for sp in ax_main.spines.values():
        sp.set_visible(False)

    if not want_left:
        ax_main.set_ylabel(ylabel if ylabel is not None else None)
        ax_main.set_yticks(yticks_idx)
        ax_main.set_yticklabels(
            (ythicks[yticks_idx] if ythicks is not None else np.arange(n_y)[yticks_idx]),
            fontsize=4,
        )
    else:
        ax_main.set_ylabel("")
        ax_main.yaxis.set_visible(False)

    if not want_bottom:
        ax_main.set_xlabel(xlabel if xlabel is not None else None)
        ax_main.set_xticks(xticks_idx)
        ax_main.set_xticklabels(
            (xthicks[xticks_idx] if xthicks is not None else np.arange(n_x)[xticks_idx]),
            rotation=45,
            ha="right",
            fontsize=4,
        )
    else:
        ax_main.set_xlabel("")
        ax_main.xaxis.set_visible(False)

    if want_left and ax_left is not None:
        y_stat = aggregate(data, axis=1, method=left_method)
        yy = np.arange(n_y)

        lm = str(left_mode).lower().strip()
        if lm == "line":
            ax_left.plot(y_stat, yy, linewidth=1)
        elif lm == "scatter":
            ax_left.scatter(y_stat, yy, s=12, linewidths=0)
        elif lm == "stem":
            markerline, stemlines, baseline = ax_left.stem(yy, y_stat, orientation="horizontal")
            baseline.set_visible(False)
        elif lm == 'bar':
            ax_left.barh(yy, y_stat)

        ax_left.set_ylim(n_y - 0.5, -0.5)
        ax_left.invert_xaxis()

        ax_left.set_yticks(yticks_idx)
        ax_left.set_yticklabels(
            (ythicks[yticks_idx] if ythicks is not None else np.arange(n_y)[yticks_idx]),
            fontsize=8,
        )

        ax_left.set_ylabel(ylabel if ylabel else None)

        ax_left.set_xlabel(left_method, fontsize=8)
        ax_left.tick_params(axis="x", labelsize=8)
        ax_left.tick_params(axis="y", labelsize=8)

        ax_left.spines["top"].set_visible(False)
        ax_left.spines["right"].set_visible(False)

    if want_bottom and ax_bottom is not None:
        x_stat = aggregate(data, axis=0, method=bottom_method)
        xx = np.arange(n_x)

        bm = str(bottom_mode).lower().strip()
        if bm == "line":
            ax_bottom.plot(xx, x_stat, linewidth=1)
        elif bm == "scatter":
            ax_bottom.scatter(xx, x_stat, s=12, linewidths=0)
        elif bm == "stem":
            markerline, stemlines, baseline = ax_bottom.stem(xx, x_stat)
            baseline.set_visible(False)
        elif bm == "bar":
            ax_bottom.bar(xx, x_stat)

        ax_bottom.set_xlim(-0.5, n_x - 0.5)

        ax_bottom.set_xticks(xticks_idx)
        ax_bottom.set_xticklabels(
            (xthicks[xticks_idx] if xthicks is not None else np.arange(n_x)[xticks_idx]),
            rotation=45,
            ha="right",
            fontsize=8,
        )

        ax_bottom.set_xlabel(xlabel if xlabel is not None else None)

        ax_bottom.set_ylabel(bottom_method, fontsize=8)
        ax_bottom.tick_params(axis="x", labelsize=8)
        ax_bottom.tick_params(axis="y", labelsize=8)

        ax_bottom.spines["top"].set_visible(False)
        ax_bottom.spines["right"].set_visible(False)

    cb = None
    if colorbar:
        cb = fig.colorbar(im, ax=ax_main, cax=cax, fraction=0.02, pad=0.02)
        cb.ax.tick_params(labelsize=8)

    handles = {
        "fig": fig,
        "ax": ax_main,
        "ax_left": ax_left,
        "ax_bottom": ax_bottom,
        "im": im,
        "cb": cb,
    }

    if standalone and not return_im:
        if save_path is not None and title is not None:
            save_figure(fig, title, save_path, bbox_inches="tight")
        plt.show()
        return None

    return handles

def plot_imshow_grid(
    data,
    sig=None,
    *,
    xlabel=None,
    ylabel=None,
    subtitle=None,
    title=None,
    xthicks=None,
    ythicks=None,
    vmin=None,
    vmax=None,
    norm_mode="normal",
    cmap=None,
    colorbar=True,
    left_method=None,
    bottom_method=None,
    left_mode="line",
    bottom_mode="line",
    left_width=0.22,
    bottom_height=0.22,
    wspace=0.01,
    hspace=0.01,
    figsize=(7.5, 5.5),
    save_path=None,
    **plot_kwargs,
):

    n = data.shape[0]

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(figsize[0] * cols, figsize[1] * rows), layout="constrained")
    outer = fig.add_gridspec(rows, cols)

    axes = []

    for i in range(n):
        ss = outer[i // cols, i % cols]

        h = plot_imshow(
            data=data[i],
            sig=sig[i] if sig is not None else None,
            fig=fig,
            subspec=ss,
            title=subtitle[i] if subtitle is not None else None,
            xlabel=xlabel,
            ylabel=ylabel,
            xthicks=xthicks,
            ythicks=ythicks,
            vmin=vmin[i] if vmin is not None else None,
            vmax=vmax[i] if vmax is not None else None,
            norm_mode=norm_mode,
            cmap=cmap,
            colorbar=colorbar,
            cax=None,
            return_im=True,
            left_method=left_method,
            bottom_method=bottom_method,
            left_mode=left_mode,
            bottom_mode=bottom_mode,
            left_width=left_width,
            bottom_height=bottom_height,
            wspace=wspace,
            hspace=hspace,
            **plot_kwargs,
        )

        axes.append(h["ax"])

    for j in range(n, rows * cols):
        ax = fig.add_subplot(outer[j // cols, j % cols])
        ax.axis("off")

    fig.suptitle(title if title is not None else None)

    if save_path is not None and title is not None:
        save_figure(fig, title, save_path, bbox_inches="tight")

    plt.show()
    return None

def plot_reject_imshow(
    data,
    *,
    x_method="std",
    y_method="std",
    x_threshold=None,
    y_threshold=None,
    xlabel=None,
    ylabel=None,
    xthicks=None,
    ythicks=None,
    vmin=None,
    vmax=None,
    norm_mode="normal",
    cmap="inferno",
    c="black",
    s=20,
    alpha=1,
    save_path=None,
    title="Reject Summary",
):
    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError("data must be 2D (n_x, n_y).")

    def _parse_limits(th):
        if th is None:
            return None, None
        if np.isscalar(th):
            return None, float(th)
        th = tuple(th)
        if len(th) != 2:
            raise ValueError("threshold must be None, scalar, or (low, high).")
        low, high = th
        low = None if low is None else float(low)
        high = None if high is None else float(high)
        return low, high

    n_x, n_y = data.shape

    y_score = aggregate(data, axis=0, method=y_method)
    x_score = aggregate(data, axis=1, method=x_method)

    y_low, y_high = _parse_limits(y_threshold)
    x_low, x_high = _parse_limits(x_threshold)

    bad_y = np.zeros(n_y, dtype=bool)
    bad_x = np.zeros(n_x, dtype=bool)

    if y_low is not None:
        bad_y |= y_score < y_low
    if y_high is not None:
        bad_y |= y_score > y_high

    if x_low is not None:
        bad_x |= x_score < x_low
    if x_high is not None:
        bad_x |= x_score > x_high

    y_to_remove = np.where(bad_y)[0]
    x_to_remove = np.where(bad_x)[0]

    if xthicks is None:
        xthicks = np.arange(n_x)
    if ythicks is None:
        ythicks = np.arange(n_y)

    h = plot_imshow(
        data.T,
        sig=None,
        title=None,
        xlabel=xlabel,
        ylabel=ylabel,
        xthicks=xthicks,
        ythicks=ythicks,
        vmin=vmin,
        vmax=vmax,
        norm_mode=norm_mode,
        cmap=cmap,
        colorbar=True,
        cax=None,
        return_im=True,
        left_method=y_method,
        bottom_method=x_method,
        left_mode="scatter",
        bottom_mode="scatter",
    )

    fig = h["fig"]
    ax_main = h["ax"]
    ax_left = h["ax_left"]
    ax_bottom = h["ax_bottom"]

    if ax_left is not None:
        yy = np.arange(n_y)
        ax_left.scatter(y_score, yy, s=max(1, s * 0.25), color=c, alpha=max(0.05, alpha))
        if y_low is not None:
            ax_left.axvline(y_low, color="tab:red", linewidth=1)
        if y_high is not None:
            ax_left.axvline(y_high, color="tab:red", linewidth=1)
        if np.any(bad_y):
            ax_left.scatter(y_score[bad_y], yy[bad_y], s=s, color="tab:red", alpha=1.0)

    if ax_bottom is not None:
        xx = np.arange(n_x)
        ax_bottom.scatter(xx, x_score, s=max(1, s * 0.25), color=c, alpha=max(0.05, alpha))
        if x_low is not None:
            ax_bottom.axhline(x_low, color="tab:red", linewidth=1)
        if x_high is not None:
            ax_bottom.axhline(x_high, color="tab:red", linewidth=1)
        if np.any(bad_x):
            ax_bottom.scatter(xx[bad_x], x_score[bad_x], s=s, color="tab:red", alpha=1.0)

    if np.any(bad_y):
        for yi in np.where(bad_y)[0]:
            ax_main.axhline(yi - 0.5, color="tab:red", linewidth=0.6)
            ax_main.axhline(yi + 0.5, color="tab:red", linewidth=0.6)

    if np.any(bad_x):
        for xi in np.where(bad_x)[0]:
            ax_main.axvline(xi - 0.5, color="tab:red", linewidth=0.6)
            ax_main.axvline(xi + 0.5, color="tab:red", linewidth=0.6)

    fig.suptitle(title, fontsize=16)
    save_figure(fig, title, save_path, bbox_inches="tight")
    plt.show()

    return y_to_remove, x_to_remove

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
    seed: int = 42,
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

    fig, axes = plt.subplots(1, len(show), figsize=(3.0 * len(show), 3.2),
                              sharey=True, squeeze=False)
    
    for ax, u in zip(axes[0], show):
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

    ax0 = axes[0][0]
    step = max(1.0, round(ymax / 3))
    x0 = t[0]
    ax0.plot([x0, x0], [0, step], color="k", lw=2.5, clip_on=False)
    ax0.text(x0 - (t[-1] - t[0]) * 0.02, step / 2, f"{step:.0f} Hz",
             rotation=90, ha="right", va="center", fontsize=8)
    
    axes[0][-1].legend(loc="upper right", fontsize=8, frameon=False)
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
