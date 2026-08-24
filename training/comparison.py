"""Notebook-faithful K-fold comparisons with explicit task-column selection."""
from __future__ import annotations
import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from mean_cov_pipeline.data.datasets import Anscombe, NeuralDataset
from mean_cov_pipeline.data.selection import select_task_variables
from mean_cov_pipeline.training.lightning import LitModel, predict_loader, trial_metrics
from mean_cov_pipeline.training.seeded import run_seeded_training

def _standardize(train, valid):
    mean, std = train.mean(axis=0, keepdims=True), train.std(axis=0, keepdims=True)
    return (train - mean) / (std + 1e-8), (valid - mean) / (std + 1e-8)

def checkpoint_prefix_for(root: Path | str, condition: str, model_label: str, fold: int) -> Path:
    safe = model_label.replace(" / ", "_")
    return Path(root) / "checkpoints" / condition / safe / f"fold{fold}" / f"{condition}_{safe}_fold{fold}"

def _train_one(conf, mean_type, cov_type, train_dataset, valid_dataset, *, checkpoint_prefix=None, frozen_state_dict=None, use_multi_seed=False):
    train_loader = DataLoader(train_dataset, batch_size=conf.training.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=conf.training.batch_size, shuffle=False)
    histories, model, seeds = run_seeded_training(conf, lambda: LitModel(conf, mean_type, cov_type).to(conf.device), train_loader, valid_loader, n_seeds=conf.training.n_seeds if use_multi_seed else 1, frozen_state_dict=frozen_state_dict, checkpoint_prefix=checkpoint_prefix, verbose=True)
    return model, histories, seeds

def run_model_comparisons(training_datasets: Mapping[str, np.ndarray], x_position_vars: np.ndarray, prep: Mapping, conf, device, *, model_variants: Sequence[tuple[str, str]] | None = None, n_splits: int = 10, force_retrain: bool = False, allowed_vars_dict: Mapping[str, Sequence[str] | None] | None = None, use_multi_seed: bool = False, trainer_factory=None, checkpoint_root: Path | str | None = None, checkpoint_prefix_factory=None, results_root: Path | str | None = None, shap_hook=None):
    """Full notebook comparison behavior, without globals or zero-padding.

    Completed CSV folds are skipped. Identity means are cached/restored then
    frozen for their matching shared-covariance fold. ``shap_hook`` is a lazy
    callable integration seam for the original conditional/identity SHAP run.
    """
    variants = model_variants or [("zero", "identity"), ("baseline", "identity"), ("baseline", "shared"), ("conditional", "identity"), ("conditional", "shared")]
    root = Path(results_root or checkpoint_root or getattr(getattr(conf, "paths", object()), "results", ".")); root.mkdir(parents=True, exist_ok=True)
    hp, up, sp = root / "training_history_full.csv", root / "unit_metrics_full.csv", root / "seed_stability_summary.csv"
    old_h = pd.read_csv(hp) if not force_retrain and hp.exists() else pd.DataFrame(); old_u = pd.read_csv(up) if not force_retrain and up.exists() else pd.DataFrame(); old_s = pd.read_csv(sp) if not force_retrain and sp.exists() else pd.DataFrame()
    histories, units, summaries = ([old_h] if not old_h.empty else []), ([old_u] if not old_u.empty else []), ([old_s] if not old_s.empty else [])
    folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=conf.seed).split(next(iter(training_datasets.values())))); cached_means = {}
    for condition, targets in training_datasets.items():
        dense, sparse = select_task_variables(prep, allowed_vars_dict.get(condition) if allowed_vars_dict else None)
        current = copy.deepcopy(conf); current.data.n_dense_vars, current.data.n_sparse_vars = dense.shape[1], sparse.shape[1]; current.data.n_vars = current.data.n_position_vars + dense.shape[1] + sparse.shape[1]
        for mean_type, cov_type in variants:
            label = f"{mean_type.capitalize()} / {cov_type.capitalize()}"
            for fold, (train_idx, valid_idx) in enumerate(folds):
                done = not old_h.empty and not old_u.empty and ((old_h.condition == condition) & (old_h.model == label) & (old_h.fold == fold)).any() and ((old_u.condition == condition) & (old_u.model_label == label) & (old_u.fold == fold)).any()
                if done: continue
                prefix = None
                if checkpoint_root is not None:
                    prefix = (checkpoint_prefix_factory or checkpoint_prefix_for)(checkpoint_root, condition, label, fold); Path(prefix).parent.mkdir(parents=True, exist_ok=True)
                pos_tr, pos_va = _standardize(x_position_vars[train_idx], x_position_vars[valid_idx]); den_tr, den_va = _standardize(dense[train_idx], dense[valid_idx]); y_tr, y_va = targets[train_idx], targets[valid_idx]
                y_mean = Anscombe()(torch.as_tensor(y_tr, dtype=torch.float32, device=device).transpose(1, 2)).mean((0, 1), keepdim=True)
                train_ds, valid_ds = NeuralDataset(pos_tr, den_tr, sparse[train_idx], y_tr, y_mean), NeuralDataset(pos_va, den_va, sparse[valid_idx], y_va, y_mean)
                frozen = None
                if cov_type == "shared" and mean_type in {"baseline", "conditional"}:
                    frozen = cached_means.get((condition, mean_type, fold)); artifact = root / f"ckpt_{condition}_{mean_type.capitalize()}_Identity_fold{fold}.pt"
                    if frozen is None and artifact.exists():
                        state = torch.load(artifact, map_location=device, weights_only=True); frozen = {k.removeprefix("full_model.mean_model."): v for k, v in state.items() if k.startswith("full_model.mean_model.")}
                if trainer_factory is None: model, run_h, seed_sum = _train_one(current, mean_type, cov_type, train_ds, valid_ds, checkpoint_prefix=prefix, frozen_state_dict=frozen, use_multi_seed=use_multi_seed)
                else:
                    model, result = trainer_factory(current, mean_type, cov_type, train_ds, valid_ds); run_h, seed_sum = ([result] if isinstance(result, dict) else []), pd.DataFrame()
                torch.save(model.state_dict(), root / f"ckpt_{condition}_{label.replace(' / ', '_')}_fold{fold}.pt")
                if cov_type == "identity" and mean_type in {"baseline", "conditional"}: cached_means[(condition, mean_type, fold)] = copy.deepcopy(model.full_model.mean_model.state_dict())
                for seed, item in enumerate(run_h):
                    if item and item.get("epoch"): histories.append(pd.DataFrame(item).assign(condition=condition, fold=fold, model=label, seed=seed))
                if not seed_sum.empty: summaries.append(seed_sum.assign(condition=condition, fold=fold, model=label))
                vl, tl = DataLoader(valid_ds, batch_size=current.training.batch_size), DataLoader(train_ds, batch_size=current.training.batch_size, shuffle=True)
                truth, pred = predict_loader(current, model, vl, y_mean, "all"); vp, vm, vr = trial_metrics(truth, pred)
                ttruth, tpred = predict_loader(current, model, tl, y_mean, "all"); tp, tm, tr = trial_metrics(ttruth, tpred)
                frame = pd.DataFrame({"unit": np.arange(current.data.n_units), "pearson": vp.mean(1), "mse": vm.mean(1), "r2": vr.mean(1), "train_pearson": tp.mean(1), "train_mse": tm.mean(1), "train_r2": tr.mean(1), "condition": condition, "fold": fold, "model_label": label})
                if fold == n_splits - 1:
                    for key, values in {"pearson_trials":vp,"mse_trials":vm,"r2_trials":vr,"train_pearson_trials":tp,"train_mse_trials":tm,"train_r2_trials":tr}.items(): frame[key] = [values[i].tolist() for i in range(current.data.n_units)]
                units.append(frame)
                if shap_hook and condition in {"real","step3","step4","step5","step6a","step7"} and (mean_type, cov_type) == ("conditional", "identity"):
                    shap_hook(conf=current, model=model.to(device), train_dataset=train_ds, valid_dataset=valid_ds, save_path=root / f"shap_values_{condition}_{label.replace(' / ', '_')}_fold{fold}.npz")
                if histories: pd.concat(histories, ignore_index=True).to_csv(hp, index=False)
                if units: pd.concat(units, ignore_index=True).to_csv(up, index=False)
                if summaries: pd.concat(summaries, ignore_index=True).to_csv(sp, index=False)
    return pd.concat(histories, ignore_index=True) if histories else pd.DataFrame(), pd.concat(units, ignore_index=True) if units else pd.DataFrame(), pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
