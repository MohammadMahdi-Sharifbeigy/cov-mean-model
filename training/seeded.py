"""Notebook-compatible seeded Lightning training and checkpoint callbacks."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:  # Keep data-selection imports usable without Lightning installed.
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:  # pragma: no cover - exercised only in minimal installs
    L = None


if L is not None:
    class MetricHistory(L.Callback):
        """Record the epoch metrics emitted by ``LitModel``."""

        def __init__(self) -> None:
            super().__init__()
            self.history = {name: [] for name in (
                "epoch", "train_loss_epoch", "valid_loss_epoch",
                "train_correlation", "valid_correlation", "train_mse_epoch",
                "valid_mse_epoch", "learning_rate",
            )}

        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            metrics = trainer.callback_metrics
            self.history["epoch"].append(int(trainer.current_epoch))
            for name in self.history:
                if name in {"epoch", "learning_rate"}:
                    continue
                value = metrics.get(name)
                self.history[name].append(float(value.detach().cpu()) if value is not None else np.nan)
            optimizers = trainer.optimizers
            self.history["learning_rate"].append(
                float(optimizers[0].param_groups[0]["lr"]) if optimizers else np.nan
            )


    class TQDMEpochProgressBar(L.Callback):
        """A quiet-by-default progress callback retained from the notebook."""

        def __init__(self) -> None:
            super().__init__()
            self.pbar = None

        def on_fit_start(self, trainer, pl_module) -> None:
            from tqdm.auto import tqdm
            self.pbar = tqdm(total=trainer.max_epochs, initial=trainer.current_epoch, desc="Epochs", leave=False)

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if self.pbar is None:
                return
            self.pbar.update(1)
            metrics = trainer.callback_metrics
            values = {key: metrics.get(key, float("nan")) for key in ("train_loss_epoch", "valid_loss_epoch", "train_mse_epoch")}
            self.pbar.set_postfix({key: f"{float(value):.4f}" for key, value in values.items()})

        def _close(self) -> None:
            if self.pbar is not None:
                self.pbar.close()
                self.pbar = None

        def on_fit_end(self, trainer, pl_module) -> None:
            self._close()

        def teardown(self, trainer, pl_module, stage) -> None:
            self._close()
else:
    class MetricHistory:  # pragma: no cover - only used without Lightning
        def __init__(self) -> None:
            self.history = {}

    class TQDMEpochProgressBar:  # pragma: no cover
        pass


def run_seeded_training(
    conf, builder_fn: Callable[[], torch.nn.Module], train_loader, valid_loader, *,
    n_seeds: int = 1, seeds: Sequence[int] | None = None,
    frozen_state_dict: dict | None = None, checkpoint_prefix: Path | str | None = None,
    verbose: bool = False,
) -> tuple[list[dict], torch.nn.Module, pd.DataFrame]:
    """Train fresh initializations, resuming and retaining ``best``/``last`` per seed.

    ``checkpoint_prefix`` is scoped by the comparison caller to one condition,
    model and fold.  This deliberately retains the notebook's current layout;
    it does not introduce a retention policy for future large seed sweeps.
    """
    if L is None:  # pragma: no cover
        raise ImportError("run_seeded_training requires the lightning package")
    if seeds is None:
        seeds = list(range(n_seeds))
    checkpoint_prefix = Path(checkpoint_prefix) if checkpoint_prefix is not None else None
    histories, rows, best_model, best_loss = [], [], None, float("inf")

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = builder_fn()
        if frozen_state_dict is not None:
            model.full_model.mean_model.load_state_dict(copy.deepcopy(frozen_state_dict), strict=True)
            for parameter in model.full_model.mean_model.parameters():
                parameter.requires_grad = False

        metric_callback = MetricHistory()
        callbacks = [metric_callback, EarlyStopping(
            monitor="valid_loss_epoch", min_delta=conf.training.min_delta,
            patience=conf.training.patience, mode="min",
        )]
        checkpoint_callback = None
        resume_path = None
        if checkpoint_prefix is not None:
            checkpoint_dir = Path(f"{checkpoint_prefix}_seed{seed}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_callback = ModelCheckpoint(
                dirpath=str(checkpoint_dir), filename="best", save_last=True,
                monitor="valid_loss_epoch", mode="min", save_top_k=1,
            )
            candidate = checkpoint_dir / "last.ckpt"
            resume_path = str(candidate) if candidate.exists() else None
            callbacks.append(checkpoint_callback)
        if verbose:
            callbacks.append(TQDMEpochProgressBar())

        trainer = L.Trainer(
            max_epochs=conf.training.max_epoch, callbacks=callbacks, logger=False,
            enable_progress_bar=False, enable_checkpointing=checkpoint_callback is not None,
            enable_model_summary=False,
            accelerator="gpu" if torch.device(conf.device).type == "cuda" else "cpu", devices=1,
        )
        trainer.fit(model, train_loader, valid_loader, ckpt_path=resume_path)
        if checkpoint_callback is not None and checkpoint_callback.best_model_path:
            best_checkpoint = torch.load(checkpoint_callback.best_model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(best_checkpoint["state_dict"])

        history = metric_callback.history
        histories.append(history)
        losses = np.asarray(history.get("valid_loss_epoch", [np.nan]), dtype=float)
        train_losses = np.asarray(history.get("train_loss_epoch", [np.nan]), dtype=float)
        valid_mses = np.asarray(history.get("valid_mse_epoch", [np.nan]), dtype=float)
        valid_losses = losses[~np.isnan(losses)]
        final_loss = float(valid_losses[-1]) if valid_losses.size else np.nan
        final_train = train_losses[~np.isnan(train_losses)]
        final_mse = valid_mses[~np.isnan(valid_mses)]
        convergence_epoch = int(trainer.current_epoch)
        rows.append({
            "seed": seed, "final_valid_loss": final_loss,
            "final_train_loss": float(final_train[-1]) if final_train.size else np.nan,
            "final_valid_mse": float(final_mse[-1]) if final_mse.size else np.nan,
            "convergence_epoch": convergence_epoch,
            "mean_jitter": float(pd.Series(losses).rolling(5, min_periods=2).std().mean()),
            "stopped_early": convergence_epoch < conf.training.max_epoch - 1,
        })
        if final_loss < best_loss:
            best_loss, best_model = final_loss, copy.deepcopy(model)

    if best_model is None:
        raise RuntimeError("No seeded training run produced a model")
    return histories, best_model, pd.DataFrame(rows)
