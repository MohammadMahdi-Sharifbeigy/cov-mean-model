"""Lightning wrapper, likelihood and prediction helpers."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:  # Keep selector-only tests importable in minimal environments.
    import lightning as L
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
except ImportError:  # pragma: no cover - exercised only without Lightning
    L = None

from mean_cov_pipeline.data.datasets import Anscombe
from mean_cov_pipeline.models.gaussian import ConditionalGaussian, FullModel


class MVNNLLLoss(nn.Module):
    def forward(self, out, y):
        mean, factor = out
        batch = mean.shape[0]; mean, y = mean.reshape(batch, -1), y.reshape(batch, -1)
        diff = (y - mean).unsqueeze(-1); dimensions = mean.shape[1]
        if factor.dim() == 2 or factor.shape[0] == 1:
            factor = factor.unsqueeze(0) if factor.dim() == 2 else factor
            solution = torch.linalg.solve_triangular(factor, diff.squeeze(-1).transpose(0, 1).unsqueeze(0), upper=False).squeeze(0).transpose(0, 1)
            logdet = 2 * torch.log(torch.diagonal(factor[0], dim1=-2, dim2=-1)).sum().expand(batch)
        else:
            solution = torch.linalg.solve_triangular(factor, diff, upper=False).squeeze(-1)
            logdet = 2 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1)
        return .5 * (dimensions * math.log(2 * math.pi) + logdet + (solution * solution).sum(-1)) / dimensions


if L is not None:
    class LitModel(L.LightningModule):
        def __init__(self, conf, mean_model, cov_model):
            super().__init__(); self.save_hyperparameters(ignore=["conf"]); self.optimizer_name = conf.optimization.optimizer_type.name
            self.optimizer_params = conf.optimization.optimizer_type[self.optimizer_name]
            scheduler_type = conf.optimization.scheduler_type
            self.scheduler_name = scheduler_type.name
            self.scheduler_params = scheduler_type[self.scheduler_name]
            self.full_model, self.mvn_nll_loss = FullModel(conf, mean_model, cov_model), MVNNLLLoss()
            self.anscombe = Anscombe()
        def forward(self, x): return self.full_model(x)
        def _step(self, batch, phase):
            x, y = batch; x = tuple(t.to(self.device) for t in x); y = y.to(self.device)
            eval_mean_model(self)
            output = self(x); loss = self.mvn_nll_loss(output, y).mean(); mse = F.mse_loss(output[0], y)
            if phase == "train":
                if not loss.requires_grad: loss.requires_grad_(True)
                self.log("train_loss_step", loss, on_step=True, on_epoch=False, logger=False)
                self.log("learning_rate_step", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=False, on_epoch=True, logger=False)
            self.log(f"{phase}_loss_epoch", loss, on_step=False, on_epoch=True, logger=False, prog_bar=True)
            self.log(f"{phase}_mse_epoch", mse, on_step=False, on_epoch=True, logger=False, prog_bar=True)
            return loss
        def training_step(self, batch, batch_idx): return self._step(batch, "train")
        def validation_step(self, batch, batch_idx): self._step(batch, "valid")
        def on_validation_epoch_end(self):
            loss = self.trainer.callback_metrics.get("valid_loss_epoch")
            if loss is not None and self.scheduler_name == "CosineReduce":
                self.scheduler.step_on_plateau(loss.item())
        def configure_optimizers(self):
            parameters = [p for p in self.parameters() if p.requires_grad]
            optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[self.optimizer_name]
            optimizer = optimizer_class(parameters, lr=self.optimizer_params.lr, weight_decay=self.optimizer_params.weight_decay)
            if self.scheduler_name == "Reduce":
                scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=self.scheduler_params.factor, patience=self.scheduler_params.patience, min_lr=self.scheduler_params.min_lr)
            elif self.scheduler_name == "Cosine":
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=self.scheduler_params.T_mult, eta_min=self.scheduler_params.eta_min, last_epoch=self.scheduler_params.last_epoch)
            else:
                return optimizer
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "valid_loss_epoch", "interval": "epoch", "frequency": 1}}
else:
    class LitModel(nn.Module):
        def __init__(self, *_args, **_kwargs): raise ImportError("LitModel requires the lightning package")


@torch.inference_mode()
def predict_loader(conf, lit_model, loader, y_mean, variant="all"):
    device = conf.device; model = lit_model.to(device).eval(); conditional = ConditionalGaussian(conf, variant).to(device); transform = Anscombe().to(device)
    targets, predictions, precision = [], [], None
    for x, y in loader:
        x, y = tuple(item.to(device) for item in x), y.to(device)
        mean, factor = model(x); covariance = factor @ factor.transpose(-1, -2)
        if covariance.size(0) == 1 and variant == "all":
            precision = torch.linalg.inv(covariance) if precision is None else precision
            prediction = conditional.predict_with_lambda(mean, precision, y)
        else: prediction = conditional.predict(mean, covariance, y)
        targets.append(transform.inv(y + y_mean)); predictions.append(transform.inv(prediction + y_mean))
    return torch.cat(targets).cpu().numpy(), torch.cat(predictions).cpu().numpy()


def trial_metrics(targets: np.ndarray, predictions: np.ndarray):
    mse = np.mean((predictions - targets) ** 2, axis=1).T
    centered_target = targets - targets.mean(axis=1, keepdims=True); centered_prediction = predictions - predictions.mean(axis=1, keepdims=True)
    denominator = np.sqrt((centered_target ** 2).sum(axis=1) * (centered_prediction ** 2).sum(axis=1))
    pearson = np.divide((centered_target * centered_prediction).sum(axis=1), denominator, out=np.zeros_like(denominator), where=denominator > 1e-8).T
    total = (centered_target ** 2).sum(axis=1); r2 = np.divide(total - ((targets - predictions) ** 2).sum(axis=1), total, out=np.zeros_like(total), where=total > 1e-8).T
    return pearson, mse, r2


def eval_mean_model(module):
    """Keep a frozen two-stage mean network in evaluation mode."""
    if not any(parameter.requires_grad for parameter in module.full_model.mean_model.parameters()):
        module.full_model.mean_model.eval()
