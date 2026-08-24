"""Mean and covariance neural modules retained from the experiment notebook."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, conf):
        super().__init__()
        data = conf.data
        self.n_position_vars, self.n_dense_vars, self.n_sparse_vars = data.n_position_vars, data.n_dense_vars, data.n_sparse_vars
        self.n_vars, self.n_bins, self.n_units, self.device = data.n_vars, data.n_bins, data.n_units, conf.device


class Time2Vec(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.w0, self.b0 = nn.Parameter(torch.randn(1)), nn.Parameter(torch.randn(1))
        self.w, self.b = nn.Parameter(torch.randn(n_hidden - 1)), nn.Parameter(torch.randn(n_hidden - 1))
    def forward(self, times):
        times = times.unsqueeze(-1)
        return torch.cat([self.w0 * times + self.b0, torch.sin(self.w * times + self.b)], dim=-1)


class Time2VecPositionalEncoding(nn.Module):
    """Notebook positional encoder, retained as a named module for checkpoints."""
    def __init__(self, n_hidden):
        super().__init__(); self.n_hidden = n_hidden; self.t2v = Time2Vec(n_hidden)
    def forward(self, n_bins, device):
        return self.t2v(torch.arange(n_bins, device=device).float()).unsqueeze(0)


class Transformer:
    def __init__(self, conf, **kwargs):
        super().__init__(conf, **kwargs)
        settings = conf.model_type[conf.model_type.name]
        self.n_hidden = settings.n_hidden
        self.vars_proj = nn.Linear(self.n_vars, self.n_hidden)
        self.positional_encoding = Time2VecPositionalEncoding(self.n_hidden)
        layer = nn.TransformerEncoderLayer(self.n_hidden, settings.n_heads, self.n_hidden * 4, settings.dropout, settings.nonlinearity, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, settings.n_layers)
        self.dropout_layer = nn.Dropout(settings.dropout)
    def forward_transformer(self, x):
        position, dense, sparse = x
        dense = dense.unsqueeze(1).expand(-1, self.n_bins, -1)
        sparse = sparse.unsqueeze(1).expand(-1, self.n_bins, -1)
        stacked = self.vars_proj(torch.cat([position, dense, sparse], dim=2))
        encoding = self.positional_encoding(self.n_bins, stacked.device)
        return self.dropout_layer(self.transformer_encoder(stacked + encoding))


class MeanModel(Model):
    pass


class ZeroMeanModel(MeanModel):
    def forward(self, x):
        return torch.zeros(x[0].size(0), self.n_bins, self.n_units, device=x[0].device, dtype=x[0].dtype)


class BaselineMeanModel(MeanModel):
    def __init__(self, conf):
        super().__init__(conf); self.theta = nn.Parameter(torch.empty(self.n_bins, self.n_units, device=self.device)); nn.init.normal_(self.theta, std=.1)
    def forward(self, x): return self.theta.unsqueeze(0).expand(x[0].size(0), -1, -1)


class ConditionalMeanModel(Transformer, Model):
    def __init__(self, conf):
        copied = copy.deepcopy(conf); copied.model_type.name = "mean"; super().__init__(copied); self.head = nn.Linear(self.n_hidden, self.n_units)
    def forward(self, x): return self.head(self.forward_transformer(x))


class CovModel(Model):
    def __init__(self, conf):
        super().__init__(conf); self.n_latent = conf.model_type.cov.n_latent
        self.register_buffer("I", torch.eye(self.n_bins * self.n_units, device=self.device))
        self.length_scales, self.noise = nn.Parameter(torch.empty(self.n_latent, device=self.device)), nn.Parameter(torch.empty(self.n_bins, self.n_units, device=self.device))
        nn.init.normal_(self.length_scales, mean=math.log(math.expm1(.5)), std=1); nn.init.normal_(self.noise, mean=-20, std=1)
    def build_covariance_matrix(self, loadings):
        grid = torch.arange(self.n_bins, device=loadings.device, dtype=torch.float32)
        distances = grid[None, :] - grid[:, None]
        kernels = torch.exp(-.5 * (distances.unsqueeze(0) / F.softplus(self.length_scales).view(-1, 1, 1)) ** 2)
        blocks = torch.einsum("btok,kts,bsuk->btosu", loadings, kernels, loadings)
        covariance = blocks.reshape(loadings.size(0), self.n_bins * self.n_units, -1) + self.I.unsqueeze(0)
        covariance.diagonal(dim1=-2, dim2=-1).add_(torch.sigmoid(self.noise.flatten()))
        return torch.linalg.cholesky(covariance)


class IdentityCovModel(CovModel):
    def forward(self, x): return self.I.unsqueeze(0)


class SharedCovModel(CovModel):
    def __init__(self, conf):
        super().__init__(conf); self.lambda_matrix = nn.Parameter(torch.empty(self.n_bins, self.n_units, self.n_latent, device=self.device)); nn.init.normal_(self.lambda_matrix, std=.1)
    def forward(self, x): return self.build_covariance_matrix(self.lambda_matrix.unsqueeze(0))


class FullModel(Model):
    def __init__(self, conf, mean_model, cov_model):
        super().__init__(conf)
        self.mean_model = {"zero": ZeroMeanModel, "baseline": BaselineMeanModel, "conditional": ConditionalMeanModel}[mean_model](conf)
        self.cov_model = {"identity": IdentityCovModel, "shared": SharedCovModel}[cov_model](conf)
    def forward(self, x): return self.mean_model(x), self.cov_model(x)


class ConditionalGaussian(Model):
    def __init__(self, conf, variant): super().__init__(conf); self.variant = variant
    def predict(self, mean, covariance, y, **_):
        if self.variant == "mean": return mean
        if self.variant == "all": return self._predict_all(mean, covariance, y)
        if self.variant in {"past", "others", "past_and_others"}:
            return self._predict_subset(mean, covariance, y, self.variant)
        raise ValueError(f"Variant is not supported: {self.variant}")
    def predict_with_lambda(self, mean, precision, y):
        batched = mean.dim() == 3
        if not batched: mean, y = mean.unsqueeze(0), y.unsqueeze(0)
        batch, count = mean.size(0), self.n_bins * self.n_units
        precision = precision.expand(batch, -1, -1); diagonal = torch.diagonal(precision, dim1=-2, dim2=-1).unsqueeze(-1)
        off = precision.clone(); torch.diagonal(off, dim1=-2, dim2=-1).zero_()
        result = (mean.reshape(batch, count, 1) - off @ (y.reshape(batch, count, 1) - mean.reshape(batch, count, 1)) / (diagonal + 1e-8)).reshape_as(mean)
        return result if batched else result.squeeze(0)
    def _predict_all(self, mean, covariance, y):
        batched = mean.dim() == 3
        if not batched: mean, covariance, y = mean.unsqueeze(0), covariance.unsqueeze(0) if covariance.dim() == 2 else covariance, y.unsqueeze(0)
        result = self.predict_with_lambda(mean, torch.linalg.inv(covariance[:1] if covariance.size(0) == 1 else covariance), y)
        return result if batched else result.squeeze(0)

    def _predict_subset(self, mean, covariance, y, variant):
        """Exact conditional means using only the requested observed variables.

        This restores the notebook's public variants; unlike its unfinished
        private helpers, it has no undeclared index variables and works for
        both a shared covariance and one covariance per trial.
        """
        batched = mean.dim() == 3
        if not batched:
            mean, y = mean.unsqueeze(0), y.unsqueeze(0)
            covariance = covariance.unsqueeze(0) if covariance.dim() == 2 else covariance
        batch, count = mean.size(0), self.n_bins * self.n_units
        flat_mean, flat_y = mean.reshape(batch, count), y.reshape(batch, count)
        output = flat_mean.clone()
        for target in range(count):
            time, unit = divmod(target, self.n_units)
            indices = []
            if variant in {"past", "past_and_others"}:
                indices.extend(range(time * self.n_units))
            if variant in {"others", "past_and_others"}:
                indices.extend(time * self.n_units + other for other in range(self.n_units) if other != unit)
            if not indices:
                continue
            idx = torch.as_tensor(indices, device=mean.device)
            for b in range(batch):
                cov = covariance[0 if covariance.size(0) == 1 else b]
                cross, observed = cov[target, idx], cov[idx][:, idx]
                update = cross @ torch.linalg.solve(observed, (flat_y[b, idx] - flat_mean[b, idx]))
                output[b, target] = flat_mean[b, target] + update
        output = output.reshape_as(mean)
        return output if batched else output.squeeze(0)
