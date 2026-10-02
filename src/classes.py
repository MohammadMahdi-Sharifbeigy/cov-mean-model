from __future__ import annotations

import copy
import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning.pytorch.callbacks import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset


class DotDict(dict):
    def __init__(self, d=None):
        super().__init__()
        if d:
            for k, v in d.items():
                self[k] = self._wrap(v)

    def _wrap(self, value):
        if isinstance(value, dict):
            return DotDict(value)
        if isinstance(value, list):
            return [self._wrap(v) for v in value]
        return value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = self._wrap(value)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)

    def to_dict(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, DotDict):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, DotDict) else x for x in v]
            else:
                out[k] = v
        return out

    def __deepcopy__(self, memo):
        return DotDict(copy.deepcopy(self.to_dict(), memo))


class Anscombe(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 2.0 * torch.sqrt(x + 3.0 / 8.0)

    def inv(self, x):
        return (x / 2.0) ** 2 - 3.0 / 8.0

     
class NeuralDataset(Dataset):
    def __init__(self, x_position_vars, x_dense_vars, x_sparse_vars, Y, mean):
        self.x_position_vars = torch.tensor(x_position_vars, dtype=torch.float32).transpose(1, 2)
        self.x_dense_vars = torch.tensor(x_dense_vars, dtype=torch.float32)
        self.x_sparse_vars = torch.tensor(x_sparse_vars, dtype=torch.float32)
        
        Y_tensor = torch.tensor(Y, dtype=torch.float32).transpose(1, 2)
        anscombe = Anscombe()
        Y_anscombed = anscombe.forward(Y_tensor)
        self.Y = Y_anscombed - mean.cpu()
        
    def __len__(self):
        return self.x_position_vars.size(0)

    def __getitem__(self, idx):
        X = (self.x_position_vars[idx], self.x_dense_vars[idx], self.x_sparse_vars[idx])
        y = self.Y[idx]
        return X, y


class MVNNLLLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()

    def forward(self, out, y):
        mean, L = out
        
        if not torch.is_tensor(mean):
            mean = torch.stack(list(mean), dim=0)
        if not torch.is_tensor(y):
            y = torch.stack(list(y), dim=0)
        if not torch.is_tensor(L):
            L = torch.stack(list(L), dim=0)

        B = mean.shape[0]
        mean = mean.reshape(B, -1)
        y = y.reshape(B, -1)
        D = mean.shape[1]

        if L.dim() == 2:
            L = L.unsqueeze(0).expand(B, -1, -1)

        diff = (y - mean).unsqueeze(-1)
        sol = torch.linalg.solve_triangular(L, diff, upper=False).squeeze(-1)
        maha = (sol * sol).sum(dim=-1)
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
        log_prob = -0.5 * (D * math.log(2.0 * math.pi) + logdet + maha)
        loss = (-log_prob / D)
        
        return loss


class Model(nn.Module):
    def __init__(self, Conf=None, **kwargs):
        super().__init__()
        self.n_position_vars = Conf.data.n_position_vars
        self.n_dense_vars = Conf.data.n_dense_vars
        self.n_sparse_vars = Conf.data.n_sparse_vars
        self.n_vars = Conf.data.n_vars
        self.n_bins = Conf.data.n_bins
        self.n_units = Conf.data.n_units
        self.device = Conf.device


class Time2Vec(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(n_hidden - 1))
        self.b = nn.Parameter(torch.randn(n_hidden - 1))

    def forward(self, t):
        t = t.unsqueeze(-1)
        linear = self.w0 * t + self.b0
        periodic = torch.sin(self.w * t + self.b)
        return torch.cat([linear, periodic], dim=-1)


class Time2VecPositionalEncoding(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.t2v = Time2Vec(n_hidden)

    def forward(self, n_bins, device):
        t = torch.arange(n_bins, device=device).float()
        enc = self.t2v(t)
        return enc.unsqueeze(0)


class Transformer():
    def __init__(self, Conf=None, **kwargs):
        super().__init__(Conf=Conf, **kwargs)
        self.n_hidden = Conf.model_type[Conf.model_type.name].n_hidden
        self.n_heads = Conf.model_type[Conf.model_type.name].n_heads
        self.n_layers = Conf.model_type[Conf.model_type.name].n_layers
        self.nonlinearity = Conf.model_type[Conf.model_type.name].nonlinearity
        self.dropout = Conf.model_type[Conf.model_type.name].dropout

        self.vars_proj = nn.Linear(self.n_vars, self.n_hidden)
        
        self.positional_encoding = Time2VecPositionalEncoding(self.n_hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_hidden,
            nhead=self.n_heads,
            dim_feedforward=self.n_hidden * 4,
            dropout=self.dropout,
            activation=self.nonlinearity,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.dropout_layer = nn.Dropout(self.dropout)
        

    def forward_transformer(self, x):
        x_position_vars, x_dense_vars, x_sparse_vars = x
        
        dense_vars = x_dense_vars.unsqueeze(1).expand(-1, self.n_bins, -1)
        sparse_vars = x_sparse_vars.unsqueeze(1).expand(-1, self.n_bins, -1)
        stacked = torch.cat([x_position_vars, dense_vars, sparse_vars], dim=2)
        stacked = self.vars_proj(stacked)
        
        pos_enc = self.positional_encoding(self.n_bins, stacked.device)
        combined = stacked + pos_enc
        transformed = self.transformer_encoder(combined)
        transformed = self.dropout_layer(transformed)
        return transformed


class MeanModel(Model):
    def __init__(self, Conf):
        super().__init__(Conf)


class ZeroMeanModel(MeanModel):
    def __init__(self, Conf):
        super().__init__(Conf)

    def forward(self, x):
        batchsize = x[0].size(0)
        return torch.zeros(batchsize, self.n_bins, self.n_units, device=x[0].device, dtype=x[0].dtype)


class BaselineMeanModel(MeanModel):
    def __init__(self, Conf):
        super().__init__(Conf)
        self.theta = nn.Parameter(torch.empty(self.n_bins, self.n_units, device=self.device))

        with torch.no_grad():
            nn.init.normal_(self.theta, mean=0, std=1e-1)
            
    def forward(self, x):
        batch_size = x[0].size(0)
        mean = self.theta
        return mean.view(1, self.n_bins, self.n_units).expand(batch_size, -1, -1)


class ConditionalMeanModel(Transformer, MeanModel):
    def __init__(self, Conf):
        Conf_ = copy.deepcopy(Conf)
        Conf_.model_type.name = "mean"
        super().__init__(Conf=Conf_)

        self.output_size = self.n_units
        self.head = nn.Linear(self.n_hidden, self.output_size)

    def forward(self, x):
        transformed = self.forward_transformer(x)
        mean = self.head(transformed)
        return mean
             

class CovModel(Model):
    def __init__(self, Conf):
        super().__init__(Conf)

        self.register_buffer(
            "I",
            torch.eye(
                self.n_bins * self.n_units,
                dtype=torch.float32,
                device=self.device,
            ),
        )

        self.noise = nn.Parameter(
            torch.empty(
                self.n_bins,
                self.n_units,
                device=self.device,
            )
        )

        with torch.no_grad():
            with torch.no_grad():nn.init.normal_(self.noise, mean=math.log(0.01 / (1.0 - 0.01)), std=0.1)

    def kernel(self, fun, n_samples1, n_samples2, length_scales):
        i_grid = torch.arange(
            n_samples1,
            dtype=torch.float32,
            device=length_scales.device,
        ).view(-1, 1)

        j_grid = torch.arange(
            n_samples2,
            dtype=torch.float32,
            device=length_scales.device,
        ).view(1, -1)

        return fun(i_grid, j_grid, length_scales)

    def squared_exponential_kernel(self, x1, x2, length_scales):
        distances = (x2 - x1).unsqueeze(0)
        fractions = distances / length_scales[:, None, None]

        return torch.exp(-0.5 * fractions.square())

    def build_component_covariance(self, lambda_matrix, length_scales):

        length_scales = F.softplus(length_scales)

        K = self.kernel(
            self.squared_exponential_kernel,
            self.n_bins,
            self.n_bins,
            length_scales,
        )

        blocks = torch.einsum(
            "tuk,kts,svk->tusv",
            lambda_matrix,
            K,
            lambda_matrix,
        )

        return blocks.reshape(
            self.n_bins * self.n_units,
            self.n_bins * self.n_units,
        )

    def add_noise_and_cholesky(self, covariance):
        covariance = covariance + self.I

        noise_diag = torch.sigmoid(self.noise.flatten())
        covariance.diagonal().add_(noise_diag)

        return torch.linalg.cholesky(covariance)


class FunctionalGroupCovModel(CovModel):
    def __init__(self, Conf):
        super().__init__(Conf)

        group_labels = list(Conf.data.unit_groups)
        self.component_configs = Conf.model_type.cov.components

        self.group_labels = tuple(sorted(set(group_labels)))

        self.component_unit_indices = {}
        self.component_loadings = nn.ParameterDict()
        self.component_length_scales = nn.ParameterDict()

        for component_name, component_config in self.component_configs.items():
            allowed_groups = set(component_config.groups)
            n_latent = int(component_config.n_latent)

            unit_indices = torch.tensor(
                [
                    unit_idx
                    for unit_idx, unit_group in enumerate(group_labels)
                    if unit_group in allowed_groups
                ],
                dtype=torch.long,
                device=self.device,
            )
            
            self.component_unit_indices[component_name] = unit_indices

            self.component_loadings[component_name] = nn.Parameter(
                torch.empty(
                    self.n_bins,
                    unit_indices.numel(),
                    n_latent,
                    device=self.device,
                )
            )

            self.component_length_scales[component_name] = nn.Parameter(
                torch.empty(
                    n_latent,
                    device=self.device,
                )
            )

            with torch.no_grad():
                nn.init.normal_(
                    self.component_loadings[component_name],
                    mean=0.0,
                    std=0.1,
                )

                initial_length_scale = math.log(math.expm1(0.5))
                self.component_length_scales[component_name].fill_(
                    initial_length_scale
                )

    def full_loading_matrix(self, component_name):
        unit_idx = self.component_unit_indices[component_name]

        local_loadings = self.component_loadings[component_name]

        lambda_matrix = torch.zeros(
            self.n_bins,
            self.n_units,
            local_loadings.size(-1),
            dtype=local_loadings.dtype,
            device=local_loadings.device,
        )

        lambda_matrix[:, unit_idx, :] = local_loadings

        return lambda_matrix
    
    def covariance_matrix(self):
        first_component_name = next(iter(self.component_configs))

        first_loading = self.component_loadings[
            first_component_name
        ]

        covariance = torch.zeros(
            self.n_bins * self.n_units,
            self.n_bins * self.n_units,
            dtype=first_loading.dtype,
            device=first_loading.device,
        )

        for component_name in self.component_configs:
            lambda_matrix = self.full_loading_matrix(
                component_name
            )

            length_scales = self.component_length_scales[
                component_name
            ]

            covariance = covariance + self.build_component_covariance(
                lambda_matrix,
                length_scales,
            )

        return covariance

    def forward(self, x):
        batch_size = x[0].size(0)

        covariance = self.covariance_matrix()
        L = self.add_noise_and_cholesky(covariance)

        return L.unsqueeze(0).expand(batch_size, -1, -1)


class IdentityCovModel(CovModel):
    def __init__(self, Conf):
        super().__init__(Conf)

    def forward(self, x):
        batch_size = x[0].size(0)
        L = self.I.unsqueeze(0).expand(batch_size, -1, -1)
        return L
            

class SharedCovModel(CovModel):
    def __init__(self, Conf):
        super().__init__(Conf)

        self.n_latent = Conf.model_type.cov.n_latent

        self.lambda_matrix = nn.Parameter(
            torch.empty(
                self.n_bins,
                self.n_units,
                self.n_latent,
                device=self.device,
            )
        )

        self.length_scales = nn.Parameter(
            torch.empty(
                self.n_latent,
                device=self.device,
            )
        )

        with torch.no_grad():
            nn.init.normal_(
                self.lambda_matrix,
                mean=0.0,
                std=0.1,
            )

            initial_length_scale = math.log(math.expm1(0.5))
            self.length_scales.fill_(initial_length_scale)

    def forward(self, x):
        batch_size = x[0].size(0)

        covariance = self.build_component_covariance(
            self.lambda_matrix,
            self.length_scales,
        )

        L = self.add_noise_and_cholesky(covariance)

        return L.unsqueeze(0).expand(batch_size, -1, -1)


class FullModel(Model):
    def __init__(self, Conf, mean_model, cov_model):
        super().__init__(Conf)
        if mean_model == 'zero':
            self.mean_model = ZeroMeanModel(Conf)
        elif mean_model == 'baseline':
            self.mean_model = BaselineMeanModel(Conf)
        elif mean_model == 'conditional':
            self.mean_model = ConditionalMeanModel(Conf)

        if cov_model == 'identity':
            self.cov_model = IdentityCovModel(Conf)
        elif cov_model == 'shared':
            self.cov_model = SharedCovModel(Conf)
        elif cov_model == "functional_groups":
            self.cov_model = FunctionalGroupCovModel(Conf)
    
    def forward(self, x):
        mean = self.mean_model(x)
        L = self.cov_model(x)
        return mean, L


class ConditionalGaussian(Model):
    def __init__(self, Conf, variant):
        super().__init__(Conf)
        self.variant = variant

    def predict(self, mean, cov, y, **kwargs):
        if self.variant == 'mean':
            return mean
        elif self.variant == 'past':
            return self.__predict_by_past(mean, cov, y)
        elif self.variant == 'others':
            return self.__predict_by_others(mean, cov, y)
        elif self.variant == 'past_and_others':
            return self.__predict_by_past_and_others(mean, cov, y)
        elif self.variant == 'all':
            return self.__predict_by_all(mean, cov, y)
        else:
            raise ValueError(f'Variant is not supported: {self.variant}')
    
    def __predict_by_all(self, mean, cov, y, max_t_past_prediction=None):
        # Add batching support if mean is (batch, bins, units)
        is_batched = mean.dim() == 3
        if not is_batched:
            mean = mean.unsqueeze(0)
            if cov.dim() == 2:
                cov = cov.unsqueeze(0)
            y = y.unsqueeze(0)
            
        B = mean.size(0)
        N = self.n_bins * self.n_units
        y_flat = y.reshape(B, N, 1)
        mean_flat = mean.reshape(B, N, 1)
        
        is_cov_shared = (cov.shape[0] == 1) or (cov.dim() == 2)
        cov_to_use = cov[0:1] if is_cov_shared else cov

        # 1. Compute Precision Matrix (\Lambda = \Sigma^{-1}) ONLY ONCE per batch!
        # Memory footprint drops from 1.29 GiB -> ~25 MB!
        Lambda = torch.linalg.inv(cov_to_use) # shape: (1 or B, N, N)
        
        # 2. Extract diagonal precision values (\Lambda_{kk})
        Lambda_diag = torch.diagonal(Lambda, dim1=-2, dim2=-1).unsqueeze(-1)
        
        # 3. Zero out the diagonal to isolate off-diagonal interactions (\Lambda_{k, -k})
        Lambda_off = Lambda.clone()
        torch.diagonal(Lambda_off, dim1=-2, dim2=-1).zero_()
        
        # 4. Apply conditioning formula in one single matrix multiplication:
        #    update = - (1 / \Lambda_{kk}) * \Lambda_{k, -k} (y_{-k} - \mu_{-k})
        diff = y_flat - mean_flat
        update = -torch.matmul(Lambda_off, diff) / (Lambda_diag + 1e-8)
        
        conditioned_mean = (mean_flat + update).reshape(B, self.n_bins, self.n_units)
        
        if not is_batched:
            conditioned_mean = conditioned_mean.squeeze(0)
            
        return conditioned_mean
    
    def __predict_by_past(self, mean, cov, y, max_t_past_prediction=None):
        y_flat = y.flatten()
        mean_flat = mean.flatten()
        conditioned_mean = mean.clone().reshape((-1, self.n_units))
    
        for t in range(1, conditioned_mean.shape[0]):
            conditioned_indices = torch.arange(t * self.n_units, device=mean.device).reshape(t, self.n_units).t()
            product_indices = torch.stack((
                conditioned_indices.flatten().repeat_interleave(t).reshape(self.n_units, t, t),
                conditioned_indices.unsqueeze(1).repeat(1, t, 1)
            ), dim=-1)
    
            row_indices = torch.arange(t * self.n_units, (t + 1) * self.n_units, device=mean.device)
            cross_cov = cov[row_indices.unsqueeze(-1), conditioned_indices].unsqueeze(1)
            partial_cov = cov[product_indices[..., 0], product_indices[..., 1]].reshape(self.n_units, t, t)
            diff = (y_flat[conditioned_indices] - mean_flat[conditioned_indices]).unsqueeze(-1)
    
            conditioned_mean[t] += torch.matmul(torch.matmul(cross_cov, torch.linalg.inv(partial_cov)), diff).reshape(self.n_units)
        return conditioned_mean
    
    def __predict_by_others(self, mean, cov, y):
        nr_others = self.n_units - 1
        y_flat = y.flatten()
        mean_flat = mean.flatten()
        conditioned_mean = mean.clone().reshape((-1, self.n_units))
    
        for t in range(conditioned_mean.shape[0]):
            time_indices = torch.arange(t * self.n_units, (t + 1) * self.n_units, device=mean.device)
            conditioned_indices = time_indices.unsqueeze(0).repeat(self.n_units, 1)
            conditioned_indices = conditioned_indices[~torch.eye(self.n_units, dtype=torch.bool, device=mean.device)].reshape(self.n_units, nr_others)
    
            product_indices = torch.stack((
                conditioned_indices.flatten().repeat_interleave(nr_others).reshape(self.n_units, nr_others, nr_others),
                conditioned_indices.unsqueeze(1).repeat(1, nr_others, 1)
            ), dim=-1)
    
            cross_cov = cov[time_indices.unsqueeze(-1), conditioned_indices].unsqueeze(1)
            partial_cov = cov[product_indices[..., 0], product_indices[..., 1]].reshape(self.n_units, nr_others, nr_others)
            diff = (y_flat[conditioned_indices] - mean_flat[conditioned_indices]).unsqueeze(-1)
    
            conditioned_mean[t] += torch.matmul(torch.matmul(cross_cov, torch.linalg.inv(partial_cov)), diff).reshape(self.n_units)
        
        return conditioned_mean
    
    def __predict_by_past_and_others(self, mean, cov, y, max_t_past_prediction=None):
        nr_others = self.n_units - 1
        y_flat = y.flatten()
        mean_flat = mean.flatten()
        conditioned_mean = mean.clone().reshape((-1, self.n_units))
    
        for t in range(conditioned_mean.shape[0]):
            nr_past_and_others = t + nr_others
    
            past_indices = torch.arange(t * self.n_units, device=mean.device).reshape(t, self.n_units).t()
            time_indices = torch.arange(t * self.n_units, (t + 1) * self.n_units, device=mean.device)
            others_indices = time_indices.unsqueeze(0).repeat(self.n_units, 1)
            others_indices = others_indices[~torch.eye(self.n_units, dtype=torch.bool, device=mean.device)].reshape(self.n_units, nr_others)
    
            conditioned_indices = torch.cat((past_indices, others_indices), dim=-1)

            product_indices = torch.stack((
                conditioned_indices.flatten().repeat_interleave(nr_past_and_others).reshape(self.n_units, nr_past_and_others, nr_past_and_others),
                conditioned_indices.unsqueeze(1).repeat(1, nr_past_and_others, 1)
            ), dim=-1)
    
            cross_cov = cov[time_indices.unsqueeze(-1), conditioned_indices].unsqueeze(1)
            partial_cov = cov[product_indices[..., 0], product_indices[..., 1]].reshape(self.n_units, nr_past_and_others, nr_past_and_others)
            diff = (y_flat[conditioned_indices] - mean_flat[conditioned_indices]).unsqueeze(-1)
    
            conditioned_mean[t] += torch.matmul(torch.matmul(cross_cov, torch.linalg.inv(partial_cov)), diff).reshape(self.n_units)
        
        return conditioned_mean


def eval_mean_model(module):
    if not any(p.requires_grad for p in module.full_model.mean_model.parameters()):
        module.full_model.mean_model.eval()

class History(L.Callback):
    def __init__(self):
        self.train_loss_epoch = []
        self.valid_loss_epoch = []
        self.lr = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss_epoch.append(trainer.callback_metrics["train_loss_epoch"].item())

    def on_validation_end(self, trainer, pl_module):
        self.valid_loss_epoch.append(trainer.callback_metrics["valid_loss_epoch"].item())
        self.lr.append(trainer.optimizers[0].param_groups[0]["lr"])

class LitModel(L.LightningModule):
    def __init__(self, Conf, mean_model, cov_model):
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer_type", "scheduler_type", "data", "model_params"])        
        self.optimizer_params = Conf.optimization.Adam
        self.scheduler_params = Conf.optimization.Reduce
        self.full_model = FullModel(Conf, mean_model, cov_model)
        self.mvn_nll_loss = MVNNLLLoss()
        self.anscombe = Anscombe()
     
    def forward(self, x):
        return self.full_model(x)

    def training_step(self, batch):
        x, y = batch
        x = tuple(t.to(self.device, non_blocking=True) for t in x)
        y = y.to(self.device, non_blocking=True)
        
        eval_mean_model(self)
        out = self(x)
        loss = self.mvn_nll_loss(out, y).mean()
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, logger=False, prog_bar=True)
        
        return loss

    def validation_step(self, batch):
        x, y = batch
        x = tuple(t.to(self.device, non_blocking=True) for t in x)
        y = y.to(self.device, non_blocking=True)
        
        eval_mean_model(self)
        out = self(x)
        loss = self.mvn_nll_loss(out, y).mean()
        
        self.log("valid_loss_epoch", loss, on_step=False, on_epoch=True, logger=False, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.optimizer_params.lr,
            weight_decay=self.optimizer_params.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_params.factor,
            patience=self.scheduler_params.patience,
            min_lr=self.scheduler_params.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss_epoch",
                "interval": "epoch",
                "frequency": 1,
            },
        }

