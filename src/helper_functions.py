from __future__ import annotations

import copy
import os
import random
import pickle
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from torch.utils.data import DataLoader

from tqdm import tqdm
from src.classes import (
    Anscombe,
    ConditionalGaussian,
    History,
    LitModel,
    NeuralDataset,
)

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)


def save_cache(**variables):
    cache_path = Path(".") / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)

    for name, variable in variables.items():
        file_path = cache_path / f"{name}.pkl"

        with open(file_path, "wb") as file:
            pickle.dump(variable, file)

        print(f"Saved: {name}")


def load_cache(name):
    file_path = Path(".") / "cache" / f"{name}.pkl"

    if not file_path.exists():
        print(f"Variable '{name}' was not found in cache.")
        return None

    with open(file_path, "rb") as file:
        variable = pickle.load(file)

    print(f"Loaded: {name}")
    return variable


def prepare_dataset(Conf, x_position_vars, x_dense_vars, x_sparse_vars, Y):

    seed_everything(Conf.seed)

    split_rng = np.random.default_rng(Conf.seed)
    train_idx = split_rng.choice(Conf.data.n_trials, size=int(Conf.data.n_trials * 0.8), replace=False)
    valid_idx = np.setdiff1d(np.arange(Conf.data.n_trials), train_idx)

    x_position_vars_train = x_position_vars[train_idx]
    x_position_vars_valid = x_position_vars[valid_idx]

    x_dense_vars_train = x_dense_vars[train_idx]
    x_dense_vars_valid = x_dense_vars[valid_idx]

    x_sparse_vars_train = x_sparse_vars[train_idx]
    x_sparse_vars_valid = x_sparse_vars[valid_idx]

    Y_train = Y[train_idx]
    Y_valid = Y[valid_idx]

    position_Y_mean = x_position_vars_train.mean(axis=0, keepdims=True)
    position_std = x_position_vars_train.std(axis=0, keepdims=True)
    x_position_vars_train = (x_position_vars_train - position_Y_mean) / (position_std + 1e-8)
    x_position_vars_valid = (x_position_vars_valid - position_Y_mean) / (position_std + 1e-8)

    dense_task_Y_mean = x_dense_vars_train.mean(axis=0, keepdims=True)
    dense_task_std = x_dense_vars_train.std(axis=0, keepdims=True)
    x_dense_vars_train = (x_dense_vars_train - dense_task_Y_mean) / (dense_task_std + 1e-8)
    x_dense_vars_valid = (x_dense_vars_valid - dense_task_Y_mean) / (dense_task_std + 1e-8)

    sparse_task_mean = x_sparse_vars_train.mean(axis=0, keepdims=True)
    sparse_task_std = x_sparse_vars_train.std(axis=0, keepdims=True)
    x_sparse_vars_train = (x_sparse_vars_train - sparse_task_mean) / (sparse_task_std + 1e-8)
    x_sparse_vars_valid = (x_sparse_vars_valid - sparse_task_mean) / (sparse_task_std + 1e-8)

    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32, device=Conf.device).transpose(1, 2)
    anscombe = Anscombe()
    Y_train_anscombed = anscombe.forward(Y_train_tensor)
    Y_mean = torch.mean(Y_train_anscombed, dim=(0, 1), keepdim=True)    

    train_dataset = NeuralDataset(x_position_vars_train, x_dense_vars_train, x_sparse_vars_train, Y_train, Y_mean)
    valid_dataset = NeuralDataset(x_position_vars_valid, x_dense_vars_valid, x_sparse_vars_valid, Y_valid, Y_mean)

    return train_dataset, valid_dataset, Y_mean



def _build_datasets_from_indices(
    Conf,
    x_position_vars,
    x_dense_vars,
    x_sparse_vars,
    Y,
    train_idx,
    valid_idx,
):
    """Shared normalization + NeuralDataset construction for a given train/valid split.

    Used internally by prepare_kfold_datasets. Contains the same normalization
    and Anscombe-transform logic as prepare_dataset, kept separate so that
    prepare_dataset itself remains untouched.

    Returns
    -------
    train_dataset : NeuralDataset
    valid_dataset : NeuralDataset
    Y_mean        : torch.Tensor  shape (1, 1, n_units)
    """
    x_position_vars_train = x_position_vars[train_idx]
    x_position_vars_valid = x_position_vars[valid_idx]

    x_dense_vars_train = x_dense_vars[train_idx]
    x_dense_vars_valid = x_dense_vars[valid_idx]

    x_sparse_vars_train = x_sparse_vars[train_idx]
    x_sparse_vars_valid = x_sparse_vars[valid_idx]

    Y_train = Y[train_idx]
    Y_valid = Y[valid_idx]

    # Normalise using train-split statistics only
    position_mean = x_position_vars_train.mean(axis=0, keepdims=True)
    position_std  = x_position_vars_train.std(axis=0, keepdims=True)
    x_position_vars_train = (x_position_vars_train - position_mean) / (position_std + 1e-8)
    x_position_vars_valid = (x_position_vars_valid - position_mean) / (position_std + 1e-8)

    dense_mean = x_dense_vars_train.mean(axis=0, keepdims=True)
    dense_std  = x_dense_vars_train.std(axis=0, keepdims=True)
    x_dense_vars_train = (x_dense_vars_train - dense_mean) / (dense_std + 1e-8)
    x_dense_vars_valid = (x_dense_vars_valid - dense_mean) / (dense_std + 1e-8)

    sparse_mean = x_sparse_vars_train.mean(axis=0, keepdims=True)
    sparse_std  = x_sparse_vars_train.std(axis=0, keepdims=True)
    x_sparse_vars_train = (x_sparse_vars_train - sparse_mean) / (sparse_std + 1e-8)
    x_sparse_vars_valid = (x_sparse_vars_valid - sparse_mean) / (sparse_std + 1e-8)

    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32, device=Conf.device).transpose(1, 2)
    anscombe = Anscombe()
    Y_mean = torch.mean(anscombe.forward(Y_train_tensor), dim=(0, 1), keepdim=True)

    train_dataset = NeuralDataset(
        x_position_vars_train, x_dense_vars_train, x_sparse_vars_train, Y_train, Y_mean
    )
    valid_dataset = NeuralDataset(
        x_position_vars_valid, x_dense_vars_valid, x_sparse_vars_valid, Y_valid, Y_mean
    )

    return train_dataset, valid_dataset, Y_mean


def prepare_kfold_datasets(
    Conf,
    x_position_vars,
    x_dense_vars,
    x_sparse_vars,
    Y,
    k: int = 10,
):
    """K-fold cross-validation split with the same seeding and normalization
    discipline as :func:`prepare_dataset`.

    Indices are shuffled once with ``np.random.default_rng(Conf.seed)``, then
    cut into *k* equal (±1) consecutive blocks.  For fold *i*:
    - validation = block *i*
    - training   = all other blocks

    ``prepare_dataset`` is left completely untouched.

    Parameters
    ----------
    Conf : DotDict
        Must have ``Conf.seed``, ``Conf.data.n_trials``, and ``Conf.device``.
    x_position_vars : ndarray  (T, P, B)
    x_dense_vars    : ndarray  (T, D)
    x_sparse_vars   : ndarray  (T, S)
    Y               : ndarray  (T, N, B)
    k               : int, default 10

    Returns
    -------
    folds : list of k tuples ``(train_dataset, valid_dataset, Y_mean)``
        Each element has the same structure as the return value of
        ``prepare_dataset``.
    """
    seed_everything(Conf.seed)

    n_trials = Conf.data.n_trials
    rng = np.random.default_rng(Conf.seed)
    shuffled_idx = rng.permutation(n_trials)

    # Split shuffled indices into k equal (±1) blocks
    fold_sizes = np.full(k, n_trials // k, dtype=int)
    fold_sizes[: n_trials % k] += 1          # distribute the remainder
    fold_blocks = np.split(shuffled_idx, np.cumsum(fold_sizes)[:-1])

    folds = []
    for fold_i in range(k):
        valid_idx = fold_blocks[fold_i]
        train_idx = np.concatenate(
            [fold_blocks[j] for j in range(k) if j != fold_i]
        )
        train_ds, valid_ds, Y_mean = _build_datasets_from_indices(
            Conf,
            x_position_vars,
            x_dense_vars,
            x_sparse_vars,
            Y,
            train_idx,
            valid_idx,
        )
        folds.append((train_ds, valid_ds, Y_mean))

    return folds


def shuffle_dataset(Conf, dataset):

    generator = torch.Generator()
    generator.manual_seed(Conf.seed)

    indices = torch.randperm(len(dataset), generator=generator)

    dataset_shuffle = copy.deepcopy(dataset)
    dataset_shuffle.Y = dataset.Y[indices]

    return dataset_shuffle


def prepare_loader(Conf, train_dataset, valid_dataset):

    seed_everything(Conf.seed)

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(Conf.seed)
    train_loader = DataLoader(train_dataset, batch_size=Conf.training.batch_size, shuffle=True, generator=train_loader_generator, num_workers=0, pin_memory=True, persistent_workers=False)
    valid_loader_generator = torch.Generator()
    valid_loader_generator.manual_seed(Conf.seed)
    valid_loader = DataLoader(valid_dataset, batch_size=Conf.training.batch_size, shuffle=False, generator=valid_loader_generator, num_workers=0, pin_memory=True, persistent_workers=False)

    return train_loader, valid_loader, (train_loader_generator, valid_loader_generator)


def build_lit_model(Conf, loader_generators, mean_model, cov_model, enable_progress_bar_epoch, checkpoint_dir=None):

    train_loader_generator, valid_loader_generator = loader_generators
    seed_everything(Conf.seed)
    train_loader_generator.manual_seed(Conf.seed)
    valid_loader_generator.manual_seed(Conf.seed)


    early_stop = EarlyStopping(
        monitor="valid_loss_epoch",
        mode="min",
        min_delta=Conf.training.min_delta,
        patience=Conf.training.patience,
    )

    history = History()
    
    callbacks = [early_stop, history]
    logger = False
    enable_checkpointing = False

    if checkpoint_dir is not None:
        from lightning.pytorch.callbacks import ModelCheckpoint
        from lightning.pytorch.loggers import CSVLogger
        import os
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_cb = ModelCheckpoint(
            dirpath=os.path.join(checkpoint_dir, "checkpoints"),
            filename="best-{epoch:02d}-{valid_loss_epoch:.4f}",
            monitor="valid_loss_epoch",
            mode="min",
            save_top_k=1,
            save_last=True
        )
        callbacks.append(checkpoint_cb)
        logger = CSVLogger(save_dir=checkpoint_dir, name="logs")
        enable_checkpointing = True

    trainer = L.Trainer(
        max_epochs=Conf.training.max_epoch,
        accelerator="gpu" if Conf.device.type == "cuda" else "cpu",
        devices=1,
        precision="16-mixed",
        deterministic=True,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar_epoch,
        enable_checkpointing=enable_checkpointing,
        enable_model_summary=False,
        check_val_every_n_epoch=1,
    )
    
    lit_model = LitModel(Conf, mean_model, cov_model).to(Conf.device)
    trainer.history = history
    
    return trainer, lit_model


def compute_shap_values(Conf, lit_model, background_dataset, explain_dataset, n_permutations=100):

    seed = Conf.seed
    seed_everything(seed)
    device = Conf.device
    shap_generator = torch.Generator(device=device)
    shap_generator.manual_seed(Conf.seed)

    n_bins = Conf.data.n_bins
    n_units = Conf.data.n_units
    n_vars = Conf.data.n_vars
    n_states = n_vars + 1
    n_background_trials = len(background_dataset)
    n_explain_trials = len(explain_dataset)

    mean_model = lit_model.full_model.mean_model.to(device).eval()

    background = [background_dataset.x_position_vars.to(device), background_dataset.x_dense_vars.to(device), background_dataset.x_sparse_vars.to(device)]
    explain = [explain_dataset.x_position_vars.to(device), explain_dataset.x_dense_vars.to(device), explain_dataset.x_sparse_vars.to(device)]

    shap_values = torch.empty(n_explain_trials, n_vars, n_bins, n_units, device=device)
    base_values = torch.empty(n_explain_trials, n_bins, n_units, device=device)

    with torch.inference_mode():

        for explain_trial_idx in range(n_explain_trials):

            permutation = torch.rand(n_permutations, n_vars, device=device, generator=shap_generator).argsort(dim=1)
            order = permutation.argsort(dim=1)
            
            included = order[:, None] < torch.arange(n_states, device=device)[None, :, None]

            background_idx = torch.randint(n_background_trials, (n_permutations,), device=device, generator=shap_generator)

            position = torch.where(included[:, :, None, :4], explain[0][[explain_trial_idx], None, :, :].clone(), background[0][background_idx, None, :, :].clone())
            dense = torch.where(included[:, :, 4:7], explain[1][[explain_trial_idx], None, :].clone(), background[1][background_idx, None, :].clone())
            sparse = torch.where(included[:, :, 7:], explain[2][[explain_trial_idx], None, :].clone(), background[2][background_idx, None, :].clone())

            outputs = mean_model((position.flatten(0, 1), dense.flatten(0, 1), sparse.flatten(0, 1))).reshape(n_permutations, n_states, n_bins, n_units)

            contributions = outputs[:, 1:] - outputs[:, :-1]

            values = torch.zeros(n_vars, n_bins, n_units, device=device)
            values.index_add_(0, permutation.reshape(-1), contributions.flatten(0, 1))

            shap_values[explain_trial_idx] = values / n_permutations
            base_values[explain_trial_idx] = outputs[:, 0].mean(dim=0)

    return [i.cpu().numpy() for i in background], [i.cpu().numpy() for i in explain], shap_values.cpu().numpy(), base_values.cpu().numpy()


def predict_loader(Conf, lit_model, loader, Y_mean, variant):
    device = Conf.device
    lit_model = lit_model.to(device)
    lit_model.eval()

    conditional_gaussian = ConditionalGaussian(Conf, variant).to(device)
    anscombe = Anscombe().to(device)
    Y_mean = torch.as_tensor(Y_mean, dtype=torch.float32, device=device).mean(dim=0, keepdim=True)

    Y = []
    Y_hat = []

    with torch.inference_mode():
        for x, y in loader:
            x = tuple(t.to(device, non_blocking=True) for t in x)
            y = y.to(device, non_blocking=True)

            mean, L = lit_model(x)
            cov = L @ L.transpose(-1, -2)

            y_hat = torch.stack([
                conditional_gaussian.predict(mean_sample, cov_sample, y_sample)
                for mean_sample, cov_sample, y_sample in zip(mean, cov, y)
            ])

            Y.append(anscombe.inv(y + Y_mean))
            Y_hat.append(anscombe.inv(y_hat + Y_mean))

    Y = torch.cat(Y).cpu().numpy()
    Y_hat = torch.cat(Y_hat).cpu().numpy()

    return Y, Y_hat


import warnings
def compute_correlation(y, y_hat):
    if np.std(y) > 1e-8 and np.std(y_hat) > 1e-8:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.corrcoef(y.flatten(), y_hat.flatten())[0, 1]
    return 0.0
compute_r2 = lambda y, y_hat: (
    1 - np.sum((y.flatten() - y_hat.flatten()) ** 2) / np.sum((y.flatten() - np.mean(y.flatten())) ** 2)
    if np.sum((y.flatten() - np.mean(y.flatten())) ** 2) > 1e-3
    else np.nan
)
compute_mse = lambda y, y_hat: np.mean((y - y_hat) ** 2)


def compute_metrics(Conf, Y, Y_hat):
    n_units = Conf.data.n_units

    correlation_trial = []
    r2_trial = []
    mse_trial = []

    for unit_idx in range(n_units):
        y = Y[:, :, unit_idx]
        y_hat = Y_hat[:, :, unit_idx]

        corr = [
            compute_correlation(y_trial, y_hat_trial)
            for y_trial, y_hat_trial in zip(y, y_hat)
        ]
        r2 = [
            compute_r2(y_trial, y_hat_trial)
            for y_trial, y_hat_trial in zip(y, y_hat)
        ]
        mse = [
            compute_mse(y_trial, y_hat_trial)
            for y_trial, y_hat_trial in zip(y, y_hat)
        ]

        correlation_trial.append(corr)
        r2_trial.append(r2)
        mse_trial.append(mse)

    return correlation_trial, r2_trial, mse_trial

class MetricHistory(L.Callback):
    def __init__(self):
        super().__init__()
        self.history = {
            'epoch': [],
            'train_loss_epoch': [],
            'valid_loss_epoch': [],
            'train_correlation': [],
            'valid_correlation': [],
            'train_mse_epoch': [],  # <--- NEW
            'valid_mse_epoch': [],  # <--- NEW
            'learning_rate': [],
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        self.history['epoch'].append(int(trainer.current_epoch))
        self.history['train_loss_epoch'].append(
            float(metrics['train_loss_epoch'].detach().cpu()) if 'train_loss_epoch' in metrics else np.nan
        )
        self.history['valid_loss_epoch'].append(
            float(metrics['valid_loss_epoch'].detach().cpu()) if 'valid_loss_epoch' in metrics else np.nan
        )
        self.history['train_correlation'].append(
            float(metrics['train_correlation'].detach().cpu()) if 'train_correlation' in metrics else np.nan
        )
        self.history['valid_correlation'].append(
            float(metrics['valid_correlation'].detach().cpu()) if 'valid_correlation' in metrics else np.nan
        )
        
        self.history['train_mse_epoch'].append(
            float(metrics['train_mse_epoch'].detach().cpu()) if 'train_mse_epoch' in metrics else np.nan
        )
        self.history['valid_mse_epoch'].append(
            float(metrics['valid_mse_epoch'].detach().cpu()) if 'valid_mse_epoch' in metrics else np.nan
        )

        lr = trainer.optimizers[0].param_groups[0]['lr'] if len(trainer.optimizers) > 0 else np.nan
        self.history['learning_rate'].append(float(lr))

def compute_group_metrics(Y, Y_hat):
    n_trials, _, n_units = Y.shape
    correlations_trial = []
    r2s_trial = []
    mses_trial = []

    for unit_idx in range(n_units):
        y = Y[:, :, unit_idx]
        y_hat = Y_hat[:, :, unit_idx]

        correlations_trial.append([
            compute_correlation(y_trial, y_hat_trial)
            for y_trial, y_hat_trial in zip(y, y_hat)
        ])
        r2s_trial.append([
            compute_r2(y_trial, y_hat_trial)
            for y_trial, y_hat_trial in zip(y, y_hat)
        ])
        mses_trial.append([
            compute_mse(y_trial, y_hat_trial)
            for y_trial, y_hat_trial in zip(y, y_hat)
        ])

    return (
        np.asarray(correlations_trial),
        np.asarray(r2s_trial),
        np.asarray(mses_trial),
    )


def predict_loader_group_conditionings(Conf, lit_model, loader, Y_mean, group_names, baseline="mean"):
    device = Conf.device
    lit_model = lit_model.to(device)
    lit_model.eval()
    anscombe = Anscombe().to(device)
    Y_mean = torch.as_tensor(
        Y_mean,
        dtype=torch.float32,
        device=device,
    ).mean(dim=0, keepdim=True)

    unit_groups = np.asarray(Conf.data.unit_groups)
    group_indices = {
        group_name: np.where(unit_groups == group_name)[0]
        for group_name in group_names
    }

    Y = {group_name: [] for group_name in group_names}
    Y_hat_baseline = {group_name: [] for group_name in group_names}
    Y_hat_model = {
        (target_group, source_group): []
        for target_group in group_names
        for source_group in group_names
    }

    with torch.inference_mode():
        for x, y in loader:
            x = tuple(t.to(device, non_blocking=True) for t in x)
            y = y.to(device, non_blocking=True)

            mean, L = lit_model(x)
            cov = L @ L.transpose(-1, -2)

            baseline_predictions = {
                target_group: mean.clone()
                for target_group in group_names
            }

            model_predictions = {
                (target_group, source_group): mean.clone()
                for target_group in group_names
                for source_group in group_names
            }

            for sample_idx, (mean_sample, cov_sample, y_sample) in enumerate(zip(mean, cov, y)):
                y_flat = y_sample.reshape(-1)
                mean_flat = mean_sample.reshape(-1)

                for time_idx in range(Conf.data.n_bins):
                    for target_group in group_names:
                        target_indices = group_indices[target_group]

                        if baseline == "within":
                            baseline_source_indices = group_indices[target_group]
                        elif baseline == "mean":
                            baseline_source_indices = np.array([], dtype=int)
                        else:
                            raise ValueError(f"Baseline is not supported: {baseline}")

                        for target_idx in target_indices:
                            target_flat_idx = time_idx * Conf.data.n_units + target_idx

                            if baseline_source_indices.size > 0:
                                conditioning_indices = baseline_source_indices[
                                    baseline_source_indices != target_idx
                                ]

                                if conditioning_indices.size > 0:
                                    conditioning_flat_indices = torch.as_tensor(
                                        time_idx * Conf.data.n_units + conditioning_indices,
                                        dtype=torch.long,
                                        device=device,
                                    )

                                    cross_cov = cov_sample[
                                        target_flat_idx,
                                        conditioning_flat_indices,
                                    ]

                                    partial_cov = cov_sample[
                                        conditioning_flat_indices[:, None],
                                        conditioning_flat_indices[None, :],
                                    ]

                                    diff = (
                                        y_flat[conditioning_flat_indices]
                                        - mean_flat[conditioning_flat_indices]
                                    )

                                    baseline_predictions[target_group][
                                        sample_idx,
                                        time_idx,
                                        target_idx,
                                    ] += cross_cov @ torch.linalg.solve(partial_cov, diff)

                            for source_group in group_names:
                                if baseline == "within":
                                    source_indices = np.unique(
                                        np.concatenate([
                                            group_indices[target_group],
                                            group_indices[source_group],
                                        ])
                                    )
                                else:
                                    source_indices = group_indices[source_group]

                                conditioning_indices = source_indices[
                                    source_indices != target_idx
                                ]

                                if conditioning_indices.size == 0:
                                    continue

                                conditioning_flat_indices = torch.as_tensor(
                                    time_idx * Conf.data.n_units + conditioning_indices,
                                    dtype=torch.long,
                                    device=device,
                                )

                                cross_cov = cov_sample[
                                    target_flat_idx,
                                    conditioning_flat_indices,
                                ]

                                partial_cov = cov_sample[
                                    conditioning_flat_indices[:, None],
                                    conditioning_flat_indices[None, :],
                                ]

                                diff = (
                                    y_flat[conditioning_flat_indices]
                                    - mean_flat[conditioning_flat_indices]
                                )

                                model_predictions[target_group, source_group][
                                    sample_idx,
                                    time_idx,
                                    target_idx,
                                ] += cross_cov @ torch.linalg.solve(partial_cov, diff)

            for target_group in group_names:
                target_indices = group_indices[target_group]

                Y[target_group].append(
                    anscombe.inv(y[:, :, target_indices] + Y_mean[:, :, target_indices])
                )

                Y_hat_baseline[target_group].append(
                    anscombe.inv(
                        baseline_predictions[target_group][:, :, target_indices]
                        + Y_mean[:, :, target_indices]
                    )
                )

                for source_group in group_names:
                    Y_hat_model[target_group, source_group].append(
                        anscombe.inv(
                            model_predictions[target_group, source_group][:, :, target_indices]
                            + Y_mean[:, :, target_indices]
                        )
                    )

    Y = {
        group_name: torch.cat(values).cpu().numpy()
        for group_name, values in Y.items()
    }

    Y_hat_baseline = {
        group_name: torch.cat(values).cpu().numpy()
        for group_name, values in Y_hat_baseline.items()
    }

    Y_hat_model = {
        group_pair: torch.cat(values).cpu().numpy()
        for group_pair, values in Y_hat_model.items()
    }

    return Y, Y_hat_baseline, Y_hat_model


def compute_group_influence_matrix(
    Conf,
    lit_model,
    loader,
    Y_mean,
    group_names,
    baseline="mean",
    alpha=0.05,
):
    n_groups = len(group_names)

    correlation_matrix = np.full((n_groups, n_groups), np.nan)
    r2_matrix = np.full((n_groups, n_groups), np.nan)
    mse_matrix = np.full((n_groups, n_groups), np.nan)

    Y, Y_hat_baseline, Y_hat_model = predict_loader_group_conditionings(
        Conf=Conf,
        lit_model=lit_model,
        loader=loader,
        Y_mean=Y_mean,
        group_names=group_names,
        baseline=baseline,
    )

    for target_group_idx, target_group in enumerate(group_names):
        correlation_trial_baseline, r2_trial_baseline, mse_trial_baseline = (
            compute_group_metrics(
                Y[target_group],
                Y_hat_baseline[target_group],
            )
        )

        for source_group_idx, source_group in enumerate(group_names):
            correlation_trial_model, r2_trial_model, mse_trial_model = (
                compute_group_metrics(
                    Y[target_group],
                    Y_hat_model[target_group, source_group],
                )
            )

            correlation_p_values = []
            r2_p_values = []
            mse_p_values = []

            correlation_better = []
            r2_better = []
            mse_better = []

            for unit_idx in range(correlation_trial_baseline.shape[0]):
                try:
                    _, correlation_p_value = wilcoxon(
                        correlation_trial_baseline[unit_idx],
                        correlation_trial_model[unit_idx],
                    )
                except ValueError:
                    correlation_p_value = 1.0

                try:
                    _, r2_p_value = wilcoxon(
                        r2_trial_baseline[unit_idx],
                        r2_trial_model[unit_idx],
                    )
                except ValueError:
                    r2_p_value = 1.0

                try:
                    _, mse_p_value = wilcoxon(
                        mse_trial_baseline[unit_idx],
                        mse_trial_model[unit_idx],
                    )
                except ValueError:
                    mse_p_value = 1.0

                correlation_p_values.append(correlation_p_value)
                r2_p_values.append(r2_p_value)
                mse_p_values.append(mse_p_value)

                correlation_better.append(
                    np.nanmean(correlation_trial_model[unit_idx])
                    > np.nanmean(correlation_trial_baseline[unit_idx])
                )

                r2_better.append(
                    np.nanmean(r2_trial_model[unit_idx])
                    > np.nanmean(r2_trial_baseline[unit_idx])
                )

                mse_better.append(
                    np.nanmean(mse_trial_model[unit_idx])
                    < np.nanmean(mse_trial_baseline[unit_idx])
                )

            correlation_reject, _, _, _ = multipletests(
                correlation_p_values,
                alpha=alpha,
                method="fdr_bh",
            )

            r2_reject, _, _, _ = multipletests(
                r2_p_values,
                alpha=alpha,
                method="fdr_bh",
            )

            mse_reject, _, _, _ = multipletests(
                mse_p_values,
                alpha=alpha,
                method="fdr_bh",
            )

            correlation_matrix[target_group_idx, source_group_idx] = np.mean(
                correlation_reject & np.asarray(correlation_better)
            )

            r2_matrix[target_group_idx, source_group_idx] = np.mean(
                r2_reject & np.asarray(r2_better)
            )

            mse_matrix[target_group_idx, source_group_idx] = np.mean(
                mse_reject & np.asarray(mse_better)
            )

    return correlation_matrix, r2_matrix, mse_matrix

class EpochProgressBar(L.Callback):
    def __init__(self, max_epochs, model_name):
        super().__init__()
        self.max_epochs = max_epochs
        self.model_name = model_name
        self.pbar = None

    def on_fit_start(self, trainer, pl_module):
        # Initialize tqdm progress bar
        self.pbar = tqdm(total=self.max_epochs, desc=f"Training {self.model_name}", initial=trainer.current_epoch, leave=False)

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        
        train_loss = metrics.get("train_loss_epoch")
        val_loss = metrics.get("valid_loss_epoch")
        
        postfix = {}
        if train_loss is not None:
            postfix["train_loss"] = f"{train_loss.item():.4f}"
        if val_loss is not None:
            postfix["val_loss"] = f"{val_loss.item():.4f}"
            
        self.pbar.set_postfix(postfix)
        self.pbar.update(1)

    def on_fit_end(self, trainer, pl_module):
        if getattr(self, 'pbar', None) is not None:
            if getattr(trainer, 'should_stop', False):
                self.pbar.total = self.pbar.n
                self.pbar.refresh()
        if self.pbar is not None:
            self.pbar.close()