"""SHAP and recovery computations extracted from the experiment notebook."""
from __future__ import annotations

import numpy as np
import torch
from scipy import stats

def seed_everything(seed, deterministic=True):
    """Notebook-compatible local seeding used by permutation SHAP."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_shap_values(Conf, lit_model, background_dataset, explain_dataset, n_permutations=10):
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

    # ---> DYNAMIC SLICING INDICES (prevents size mismatch errors when variables are hidden)
    n_pos = Conf.data.n_position_vars
    n_den = Conf.data.n_dense_vars
    pos_end = n_pos
    den_end = n_pos + n_den

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

            # Slicing uses dynamic variables pos_end and den_end instead of hardcoded numbers
            position = torch.where(included[:, :, None, :pos_end], explain[0][[explain_trial_idx], None, :, :].clone(), background[0][background_idx, None, :, :].clone())
            dense = torch.where(included[:, :, pos_end:den_end], explain[1][[explain_trial_idx], None, :].clone(), background[1][background_idx, None, :].clone())
            sparse = torch.where(included[:, :, den_end:], explain[2][[explain_trial_idx], None, :].clone(), background[2][background_idx, None, :].clone())

            outputs = mean_model((position.flatten(0, 1), dense.flatten(0, 1), sparse.flatten(0, 1))).reshape(n_permutations, n_states, n_bins, n_units)
            contributions = outputs[:, 1:] - outputs[:, :-1]
            values = torch.zeros(n_vars, n_bins, n_units, device=device)
            values.index_add_(0, permutation.reshape(-1), contributions.flatten(0, 1))

            shap_values[explain_trial_idx] = values / n_permutations
            base_values[explain_trial_idx] = outputs[:, 0].mean(dim=0)

    return [i.cpu().numpy() for i in background], [i.cpu().numpy() for i in explain], shap_values.cpu().numpy(), base_values.cpu().numpy()
