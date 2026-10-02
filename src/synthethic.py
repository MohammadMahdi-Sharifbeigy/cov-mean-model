# @title data generation 

import warnings as _warnings
import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed

STEP1_NOISE = 'poisson'
SEED = 1   # module-level default; matches the project-wide seed set in the notebook


def _build_design_matrix(X_dense, X_sparse):
    """Concatenate dense + sparse task vars and add statsmodels intercept.

    Returns X : (T, V+1)
    """
    X_raw = np.concatenate([X_dense, X_sparse], axis=1)  # (T, V)
    return sm.add_constant(X_raw, prepend=True)          # (T, V+1)


def _fit_one_glm(y_nb, X_design):
    """Fit Poisson log-linear GLM for one (unit, bin).

    Model:  y[t] ~ Poisson(lambda_t),  log lambda_t = X_design[t] @ beta

    Falls back to intercept-only (PSTH mean) on silent neuron or failure.

    Returns beta : (V+1,)
    """
    V1 = X_design.shape[1]
    mu = float(y_nb.mean())

    # Silent unit/bin -> Poisson(0)
    if mu == 0.0:
        beta = np.zeros(V1)
        beta[0] = np.log(1e-8)
        return beta

    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore')
            glm = sm.GLM(
                y_nb, X_design,
                family=sm.families.Poisson(link=sm.families.links.Log()),
            )
            res = glm.fit(maxiter=200, disp=False)
        return np.asarray(res.params, dtype=float)
    except Exception:
        # Fallback: intercept-only
        beta = np.zeros(V1)
        beta[0] = np.log(max(mu, 1e-8))
        return beta


def fit_glm_per_unit_bin(counts, X_dense, X_sparse, n_jobs=-1, seed=SEED):
    """Fit one Poisson GLM per (neuron, bin) using ALL trials.

    The GLM is used only to generate the synthetic null distribution.
    Fitting on all trials gives the most stable beta estimates.

    Parameters
    ----------
    counts   : (T, N, K)
    X_dense  : (T, D)  continuous task vars (MinMax-scaled)
    X_sparse : (T, S)  binary task vars
    n_jobs   : int  (-1 = all cores)

    Returns
    -------
    betas : (N, K, V+1)  intercept first, then dense cols, then sparse cols
    """
    T, N, K = counts.shape
    X_design = _build_design_matrix(X_dense, X_sparse)
    V1 = X_design.shape[1]

    tasks = [
        (counts[:, n, b], X_design)
        for n in range(N)
        for b in range(K)
    ]

    print(f'Fitting {N*K} Poisson GLMs  ({N} units x {K} bins) ...')
    results_flat = Parallel(n_jobs=n_jobs, prefer='threads', verbose=0)(
        delayed(_fit_one_glm)(y_nb, Xd) for y_nb, Xd in tasks
    )

    betas = np.array(results_flat, dtype=float).reshape(N, K, V1)
    print(f'  betas shape: {betas.shape}  (neurons, bins, V+1)')
    return betas


def predict_glm_means(betas, X_dense, X_sparse):
    """Reconstruct GLM-predicted lambda[t,n,b] = exp(X[t] @ beta[n,b]).

    Parameters
    ----------
    betas    : (N, K, V+1)
    X_dense  : (T, D)
    X_sparse : (T, S)

    Returns
    -------
    lambda_hat : (T, N, K)  predicted Poisson rates (counts/bin)
    """
    X_design = _build_design_matrix(X_dense, X_sparse)  # (T, V+1)
    N, K, V1 = betas.shape

    # eta[t,n,b] = sum_v X[t,v] * betas[n,b,v]
    # (T,1,1,V1) * (1,N,K,V1) -> sum over V1 -> (T,N,K)
    eta = (
        X_design[:, np.newaxis, np.newaxis, :]
        * betas[np.newaxis, :, :, :]
    ).sum(axis=-1)

    lambda_hat = np.exp(np.clip(eta, -30, 30))
    return lambda_hat

def generate_step1(prep: dict, noise: str = STEP1_NOISE, seed: int = SEED) -> np.ndarray:
    """NB-B baseline target: preserve each neuron's marginal firing statistics,
    destroy all task / temporal / cross-neuron covariance (every bin i.i.d.).

    Isolates the baseline co-fluctuation confound (s41593-024-01575-w, 2024):
    a model that beats its peers here is exploiting spurious baseline variance,
    not task signal.
    """
    rng = np.random.default_rng(seed + 202)
    counts = prep["counts"]
    mu = counts.mean(axis=(0, 2), keepdims=True)        # per-neuron mean count
    if noise == "poisson":
        synth = rng.poisson(lam=np.broadcast_to(mu, counts.shape)).astype(float)
    elif noise == "gaussian":
        sigma = counts.std(axis=(0, 2), keepdims=True)
        synth = np.clip(rng.normal(mu, sigma, size=counts.shape), 0, None)
    else:
        raise ValueError(f"unknown STEP1_NOISE: {noise}")
    return synth * prep["spike_scale"]

def generate_step2_poisson_binwise(
    prep: dict, seed: int = SEED
) -> np.ndarray:
    """NB-C Step-2a: Poisson(Î¼[n,b]) synthetic target.

    Preserves the per-neuron, per-bin mean firing profile (Î¼[n,b] =
    mean_t C[t,n,b]) â€” the event-locked PSTH â€” while destroying all
    cross-trial shared fluctuations and all cross-neuron covariance.
    Each (trial, neuron, bin) cell is drawn i.i.d.

    This is the minimal structured null: real task structure encoded in
    the PSTH is preserved; anything beyond that (correlations, noise
    correlations, overdispersion) is absent.

    Methodological grounding: Shahidi et al. (2019, Nat. Neurosci.)
    introduce per-neuron-rate synthetic nulls to isolate variance
    attributable to the mean firing profile vs. genuine covariance
    structure. The binwise extension sharpens resolution to the
    temporal level.

    Returns firing-rate units (counts Ã— spike_scale), identical units
    to prep['Y_rate'], so the target flows unchanged into NeuralDataset
    and the Anscombe transform.
    """
    rng = np.random.default_rng(seed + 303)
    counts = prep["counts"]                   # (T, N, K) spike counts
    T, N, K = counts.shape

    # Î¼[n, b]: per-neuron per-bin mean across trials â€” shape (1, N, K)
    mu_nb = counts.mean(axis=0, keepdims=True)  # broadcast over trials

    # Log zero-mean cells; they generate all-zero Poisson draws (correct).
    n_zero = int((mu_nb == 0).sum())
    if n_zero > 0:
        import warnings as _w
        _w.warn(
            f"generate_step2_poisson_binwise: {n_zero} (neuron,bin) "
            "cells with Î¼=0; those entries will be 0 in the synthetic tensor.",
            RuntimeWarning, stacklevel=2,
        )

    # Vectorised draw: rng.poisson broadcasts (1,N,K) lambda â†’ (T,N,K)
    synth = rng.poisson(
        lam=np.broadcast_to(mu_nb, (T, N, K))
    ).astype(float)

    assert synth.shape == counts.shape, (
        f"step2_poisson shape mismatch: {synth.shape} vs {counts.shape}"
    )
    assert synth.dtype == float, "step2_poisson dtype must be float"
    assert np.all(synth >= 0), "step2_poisson contains negative values"

    return synth * prep["spike_scale"]

def generate_step3_glm_poisson(prep, betas, seed=SEED, X_dense=None, X_sparse=None):
    """Step-3 null: Poisson( lambda_hat[t,n,b] ).

    Variable-conditioned mean, no cross-neuron covariance, Fano factor = 1.

    Returns Y_synth : (T, N, K) firing rates (counts x spike_scale)
    """
    rng         = np.random.default_rng(seed + 505)
    counts      = prep['counts']
    X_dense     = X_dense if X_dense is not None else prep['X_dense_task']
    X_sparse    = X_sparse if X_sparse is not None else prep['X_sparse_task']
    spike_scale = prep['spike_scale']
    T, N, K     = counts.shape

    lambda_hat = predict_glm_means(betas, X_dense, X_sparse)  # (T,N,K)

    n_large = int((lambda_hat > 1000).sum())
    if n_large > 0:
        _warnings.warn(
            f'generate_step3a: {n_large} cells with lambda>1000 - check GLM fit.',
            RuntimeWarning, stacklevel=2,
        )

    synth = rng.poisson(lam=lambda_hat).astype(float)

    assert synth.shape == counts.shape
    assert np.all(synth >= 0)

    return synth * spike_scale

def filter_task_vars(prep, selected_vars=None):
    """Filter task variables by name for Step 3 GLM."""
    if selected_vars is None:
        return prep['X_dense_task'], prep['X_sparse_task']
        
    dense_cols = []
    sparse_cols = []
    task_names = list(prep['task_var_names'])
    
    for var in selected_vars:
        if var in task_names:
            idx = task_names.index(var)
            if idx in prep['dense_indices']:
                dense_cols.append(prep['dense_indices'].index(idx))
            elif idx in prep['sparse_indices']:
                sparse_cols.append(prep['sparse_indices'].index(idx))
        else:
            print(f"Warning: '{var}' not found in task_var_names")
            
    if dense_cols:
        X_dense = prep['X_dense_task'][:, dense_cols]
    else:
        X_dense = np.zeros((prep['X_dense_task'].shape[0], 0))
        
    if sparse_cols:
        X_sparse = prep['X_sparse_task'][:, sparse_cols]
    else:
        X_sparse = np.zeros((prep['X_sparse_task'].shape[0], 0))
        
    return X_dense, X_sparse

def generate_step4_random_latent_bias(prep, lambda_hat, n_latents=3, latent_strength=0.4, seed=SEED):
    rng = np.random.default_rng(seed)
    T, N, K = lambda_hat.shape
    
    # 1. Generate low-rank latent fluctuations
    Z = rng.normal(size=(T, n_latents))
    W = rng.normal(size=(n_latents, N))
    raw_proj = Z @ W
    
    # 2. Standardize across time so each unit's perturbation has mean=0 and std=1
    proj_norm = (raw_proj - np.mean(raw_proj, axis=0, keepdims=True)) / (np.std(raw_proj, axis=0, keepdims=True) + 1e-8)
    
    # 3. Scale by latent_strength (sigma) and subtract sigma^2 / 2 for exact log-normal mean compensation!
    sigma = latent_strength
    log_perturbation = (proj_norm[:, :, np.newaxis] * sigma) - (0.5 * sigma**2)
    
    # E[exp(log_perturbation)] is precisely 1.0 -> mean firing rates stay unaffected!
    modulated_lambda = lambda_hat * np.exp(log_perturbation)
    modulated_lambda = np.clip(modulated_lambda, 1e-8, 1000.0)
    
    synth = rng.poisson(lam=modulated_lambda).astype(float)
    return synth * prep.get('spike_scale', 1.0)

def generate_step5_shared_noise(prep, lambda_hat, cov_type='global', n_blocks=4, seed=SEED, noise_scale=1.0, **kwargs):
    """
    Step-5 null: Adds synthetic shared noise sampled from the empirical noise 
    correlation (residual covariance) matrix.
    
    cov_type options:
      - 'global': A single N x N covariance matrix across all bins and trials.
      - 'binwise': K different N x N covariance matrices (one for each bin).
      - 'structured': A block-structured N x N covariance matrix (random boxes).
    """
    rng = np.random.default_rng(seed)
    
    Y_true = prep['counts']
    T, N, K = Y_true.shape
    
    eps = 1e-8
    # residuals: (T, N, K)
    residuals = (Y_true - lambda_hat) / np.sqrt(lambda_hat + eps)
    
    synth_noise = np.zeros_like(residuals)
    
    if cov_type == 'global':
        # Compute covariance across T and K
        res_flat = residuals.transpose(0, 2, 1).reshape(T * K, N) # (T*K, N)
        Sigma = np.cov(res_flat, rowvar=False) # (N, N)
        
        # Sample noise
        noise_flat = rng.multivariate_normal(mean=np.zeros(N), cov=Sigma * noise_scale, size=T * K)
        synth_noise = noise_flat.reshape(T, K, N).transpose(0, 2, 1)
        
    elif cov_type == 'binwise':
        # Compute covariance for each bin separately
        for k in range(K):
            res_k = residuals[:, :, k] # (T, N)
            Sigma_k = np.cov(res_k, rowvar=False) # (N, N)
            
            # Sample noise for this bin
            synth_noise[:, :, k] = rng.multivariate_normal(mean=np.zeros(N), cov=Sigma_k * noise_scale, size=T)
            
    elif cov_type == 'structured':
        # Compute global covariance first
        res_flat = residuals.transpose(0, 2, 1).reshape(T * K, N)
        Sigma = np.cov(res_flat, rowvar=False) # (N, N)
        
        # Create a block-diagonal mask or structured boxes
        block_size = int(np.ceil(N / n_blocks))
        Sigma_structured = np.zeros_like(Sigma)
        
        for i in range(n_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, N)
            if start < N:
                Sigma_structured[start:end, start:end] = Sigma[start:end, start:end]
            
        # Ensure it's positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(Sigma_structured)
        eigvals = np.clip(eigvals, 1e-8, None)
        Sigma_structured = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        noise_flat = rng.multivariate_normal(mean=np.zeros(N), cov=Sigma_structured * noise_scale, size=T * K)
        synth_noise = noise_flat.reshape(T, K, N).transpose(0, 2, 1)

    # Add synthetic noise back to the predicted mean
    new_lambda = lambda_hat + synth_noise * np.sqrt(lambda_hat + eps)
    new_lambda = np.clip(new_lambda, 1e-8, 1000.0)
    
    synth = rng.poisson(lam=new_lambda).astype(float)
    
    if kwargs.get('return_noise', False):
        return synth * prep.get('spike_scale', 1.0), synth_noise
    return synth * prep.get('spike_scale', 1.0)

# =========================================================================
# Step 6a: Deep Mean Model Noise Correlation Extraction & Injection
# =========================================================================
import torch
import numpy as np

def extract_empirical_covariance_from_deep_model(prep, Conf, device, x_position_vars):
    import os
    import numpy as np
    checkpoint_file = 'step6a_extraction_checkpoint.npz'
    if os.path.exists(checkpoint_file):
        print(f"Loading cached extraction from {checkpoint_file} to save time...")
        data = np.load(checkpoint_file)
        return data['empirical_covs'], data['residuals_real']

    print("-> Training ConditionalMeanModel on real data to extract empirical residuals...")
    from copy import deepcopy
    import torch
    from torch import nn
    from src.classes import NeuralDataset, LitModel
    from src.helper_functions import EpochProgressBar, MetricHistory

    class Anscombe(nn.Module):
        def forward(self, x): return 2.0 * torch.sqrt(x + 3.0 / 8.0)
        def inv(self, x): return (x / 2.0) ** 2 - 3.0 / 8.0
    ans = Anscombe()

    # Zero out all task variables EXCEPT 'tslp' and 'rew' to match the Conf dimensions
    x_dense_3 = np.zeros_like(prep['X_dense_task'])
    x_sparse_3 = np.zeros_like(prep['X_sparse_task'])
    task_names = list(prep['task_var_names'])
    for var in ['tslp', 'rew']:
        if var in task_names:
            idx = task_names.index(var)
            if idx in prep['dense_indices']:
                col_idx = prep['dense_indices'].index(idx)
                x_dense_3[:, col_idx] = prep['X_dense_task'][:, col_idx]
            elif idx in prep['sparse_indices']:
                col_idx = prep['sparse_indices'].index(idx)
                x_sparse_3[:, col_idx] = prep['X_sparse_task'][:, col_idx]

    Y_tensor = torch.tensor(prep['counts'], dtype=torch.float32).transpose(1, 2)
    Y_mean = torch.mean(ans.forward(Y_tensor), dim=(0, 1), keepdim=True)
    
    # Standardize dense and position variables exactly like run_model_comparisons does!
    x_pos_norm = x_position_vars.copy()
    pos_mean, pos_std = x_pos_norm.mean(axis=0, keepdims=True), x_pos_norm.std(axis=0, keepdims=True)
    x_pos_norm = (x_pos_norm - pos_mean) / (pos_std + 1e-8)
    
    den_mean, den_std = x_dense_3.mean(axis=0, keepdims=True), x_dense_3.std(axis=0, keepdims=True)
    x_dense_norm = (x_dense_3 - den_mean) / (den_std + 1e-8)
    
    dataset_real = NeuralDataset(
        x_position_vars=x_pos_norm,
        x_dense_vars=x_dense_norm,
        x_sparse_vars=x_sparse_3,
        Y=prep['counts'],
        mean=Y_mean
    )

    
    # Train Mean Model only
    mean_model = LitModel(
        Conf, 
        mean_model='conditional', 
        cov_model='identity'
    ).to(device)
    
    from torch.utils.data import DataLoader
    from lightning import Trainer
    import logging
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    
    train_loader = DataLoader(dataset_real, batch_size=Conf.training.batch_size, shuffle=True)
    # Add the custom progress bar defined earlier in the notebook
    progress_bar = EpochProgressBar(max_epochs=Conf.training.max_epoch, model_name="Deep Mean Model (Step 6a Extraction)")
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping
    early_stop_callback = EarlyStopping(
        monitor='valid_loss_epoch',
        patience=Conf.training.patience,
        mode='min',
        verbose=False
    )
    
    trainer = Trainer(
        max_epochs=Conf.training.max_epoch, 
        accelerator='gpu' if device.type == 'cuda' else 'cpu',
        devices=1,
        enable_progress_bar=False, 
        enable_model_summary=False,
        callbacks=[progress_bar, MetricHistory(), early_stop_callback]
    )
    # Pass train_loader as validation loader to satisfy ReduceLROnPlateau which monitors 'valid_loss_epoch'
    trainer.fit(mean_model, train_dataloaders=train_loader, val_dataloaders=train_loader)
    
    # Extract predictions (lambda_hat)
    mean_model = mean_model.to(device) # Trainer pulled model to CPU, bring it back!
    mean_model.eval()
    val_loader = DataLoader(dataset_real, batch_size=Conf.training.batch_size, shuffle=False)
    lambda_hat_list = []
    
    with torch.no_grad():
        for batch in val_loader:
            x, _ = batch
            x = [t.to(device) for t in x]
            out_mean, L = mean_model(x)
            y_hat = ans.inv(out_mean.cpu() + Y_mean.cpu())
            lambda_hat_list.append(y_hat.numpy())
            
    lambda_hat = np.concatenate(lambda_hat_list, axis=0)
    lambda_hat = np.transpose(lambda_hat, (0, 2, 1))
    
    Y_real = prep['counts'] * prep.get('spike_scale', 1.0)

    
    T, N, K = Y_real.shape
    
    # residuals (T, N, K)
    residuals = Y_real - lambda_hat
    
    # Calculate unit x unit x bin empirical covariance
    empirical_covs = np.zeros((K, N, N))
    for k in range(K):
        res_k = residuals[:, :, k] # (T, N)
        empirical_covs[k] = np.cov(res_k, rowvar=False)
        
    print(f"Saving extraction to {checkpoint_file}...")
    np.savez(checkpoint_file, empirical_covs=empirical_covs, residuals_real=residuals)
    return empirical_covs, residuals

def generate_step6_data(prep, empirical_covs, residuals_real, lambda_hat_step3, seed=SEED, return_noise=False):
    """
    Injects empirical noise correlation into step 3 synthetic means.
    """
    rng = np.random.default_rng(seed)
    T, N, K = lambda_hat_step3.shape
    
    synth_noise = np.zeros_like(lambda_hat_step3)
    
    for k in range(K):
        Sigma_k = empirical_covs[k]
        # Sample noise (T, N)
        noise_k = rng.multivariate_normal(mean=np.zeros(N), cov=Sigma_k, size=T)
        synth_noise[:, :, k] = noise_k
        
    # Variance Matching
    std_real = np.std(residuals_real)
    std_synth = np.std(synth_noise)
    
    scale_factor = std_real / (std_synth + 1e-8)
    print(f"-> [Step 6] Variance Control Scale Factor: {scale_factor:.4f}")
    
    synth_noise_scaled = synth_noise * scale_factor
    
    eps = 1e-8
    new_lambda = lambda_hat_step3 + synth_noise_scaled * np.sqrt(lambda_hat_step3 + eps)
    new_lambda = np.clip(new_lambda, 1e-8, 1000.0)
    
    synth = rng.poisson(lam=new_lambda).astype(float)
    synth_out = synth * prep.get('spike_scale', 1.0)

    if return_noise:
        return synth_out, synth_noise_scaled
    return synth_out


def make_positive_definite(C, eps=1e-6):
    """
    Project a symmetric matrix onto the PD cone and renormalize
    the diagonal back to 1 (i.e. treat it as a correlation matrix).
    """
    # enforce symmetry
    C = (C + C.T) / 2

    # eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(C)

    # clip negative / too-small eigenvalues
    eigvals[eigvals < eps] = eps

    # reconstruct
    C_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # renormalize diagonal back to 1
    d = np.sqrt(np.diag(C_pd))
    C_pd = C_pd / np.outer(d, d)

    # small jitter for numerical safety
    C_pd += eps * np.eye(C_pd.shape[0])

    return C_pd

def generate_step7_beta_factor_cov(
    prep,
    betas,
    lambda_hat,
    cov_mode='binwise',
    noise_scale=5.0,
    ridge=1e-6,
    seed=SEED,
    sort_by_tuning=True,
    n_clusters=5,
    ):
    """
    Step-7 null: Synthetic covariance built purely from GLM tuning coefficients.
    Contains ZERO real-data information in Sigma.

    The covariance structure is derived from the **normalized** Î²-factor Gram matrix
    (cosine similarity between neuron tuning vectors):

        B_norm_b = B_b / ||B_b||_2        # unit-length tuning vectors
        Sigma_b  = B_norm_b @ B_norm_b.T  # cosine similarity, entries in [-1, +1]

    KEY FIX vs raw B@B.T:
    - Raw B@B.T is dominated by the absolute magnitude of betas (firing rate scale)
      â†’ all-positive matrix, looks like Step 4 (no visible structure)
    - Normalized cosine similarity captures DIRECTION only, independent of magnitude
      â†’ entries span [-1, +1], directly encoding same/opposite/orthogonal tuning

    This produces the signed block structure matching the target image:
      - neurons with same-direction tuning   â†’ cos â‰ˆ +1 â†’ positive entry (blue blocks)
      - neurons with opposite-direction tuning â†’ cos â‰ˆ -1 â†’ negative entry (red blocks)
      - neurons with orthogonal tuning        â†’ cos â‰ˆ  0 â†’ near-zero entry (white)

    Normalization guarantees the matrix is still PSD (it is the Gram matrix of
    real unit vectors) while allowing negative off-diagonal entries.

    Parameters
    ----------
    prep        : dict   â€” dataset prep dict (needs 'counts', 'spike_scale')
    betas       : ndarray (N, K, V+1) â€” GLM betas; step3 betas by default,
                  pass step4 betas to include the hidden variable dimension
    lambda_hat  : ndarray (T, N, K) â€” GLM predicted rates (same betas as above)
    cov_mode    : str  â€” 'binwise' (separate Sigma per bin, recommended)
                         'global'  (average tuning across bins, single Sigma)
    noise_scale : float â€” scalar multiplier on Sigma before sampling;
                  controls noise strength; 5.0 recommended (makes correlated
                  noise dominate Poisson noise so block structure is visible)
    ridge       : float â€” added to Sigma diagonal for numerical PD guarantee
    seed        : int
    sort_by_tuning : bool â€” if True, cluster neurons by their mean tuning
                  direction (K-means on normalized mean beta vectors across bins)
                  and return (synth, sort_idx). The data itself is NOT reordered;
                  sort_idx is for visualization only (pass to plot calls).
    n_clusters  : int â€” number of K-means clusters for tuning-based sort

    Pseudocode (per bin b, in 'binwise' mode)
    -----------------------------------------
    1.  B_b      = betas[:, b, 1:]                       # (N, V)  exclude intercept
    2.  B_norm_b = B_b / (||B_b||_row + eps)             # (N, V)  unit-length tuning vectors
    3.  Gram     = B_norm_b @ B_norm_b.T                 # (N, N)  cosine similarity âˆˆ [-1,+1]
    4.  Sigma_b  = Gram * noise_scale + ridge * I        # scale + numerical PD fix
    5.  Make PSD: eigendecompose, clip eigenvalues to ridge, reconstruct
    6.  noise_b  ~ MVN(0, Sigma_b),  size=T              # (T, N)  shared signed noise
    7.  synth_noise[:, :, b] = noise_b
    (end loop)
    8.  new_lambda = lambda_hat + synth_noise * sqrt(lambda_hat + eps)
    9.  new_lambda = clip(new_lambda, 1e-8, 1000)
    10. synth      = Poisson(new_lambda) * spike_scale

    Returns
    -------
    synth     : ndarray (T, N, K) â€” synthetic firing rates (counts Ã— spike_scale)
                Neurons are in the ORIGINAL order (not sorted) â€” safe for training.
    sort_idx  : ndarray (N,) â€” neuron indices that sort by tuning cluster
                (only returned when sort_by_tuning=True). Use this to reorder
                Y_synth and lambda_hat for visualization:
                    Y_sorted = synth[:, sort_idx, :]
                    lhat_sorted = lambda_hat[:, sort_idx, :]
    """
    rng = np.random.default_rng(seed)

    T, N, K = lambda_hat.shape
    V = betas.shape[2] - 1  # exclude intercept column
    eps = 1e-8
    I_N = np.eye(N)

    synth_noise = np.zeros((T, N, K), dtype=float)

    if cov_mode == 'global':
        # ---- single Sigma from mean tuning across all bins ----
        B_mean = betas[:, :, 1:].mean(axis=1)                            # (N, V)
        norms  = np.linalg.norm(B_mean, axis=1, keepdims=True) + eps
        B_norm = B_mean / norms                                          # unit vectors
        Gram   = B_norm @ B_norm.T                                       # (N, N) cosine sim âˆˆ [-1,+1]
        Sigma  = Gram * noise_scale + ridge * I_N
        # Ensure PSD (cosine gram is PSD by construction; ridge makes it strictly PD)
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.clip(eigvals, ridge, None)
        Sigma   = eigvecs @ np.diag(eigvals) @ eigvecs.T
        noise_flat  = rng.multivariate_normal(np.zeros(N), Sigma, size=T * K)  # (T*K, N)
        synth_noise = noise_flat.reshape(T, K, N).transpose(0, 2, 1)           # (T, N, K)

    elif cov_mode == 'binwise':
        # ---- per-bin Sigma: cosine similarity of that bin's tuning vectors ----
        for b in range(K):
            B_b    = betas[:, b, 1:]                                     # (N, V)
            norms  = np.linalg.norm(B_b, axis=1, keepdims=True) + eps
            B_norm = B_b / norms                                         # (N, V) unit vectors
            Gram   = B_norm @ B_norm.T                                   # (N, N) cosine sim âˆˆ [-1,+1]
            Sigma_b = Gram * noise_scale + ridge * I_N
            # Ensure PSD (small clip for numerical stability)
            eigvals, eigvecs = np.linalg.eigh(Sigma_b)
            eigvals  = np.clip(eigvals, ridge, None)
            Sigma_b  = eigvecs @ np.diag(eigvals) @ eigvecs.T
            synth_noise[:, :, b] = rng.multivariate_normal(np.zeros(N), Sigma_b, size=T)

    else:
        raise ValueError(f"generate_step7_beta_factor_cov: unknown cov_mode '{cov_mode}'. "
                         "Choose 'binwise' or 'global'.")

    # ---- inject noise proportionally into lambda (same formula as step5) ----
    new_lambda = lambda_hat + synth_noise * np.sqrt(lambda_hat + eps)
    new_lambda = np.clip(new_lambda, eps, 1000.0)

    synth = rng.poisson(lam=new_lambda).astype(float)

    # ---- sanity checks ----
    assert synth.shape == (T, N, K), f"step7 shape mismatch: {synth.shape}"
    assert np.all(synth >= 0), "step7 contains negative values"

    synth_out = synth * prep.get('spike_scale', 1.0)

    if sort_by_tuning:
        # Cluster neurons by mean tuning direction across all bins
        from sklearn.cluster import KMeans
        B_all = betas[:, :, 1:].mean(axis=1)                          # (N, V) mean tuning
        norms_all = np.linalg.norm(B_all, axis=1, keepdims=True) + eps
        B_all_norm = B_all / norms_all                                 # unit vectors
        km = KMeans(n_clusters=min(n_clusters, N), random_state=seed, n_init=10)
        labels = km.fit_predict(B_all_norm)
        # Secondary sort: within each cluster, sort by angle in beta space
        sort_idx = np.lexsort((np.arctan2(B_all_norm[:, 0], B_all_norm[:, -1]), labels))
        return synth_out, sort_idx

    return synth_out

def generate_step7_kronecker_poisson(
    lambda_hat,
    per_neuron_cov,   # (N, N)
    per_bin_cov,      # (B, B)
    ridge=1e-6,
    seed=SEED,
    spike_scale=1.0,
    return_noise=False,
    ):
    """
    Step 7c: Same Kronecker joint covariance as 7a,
    but with an added Poisson shot-noise layer on top.
    """
    rng = np.random.default_rng(seed)
    T, N, B = lambda_hat.shape
    eps = 1e-8

    # 1. Build joint neuron-bin covariance (same as 7a)
    hidden_cov_matrix = np.kron(per_neuron_cov, per_bin_cov)
    hidden_cov_matrix = make_positive_definite(hidden_cov_matrix, eps=ridge)

    # 2. Cholesky factor over the full (N*B) x (N*B) space
    L2 = np.linalg.cholesky(hidden_cov_matrix)

    # 3. Draw independent standard normal noise and color it
    z = rng.normal(size=(T, N * B))
    noise_flat = z @ L2.T
    eta = noise_flat.reshape(T, N, B)

    # 4. Inject noise into the rate (same formula as 7a)
    new_lambda = lambda_hat + eta * np.sqrt(lambda_hat + eps)
    new_lambda = np.clip(new_lambda, eps, 1000.0)

    # 5. Extra Poisson layer on top
    synth = rng.poisson(lam=new_lambda).astype(np.float32)

    assert synth.shape == (T, N, B), f"step7 shape mismatch: {synth.shape}"
    assert np.all(synth >= 0), "step7 contains negative values"

    synth_out = synth * spike_scale
    if return_noise:
        return synth_out, eta
    return synth_out


def generate_positive_control_shared_scalar(
    n_trials, n_channels, n_bins, sigma_eps=1.0, seed=SEED
):
    """Positive-control synthetic dataset.

    For each trial, draw a single scalar epsilon_t ~ N(0, sigma_eps^2) and
    broadcast it identically to every (channel, bin) entry of that trial.
    All other structure is zero.  The resulting covariance is rank-1 with an
    all-ones correlation matrix — a closed-form ground truth for testing
    factor-analytic identifiability of SharedCovModel.

    Returns
    -------
    Y       : ndarray (n_trials, n_channels, n_bins)
    epsilon : ndarray (n_trials,)  — drawn scalar per trial (for verification)
    """
    rng = np.random.default_rng(seed)
    epsilon = rng.normal(loc=0.0, scale=sigma_eps, size=(n_trials,))
    Y = np.zeros((n_trials, n_channels, n_bins))
    Y = Y + epsilon[:, None, None]   # broadcast to all (channel, bin)
    return Y, epsilon
