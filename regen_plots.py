import pickle
import numpy as np
from src.plot_functions import plot_model_metric_improvement, plot_model_metrics_comparison

with open('cache/synth_results.pkl', 'rb') as f:
    results = pickle.load(f)

res_zero = results['step2']['Zero / Identity'][0]
res_base = results['step2']['Baseline / Identity'][0]
res_cond = results['step2']['Conditional / Identity'][0]

# 1. Baseline / Identity improvement over Zero / Identity
plot_model_metric_improvement(
    mse_trial_baseline=res_zero['mse'],
    mse_trial_model=res_base['mse'],
    corr_trial_baseline=res_zero['corr'],
    corr_trial_model=res_base['corr'],
    r2_trial_baseline=res_zero['r2'],
    r2_trial_model=res_base['r2'],
    title="step2: Baseline / Identity improvement over Zero / Identity",
    file_name="baseline-vs-zero-improvement",
    file_path="./plot/synthetic/step2/"
)

# 2. Baseline / Identity vs Zero / Identity comparison
def get_unit_means(trial_metric):
    return [np.nanmean(u) for u in trial_metric]

plot_model_metrics_comparison(
    correlations_x_train=get_unit_means(res_zero['corr']),
    correlations_x_valid=get_unit_means(res_zero['corr']),
    r2s_x_train=get_unit_means(res_zero['r2']),
    r2s_x_valid=get_unit_means(res_zero['r2']),
    mses_x_train=get_unit_means(res_zero['mse']),
    mses_x_valid=get_unit_means(res_zero['mse']),
    correlations_y_train=get_unit_means(res_base['corr']),
    correlations_y_valid=get_unit_means(res_base['corr']),
    r2s_y_train=get_unit_means(res_base['r2']),
    r2s_y_valid=get_unit_means(res_base['r2']),
    mses_y_train=get_unit_means(res_base['mse']),
    mses_y_valid=get_unit_means(res_base['mse']),
    x_label="Zero / Identity",
    y_label="Baseline / Identity",
    file_name="baseline-vs-zero-comparison",
    file_path="./plot/synthetic/step2/",
    title="step2: Baseline / Identity vs Zero / Identity"
)

# 3. Conditional / Identity improvement over Baseline / Identity
plot_model_metric_improvement(
    mse_trial_baseline=res_base['mse'],
    mse_trial_model=res_cond['mse'],
    corr_trial_baseline=res_base['corr'],
    corr_trial_model=res_cond['corr'],
    r2_trial_baseline=res_base['r2'],
    r2_trial_model=res_cond['r2'],
    title="step2: Conditional / Identity improvement over Baseline / Identity",
    file_name="conditional-vs-baseline-improvement",
    file_path="./plot/synthetic/step2/"
)

# 4. Conditional / Identity vs Baseline / Identity comparison
plot_model_metrics_comparison(
    correlations_x_train=get_unit_means(res_base['corr']),
    correlations_x_valid=get_unit_means(res_base['corr']),
    r2s_x_train=get_unit_means(res_base['r2']),
    r2s_x_valid=get_unit_means(res_base['r2']),
    mses_x_train=get_unit_means(res_base['mse']),
    mses_x_valid=get_unit_means(res_base['mse']),
    correlations_y_train=get_unit_means(res_cond['corr']),
    correlations_y_valid=get_unit_means(res_cond['corr']),
    r2s_y_train=get_unit_means(res_cond['r2']),
    r2s_y_valid=get_unit_means(res_cond['r2']),
    mses_y_train=get_unit_means(res_cond['mse']),
    mses_y_valid=get_unit_means(res_cond['mse']),
    x_label="Baseline / Identity",
    y_label="Conditional / Identity",
    file_name="conditional-vs-baseline-comparison",
    file_path="./plot/synthetic/step2/",
    title="step2: Conditional / Identity vs Baseline / Identity"
)
