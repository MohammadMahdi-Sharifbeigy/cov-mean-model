"""Synthetic-data generation for the mean/covariance experiment."""

from .glm import fit_glm_per_unit_bin, predict_glm_means
from .steps import (
    generate_step1,
    generate_step2_poisson_binwise,
    generate_step3_glm_poisson,
    generate_step5_shared_noise,
    generate_step6a_data,
    generate_step7_synthetic_pattern_cov,
)

__all__ = [
    "fit_glm_per_unit_bin",
    "predict_glm_means",
    "generate_step1",
    "generate_step2_poisson_binwise",
    "generate_step3_glm_poisson",
    "generate_step5_shared_noise",
    "generate_step6a_data",
    "generate_step7_synthetic_pattern_cov",
]
