# ==============================================================================
# golf_model/validation/statistical_tests.py
# ==============================================================================
#
# STATISTICAL SIGNIFICANCE TESTS
# ---------------------------------
# Tests whether the model's improvement over a baseline is 
# statistically significant (not just noise).
#
# Primary test: Diebold-Mariano (1995)
#   H0: Model and baseline have equal predictive accuracy.
#   H1: Model has better predictive accuracy.
#   
#   Why this test? It's designed for comparing forecast accuracy 
#   between two competing models, handles serial correlation in 
#   forecast errors, and doesn't require nested models.
#
# ==============================================================================

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from utils.logger import get_logger

logger = get_logger(__name__)


def diebold_mariano_test(
    model_losses: np.ndarray,
    baseline_losses: np.ndarray,
    h: int = 1,
    alternative: str = "less",
) -> Dict:
    """
    Diebold-Mariano test for equal predictive accuracy.
    
    Tests whether the model has significantly lower loss than baseline.
    
    Test statistic: DM = d̄ / √(V̂(d̄))
    where d_t = L_baseline(t) - L_model(t) is the loss differential.
    
    Under H0 (equal accuracy), DM ~ N(0, 1) asymptotically.
    
    Parameters
    ----------
    model_losses : np.ndarray
        Per-event loss for the model (e.g., Brier Score per tournament).
    baseline_losses : np.ndarray
        Per-event loss for the baseline (e.g., market Brier Score).
    h : int
        Forecast horizon (1 for one-step-ahead). Used for HAC variance.
    alternative : str
        "less" — H1: model losses < baseline losses (model is better).
        "two-sided" — H1: model losses ≠ baseline losses.
        "greater" — H1: model losses > baseline losses (baseline better).
        
    Returns
    -------
    dict with keys:
        dm_statistic : float — DM test statistic
        p_value : float — p-value
        significant : bool — p < 0.05
        mean_diff : float — mean loss differential (negative = model better)
        n_events : int
        
    References
    ----------
    Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy."
    Journal of Business & Economic Statistics, 13(3), 253-263.
    """
    model_l = np.asarray(model_losses, dtype=np.float64)
    baseline_l = np.asarray(baseline_losses, dtype=np.float64)

    if len(model_l) != len(baseline_l):
        raise ValueError("Loss arrays must have equal length")

    n = len(model_l)
    if n < 10:
        logger.warning("Diebold-Mariano with n=%d may be unreliable (need ≥30)", n)

    # Loss differential: positive = model is better (lower loss)
    d = baseline_l - model_l
    d_bar = np.mean(d)

    # HAC variance estimator (Newey-West style)
    # For h=1: just the sample variance with first-order autocorrelation correction
    gamma_0 = np.var(d, ddof=1)

    # Autocovariance at lag k
    if h > 1:
        autocovariances = []
        for k in range(1, h):
            gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
            autocovariances.append(gamma_k)
        var_d_bar = (gamma_0 + 2 * sum(autocovariances)) / n
    else:
        var_d_bar = gamma_0 / n

    if var_d_bar <= 0:
        logger.warning("Non-positive variance in DM test. Using sample variance.")
        var_d_bar = gamma_0 / n

    # DM statistic
    dm_stat = d_bar / np.sqrt(var_d_bar)

    # P-value
    if alternative == "less":
        # H1: model is better → d_bar > 0 → right tail
        p_value = 1 - sp_stats.norm.cdf(dm_stat)
    elif alternative == "greater":
        p_value = sp_stats.norm.cdf(dm_stat)
    else:  # two-sided
        p_value = 2 * (1 - sp_stats.norm.cdf(abs(dm_stat)))

    result = {
        "dm_statistic": round(float(dm_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_005": p_value < 0.05,
        "significant_010": p_value < 0.10,
        "mean_loss_diff": round(float(d_bar), 6),
        "model_better_pct": round(float(np.mean(d > 0)) * 100, 1),
        "n_events": n,
        "alternative": alternative,
    }

    logger.info(
        "Diebold-Mariano test: DM=%.3f, p=%.4f, %s | "
        "model better in %.0f%% of events",
        dm_stat, p_value,
        "SIGNIFICANT" if p_value < 0.05 else "NOT significant",
        result["model_better_pct"],
    )

    return result


def likelihood_ratio_test(
    log_lik_full: float,
    log_lik_restricted: float,
    df: int = 1,
) -> Dict:
    """
    Likelihood ratio test for nested models.
    
    LR = -2 × (ℓ_restricted - ℓ_full) ~ χ²(df)
    
    Tests whether the full model (with course-fit, time-weighting, etc.)
    significantly improves over the restricted model.
    
    Parameters
    ----------
    log_lik_full : float
        Log-likelihood of the full model.
    log_lik_restricted : float
        Log-likelihood of the restricted (simpler) model.
    df : int
        Degrees of freedom (number of additional parameters).
        
    Returns
    -------
    dict
        Test statistic, p-value, significance.
    """
    lr_stat = -2 * (log_lik_restricted - log_lik_full)
    p_value = 1 - sp_stats.chi2.cdf(lr_stat, df)

    return {
        "lr_statistic": round(float(lr_stat), 4),
        "p_value": round(float(p_value), 6),
        "df": df,
        "significant": p_value < 0.05,
        "log_lik_full": round(log_lik_full, 2),
        "log_lik_restricted": round(log_lik_restricted, 2),
    }


def bootstrap_test(
    model_losses: np.ndarray,
    baseline_losses: np.ndarray,
    n_bootstrap: int = 10000,
    seed: Optional[int] = 42,
) -> Dict:
    """
    Non-parametric bootstrap test for model comparison.
    
    Bootstraps the loss differential to construct a confidence interval.
    If 0 is outside the CI → significant difference.
    
    More robust than DM for small samples or non-normal differentials.
    
    Parameters
    ----------
    model_losses : np.ndarray
    baseline_losses : np.ndarray
    n_bootstrap : int
    seed : int
        
    Returns
    -------
    dict
        Bootstrap CI, p-value, significance.
    """
    rng = np.random.default_rng(seed)
    d = baseline_losses - model_losses  # positive = model better
    n = len(d)

    # Bootstrap
    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(d, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # P-value: fraction of bootstrap samples where model is NOT better
    p_value = float(np.mean(boot_means <= 0))

    return {
        "mean_diff": round(float(np.mean(d)), 6),
        "ci_95_lower": round(ci_lower, 6),
        "ci_95_upper": round(ci_upper, 6),
        "p_value": round(p_value, 4),
        "significant": ci_lower > 0,  # Entire CI above 0
        "n_bootstrap": n_bootstrap,
    }
