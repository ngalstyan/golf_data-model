# ==============================================================================
# golf_model/validation/calibration.py
# ==============================================================================
#
# CALIBRATION ANALYSIS
# ----------------------
# A well-calibrated model means: when it says "5% chance of winning,"
# the player should win approximately 5% of the time across many such
# predictions.
#
# Calibration is NECESSARY but not SUFFICIENT for profitable betting.
# A model can be perfectly calibrated but have no edge over the market
# if the market is also perfectly calibrated.
#
# Tools:
#   1. Calibration curves (reliability diagrams)
#   2. Murphy decomposition: BS = Reliability - Resolution + Uncertainty
#   3. PIT histograms (Probability Integral Transform)
#   4. Expected Calibration Error (ECE)
#
# ==============================================================================

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def calibration_curve(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve data (reliability diagram).
    
    Groups predictions into bins and compares predicted probability
    against observed frequency within each bin.
    
    Parameters
    ----------
    probabilities : np.ndarray
        Predicted probabilities.
    outcomes : np.ndarray
        Binary outcomes (0 or 1).
    n_bins : int
        Number of bins.
    strategy : str
        "uniform" — equal-width bins (0.0-0.1, 0.1-0.2, ...).
        "quantile" — equal-count bins (same number of predictions per bin).
        
    Returns
    -------
    tuple of (bin_centers, observed_freq, bin_counts)
        bin_centers : midpoint of each bin
        observed_freq : fraction of positive outcomes in each bin
        bin_counts : number of predictions in each bin
    """
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    o = np.asarray(outcomes, dtype=np.float64).ravel()

    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(p, quantiles)
        bin_edges = np.unique(bin_edges)  # Remove duplicates
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    bin_centers = []
    observed_freqs = []
    bin_counts = []

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        count = mask.sum()
        if count == 0:
            continue

        bin_centers.append((lo + hi) / 2)
        observed_freqs.append(o[mask].mean())
        bin_counts.append(count)

    return (
        np.array(bin_centers),
        np.array(observed_freqs),
        np.array(bin_counts),
    )


def expected_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).
    
    ECE = Σ_b (n_b / N) × |observed_freq_b - predicted_prob_b|
    
    Weighted average of per-bin absolute calibration error.
    
    Parameters
    ----------
    probabilities : np.ndarray
    outcomes : np.ndarray
    n_bins : int
        
    Returns
    -------
    float
        ECE (lower = better calibrated). 0.0 = perfect.
    """
    centers, observed, counts = calibration_curve(
        probabilities, outcomes, n_bins, "uniform"
    )

    if len(counts) == 0:
        return 0.0

    total = counts.sum()
    ece = np.sum(counts / total * np.abs(observed - centers))
    return float(ece)


def murphy_decomposition(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Murphy decomposition of the Brier Score.
    
    BS = Reliability - Resolution + Uncertainty
    
    where:
        Reliability (lower = better):
            How close predicted probs are to observed frequencies.
            REL = (1/N) Σ_b n_b × (observed_b - predicted_b)²
            
        Resolution (higher = better):
            How much the model separates events into different probability bins.
            RES = (1/N) Σ_b n_b × (observed_b - base_rate)²
            
        Uncertainty (fixed):
            UNC = base_rate × (1 - base_rate)
            Inherent unpredictability of the outcome.
    
    A good model has LOW reliability and HIGH resolution.
    
    Parameters
    ----------
    probabilities : np.ndarray
    outcomes : np.ndarray
    n_bins : int
        
    Returns
    -------
    dict with keys: reliability, resolution, uncertainty, brier_score
    """
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    o = np.asarray(outcomes, dtype=np.float64).ravel()
    N = len(p)

    if N == 0:
        return {"reliability": 0, "resolution": 0, "uncertainty": 0, "brier_score": 0}

    base_rate = o.mean()
    uncertainty = base_rate * (1 - base_rate)

    centers, observed, counts = calibration_curve(p, o, n_bins, "uniform")

    if len(counts) == 0:
        bs = float(np.mean((p - o) ** 2))
        return {
            "reliability": 0, "resolution": 0,
            "uncertainty": uncertainty, "brier_score": bs,
        }

    reliability = np.sum(counts * (observed - centers) ** 2) / N
    resolution = np.sum(counts * (observed - base_rate) ** 2) / N
    brier = reliability - resolution + uncertainty

    return {
        "reliability": round(float(reliability), 6),
        "resolution": round(float(resolution), 6),
        "uncertainty": round(float(uncertainty), 6),
        "brier_score": round(float(brier), 6),
        "brier_actual": round(float(np.mean((p - o) ** 2)), 6),
    }


def pit_histogram(
    cdf_values: np.ndarray,
    n_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Probability Integral Transform (PIT) histogram.
    
    If the model is well-calibrated, the PIT values should be 
    uniformly distributed on [0, 1]. The histogram should be flat.
    
    Systematic deviations indicate:
        - U-shape: underdispersed (overconfident)
        - ∩-shape: overdispersed (underconfident)
        - Skew: systematic bias
    
    Parameters
    ----------
    cdf_values : np.ndarray
        CDF evaluated at the actual outcome for each observation.
        For outright winner: CDF = cumulative probability of all players
        ranked better than the actual winner.
    n_bins : int
        
    Returns
    -------
    tuple of (bin_centers, counts)
        For plotting: bar chart should be roughly flat at 1/n_bins.
    """
    vals = np.asarray(cdf_values, dtype=np.float64)
    vals = np.clip(vals, 0, 1)

    counts, edges = np.histogram(vals, bins=n_bins, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2

    # Normalize to density
    counts = counts / counts.sum() if counts.sum() > 0 else counts

    return centers, counts


def calibration_report(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    model_name: str = "Model",
) -> Dict:
    """
    Generate a comprehensive calibration report.
    
    Returns
    -------
    dict
        Full calibration analysis.
    """
    murphy = murphy_decomposition(probabilities, outcomes)
    ece = expected_calibration_error(probabilities, outcomes)

    report = {
        "model_name": model_name,
        "n_predictions": len(probabilities),
        "brier_score": murphy["brier_actual"],
        "log_loss": float(-np.mean(
            outcomes * np.log(np.clip(probabilities, 1e-15, 1)) +
            (1 - outcomes) * np.log(np.clip(1 - probabilities, 1e-15, 1))
        )),
        "ece": round(ece, 6),
        "murphy_reliability": murphy["reliability"],
        "murphy_resolution": murphy["resolution"],
        "murphy_uncertainty": murphy["uncertainty"],
        "base_rate": float(outcomes.mean()),
    }

    logger.info(
        "Calibration report [%s]: Brier=%.4f, ECE=%.4f, "
        "Reliability=%.6f, Resolution=%.6f",
        model_name, report["brier_score"], report["ece"],
        report["murphy_reliability"], report["murphy_resolution"],
    )

    return report
