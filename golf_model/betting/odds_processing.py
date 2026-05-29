# ==============================================================================
# golf_model/betting/odds_processing.py
# ==============================================================================
#
# ODDS PROCESSING & OVERROUND REMOVAL
# --------------------------------------
# Bookmaker odds include a built-in margin (overround/vig). Before comparing
# model probabilities to market probabilities, we must remove this margin
# to extract the bookmaker's "true" implied probabilities.
#
# Three methods implemented:
#
# 1. SHIN'S METHOD (default, recommended):
#    Shin (1991, 1993) models the market as containing an "insider" fraction z.
#    The bookmaker protects against informed bettors by widening odds on 
#    favorites and narrowing on longshots. This creates a specific shape 
#    of distortion that Shin's method corrects.
#    
#    Solves: π_i = (√(z² + 4(1-z) p̃_i / S) - z) / (2(1-z))
#    where p̃_i are raw implied probs and S = sum(p̃_i) > 1.
#    
#    Why preferred: Golf outright markets have known favorite-longshot bias.
#    Shin's method explicitly accounts for this structure.
#
# 2. PROPORTIONAL (simple):
#    π_i = p̃_i / S
#    Just scales all probabilities equally. Fast but doesn't correct for
#    the structure of the bias (favorites are over-charged proportionally 
#    less than longshots).
#
# 3. POWER METHOD:
#    Find k such that Σ p̃_i^k = 1.
#    π_i = p̃_i^k
#    A middle ground — accounts for some structure without the full 
#    Shin model.
#
# Reference:
#   Shin, H.S. (1991). "Optimal Betting Odds Against Insider Traders."
#   Shin, H.S. (1993). "Measuring the Incidence of Insider Trading in a Market."
#   Clarke et al. (2017). "Adjusting bookmaker's odds."
#
# ==============================================================================

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


def decimal_odds_to_implied_prob(odds: np.ndarray) -> np.ndarray:
    """
    Convert decimal odds to raw implied probabilities.
    
    Formula: p̃_i = 1 / odds_i
    
    These sum to MORE than 1.0 due to bookmaker overround.
    
    Parameters
    ----------
    odds : np.ndarray
        Decimal odds (e.g., 21.0 for 20/1). Must be > 1.0.
        
    Returns
    -------
    np.ndarray
        Raw implied probabilities (sum > 1.0).
    """
    odds = np.asarray(odds, dtype=np.float64)
    if np.any(odds <= 1.0):
        logger.warning("Found odds <= 1.0. Clipping to 1.01.")
        odds = np.maximum(odds, 1.01)
    return 1.0 / odds


def implied_prob_to_decimal_odds(probs: np.ndarray) -> np.ndarray:
    """Convert true probabilities to fair decimal odds."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 1e-10)  # Avoid division by zero
    return 1.0 / probs


def compute_overround(implied_probs: np.ndarray) -> float:
    """
    Compute the bookmaker's overround (vig/juice).
    
    Overround = Σ p̃_i - 1.0
    
    Example: if implied probs sum to 1.15, overround is 0.15 (15%).
    Typical PGA Tour outright markets: 20–40% overround.
    """
    return float(np.sum(implied_probs) - 1.0)


# ==============================================================================
# METHOD 1: SHIN'S METHOD (recommended)
# ==============================================================================

def remove_overround_shin(
    implied_probs: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, float]:
    """
    Remove overround using Shin's (1993) method.
    
    Solves for the insider fraction z such that the corrected 
    probabilities sum to 1.0.
    
    Parameters
    ----------
    implied_probs : np.ndarray
        Raw implied probabilities (sum > 1.0).
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance for z.
        
    Returns
    -------
    tuple of (true_probs, z)
        true_probs : np.ndarray — corrected probabilities summing to ~1.0.
        z : float — estimated insider trading fraction (typically 0.01–0.10).
        
    Notes
    -----
    The Shin correction formula for each outcome i:
        π_i = (√(z² + 4(1-z)·p̃_i/S) - z) / (2(1-z))
        
    where S = Σ p̃_i is the overround-inclusive sum.
    
    We find z by root-finding: f(z) = Σ π_i(z) - 1 = 0.
    """
    p = np.asarray(implied_probs, dtype=np.float64)
    S = np.sum(p)

    if S <= 1.0:
        # No overround to remove
        return p / np.sum(p), 0.0

    n = len(p)

    def shin_probs(z):
        """Compute Shin-corrected probabilities for given z."""
        numerator = np.sqrt(z**2 + 4 * (1 - z) * p / S) - z
        denominator = 2 * (1 - z)
        return numerator / denominator

    def objective(z):
        """Sum of Shin probs should equal 1."""
        return np.sum(shin_probs(z)) - 1.0

    # Solve for z in (0, 1) — typically z ∈ [0.01, 0.15]
    try:
        z_solution = optimize.brentq(objective, 0.0001, 0.5, xtol=tol, maxiter=max_iter)
        true_probs = shin_probs(z_solution)

        # Safety: ensure non-negative and normalized
        true_probs = np.maximum(true_probs, 0.0)
        true_probs /= np.sum(true_probs)

        return true_probs, float(z_solution)

    except ValueError as e:
        logger.warning(
            "Shin's method failed to converge: %s. Falling back to proportional.", e
        )
        return remove_overround_proportional(p), 0.0


# ==============================================================================
# METHOD 2: PROPORTIONAL (simple)
# ==============================================================================

def remove_overround_proportional(
    implied_probs: np.ndarray,
) -> np.ndarray:
    """
    Remove overround by proportional scaling.
    
    π_i = p̃_i / Σ p̃_j
    
    Simple and fast. Doesn't account for the structure of the bias.
    """
    p = np.asarray(implied_probs, dtype=np.float64)
    total = np.sum(p)
    if total == 0:
        return p
    return p / total


# ==============================================================================
# METHOD 3: POWER METHOD
# ==============================================================================

def remove_overround_power(
    implied_probs: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Remove overround using the power method.
    
    Find k such that Σ p̃_i^k = 1.
    Then π_i = p̃_i^k.
    
    k > 1 when overround is positive (probabilities sum > 1).
    This compresses probabilities, with larger compression on smaller probs.
    """
    p = np.asarray(implied_probs, dtype=np.float64)
    p = np.maximum(p, 1e-15)  # Avoid log(0)

    def objective(k):
        return np.sum(p ** k) - 1.0

    try:
        k_solution = optimize.brentq(objective, 0.5, 5.0, xtol=tol)
        true_probs = p ** k_solution
        true_probs /= np.sum(true_probs)  # Safety normalization
        return true_probs
    except ValueError:
        logger.warning("Power method failed. Falling back to proportional.")
        return remove_overround_proportional(p)


# ==============================================================================
# UNIFIED INTERFACE
# ==============================================================================

def remove_overround(
    implied_probs: np.ndarray,
    method: str = "shin",
) -> np.ndarray:
    """
    Remove bookmaker overround using the specified method.
    
    Parameters
    ----------
    implied_probs : np.ndarray
        Raw implied probabilities (sum > 1.0).
    method : str
        One of "shin", "proportional", "power".
        
    Returns
    -------
    np.ndarray
        True probabilities summing to ~1.0.
    """
    method = method.lower()

    if method == "shin":
        probs, z = remove_overround_shin(implied_probs)
        return probs
    elif method == "proportional":
        return remove_overround_proportional(implied_probs)
    elif method == "power":
        return remove_overround_power(implied_probs)
    else:
        raise ValueError(f"Unknown overround method: {method}. "
                         f"Use 'shin', 'proportional', or 'power'.")


def process_tournament_odds(
    odds_df: "pd.DataFrame",
    book: str = "pinnacle",
    market: str = "win",
    method: str = "shin",
) -> "pd.DataFrame":
    """
    Full pipeline: extract odds for one book/market → devig → return clean probs.
    
    Parameters
    ----------
    odds_df : pd.DataFrame
        Raw odds data (must contain: player_id, book, decimal_odds).
    book : str
        Sportsbook to process (e.g., "pinnacle").
    market : str
        Market type (e.g., "win").
    method : str
        Overround removal method.
        
    Returns
    -------
    pd.DataFrame
        Columns: player_id, decimal_odds, implied_prob, true_prob.
    """
    import pandas as pd

    # Normalize column names: support both loader conventions and legacy names
    col_map = {}
    if "bookmaker" in odds_df.columns and "book" not in odds_df.columns:
        col_map["bookmaker"] = "book"
    if "close_odds" in odds_df.columns and "decimal_odds" not in odds_df.columns:
        col_map["close_odds"] = "decimal_odds"
    if "dg_id" in odds_df.columns and "player_id" not in odds_df.columns:
        col_map["dg_id"] = "player_id"

    if col_map:
        odds_df = odds_df.rename(columns=col_map)

    # Filter to specific book and market
    mask = odds_df["book"].str.lower() == book.lower()
    if "market" in odds_df.columns:
        mask &= odds_df["market"].str.lower() == market.lower()

    book_odds = odds_df[mask].copy()

    if len(book_odds) == 0:
        logger.warning("No odds found for book='%s', market='%s'", book, market)
        return pd.DataFrame()

    # Convert to implied probabilities
    book_odds["implied_prob"] = decimal_odds_to_implied_prob(
        book_odds["decimal_odds"].values
    )

    overround = compute_overround(book_odds["implied_prob"].values)
    logger.info(
        "Processing %s %s odds | %d players | overround=%.1f%% | method=%s",
        book, market, len(book_odds), overround * 100, method,
    )

    # Remove overround
    true_probs = remove_overround(book_odds["implied_prob"].values, method=method)
    book_odds["true_prob"] = true_probs

    logger.info(
        "Devigged probs sum=%.6f | max=%.4f | min=%.6f",
        true_probs.sum(), true_probs.max(), true_probs.min(),
    )

    return book_odds[["player_id", "decimal_odds", "implied_prob", "true_prob"]].copy()
