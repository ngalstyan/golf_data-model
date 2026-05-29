# ==============================================================================
# golf_model/utils/helpers.py
# ==============================================================================
#
# GENERAL UTILITY FUNCTIONS
# --------------------------
# Small, reusable functions used across multiple modules.
# Each function is pure (no side effects) and independently testable.
#
# ==============================================================================

import hashlib
import re
from datetime import datetime, date
from typing import Optional, Union

import numpy as np
import pandas as pd


# ==============================================================================
# PLAYER ID NORMALIZATION
# ==============================================================================
# DataGolf uses numeric player IDs, but names may appear in different formats
# across data sources. These helpers ensure consistent player identification.
# ==============================================================================

def normalize_player_name(name: str) -> str:
    """
    Normalize a golfer's name for consistent matching across data sources.
    
    Transformations:
        - Lowercase
        - Strip whitespace
        - Remove accents (é → e, ñ → n)
        - Remove suffixes (Jr., III, etc.)
        - "Last, First" → "first last"
    
    Parameters
    ----------
    name : str
        Raw player name from any source.
        
    Returns
    -------
    str
        Normalized name string.
        
    Examples
    --------
    >>> normalize_player_name("Scheffler, Scottie")
    'scottie scheffler'
    >>> normalize_player_name("Hovland, Viktor")
    'viktor hovland'
    >>> normalize_player_name("Homa, Max Jr.")
    'max homa'
    """
    if not name or not isinstance(name, str):
        return ""

    # Lowercase and strip
    name = name.lower().strip()

    # Handle "Last, First" format → "First Last"
    if "," in name:
        parts = name.split(",", 1)
        name = f"{parts[1].strip()} {parts[0].strip()}"

    # Remove common suffixes
    for suffix in [" jr.", " jr", " sr.", " sr", " iii", " ii", " iv"]:
        name = name.replace(suffix, "")

    # Remove accents (basic transliteration)
    accent_map = {
        "á": "a", "à": "a", "â": "a", "ä": "a", "ã": "a",
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "í": "i", "ì": "i", "î": "i", "ï": "i",
        "ó": "o", "ò": "o", "ô": "o", "ö": "o", "õ": "o",
        "ú": "u", "ù": "u", "û": "u", "ü": "u",
        "ñ": "n", "ø": "o", "å": "a", "æ": "ae",
    }
    for accented, plain in accent_map.items():
        name = name.replace(accented, plain)

    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name).strip()

    return name


def normalize_player_id(player_id: Union[int, str]) -> int:
    """
    Ensure player ID is a consistent integer.
    
    DataGolf uses integer IDs. This handles cases where IDs 
    arrive as strings (from CSV parsing) or floats (from NaN handling).
    
    Parameters
    ----------
    player_id : int or str
        Raw player ID.
        
    Returns
    -------
    int
        Validated integer player ID.
        
    Raises
    ------
    ValueError
        If player_id cannot be converted to a valid integer.
    """
    try:
        pid = int(float(player_id))
        if pid <= 0:
            raise ValueError(f"Player ID must be positive, got {pid}")
        return pid
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid player ID: {player_id!r}") from e


# ==============================================================================
# DATE & SEASON UTILITIES
# ==============================================================================
# The PGA Tour season wraps around the calendar year (October start).
# These helpers handle the season ↔ calendar date mapping.
# ==============================================================================

def season_from_date(d: Union[str, date, datetime, pd.Timestamp]) -> int:
    """
    Determine the PGA Tour season year from a calendar date.
    
    PGA Tour seasons typically start in October of the prior year.
    For example, a tournament in January 2024 is part of the 2024 season.
    A tournament in October 2023 is part of the 2024 season.
    
    Simplified rule used here: 
        If month >= 10 → next year's season.
        Otherwise → current year's season.
    
    Parameters
    ----------
    d : str, date, datetime, or pd.Timestamp
        The calendar date of the event.
        
    Returns
    -------
    int
        PGA Tour season year.
        
    Examples
    --------
    >>> season_from_date("2024-01-15")
    2024
    >>> season_from_date("2023-10-05")
    2024
    >>> season_from_date("2023-09-30")
    2023
    """
    if isinstance(d, str):
        d = pd.Timestamp(d)
    elif isinstance(d, date) and not isinstance(d, datetime):
        d = pd.Timestamp(d)

    if d.month >= 10:
        return d.year + 1
    return d.year


def days_between(
    date1: Union[str, date, datetime, pd.Timestamp],
    date2: Union[str, date, datetime, pd.Timestamp],
) -> int:
    """
    Calculate absolute number of days between two dates.
    
    Parameters
    ----------
    date1, date2 : date-like
        Two dates in any parseable format.
        
    Returns
    -------
    int
        Absolute number of days between the dates.
    """
    d1 = pd.Timestamp(date1)
    d2 = pd.Timestamp(date2)
    return abs((d2 - d1).days)


def date_to_str(d: Union[str, date, datetime, pd.Timestamp]) -> str:
    """Convert any date-like to ISO format string YYYY-MM-DD."""
    return pd.Timestamp(d).strftime("%Y-%m-%d")


# ==============================================================================
# MATHEMATICAL UTILITIES
# ==============================================================================

def exponential_weights(
    n: int,
    half_life: float,
    normalize: bool = True,
) -> np.ndarray:
    """
    Generate exponential decay weights for a sequence of n observations.
    
    Weight for observation j (where j=0 is most recent):
        w_j = exp(-λ * j)
    
    where λ = ln(2) / half_life.
    
    Parameters
    ----------
    n : int
        Number of observations.
    half_life : float
        Number of steps (rounds or days) until weight drops to 50%.
    normalize : bool, default True
        If True, weights sum to 1.0.
        
    Returns
    -------
    np.ndarray
        Array of shape (n,) with weights. Index 0 = most recent.
        
    Examples
    --------
    >>> exponential_weights(5, half_life=3, normalize=True)
    array([0.3012, 0.2390, 0.1897, 0.1506, 0.1195])  # approx, sums to 1.0
    
    Notes
    -----
    This is the core weight function used in the EWMA time-weighting
    component (Phase 2). The half-life determines how quickly old
    rounds are downweighted. 
    
    Mathematical derivation:
        We want w(t_half) = 0.5 * w(0)
        exp(-λ * t_half) = 0.5
        -λ * t_half = ln(0.5) = -ln(2)
        λ = ln(2) / t_half
    """
    if n <= 0:
        return np.array([])

    lam = np.log(2) / half_life
    j = np.arange(n, dtype=np.float64)
    weights = np.exp(-lam * j)

    if normalize and weights.sum() > 0:
        weights /= weights.sum()

    return weights


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    fill_value: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Division with zero-safe handling.
    
    Returns fill_value wherever denominator is 0 or NaN.
    Avoids RuntimeWarning from numpy division.
    
    Parameters
    ----------
    numerator : float or array
    denominator : float or array
    fill_value : float, default 0.0
        Value to use where division is undefined.
        
    Returns
    -------
    float or array
        Result of numerator / denominator with safe handling.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    result = np.where(
        (den == 0) | np.isnan(den),
        fill_value,
        num / np.where(den == 0, 1.0, den),  # Avoid actual division by zero
    )
    return float(result) if result.ndim == 0 else result


# ==============================================================================
# DATA QUALITY UTILITIES
# ==============================================================================

def check_missing_pct(df: pd.DataFrame, column: str) -> float:
    """Return percentage of missing values in a column (0.0 to 100.0)."""
    if column not in df.columns:
        return 100.0
    return (df[column].isna().sum() / len(df)) * 100.0


def summarize_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> str:
    """
    Generate a concise summary string for logging.
    
    Parameters
    ----------
    df : pd.DataFrame
    name : str
        Label for this DataFrame.
        
    Returns
    -------
    str
        Summary like "Rounds: 45,231 rows × 18 cols | 3 players, 2019–2023"
    """
    n_rows, n_cols = df.shape
    summary = f"{name}: {n_rows:,} rows × {n_cols} cols"

    if "player_id" in df.columns:
        n_players = df["player_id"].nunique()
        summary += f" | {n_players:,} players"

    if "date" in df.columns or "event_date" in df.columns:
        date_col = "date" if "date" in df.columns else "event_date"
        try:
            dates = pd.to_datetime(df[date_col])
            summary += f" | {dates.min().year}–{dates.max().year}"
        except Exception:
            pass

    return summary
