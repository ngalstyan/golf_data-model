# ==============================================================================
# golf_model/data/schemas.py
# ==============================================================================
#
# DATA CONTRACTS (SCHEMAS)
# -------------------------
# Every DataFrame entering the system is validated against these schemas.
# If the data doesn't match, we fail LOUDLY — not silently.
#
# Purpose:
#   - Catch data issues at ingestion, not deep inside the model.
#   - Document exactly what each DataFrame should contain.
#   - Enable automated testing of data pipelines.
#
# Design:
#   Each schema is a dataclass containing:
#     - REQUIRED_COLUMNS: dict of {column_name: expected_dtype}
#     - OPTIONAL_COLUMNS: dict of {column_name: expected_dtype}
#     - validate(df): method that checks a DataFrame against the schema
#
# Usage:
#   from data.schemas import RoundsSchema, validate_dataframe
#   issues = validate_dataframe(df, RoundsSchema)
#   if issues:
#       for issue in issues:
#           logger.error(issue)
#
# ==============================================================================

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# SCHEMA DEFINITIONS
# ==============================================================================
# 
# These map directly to the DataGolf API outputs and our CSV files:
#
# 1. RoundsSchema    — Round-level player performance (SG data)
# 2. EventsSchema    — Tournament metadata (event list)
# 3. OddsSchema      — Historical bookmaker odds
# 4. CourseSchema     — Course characteristics (γ_c vector)
# 5. PlayerSchema     — Player metadata & ratings
# ==============================================================================


@dataclass(frozen=True)
class RoundsSchema:
    """
    Schema for round-level strokes-gained data.
    
    This is the CORE dataset of the entire model. Each row represents
    one golfer's performance in one round of one tournament.
    
    Source: DataGolf API → historical-raw-data/rounds endpoint
    File:   rounds_YYYY.csv or sg_rounds.csv
    
    Key columns:
        player_id (int)     — DataGolf unique player identifier
        event_id (int)      — DataGolf unique event identifier  
        round_num (int)     — Round number within tournament (1–4)
        course_id (int)     — Course identifier (some events use multiple courses)
        date (str)          — Calendar date of the round (YYYY-MM-DD)
        sg_total (float)    — Total strokes gained vs field (Broadie methodology)
        sg_ott (float)      — Strokes gained: Off-the-Tee (driving)
        sg_app (float)      — Strokes gained: Approach (iron play)
        sg_arg (float)      — Strokes gained: Around-the-Green (chipping)
        sg_putt (float)     — Strokes gained: Putting
        
    Mathematical role:
        This provides Y_{i,r,t} in the observation equation:
        Y_{i,r,t} = μ_{i,t} + γ_{c(t)} · δ_i + ε_{i,r,t}
        
        Each SG component (OTT, APP, ARG, PUTT) is decomposed separately
        in the hierarchical model's sub-component structure.
    """

    # Column name → expected pandas dtype string
    REQUIRED_COLUMNS: Dict[str, str] = field(default_factory=lambda: {
        "dg_id":    "int",       # DataGolf player ID
        "event_id":     "int",       # DataGolf event ID
        "round_num":    "int",       # 1, 2, 3, or 4
        "year":         "int",       # YYYY-MM-DD (will be parsed to datetime)
        "sg_total":     "float",     # Total strokes gained
        "sg_ott":       "float",     # Off-the-Tee
        "sg_app":       "float",     # Approach
        "sg_arg":       "float",     # Around-the-Green
        "sg_putt":      "float",     # Putting
    })

    OPTIONAL_COLUMNS: Dict[str, str] = field(default_factory=lambda: {
        "player_name":      "str",   # Human-readable name
        "event_name":       "str",   # Tournament name
        "course_id":        "int",   # Course identifier (if multi-course)
        "course_name":      "str",   # Course name
        "score":            "int",   # Raw stroke score
        "score_vs_par":     "int",   # Score relative to par
        "tee_time":         "str",   # Tee time (for wave assignment)
        "season":           "int",   # PGA Tour season year
        "tour":             "str",   # Tour identifier (pga, kft, euro)
        "made_cut":         "bool",  # Whether player made the cut
        "finish_position":  "str",   # Final position (e.g., "T15", "1")
    })

    # Validation rules beyond type checking
    ROUND_NUM_RANGE: tuple = (1, 4)
    SG_REASONABLE_RANGE: tuple = (-25.0, 15.0)  # SG outside this = likely error


@dataclass(frozen=True)
class EventsSchema:
    """
    Schema for tournament/event metadata.
    
    Source: DataGolf API → historical-raw-data/event-list endpoint
    File:   events.csv or tournament_list.csv
    
    Key columns:
        event_id (int)      — DataGolf unique event identifier
        event_name (str)    — Tournament name
        tour (str)          — Tour (pga, kft, euro)
        calendar_year (int) — Calendar year of the event
        season (int)        — PGA Tour season year
        course_id (int)     — Primary course identifier
        course_name (str)   — Primary course name
        start_date (str)    — Tournament start date (YYYY-MM-DD)
        
    Mathematical role:
        Provides tournament-level metadata for indexing (the 't' subscript)
        and course identification (for γ_{c(t)} lookup).
    """

    REQUIRED_COLUMNS: Dict[str, str] = field(default_factory=lambda: {
        "event_id":         "int",
        "event_name":       "str",
        "calendar_year":    "int",
    })

    OPTIONAL_COLUMNS: Dict[str, str] = field(default_factory=lambda: {
        "tour":             "str",
        "start_date":       "str",
        "season":           "int",
        "course_id":        "int",
        "course_name":      "str",
        "end_date":         "str",
        "purse":            "float",   # Tournament purse ($)
        "field_size":       "int",     # Number of players
        "sg_categories":    "bool",    # Whether SG data is available
        "latitude":         "float",   # Course location (for weather)
        "longitude":        "float",   # Course location (for weather)
    })


@dataclass(frozen=True)
class OddsSchema:
    """
    Schema for historical bookmaker odds.
    
    Source: DataGolf API → historical-odds/outrights endpoint
    File:   odds_outrights.csv
    
    Key columns:
        event_id (int)          — Links to EventsSchema
        player_id (int)         — Links to RoundsSchema
        book (str)              — Sportsbook name (e.g., "pinnacle")
        market (str)            — Market type (e.g., "win", "top_5")
        decimal_odds (float)    — Decimal odds (e.g., 21.0 for 20/1)
        
    Mathematical role:
        Provides P_market for edge calculation:
        Edge_i = P_model(i) - P_market(i)
        
        Pinnacle closing odds serve as the "sharp" benchmark.
        Softer books (DraftKings, etc.) are where bets are placed.
    """

    REQUIRED_COLUMNS: Dict[str, str] = field(default_factory=lambda: {
    "event_id":         "int",
    "dg_id":            "int",      # DataGolf player ID (was: player_id)
    "bookmaker":        "str",      # Sportsbook name (was: book)
    "close_odds":       "float",    # Closing decimal odds (was: decimal_odds)
    })

    OPTIONAL_COLUMNS: Dict[str, str] = field(default_factory=lambda: {
    "player_name":          "str",
    "event_name":           "str",
    "market":               "str",
    "open_odds":            "float",   # Opening decimal odds
    "open_time":            "str",
    "close_time":           "str",
    "implied_prob_open":    "float",
    "implied_prob_close":   "float",
    "outcome":              "str",
    "bet_outcome_numeric":  "float",
    "bet_outcome_text":     "str",
    "event_completed":      "str",
    "season":               "int",
    "calendar_year":        "int",
    })

    MIN_VALID_ODDS: float = 1.01  # Below this = clearly wrong


@dataclass(frozen=True)
class CourseSchema:
    """
    Schema for course characteristics (γ_c vector).
    
    Source: Manual research + DataGolf course metadata + public sources
    File:   course_features.csv
    
    Key columns match COURSE_FEATURE_NAMES from settings:
        course_id (int)         — Links to EventsSchema
        length_yards (float)    — Total course length
        rough_height_in (float) — Primary rough height (inches)
        green_speed_stimp (float) — Stimpmeter reading
        wind_exposure (float)   — Wind exposure index (0–1)
        elevation_ft (float)    — Elevation (feet above sea level)
        fairway_width_avg (float) — Avg fairway width (yards)
        green_size_sqft (float) — Avg green size (sq ft)
        water_hazard_pct (float) — Pct holes with water in play
        
    Mathematical role:
        This IS γ_c — the course characteristic vector in:
        CourseFit_{i,c} = γ_c · δ_i
        
        Each column is Z-score standardized across all courses before use.
    """

    REQUIRED_COLUMNS: Dict[str, str] = field(default_factory=lambda: {
        "course_id":    "int",
    })

    OPTIONAL_COLUMNS: Dict[str, str] = field(default_factory=lambda: {
        "course_name":          "str",
        "length_yards":         "float",
        "rough_height_in":      "float",
        "green_speed_stimp":    "float",
        "wind_exposure":        "float",
        "elevation_ft":         "float",
        "fairway_width_avg":    "float",
        "green_size_sqft":      "float",
        "water_hazard_pct":     "float",
        "par":                  "int",
        "latitude":             "float",
        "longitude":            "float",
    })


@dataclass(frozen=True)
class PlayerSchema:
    """
    Schema for player metadata and ratings.
    
    Source: DataGolf API → player-list, rankings endpoints
    File:   players.csv
    """

    REQUIRED_COLUMNS: Dict[str, str] = field(default_factory=lambda: {
        "player_id":    "int",
        "player_name":  "str",
    })

    OPTIONAL_COLUMNS: Dict[str, str] = field(default_factory=lambda: {
        "country":          "str",
        "tour":             "str",
        "dg_skill_estimate": "float",  # DataGolf's own skill rating
        "owgr_rank":        "int",     # Official World Golf Ranking
        "amateur":          "bool",
    })


# ==============================================================================
# VALIDATION ENGINE
# ==============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    schema: object,
    strict: bool = False,
) -> List[str]:
    """
    Validate a DataFrame against a schema.
    
    Checks performed:
        1. All REQUIRED_COLUMNS present
        2. No REQUIRED_COLUMNS entirely null
        3. Types are coercible to expected dtypes
        4. Schema-specific range checks (if defined)
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
        
    schema : Schema dataclass instance
        One of: RoundsSchema(), EventsSchema(), OddsSchema(), etc.
        
    strict : bool, default False
        If True, also warns about unexpected columns not in the schema.
        
    Returns
    -------
    List[str]
        List of issue descriptions. Empty list = validation passed.
        
    Examples
    --------
    >>> issues = validate_dataframe(rounds_df, RoundsSchema())
    >>> if issues:
    ...     for issue in issues:
    ...         print(f"  ⚠ {issue}")
    ... else:
    ...     print("  ✓ Schema validation passed")
    """
    issues: List[str] = []
    schema_name = type(schema).__name__

    if df is None or df.empty:
        issues.append(f"{schema_name}: DataFrame is empty or None")
        return issues

    # --- Check 1: Required columns present ---
    required = schema.REQUIRED_COLUMNS
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        issues.append(
            f"{schema_name}: Missing required columns: {missing_cols}"
        )
        # Can't do further checks on missing columns
        required = {k: v for k, v in required.items() if k in df.columns}

    # --- Check 2: Required columns not entirely null ---
    for col in required:
        if col in df.columns and df[col].isna().all():
            issues.append(
                f"{schema_name}: Column '{col}' is entirely null"
            )

    # --- Check 3: Type coercibility ---
    for col, expected_type in required.items():
        if col not in df.columns:
            continue

        actual_dtype = df[col].dtype

        if expected_type == "int":
            # Allow int or float (pandas often reads ints as float64 due to NaN)
            if not (np.issubdtype(actual_dtype, np.integer) or
                    np.issubdtype(actual_dtype, np.floating)):
                issues.append(
                    f"{schema_name}: Column '{col}' expected int, "
                    f"got {actual_dtype}"
                )

        elif expected_type == "float":
            if not np.issubdtype(actual_dtype, np.number):
                issues.append(
                    f"{schema_name}: Column '{col}' expected float, "
                    f"got {actual_dtype}"
                )

        elif expected_type == "str":
            if not (actual_dtype == object or pd.api.types.is_string_dtype(actual_dtype)):
                issues.append(
                    f"{schema_name}: Column '{col}' expected str, "
                    f"got {actual_dtype}"
                )

        elif expected_type == "bool":
            if not (actual_dtype == bool or actual_dtype == object):
                issues.append(
                    f"{schema_name}: Column '{col}' expected bool, "
                    f"got {actual_dtype}"
                )

    # --- Check 4: Schema-specific range validation ---
    # Rounds: check round_num range
    if hasattr(schema, "ROUND_NUM_RANGE") and "round_num" in df.columns:
        lo, hi = schema.ROUND_NUM_RANGE
        out_of_range = df[
            (df["round_num"] < lo) | (df["round_num"] > hi)
        ]
        if len(out_of_range) > 0:
            issues.append(
                f"{schema_name}: {len(out_of_range)} rows have "
                f"round_num outside [{lo}, {hi}]"
            )

    # Rounds: check SG values in reasonable range
    if hasattr(schema, "SG_REASONABLE_RANGE"):
        lo, hi = schema.SG_REASONABLE_RANGE
        sg_cols = [c for c in ["sg_total", "sg_ott", "sg_app", "sg_arg", "sg_putt"]
                   if c in df.columns]
        for col in sg_cols:
            extreme = df[(df[col] < lo) | (df[col] > hi)][col]
            if len(extreme) > 0:
                issues.append(
                    f"{schema_name}: {len(extreme)} rows have {col} "
                    f"outside [{lo}, {hi}]. "
                    f"Range: [{extreme.min():.2f}, {extreme.max():.2f}]"
                )

    # Odds: check minimum valid odds
    if hasattr(schema, "MIN_VALID_ODDS") and "close_odds" in df.columns:
        bad_odds = df[df["close_odds"] < schema.MIN_VALID_ODDS]
        if len(bad_odds) > 0:
            issues.append(
                f"{schema_name}: {len(bad_odds)} rows have decimal_odds "
                f"< {schema.MIN_VALID_ODDS}"
            )

    # --- Check 5 (optional): Unexpected columns ---
    if strict:
        expected_cols = set(required.keys())
        if hasattr(schema, "OPTIONAL_COLUMNS"):
            expected_cols |= set(schema.OPTIONAL_COLUMNS.keys())
        unexpected = set(df.columns) - expected_cols
        if unexpected:
            issues.append(
                f"{schema_name}: Unexpected columns: {sorted(unexpected)}"
            )

    return issues
