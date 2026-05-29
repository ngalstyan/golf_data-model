# ==============================================================================
# golf_model/features/strokes_gained.py
# ==============================================================================
#
# STROKES-GAINED DECOMPOSITION & NORMALIZATION
# -----------------------------------------------
# Processes raw round-level SG data into model-ready features.
#
# Mathematical Background (Broadie, 2012):
#   SG measures how many strokes a player gains vs the field average
#   in a specific category per round. By definition:
#     Σ SG_total across all players in a round ≈ 0
#
#   Sub-components:
#     SG_total = SG_OTT + SG_APP + SG_ARG + SG_PUTT
#
#   In our model, these map to the observation equation:
#     Y_{i,r,t}^{(k)} = μ_{i,t}^{(k)} + ε_{i,r,t}^{(k)}
#   where k ∈ {OTT, APP, ARG, PUTT}.
#
# This module handles:
#   1. Validation that SG components sum to total (within tolerance)
#   2. Field-strength adjustment: normalize SG by tournament field quality
#   3. Per-round aggregation and summary statistics
#   4. SG profile computation (player's relative strengths across categories)
#
# ==============================================================================

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class StrokesGainedProcessor:
    """
    Process and validate strokes-gained data.
    
    Parameters
    ----------
    settings : Settings
        Project configuration.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.cfg = settings or Settings()
        self.sg_cols = self.cfg.SG_COMPONENTS       # [sg_ott, sg_app, sg_arg, sg_putt]
        self.sg_total = self.cfg.SG_TOTAL_COL       # sg_total

    def validate_sg_decomposition(
        self,
        df: pd.DataFrame,
        tolerance: float = 0.05,
    ) -> pd.DataFrame:
        """
        Verify SG components sum to SG total within tolerance.
        
        This is a critical data quality check. If components don't sum
        to total, the data has an error or uses a different decomposition.
        
        Parameters
        ----------
        df : pd.DataFrame
            Round-level data with SG columns.
        tolerance : float, default 0.05
            Maximum allowed absolute difference between sum and total.
            
        Returns
        -------
        pd.DataFrame
            Input DataFrame with added '_sg_residual' column showing
            the difference between component sum and reported total.
        """
        df = df.copy()

        # Check all required columns exist
        missing = [c for c in self.sg_cols + [self.sg_total] if c not in df.columns]
        if missing:
            logger.warning("Missing SG columns for decomposition check: %s", missing)
            return df

        # Compute component sum
        component_sum = df[self.sg_cols].sum(axis=1)
        df["_sg_residual"] = df[self.sg_total] - component_sum

        # Report violations
        violations = df[df["_sg_residual"].abs() > tolerance]
        if len(violations) > 0:
            logger.warning(
                "SG decomposition check: %d/%d rounds (%.1f%%) have "
                "|residual| > %.3f. Max residual: %.4f",
                len(violations), len(df),
                100 * len(violations) / len(df),
                tolerance,
                df["_sg_residual"].abs().max(),
            )
        else:
            logger.info(
                "SG decomposition check passed: all %d rounds within tolerance",
                len(df),
            )

        return df

    def compute_sg_profiles(
        self,
        df: pd.DataFrame,
        min_rounds: int = 20,
    ) -> pd.DataFrame:
        """
        Compute each player's SG profile: average SG in each sub-category.
        
        The profile vector [SG_OTT, SG_APP, SG_ARG, SG_PUTT] reveals
        WHERE a player gains/loses strokes — critical for course-fit modeling.
        
        A player with high SG_OTT but low SG_PUTT is a "bomber" who thrives
        on long courses but struggles on those demanding precision putting.
        
        Parameters
        ----------
        df : pd.DataFrame
            Round-level data.
        min_rounds : int, default 20
            Minimum rounds for a reliable profile estimate.
            
        Returns
        -------
        pd.DataFrame
            One row per player with columns:
            [player_id, n_rounds, sg_total_mean, sg_ott_mean, ..., sg_putt_mean,
             sg_total_std, sg_ott_std, ..., sg_putt_std]
        """
        if "player_id" not in df.columns:
            raise ValueError("DataFrame must contain 'player_id' column")

        all_sg = [self.sg_total] + self.sg_cols

        # Group by player and compute stats
        grouped = df.groupby("player_id")

        # Means
        means = grouped[all_sg].mean()
        means.columns = [f"{c}_mean" for c in means.columns]

        # Standard deviations
        stds = grouped[all_sg].std()
        stds.columns = [f"{c}_std" for c in stds.columns]

        # Round counts
        counts = grouped[self.sg_total].count().rename("n_rounds")

        # Combine
        profiles = pd.concat([counts, means, stds], axis=1).reset_index()

        # Flag players below minimum rounds
        profiles["sufficient_data"] = profiles["n_rounds"] >= min_rounds

        n_sufficient = profiles["sufficient_data"].sum()
        logger.info(
            "SG profiles computed: %d players total, %d with ≥%d rounds",
            len(profiles), n_sufficient, min_rounds,
        )

        return profiles

    def normalize_by_field_strength(
        self,
        rounds_df: pd.DataFrame,
        field_strength_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Adjust SG values by tournament field strength.
        
        Motivation:
            Gaining 2.0 SG/round against the Korn Ferry Tour field is
            NOT equivalent to gaining 2.0 SG/round against a major
            championship field. Field-strength adjustment corrects this.
        
        Method:
            Adjusted_SG_{i,r,t} = Raw_SG_{i,r,t} + FieldStrength_t
            
            where FieldStrength_t is the average skill level of
            tournament t's field (measured in SG vs PGA Tour average).
        
        Parameters
        ----------
        rounds_df : pd.DataFrame
            Round-level data with SG columns and 'event_id'.
        field_strength_df : pd.DataFrame
            Field strength per event: columns ['event_id', 'field_strength'].
            
        Returns
        -------
        pd.DataFrame
            Rounds with adjusted SG columns (suffixed '_adj').
        """
        df = rounds_df.copy()

        if "event_id" not in df.columns:
            logger.warning("Cannot adjust by field strength: no 'event_id' column")
            return df

        # Merge field strength
        df = df.merge(
            field_strength_df[["event_id", "field_strength"]],
            on="event_id",
            how="left",
        )

        missing_fs = df["field_strength"].isna().sum()
        if missing_fs > 0:
            logger.warning(
                "%d rounds (%.1f%%) have no field strength data — "
                "using 0.0 (no adjustment)",
                missing_fs, 100 * missing_fs / len(df),
            )
            df["field_strength"] = df["field_strength"].fillna(0.0)

        # Adjust each SG component
        for col in [self.sg_total] + self.sg_cols:
            if col in df.columns:
                df[f"{col}_adj"] = df[col] + df["field_strength"]

        logger.info(
            "Field-strength adjustment applied: mean FS = %.3f",
            df["field_strength"].mean(),
        )

        return df


class FieldStrengthCalculator:
    """
    Compute tournament field strength index.
    
    Field strength measures the average quality of players in a 
    tournament's field, expressed in strokes gained vs PGA Tour average.
    
    Methodology:
        1. Each player has a "true skill" estimate (rolling SG average).
        2. For each tournament, average the skills of all participants.
        3. Express relative to PGA Tour season average (which is 0 by definition).
        
    Higher field strength = harder field = SG values more impressive.
    
    Parameters
    ----------
    settings : Settings
        Project configuration.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.cfg = settings or Settings()

    def compute_field_strength(
        self,
        rounds_df: pd.DataFrame,
        lookback_rounds: int = 50,
    ) -> pd.DataFrame:
        """
        Compute field strength for each tournament.
        
        Algorithm:
            For each tournament t:
              1. Get the set of players P_t who participated.
              2. For each player i ∈ P_t, compute their rolling
                 average SG_total over their most recent `lookback_rounds`
                 rounds BEFORE tournament t.
              3. FieldStrength_t = mean of these player skill estimates.
              
        Parameters
        ----------
        rounds_df : pd.DataFrame
            Round-level data sorted by date, with columns:
            [player_id, event_id, date, sg_total].
        lookback_rounds : int, default 50
            Number of most recent rounds to estimate player skill.
            
        Returns
        -------
        pd.DataFrame
            Columns: [event_id, field_strength, field_size, avg_player_skill_std]
        """
        df = rounds_df.copy()

        if "date" not in df.columns or "sg_total" not in df.columns:
            raise ValueError("rounds_df must contain 'date' and 'sg_total' columns")

        # Ensure sorted by date
        df = df.sort_values("date").reset_index(drop=True)

        # Get unique events in chronological order
        events = (
            df.groupby("event_id")["date"]
            .min()
            .sort_values()
            .reset_index()
            .rename(columns={"date": "event_start"})
        )

        results = []

        for _, event_row in events.iterrows():
            eid = event_row["event_id"]
            event_start = event_row["event_start"]

            # Players in this event
            event_players = df[df["event_id"] == eid]["player_id"].unique()

            # Prior rounds for each player (before this event)
            prior = df[
                (df["date"] < event_start) &
                (df["player_id"].isin(event_players))
            ]

            if prior.empty:
                results.append({
                    "event_id": eid,
                    "field_strength": 0.0,
                    "field_size": len(event_players),
                    "avg_player_skill_std": np.nan,
                })
                continue

            # Rolling average SG for each player (most recent N rounds)
            player_skills = (
                prior.groupby("player_id")
                .apply(lambda g: g.nlargest(lookback_rounds, "date")["sg_total"].mean())
                .reset_index(name="skill_estimate")
            )

            # Only include players who actually have prior data
            player_skills = player_skills.dropna(subset=["skill_estimate"])

            if player_skills.empty:
                fs = 0.0
                skill_std = np.nan
            else:
                fs = player_skills["skill_estimate"].mean()
                skill_std = player_skills["skill_estimate"].std()

            results.append({
                "event_id": eid,
                "field_strength": fs,
                "field_size": len(event_players),
                "avg_player_skill_std": skill_std,
            })

        result_df = pd.DataFrame(results)
        logger.info(
            "Field strength computed for %d events | "
            "range: [%.3f, %.3f] | mean: %.3f",
            len(result_df),
            result_df["field_strength"].min(),
            result_df["field_strength"].max(),
            result_df["field_strength"].mean(),
        )

        return result_df
