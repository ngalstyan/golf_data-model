# ==============================================================================
# golf_model/features/pipeline.py
# ==============================================================================
#
# MASTER FEATURE ENGINEERING PIPELINE
# -------------------------------------
# Orchestrates all feature engineering steps in the correct order:
#   1. Validate SG decomposition
#   2. Compute field strength
#   3. Adjust SG by field strength
#   4. Apply EWMA time-weighting
#   5. Standardize course features
#   6. Package everything for the model layer
#
# This is the SINGLE entry point for transforming raw data → model features.
# Downstream code (models, simulation) never touches raw data directly.
#
# Usage:
#   from features.pipeline import FeaturePipeline
#   pipeline = FeaturePipeline(Settings())
#   features = pipeline.run(rounds_df, events_df, course_df, as_of_date=...)
#
# ==============================================================================

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from config.settings import Settings
from features.strokes_gained import StrokesGainedProcessor, FieldStrengthCalculator
from features.time_weighting import TimeWeighter
from features.course_features import CourseFeatureProcessor
from utils.logger import get_logger
from utils.helpers import summarize_dataframe

logger = get_logger(__name__)


@dataclass
class ModelFeatures:
    """
    Container for all model-ready features produced by the pipeline.
    
    Attributes
    ----------
    player_features : pd.DataFrame
        One row per player. Contains EWMA SG estimates (total and components),
        effective sample sizes, variance estimates. This maps to the
        μ_{i,t} estimates in the observation equation.
        
    course_features : pd.DataFrame
        One row per course. Standardized γ_c vectors for course-fit modeling.
        
    field_strength : pd.DataFrame
        One row per event. Field strength indices for context.
        
    rounds_enriched : pd.DataFrame
        Full round-level data with field-strength adjustments applied.
        Used for model fitting (not prediction).
        
    metadata : dict
        Pipeline run metadata: as_of_date, n_players, n_rounds, etc.
    """
    player_features: pd.DataFrame
    course_features: pd.DataFrame
    field_strength: pd.DataFrame
    rounds_enriched: pd.DataFrame
    metadata: Dict


class FeaturePipeline:
    """
    End-to-end feature engineering pipeline.
    
    Chains all feature processors in the correct dependency order.
    Ensures temporal integrity: no future data leaks into features.
    
    Parameters
    ----------
    settings : Settings
        Project configuration.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.cfg = settings or Settings()
        self.sg_processor = StrokesGainedProcessor(self.cfg)
        self.field_calc = FieldStrengthCalculator(self.cfg)
        self.time_weighter = TimeWeighter(self.cfg)
        self.course_processor = CourseFeatureProcessor(self.cfg)

    def run(
        self,
        rounds_df: pd.DataFrame,
        events_df: Optional[pd.DataFrame] = None,
        course_df: Optional[pd.DataFrame] = None,
        as_of_date: Optional[pd.Timestamp] = None,
        ewma_method: str = "dual",
    ) -> ModelFeatures:
        """
        Execute the full feature engineering pipeline.
        
        Parameters
        ----------
        rounds_df : pd.DataFrame
            Round-level SG data. Must contain:
            [player_id, event_id, date, sg_total, sg_ott, sg_app, sg_arg, sg_putt]
            
        events_df : pd.DataFrame, optional
            Tournament metadata. Used for field strength context.
            
        course_df : pd.DataFrame, optional
            Course characteristics. Used for course-fit features.
            If None, course features are skipped.
            
        as_of_date : pd.Timestamp, optional
            Reference date for EWMA computation. All data after this date
            is excluded. If None, uses max date in rounds_df.
            
        ewma_method : str, default "dual"
            EWMA weighting: "rounds", "days", or "dual".
            
        Returns
        -------
        ModelFeatures
            Container with all model-ready features.
        """
        logger.info("=" * 60)
        logger.info("FEATURE PIPELINE: Starting")
        logger.info("=" * 60)

        # --- Step 0: Temporal filter ---
        rounds = rounds_df.copy()
        rounds["date"] = pd.to_datetime(rounds["date"])

        if as_of_date is not None:
            n_before = len(rounds)
            rounds = rounds[rounds["date"] <= as_of_date]
            n_after = len(rounds)
            logger.info(
                "Step 0: Temporal filter as_of=%s | %d → %d rounds",
                as_of_date.strftime("%Y-%m-%d"), n_before, n_after,
            )
        else:
            as_of_date = rounds["date"].max()
            logger.info(
                "Step 0: No as_of_date specified, using max date: %s",
                as_of_date.strftime("%Y-%m-%d"),
            )

        # --- Step 1: Validate SG decomposition ---
        logger.info("Step 1: Validating SG decomposition...")
        rounds = self.sg_processor.validate_sg_decomposition(rounds)

        # --- Step 2: Compute field strength ---
        logger.info("Step 2: Computing field strength...")
        field_strength = self.field_calc.compute_field_strength(rounds)

        # --- Step 3: Adjust SG by field strength ---
        logger.info("Step 3: Applying field-strength adjustment...")
        rounds_enriched = self.sg_processor.normalize_by_field_strength(
            rounds, field_strength
        )

        # --- Step 4: EWMA time-weighted player features ---
        logger.info("Step 4: Computing EWMA time-weighted player features...")
        player_features = self.time_weighter.compute_weighted_sg(
            rounds_enriched,
            as_of_date=as_of_date,
            method=ewma_method,
        )

        # Add SG profiles (full-history, unweighted) for reference
        sg_profiles = self.sg_processor.compute_sg_profiles(
            rounds_enriched,
            min_rounds=self.cfg.MIN_ROUNDS_FOR_ESTIMATE,
        )
        player_features = player_features.merge(
            sg_profiles[["player_id", "n_rounds", "sufficient_data"]],
            on="player_id",
            how="left",
            suffixes=("_ewma", "_total"),
        )

        # --- Step 5: Course features ---
        if course_df is not None and not course_df.empty:
            logger.info("Step 5: Processing course features...")
            course_df_imputed = self.course_processor.impute_missing_features(course_df)
            course_features = self.course_processor.standardize_features(course_df_imputed)
        else:
            logger.info("Step 5: No course data provided — skipping course features")
            course_features = pd.DataFrame()

        # --- Metadata ---
        metadata = {
            "as_of_date": as_of_date.strftime("%Y-%m-%d"),
            "n_rounds": len(rounds_enriched),
            "n_players": player_features["player_id"].nunique(),
            "n_events": rounds_enriched["event_id"].nunique(),
            "n_courses": len(course_features) if not course_features.empty else 0,
            "ewma_method": ewma_method,
            "ewma_half_life_rounds": self.cfg.EWMA_HALF_LIFE_ROUNDS,
            "ewma_half_life_days": self.cfg.EWMA_HALF_LIFE_DAYS,
            "date_range": (
                f"{rounds_enriched['date'].min().strftime('%Y-%m-%d')} → "
                f"{rounds_enriched['date'].max().strftime('%Y-%m-%d')}"
            ),
        }

        logger.info("=" * 60)
        logger.info("FEATURE PIPELINE: Complete")
        logger.info("  Players: %d", metadata["n_players"])
        logger.info("  Rounds:  %d", metadata["n_rounds"])
        logger.info("  Events:  %d", metadata["n_events"])
        logger.info("  Period:  %s", metadata["date_range"])
        logger.info("=" * 60)

        return ModelFeatures(
            player_features=player_features,
            course_features=course_features,
            field_strength=field_strength,
            rounds_enriched=rounds_enriched,
            metadata=metadata,
        )

    def run_for_tournament(
        self,
        rounds_df: pd.DataFrame,
        event_id: int,
        course_df: Optional[pd.DataFrame] = None,
    ) -> ModelFeatures:
        """
        Run feature pipeline for a specific upcoming tournament.
        
        Convenience method that automatically sets as_of_date to the
        start of the specified tournament.
        
        Parameters
        ----------
        rounds_df : pd.DataFrame
            Full historical rounds data.
        event_id : int
            The tournament to generate predictions for.
        course_df : pd.DataFrame, optional
            Course characteristics data.
            
        Returns
        -------
        ModelFeatures
            Features computed as of the tournament start date.
        """
        rounds = rounds_df.copy()
        rounds["date"] = pd.to_datetime(rounds["date"])

        # Find tournament start date
        event_rounds = rounds[rounds["event_id"] == event_id]
        if event_rounds.empty:
            raise ValueError(f"Event {event_id} not found in rounds data")

        tournament_start = event_rounds["date"].min()

        # Use day before tournament as as_of_date (don't include tournament data)
        as_of = tournament_start - pd.Timedelta(days=1)

        logger.info(
            "Running feature pipeline for event %d (start: %s, as_of: %s)",
            event_id,
            tournament_start.strftime("%Y-%m-%d"),
            as_of.strftime("%Y-%m-%d"),
        )

        return self.run(
            rounds_df=rounds_df,
            course_df=course_df,
            as_of_date=as_of,
        )
