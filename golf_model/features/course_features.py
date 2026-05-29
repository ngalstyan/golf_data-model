# ==============================================================================
# golf_model/features/course_features.py
# ==============================================================================
#
# COURSE CHARACTERISTIC FEATURES (γ_c VECTOR)
# ----------------------------------------------
# Builds the course feature vector used in the course-fit interaction model.
#
# Mathematical Role:
#   In the course-fit extension (Phase 4), each player i has a 
#   K-dimensional "course preference" vector δ_i, and each course c
#   has a K-dimensional "characteristic" vector γ_c.
#
#   The course-fit adjustment is:
#     CourseFit_{i,c} = γ_c · δ_i  (dot product)
#
#   This module builds γ_c from raw course data.
#
# Feature Engineering:
#   Raw course attributes → Z-score standardized features.
#   Z-scoring ensures all features are on the same scale,
#   so the dot product weights them equally a priori.
#
#   Features:
#     1. length_yards       — Total course distance
#     2. rough_height_in    — Primary rough height (inches)
#     3. green_speed_stimp  — Stimpmeter green speed
#     4. wind_exposure      — Wind exposure (0–1 index)
#     5. elevation_ft       — Elevation above sea level
#     6. fairway_width_avg  — Average fairway width (yards)
#     7. green_size_sqft    — Average green size (sq ft)
#     8. water_hazard_pct   — % holes with water hazards
#
# ==============================================================================

from typing import List, Optional

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class CourseFeatureProcessor:
    """
    Process and standardize course characteristics for the γ_c vector.
    
    Parameters
    ----------
    settings : Settings
        Provides course feature column names.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.cfg = settings or Settings()
        self.feature_names = self.cfg.COURSE_FEATURE_NAMES

    def standardize_features(
        self,
        course_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Z-score standardize course features across all courses.
        
        For each feature f:
            z_f = (x_f - μ_f) / σ_f
            
        where μ_f and σ_f are the mean and std across ALL courses
        in the training set.
        
        Parameters
        ----------
        course_df : pd.DataFrame
            Course characteristics with columns matching feature_names.
            Must have 'course_id' column.
            
        Returns
        -------
        pd.DataFrame
            Same structure with standardized features (suffixed '_z').
            Also includes the raw features and standardization parameters.
        """
        df = course_df.copy()

        if "course_id" not in df.columns:
            raise ValueError("course_df must contain 'course_id' column")

        # Track standardization parameters (for applying to new courses)
        self._standardization_params = {}

        available_features = [f for f in self.feature_names if f in df.columns]
        missing_features = [f for f in self.feature_names if f not in df.columns]

        if missing_features:
            logger.warning(
                "Missing course features (will be set to 0): %s", missing_features
            )

        for feat in available_features:
            col = df[feat].astype(float)
            mu = col.mean()
            sigma = col.std()

            if sigma == 0 or np.isnan(sigma):
                logger.warning(
                    "Feature '%s' has zero variance — setting z-scores to 0", feat
                )
                df[f"{feat}_z"] = 0.0
                sigma = 1.0  # Prevent division by zero
            else:
                df[f"{feat}_z"] = (col - mu) / sigma

            self._standardization_params[feat] = {"mean": mu, "std": sigma}

        # Set missing features to 0 (neutral in z-score space)
        for feat in missing_features:
            df[f"{feat}_z"] = 0.0

        n_complete = df[[f"{f}_z" for f in self.feature_names]].notna().all(axis=1).sum()
        logger.info(
            "Course features standardized: %d courses, %d/%d features available, "
            "%d courses with complete data",
            len(df), len(available_features), len(self.feature_names), n_complete,
        )

        return df

    def get_gamma_matrix(
        self,
        course_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract the γ_c matrix: course_id × K standardized features.
        
        This is the matrix directly used in the model's course-fit term.
        
        Parameters
        ----------
        course_df : pd.DataFrame
            Must already be standardized (call standardize_features first).
            
        Returns
        -------
        pd.DataFrame
            Index: course_id, Columns: K standardized features.
        """
        z_cols = [f"{f}_z" for f in self.feature_names]
        available_z = [c for c in z_cols if c in course_df.columns]

        if not available_z:
            raise ValueError(
                "No standardized features found. "
                "Call standardize_features() first."
            )

        gamma = course_df.set_index("course_id")[available_z].copy()
        gamma.columns = [c.replace("_z", "") for c in gamma.columns]

        logger.info(
            "γ_c matrix: %d courses × %d features", len(gamma), len(gamma.columns)
        )
        return gamma

    def impute_missing_features(
        self,
        course_df: pd.DataFrame,
        method: str = "median",
    ) -> pd.DataFrame:
        """
        Impute missing course feature values.
        
        Parameters
        ----------
        course_df : pd.DataFrame
        method : str, default "median"
            Imputation method: "median", "mean", or "zero".
            
        Returns
        -------
        pd.DataFrame
            Course data with missing features imputed.
        """
        df = course_df.copy()

        for feat in self.feature_names:
            if feat not in df.columns:
                continue

            n_missing = df[feat].isna().sum()
            if n_missing == 0:
                continue

            if method == "median":
                fill_val = df[feat].median()
            elif method == "mean":
                fill_val = df[feat].mean()
            elif method == "zero":
                fill_val = 0.0
            else:
                raise ValueError(f"Unknown imputation method: {method}")

            df[feat] = df[feat].fillna(fill_val)
            logger.info(
                "Imputed %d missing values for '%s' with %s=%.2f",
                n_missing, feat, method, fill_val,
            )

        return df
