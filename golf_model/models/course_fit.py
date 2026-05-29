# ==============================================================================
# golf_model/models/course_fit.py
# ==============================================================================
#
# COURSE-FIT MODEL (Phase 4)
# ---------------------------
# Models player-specific affinity for course characteristics via the 
# interaction term:
#
#     CourseFit_{i,c} = γ_c · δ_i
#
# where:
#     γ_c ∈ R^K  = standardized course characteristic vector 
#                   (length, rough, green speed, wind, etc.)
#     δ_i ∈ R^K  = player-specific course-fit coefficients
#                   (how much player i benefits/suffers from each feature)
#
# This is FEATURE-BASED, not course fixed effects. A player who thrives 
# on long, windy courses will be predicted to do well on ANY long, windy 
# course — even one they've never played.
#
# The δ_i coefficients are estimated via Bayesian regularized regression 
# with shrinkage toward zero (most players have small course-fit effects).
#
# Mathematical framework:
#     Y_{i,r,t} = μ_{i,t} + γ_{c(t)} · δ_i + ε_{i,r,t}
#     δ_i ~ N(0, Σ_δ)      # Prior: course-fit effects shrunk toward zero
#     Σ_δ = diag(τ²_δ)     # Independent features (simplification)
#
# Validation:
#     Leave-one-course-out cross-validation: for each course c, train on 
#     all other courses, predict performance at c. If course-fit term 
#     improves log-loss, it's adding signal, not noise.
#
# ==============================================================================

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CourseFitResult:
    """
    Container for a player's estimated course-fit coefficients.
    
    Attributes
    ----------
    player_id : int
        DataGolf player ID.
    delta : np.ndarray
        Shape (K,) — estimated δ_i coefficients for each course feature.
    delta_se : np.ndarray
        Shape (K,) — standard errors of δ_i estimates.
    n_rounds : int
        Number of rounds used to estimate this player's δ_i.
    feature_names : list of str
        Names of the K course features (matches delta ordering).
    shrinkage_factor : float
        How much this player was shrunk toward zero (0 = fully shrunk,
        1 = no shrinkage). Related to n_rounds and prior variance.
    """
    player_id: int
    delta: np.ndarray
    delta_se: np.ndarray
    n_rounds: int
    feature_names: List[str]
    shrinkage_factor: float


class CourseFitModel:
    """
    Estimates player × course-feature interaction coefficients (δ_i).
    
    Approach: Bayesian ridge regression per player. For each player i:
    
        residual_{i,r,t} = Y_{i,r,t} - μ̂_{i,t}  (remove ability estimate)
        residual_{i,r,t} = γ_{c(t)} · δ_i + noise
    
    We regress residuals on course features with a N(0, τ²_δ) prior on δ_i.
    
    For players with few rounds, the prior dominates (δ_i ≈ 0).
    For players with many rounds across diverse courses, we can detect 
    genuine course-fit patterns.
    
    Parameters
    ----------
    settings : Settings
        Project configuration.
    prior_variance : float, default 0.01
        Prior variance τ²_δ for each component of δ_i.
        Small value = strong shrinkage toward zero.
        This reflects our prior belief that most course-fit effects are small.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        prior_variance: float = 0.01,
    ):
        self.settings = settings or Settings()
        self.prior_variance = prior_variance
        self.feature_names = self.settings.COURSE_FEATURE_NAMES

        # Estimated parameters (populated by fit())
        self.player_fits: Dict[int, CourseFitResult] = {}
        self.population_variance: Optional[np.ndarray] = None
        self._is_fitted = False

        logger.info(
            "CourseFitModel initialized | K=%d features, prior_var=%.4f",
            len(self.feature_names), self.prior_variance,
        )

    def fit(
        self,
        residuals_df: pd.DataFrame,
        course_features_df: pd.DataFrame,
    ) -> "CourseFitModel":
        """
        Estimate δ_i for all players via Bayesian ridge regression.
        
        Parameters
        ----------
        residuals_df : pd.DataFrame
            Must contain columns:
                - player_id (int)
                - event_id (int)
                - course_id (int)
                - residual (float) — Y_{i,r,t} - μ̂_{i,t}
                
        course_features_df : pd.DataFrame
            Must contain columns:
                - course_id (int)
                - [all feature names from settings.COURSE_FEATURE_NAMES]
            Features should already be Z-score standardized.
            
        Returns
        -------
        self
            Fitted model instance.
        """
        logger.info("Fitting course-fit model...")

        # Merge residuals with course features
        merged = residuals_df.merge(
            course_features_df[["course_id"] + self.feature_names],
            on="course_id",
            how="inner",
        )

        if len(merged) == 0:
            logger.error("No data after merging residuals with course features")
            self._is_fitted = True
            return self

        n_players = merged["player_id"].nunique()
        logger.info(
            "Fitting δ_i for %d players | %d total observations",
            n_players, len(merged),
        )

        # Fit per player
        player_ids = merged["player_id"].unique()
        for pid in player_ids:
            player_data = merged[merged["player_id"] == pid]
            result = self._fit_single_player(pid, player_data)
            self.player_fits[pid] = result

        # Compute empirical population variance of δ estimates
        all_deltas = np.array([r.delta for r in self.player_fits.values()])
        if len(all_deltas) > 1:
            self.population_variance = np.var(all_deltas, axis=0)

        self._is_fitted = True
        logger.info(
            "Course-fit model fitted | %d players estimated", len(self.player_fits)
        )

        return self

    def predict_course_fit(
        self,
        player_id: int,
        course_features: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Predict course-fit adjustment for a player at a specific course.
        
        Parameters
        ----------
        player_id : int
            DataGolf player ID.
        course_features : np.ndarray
            Shape (K,) — standardized course feature vector γ_c.
            
        Returns
        -------
        tuple of (mean, std)
            mean : float — Expected course-fit adjustment (strokes/round).
            std : float — Uncertainty in the course-fit estimate.
            
        Notes
        -----
        For unknown players, returns (0.0, prior_std) — no course-fit effect 
        but with uncertainty reflecting the prior.
        """
        if player_id not in self.player_fits:
            # Unknown player → zero adjustment with prior uncertainty
            prior_std = np.sqrt(self.prior_variance * len(self.feature_names))
            return 0.0, prior_std

        result = self.player_fits[player_id]
        gamma = np.asarray(course_features, dtype=np.float64)

        # Point estimate: γ_c · δ_i
        mean = float(np.dot(gamma, result.delta))

        # Uncertainty: propagate δ_i standard errors through the dot product
        # Var(γ · δ) = γ^T · diag(se²) · γ  (assuming independent features)
        var = float(np.dot(gamma**2, result.delta_se**2))
        std = np.sqrt(var)

        return mean, std

    def predict_course_fit_batch(
        self,
        player_ids: np.ndarray,
        course_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction for multiple players at one course.
        
        Parameters
        ----------
        player_ids : array of int
            Shape (N,) — player IDs.
        course_features : np.ndarray
            Shape (K,) — single course feature vector.
            
        Returns
        -------
        tuple of (means, stds)
            means : np.ndarray shape (N,) — course-fit adjustments
            stds : np.ndarray shape (N,) — uncertainties
        """
        means = np.zeros(len(player_ids))
        stds = np.zeros(len(player_ids))

        for i, pid in enumerate(player_ids):
            means[i], stds[i] = self.predict_course_fit(pid, course_features)

        return means, stds

    def leave_one_course_out_cv(
        self,
        residuals_df: pd.DataFrame,
        course_features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Leave-one-course-out cross-validation.
        
        For each course c:
            1. Train on all rounds NOT at course c.
            2. Predict course-fit at course c.
            3. Measure whether predictions improve residual prediction.
            
        Returns
        -------
        pd.DataFrame
            CV results with columns: course_id, n_rounds, rmse_null, 
            rmse_model, improvement_pct.
        """
        logger.info("Running leave-one-course-out CV...")

        courses = residuals_df["course_id"].unique()
        results = []

        for course_id in courses:
            # Split: train on all other courses
            train_mask = residuals_df["course_id"] != course_id
            test_mask = residuals_df["course_id"] == course_id

            train_data = residuals_df[train_mask]
            test_data = residuals_df[test_mask]

            if len(test_data) < 10:
                continue

            # Fit on training courses
            cv_model = CourseFitModel(
                settings=self.settings,
                prior_variance=self.prior_variance,
            )
            cv_model.fit(train_data, course_features_df)

            # Predict on held-out course
            course_feats = course_features_df[
                course_features_df["course_id"] == course_id
            ][self.feature_names].values

            if len(course_feats) == 0:
                continue
            course_feats = course_feats[0]

            predictions = []
            actuals = []
            for _, row in test_data.iterrows():
                pred_mean, _ = cv_model.predict_course_fit(
                    int(row["player_id"]), course_feats
                )
                predictions.append(pred_mean)
                actuals.append(row["residual"])

            predictions = np.array(predictions)
            actuals = np.array(actuals)

            # RMSE: null model (predict 0) vs course-fit model
            rmse_null = np.sqrt(np.mean(actuals ** 2))
            rmse_model = np.sqrt(np.mean((actuals - predictions) ** 2))
            improvement = (rmse_null - rmse_model) / rmse_null * 100

            results.append({
                "course_id": course_id,
                "n_rounds": len(test_data),
                "rmse_null": round(rmse_null, 4),
                "rmse_model": round(rmse_model, 4),
                "improvement_pct": round(improvement, 2),
            })

        cv_df = pd.DataFrame(results)
        if len(cv_df) > 0:
            avg_improvement = cv_df["improvement_pct"].mean()
            logger.info(
                "LOCO-CV complete | %d courses | avg improvement: %.2f%%",
                len(cv_df), avg_improvement,
            )
        return cv_df

    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================

    def _fit_single_player(
        self,
        player_id: int,
        player_data: pd.DataFrame,
    ) -> CourseFitResult:
        """
        Bayesian ridge regression for a single player's δ_i.
        
        Closed-form posterior for Bayesian linear regression:
            Prior:      δ_i ~ N(0, τ² I)
            Likelihood: residual = Γ · δ_i + ε, ε ~ N(0, σ²)
            
            Posterior mean:  δ̂ = (Γ^T Γ + (σ²/τ²) I)^{-1} Γ^T y
            Posterior cov:   Σ_post = σ² (Γ^T Γ + (σ²/τ²) I)^{-1}
            
        where Γ is the matrix of course features for this player's rounds.
        """
        K = len(self.feature_names)
        n = len(player_data)

        # Design matrix Γ: (n_rounds × K)
        Gamma = player_data[self.feature_names].values.astype(np.float64)
        y = player_data["residual"].values.astype(np.float64)

        # Estimate observation noise σ² from residuals
        sigma2 = max(np.var(y), 0.5)  # Floor at 0.5 to avoid numerical issues

        # Regularization strength: σ² / τ²_δ
        # Higher = more shrinkage toward zero
        reg_strength = sigma2 / self.prior_variance

        # Posterior precision: Γ^T Γ + λI
        GtG = Gamma.T @ Gamma
        precision = GtG + reg_strength * np.eye(K)

        try:
            # Posterior mean: (Γ^T Γ + λI)^{-1} Γ^T y
            cov_post = np.linalg.inv(precision) * sigma2
            delta = np.linalg.solve(precision, Gamma.T @ y)
            delta_se = np.sqrt(np.diag(cov_post))
        except np.linalg.LinAlgError:
            # Singular matrix (likely all identical course features)
            logger.warning(
                "Singular precision matrix for player %d. Using prior.", player_id
            )
            delta = np.zeros(K)
            delta_se = np.full(K, np.sqrt(self.prior_variance))

        # Shrinkage factor: ratio of data precision to total precision
        # 1.0 = fully data-driven, 0.0 = fully prior-driven
        data_precision = np.diag(GtG).mean() if n > 0 else 0
        total_precision = data_precision + reg_strength
        shrinkage = data_precision / total_precision if total_precision > 0 else 0.0

        return CourseFitResult(
            player_id=player_id,
            delta=delta,
            delta_se=delta_se,
            n_rounds=n,
            feature_names=self.feature_names,
            shrinkage_factor=float(shrinkage),
        )

    def get_top_course_fit_players(
        self,
        course_features: np.ndarray,
        n_top: int = 20,
    ) -> pd.DataFrame:
        """
        Rank players by predicted course-fit advantage at a given course.
        
        Parameters
        ----------
        course_features : np.ndarray
            Shape (K,) — standardized features for the target course.
        n_top : int
            Number of top players to return.
            
        Returns
        -------
        pd.DataFrame
            Sorted by predicted course-fit advantage (descending).
        """
        if not self.player_fits:
            return pd.DataFrame()

        records = []
        for pid, result in self.player_fits.items():
            mean, std = self.predict_course_fit(pid, course_features)
            records.append({
                "player_id": pid,
                "course_fit_mean": round(mean, 4),
                "course_fit_std": round(std, 4),
                "n_rounds": result.n_rounds,
                "shrinkage": round(result.shrinkage_factor, 3),
            })

        df = pd.DataFrame(records)
        return df.nlargest(n_top, "course_fit_mean").reset_index(drop=True)
