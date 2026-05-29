# ==============================================================================
# golf_model/models/baseline.py
# ==============================================================================
#
# BASELINE MODEL (Phase 1)
# --------------------------
# Simple weighted-average SG regression. NO Bayesian machinery.
# This is the "beat this or stop" baseline.
#
# If the full Bayesian model can't beat this simple approach on the
# holdout set, then the added complexity provides no value.
#
# Model:
#   E[SG_{i,t}] = EWMA_SG_i(t)  (time-weighted average of past SG)
#
#   That's it. The prediction for player i in tournament t is simply
#   their exponentially-weighted average SG up to time t.
#
#   Win probability is derived by:
#     1. Assume SG_total ~ N(EWMA_SG_i, σ²_i) per round
#     2. Simulate 4 rounds
#     3. Sum to get tournament total
#     4. Player with lowest total (most negative SG = best) wins
#     5. Repeat N times → P(win) = fraction of sims won
#
# ==============================================================================

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class BaselineModel:
    """
    Weighted-average SG baseline model.
    
    Predicts tournament performance using EWMA SG estimates.
    Generates win probabilities via simple Gaussian Monte Carlo simulation.
    
    Parameters
    ----------
    settings : Settings
        Project configuration.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.cfg = settings or Settings()
        self.is_fitted = False
        self.player_estimates: Optional[pd.DataFrame] = None
        self.metadata: Dict = {}

    def fit(
        self,
        player_features: pd.DataFrame,
    ) -> "BaselineModel":
        """
        "Fit" the baseline model.
        
        For the baseline, fitting is trivial: just store the player 
        EWMA estimates. The heavy lifting was done in the feature pipeline.
        
        Parameters
        ----------
        player_features : pd.DataFrame
            Output of FeaturePipeline. Must contain:
            [player_id, ewma_sg_total, ewma_sg_total_var, effective_sample_size]
            
        Returns
        -------
        self
        """
        required_cols = ["player_id", "ewma_sg_total"]
        missing = [c for c in required_cols if c not in player_features.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.player_estimates = player_features.copy()
        self.is_fitted = True

        self.metadata = {
            "model_type": "baseline_ewma",
            "n_players": len(player_features),
            "fit_timestamp": datetime.now().isoformat(),
            "mean_ewma_sg": player_features["ewma_sg_total"].mean(),
        }

        logger.info(
            "Baseline model fitted: %d players | mean EWMA SG: %.3f",
            len(player_features),
            player_features["ewma_sg_total"].mean(),
        )

        return self

    def predict_tournament(
        self,
        field_player_ids: List[int],
        n_simulations: Optional[int] = None,
        n_rounds: int = 4,
    ) -> pd.DataFrame:
        """
        Predict win probabilities for a tournament field.
        
        Algorithm:
            For each simulation s = 1..N:
              For each player i in the field:
                For each round r = 1..4:
                  Draw SG_{i,r} ~ N(μ_i, σ²_i)
                TournamentTotal_i = Σ SG_{i,r}
              Winner_s = argmax(TournamentTotal)  [highest SG wins]
            P(win|i) = #{s : Winner_s = i} / N
        
        Parameters
        ----------
        field_player_ids : list of int
            Player IDs in the tournament field.
        n_simulations : int, optional
            Number of MC simulations. Default: Settings.N_SIMULATIONS.
        n_rounds : int, default 4
            Number of rounds (4 for standard, 3 for some events).
            
        Returns
        -------
        pd.DataFrame
            Columns: [player_id, win_prob, mean_total_sg, std_total_sg]
            Sorted by win_prob descending.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_sims = n_simulations or self.cfg.N_SIMULATIONS
        rng = np.random.default_rng(self.cfg.RANDOM_SEED)

        # Get estimates for field players
        field_df = self.player_estimates[
            self.player_estimates["player_id"].isin(field_player_ids)
        ].copy()

        # Handle players not in our estimates (use population prior)
        known_ids = set(field_df["player_id"])
        unknown_ids = set(field_player_ids) - known_ids

        if unknown_ids:
            logger.warning(
                "%d players in field not in estimates — using population mean",
                len(unknown_ids),
            )
            pop_mean = self.player_estimates["ewma_sg_total"].mean()
            pop_var = self.player_estimates["ewma_sg_total"].var()

            unknown_rows = pd.DataFrame({
                "player_id": list(unknown_ids),
                "ewma_sg_total": pop_mean,
                "ewma_sg_total_var": pop_var,
                "effective_sample_size": 0.0,
            })
            field_df = pd.concat([field_df, unknown_rows], ignore_index=True)

        n_players = len(field_df)
        player_ids = field_df["player_id"].values
        means = field_df["ewma_sg_total"].values
        variances = field_df["ewma_sg_total_var"].fillna(
            field_df["ewma_sg_total_var"].median()
        ).values

        # Ensure variances are positive
        variances = np.maximum(variances, 0.01)
        stds = np.sqrt(variances)

        logger.info(
            "Simulating %d tournaments | %d players | %d rounds",
            n_sims, n_players, n_rounds,
        )

        # --- Monte Carlo Simulation ---
        # Shape: (n_simulations, n_players, n_rounds)
        round_scores = rng.normal(
            loc=means[np.newaxis, :, np.newaxis],
            scale=stds[np.newaxis, :, np.newaxis],
            size=(n_sims, n_players, n_rounds),
        )

        # Tournament totals: sum across rounds
        # Shape: (n_simulations, n_players)
        tournament_totals = round_scores.sum(axis=2)

        # Winner = player with HIGHEST total SG (most strokes gained)
        # Shape: (n_simulations,)
        winners = np.argmax(tournament_totals, axis=1)

        # Win counts
        win_counts = np.bincount(winners, minlength=n_players)
        win_probs = win_counts / n_sims

        # Summary stats
        results = pd.DataFrame({
            "player_id": player_ids,
            "win_prob": win_probs,
            "mean_total_sg": tournament_totals.mean(axis=0),
            "std_total_sg": tournament_totals.std(axis=0),
            "ewma_sg_per_round": means,
        })

        results = results.sort_values("win_prob", ascending=False).reset_index(drop=True)

        logger.info(
            "Simulation complete | Top 5: %s",
            results.head(5)[["player_id", "win_prob"]].to_dict("records"),
        )

        return results


class ModelRegistry:
    """
    Save, load, and version trained models with metadata.
    
    Every trained model is saved as a pickle with accompanying
    JSON metadata. This ensures reproducibility and easy comparison.
    
    Parameters
    ----------
    settings : Settings
        Provides MODELS_DIR path.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.cfg = settings or Settings()
        self.models_dir = self.cfg.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: object,
        name: str,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Save a trained model to disk with metadata.
        
        Parameters
        ----------
        model : object
            Any trained model object (BaselineModel, BayesianModel, etc.).
        name : str
            Model name (e.g., "baseline_v1", "bayesian_2024").
        metadata : dict, optional
            Additional metadata to save.
            
        Returns
        -------
        Path
            Path to saved model file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}"

        # Save model
        model_path = self.models_dir / f"{filename}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metadata
        meta = {
            "name": name,
            "timestamp": timestamp,
            "model_type": type(model).__name__,
            **(metadata or {}),
            **(getattr(model, "metadata", {})),
        }
        meta_path = self.models_dir / f"{filename}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info("Model saved: %s → %s", name, model_path)
        return model_path

    def load_model(self, name: str) -> object:
        """
        Load the most recent version of a named model.
        
        Parameters
        ----------
        name : str
            Model name prefix.
            
        Returns
        -------
        object
            The loaded model.
        """
        pattern = f"{name}_*.pkl"
        candidates = sorted(self.models_dir.glob(pattern))

        if not candidates:
            raise FileNotFoundError(
                f"No model found matching '{name}' in {self.models_dir}"
            )

        latest = candidates[-1]  # Last alphabetically = most recent timestamp
        with open(latest, "rb") as f:
            model = pickle.load(f)

        logger.info("Model loaded: %s", latest)
        return model

    def list_models(self) -> List[Dict]:
        """List all saved models with their metadata."""
        models = []
        for meta_file in sorted(self.models_dir.glob("*_meta.json")):
            with open(meta_file, "r") as f:
                models.append(json.load(f))
        return models
