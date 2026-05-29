# ==============================================================================
# golf_model/models/bayesian_core.py
# ==============================================================================
#
# HIERARCHICAL BAYESIAN STROKES-GAINED MODEL (Phase 3)
# -------------------------------------------------------
# The core probabilistic model of the entire system.
#
# Full generative model:
#
#   POPULATION LEVEL:
#     μ_pop        ~ Normal(0, σ²_pop_prior)
#     σ_pop        ~ HalfNormal(σ_pop_prior)
#
#   PLAYER LEVEL (for each player i = 1..N):
#     μ_i          ~ Normal(μ_pop, σ²_pop)        ← BAYESIAN SHRINKAGE
#     τ_i          ~ HalfNormal(τ_prior)
#
#   OBSERVATION LEVEL (for each round r of player i in event t):
#     Y_{i,r,t}    ~ StudentT(ν, μ_i, τ²_i)      ← HEAVY-TAILED NOISE
#
#   NOISE PARAMETERS:
#     ν            ~ Gamma(α_ν, β_ν) + 2
#
# The key insight of Bayesian shrinkage:
#   Players with few rounds → estimate shrunk toward population mean
#   Players with many rounds → estimate close to their sample mean
#   This automatically handles the "small sample" problem in golf.
#
# Implementation uses PyMC v5 for MCMC inference (NUTS sampler).
# Alternative: ADVI for fast approximate inference during development.
#
# Usage:
#   from models.bayesian_core import HierarchicalBayesianModel
#   model = HierarchicalBayesianModel(Settings())
#   model.fit(player_features, rounds_enriched)
#   posteriors = model.get_posteriors()
#
# ==============================================================================

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import Settings
from models.priors import PriorConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class HierarchicalBayesianModel:
    """
    Hierarchical Bayesian model for player ability estimation.
    
    Uses PyMC for Bayesian inference with NUTS (MCMC) or ADVI 
    (variational inference) backends.
    
    Parameters
    ----------
    settings : Settings
        Project configuration (MCMC params, convergence thresholds).
    prior_config : PriorConfig, optional
        Prior specifications. Uses defaults if not provided.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        prior_config: Optional[PriorConfig] = None,
    ):
        self.cfg = settings or Settings()
        self.priors = prior_config or PriorConfig()
        self.trace = None               # PyMC InferenceData after fitting
        self.model = None                # PyMC Model object
        self.player_id_map: Dict = {}    # Maps player_id → model index
        self.is_fitted = False
        self.metadata: Dict = {}

    def build_model(
        self,
        rounds_df: pd.DataFrame,
        player_features: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Construct the PyMC model graph.
        
        This builds the probabilistic model but does NOT run inference.
        Call fit() after build_model() to run MCMC/ADVI.
        
        Parameters
        ----------
        rounds_df : pd.DataFrame
            Round-level data. Must contain:
            [player_id, sg_total, event_id].
            Each row = one round of one player.
            
        player_features : pd.DataFrame, optional
            Player EWMA features for informative initialization.
            Not used in the model itself — only for init values.
        """
        try:
            import pymc as pm
        except ImportError:
            raise ImportError(
                "PyMC is required for the Bayesian model. "
                "Install with: pip install pymc>=5.10"
            )

        df = rounds_df.copy()

        # Map player IDs to contiguous indices (required by PyMC)
        unique_players = df["player_id"].unique()
        self.player_id_map = {pid: idx for idx, pid in enumerate(unique_players)}
        self.reverse_id_map = {idx: pid for pid, idx in self.player_id_map.items()}
        n_players = len(unique_players)

        # Create index column
        df["player_idx"] = df["player_id"].map(self.player_id_map)

        # Extract observed data
        observed_sg = df["sg_total"].values
        player_indices = df["player_idx"].values.astype(int)

        logger.info(
            "Building Bayesian model: %d players, %d rounds",
            n_players, len(df),
        )

        # ---- BUILD MODEL ----
        with pm.Model() as self.model:

            # --- Population-level hyperpriors ---
            # μ_pop: the "average" PGA Tour player ability
            mu_pop = pm.Normal(
                "mu_pop",
                mu=self.priors.pop_mean_mu,
                sigma=self.priors.pop_mean_sigma,
            )

            # σ_pop: how much true abilities vary across players
            sigma_pop = pm.HalfNormal(
                "sigma_pop",
                sigma=self.priors.pop_std_sigma,
            )

            # --- Player-level parameters ---
            # μ_i ~ Normal(μ_pop, σ_pop)
            # This is WHERE Bayesian shrinkage happens:
            #   - Players with few rounds: μ_i pulled toward μ_pop
            #   - Players with many rounds: μ_i stays near their sample mean
            mu_player = pm.Normal(
                "mu_player",
                mu=mu_pop,
                sigma=sigma_pop,
                shape=n_players,
            )

            # τ_i: round-to-round noise per player
            tau_player = pm.HalfNormal(
                "tau_player",
                sigma=self.priors.round_noise_sigma,
                shape=n_players,
            )

            # --- Observation noise parameters ---
            # ν: degrees of freedom for Student-t (controls tail heaviness)
            # ν = 2 + Gamma(α, β) ensures ν > 2 (finite variance)
            nu_offset = pm.Gamma(
                "nu_offset",
                alpha=self.priors.nu_alpha,
                beta=self.priors.nu_beta,
            )
            nu = pm.Deterministic("nu", nu_offset + 2.0)

            # --- Observation model ---
            # Y_{i,r,t} ~ StudentT(ν, μ_i, τ_i)
            # Student-t handles outlier rounds (blowups, withdrawal rounds)
            # better than Gaussian.
            y_obs = pm.StudentT(
                "y_obs",
                nu=nu,
                mu=mu_player[player_indices],
                sigma=tau_player[player_indices],
                observed=observed_sg,
            )

        logger.info("PyMC model built successfully | %d parameters", n_players * 2 + 3)

    def fit(
        self,
        rounds_df: pd.DataFrame,
        method: str = "mcmc",
        player_features: Optional[pd.DataFrame] = None,
    ) -> "HierarchicalBayesianModel":
        """
        Run Bayesian inference.
        
        Parameters
        ----------
        rounds_df : pd.DataFrame
            Round-level data for training.
        method : str, default "mcmc"
            Inference method:
            - "mcmc": Full NUTS sampling (accurate but slow)
            - "advi": Variational inference (fast but approximate)
        player_features : pd.DataFrame, optional
            EWMA features for initialization hints.
            
        Returns
        -------
        self
        """
        import pymc as pm
        import arviz as az

        # Build model if not already built
        if self.model is None:
            self.build_model(rounds_df, player_features)

        with self.model:
            if method == "mcmc":
                logger.info(
                    "Running MCMC: %d draws × %d chains, %d warmup | target_accept=%.2f",
                    self.cfg.MCMC_DRAWS, self.cfg.MCMC_CHAINS,
                    self.cfg.MCMC_TUNE, self.cfg.MCMC_TARGET_ACCEPT,
                )
                self.trace = pm.sample(
                    draws=self.cfg.MCMC_DRAWS,
                    tune=self.cfg.MCMC_TUNE,
                    chains=self.cfg.MCMC_CHAINS,
                    target_accept=self.cfg.MCMC_TARGET_ACCEPT,
                    random_seed=self.cfg.RANDOM_SEED,
                    return_inferencedata=True,
                    progressbar=True,
                )

            elif method == "advi":
                logger.info(
                    "Running ADVI: max_iter=%d", self.cfg.ADVI_MAX_ITER
                )
                approx = pm.fit(
                    n=self.cfg.ADVI_MAX_ITER,
                    method="advi",
                    random_seed=self.cfg.RANDOM_SEED,
                )
                self.trace = approx.sample(
                    draws=self.cfg.MCMC_DRAWS,
                    random_seed=self.cfg.RANDOM_SEED,
                )

            else:
                raise ValueError(f"Unknown method: {method}. Use 'mcmc' or 'advi'.")

        self.is_fitted = True

        # Run convergence diagnostics
        diagnostics = self.check_convergence()

        self.metadata = {
            "model_type": "hierarchical_bayesian",
            "inference_method": method,
            "n_players": len(self.player_id_map),
            "n_rounds": len(rounds_df),
            "draws": self.cfg.MCMC_DRAWS,
            "chains": self.cfg.MCMC_CHAINS,
            "diagnostics": diagnostics,
        }

        logger.info("Bayesian model fitting complete")
        return self

    def check_convergence(self) -> Dict:
        """
        Run standard convergence diagnostics.
        
        Checks:
            1. R-hat (Gelman-Rubin): Should be < 1.01 for all parameters.
               R-hat > 1.01 indicates chains haven't converged.
               
            2. ESS (Effective Sample Size): Should be > 400.
               Low ESS means autocorrelated samples → unreliable estimates.
               
            3. Divergences: Should be 0.
               Divergences indicate the sampler struggled with geometry.
        
        Returns
        -------
        dict
            Diagnostic summary with pass/fail per check.
        """
        import arviz as az

        if self.trace is None:
            return {"error": "No trace available"}

        diagnostics = {}

        # R-hat
        rhat = az.rhat(self.trace)
        rhat_max = max(
            float(rhat[var].max()) for var in rhat.data_vars
            if "mu_player" in var or "sigma" in var or "nu" in var
        )
        diagnostics["rhat_max"] = rhat_max
        diagnostics["rhat_pass"] = rhat_max < self.cfg.CONVERGENCE_RHAT_THRESHOLD

        # ESS
        ess = az.ess(self.trace)
        ess_min = min(
            float(ess[var].min()) for var in ess.data_vars
            if "mu_player" in var or "sigma" in var or "nu" in var
        )
        diagnostics["ess_min"] = ess_min
        diagnostics["ess_pass"] = ess_min > self.cfg.CONVERGENCE_ESS_THRESHOLD

        # Divergences
        if hasattr(self.trace, "sample_stats"):
            try:
                n_divergent = int(self.trace.sample_stats["diverging"].sum())
            except (KeyError, AttributeError):
                n_divergent = 0
        else:
            n_divergent = 0
        diagnostics["n_divergences"] = n_divergent
        diagnostics["divergence_pass"] = n_divergent == 0

        # Overall
        diagnostics["all_pass"] = all([
            diagnostics["rhat_pass"],
            diagnostics["ess_pass"],
            diagnostics["divergence_pass"],
        ])

        # Log results
        status = "✓ PASSED" if diagnostics["all_pass"] else "✗ FAILED"
        logger.info(
            "Convergence diagnostics: %s | R-hat max=%.4f | ESS min=%d | Divergences=%d",
            status, rhat_max, ess_min, n_divergent,
        )

        return diagnostics

    def get_posteriors(self) -> pd.DataFrame:
        """
        Extract posterior estimates for all players.
        
        Returns
        -------
        pd.DataFrame
            Columns: [player_id, mu_mean, mu_std, mu_q05, mu_q25, mu_q50,
                       mu_q75, mu_q95, tau_mean, tau_std]
                       
            mu = posterior distribution of true ability (SG/round)
            tau = posterior distribution of round-to-round variability
        """
        import arviz as az

        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Extract posterior samples for mu_player
        mu_samples = self.trace.posterior["mu_player"].values  # (chains, draws, players)
        # Reshape to (total_samples, players)
        mu_flat = mu_samples.reshape(-1, mu_samples.shape[-1])

        tau_samples = self.trace.posterior["tau_player"].values
        tau_flat = tau_samples.reshape(-1, tau_samples.shape[-1])

        results = []
        for idx, player_id in self.reverse_id_map.items():
            mu_post = mu_flat[:, idx]
            tau_post = tau_flat[:, idx]

            results.append({
                "player_id": player_id,
                "mu_mean": np.mean(mu_post),
                "mu_std": np.std(mu_post),
                "mu_q05": np.percentile(mu_post, 5),
                "mu_q25": np.percentile(mu_post, 25),
                "mu_q50": np.percentile(mu_post, 50),
                "mu_q75": np.percentile(mu_post, 75),
                "mu_q95": np.percentile(mu_post, 95),
                "tau_mean": np.mean(tau_post),
                "tau_std": np.std(tau_post),
            })

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("mu_mean", ascending=False).reset_index(drop=True)

        logger.info(
            "Posteriors extracted for %d players | "
            "top ability: %.3f | bottom: %.3f",
            len(result_df),
            result_df["mu_mean"].max(),
            result_df["mu_mean"].min(),
        )

        return result_df

    def get_population_params(self) -> Dict:
        """Extract posterior estimates for population-level parameters."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        post = self.trace.posterior

        return {
            "mu_pop_mean": float(post["mu_pop"].mean()),
            "mu_pop_std": float(post["mu_pop"].std()),
            "sigma_pop_mean": float(post["sigma_pop"].mean()),
            "sigma_pop_std": float(post["sigma_pop"].std()),
            "nu_mean": float(post["nu"].mean()),
            "nu_std": float(post["nu"].std()),
        }
