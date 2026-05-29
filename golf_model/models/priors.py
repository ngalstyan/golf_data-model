# ==============================================================================
# golf_model/models/priors.py
# ==============================================================================
#
# PRIOR SPECIFICATIONS
# ----------------------
# Defines all prior distributions for the Bayesian model.
# Centralized here so they're documented, adjustable, and auditable.
#
# Mathematical framework:
#   The hierarchical model has these random variables:
#
#   Population level:
#     μ_pop    ~ Normal(0, σ²_pop)          Population mean SG (≈0 by construction)
#     σ²_pop   ~ InvGamma(α_pop, β_pop)     Population variance of true abilities
#
#   Player level:
#     μ_i      ~ Normal(μ_pop, σ²_pop)      Player i's true ability (SHRINKAGE!)
#     τ²_i     ~ InvGamma(α_τ, β_τ)         Player i's round-to-round variance
#
#   Sub-components (k ∈ {OTT, APP, ARG, PUTT}):
#     μ_i^(k)  ~ Normal(μ_pop^(k), σ²_pop^(k))  Component-specific abilities
#
#   Observation noise:
#     ε_{i,r,t} ~ StudentT(ν, 0, τ²_i)     Heavy-tailed noise per round
#     ν          ~ Gamma(α_ν, β_ν) + 2      Degrees of freedom (ν > 2)
#
#   Course-fit (Phase 4):
#     δ_i^(k)  ~ Normal(0, σ²_δ)            Player's course-fit coefficients
#     σ²_δ     ~ HalfNormal(σ_δ_prior)      Course-fit variance
#
# Why these priors?
#   - Normal priors for abilities: maximally uninformative given SG ∈ [-3, +3]
#   - InvGamma for variances: conjugate, controls shrinkage strength
#   - StudentT for noise: handles blowup rounds (80+ scores) gracefully
#   - Hierarchical structure: automatic Bayesian shrinkage (Efron & Morris)
#
# ==============================================================================

from dataclasses import dataclass, field
from typing import Dict, Optional

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PriorConfig:
    """
    Complete prior specification for the hierarchical model.
    
    All hyperparameters are set to weakly informative defaults
    based on empirical PGA Tour SG distributions.
    
    The key insight: PGA Tour SG_total has population std ≈ 1.5 strokes,
    with individual round-to-round std ≈ 2.5–3.5 strokes.
    """

    # ==================================================================
    # POPULATION-LEVEL PRIORS
    # ==================================================================

    # μ_pop ~ Normal(pop_mean_mu, pop_mean_sigma)
    # Population mean ability. Centered at 0 (SG is zero-sum by definition).
    pop_mean_mu: float = 0.0
    pop_mean_sigma: float = 0.5       # Weakly informative

    # σ_pop ~ HalfNormal(pop_std_sigma)
    # Population standard deviation of true abilities.
    # PGA Tour: ~1.5 SG spread between best and worst regulars.
    pop_std_sigma: float = 2.0        # Prior on the scale of population spread

    # ==================================================================
    # PLAYER-LEVEL PRIORS
    # ==================================================================

    # τ_i (round-to-round std for player i)
    # τ_i ~ HalfNormal(round_noise_sigma)
    # Typical PGA Tour round-to-round std: 2.5–3.5 strokes
    round_noise_sigma: float = 4.0    # Weakly informative upper bound

    # ==================================================================
    # OBSERVATION MODEL
    # ==================================================================

    # ν (Student-t degrees of freedom)
    # ν ~ Gamma(nu_alpha, nu_beta) + 2
    # The +2 ensures ν > 2 (finite variance).
    # Prior mode ≈ (α-1)/β + 2 ≈ 5.0 (moderate heavy tails)
    nu_alpha: float = 3.0
    nu_beta: float = 1.0

    # ρ (within-tournament round correlation)
    # ρ ~ Beta(rho_a, rho_b)
    # Expected value: a/(a+b) = 2/12 ≈ 0.17
    rho_a: float = 2.0
    rho_b: float = 10.0

    # ==================================================================
    # COURSE-FIT PRIORS (Phase 4)
    # ==================================================================

    # δ_i^(k) ~ Normal(0, σ_delta)
    # Course preference coefficients per player per feature
    delta_sigma: float = 0.3         # Prior std for course-fit coefficients

    # σ_delta ~ HalfNormal(delta_hyper_sigma)
    # Hyperprior on course-fit coefficient scale
    delta_hyper_sigma: float = 0.5

    # ==================================================================
    # SUB-COMPONENT PRIORS (k ∈ {OTT, APP, ARG, PUTT})
    # ==================================================================

    # Each component's population std (how much players vary in this skill)
    # These are DIFFERENT because skills vary differently:
    #   - Putting has the smallest spread (~0.3 SG)
    #   - Approach has the largest spread (~0.5 SG)
    component_pop_sigma: Dict[str, float] = field(default_factory=lambda: {
        "sg_ott":  0.40,   # Off-the-Tee: moderate spread
        "sg_app":  0.50,   # Approach: largest spread (most skill differentiation)
        "sg_arg":  0.30,   # Around-the-Green: smaller spread
        "sg_putt": 0.35,   # Putting: moderate spread, high noise
    })

    # Component round-to-round noise (how variable each component is per round)
    component_noise_sigma: Dict[str, float] = field(default_factory=lambda: {
        "sg_ott":  1.0,
        "sg_app":  1.2,
        "sg_arg":  0.8,
        "sg_putt": 1.0,
    })

    def validate(self) -> list:
        """Check that all priors are valid."""
        issues = []
        if self.pop_mean_sigma <= 0:
            issues.append("pop_mean_sigma must be positive")
        if self.pop_std_sigma <= 0:
            issues.append("pop_std_sigma must be positive")
        if self.nu_alpha <= 0 or self.nu_beta <= 0:
            issues.append("nu_alpha and nu_beta must be positive")
        if self.rho_a <= 0 or self.rho_b <= 0:
            issues.append("rho_a and rho_b must be positive")
        return issues
