# ==============================================================================
# golf_model/betting/kelly.py
# ==============================================================================
#
# KELLY CRITERION BET SIZING
# ----------------------------
# The Kelly Criterion determines the mathematically optimal fraction of 
# bankroll to wager on each bet, maximizing long-run expected log-utility 
# (geometric growth rate).
#
# Full Kelly: f* = (b·p - q) / b
#   where b = decimal_odds - 1 (net payout per unit wagered)
#         p = true probability of winning
#         q = 1 - p = probability of losing
#
# Why fractional Kelly?
#   Full Kelly assumes PERFECT probability estimates. With estimation error:
#   - Overbetting is FAR more costly than underbetting.
#   - Full Kelly gives ~50% peak-to-trough drawdowns.
#   - Fractional Kelly (25-33%) sacrifices growth for survivability.
#   
#   Rule of thumb: use Kelly fraction ≈ 1 / (2 × estimation_uncertainty_ratio).
#   With ~50% estimation uncertainty → 25% Kelly is appropriate.
#
# Correlated bets (same tournament):
#   Outright winner bets in the same tournament are mutually exclusive —
#   at most one can win. This means the true portfolio risk is lower than
#   the sum of individual bet risks. We handle this by:
#   1. Computing individual Kelly fractions.
#   2. Capping total tournament exposure.
#   3. Scaling down proportionally if exposure exceeds cap.
#
# Reference:
#   Kelly, J.L. (1956). "A New Interpretation of Information Rate."
#   Thorp, E.O. (2006). "The Kelly Criterion in Blackjack, Sports Betting."
#
# ==============================================================================

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import Settings
from betting.edge_detection import BettingOpportunity
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BetRecommendation:
    """
    A sized bet recommendation ready for execution.
    
    Attributes
    ----------
    player_id : int
    player_name : str
    p_model : float
        Model probability.
    decimal_odds : float
        Available decimal odds.
    book : str
        Sportsbook.
    full_kelly : float
        Full Kelly fraction (f*).
    fractional_kelly : float
        Adjusted Kelly fraction (f* × kelly_fraction setting).
    stake_dollars : float
        Dollar amount to wager.
    expected_value : float
        E[profit] = stake × (odds × p_model - 1).
    edge_pct : float
        Edge as percentage.
    """
    player_id: int
    player_name: str = ""
    p_model: float = 0.0
    decimal_odds: float = 0.0
    book: str = ""
    full_kelly: float = 0.0
    fractional_kelly: float = 0.0
    stake_dollars: float = 0.0
    expected_value: float = 0.0
    edge_pct: float = 0.0


def kelly_fraction(
    p_win: float,
    decimal_odds: float,
) -> float:
    """
    Compute the full Kelly fraction.
    
    f* = (b·p - q) / b
    
    where:
        b = decimal_odds - 1 (net payout per unit)
        p = probability of winning
        q = 1 - p = probability of losing
    
    Parameters
    ----------
    p_win : float
        True probability of winning (0 < p < 1).
    decimal_odds : float
        Decimal odds offered (> 1.0).
        
    Returns
    -------
    float
        Kelly fraction. Negative = don't bet (no edge).
    """
    if p_win <= 0 or p_win >= 1 or decimal_odds <= 1.0:
        return 0.0

    b = decimal_odds - 1.0
    q = 1.0 - p_win
    f_star = (b * p_win - q) / b

    return f_star


def kelly_fraction_with_uncertainty(
    p_win: float,
    p_std: float,
    decimal_odds: float,
    n_samples: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Kelly fraction accounting for probability estimation uncertainty.
    
    Instead of using a point estimate for p_win, sample from
    p ~ Beta distribution and compute E[f*(p)] across samples.
    
    This naturally reduces bet sizes when uncertainty is high,
    because the Kelly function is concave in p for p > 0.
    
    Parameters
    ----------
    p_win : float
        Point estimate of P(win).
    p_std : float
        Standard deviation of the probability estimate.
    decimal_odds : float
    n_samples : int
    rng : np.random.Generator
        
    Returns
    -------
    tuple of (mean_kelly, kelly_std)
        Mean and std of Kelly fraction across probability samples.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Fit Beta distribution to match (p_win, p_std)
    # Beta(α, β): mean = α/(α+β), var = αβ/((α+β)²(α+β+1))
    if p_std <= 0 or p_win <= 0 or p_win >= 1:
        return kelly_fraction(p_win, decimal_odds), 0.0

    # Method of moments for Beta parameters
    mean = p_win
    var = min(p_std ** 2, mean * (1 - mean) * 0.99)  # Cap variance

    if var <= 0:
        return kelly_fraction(p_win, decimal_odds), 0.0

    alpha = mean * (mean * (1 - mean) / var - 1)
    beta = (1 - mean) * (mean * (1 - mean) / var - 1)

    if alpha <= 0 or beta <= 0:
        return kelly_fraction(p_win, decimal_odds), 0.0

    # Sample probabilities and compute Kelly for each
    p_samples = rng.beta(alpha, beta, size=n_samples)
    kelly_samples = np.array([kelly_fraction(p, decimal_odds) for p in p_samples])

    # Only consider non-negative Kelly (would bet)
    kelly_samples = np.maximum(kelly_samples, 0.0)

    return float(np.mean(kelly_samples)), float(np.std(kelly_samples))


class KellyCalculator:
    """
    Full Kelly bet sizing with portfolio-level constraints.
    
    Parameters
    ----------
    settings : Settings
        Project configuration.
    bankroll : float, optional
        Current bankroll. If None, uses settings.INITIAL_BANKROLL.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        bankroll: Optional[float] = None,
    ):
        self.settings = settings or Settings()
        self.bankroll = bankroll or self.settings.INITIAL_BANKROLL
        self.kelly_fraction_multiplier = self.settings.KELLY_FRACTION
        self.max_single_bet_pct = self.settings.MAX_SINGLE_BET_PCT
        self.max_tournament_exposure = self.settings.MAX_TOURNAMENT_EXPOSURE_PCT

        logger.info(
            "KellyCalculator | bankroll=$%.0f | kelly_frac=%.0f%% | "
            "max_single=%.1f%% | max_tournament=%.1f%%",
            self.bankroll,
            self.kelly_fraction_multiplier * 100,
            self.max_single_bet_pct * 100,
            self.max_tournament_exposure * 100,
        )

    def size_bets(
        self,
        opportunities: List[BettingOpportunity],
    ) -> List[BetRecommendation]:
        """
        Convert betting opportunities into sized bet recommendations.
        
        Process:
        1. Compute full Kelly for each opportunity.
        2. Apply fractional Kelly multiplier.
        3. Cap individual bets at max_single_bet_pct.
        4. Cap total tournament exposure at max_tournament_exposure.
        5. Scale down proportionally if over tournament cap.
        
        Parameters
        ----------
        opportunities : list of BettingOpportunity
            From EdgeDetector.find_edges().
            
        Returns
        -------
        list of BetRecommendation
            Sized bets ready for execution.
        """
        if not opportunities:
            return []

        bets = []

        for opp in opportunities:
            full_k = kelly_fraction(opp.p_model, opp.decimal_odds)

            if full_k <= 0:
                continue

            # Apply fractional Kelly
            frac_k = full_k * self.kelly_fraction_multiplier

            # Cap at maximum single bet percentage
            frac_k = min(frac_k, self.max_single_bet_pct)

            # Compute dollar stake
            stake = frac_k * self.bankroll

            # Expected value
            ev = stake * (opp.decimal_odds * opp.p_model - 1.0)

            bets.append(BetRecommendation(
                player_id=opp.player_id,
                player_name=opp.player_name,
                p_model=opp.p_model,
                decimal_odds=opp.decimal_odds,
                book=opp.book,
                full_kelly=round(full_k, 6),
                fractional_kelly=round(frac_k, 6),
                stake_dollars=round(stake, 2),
                expected_value=round(ev, 2),
                edge_pct=round(opp.edge * 100, 1),
            ))

        # Apply tournament-level exposure cap
        bets = self._apply_tournament_cap(bets)

        # Log summary
        total_stake = sum(b.stake_dollars for b in bets)
        total_ev = sum(b.expected_value for b in bets)
        logger.info(
            "Sized %d bets | total_stake=$%.2f (%.1f%% of bankroll) | "
            "total_EV=$%.2f",
            len(bets), total_stake, total_stake / self.bankroll * 100, total_ev,
        )

        return bets

    def _apply_tournament_cap(
        self,
        bets: List[BetRecommendation],
    ) -> List[BetRecommendation]:
        """
        Ensure total tournament exposure doesn't exceed cap.
        
        If total stakes > max_tournament_exposure * bankroll,
        scale all bets down proportionally.
        
        Why not just remove the smallest bets?
        Because proportional scaling preserves the relative Kelly 
        weighting across bets (higher-edge bets keep larger share).
        """
        if not bets:
            return bets

        max_exposure = self.max_tournament_exposure * self.bankroll
        total_stake = sum(b.stake_dollars for b in bets)

        if total_stake <= max_exposure:
            return bets

        # Scale down proportionally
        scale_factor = max_exposure / total_stake
        logger.info(
            "Tournament exposure $%.2f exceeds cap $%.2f. "
            "Scaling all bets by %.2f%%",
            total_stake, max_exposure, scale_factor * 100,
        )

        for bet in bets:
            bet.stake_dollars = round(bet.stake_dollars * scale_factor, 2)
            bet.fractional_kelly *= scale_factor
            bet.expected_value = round(bet.expected_value * scale_factor, 2)

        return bets

    def update_bankroll(self, new_bankroll: float):
        """Update bankroll (e.g., after settling bets)."""
        old = self.bankroll
        self.bankroll = new_bankroll
        logger.info("Bankroll updated: $%.2f → $%.2f", old, new_bankroll)

    def bets_to_dataframe(self, bets: List[BetRecommendation]) -> "pd.DataFrame":
        """Convert bet recommendations to a display DataFrame."""
        import pandas as pd

        if not bets:
            return pd.DataFrame()

        records = [{
            "player": b.player_name or f"ID:{b.player_id}",
            "p_model": f"{b.p_model:.3%}",
            "odds": b.decimal_odds,
            "edge": f"{b.edge_pct:.1f}%",
            "full_kelly": f"{b.full_kelly:.4%}",
            "frac_kelly": f"{b.fractional_kelly:.4%}",
            "stake": f"${b.stake_dollars:.2f}",
            "EV": f"${b.expected_value:.2f}",
            "book": b.book,
        } for b in bets]

        return pd.DataFrame(records)
