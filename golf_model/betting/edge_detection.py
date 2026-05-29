# ==============================================================================
# golf_model/betting/edge_detection.py
# ==============================================================================
#
# EDGE DETECTION
# ---------------
# Compares model probabilities (from Monte Carlo simulation) against 
# market probabilities (from devigged bookmaker odds) to identify 
# betting opportunities where P_model > P_market.
#
# Key concepts:
#
# 1. Edge = P_model / P_market - 1
#    Positive edge = model thinks player is underpriced.
#    Example: P_model = 5%, P_market = 3.5% → Edge = 43%
#
# 2. Not all edges are bettable. We filter by:
#    - Minimum edge threshold (default 5%)
#    - Minimum Kelly fraction (avoids tiny bets)
#    - Maximum individual bet exposure
#    - Sufficient model confidence
#
# 3. Closing Line Value (CLV):
#    After-the-fact check: did our model spot value that the market 
#    eventually recognized? If opening edge → closing edge shrinks,
#    the market moved toward our model = good signal.
#
# ==============================================================================

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BettingOpportunity:
    """
    A single identified betting opportunity.
    
    Attributes
    ----------
    player_id : int
        DataGolf player ID.
    player_name : str
        Human-readable name (for logging).
    p_model : float
        Model's estimated probability of winning.
    p_market : float
        Market's implied probability (after devigging).
    edge : float
        Fractional edge: P_model / P_market - 1.
    decimal_odds : float
        Available decimal odds at the sportsbook.
    book : str
        Sportsbook offering these odds.
    kelly_fraction : float
        Optimal Kelly stake as fraction of bankroll.
    confidence : str
        "high", "medium", or "low" based on edge magnitude and model certainty.
    """
    player_id: int
    player_name: str = ""
    p_model: float = 0.0
    p_market: float = 0.0
    edge: float = 0.0
    decimal_odds: float = 0.0
    book: str = ""
    kelly_fraction: float = 0.0
    confidence: str = "low"


class EdgeDetector:
    """
    Detects betting edges by comparing model vs market probabilities.
    
    Parameters
    ----------
    settings : Settings
        Project configuration (provides thresholds).
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.min_edge = self.settings.MIN_EDGE_THRESHOLD  # Ratio, e.g., 1.05
        self.min_kelly = self.settings.MIN_KELLY_FRACTION
        self.min_bet_probability = getattr(self.settings, 'MIN_BET_PROBABILITY', 0.0)

        logger.info(
            "EdgeDetector initialized | min_edge_ratio=%.2f, min_kelly=%.4f",
            self.min_edge, self.min_kelly,
        )

    def find_edges(
        self,
        model_probs: Dict[int, float],
        market_probs: pd.DataFrame,
        player_names: Optional[Dict[int, str]] = None,
    ) -> List[BettingOpportunity]:
        """
        Compare model probabilities against market for all players.
        
        Parameters
        ----------
        model_probs : dict
            {player_id: P_model(win)} from Monte Carlo simulation.
            
        market_probs : pd.DataFrame
            Must contain: player_id, true_prob, decimal_odds, book.
            From odds_processing.process_tournament_odds().
            
        player_names : dict, optional
            {player_id: "Name"} for display.
            
        Returns
        -------
        list of BettingOpportunity
            Only opportunities that pass all filters, sorted by edge descending.
        """
        opportunities = []

        if market_probs.empty:
            logger.warning("No market probabilities provided")
            return opportunities

        for _, row in market_probs.iterrows():
            pid = int(row["player_id"])
            p_market = row["true_prob"]
            decimal_odds = row["decimal_odds"]
            book = row.get("book", "unknown")

            if pid not in model_probs:
                continue

            p_model = model_probs[pid]

            if p_market <= 0 or p_model <= 0:
                continue

            # Edge ratio: P_model / P_market
            edge_ratio = p_model / p_market
            edge_pct = edge_ratio - 1.0  # Fractional edge

            # Kelly fraction: f* = (b*p - q) / b
            # where b = decimal_odds - 1, p = P_model, q = 1 - P_model
            b = decimal_odds - 1.0
            kelly = (b * p_model - (1 - p_model)) / b if b > 0 else 0.0

            # Apply filters
            if edge_ratio < self.min_edge:
                continue
            if kelly < self.min_kelly:
                continue
            if p_market < self.min_bet_probability:
                continue

            # Confidence level
            if edge_pct > 0.20 and kelly > 0.01:
                confidence = "high"
            elif edge_pct > 0.10:
                confidence = "medium"
            else:
                confidence = "low"

            name = (player_names or {}).get(pid, f"Player {pid}")

            opportunities.append(BettingOpportunity(
                player_id=pid,
                player_name=name,
                p_model=round(p_model, 6),
                p_market=round(p_market, 6),
                edge=round(edge_pct, 4),
                decimal_odds=decimal_odds,
                book=book,
                kelly_fraction=round(kelly, 6),
                confidence=confidence,
            ))

        # Sort by edge (descending)
        opportunities.sort(key=lambda x: x.edge, reverse=True)

        logger.info(
            "Edge detection: %d opportunities found from %d players | "
            "high=%d, medium=%d, low=%d",
            len(opportunities),
            len(model_probs),
            sum(1 for o in opportunities if o.confidence == "high"),
            sum(1 for o in opportunities if o.confidence == "medium"),
            sum(1 for o in opportunities if o.confidence == "low"),
        )

        return opportunities

    def opportunities_to_dataframe(
        self,
        opportunities: List[BettingOpportunity],
    ) -> pd.DataFrame:
        """Convert list of opportunities to a display DataFrame."""
        if not opportunities:
            return pd.DataFrame()

        records = []
        for opp in opportunities:
            records.append({
                "player_id": opp.player_id,
                "player_name": opp.player_name,
                "p_model": opp.p_model,
                "p_market": opp.p_market,
                "edge_pct": round(opp.edge * 100, 1),
                "decimal_odds": opp.decimal_odds,
                "book": opp.book,
                "kelly_frac": opp.kelly_fraction,
                "confidence": opp.confidence,
            })

        return pd.DataFrame(records)

    def compute_clv(
        self,
        bet_probs: Dict[int, float],
        closing_probs: Dict[int, float],
        opening_probs: Dict[int, float],
    ) -> pd.DataFrame:
        """
        Compute Closing Line Value for placed bets.
        
        CLV measures whether the market moved toward our model's 
        assessment between when we bet (opening) and when the market 
        closed (closing).
        
        CLV_i = P_close(i) / P_open(i) - 1
        
        Positive CLV = market moved toward us = strong signal of edge.
        
        Parameters
        ----------
        bet_probs : dict
            {player_id: P at time of bet} for all bets placed.
        closing_probs : dict
            {player_id: closing probability} from final odds.
        opening_probs : dict
            {player_id: opening probability} at time of bet.
            
        Returns
        -------
        pd.DataFrame
            CLV analysis for each bet.
        """
        records = []
        for pid in bet_probs:
            if pid in closing_probs and pid in opening_probs:
                p_close = closing_probs[pid]
                p_open = opening_probs[pid]

                clv = (p_close / p_open - 1) if p_open > 0 else 0

                records.append({
                    "player_id": pid,
                    "p_model": bet_probs[pid],
                    "p_open": p_open,
                    "p_close": p_close,
                    "clv_pct": round(clv * 100, 2),
                    "market_moved_toward_model": clv > 0,
                })

        df = pd.DataFrame(records)
        if len(df) > 0:
            avg_clv = df["clv_pct"].mean()
            pct_positive = (df["clv_pct"] > 0).mean() * 100
            logger.info(
                "CLV analysis: avg=%.1f%%, positive=%.0f%% of bets",
                avg_clv, pct_positive,
            )
        return df
