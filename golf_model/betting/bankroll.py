# ==============================================================================
# golf_model/betting/bankroll.py
# ==============================================================================
#
# BANKROLL MANAGEMENT & P&L TRACKING
# ------------------------------------
# Tracks bankroll over time, records all bets and outcomes, computes 
# performance metrics (ROI, Sharpe, drawdown), and enforces risk limits.
#
# Design:
#   - Append-only bet log (never modify past records).
#   - Bankroll recalculated from initial + sum of P&L.
#   - All timestamps in UTC for consistency.
#   - CSV export for external analysis.
#
# ==============================================================================

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import Settings
from betting.kelly import BetRecommendation
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BetRecord:
    """
    A single bet record with full lifecycle tracking.
    
    Created when bet is placed, updated when settled.
    """
    bet_id: str                          # Unique identifier
    event_id: int                        # Tournament
    event_name: str = ""
    player_id: int = 0
    player_name: str = ""
    market: str = "win"                  # "win", "top_5", "top_10"
    book: str = ""                       # Sportsbook
    decimal_odds: float = 0.0
    stake: float = 0.0                   # Dollar amount wagered
    p_model: float = 0.0                 # Model probability at time of bet
    p_market: float = 0.0                # Market probability at time of bet
    edge_pct: float = 0.0               # Edge at time of bet
    kelly_fraction: float = 0.0          # Kelly fraction used
    placed_at: str = ""                  # ISO timestamp when bet placed
    settled: bool = False                # Has this bet been resolved?
    won: bool = False                    # Did this bet win?
    pnl: float = 0.0                     # Profit/loss (positive = profit)
    settled_at: str = ""                 # ISO timestamp when settled
    bankroll_after: float = 0.0          # Bankroll after this bet settled


class BankrollTracker:
    """
    Tracks bankroll, records bets, and computes performance metrics.
    
    Parameters
    ----------
    settings : Settings
        Project configuration.
    initial_bankroll : float, optional
        Starting bankroll. Defaults to settings.INITIAL_BANKROLL.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        initial_bankroll: Optional[float] = None,
    ):
        self.settings = settings or Settings()
        self.initial_bankroll = initial_bankroll or self.settings.INITIAL_BANKROLL
        self.current_bankroll = self.initial_bankroll

        # Full bet history
        self.bet_log: List[BetRecord] = []
        self._bet_counter = 0

        # Bankroll snapshots (after each settlement)
        self.bankroll_history: List[Tuple[str, float]] = [
            (datetime.utcnow().isoformat(), self.initial_bankroll)
        ]

        logger.info("BankrollTracker initialized | bankroll=$%.2f", self.initial_bankroll)

    # ==========================================================================
    # BET LIFECYCLE
    # ==========================================================================

    def place_bet(
        self,
        recommendation: BetRecommendation,
        event_id: int,
        event_name: str = "",
        market: str = "win",
    ) -> BetRecord:
        """
        Record a new bet placement.
        
        This deducts the stake from available bankroll and creates
        a BetRecord in the log.
        
        Parameters
        ----------
        recommendation : BetRecommendation
            Sized bet from KellyCalculator.
        event_id : int
        event_name : str
        market : str
            
        Returns
        -------
        BetRecord
            The created bet record.
        """
        self._bet_counter += 1
        bet_id = f"BET-{self._bet_counter:06d}"
        now = datetime.utcnow().isoformat()

        record = BetRecord(
            bet_id=bet_id,
            event_id=event_id,
            event_name=event_name,
            player_id=recommendation.player_id,
            player_name=recommendation.player_name,
            market=market,
            book=recommendation.book,
            decimal_odds=recommendation.decimal_odds,
            stake=recommendation.stake_dollars,
            p_model=recommendation.p_model,
            edge_pct=recommendation.edge_pct,
            kelly_fraction=recommendation.fractional_kelly,
            placed_at=now,
        )

        self.bet_log.append(record)
        self.current_bankroll -= recommendation.stake_dollars

        logger.info(
            "Bet placed: %s | %s | $%.2f @ %.1f | bankroll=$%.2f",
            bet_id, recommendation.player_name,
            recommendation.stake_dollars, recommendation.decimal_odds,
            self.current_bankroll,
        )

        return record

    def settle_bet(
        self,
        bet_id: str,
        won: bool,
    ) -> BetRecord:
        """
        Settle an existing bet.
        
        Parameters
        ----------
        bet_id : str
            The bet ID to settle.
        won : bool
            Whether the bet won.
            
        Returns
        -------
        BetRecord
            Updated bet record with P&L.
        """
        record = self._find_bet(bet_id)
        if record is None:
            raise ValueError(f"Bet not found: {bet_id}")
        if record.settled:
            raise ValueError(f"Bet already settled: {bet_id}")

        record.settled = True
        record.won = won
        record.settled_at = datetime.utcnow().isoformat()

        if won:
            # Win: return stake + profit
            payout = record.stake * record.decimal_odds
            record.pnl = payout - record.stake  # Net profit
            self.current_bankroll += payout
        else:
            # Loss: stake already deducted
            record.pnl = -record.stake

        record.bankroll_after = self.current_bankroll
        self.bankroll_history.append(
            (record.settled_at, self.current_bankroll)
        )

        logger.info(
            "Bet settled: %s | %s | %s | P&L=$%.2f | bankroll=$%.2f",
            bet_id, record.player_name,
            "WON" if won else "LOST",
            record.pnl, self.current_bankroll,
        )

        return record

    def settle_tournament(
        self,
        event_id: int,
        winner_player_id: int,
        top5_ids: Optional[List[int]] = None,
        top10_ids: Optional[List[int]] = None,
    ):
        """
        Settle all bets for a completed tournament.
        
        Parameters
        ----------
        event_id : int
        winner_player_id : int
            DataGolf player ID of the tournament winner.
        top5_ids : list of int, optional
            Players who finished top 5.
        top10_ids : list of int, optional
            Players who finished top 10.
        """
        top5_ids = set(top5_ids or [])
        top10_ids = set(top10_ids or [])

        event_bets = [
            b for b in self.bet_log
            if b.event_id == event_id and not b.settled
        ]

        for bet in event_bets:
            if bet.market == "win":
                won = bet.player_id == winner_player_id
            elif bet.market == "top_5":
                won = bet.player_id in top5_ids
            elif bet.market == "top_10":
                won = bet.player_id in top10_ids
            else:
                won = bet.player_id == winner_player_id

            self.settle_bet(bet.bet_id, won)

        logger.info(
            "Tournament %d settled | %d bets processed", event_id, len(event_bets)
        )

    # ==========================================================================
    # PERFORMANCE METRICS
    # ==========================================================================

    def compute_metrics(self) -> Dict:
        """
        Compute comprehensive performance metrics from bet history.
        
        Returns
        -------
        dict with keys:
            n_bets, n_wins, win_rate, total_staked, total_pnl, roi,
            avg_odds, avg_edge, sharpe_ratio, max_drawdown, max_drawdown_pct,
            current_bankroll, bankroll_growth
        """
        settled = [b for b in self.bet_log if b.settled]

        if not settled:
            return {"n_bets": 0, "message": "No settled bets yet"}

        n_bets = len(settled)
        n_wins = sum(1 for b in settled if b.won)
        total_staked = sum(b.stake for b in settled)
        total_pnl = sum(b.pnl for b in settled)

        pnls = np.array([b.pnl for b in settled])
        stakes = np.array([b.stake for b in settled])
        returns = pnls / stakes  # Return per unit staked

        # ROI (return on investment)
        roi = total_pnl / total_staked if total_staked > 0 else 0

        # Sharpe ratio (annualized, assuming ~40 events/year)
        if len(returns) > 1 and np.std(returns) > 0:
            # Per-bet Sharpe × sqrt(bets_per_year) for annualized
            per_bet_sharpe = np.mean(returns) / np.std(returns)
            bets_per_year = min(len(settled), 40 * 5)  # Rough estimate
            annualized_sharpe = per_bet_sharpe * np.sqrt(bets_per_year)
        else:
            annualized_sharpe = 0.0

        # Maximum drawdown
        bankroll_values = [self.initial_bankroll]
        running = self.initial_bankroll
        for b in settled:
            running += b.pnl
            bankroll_values.append(running)

        bankroll_arr = np.array(bankroll_values)
        peak = np.maximum.accumulate(bankroll_arr)
        drawdown = (peak - bankroll_arr) / peak
        max_dd_pct = float(np.max(drawdown))
        max_dd_dollar = float(np.max(peak - bankroll_arr))

        return {
            "n_bets": n_bets,
            "n_wins": n_wins,
            "win_rate": round(n_wins / n_bets, 4) if n_bets > 0 else 0,
            "total_staked": round(total_staked, 2),
            "total_pnl": round(total_pnl, 2),
            "roi": round(roi, 4),
            "roi_pct": round(roi * 100, 2),
            "avg_odds": round(np.mean([b.decimal_odds for b in settled]), 1),
            "avg_edge_pct": round(np.mean([b.edge_pct for b in settled]), 1),
            "sharpe_ratio": round(annualized_sharpe, 2),
            "max_drawdown_pct": round(max_dd_pct * 100, 1),
            "max_drawdown_dollar": round(max_dd_dollar, 2),
            "current_bankroll": round(self.current_bankroll, 2),
            "bankroll_growth_pct": round(
                (self.current_bankroll / self.initial_bankroll - 1) * 100, 1
            ),
        }

    # ==========================================================================
    # EXPORT & UTILITY
    # ==========================================================================

    def to_dataframe(self) -> pd.DataFrame:
        """Export full bet log as DataFrame."""
        if not self.bet_log:
            return pd.DataFrame()

        records = []
        for b in self.bet_log:
            records.append({
                "bet_id": b.bet_id,
                "event_id": b.event_id,
                "event_name": b.event_name,
                "player_id": b.player_id,
                "player_name": b.player_name,
                "market": b.market,
                "book": b.book,
                "odds": b.decimal_odds,
                "stake": b.stake,
                "p_model": b.p_model,
                "edge_pct": b.edge_pct,
                "kelly": b.kelly_fraction,
                "placed_at": b.placed_at,
                "settled": b.settled,
                "won": b.won,
                "pnl": b.pnl,
                "settled_at": b.settled_at,
                "bankroll_after": b.bankroll_after,
            })

        return pd.DataFrame(records)

    def save_log(self, filepath: Optional[Path] = None):
        """Save bet log to CSV."""
        if filepath is None:
            filepath = self.settings.OUTPUTS_DIR / "bet_log.csv"

        df = self.to_dataframe()
        if not df.empty:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath, index=False)
            logger.info("Bet log saved to %s (%d records)", filepath, len(df))

    def bankroll_series(self) -> pd.Series:
        """Return bankroll over time as a pandas Series."""
        if not self.bankroll_history:
            return pd.Series(dtype=float)

        dates, values = zip(*self.bankroll_history)
        return pd.Series(values, index=pd.to_datetime(dates), name="bankroll")

    def _find_bet(self, bet_id: str) -> Optional[BetRecord]:
        """Find a bet record by ID."""
        for b in self.bet_log:
            if b.bet_id == bet_id:
                return b
        return None
