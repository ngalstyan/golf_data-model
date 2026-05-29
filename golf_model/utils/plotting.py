# ==============================================================================
# golf_model/utils/plotting.py
# ==============================================================================
#
# VISUALIZATION TEMPLATES
# -------------------------
# Standardized matplotlib/seaborn plot templates for golf model analysis.
# Used in Jupyter notebooks for exploration and in validation reports.
#
# All functions return (fig, ax) tuples for further customization.
#
# ==============================================================================

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# Lazy imports for matplotlib/seaborn (may not be available in all contexts)
def _setup_style():
    """Configure matplotlib style for golf model plots."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


# Color palette
COLORS = {
    "model": "#2563eb",      # Blue
    "market": "#dc2626",     # Red
    "win": "#16a34a",        # Green
    "loss": "#ef4444",       # Light red
    "neutral": "#6b7280",    # Gray
    "highlight": "#f59e0b",  # Amber
}


def plot_calibration_curve(
    bin_centers: np.ndarray,
    observed_freqs: np.ndarray,
    bin_counts: np.ndarray,
    model_name: str = "Model",
    ax=None,
) -> Tuple:
    """
    Plot calibration / reliability diagram.
    
    Perfect calibration = points on the diagonal.
    Above diagonal = underconfident. Below = overconfident.
    """
    import matplotlib.pyplot as plt

    _setup_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color=COLORS["neutral"], label="Perfect", alpha=0.8)

    # Calibration curve
    ax.plot(
        bin_centers, observed_freqs, "o-",
        color=COLORS["model"], linewidth=2, markersize=8, label=model_name,
    )

    # Histogram of predictions (bottom)
    ax2 = ax.twinx()
    ax2.bar(
        bin_centers, bin_counts, width=0.08, alpha=0.15,
        color=COLORS["model"], label="Count",
    )
    ax2.set_ylabel("Prediction count", color=COLORS["neutral"])

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    return fig, ax


def plot_backtest_cumulative_pnl(
    pnl_series: np.ndarray,
    event_names: Optional[List[str]] = None,
    ax=None,
) -> Tuple:
    """Plot cumulative P&L over backtested events."""
    import matplotlib.pyplot as plt

    _setup_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    cum_pnl = np.cumsum(pnl_series)
    x = range(len(cum_pnl))

    # Color by positive/negative
    ax.fill_between(x, 0, cum_pnl, where=(cum_pnl >= 0),
                     color=COLORS["win"], alpha=0.3)
    ax.fill_between(x, 0, cum_pnl, where=(cum_pnl < 0),
                     color=COLORS["loss"], alpha=0.3)
    ax.plot(x, cum_pnl, color=COLORS["model"], linewidth=2)
    ax.axhline(y=0, color=COLORS["neutral"], linestyle="--", alpha=0.5)

    ax.set_xlabel("Tournament #")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title("Backtest: Cumulative Profit & Loss")

    return fig, ax


def plot_bankroll_trajectory(
    bankroll_values: np.ndarray,
    initial_bankroll: float,
    ax=None,
) -> Tuple:
    """Plot bankroll over time with drawdown shading."""
    import matplotlib.pyplot as plt

    _setup_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    x = range(len(bankroll_values))
    peak = np.maximum.accumulate(bankroll_values)

    ax.plot(x, bankroll_values, color=COLORS["model"], linewidth=2, label="Bankroll")
    ax.plot(x, peak, color=COLORS["neutral"], linewidth=1, linestyle="--",
            alpha=0.5, label="Peak")
    ax.fill_between(x, bankroll_values, peak,
                     color=COLORS["loss"], alpha=0.15, label="Drawdown")
    ax.axhline(y=initial_bankroll, color=COLORS["neutral"],
               linestyle=":", alpha=0.5, label=f"Initial (${initial_bankroll:,.0f})")

    ax.set_xlabel("Event #")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Bankroll Trajectory")
    ax.legend()

    return fig, ax


def plot_brier_comparison(
    model_briers: np.ndarray,
    market_briers: np.ndarray,
    event_names: Optional[List[str]] = None,
    ax=None,
) -> Tuple:
    """Plot per-event Brier Score comparison (model vs market)."""
    import matplotlib.pyplot as plt

    _setup_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig = ax.figure

    n = len(model_briers)
    x = range(n)

    ax.bar(x, market_briers, alpha=0.4, color=COLORS["market"],
           label=f"Market (avg={np.mean(market_briers):.4f})")
    ax.bar(x, model_briers, alpha=0.6, color=COLORS["model"],
           label=f"Model (avg={np.mean(model_briers):.4f})")

    ax.set_xlabel("Tournament")
    ax.set_ylabel("Brier Score (lower = better)")
    ax.set_title("Brier Score: Model vs Market")
    ax.legend()

    return fig, ax


def plot_edge_distribution(
    edges_pct: np.ndarray,
    ax=None,
) -> Tuple:
    """Plot distribution of detected edges."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    _setup_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    sns.histplot(edges_pct, bins=30, ax=ax, color=COLORS["model"], alpha=0.7)
    ax.axvline(x=0, color=COLORS["neutral"], linestyle="--")
    ax.axvline(x=5, color=COLORS["highlight"], linestyle="--",
               label="Min threshold (5%)")

    ax.set_xlabel("Edge (%)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Detected Edges")
    ax.legend()

    return fig, ax


def plot_posterior_distributions(
    player_posteriors: Dict[str, np.ndarray],
    title: str = "Posterior Skill Distributions",
    ax=None,
) -> Tuple:
    """
    Plot overlapping posterior distributions for multiple players.
    
    Parameters
    ----------
    player_posteriors : dict
        {"Player Name": np.ndarray of posterior samples}
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    _setup_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    colors = plt.cm.Set2(np.linspace(0, 1, len(player_posteriors)))

    for (name, samples), color in zip(player_posteriors.items(), colors):
        sns.kdeplot(samples, ax=ax, label=name, color=color, linewidth=2)

    ax.axvline(x=0, color=COLORS["neutral"], linestyle="--", alpha=0.5,
               label="Tour average")
    ax.set_xlabel("Strokes Gained per Round")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    return fig, ax


def plot_sg_decomposition(
    player_name: str,
    sg_components: Dict[str, float],
    ax=None,
) -> Tuple:
    """
    Bar chart of SG component decomposition for a single player.
    
    Parameters
    ----------
    player_name : str
    sg_components : dict
        {"OTT": 0.45, "APP": 0.32, "ARG": -0.10, "PUTT": 0.15}
    """
    import matplotlib.pyplot as plt

    _setup_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    components = list(sg_components.keys())
    values = list(sg_components.values())
    colors = [COLORS["win"] if v >= 0 else COLORS["loss"] for v in values]

    bars = ax.barh(components, values, color=colors, alpha=0.8, height=0.5)
    ax.axvline(x=0, color=COLORS["neutral"], linewidth=1)

    # Value labels
    for bar, val in zip(bars, values):
        x_pos = val + 0.02 if val >= 0 else val - 0.02
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}", va="center", ha=ha, fontsize=10)

    total = sum(values)
    ax.set_xlabel("Strokes Gained per Round")
    ax.set_title(f"{player_name} — SG Decomposition (Total: {total:+.2f})")

    return fig, ax
