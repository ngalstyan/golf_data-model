# ==============================================================================
# golf_model/run_pipeline.py
# ==============================================================================
#
# END-TO-END PIPELINE ORCHESTRATOR
# ----------------------------------
# Chains all modules together into a single executable pipeline.
#
# Three execution modes:
#   1. TRAIN   — Fit model on historical data.
#   2. BACKTEST — Expanding-window validation on holdout.
#   3. PREDICT  — Generate predictions for an upcoming tournament.
#
# Usage:
#   # From command line:
#   python run_pipeline.py --mode train
#   python run_pipeline.py --mode backtest
#   python run_pipeline.py --mode predict --event_id 12345
#
#   # From Python/Jupyter:
#   from run_pipeline import Pipeline
#   pipe = Pipeline()
#   pipe.train()
#   results = pipe.backtest()
#   predictions = pipe.predict(event_id=12345)
#
# ==============================================================================

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import Settings
from data.loader import DataLoader
from features.pipeline import FeaturePipeline
from models.baseline import BaselineModel, ModelRegistry
from validation.backtest import BacktestEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class Pipeline:
    """
    Master orchestrator for the golf betting model.
    
    Chains together:
        Data loading → Feature engineering → Model fitting →
        Simulation → Edge detection → Bet sizing → Validation
    
    Parameters
    ----------
    settings : Settings, optional
        Project configuration. Uses defaults if not provided.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()

        # Validate settings
        warnings = self.settings.validate()
        for w in warnings:
            logger.warning("Config: %s", w)

        # Initialize components
        self.loader = DataLoader(self.settings)
        self.feature_pipeline = FeaturePipeline(self.settings)
        self.model = BaselineModel(self.settings)  # Start with baseline
        self.registry = ModelRegistry(self.settings)
        self.backtest_engine = BacktestEngine(self.settings)

        # Data containers (populated by train/load steps)
        self._rounds_df: Optional[pd.DataFrame] = None
        self._events_df: Optional[pd.DataFrame] = None
        self._odds_df: Optional[pd.DataFrame] = None

        logger.info("Pipeline initialized | mode=ready")

    # ==========================================================================
    # MODE 1: TRAIN
    # ==========================================================================

    def train(self) -> Dict:
        """
        Train the model on all available training data.
        
        Steps:
            1. Load CSV data (rounds, events, odds).
            2. Run feature engineering pipeline.
            3. Fit the model.
            4. Save model artifacts.
            5. Return training summary.
            
        Returns
        -------
        dict
            Training summary with data shapes, model metadata.
        """
        logger.info("=" * 60)
        logger.info("PIPELINE: TRAINING MODE")
        logger.info("=" * 60)

        # Step 1: Load data
        logger.info("Step 1/4: Loading data...")
        self._load_data()

        train_rounds = self._rounds_df[
            self._rounds_df["season"].isin(self.settings.TRAIN_SEASONS)
        ] if "season" in self._rounds_df.columns else self._rounds_df

        logger.info(
            "Training data: %d rounds from %d players",
            len(train_rounds), train_rounds["player_id"].nunique(),
        )

        # Step 2: Feature engineering
        logger.info("Step 2/4: Engineering features...")
        features = self.feature_pipeline.run(train_rounds)

        # Step 3: Fit model
        logger.info("Step 3/4: Fitting model...")
        self.model.fit(features)

        # Step 4: Save
        logger.info("Step 4/4: Saving model...")
        model_path = self.registry.save(
            self.model,
            metadata={
                "train_seasons": self.settings.TRAIN_SEASONS,
                "n_rounds": len(train_rounds),
                "n_players": train_rounds["player_id"].nunique(),
            },
        )

        summary = {
            "status": "success",
            "n_training_rounds": len(train_rounds),
            "n_players": int(train_rounds["player_id"].nunique()),
            "train_seasons": self.settings.TRAIN_SEASONS,
            "model_path": str(model_path) if model_path else None,
        }

        logger.info("Training complete: %s", summary)
        return summary

    # ==========================================================================
    # MODE 2: BACKTEST
    # ==========================================================================

    def backtest(self) -> "BacktestResult":
        """
        Run expanding-window backtest on holdout data.
        
        Steps:
            1. Load all data.
            2. Define the fit-and-predict function.
            3. Run BacktestEngine.
            4. Generate validation report.
            
        Returns
        -------
        BacktestResult
            Full backtest results with gate verdicts.
        """
        logger.info("=" * 60)
        logger.info("PIPELINE: BACKTEST MODE")
        logger.info("=" * 60)

        # Load data
        self._load_data()

        # Define the callback that the backtest engine will call
        def fit_and_predict(
            train_rounds: pd.DataFrame,
            event_info: dict,
            event_rounds: pd.DataFrame,
        ) -> Dict[int, float]:
            """
            Called by BacktestEngine for each holdout event.
            
            1. Run feature pipeline on training data.
            2. Fit model.
            3. Predict win probabilities for event players.
            """
            # Feature engineering on training data
            features = self.feature_pipeline.run(train_rounds)

            # Fit model
            model = BaselineModel(self.settings)
            model.fit(features)

            # Get event player list
            player_ids = event_rounds["player_id"].unique()

            # Predict (baseline model returns SG estimates)
            predictions = model.predict(player_ids)

            # Convert SG estimates to win probabilities
            # Simple approach: softmax-like conversion
            # (full version uses Monte Carlo simulation)
            if predictions is not None and len(predictions) > 0:
                sg_values = np.array([
                    predictions.get(pid, 0.0) for pid in player_ids
                ])

                # Temperature-scaled softmax: higher SG = higher P(win)
                # Temperature controls how much skill separates players
                temperature = 0.5  # Lower = more separation
                exp_sg = np.exp(sg_values / temperature)
                probs = exp_sg / exp_sg.sum()

                return dict(zip(player_ids.tolist(), probs.tolist()))

            return {}

        # Run backtest
        result = self.backtest_engine.run(
            events_df=self._events_df,
            rounds_df=self._rounds_df,
            odds_df=self._odds_df if self._odds_df is not None else pd.DataFrame(),
            fit_and_predict_fn=fit_and_predict,
        )

        # Print summary
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("-" * 60)
        logger.info("Events: %d", result.total_events)
        logger.info("Model avg Brier: %.5f", result.model_avg_brier)
        logger.info("Market avg Brier: %.5f", result.market_avg_brier)
        logger.info("ROI: %.1f%%", result.roi_pct)
        logger.info("Sharpe: %.2f", result.sharpe)
        logger.info("Max Drawdown: %.1f%%", result.max_dd_pct)
        logger.info("-" * 60)
        logger.info("Gate 1 (Calibration): %s",
                     "PASS" if result.gate_1_passed else "FAIL")
        logger.info("Gate 3 (Betting): %s",
                     "PASS" if result.gate_3_passed else "FAIL")
        logger.info("=" * 60)

        return result

    # ==========================================================================
    # MODE 3: PREDICT
    # ==========================================================================

    def predict(
        self,
        event_id: int,
        field_player_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions for an upcoming tournament.
        
        Steps:
            1. Load latest data.
            2. Load fitted model.
            3. Generate features for field players.
            4. Run simulation (or baseline prediction).
            5. Return sorted predictions.
            
        Parameters
        ----------
        event_id : int
            DataGolf event ID for the target tournament.
        field_player_ids : list of int, optional
            Override field. If None, inferred from data.
            
        Returns
        -------
        pd.DataFrame
            Predictions sorted by P(win) descending.
        """
        logger.info("=" * 60)
        logger.info("PIPELINE: PREDICT MODE | event=%d", event_id)
        logger.info("=" * 60)

        # Load data
        self._load_data()

        # Feature engineering on all available data
        features = self.feature_pipeline.run(self._rounds_df)

        # Fit model on all available data (for live predictions)
        self.model.fit(features)

        # Determine field
        if field_player_ids is None:
            # Try to get from event data
            event_rounds = self._rounds_df[self._rounds_df["event_id"] == event_id]
            if len(event_rounds) > 0:
                field_player_ids = event_rounds["player_id"].unique().tolist()
            else:
                logger.warning("No field data for event %d. Using all known players.", event_id)
                field_player_ids = self._rounds_df["player_id"].unique().tolist()

        # Predict
        predictions = self.model.predict(field_player_ids)

        if predictions is None or len(predictions) == 0:
            logger.warning("No predictions generated")
            return pd.DataFrame()

        # Convert to probabilities
        player_ids = np.array(list(predictions.keys()))
        sg_values = np.array(list(predictions.values()))

        temperature = 0.5
        exp_sg = np.exp(sg_values / temperature)
        probs = exp_sg / exp_sg.sum()

        # Build output DataFrame
        result_df = pd.DataFrame({
            "player_id": player_ids,
            "sg_estimate": np.round(sg_values, 4),
            "p_win": np.round(probs, 6),
            "implied_odds": np.round(1.0 / np.maximum(probs, 1e-10), 1),
        })
        result_df = result_df.sort_values("p_win", ascending=False).reset_index(drop=True)
        result_df["rank"] = range(1, len(result_df) + 1)

        logger.info("Predictions generated for %d players", len(result_df))

        return result_df

    # ==========================================================================
    # PRIVATE HELPERS
    # ==========================================================================

    def _load_data(self):
        """Load all required data files."""
        if self._rounds_df is not None:
            return  # Already loaded

        try:
            self._rounds_df = self.loader.load_rounds()
        except FileNotFoundError as e:
            logger.error("Cannot load rounds: %s", e)
            raise

        try:
            self._events_df = self.loader.load_events()
        except FileNotFoundError:
            logger.warning("No events data found. Creating minimal events from rounds.")
            self._events_df = self._infer_events()

        try:
            self._odds_df = self.loader.load_odds()
        except FileNotFoundError:
            logger.warning("No odds data found. Betting analysis will be limited.")
            self._odds_df = pd.DataFrame()

    def _infer_events(self) -> pd.DataFrame:
        """Create minimal events DataFrame from rounds data."""
        if self._rounds_df is None or self._rounds_df.empty:
            return pd.DataFrame(columns=["event_id", "event_name", "calendar_year", "start_date"])

        events = self._rounds_df.groupby("event_id").agg(
            start_date=("date", "min"),
            n_rounds=("round_num", "max"),
            n_players=("player_id", "nunique"),
        ).reset_index()

        if "event_name" in self._rounds_df.columns:
            names = self._rounds_df.groupby("event_id")["event_name"].first()
            events = events.merge(names, on="event_id", how="left")
        else:
            events["event_name"] = "Event " + events["event_id"].astype(str)

        events["calendar_year"] = pd.to_datetime(events["start_date"]).dt.year

        return events


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Golf Betting Model — Method 1 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py --mode train
    python run_pipeline.py --mode backtest
    python run_pipeline.py --mode predict --event_id 12345
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "backtest", "predict"],
        required=True,
        help="Pipeline execution mode.",
    )
    parser.add_argument(
        "--event_id",
        type=int,
        default=None,
        help="Event ID for predict mode.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override DATA_DIR path.",
    )

    args = parser.parse_args()

    # Configure
    settings_kwargs = {}
    if args.data_dir:
        settings_kwargs["DATA_DIR"] = Path(args.data_dir)

    settings = Settings(**settings_kwargs) if settings_kwargs else Settings()
    pipe = Pipeline(settings)

    # Execute
    if args.mode == "train":
        pipe.train()
    elif args.mode == "backtest":
        pipe.backtest()
    elif args.mode == "predict":
        if args.event_id is None:
            print("Error: --event_id required for predict mode")
            sys.exit(1)
        predictions = pipe.predict(args.event_id)
        print(predictions.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
