# Golf Betting Model — Method 1
## Hierarchical Bayesian Strokes-Gained Decomposition

A quantitative golf betting system that estimates player win probabilities via hierarchical Bayesian modeling of strokes-gained data, then detects edges against bookmaker odds.

### Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate golf_model

# 2. Set your API keys
cp .env.example .env
#    Then edit .env and fill in GOLF_DATAGOLF_API_KEY and ANTHROPIC_API_KEY

# 3. Pull data into data/raw/ (requires a DataGolf Scratch PLUS subscription)
python scripts/14_pull_season_data.py --key YOUR_DATAGOLF_API_KEY

# 4. Run pipeline
python run_pipeline.py --mode train
python run_pipeline.py --mode backtest
python run_pipeline.py --mode predict --event_id 12345
```

### Data & Credentials

This repository contains **code only**. The following are intentionally excluded
(see `.gitignore`) and must be supplied locally:

- **`.env`** — your `GOLF_DATAGOLF_API_KEY` and `ANTHROPIC_API_KEY` (copy from `.env.example`).
- **`data/`** — DataGolf strokes-gained, odds, and schedule data is a paid tier and
  cannot be redistributed. Populate `data/raw/` yourself via the `scripts/` pullers
  (or your own DataGolf export).
- **`artifacts/`** and **`cache/`** — trained models, run logs, and cached research are
  regenerated locally by running the pipeline; they are not version-controlled.

### Project Structure

```
golf_model/
├── config/settings.py          # All configuration in one place
├── data/                       # Data loading, schemas, API clients
├── features/                   # SG decomposition, time-weighting, course features
├── models/                     # Baseline, Bayesian core, course-fit
├── simulation/                 # Monte Carlo tournament simulation
├── betting/                    # Odds processing, edge detection, Kelly sizing
├── validation/                 # Metrics, calibration, backtesting
├── utils/                      # Logging, plotting, helpers
├── notebooks/                  # Jupyter notebooks for exploration
└── run_pipeline.py             # End-to-end orchestrator
```

### Validation Gates

The model must pass three gates on 2023-2024 holdout data before live deployment:

1. **Calibration** — Model Brier Score < Market Brier Score
2. **Significance** — Diebold-Mariano p < 0.05
3. **Betting Viability** — ROI > 0%, Sharpe > 0.5, Max DD < 40%
