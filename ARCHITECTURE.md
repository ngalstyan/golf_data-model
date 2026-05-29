# Golf Betting Model — Method 1: Architecture Plan
## Hierarchical Bayesian Strokes-Gained Decomposition

---

## 1. PROJECT STRUCTURE

```
golf_model/
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Central config: paths, API keys, hyperparameters
│
├── data/
│   ├── __init__.py
│   ├── schemas.py               # Data contracts: expected columns, types, validation
│   ├── loader.py                # Load & validate CSVs from disk
│   ├── api_client.py            # DataGolf API client (fresh build)
│   └── weather.py               # Open-Meteo free-tier historical weather
│
├── features/
│   ├── __init__.py
│   ├── strokes_gained.py        # SG decomposition, normalization, aggregation
│   ├── time_weighting.py        # EWMA exponential decay (dual weighting)
│   ├── course_features.py       # γ_c vector: course characteristic profiles
│   ├── field_strength.py        # Tournament field strength index
│   └── pipeline.py              # Master feature engineering orchestrator
│
├── models/
│   ├── __init__.py
│   ├── baseline.py              # Phase 1: Weighted SG regression (no Bayes)
│   ├── bayesian_core.py         # Phase 3: Hierarchical Bayesian (PyMC/NumPyro)
│   ├── course_fit.py            # Phase 4: Player × course interactions (δ_i)
│   ├── priors.py                # Prior specifications & hyperparameter defaults
│   └── model_registry.py        # Model versioning, save/load, metadata
│
├── simulation/
│   ├── __init__.py
│   ├── monte_carlo.py           # Core MC engine: draw abilities → simulate rounds
│   └── tournament.py            # Tournament structure: cuts, scoring, ties
│
├── betting/
│   ├── __init__.py
│   ├── odds_processing.py       # Overround removal (Shin, proportional, power)
│   ├── kelly.py                 # Full & fractional Kelly, bet sizing
│   ├── edge_detection.py        # P_model vs P_market, threshold filters
│   └── bankroll.py              # Bankroll tracker, exposure limits, P&L log
│
├── validation/
│   ├── __init__.py
│   ├── metrics.py               # Log-loss, Brier Score, ROI, Sharpe, CLV
│   ├── calibration.py           # Calibration plots, Murphy decomposition, PIT
│   ├── backtest.py              # Expanding-window backtesting engine
│   └── statistical_tests.py     # Diebold-Mariano, likelihood ratio, bootstrap
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                # Structured logging (file + console)
│   ├── plotting.py              # Matplotlib/seaborn templates for golf viz
│   └── helpers.py               # Date math, player ID normalization, etc.
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_bayesian_model.ipynb
│   ├── 05_course_fit.ipynb
│   ├── 06_simulation.ipynb
│   ├── 07_betting_integration.ipynb
│   ├── 08_full_backtest.ipynb
│   └── 09_live_deployment.ipynb
│
├── tests/                       # Unit tests (pytest)
│   ├── __init__.py
│   ├── test_loader.py
│   ├── test_features.py
│   ├── test_kelly.py
│   └── test_simulation.py
│
├── environment.yml              # Conda environment specification
├── README.md                    # Project documentation
└── run_pipeline.py              # End-to-end pipeline orchestrator
```

---

## 2. PIPE-BY-PIPE BUILD ORDER

Each pipe is self-contained, testable, and builds on the previous one.
The architecture enforces: **no pipe depends on a pipe that comes after it.**

### Pipe 1: Foundation (config + data schemas + loader)
**Files:** `config/settings.py`, `data/schemas.py`, `data/loader.py`, `utils/logger.py`, `utils/helpers.py`, `environment.yml`
**Purpose:** Centralized configuration, data validation contracts, CSV loading with schema enforcement, structured logging.
**Tests with:** Load your CSVs, validate they match expected schemas, log any issues.

### Pipe 2: DataGolf API Client
**Files:** `data/api_client.py`
**Purpose:** Fresh API client for DataGolf endpoints (event lists, round-level SG, player ratings, historical odds). Clean retry logic, rate limiting, response caching.
**Tests with:** Pull a single tournament's data, verify schema compliance.

### Pipe 3: Feature Engineering — Strokes Gained
**Files:** `features/strokes_gained.py`, `features/field_strength.py`
**Purpose:** SG decomposition processing (OTT, APP, ARG, PUTT), field-strength-adjusted SG, per-round normalization.
**Tests with:** Compute SG aggregates for known players, verify against DataGolf published values.

### Pipe 4: Feature Engineering — Time Weighting
**Files:** `features/time_weighting.py`
**Purpose:** Exponential decay EWMA. Dual weighting (time-based + sequence-based). Configurable half-life λ.
**Tests with:** Compute time-weighted SG for a player, verify recent rounds dominate.

### Pipe 5: Feature Engineering — Course Features
**Files:** `features/course_features.py`, `features/pipeline.py`
**Purpose:** γ_c course characteristic vectors (length, rough, green speed, wind, elevation). Z-score standardization across courses. Master pipeline that chains Pipes 3–5.
**Tests with:** Generate course feature matrix, verify standardization.

### Pipe 6: Baseline Model (Phase 1)
**Files:** `models/baseline.py`, `models/model_registry.py`
**Purpose:** Simple weighted-average SG regression. No Bayesian machinery. This is the "beat this or stop" baseline. Model save/load with metadata tracking.
**Tests with:** Predict next-tournament SG for all players, measure RMSE vs naive field average.

### Pipe 7: Bayesian Core Model (Phase 3)
**Files:** `models/bayesian_core.py`, `models/priors.py`
**Purpose:** Full hierarchical Bayesian model in PyMC. Population-level priors, player-specific shrinkage, sub-component decomposition, MCMC/ADVI inference. Convergence diagnostics (R-hat, ESS, trace plots).
**Tests with:** Fit on training data, verify convergence, compare posteriors against baseline.

### Pipe 8: Course-Fit Model (Phase 4)
**Files:** `models/course_fit.py`
**Purpose:** Player × course-feature interaction coefficients (δ_i). Feature-based (not course fixed effects). Leave-one-course-out validation.
**Tests with:** Check whether course-fit term significantly improves log-loss.

### Pipe 9: Monte Carlo Tournament Simulation
**Files:** `simulation/monte_carlo.py`, `simulation/tournament.py`
**Purpose:** Draw from posterior → simulate 4 rounds × N players → determine winner. Handle cuts (top 65+ties after R2), playoff tiebreakers. 100K+ simulations per tournament. Output: P(win) for each player.
**Tests with:** Simulate a historical tournament, compare P(win) distribution shape.

### Pipe 10: Odds Processing & Edge Detection
**Files:** `betting/odds_processing.py`, `betting/edge_detection.py`
**Purpose:** Overround removal (Shin's method primary, proportional fallback). Convert decimal odds → implied probabilities. Edge = P_model - P_market. Threshold filters (min edge, min Kelly fraction).
**Tests with:** Process historical Pinnacle odds, verify probabilities sum to ~1.0 after Shin correction.

### Pipe 11: Kelly Criterion & Bankroll Management
**Files:** `betting/kelly.py`, `betting/bankroll.py`
**Purpose:** Full Kelly, fractional Kelly (default 25%), correlated bet handling for same-tournament outrights. Bankroll tracking, exposure caps (2% single bet, 8% tournament), P&L logging.
**Tests with:** Calculate Kelly stakes for known edge/odds combos, verify against hand calculations.

### Pipe 12: Validation Framework
**Files:** `validation/metrics.py`, `validation/calibration.py`, `validation/statistical_tests.py`, `validation/backtest.py`
**Purpose:** All three gates: calibration (Brier, log-loss, Murphy), significance (Diebold-Mariano, PIT), betting viability (ROI, Sharpe, CLV, drawdown). Expanding-window backtesting engine.
**Tests with:** Run full backtest on 2023–2024 holdout, generate validation report.

### Pipe 13: Weather Integration (Optional Enhancement)
**Files:** `data/weather.py`
**Purpose:** Open-Meteo free-tier historical weather for wave adjustments. Wind speed, temperature, precipitation at tee-time hour.
**Tests with:** Pull weather for historical tournament, cross-reference AM/PM wave assignments.

### Pipe 14: Visualization & Reporting
**Files:** `utils/plotting.py`, all notebooks
**Purpose:** Standardized plot templates (calibration curves, posterior distributions, bankroll trajectories, feature importance). Jupyter notebooks that call the modules for interactive exploration.

### Pipe 15: Pipeline Orchestrator
**Files:** `run_pipeline.py`
**Purpose:** End-to-end execution: load data → engineer features → fit model → simulate → detect edges → size bets → log results. Configurable for training, backtesting, or live prediction modes.

---

## 3. DATA FLOW

```
[CSV Files on Disk]
        │
        ▼
┌─── data/loader.py ───┐
│  Validate schemas     │
│  Type enforcement     │
│  Missing data flags   │
└───────┬───────────────┘
        │
        ▼
┌─── features/pipeline.py ─────────────────────────────┐
│                                                       │
│  strokes_gained.py  →  Decompose & normalize SG      │
│  time_weighting.py  →  Apply EWMA decay              │
│  course_features.py →  Build γ_c vectors              │
│  field_strength.py  →  Compute field strength index   │
│                                                       │
│  OUTPUT: player_features DataFrame                    │
│          course_features DataFrame                    │
│          tournament_metadata DataFrame                │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌─── models/ ───────────────────────────────────────────┐
│                                                       │
│  baseline.py      →  Phase 1: weighted SG regression  │
│  bayesian_core.py →  Phase 3: hierarchical Bayesian   │
│  course_fit.py    →  Phase 4: δ_i interactions        │
│                                                       │
│  OUTPUT: posterior distributions over μ_i, τ²_i, δ_i  │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌─── simulation/monte_carlo.py ─────────────────────────┐
│                                                       │
│  Draw μ_i from posterior                              │
│  Add course-fit: γ_c · δ_i                            │
│  Simulate 4 rounds with ε ~ t(ν, 0, σ²)              │
│  Determine winner across 100K+ simulations            │
│                                                       │
│  OUTPUT: P(win|player, tournament) for all players    │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌─── betting/ ──────────────────────────────────────────┐
│                                                       │
│  odds_processing.py →  Shin overround removal         │
│  edge_detection.py  →  P_model - P_market             │
│  kelly.py           →  Fractional Kelly sizing        │
│  bankroll.py        →  Exposure limits, P&L log       │
│                                                       │
│  OUTPUT: bet recommendations with stakes              │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌─── validation/ ───────────────────────────────────────┐
│                                                       │
│  backtest.py          →  Expanding-window engine      │
│  metrics.py           →  Brier, log-loss, ROI, Sharpe │
│  calibration.py       →  Murphy decomposition, PIT    │
│  statistical_tests.py →  Diebold-Mariano, bootstrap   │
│                                                       │
│  OUTPUT: validation report (pass/fail per gate)       │
└───────────────────────────────────────────────────────┘
```

---

## 4. TECHNOLOGY STACK

| Component        | Library                 | Why                                              |
|------------------|-------------------------|--------------------------------------------------|
| Data handling     | pandas, numpy           | Industry standard for tabular data               |
| Bayesian models   | PyMC (v5+)              | Best Python Bayesian library, NUTS + ADVI        |
| MCMC backend      | JAX / NumPyro (alt)     | Faster GPU-accelerated alternative to PyMC       |
| Monte Carlo sim   | numpy + numba           | Vectorized simulation, JIT-compiled inner loops  |
| Optimization      | scipy.optimize          | Cross-validation for λ, ν hyperparameters        |
| Statistics        | scipy.stats, statsmodels| Diebold-Mariano, t-tests, calibration            |
| Visualization     | matplotlib, seaborn     | Publication-quality plots in notebooks           |
| HTTP/API          | requests, tenacity      | API calls with retry logic                       |
| Logging           | logging (stdlib)        | Structured logging with file rotation            |
| Testing           | pytest                  | Unit tests for all modules                       |
| Environment       | conda (Anaconda)        | Reproducible environment with environment.yml    |

---

## 5. KEY DESIGN PRINCIPLES

1. **Strict separation of concerns** — Each module does ONE thing. No model code in data loaders. No plotting in models.

2. **Configuration over hardcoding** — ALL magic numbers live in `config/settings.py`. Hyperparameters, file paths, API keys, thresholds — one place to change everything.

3. **Schema enforcement** — Every DataFrame entering the system is validated against `data/schemas.py`. Bad data fails loudly, not silently.

4. **Immutable data pipeline** — Raw CSVs are NEVER modified. All transformations create new DataFrames. Reproducibility guaranteed.

5. **Model registry** — Every trained model is saved with metadata (training dates, hyperparameters, validation scores). You can always reproduce or roll back.

6. **Expanding-window by default** — The backtesting engine enforces temporal ordering. No future data leakage is possible by construction.

7. **Logging everywhere** — Every module logs what it does. Debug issues by reading logs, not re-running notebooks.

8. **Type hints throughout** — Every function has typed parameters and return values. Self-documenting code.

---

## 6. TRAIN / HOLDOUT SPLIT

```
DATA TIMELINE:
|========= TRAINING =========|==== HOLDOUT ====|=== LIVE ===|
   2019         2020    2021    2022  │  2023      2024  │  2025+
                                      │                  │
                               Train  │  Validate        │  Deploy
                               cutoff │  (never train    │  (real $)
                                      │   on this data)  │
```

- **Training:** 2019–2022 (~200 events). All model parameters learned here.
- **Holdout:** 2023–2024 (~100 events). Strict out-of-sample validation only.
- **Live:** 2025+ (real deployment if holdout gates pass).
- **Rule:** Holdout odds NEVER used for parameter optimization. Only for measuring edge.

---

## 7. CONDA ENVIRONMENT

```yaml
name: golf_model
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pandas>=2.0
  - numpy>=1.24
  - scipy>=1.11
  - scikit-learn>=1.3
  - matplotlib>=3.7
  - seaborn>=0.12
  - requests>=2.31
  - jupyter>=1.0
  - notebook>=7.0
  - ipykernel>=6.25
  - pytest>=7.4
  - pip:
    - pymc>=5.10
    - arviz>=0.17
    - numba>=0.58
    - tenacity>=8.2
    - tqdm>=4.66
```
