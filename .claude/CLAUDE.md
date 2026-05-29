# CLAUDE.md — Golf Betting Model (Method 1)

## What This Project Is

A quantitative golf betting system that predicts PGA Tour tournament outcomes using hierarchical Bayesian strokes-gained (SG) modeling. It estimates player win probabilities via Monte Carlo simulation, detects edges against sportsbook odds, and sizes bets with fractional Kelly Criterion.

**Core prediction:** P(win), P(top5), P(top10), P(make_cut), P(A beats B) for every player in a tournament field.

**Mathematical model:** `Y_{i,r,t} = μ_{i,t} + γ_{c(t)} · δ_i + ε_{i,r,t}` where μ_i is player skill (Bayesian shrinkage), γ_c · δ_i is course-fit interaction, and ε ~ Student-t(ν) for heavy-tailed noise.

**Deployment status:** H2H matchup betting is live (all gates passed). Outright winner betting fails calibration gate — avoid until improved.

## Tech Stack

- **Python 3.11** via Conda (`environment.yml`)
- **Core:** pandas >=2.0, numpy >=1.24, scipy >=1.11, scikit-learn >=1.3
- **Bayesian:** PyMC >=5.10, ArviZ >=0.17
- **Performance:** numba >=0.58 (JIT for Monte Carlo)
- **Data:** DataGolf API (Scratch PLUS tier), Open-Meteo weather API
- **Research:** Anthropic Claude API (Sonnet) for pre-bet news intelligence
- **No database** — all data in CSVs, validated via schemas

## Project Structure

```
golf_data-model/                    # Project root (NOT a git repo)
├── ARCHITECTURE.md                 # 15-pipe architecture spec (read for deep design)
├── CLAUDE.md                       # This file
├── golf_model/                     # Main Python package
│   ├── config/settings.py          # ALL config: paths, API keys, hyperparameters, thresholds
│   ├── data/
│   │   ├── schemas.py              # DataFrame validation contracts (Rounds, Events, Odds, Course, Player)
│   │   ├── loader.py               # CSV loading with schema enforcement
│   │   ├── api_client.py           # DataGolf API client (retry, rate-limit, caching)
│   │   └── weather.py              # Open-Meteo weather integration
│   ├── features/
│   │   ├── pipeline.py             # Master orchestrator — chains all feature steps
│   │   ├── strokes_gained.py       # SG decomposition, field-strength normalization
│   │   ├── time_weighting.py       # EWMA dual decay (rounds + calendar days)
│   │   ├── course_features.py      # 8D γ_c course vectors, Z-score standardization
│   │   └── field_strength.py       # Tournament field quality index
│   ├── models/
│   │   ├── baseline.py             # Phase 1: Weighted SG regression (benchmark)
│   │   ├── bayesian_core.py        # Phase 3: Hierarchical Bayesian (PyMC, NUTS MCMC)
│   │   ├── course_fit.py           # Phase 4: Player x course interactions (δ_i)
│   │   └── priors.py               # Prior specifications
│   ├── simulation/
│   │   ├── monte_carlo.py          # 100K MC iterations, numba-vectorized
│   │   └── tournament.py           # PGA cut rules (top_65_ties), playoff logic
│   ├── betting/
│   │   ├── odds_processing.py      # Overround removal (Shin, proportional, power)
│   │   ├── edge_detection.py       # Edge = P_model / P_market, threshold filters
│   │   ├── kelly.py                # Full & fractional Kelly sizing
│   │   └── bankroll.py             # Bankroll tracker, exposure limits, P&L log
│   ├── validation/
│   │   ├── backtest.py             # Expanding-window backtesting engine
│   │   ├── metrics.py              # Brier, log-loss, ROI, Sharpe, CLV
│   │   ├── calibration.py          # Murphy decomposition, PIT, calibration plots
│   │   └── statistical_tests.py    # Diebold-Mariano, bootstrap, t-tests
│   ├── research/
│   │   ├── __init__.py             # Re-exports NewsResearchAgent
│   │   └── news_agent.py           # Pre-bet news intelligence (Claude API, Google News RSS)
│   ├── utils/
│   │   ├── logger.py               # Structured logging (file + console rotation)
│   │   ├── plotting.py             # Matplotlib/seaborn templates
│   │   └── helpers.py              # Date math, player ID normalization
│   ├── notebooks/                  # Sequential pipeline (01-09), see detailed section below
│   ├── tests/                      # pytest (mostly empty — tests live in notebooks)
│   ├── artifacts/                  # Trained models + metadata JSONs
│   ├── run_pipeline.py             # End-to-end orchestrator (train/backtest/predict)
│   ├── environment.yml             # Conda env spec
│   └── .env                        # GOLF_DATAGOLF_API_KEY, ANTHROPIC_API_KEY (NEVER commit)
├── scripts/                        # Data pulling scripts (00-13, numbered)
├── logs/                           # Script execution logs
└── old/                            # Legacy archived code
```

## Data Architecture

### Sources
- **DataGolf API** (`feeds.datagolf.com`): SG rounds, odds, schedules, player list, skill ratings
- **Open-Meteo API**: Historical weather (free, no auth)

### Data Splits (temporal, strict separation)
| Split | Period | Purpose |
|-------|--------|---------|
| Training | 2017-2022 | Model parameter learning (287K rounds, 3,630 players, 333 events) |
| Holdout | 2023-2024 | Expanding-window validation — gate checks (37K rounds, 190 events) |
| Live | 2025-2026 | Real deployment with real money |

### Raw Data (`golf_model/data/raw/`)
- `sg_rounds_*.csv` — Round-level SG data (PGA, Euro, KFT tours; 36 columns)
- `odds_*.csv` — Historical bookmaker closing odds (DraftKings, Pinnacle)
- `matchup_odds_*.csv` — Head-to-head betting odds
- `schedule_*.csv` — Tournament schedules
- `player_list.csv` — Player metadata
- `holdout/` subdirectory — 2023-2024 data (physically separated)

### Schema Enforcement (`data/schemas.py`)
Every DataFrame is validated on load. Key schemas:
- **RoundsSchema**: dg_id, event_id, round_num, sg_total, sg_ott/app/arg/putt (SG ∈ [-25, 15])
- **OddsSchema**: event_id, dg_id, bookmaker, close_odds, implied_prob
- **EventsSchema**: event_id, event_name, calendar_year
- **CourseSchema**: course_id + 8 feature columns

## Pipeline Flow

```
CSV files → Loader (schema validation)
  → Features: SG decomposition → field-strength adjustment → EWMA time-weighting → course features
    → Models: baseline / Bayesian / course-fit
      → Simulation: 50K-100K Monte Carlo → P(win), P(A>B) per player
        → Betting: devig odds → edge detection → Kelly sizing
          → Research: news intelligence briefing (read-only, human reviews before betting)
            → Validation: Brier, log-loss, ROI, Sharpe, backtest gates
```

## Key Hyperparameters (all in `config/settings.py`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| EWMA_HALF_LIFE_ROUNDS | 25 | 50% weight decay after 25 rounds |
| EWMA_HALF_LIFE_DAYS | 120 | 50% weight decay after 120 days |
| OBSERVATION_DF (ν) | 6.0 | Student-t degrees of freedom (heavy tails) |
| ROUND_CORRELATION_RHO | 0.15 | Within-tournament round correlation |
| N_SIMULATIONS | 100,000 | Monte Carlo iterations (50K for backtest speed) |
| KELLY_FRACTION | 0.25 | Conservative fractional Kelly |
| PROB_TEMPERATURE | 0.35 | Probability sharpening factor |
| MATCHUP_MIN_EDGE | 0.08 | 8% probability edge for H2H bets (grid-search optimal) |
| MIN_EDGE_THRESHOLD | 1.50 | 50% edge for outright bets (high conviction) |
| MIN_BET_PROBABILITY | 0.05 | Skip outrights if model P < 5% |
| COURSE_SG_BLEND | 0.50 | 50% weight on course-specific SG (min 40 rounds) |
| RECENT_FORM_BLEND | 0.40 | 40% weight on last 8 rounds |
| COURSE_FIT_SHRINKAGE (τ) | 0.50 | Bayesian shrinkage on course-fit coefficients |
| T_SCALE_FACTOR | 0.8165 | sqrt((ν-2)/ν) correction for t-distribution |

## Run Modes

```bash
conda activate golf_model

# Train model on 2019-2022 data
python run_pipeline.py --mode train

# Expanding-window backtest on 2023-2024 holdout
python run_pipeline.py --mode backtest

# Live prediction for a specific tournament
python run_pipeline.py --mode predict --event_id 12345
```

## Notebooks — Detailed Workflow

### 01_data_exploration.ipynb — Load & Validate Raw Data
- Loads all CSVs via `DataLoader` with schema validation
- **Coverage:** 287K rounds, 3,630 players, 333 events, tours: PGA (128K), Euro (91K), KFT (68K)
- **Missing data:** SG components ~64.5% missing (older tours), start_hole/teetime ~23.7%
- Plots SG distributions, rounds-per-player histogram
- Loads events (1,228) and odds (82,981 rows; DraftKings + Pinnacle)
- **Imports:** `DataLoader`, `RoundsSchema`, `EventsSchema`, `OddsSchema`, `Settings`

### 02_feature_engineering.ipynb — SG Processing & Time Weighting
- Runs full `FeaturePipeline` on 287K rounds
- **Field strength:** per-event quality adjustment (range -1.074 to +0.968, mean -0.016)
- **EWMA time weighting:** dual half-life (25 rounds / 120 days), exponential decay visualization
- **Course features:** attempted but skipped (no course_features.csv yet)
- **Output:** player_features (EWMA estimates for 3,630 players), field_strength (333 events), rounds_enriched
- **SG decomposition warning:** 63.7% of rounds have |residual| > 0.05 (data quality)
- **Imports:** `FeaturePipeline`, `StrokesGainedProcessor`, `FieldStrengthCalculator`, `TimeWeighter`, `CourseFeatureProcessor`

### 03_baseline_model.ipynb — Phase 1 Benchmark
- Splits: train (2019-2022: 171K rounds, 2,721 players) vs holdout (2023-2024: 6,394 rounds, 718 players)
- Fits `BaselineModel` (EWMA SG averages) on training features
- **Results:** RMSE 2.04, naive baseline RMSE 2.13, **4.4% improvement**, Pearson correlation 0.33
- Saves model artifact via `ModelRegistry` to `artifacts/models/`
- Scatter plot: predicted vs actual SG
- **Imports:** `BaselineModel`, `ModelRegistry`, `FeaturePipeline`

### 04_bayesian_model.ipynb — Phase 3 Hierarchical Bayesian
- Fits `HierarchicalBayesianModel` on training data (5,445 parameters)
- **Priors:** μ_pop ~ N(0,1), τ_pop ~ HalfNormal(1), σ_i ~ HalfNormal(2), ν ~ Exponential(1)+2
- **MCMC:** 2,000 draws × 4 chains, 1,000 warmup, target_accept=0.90
- **Fitting time:** ~4,327 seconds (~1.2 hours)
- **Convergence:** R-hat max=1.0048 (PASS), ESS min=1,071 (PASS), divergences=0 (PASS)
- Trace plots + posterior distributions for top players
- **Imports:** `HierarchicalBayesianModel`, `PriorConfig`, `TimeWeighter`, PyMC, ArviZ

### 05_course_fit.ipynb — Phase 4 Player x Course Interactions
- **Status: SKIPPED** — no course_features.csv available
- Describes 8 course features: length_yards, rough_height_in, green_speed_stimp, wind_exposure, elevation_ft, fairway_width_avg, green_size_sqft, water_hazard_pct
- Would fit `CourseFitModel` with Bayesian ridge (prior N(0, 0.01) on δ_i)
- Leave-one-course-out cross-validation planned
- **Imports:** `CourseFitModel`, `CourseFeatureProcessor`, `DataLoader`

### 06_simulation.ipynb — Monte Carlo Tournament Simulation
- Tests `MonteCarloSimulator` on 30-player synthetic field (50K iterations)
- **Algorithm per sim:** draw abilities → add course-fit → simulate 4 rounds (Student-t noise + ρ=0.15 autocorrelation) → apply cut (top 65+ties) → playoff → record winner
- **Results:** P(win) sums to 1.0000, max SE=0.002226 (converged)
- Favorite: P(win)=0.5475 (synthetic), second: 0.1112
- **Imports:** `MonteCarloSimulator`, `TournamentConfig`, `PlayerAbility`, `SimulationResult`

### 07_betting_integration.ipynb — Odds, Edges & Kelly Sizing
- **Devigging demo:** Shin's method (preferred for golf), proportional fallback, on 10-player field with 25% overround
- **Kelly demo:** 5% edge @ 25:1 → 25% Kelly stake $13.02; 10% edge @ 12:1 → $22.73
- **Outright edge detection:** 6 edges found on synthetic 10-player tournament, total stake $52.28 (1% of $5K bankroll)
- **H2H matchup demo:** Market A 1.85/B 2.05 (2.8% vig), model A 60%, edge 7.4%, Kelly stake $161.76
- **Key insight:** H2H markets (2 outcomes, 2-3% vig) produce many more bets than outrights (150 outcomes, 25% vig)
- **Imports:** `odds_processing`, `EdgeDetector`, `KellyCalculator`, `BankrollTracker`

### 08_full_backtest.ipynb — Expanding-Window Validation (CRITICAL)
- **Data:** training + holdout + 2025-2026 combined (287K rounds, 82K odds)
- **fit_and_predict callback:** For each holdout event:
  1. EWMA skill estimates (25r/120d half-life)
  2. Course-specific SG component weighting (50% blend, min 40 historical rounds)
  3. Recent-form boost (40% of last 8 rounds)
  4. Bayesian course-fit shrinkage (τ=0.50)
  5. t-distribution scale correction (0.8165)
  6. 50K MC simulations with H2H pairwise probabilities
- **Caches** 58 holdout event predictions to pickle for fast re-evaluation
- **Outright results (2023-2024):**
  - 23 events with bets, 28 total bets
  - ROI=84.4%, Sharpe=1.97, MaxDD=11.4%, P&L=+$2,642
  - **WARNING:** Top 3 events = 100% of gains (concentration risk)
  - Model overconfident: probability std 1.88x market std
- **H2H matchup results (2023-2024):**
  - 1,854 bets across 55 events
  - Win rate=56.6%, ROI=6.0%, Sharpe=1.48, MaxDD=27.9%, P&L=+$37,176
  - Avg edge 13.4%, avg odds 1.94, avg stake $336
- **Edge grid search:** MATCHUP_MIN_EDGE=0.08 is optimal (Sharpe-ROI tradeoff)
- **Flat vs Kelly sizing:** Flat $25 safer (MaxDD 10%), Kelly higher absolute P&L ($37K vs $6K)
- **Gate verdicts:**
  - Gate 1 (Calibration): **FAIL** — Brier 0.013 > Market 0.011
  - Gate 2 (Significance): **FAIL** — DM p=0.99, model better in only 18% of events
  - Gate 3 (Betting): **PASS** — ROI/Sharpe/MaxDD all pass
  - H2H gates: **ALL PASS** — sample ≥100, WR>50%, ROI>0%, Sharpe>0.3
- **Imports:** `BacktestEngine`, `BacktestCache`, `EdgeDetector`, `KellyCalculator`, `MonteCarloSimulator`, `TimeWeighter`, `diebold_mariano_test`, `bootstrap_test`

### 09_live_deployment.ipynb — Production Prediction & Betting
- **Weekly production notebook** — run this for each tournament
- Loads ALL historical rounds (no holdout restriction for live)
- Selects target tournament by event_id
- Computes EWMA + course-fit + recent-form (same callback as backtest)
- Runs 50K MC simulation with `compute_h2h=True` for pairwise probabilities
- Loads live sportsbook odds (DraftKings/Pinnacle)
- Edge detection at MATCHUP_MIN_EDGE=0.08
- Fractional Kelly sizing (25%) with exposure caps
- **News research briefing** (Step 6a) — researches recent news about players in identified bets via Google News RSS + Claude API analysis. Read-only: displays injury/form/equipment/personal intel per matchup for human review. Never modifies bets. Cached per player per week (~$0.30/week Claude API cost).
- Logs bets to CSV for P&L tracking
- **Imports:** `DataLoader`, `TimeWeighter`, `MonteCarloSimulator`, `TournamentConfig`, `PlayerAbility`, `EdgeDetector`, `KellyCalculator`, `BankrollTracker`, `NewsResearchAgent`

## Weekly Operational Routine

1. **Pull fresh data:** `python scripts/14_pull_season_data.py --key YOUR_API_KEY`
2. **Open notebook 09** (live_deployment) — run cells 1-7 for predictions & bets
3. **Review news briefing** (Step 6a) — read the research report for each matchup
4. **Place bets** on sportsbook (DraftKings, Pinnacle) — use your judgment informed by both model edges and news briefing
5. **Post-tournament:** settle bets in notebook, update P&L log

## Data Pulling Scripts (`scripts/`)

Numbered 00-14, each pulls from DataGolf API:
- `00` diagnose API, `01` player list, `02` schedule, `03` SG rounds, `04` odds
- `05` validate all data, `06` skill ratings, `07` predictions archive
- `08` matchup odds, `09` course fit, `10` KFT/Euro tours
- `11` holdout matchup odds, `12` 2025 data (legacy), `13` 2026 data (legacy)
- **`14` unified season pull (2025+2026)** — smart incremental, use this for weekly routine

## Validation Gates (3-gate system)

Must pass on holdout (2023-2024) before live deployment:

| Gate | Criterion | Outright Status | H2H Status |
|------|-----------|----------------|------------|
| 1. Calibration | Model Brier < Market Brier | FAIL (0.013 > 0.011) | N/A |
| 2. Significance | Diebold-Mariano p < 0.05 | FAIL (p=0.99) | N/A |
| 3. Betting | ROI > 0%, Sharpe > 0.5, MaxDD < 40% | PASS | PASS |
| H2H: Sample | ≥100 bets | — | PASS (1,854) |
| H2H: Win Rate | >50% | — | PASS (56.6%) |
| H2H: ROI | >0% | — | PASS (6.0%) |
| H2H: Sharpe | >0.3 | — | PASS (1.48) |

**Known risks:**
- Outright P&L concentrated in 3 events (100% of gains) — concentration risk
- Model probability std is 1.88x market std — overconfident on longshots
- H2H matchups are the reliable, diversified edge

## Critical Conventions

1. **All config in `settings.py`** — no magic numbers scattered in code
2. **Raw CSVs are immutable** — never modify files in `data/raw/`
3. **Schema enforcement** — every DataFrame validated on load via `schemas.py`
4. **Expanding-window backtesting only** — no lookahead, no future data leakage
5. **Temporal integrity** — `as_of_date` parameter prevents using future data in features
6. **Model artifacts saved with metadata** — JSON sidecar with training date, hyperparams, scores
7. **Structured logging everywhere** — debug via logs, not re-running notebooks
8. **Type hints throughout** — self-documenting code
9. **Conda environment** — use `environment.yml`, not pip
10. **SG decomposition must sum** — sg_ott + sg_app + sg_arg + sg_putt ≈ sg_total

## Sensitive Files

- `golf_model/.env` — contains `GOLF_DATAGOLF_API_KEY` and `ANTHROPIC_API_KEY` (never commit, never display)
- `golf_model/data/raw/` — proprietary DataGolf data (paid API tier)

## Not Yet Implemented

- `models/model_registry.py` — referenced in ARCHITECTURE.md but not created
- `tests/` — mostly empty (validation lives in notebooks)
- Course features CSV — not populated (Phase 4 course-fit is optional)
- Git repository — project is not version-controlled yet
- News agent: Twitter/X integration — free tier too limited; placeholder for future
- News agent: podcast transcript analysis — potential future enhancement
