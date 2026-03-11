# boatrace-ai

Public starter repository for building a boat race prediction pipeline.

This repository is intentionally scoped as a standalone project rather than a subdirectory inside `coconala-tools`.
The expected workflow is:

1. Collect race, player, weather, and result data.
2. Build training datasets and feature tables.
3. Train baseline models for win/place probability.
4. Evaluate model quality and backtest betting rules.
5. Run daily prediction jobs with versioned outputs.

## Current status

Current repository includes:

- a package layout under `src/boatrace_ai/`
- a JSON config template under `configs/`
- architecture and roadmap documents under `docs/`
- a working `collect` command that stores one-race-per-file raw JSON from official pages
- a working `build-dataset` command that expands raw race records into entrant-level rows
- a working `train` command that fits a holdout-tested gradient boosting win model, learns a betting policy, and saves artifacts
- out-of-fold probability calibration that is applied only when it improves training-side calibration metrics
- a working `backtest` command that runs holdout evaluation with flat, Kelly, and capped-Kelly bankroll simulation
- a working `predict` command that uses the latest trained model when available, otherwise falls back to the baseline scorer
- ROI-oriented trifecta recommendations that use official live `odds3t` when available and historical payout priors as fallback
- `note-morning` / `note-evening` commands that turn prediction and verification JSON into note-style article HTML
- parser, dataset, and training tests that run without live network access

## Project layout

```text
boatrace-ai/
├── configs/
├── docs/
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
├── src/
│   └── boatrace_ai/
│       ├── collect/
│       ├── evaluate/
│       ├── features/
│       ├── predict/
│       ├── train/
│       ├── __init__.py
│       ├── __main__.py
│       └── cli.py
└── tests/
```

## Planning docs

- `docs/implementation-plan.md`
  Current implementation, corrected assumptions, target architecture, and roadmap.
- `docs/source-registry.md`
  Source-by-source collection policy, cadence, and usage notes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## CLI

```bash
python -m boatrace_ai collect --start-date 2026-03-08 --end-date 2026-03-09 --venue 24
python -m boatrace_ai build-dataset --input-dir data/raw --output-dir data/processed
python -m boatrace_ai train --dataset-path data/processed/entrants.csv --raw-dir data/raw --train-end-date 2026-03-08
python -m boatrace_ai backtest --dataset-path data/processed/entrants.csv --raw-dir data/raw --train-end-date 2026-03-09 --bankroll-mode all --kelly-cap-fraction 0.05 --max-bet-bankroll-fraction 0.02 --max-race-exposure-fraction 0.15
python -m boatrace_ai predict --race-date 2026-03-10 --min-probability 0.12
python -m boatrace_ai predict --race-date 2026-03-10 --venue 24 --race-no 12
python -m boatrace_ai note-morning --race-date 2026-03-10
python -m boatrace_ai note-evening --race-date 2026-03-09
```

`collect` stores raw JSON under `data/raw/YYYYMMDD/`, including `beforeinfo`, `result`, and `trifecta_odds` when available.
`build-dataset` writes entrant-level rows to `data/processed/entrants.csv`.
`train` saves a `joblib` artifact under `artifacts/models/` together with holdout metrics, walk-forward backtest metrics, payout priors, historical odds-aware backtest inputs, and the selected betting policy.
`backtest` saves a JSON file under `artifacts/backtests/` and reports holdout metrics plus bankroll simulation for `flat`, `kelly`, and `kelly_capped`.
`predict` saves a JSON file under `artifacts/predictions/` and prints the top win candidates, leading trifecta combinations, and the highest-EV bets.
`note-morning` and `note-evening` write HTML and title text files under `artifacts/note/`.

The current trained path uses official race-card features plus `beforeinfo` features such as exhibition time, tilt, adjusted weight, start display ST, and weather.
The win model is a `HistGradientBoostingClassifier`, and trifecta candidates are derived from entrant win weights with a Plackett-Luce style ranking step.
The training pipeline fits an out-of-fold Platt calibrator when it improves Brier score and log loss, and the saved artifact reuses that calibrator for prediction.
ROI-oriented recommendation logic continues to use raw model ranking weights so calibration does not collapse bet coverage.
Expected value is computed from official live trifecta odds when that page is available; otherwise the system falls back to smoothed historical payout estimates by venue and combination.
When the historical data contains large date gaps, model fitting still uses all complete venue-days, but betting-policy selection and walk-forward checks focus on the most recent contiguous block.
Venue-days that are missing races are treated as incomplete and are excluded from training/backtest evaluation until the missing raw files are collected.
The betting policy now filters on `min_expected_value`, `min_probability`, and `min_edge`, and the search includes single-ticket candidate pools for higher-ROI, lower-volume betting.
The backtest path now includes bankroll simulation for flat staking, fractional Kelly staking, capped Kelly, optional daily stop-loss / take-profit, and optional race-level exposure caps.
The default risk template keeps Kelly-style staking conservative with `kelly_cap_fraction=0.05`, `max_bet_bankroll_fraction=0.02`, and `max_race_exposure_fraction=0.15`.

Items such as official bulk-download ingestion, external weather/tide joins, model serving APIs, UI, schedulers, and monitoring are target architecture items documented in `docs/implementation-plan.md`; they are not all implemented in the current codebase.

## Prediction Output

Prediction JSON now contains:

- `races`: full entrant/trifecta prediction details
- `race_predictions`: simplified race-level view for downstream publishing
- `recommendations`: ROI-ranked trifecta bets with probability, payout estimate, and expected value
- `recommendations[*].edge`: model probability minus market implied probability
- `betting_policy`: threshold and per-race cap used to select bets
- `model_metrics`: holdout metrics plus walk-forward recommendation ROI embedded from the trained artifact

## note Commands

`note-morning` reads the latest prediction file for a date and generates a morning article focused on expected ROI.
`note-evening` re-fetches official race results, verifies recommended bets, updates cumulative ROI, and generates a review article plus verification JSON.

## Suggested roadmap

### Phase 1

- Decide data source policy and legal constraints.
- Normalize race-level, racer-level, and venue-level tables.
- Define the target labels for 1st place, top-3, and trifecta ranking tasks.

### Phase 2

- Build a baseline feature store.
- Train a simple gradient boosting model.
- Track offline metrics by venue, weather, and race class.

### Phase 3

- Add calibration, richer ranking models, and bankroll-aware odds filtering.
- Backtest bet selection rules with bankroll constraints.
- Automate scheduled collection and daily inference.
