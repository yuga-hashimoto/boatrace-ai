"""Holdout backtest helpers with bankroll simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from boatrace_ai.betting import (
    BET_UNIT_YEN,
    build_payout_model,
    compare_bankroll_strategies,
    merge_betting_policy,
)
from boatrace_ai.calibration import apply_probability_calibration
from boatrace_ai.features.dataset import FEATURE_COLUMNS, rows_to_matrix
from boatrace_ai.train.model import (
    MAX_POLICY_SELECTION_DATES,
    _build_race_probability_records,
    _derive_betting_policy,
    _evaluate_predictions,
    _filter_rows_by_complete_groups,
    _fit_exacta_model,
    _fit_model,
    _load_complete_group_keys,
    _load_dataset_rows,
    _load_race_odds_index,
    _prefix_calibration_summary,
    _fit_probability_calibrator_from_rows,
    _select_policy_training_rows,
    _split_rows_by_date,
)
from boatrace_ai.trifecta import (
    EXACTA_FEATURE_COLUMNS,
    attach_win_probability_features,
    predict_staged_trifecta_probability_maps,
)


@dataclass(frozen=True)
class BacktestResult:
    train_rows: int
    test_rows: int
    train_dates: list[str]
    test_dates: list[str]
    feature_columns: list[str]
    betting_policy: dict[str, Any]
    metrics: dict[str, Any]
    bankroll: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_holdout_backtest(
    *,
    dataset_path: Path,
    raw_dir: Path | None,
    train_end_date: str | None,
    random_state: int = 42,
    bankroll_mode: str = "both",
    starting_bankroll_yen: int = 10_000,
    flat_bet_yen: int = BET_UNIT_YEN,
    kelly_fraction: float = 0.25,
    kelly_cap_fraction: float | None = 0.1,
    max_bet_yen: int | None = None,
    max_bet_bankroll_fraction: float | None = None,
    max_daily_bets: int | None = None,
    max_daily_investment_yen: int | None = None,
    max_race_exposure_fraction: float | None = None,
    daily_stop_loss_yen: int | None = None,
    daily_take_profit_yen: int | None = None,
    betting_policy_override: dict[str, Any] | None = None,
    clear_derived_filters: bool = False,
) -> BacktestResult:
    rows = _load_dataset_rows(dataset_path)
    labeled_rows = [row for row in rows if row.get("is_win") not in ("", None)]
    complete_group_keys = _load_complete_group_keys(raw_dir)
    if complete_group_keys:
        labeled_rows = _filter_rows_by_complete_groups(labeled_rows, complete_group_keys)
    if len(labeled_rows) < 20:
        raise ValueError("Need at least 20 labeled entrant rows to run a backtest.")

    train_rows, test_rows = _split_rows_by_date(labeled_rows, train_end_date)
    if not train_rows or not test_rows:
        raise ValueError("Backtest requires both train and test splits.")

    feature_columns = FEATURE_COLUMNS
    x_train = np.array(rows_to_matrix(train_rows, feature_columns), dtype=float)
    y_train = np.array([int(row["is_win"]) for row in train_rows], dtype=int)
    model = _fit_model(x_train, y_train, random_state=random_state)
    train_raw_probabilities = model.predict_proba(x_train)[:, 1]
    train_rows_with_win_features = attach_win_probability_features(
        train_rows,
        train_raw_probabilities.tolist(),
    )
    exacta_model = _fit_exacta_model(train_rows_with_win_features, random_state=random_state)
    calibrator, calibration_summary = _fit_probability_calibrator_from_rows(
        train_rows,
        random_state=random_state,
    )
    payout_model = build_payout_model(train_rows)
    odds_index = _load_race_odds_index(raw_dir)

    x_test = np.array(rows_to_matrix(test_rows, feature_columns), dtype=float)
    y_test = np.array([int(row["is_win"]) for row in test_rows], dtype=int)
    raw_probabilities = model.predict_proba(x_test)[:, 1]
    test_rows_with_win_features = attach_win_probability_features(
        test_rows,
        raw_probabilities.tolist(),
    )
    probabilities = apply_probability_calibration(
        raw_probabilities,
        calibrator,
    )
    trifecta_probability_maps = (
        predict_staged_trifecta_probability_maps(
            test_rows_with_win_features,
            exacta_model=exacta_model,
            exacta_feature_columns=list(EXACTA_FEATURE_COLUMNS),
        )
        if exacta_model is not None
        else None
    )
    if betting_policy_override and clear_derived_filters:
        betting_policy = {}
    else:
        policy_rows = _select_policy_training_rows(
            train_rows,
            evaluation_rows=test_rows,
            max_dates=MAX_POLICY_SELECTION_DATES,
        )
        betting_policy = _derive_betting_policy(
            policy_rows,
            odds_index=odds_index,
            random_state=random_state,
        )
    betting_policy = merge_betting_policy(
        betting_policy,
        betting_policy_override,
        clear_derived_filters=clear_derived_filters,
    )
    race_records = _build_race_probability_records(
        test_rows_with_win_features,
        raw_probabilities,
        odds_index=odds_index,
        trifecta_probability_maps=trifecta_probability_maps,
    )

    metrics = _evaluate_predictions(
        test_rows_with_win_features,
        y_test,
        probabilities,
        betting_probabilities=raw_probabilities,
        trifecta_probability_maps=trifecta_probability_maps,
        payout_model=payout_model,
        betting_policy=betting_policy,
        odds_index=odds_index,
    )
    metrics.update(_prefix_calibration_summary(calibration_summary))
    bankroll = compare_bankroll_strategies(
        race_records,
        payout_model,
        betting_policy,
        starting_bankroll_yen=starting_bankroll_yen,
        flat_bet_yen=flat_bet_yen,
        kelly_fraction=kelly_fraction,
        kelly_cap_fraction=kelly_cap_fraction,
        max_bet_yen=max_bet_yen,
        max_bet_bankroll_fraction=max_bet_bankroll_fraction,
        max_daily_bets=max_daily_bets,
        max_daily_investment_yen=max_daily_investment_yen,
        max_race_exposure_fraction=max_race_exposure_fraction,
        daily_stop_loss_yen=daily_stop_loss_yen,
        daily_take_profit_yen=daily_take_profit_yen,
        mode=bankroll_mode,
    )

    return BacktestResult(
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        train_dates=sorted({row["date"] for row in train_rows}),
        test_dates=sorted({row["date"] for row in test_rows}),
        feature_columns=feature_columns,
        betting_policy=betting_policy,
        metrics=metrics,
        bankroll=bankroll,
    )
