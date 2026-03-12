"""Model training and holdout backtesting."""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from datetime import date as calendar_date
import math
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from boatrace_ai.betting import (
    DEFAULT_BETTING_POLICY,
    ROI_FOCUSED_BETTING_POLICY,
    SINGLE_DAY_ROI_BETTING_POLICY,
    build_payout_model,
    evaluate_recommendation_strategy,
    iter_betting_policies,
    normalize_probabilities,
    score_betting_summary,
    select_betting_policy,
)
from boatrace_ai.calibration import apply_probability_calibration, fit_platt_calibrator
from boatrace_ai.collect.history import iter_race_record_paths, load_race_record
from boatrace_ai.features.dataset import FEATURE_COLUMNS, rows_to_matrix


DEFAULT_TIMEZONE = ZoneInfo("Asia/Tokyo")
MAX_TRAINING_DATE_GAP_DAYS = 3
MIN_WALK_FORWARD_TRAIN_RACES = 8
MAX_CALIBRATION_DATES = 60
CONSERVATIVE_SINGLE_DAY_POLICY = {
    **DEFAULT_BETTING_POLICY,
    "candidate_pool_size": 1,
    "min_probability": 0.3,
    "min_market_odds": 70.0,
    "max_market_odds": 120.0,
}


@dataclass(frozen=True)
class TrainingResult:
    model_path: str
    metrics: dict[str, Any]
    train_rows: int
    test_rows: int
    train_dates: list[str]
    test_dates: list[str]
    feature_columns: list[str]
    betting_policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_win_model(
    dataset_path: Path,
    output_dir: Path,
    train_end_date: str | None = None,
    raw_dir: Path | None = None,
    random_state: int = 42,
    min_recommendation_roi: float | None = None,
    min_walk_forward_recommendation_roi: float | None = None,
) -> TrainingResult:
    rows = _load_dataset_rows(dataset_path)
    labeled_rows = [row for row in rows if row.get("is_win") not in ("", None)]
    complete_group_keys = _load_complete_group_keys(raw_dir)
    if complete_group_keys:
        labeled_rows = _filter_rows_by_complete_groups(labeled_rows, complete_group_keys)
    if len(labeled_rows) < 20:
        raise ValueError("Need at least 20 labeled entrant rows to train a model.")

    train_rows, test_rows = _split_rows_by_date(labeled_rows, train_end_date)
    if not train_rows:
        raise ValueError("Training split is empty. Adjust train_end_date or collect more data.")
    policy_rows = _select_recent_training_rows(train_rows)
    feature_columns = FEATURE_COLUMNS
    x_train = np.array(rows_to_matrix(train_rows, feature_columns), dtype=float)
    y_train = np.array([int(row["is_win"]) for row in train_rows], dtype=int)
    model = _fit_model(x_train, y_train, random_state=random_state)
    calibrator, calibration_summary = _fit_probability_calibrator_from_rows(
        train_rows,
        random_state=random_state,
    )
    payout_model = build_payout_model(train_rows)
    odds_index = _load_race_odds_index(raw_dir)
    betting_policy = _derive_betting_policy(policy_rows, odds_index=odds_index, random_state=random_state)

    metrics = {
        "train_positive_rate": round(float(np.mean(y_train)), 6),
        "train_race_count": len({row["race_key"] for row in train_rows}),
    }
    metrics.update(_prefix_calibration_summary(calibration_summary))
    walk_forward_rows = _select_recent_training_rows(labeled_rows)
    walk_forward_summary = _walk_forward_backtest(
        walk_forward_rows,
        odds_index=odds_index,
        random_state=random_state,
        policy=betting_policy,
    )
    if walk_forward_summary:
        metrics.update(
            {
                "walk_forward_folds": walk_forward_summary["folds"],
                "walk_forward_dates": walk_forward_summary["dates"],
                **_prefix_summary_metrics(
                    walk_forward_summary,
                    label_prefix="walk_forward_recommendation",
                ),
            }
        )

    if test_rows:
        x_test = np.array(rows_to_matrix(test_rows, feature_columns), dtype=float)
        y_test = np.array([int(row["is_win"]) for row in test_rows], dtype=int)
        raw_probabilities = model.predict_proba(x_test)[:, 1]
        probabilities = apply_probability_calibration(raw_probabilities, calibrator)
        metrics.update(
            _evaluate_predictions(
                test_rows,
                y_test,
                probabilities,
                betting_probabilities=raw_probabilities,
                payout_model=payout_model,
                betting_policy=betting_policy,
                odds_index=odds_index,
            )
        )

    _assert_training_thresholds(
        metrics,
        min_recommendation_roi=min_recommendation_roi,
        min_walk_forward_recommendation_roi=min_walk_forward_recommendation_roi,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=DEFAULT_TIMEZONE).strftime("%Y%m%dT%H%M%S")
    artifact_path = output_dir / f"win_model_{timestamp}.joblib"
    joblib.dump(
        {
            "model_name": "hist_gradient_boosting_win_v1",
            "trained_at": datetime.now(tz=DEFAULT_TIMEZONE).isoformat(),
            "feature_columns": feature_columns,
            "train_end_date": train_end_date,
            "training_max_gap_days": MAX_TRAINING_DATE_GAP_DAYS,
            "random_state": random_state,
            "min_recommendation_roi": min_recommendation_roi,
            "min_walk_forward_recommendation_roi": min_walk_forward_recommendation_roi,
            "metrics": metrics,
            "payout_model": payout_model,
            "betting_policy": betting_policy,
            "single_day_betting_policy": dict(SINGLE_DAY_ROI_BETTING_POLICY),
            "calibrator": calibrator,
            "calibration_summary": calibration_summary,
            "raw_dir": str(raw_dir) if raw_dir else None,
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "model": model,
        },
        artifact_path,
    )

    return TrainingResult(
        model_path=str(artifact_path),
        metrics=metrics,
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        train_dates=sorted({row["date"] for row in train_rows}),
        test_dates=sorted({row["date"] for row in test_rows}),
        feature_columns=feature_columns,
        betting_policy=betting_policy,
    )


def load_model_artifact(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def find_latest_model(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("win_model_*.joblib"))
    return candidates[-1] if candidates else None


def _load_dataset_rows(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _split_rows_by_date(rows: list[dict[str, Any]], train_end_date: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique_dates = sorted({row["date"] for row in rows})
    if not unique_dates:
        return [], []

    if train_end_date:
        train_rows = [row for row in rows if row["date"] <= train_end_date]
        test_rows = [row for row in rows if row["date"] > train_end_date]
        return train_rows, test_rows

    if len(unique_dates) == 1:
        return rows, []

    split_index = max(1, math.floor(len(unique_dates) * 0.8))
    train_dates = set(unique_dates[:split_index])
    test_dates = set(unique_dates[split_index:])
    return (
        [row for row in rows if row["date"] in train_dates],
        [row for row in rows if row["date"] in test_dates],
    )


def _select_recent_training_rows(
    rows: list[dict[str, Any]],
    *,
    max_gap_days: int = MAX_TRAINING_DATE_GAP_DAYS,
) -> list[dict[str, Any]]:
    unique_dates = sorted({row["date"] for row in rows})
    if len(unique_dates) <= 1:
        return rows

    selected_dates = [unique_dates[-1]]
    for current_date, previous_date in zip(reversed(unique_dates[1:]), reversed(unique_dates[:-1]), strict=True):
        current = calendar_date.fromisoformat(current_date)
        previous = calendar_date.fromisoformat(previous_date)
        if (current - previous).days > max_gap_days:
            break
        selected_dates.append(previous_date)

    selected_set = set(selected_dates)
    return [row for row in rows if row["date"] in selected_set]


def _assert_training_thresholds(
    metrics: dict[str, Any],
    *,
    min_recommendation_roi: float | None,
    min_walk_forward_recommendation_roi: float | None,
) -> None:
    failures: list[str] = []

    if min_recommendation_roi is not None:
        recommendation_roi = metrics.get("recommendation_roi")
        if recommendation_roi is None:
            failures.append("recommendation_roi is unavailable")
        elif float(recommendation_roi) < float(min_recommendation_roi):
            failures.append(
                f"recommendation_roi={recommendation_roi} < required {min_recommendation_roi}"
            )

    if min_walk_forward_recommendation_roi is not None:
        walk_forward_roi = metrics.get("walk_forward_recommendation_roi")
        if walk_forward_roi is None:
            failures.append("walk_forward_recommendation_roi is unavailable")
        elif float(walk_forward_roi) < float(min_walk_forward_recommendation_roi):
            failures.append(
                "walk_forward_recommendation_roi="
                f"{walk_forward_roi} < required {min_walk_forward_recommendation_roi}"
            )

    if failures:
        raise ValueError("Training thresholds failed: " + ", ".join(failures))


def _evaluate_predictions(
    test_rows: list[dict[str, Any]],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    betting_probabilities: np.ndarray | None,
    payout_model: dict[str, Any] | None,
    betting_policy: dict[str, Any],
    odds_index: dict[str, dict[str, float]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "log_loss": round(float(log_loss(y_true, probabilities, labels=[0, 1])), 6),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 6),
    }

    if len(set(y_true.tolist())) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, probabilities)), 6)

    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for row, probability in zip(test_rows, probabilities, strict=True):
        grouped.setdefault(row["race_key"], []).append((row, float(probability)))

    race_count = len(grouped)
    top1_hits = 0
    top3_hits = 0
    win_stake = 0
    win_return = 0
    trifecta_stake = 0
    trifecta_return = 0

    for race_key, entrants in grouped.items():
        entrants.sort(key=lambda item: item[1], reverse=True)
        top_pick = entrants[0][0]
        if int(top_pick["is_win"]) == 1:
            top1_hits += 1
            win_return += int(top_pick.get("win_payout_yen") or 0)
        if any(int(row["is_win"]) == 1 for row, _ in entrants[:3]):
            top3_hits += 1
        win_stake += 100

        predicted_trifecta = _predict_top_trifecta(entrants)
        actual_trifecta_key = entrants[0][0].get("trifecta_key")
        if predicted_trifecta and actual_trifecta_key and predicted_trifecta == actual_trifecta_key:
            trifecta_return += int(entrants[0][0].get("trifecta_payout_yen") or 0)
        trifecta_stake += 100

    metrics.update(
        {
            "race_count": race_count,
            "top1_hit_rate": round(top1_hits / race_count, 6) if race_count else None,
            "winner_in_top3_rate": round(top3_hits / race_count, 6) if race_count else None,
            "win_bet_roi": round((win_return - win_stake) / win_stake, 6) if win_stake else None,
            "trifecta_bet_roi": round((trifecta_return - trifecta_stake) / trifecta_stake, 6) if trifecta_stake else None,
            "win_bet_return": win_return,
            "trifecta_bet_return": trifecta_return,
        }
    )

    recommendation_summary = evaluate_recommendation_strategy(
        _build_race_probability_records(
            test_rows,
            betting_probabilities if betting_probabilities is not None else probabilities,
            odds_index=odds_index,
        ),
        payout_model,
        betting_policy,
    )
    metrics.update(
        {
            "recommendation_bets": recommendation_summary["bets"],
            "recommendation_hits": recommendation_summary["hits"],
            "recommendation_hit_rate": recommendation_summary["hit_rate"],
            "recommendation_roi": recommendation_summary["roi"],
            "recommendation_return": recommendation_summary["return"],
            "recommendation_profit": recommendation_summary["profit"],
            "betting_min_expected_value": betting_policy.get("min_expected_value"),
            "betting_max_per_race": betting_policy.get("max_per_race"),
            "betting_candidate_pool_size": betting_policy.get("candidate_pool_size"),
            "betting_min_probability": betting_policy.get("min_probability"),
            "betting_min_edge": betting_policy.get("min_edge"),
            "betting_min_market_odds": betting_policy.get("min_market_odds"),
            "betting_max_market_odds": betting_policy.get("max_market_odds"),
        }
    )
    return metrics


def _fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=5,
        max_iter=250,
        min_samples_leaf=10,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model


def _derive_betting_policy(
    train_rows: list[dict[str, Any]],
    *,
    odds_index: dict[str, dict[str, float]],
    random_state: int,
) -> dict[str, Any]:
    unique_dates = sorted({row["date"] for row in train_rows})
    if len(unique_dates) < 2:
        return dict(CONSERVATIVE_SINGLE_DAY_POLICY)

    walk_forward_policy = _select_betting_policy_walk_forward(
        train_rows,
        odds_index=odds_index,
        random_state=random_state,
    )
    if walk_forward_policy:
        return walk_forward_policy
    if len(unique_dates) >= 3:
        return dict(ROI_FOCUSED_BETTING_POLICY)

    fit_rows, validation_rows = _split_rows_for_policy_selection(train_rows)
    if not fit_rows or not validation_rows:
        return dict(ROI_FOCUSED_BETTING_POLICY)

    x_fit = np.array(rows_to_matrix(fit_rows, FEATURE_COLUMNS), dtype=float)
    y_fit = np.array([int(row["is_win"]) for row in fit_rows], dtype=int)
    selector_model = _fit_model(x_fit, y_fit, random_state=random_state)

    x_validation = np.array(rows_to_matrix(validation_rows, FEATURE_COLUMNS), dtype=float)
    validation_probabilities = selector_model.predict_proba(x_validation)[:, 1]
    validation_races = _build_race_probability_records(
        validation_rows,
        validation_probabilities,
        odds_index=odds_index,
    )
    payout_model = build_payout_model(fit_rows)
    return select_betting_policy(validation_races, payout_model)


def _select_betting_policy_walk_forward(
    rows: list[dict[str, Any]],
    *,
    odds_index: dict[str, dict[str, float]],
    random_state: int,
) -> dict[str, Any] | None:
    unique_dates = sorted({row["date"] for row in rows})
    if len(unique_dates) < 3:
        return None

    fold_contexts = _prepare_walk_forward_contexts(
        rows,
        odds_index=odds_index,
        random_state=random_state,
        dates=unique_dates,
    )
    if not fold_contexts:
        return None

    best_policy: dict[str, Any] | None = None
    best_score = (-math.inf,)

    for policy in iter_betting_policies():
        summary = _summarize_walk_forward_contexts(fold_contexts, policy)
        if not summary or summary.get("bets", 0) == 0 or summary.get("roi") is None:
            continue
        score = score_betting_summary(summary)
        if score > best_score:
            best_score = score
            best_policy = policy

    if best_policy is None or best_score[2] <= 0:
        return None

    return best_policy


def _split_rows_for_policy_selection(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique_dates = sorted({row["date"] for row in rows})
    if len(unique_dates) >= 2:
        split_index = max(1, math.floor(len(unique_dates) * 0.8))
        fit_dates = set(unique_dates[:split_index])
        validation_dates = set(unique_dates[split_index:])
        if validation_dates:
            return (
                [row for row in rows if row["date"] in fit_dates],
                [row for row in rows if row["date"] in validation_dates],
            )

    race_keys = sorted({row["race_key"] for row in rows})
    if len(race_keys) < 2:
        return rows, []

    split_index = max(1, math.floor(len(race_keys) * 0.8))
    fit_keys = set(race_keys[:split_index])
    validation_keys = set(race_keys[split_index:])
    return (
        [row for row in rows if row["race_key"] in fit_keys],
        [row for row in rows if row["race_key"] in validation_keys],
    )


def _walk_forward_backtest(
    rows: list[dict[str, Any]],
    *,
    odds_index: dict[str, dict[str, float]],
    random_state: int,
    policy: dict[str, Any],
    dates: list[str] | None = None,
) -> dict[str, Any] | None:
    fold_contexts = _prepare_walk_forward_contexts(
        rows,
        odds_index=odds_index,
        random_state=random_state,
        dates=dates,
    )
    return _summarize_walk_forward_contexts(fold_contexts, policy)


def _prepare_walk_forward_contexts(
    rows: list[dict[str, Any]],
    *,
    odds_index: dict[str, dict[str, float]],
    random_state: int,
    dates: list[str] | None = None,
) -> list[dict[str, Any]]:
    unique_dates = dates or sorted({row["date"] for row in rows})
    if len(unique_dates) < 2:
        return []

    fold_contexts: list[dict[str, Any]] = []
    for test_date in unique_dates[1:]:
        fit_rows = [row for row in rows if row["date"] < test_date]
        validation_rows = [row for row in rows if row["date"] == test_date]
        fit_race_count = len({row["race_key"] for row in fit_rows})
        if not fit_rows or not validation_rows or fit_race_count < MIN_WALK_FORWARD_TRAIN_RACES:
            continue

        x_fit = np.array(rows_to_matrix(fit_rows, FEATURE_COLUMNS), dtype=float)
        y_fit = np.array([int(row["is_win"]) for row in fit_rows], dtype=int)
        selector_model = _fit_model(x_fit, y_fit, random_state=random_state)
        x_validation = np.array(rows_to_matrix(validation_rows, FEATURE_COLUMNS), dtype=float)
        validation_probabilities = selector_model.predict_proba(x_validation)[:, 1]
        validation_races = _build_race_probability_records(
            validation_rows,
            validation_probabilities,
            odds_index=odds_index,
        )
        fold_contexts.append(
            {
                "fold_date": test_date,
                "validation_races": validation_races,
                "payout_model": build_payout_model(fit_rows),
            }
        )

    return fold_contexts


def _summarize_walk_forward_contexts(
    fold_contexts: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any] | None:
    if not fold_contexts:
        return None

    fold_summaries: list[dict[str, Any]] = []
    for fold_context in fold_contexts:
        summary = evaluate_recommendation_strategy(
            fold_context["validation_races"],
            fold_context["payout_model"],
            policy,
        )
        summary["fold_date"] = fold_context["fold_date"]
        fold_summaries.append(summary)

    if not fold_summaries:
        return None

    aggregate = _aggregate_strategy_summaries(fold_summaries)
    aggregate["folds"] = len(fold_summaries)
    aggregate["dates"] = [summary["fold_date"] for summary in fold_summaries]
    return aggregate


def _aggregate_strategy_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    total_bets = sum(int(summary.get("bets") or 0) for summary in summaries)
    total_hits = sum(int(summary.get("hits") or 0) for summary in summaries)
    total_investment = sum(int(summary.get("investment") or 0) for summary in summaries)
    total_return = sum(int(summary.get("return") or 0) for summary in summaries)
    profit = total_return - total_investment
    roi = (profit / total_investment) if total_investment else None
    daily_results = [
        daily
        for summary in summaries
        for daily in summary.get("daily_results", [])
    ]
    daily_rois = [float(item["roi"]) for item in daily_results if item.get("roi") is not None]
    positive_days = sum(1 for value in daily_rois if value > 0)
    hit_rate = (total_hits / total_bets) if total_bets else None

    return {
        "bets": total_bets,
        "hits": total_hits,
        "hit_rate": round(hit_rate, 6) if hit_rate is not None else None,
        "return": total_return,
        "profit": profit,
        "roi": round(roi, 6) if roi is not None else None,
        "days": len(daily_results),
        "betting_days": len(daily_rois),
        "positive_days": positive_days,
        "daily_roi_floor": round(min(daily_rois), 6) if daily_rois else None,
        "daily_roi_median": round(float(np.median(daily_rois)), 6) if daily_rois else None,
    }


def _prefix_summary_metrics(summary: dict[str, Any], *, label_prefix: str) -> dict[str, Any]:
    return {
        f"{label_prefix}_bets": summary.get("bets"),
        f"{label_prefix}_hits": summary.get("hits"),
        f"{label_prefix}_hit_rate": summary.get("hit_rate"),
        f"{label_prefix}_return": summary.get("return"),
        f"{label_prefix}_profit": summary.get("profit"),
        f"{label_prefix}_roi": summary.get("roi"),
        f"{label_prefix}_positive_days": summary.get("positive_days"),
        f"{label_prefix}_day_roi_floor": summary.get("daily_roi_floor"),
        f"{label_prefix}_day_roi_median": summary.get("daily_roi_median"),
    }


def _fit_probability_calibrator_from_rows(
    rows: list[dict[str, Any]],
    *,
    random_state: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    unique_dates = sorted({row["date"] for row in rows})
    if len(unique_dates) < 3:
        return None, {
            "method": "platt_logit",
            "rows": 0,
            "accepted": False,
            "reason": "insufficient_dates",
            "dates": [],
        }

    calibration_dates = _select_calibration_dates(unique_dates)
    raw_probabilities: list[float] = []
    labels: list[int] = []
    dates: list[str] = []

    for calibration_date in calibration_dates:
        fit_rows = [row for row in rows if row["date"] < calibration_date]
        calibration_rows = [row for row in rows if row["date"] == calibration_date]
        fit_race_count = len({row["race_key"] for row in fit_rows})
        if not fit_rows or not calibration_rows or fit_race_count < MIN_WALK_FORWARD_TRAIN_RACES:
            continue

        x_fit = np.array(rows_to_matrix(fit_rows, FEATURE_COLUMNS), dtype=float)
        y_fit = np.array([int(row["is_win"]) for row in fit_rows], dtype=int)
        model = _fit_model(x_fit, y_fit, random_state=random_state)
        x_calibration = np.array(rows_to_matrix(calibration_rows, FEATURE_COLUMNS), dtype=float)
        raw_probabilities.extend(model.predict_proba(x_calibration)[:, 1].tolist())
        labels.extend(int(row["is_win"]) for row in calibration_rows)
        dates.append(calibration_date)

    calibrator, summary = fit_platt_calibrator(
        raw_probabilities,
        labels,
        random_state=random_state,
    )
    summary["dates"] = dates
    return calibrator, summary


def _select_calibration_dates(unique_dates: list[str]) -> list[str]:
    candidate_dates = unique_dates[1:]
    if len(candidate_dates) <= MAX_CALIBRATION_DATES:
        return candidate_dates
    return candidate_dates[-MAX_CALIBRATION_DATES:]


def _prefix_calibration_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "calibration_method": summary.get("method"),
        "calibration_rows": summary.get("rows"),
        "calibration_accepted": summary.get("accepted"),
        "calibration_reason": summary.get("reason"),
        "calibration_dates": summary.get("dates"),
        "calibration_oof_raw_log_loss": summary.get("raw_log_loss"),
        "calibration_oof_log_loss": summary.get("calibrated_log_loss"),
        "calibration_oof_raw_brier_score": summary.get("raw_brier_score"),
        "calibration_oof_brier_score": summary.get("calibrated_brier_score"),
    }


def _build_race_probability_records(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    *,
    odds_index: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row, probability in zip(rows, probabilities, strict=True):
        race_key = row["race_key"]
        record = grouped.setdefault(
            race_key,
            {
                "race_key": race_key,
                "date": row["date"],
                "venue_code": f"{int(float(row['venue_code'])):02d}",
                "venue_name": row["venue_name"],
                "race_no": int(float(row["race_no"])),
                "lane_probabilities": {},
                "odds_map": odds_index.get(race_key, {}),
                "actual_trifecta_key": row.get("trifecta_key"),
                "actual_trifecta_payout_yen": _parse_int(row.get("trifecta_payout_yen")),
            },
        )
        record["lane_probabilities"][int(float(row["lane"]))] = float(probability)

    race_records: list[dict[str, Any]] = []
    for record in grouped.values():
        lanes = sorted(record["lane_probabilities"])
        normalized = normalize_probabilities([record["lane_probabilities"][lane] for lane in lanes])
        record["lane_probabilities"] = {
            lane: probability for lane, probability in zip(lanes, normalized, strict=True)
        }
        race_records.append(record)
    return race_records


def _predict_top_trifecta(entrants: list[tuple[dict[str, Any], float]]) -> str | None:
    weights = [max(probability, 1e-6) for _, probability in entrants]
    lanes = [int(float(row["lane"])) for row, _ in entrants]
    total = sum(weights)
    best_key: str | None = None
    best_probability = -1.0

    for first_index, first_lane in enumerate(lanes):
        for second_index, second_lane in enumerate(lanes):
            if second_index == first_index:
                continue
            for third_index, third_lane in enumerate(lanes):
                if third_index in {first_index, second_index}:
                    continue
                remaining_after_first = total - weights[first_index]
                remaining_after_second = remaining_after_first - weights[second_index]
                if remaining_after_first <= 0 or remaining_after_second <= 0:
                    continue
                probability = (
                    weights[first_index] / total
                    * weights[second_index] / remaining_after_first
                    * weights[third_index] / remaining_after_second
                )
                if probability > best_probability:
                    best_probability = probability
                    best_key = f"{first_lane}-{second_lane}-{third_lane}"

    return best_key


def _parse_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _load_race_odds_index(raw_dir: Path | None) -> dict[str, dict[str, float]]:
    if raw_dir is None or not raw_dir.exists():
        return {}

    odds_index: dict[str, dict[str, float]] = {}
    for path in iter_race_record_paths(raw_dir):
        record = load_race_record(path)
        race_key = f"{record['date']}_{record['venue_code']}_{int(record['race_no']):02d}"
        odds_map = record.get("trifecta_odds") or {}
        if odds_map:
            odds_index[race_key] = {str(key): float(value) for key, value in odds_map.items()}
    return odds_index


def _load_complete_group_keys(raw_dir: Path | None) -> set[tuple[str, str]]:
    if raw_dir is None or not raw_dir.exists():
        return set()

    venue_day_races: dict[tuple[str, str], set[int]] = {}
    expected_count_by_venue: dict[str, int] = {}

    for path in iter_race_record_paths(raw_dir):
        record = load_race_record(path)
        if not record.get("card") or not record.get("result"):
            continue
        date = str(record["date"])
        venue_code = str(record["venue_code"]).zfill(2)
        race_no = int(record["race_no"])
        key = (date, venue_code)
        venue_day_races.setdefault(key, set()).add(race_no)
        expected_count_by_venue[venue_code] = max(
            expected_count_by_venue.get(venue_code, 0),
            len(venue_day_races[key]),
        )

    complete_groups: set[tuple[str, str]] = set()
    for (date, venue_code), race_numbers in venue_day_races.items():
        expected_count = expected_count_by_venue.get(venue_code, 0)
        if expected_count <= 0:
            continue
        if len(race_numbers) != expected_count:
            continue
        complete_groups.add((date, venue_code))

    return complete_groups


def _filter_rows_by_complete_groups(
    rows: list[dict[str, Any]],
    complete_group_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    if not complete_group_keys:
        return rows

    filtered: list[dict[str, Any]] = []
    for row in rows:
        venue_code = f"{int(float(row['venue_code'])):02d}"
        key = (str(row["date"]), venue_code)
        if key in complete_group_keys:
            filtered.append(row)
    return filtered
