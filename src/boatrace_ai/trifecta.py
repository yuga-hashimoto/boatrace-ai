"""Direct trifecta feature building and probability helpers."""

from __future__ import annotations

import itertools
from typing import Any


TRIFECTA_CONTEXT_COLUMNS = [
    "venue_code",
    "race_no",
    "temperature_c",
    "wind_speed_mps",
    "water_temperature_c",
    "wave_height_cm",
    "wind_direction_code",
]
TRIFECTA_ENTRANT_COLUMNS = [
    "lane",
    "grade_code",
    "age",
    "weight_kg",
    "f_count",
    "l_count",
    "average_start_timing",
    "national_win_rate",
    "national_2ren_rate",
    "national_3ren_rate",
    "local_win_rate",
    "local_2ren_rate",
    "local_3ren_rate",
    "motor_2ren_rate",
    "motor_3ren_rate",
    "boat_2ren_rate",
    "boat_3ren_rate",
    "display_weight_kg",
    "exhibition_time",
    "tilt",
    "adjusted_weight_kg",
    "start_display_st",
    "parts_exchange_count",
    "has_propeller_note",
    "model_win_score",
    "model_win_probability",
    "model_win_rank",
]
TRIFECTA_DIFF_COLUMNS = [
    "national_win_rate",
    "local_win_rate",
    "motor_2ren_rate",
    "exhibition_time",
    "start_display_st",
    "average_start_timing",
]
TRIFECTA_FEATURE_COLUMNS = (
    TRIFECTA_CONTEXT_COLUMNS
    + [f"{prefix}_{column}" for prefix in ("first", "second", "third") for column in TRIFECTA_ENTRANT_COLUMNS]
    + [
        f"{left}_minus_{right}_{column}"
        for left, right in (("first", "second"), ("first", "third"), ("second", "third"))
        for column in TRIFECTA_DIFF_COLUMNS
    ]
    + [
        "lane_sum",
        "lane_span",
        "is_first_inside",
        "is_second_inside",
        "is_third_inside",
    ]
)
EXACTA_CONTEXT_COLUMNS = list(TRIFECTA_CONTEXT_COLUMNS)
EXACTA_ENTRANT_COLUMNS = list(TRIFECTA_ENTRANT_COLUMNS)
EXACTA_DIFF_COLUMNS = list(TRIFECTA_DIFF_COLUMNS)
EXACTA_FEATURE_COLUMNS = (
    EXACTA_CONTEXT_COLUMNS
    + [f"{prefix}_{column}" for prefix in ("first", "second") for column in EXACTA_ENTRANT_COLUMNS]
    + [
        f"first_minus_second_{column}"
        for column in EXACTA_DIFF_COLUMNS
    ]
    + [
        "lane_sum",
        "lane_gap",
        "is_first_inside",
        "is_second_inside",
    ]
)


def build_trifecta_examples(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["race_key"]), []).append(row)

    examples: list[dict[str, Any]] = []
    for race_rows in grouped.values():
        examples.extend(build_trifecta_examples_from_race_rows(race_rows))
    return examples


def build_exacta_examples(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["race_key"]), []).append(row)

    examples: list[dict[str, Any]] = []
    for race_rows in grouped.values():
        examples.extend(build_exacta_examples_from_race_rows(race_rows))
    return examples


def attach_win_probability_features(
    rows: list[dict[str, Any]],
    raw_probabilities: list[float],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row, raw_probability in zip(rows, raw_probabilities, strict=True):
        grouped.setdefault(str(row["race_key"]), []).append((int(float(row["lane"])), float(raw_probability)))

    lane_features: dict[str, dict[int, dict[str, float]]] = {}
    for race_key, pairs in grouped.items():
        sorted_pairs = sorted(pairs, key=lambda item: item[0])
        normalized = _normalize_probabilities([probability for _, probability in sorted_pairs])
        ranked = sorted(sorted_pairs, key=lambda item: (-item[1], item[0]))
        rank_map = {lane: index for index, (lane, _) in enumerate(ranked, start=1)}
        lane_features[race_key] = {
            lane: {
                "model_win_score": probability,
                "model_win_probability": normalized_probability,
                "model_win_rank": float(rank_map[lane]),
            }
            for (lane, probability), normalized_probability in zip(sorted_pairs, normalized, strict=True)
        }

    enriched: list[dict[str, Any]] = []
    for row, raw_probability in zip(rows, raw_probabilities, strict=True):
        race_key = str(row["race_key"])
        lane = int(float(row["lane"]))
        updates = lane_features.get(race_key, {}).get(
            lane,
            {
                "model_win_score": float(raw_probability),
                "model_win_probability": float(raw_probability),
                "model_win_rank": float("nan"),
            },
        )
        enriched.append({**row, **updates})
    return enriched


def build_trifecta_examples_from_race_rows(
    race_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(race_rows) < 3:
        return []

    ordered_rows = sorted(race_rows, key=lambda row: int(float(row["lane"])))
    lanes = [int(float(row["lane"])) for row in ordered_rows]
    row_by_lane = {int(float(row["lane"])): row for row in ordered_rows}
    template = ordered_rows[0]
    actual_trifecta_key = str(template.get("trifecta_key") or "").strip() or None
    trifecta_payout_yen = _parse_int(template.get("trifecta_payout_yen"))

    examples: list[dict[str, Any]] = []
    for order in itertools.permutations(lanes, 3):
        first_lane, second_lane, third_lane = order
        first = row_by_lane[first_lane]
        second = row_by_lane[second_lane]
        third = row_by_lane[third_lane]
        example = {
            "race_key": str(template["race_key"]),
            "date": str(template["date"]),
            "venue_code": _normalize_venue_code(template.get("venue_code")),
            "venue_name": str(template.get("venue_name") or ""),
            "race_no": int(float(template["race_no"])),
            "combination": f"{first_lane}-{second_lane}-{third_lane}",
            "actual_trifecta_key": actual_trifecta_key,
            "actual_trifecta_payout_yen": trifecta_payout_yen,
            "is_target": 1 if actual_trifecta_key and actual_trifecta_key == f"{first_lane}-{second_lane}-{third_lane}" else 0,
        }
        for column in TRIFECTA_CONTEXT_COLUMNS:
            example[column] = _float_value(template.get(column))

        for prefix, entrant in (("first", first), ("second", second), ("third", third)):
            for column in TRIFECTA_ENTRANT_COLUMNS:
                example[f"{prefix}_{column}"] = _float_value(entrant.get(column))

        for left_name, left_row in (("first", first), ("second", second)):
            for right_name, right_row in (("second", second), ("third", third)):
                if left_name >= right_name:
                    continue
                for column in TRIFECTA_DIFF_COLUMNS:
                    left_value = _float_value(left_row.get(column))
                    right_value = _float_value(right_row.get(column))
                    example[f"{left_name}_minus_{right_name}_{column}"] = _safe_diff(left_value, right_value)

        example["lane_sum"] = float(first_lane + second_lane + third_lane)
        example["lane_span"] = float(max(order) - min(order))
        example["is_first_inside"] = 1.0 if first_lane == 1 else 0.0
        example["is_second_inside"] = 1.0 if second_lane == 1 else 0.0
        example["is_third_inside"] = 1.0 if third_lane == 1 else 0.0
        examples.append(example)

    return examples


def build_exacta_examples_from_race_rows(
    race_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(race_rows) < 2:
        return []

    ordered_rows = sorted(race_rows, key=lambda row: int(float(row["lane"])))
    lanes = [int(float(row["lane"])) for row in ordered_rows]
    row_by_lane = {int(float(row["lane"])): row for row in ordered_rows}
    template = ordered_rows[0]
    actual_trifecta_key = str(template.get("trifecta_key") or "").strip() or None
    actual_exacta_key = "-".join(actual_trifecta_key.split("-")[:2]) if actual_trifecta_key else None

    examples: list[dict[str, Any]] = []
    for first_lane, second_lane in itertools.permutations(lanes, 2):
        first = row_by_lane[first_lane]
        second = row_by_lane[second_lane]
        example = {
            "race_key": str(template["race_key"]),
            "date": str(template["date"]),
            "venue_code": _normalize_venue_code(template.get("venue_code")),
            "venue_name": str(template.get("venue_name") or ""),
            "race_no": int(float(template["race_no"])),
            "combination": f"{first_lane}-{second_lane}",
            "actual_exacta_key": actual_exacta_key,
            "is_target": 1 if actual_exacta_key and actual_exacta_key == f"{first_lane}-{second_lane}" else 0,
        }
        for column in EXACTA_CONTEXT_COLUMNS:
            example[column] = _float_value(template.get(column))

        for prefix, entrant in (("first", first), ("second", second)):
            for column in EXACTA_ENTRANT_COLUMNS:
                example[f"{prefix}_{column}"] = _float_value(entrant.get(column))

        for column in EXACTA_DIFF_COLUMNS:
            example[f"first_minus_second_{column}"] = _safe_diff(
                _float_value(first.get(column)),
                _float_value(second.get(column)),
            )

        example["lane_sum"] = float(first_lane + second_lane)
        example["lane_gap"] = float(abs(first_lane - second_lane))
        example["is_first_inside"] = 1.0 if first_lane == 1 else 0.0
        example["is_second_inside"] = 1.0 if second_lane == 1 else 0.0
        examples.append(example)

    return examples


def trifecta_rows_to_matrix(
    rows: list[dict[str, Any]],
    feature_columns: list[str] | None = None,
) -> list[list[float]]:
    feature_columns = feature_columns or list(TRIFECTA_FEATURE_COLUMNS)
    matrix: list[list[float]] = []
    for row in rows:
        matrix.append(
            [
                float("nan") if row.get(column) in (None, "") else float(row[column])
                for column in feature_columns
            ]
        )
    return matrix


def exacta_rows_to_matrix(
    rows: list[dict[str, Any]],
    feature_columns: list[str] | None = None,
) -> list[list[float]]:
    feature_columns = feature_columns or list(EXACTA_FEATURE_COLUMNS)
    matrix: list[list[float]] = []
    for row in rows:
        matrix.append(
            [
                float("nan") if row.get(column) in (None, "") else float(row[column])
                for column in feature_columns
            ]
        )
    return matrix


def predict_trifecta_probability_maps(
    rows: list[dict[str, Any]],
    *,
    model: Any,
    feature_columns: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    examples = build_trifecta_examples(rows)
    if not examples:
        return {}

    matrix = trifecta_rows_to_matrix(examples, feature_columns)
    probabilities = model.predict_proba(matrix)[:, 1].tolist()
    grouped: dict[str, list[tuple[str, float]]] = {}
    for example, probability in zip(examples, probabilities, strict=True):
        grouped.setdefault(example["race_key"], []).append((example["combination"], float(probability)))

    probability_maps: dict[str, dict[str, float]] = {}
    for race_key, items in grouped.items():
        normalized = _normalize_probabilities([probability for _, probability in items])
        probability_maps[race_key] = {
            combination: probability
            for (combination, _), probability in zip(items, normalized, strict=True)
        }
    return probability_maps


def predict_exacta_probability_maps(
    rows: list[dict[str, Any]],
    *,
    model: Any,
    feature_columns: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    examples = build_exacta_examples(rows)
    if not examples:
        return {}

    matrix = exacta_rows_to_matrix(examples, feature_columns)
    probabilities = model.predict_proba(matrix)[:, 1].tolist()
    grouped: dict[str, list[tuple[str, float]]] = {}
    for example, probability in zip(examples, probabilities, strict=True):
        grouped.setdefault(example["race_key"], []).append((example["combination"], float(probability)))

    probability_maps: dict[str, dict[str, float]] = {}
    for race_key, items in grouped.items():
        normalized = _normalize_probabilities([probability for _, probability in items])
        probability_maps[race_key] = {
            combination: probability
            for (combination, _), probability in zip(items, normalized, strict=True)
        }
    return probability_maps


def predict_staged_trifecta_probability_maps(
    rows: list[dict[str, Any]],
    *,
    exacta_model: Any,
    exacta_feature_columns: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    exacta_maps = predict_exacta_probability_maps(
        rows,
        model=exacta_model,
        feature_columns=exacta_feature_columns,
    )
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(str(row["race_key"]), []).append(row)

    trifecta_probability_maps: dict[str, dict[str, float]] = {}
    for race_key, exacta_map in exacta_maps.items():
        race_rows = grouped_rows.get(race_key, [])
        if not race_rows:
            continue
        lanes = sorted(int(float(row["lane"])) for row in race_rows)
        win_probabilities = {
            int(float(row["lane"])): float(row.get("model_win_probability") or 0.0)
            for row in race_rows
        }
        trifecta_scores: dict[str, float] = {}
        for exacta_key, exacta_probability in exacta_map.items():
            first_lane, second_lane = _parse_exacta_combination(exacta_key)
            remaining_lanes = [lane for lane in lanes if lane not in {first_lane, second_lane}]
            third_weights = _normalize_probabilities([win_probabilities.get(lane, 0.0) for lane in remaining_lanes])
            for third_lane, third_weight in zip(remaining_lanes, third_weights, strict=True):
                trifecta_scores[f"{first_lane}-{second_lane}-{third_lane}"] = (
                    float(exacta_probability) * float(third_weight)
                )
        if trifecta_scores:
            normalized = _normalize_probabilities(list(trifecta_scores.values()))
            trifecta_probability_maps[race_key] = {
                combination: probability
                for combination, probability in zip(trifecta_scores.keys(), normalized, strict=True)
            }
    return trifecta_probability_maps


def top3_lane_probabilities(
    trifecta_probability_map: dict[str, float],
) -> dict[int, float]:
    totals: dict[int, float] = {}
    for combination, probability in trifecta_probability_map.items():
        for lane in _parse_combination(combination):
            totals[lane] = totals.get(lane, 0.0) + float(probability)
    return totals


def ranked_trifectas(
    trifecta_probability_map: dict[str, float],
    *,
    top_k: int,
) -> list[tuple[tuple[int, int, int], float]]:
    ranked = sorted(
        trifecta_probability_map.items(),
        key=lambda item: (-float(item[1]), item[0]),
    )
    results: list[tuple[tuple[int, int, int], float]] = []
    for combination, probability in ranked[:top_k]:
        results.append((_parse_combination(combination), float(probability)))
    return results


def _normalize_probabilities(probabilities: list[float]) -> list[float]:
    observed = [max(float(probability), 1e-6) for probability in probabilities]
    total = sum(observed)
    if total <= 0:
        return [1.0 / len(observed) for _ in observed] if observed else []
    return [probability / total for probability in observed]


def _safe_diff(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return float("nan")
    return float(left - right)


def _parse_combination(combination: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in str(combination).split("-"))
    if len(parts) != 3:
        raise ValueError(f"Invalid trifecta combination: {combination}")
    return parts


def _parse_exacta_combination(combination: str) -> tuple[int, int]:
    parts = tuple(int(part) for part in str(combination).split("-"))
    if len(parts) != 2:
        raise ValueError(f"Invalid exacta combination: {combination}")
    return parts


def _float_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _parse_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _normalize_venue_code(value: Any) -> str:
    if value in (None, ""):
        return "00"
    return str(int(float(value))).zfill(2)
