"""Dataset building and shared feature extraction."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from boatrace_ai.collect.history import iter_race_record_paths, load_race_record


FEATURE_COLUMNS = [
    "venue_code",
    "race_no",
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
    "temperature_c",
    "wind_speed_mps",
    "water_temperature_c",
    "wave_height_cm",
    "wind_direction_code",
    "parts_exchange_count",
    "has_propeller_note",
]
META_COLUMNS = [
    "race_key",
    "date",
    "venue_name",
    "meeting_name",
    "deadline",
    "racer_id",
    "racer_name",
    "grade",
]
TARGET_COLUMNS = [
    "finish_position",
    "finish_label",
    "is_win",
    "is_top3",
    "result_start_timing",
    "race_time",
    "win_payout_yen",
    "trifecta_key",
    "trifecta_payout_yen",
]
DATASET_COLUMNS = META_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS


def build_dataset(input_dir: Path, output_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    record_paths = iter_race_record_paths(input_dir)
    for path in record_paths:
        rows.extend(build_rows_from_record(load_race_record(path)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows([{column: row.get(column) for column in DATASET_COLUMNS} for row in rows])

    return {
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "records": len(record_paths),
        "rows": len(rows),
        "feature_columns": FEATURE_COLUMNS,
    }


def build_rows_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    card = record["card"]
    beforeinfo = record.get("beforeinfo") or {}
    result = record.get("result") or {}

    before_map = {
        int(entrant["lane"]): entrant
        for entrant in (beforeinfo.get("entrants") or [])
    }
    result_map = {
        int(entrant["lane"]): entrant
        for entrant in (result.get("entrants") or [])
    }
    result_start_map = {
        int(lane): value
        for lane, value in (result.get("start_timings") or {}).items()
    }
    payouts = result.get("payouts") or []
    win_payout = _find_payout(payouts, "単勝")
    trifecta_payout = _find_payout(payouts, "3連単")
    weather = beforeinfo.get("weather") or result.get("weather") or {}

    rows: list[dict[str, Any]] = []
    race_key = f"{record['date']}_{record['venue_code']}_{int(record['race_no']):02d}"

    for entrant in card["entrants"]:
        lane = int(entrant["lane"])
        before_entrant = before_map.get(lane, {})
        result_entrant = result_map.get(lane, {})
        finish_position = result_entrant.get("finish_position")
        is_win = 1 if finish_position == 1 else 0 if finish_position is not None else None
        is_top3 = 1 if finish_position is not None and finish_position <= 3 else 0 if finish_position is not None else None

        row = {
            "race_key": race_key,
            "date": record["date"],
            "venue_name": record["venue_name"],
            "meeting_name": record["meeting_name"],
            "deadline": record.get("deadline"),
            "racer_id": entrant["racer_id"],
            "racer_name": entrant["name"],
            "grade": entrant["grade"],
            "venue_code": _parse_float(record["venue_code"]),
            "race_no": _parse_float(record["race_no"]),
            "lane": _parse_float(lane),
            "grade_code": _grade_code(entrant.get("grade")),
            "age": _parse_float(entrant.get("age")),
            "weight_kg": _parse_float(entrant.get("weight_kg")),
            "f_count": _parse_float(entrant.get("f_count")),
            "l_count": _parse_float(entrant.get("l_count")),
            "average_start_timing": _parse_float(entrant.get("average_start_timing")),
            "national_win_rate": _parse_float(entrant.get("national_win_rate")),
            "national_2ren_rate": _parse_float(entrant.get("national_2ren_rate")),
            "national_3ren_rate": _parse_float(entrant.get("national_3ren_rate")),
            "local_win_rate": _parse_float(entrant.get("local_win_rate")),
            "local_2ren_rate": _parse_float(entrant.get("local_2ren_rate")),
            "local_3ren_rate": _parse_float(entrant.get("local_3ren_rate")),
            "motor_2ren_rate": _parse_float(entrant.get("motor_2ren_rate")),
            "motor_3ren_rate": _parse_float(entrant.get("motor_3ren_rate")),
            "boat_2ren_rate": _parse_float(entrant.get("boat_2ren_rate")),
            "boat_3ren_rate": _parse_float(entrant.get("boat_3ren_rate")),
            "display_weight_kg": _parse_float(before_entrant.get("display_weight_kg")),
            "exhibition_time": _parse_float(before_entrant.get("exhibition_time")),
            "tilt": _parse_float(before_entrant.get("tilt")),
            "adjusted_weight_kg": _parse_float(before_entrant.get("adjusted_weight_kg")),
            "start_display_st": _parse_float(before_entrant.get("start_display_st")),
            "temperature_c": _parse_float(weather.get("temperature_c")),
            "wind_speed_mps": _parse_float(weather.get("wind_speed_mps")),
            "water_temperature_c": _parse_float(weather.get("water_temperature_c")),
            "wave_height_cm": _parse_float(weather.get("wave_height_cm")),
            "wind_direction_code": _wind_direction_code(weather.get("wind_direction")),
            "parts_exchange_count": _parse_float(len(before_entrant.get("parts_exchange") or [])),
            "has_propeller_note": 1.0 if before_entrant.get("propeller_note") else 0.0,
            "finish_position": finish_position,
            "finish_label": result_entrant.get("finish_label"),
            "is_win": is_win,
            "is_top3": is_top3,
            "result_start_timing": _parse_float(result_start_map.get(lane)),
            "race_time": result_entrant.get("race_time"),
            "win_payout_yen": (
                win_payout["payout_yen"]
                if win_payout and str(lane) == win_payout["combination"]
                else 0
            ),
            "trifecta_key": trifecta_payout["combination"] if trifecta_payout else None,
            "trifecta_payout_yen": trifecta_payout["payout_yen"] if trifecta_payout else None,
        }
        rows.append(row)

    return rows
def rows_to_matrix(rows: list[dict[str, Any]], feature_columns: list[str] | None = None) -> list[list[float]]:
    feature_columns = feature_columns or FEATURE_COLUMNS
    matrix: list[list[float]] = []
    for row in rows:
        values: list[float] = []
        for column in feature_columns:
            value = row.get(column)
            values.append(float("nan") if value in (None, "") else float(value))
        matrix.append(values)
    return matrix


def _find_payout(payouts: list[dict[str, Any]], bet_type: str) -> dict[str, Any] | None:
    for payout in payouts:
        if payout.get("bet_type") == bet_type:
            return payout
    return None


def _grade_code(grade: str | None) -> float:
    mapping = {"A1": 4.0, "A2": 3.0, "B1": 2.0, "B2": 1.0}
    return mapping.get((grade or "").strip(), 0.0)


def _wind_direction_code(direction: str | None) -> float:
    if not direction:
        return float("nan")
    suffix = direction.removeprefix("is-direction")
    try:
        return float(int(suffix))
    except ValueError:
        return float("nan")


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
