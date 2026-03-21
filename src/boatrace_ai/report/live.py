"""Chat-friendly live reporting for recommendations and settled outcomes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

from boatrace_ai.collect.official import OfficialBoatraceClient, RaceResult

BET_UNIT_YEN = 100
LIVE_REPORT_COMMANDS = {"predict-live", "report-live"}


def generate_live_report_message(
    *,
    race_date: str,
    current_payload: dict[str, Any],
    prediction_dir: Path,
    state_path: Path,
    result_max_workers: int = 4,
    upcoming_limit: int = 5,
    settled_limit: int = 10,
    quiet_when_empty: bool = False,
) -> tuple[str, dict[str, Any]]:
    upcoming_lines = build_upcoming_report_lines(current_payload, limit=upcoming_limit)
    reported = load_live_report_state(state_path)
    recommendation_index = load_live_recommendation_index(prediction_dir, date.fromisoformat(race_date))
    result_map = fetch_recommendation_result_map(recommendation_index.values(), max_workers=result_max_workers)
    settled_lines, new_keys = build_settled_report_lines(
        recommendations=recommendation_index.values(),
        result_map=result_map,
        reported_keys=reported,
        limit=settled_limit,
    )

    if new_keys:
        save_live_report_state(state_path, reported | new_keys)

    sections: list[str] = []
    if upcoming_lines:
        sections.append("【直前予想】")
        sections.extend(upcoming_lines)
    if settled_lines:
        sections.append("【結果】")
        sections.extend(settled_lines)

    if not sections:
        if quiet_when_empty:
            return "", {
                "upcoming_count": 0,
                "settled_count": 0,
                "reported_result_keys": len(reported),
            }
        return "現在、直前情報が揃った買い目も新規確定結果もありません。", {
            "upcoming_count": 0,
            "settled_count": 0,
            "reported_result_keys": len(reported),
        }

    return "\n".join(sections), {
        "upcoming_count": len(upcoming_lines),
        "settled_count": len(settled_lines),
        "reported_result_keys": len(reported | new_keys),
    }


def build_upcoming_report_lines(current_payload: dict[str, Any], limit: int = 5) -> list[str]:
    recommendations = list(current_payload.get("recommendations", []))
    if recommendations:
        deadline_map = {
            race.get("race_key", ""): race.get("deadline")
            for race in current_payload.get("race_predictions", [])
        }
        recommendations.sort(
            key=lambda item: (
                deadline_map.get(item.get("race_key", ""), "99:99"),
                -float(item.get("expected_value", 0)),
                item.get("race_key", ""),
            )
        )
        lines = []
        for recommendation in recommendations[:limit]:
            deadline = deadline_map.get(recommendation.get("race_key", ""), "??:??")
            market_odds = float(recommendation.get("market_odds", 0))
            probability = float(recommendation.get("probability", 0))
            expected_value = float(recommendation.get("expected_value", 0))
            lines.append(
                f"{recommendation.get('stadium_name', '?')}{int(recommendation.get('race_number', 0))}R "
                f"締切{deadline} "
                f"{recommendation.get('combination', '-')}"
                f" @{market_odds:.1f}倍 "
                f"予測{probability:.2f}% "
                f"EV{expected_value * 100:.0f}%"
            )
        return lines

    races = list(current_payload.get("race_predictions", []))
    races.sort(key=lambda item: (item.get("deadline", "99:99"), -_top_boat_win_prob(item)))
    lines = []
    for race in races[:limit]:
        top_boat = _top_boat(race)
        trifectas = race.get("trifectas", [])
        trifecta = trifectas[0] if trifectas else None
        if top_boat is None:
            continue
        line = (
            f"{race.get('stadium_name', '?')}{int(race.get('race_number', 0))}R "
            f"締切{race.get('deadline', '??:??')} "
            f"本命 {int(top_boat.get('boat_number', 0))}号艇 {top_boat.get('racer_name', '不明')} "
            f"{float(top_boat.get('win_prob', 0)):.2f}%"
        )
        if trifecta:
            line += (
                f" / 本線 {trifecta.get('combination', '-')}"
                f" {float(trifecta.get('probability', 0)):.2f}%"
            )
        lines.append(line)
    return lines


def load_live_recommendation_index(prediction_dir: Path, target_date: date) -> dict[str, dict[str, Any]]:
    date_key = target_date.strftime("%Y%m%d")
    recommendations: dict[str, dict[str, Any]] = {}
    for path in sorted(prediction_dir.glob(f"predictions_{date_key}_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("command") not in LIVE_REPORT_COMMANDS:
            continue
        for recommendation in payload.get("recommendations", []):
            key = recommendation_key(recommendation)
            recommendations[key] = recommendation
    return recommendations


def fetch_recommendation_result_map(
    recommendations: list[dict[str, Any]],
    *,
    max_workers: int = 4,
) -> dict[str, dict[str, Any]]:
    tasks = []
    for recommendation in recommendations:
        race_key = recommendation.get("race_key", "")
        if not race_key:
            continue
        date_key, venue_code, race_no = race_key.split("_")
        tasks.append(
            {
                "race_key": race_key,
                "race_date": f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:8]}",
                "venue_code": venue_code,
                "race_no": int(race_no),
            }
        )

    deduped = {
        task["race_key"]: task
        for task in tasks
    }
    task_list = list(deduped.values())
    if not task_list:
        return {}

    if len(task_list) == 1 or max_workers <= 1:
        results = [_fetch_one_result(task) for task in task_list]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(task_list))) as executor:
            futures = {executor.submit(_fetch_one_result, task): task for task in task_list}
            for future in as_completed(futures):
                results.append(future.result())

    return {
        item["race_key"]: item["result"]
        for item in results
        if item["result"] is not None
    }


def build_settled_report_lines(
    *,
    recommendations: list[dict[str, Any]],
    result_map: dict[str, dict[str, Any]],
    reported_keys: set[str],
    limit: int = 10,
) -> tuple[list[str], set[str]]:
    new_keys: set[str] = set()
    lines: list[str] = []
    ordered = sorted(
        recommendations,
        key=lambda item: (item.get("race_key", ""), item.get("combination", "")),
    )
    for recommendation in ordered:
        key = recommendation_key(recommendation)
        if key in reported_keys:
            continue
        result = result_map.get(recommendation.get("race_key", ""))
        if not result or not result.get("actual_order"):
            continue
        actual_order = result.get("actual_order", "")
        hit = recommendation.get("combination") == actual_order
        payout = int(result.get("payoff_map", {}).get(actual_order, 0))
        market_odds = float(recommendation.get("market_odds", 0))
        if hit:
            line = (
                f"{recommendation.get('stadium_name', '?')}{int(recommendation.get('race_number', 0))}R "
                f"{recommendation.get('combination', '-')} @{market_odds:.1f}倍 的中 "
                f"払戻{payout:,}円"
            )
        else:
            line = (
                f"{recommendation.get('stadium_name', '?')}{int(recommendation.get('race_number', 0))}R "
                f"{recommendation.get('combination', '-')} @{market_odds:.1f}倍 ハズレ "
                f"結果{actual_order}"
            )
            if payout > 0:
                line += f" 払戻{payout:,}円"
        lines.append(line)
        new_keys.add(key)
        if len(lines) >= limit:
            break
    return lines, new_keys


def load_live_report_state(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(value) for value in payload.get("reported_result_keys", [])}


def save_live_report_state(path: Path, reported_keys: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(),
        "reported_result_keys": sorted(reported_keys),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def recommendation_key(recommendation: dict[str, Any]) -> str:
    return f"{recommendation.get('race_key', '')}|{recommendation.get('combination', '')}"


def _top_boat(race_prediction: dict[str, Any]) -> dict[str, Any] | None:
    boats = list(race_prediction.get("boats", []))
    if not boats:
        return None
    boats.sort(key=lambda item: (int(item.get("predicted_rank", 99)), -float(item.get("win_prob", 0))))
    return boats[0]


def _top_boat_win_prob(race_prediction: dict[str, Any]) -> float:
    top_boat = _top_boat(race_prediction)
    return float(top_boat.get("win_prob", 0)) if top_boat else 0.0


def _fetch_one_result(task: dict[str, Any]) -> dict[str, Any]:
    client = OfficialBoatraceClient()
    try:
        result = client.fetch_race_result(task["race_date"], task["venue_code"], task["race_no"])
    except Exception:
        result = None
    finally:
        client.close()

    return {
        "race_key": task["race_key"],
        "result": _convert_result(result) if result is not None else None,
    }


def _convert_result(result: RaceResult) -> dict[str, Any]:
    finish_order = [entrant.lane for entrant in result.entrants if entrant.finish_position and entrant.finish_position <= 3]
    payoff_map = {
        payout.combination: int(payout.payout_yen or 0)
        for payout in result.payouts
        if payout.bet_type == "3連単" and payout.combination
    }
    return {
        "finish_order": finish_order,
        "actual_order": "-".join(str(lane) for lane in finish_order[:3]) if len(finish_order) >= 3 else "",
        "payoff_map": payoff_map,
    }
