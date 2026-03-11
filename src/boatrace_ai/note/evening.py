"""Generate an evening verification article from predictions and official results."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

from boatrace_ai.collect.official import OfficialBoatraceClient, RaceResult
from boatrace_ai.note import STADIUMS
from boatrace_ai.note.morning import find_prediction_file, load_cumulative


BET_UNIT_YEN = 100


def generate_evening_note(
    *,
    race_date: str,
    prediction_dir: Path,
    output_dir: Path,
    max_workers: int = 4,
) -> dict[str, Any]:
    target_date = datetime.strptime(race_date, "%Y-%m-%d").date()
    prediction_path = find_prediction_file(prediction_dir, target_date)
    predictions = _load_json(prediction_path)

    result_map = fetch_result_map(predictions, max_workers=max_workers)
    verified = verify_recommendations(predictions.get("recommendations", []), result_map)
    rank_stats = verify_rank_predictions(predictions.get("race_predictions", []), result_map)
    summary = compute_summary(verified)
    stadium_stats = compute_stadium_stats(verified)

    output_dir.mkdir(parents=True, exist_ok=True)
    cumulative = load_cumulative(output_dir)
    cumulative = update_cumulative(cumulative, target_date, summary)
    cumulative_path = output_dir / "cumulative_results.json"
    cumulative_path.write_text(json.dumps(cumulative, ensure_ascii=False, indent=2), encoding="utf-8")

    html = generate_html(target_date, summary, verified, stadium_stats, rank_stats, cumulative)
    title = generate_title(target_date, summary, cumulative)

    date_key = target_date.strftime("%Y%m%d")
    html_path = output_dir / f"evening_{date_key}.html"
    title_path = output_dir / f"evening_{date_key}_title.txt"
    verification_path = output_dir / f"verification_{date_key}.json"
    html_path.write_text(html, encoding="utf-8")
    title_path.write_text(title, encoding="utf-8")
    verification_path.write_text(
        json.dumps(
            {
                "date": target_date.isoformat(),
                "summary": summary,
                "verified": verified,
                "stadium_stats": stadium_stats,
                "rank_stats": rank_stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "prediction_path": str(prediction_path),
        "html_path": str(html_path),
        "title_path": str(title_path),
        "verification_path": str(verification_path),
        "cumulative_path": str(cumulative_path),
        "title": title,
        "summary": summary,
    }


def fetch_result_map(predictions: dict[str, Any], max_workers: int = 4) -> dict[str, dict[str, Any]]:
    race_predictions = predictions.get("race_predictions", [])
    tasks = [
        {
            "race_key": race["race_key"],
            "race_date": _race_date_from_key(race["race_key"]),
            "venue_code": f"{int(race.get('stadium', 0)):02d}",
            "race_no": int(race.get("race_number", 0)),
        }
        for race in race_predictions
    ]
    if not tasks:
        return {}

    if len(tasks) == 1 or max_workers <= 1:
        results = [_fetch_one_result(task) for task in tasks]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
            futures = {executor.submit(_fetch_one_result, task): task for task in tasks}
            for future in as_completed(futures):
                results.append(future.result())

    return {
        item["race_key"]: item["result"]
        for item in results
        if item["result"] is not None
    }


def verify_recommendations(
    recommendations: list[dict[str, Any]],
    result_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for recommendation in recommendations:
        race_key = recommendation.get("race_key", "")
        result = result_map.get(race_key)
        actual_order = result["actual_order"] if result else ""
        actual_payout = result["payoff_map"].get(recommendation.get("combination", ""), 0) if result else 0
        hit = bool(result and recommendation.get("combination") == actual_order)
        verified.append(
            {
                "race_key": race_key,
                "stadium": int(recommendation.get("stadium", 0)),
                "stadium_name": recommendation.get("stadium_name", STADIUMS.get(int(recommendation.get("stadium", 0)), "?")),
                "race_number": int(recommendation.get("race_number", 0)),
                "bet_type": recommendation.get("bet_type", "3連単"),
                "combination": recommendation.get("combination", ""),
                "probability": float(recommendation.get("probability", 0)),
                "expected_value": float(recommendation.get("expected_value", 0)),
                "avg_payout": int(float(recommendation.get("avg_payout", 0))),
                "hit": hit,
                "payout": actual_payout if hit else 0,
                "actual_order": actual_order,
                "race_found": result is not None,
            }
        )
    return verified


def verify_rank_predictions(
    race_predictions: list[dict[str, Any]],
    result_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    stats = {
        "total": 0,
        "top1_hit": 0,
        "top2_hit": 0,
        "top3_hit": 0,
        "details": [],
    }

    for race in race_predictions:
        race_key = race.get("race_key", "")
        result = result_map.get(race_key)
        if result is None:
            continue

        top_boat = None
        for boat in race.get("boats", []):
            if int(boat.get("predicted_rank", 99)) == 1:
                top_boat = int(boat.get("boat_number", 0))
                break
        if top_boat is None:
            continue

        finish_order = result["finish_order"]
        stats["total"] += 1
        actual_place = finish_order.index(top_boat) + 1 if top_boat in finish_order else 99
        stats["details"].append(
            {
                "race_key": race_key,
                "predicted_first": top_boat,
                "actual_place": actual_place,
            }
        )
        if actual_place == 1:
            stats["top1_hit"] += 1
        if actual_place <= 2:
            stats["top2_hit"] += 1
        if actual_place <= 3:
            stats["top3_hit"] += 1

    return stats


def compute_summary(verified: list[dict[str, Any]]) -> dict[str, Any]:
    total_bets = len(verified)
    hits = sum(1 for item in verified if item["hit"])
    investment = total_bets * BET_UNIT_YEN
    total_payout = sum(int(item["payout"]) for item in verified)
    profit = total_payout - investment
    return {
        "total_bets": total_bets,
        "hits": hits,
        "hit_rate": (hits / total_bets * 100) if total_bets else 0.0,
        "investment": investment,
        "total_payout": total_payout,
        "roi": (total_payout / investment * 100) if investment else 0.0,
        "profit": profit,
    }


def compute_stadium_stats(verified: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, int]] = defaultdict(
        lambda: {"bets": 0, "hits": 0, "investment": 0, "payout": 0}
    )
    for item in verified:
        key = (int(item["stadium"]), item["stadium_name"])
        grouped[key]["bets"] += 1
        grouped[key]["investment"] += BET_UNIT_YEN
        if item["hit"]:
            grouped[key]["hits"] += 1
            grouped[key]["payout"] += int(item["payout"])

    rows: list[dict[str, Any]] = []
    for (stadium, stadium_name), stats in sorted(grouped.items()):
        roi = (stats["payout"] / stats["investment"] * 100) if stats["investment"] else 0.0
        rows.append(
            {
                "stadium": stadium,
                "stadium_name": stadium_name,
                "bets": stats["bets"],
                "hits": stats["hits"],
                "investment": stats["investment"],
                "payout": stats["payout"],
                "roi": roi,
            }
        )
    return rows


def update_cumulative(
    cumulative: dict[str, Any],
    target_date: date,
    summary: dict[str, Any],
) -> dict[str, Any]:
    daily_results = list(cumulative.get("daily_results", []))
    daily_entry = {
        "date": target_date.isoformat(),
        "total_bets": summary["total_bets"],
        "hits": summary["hits"],
        "investment": summary["investment"],
        "payout": summary["total_payout"],
        "roi": summary["roi"],
    }
    existing_dates = {item["date"] for item in daily_results}
    if daily_entry["date"] in existing_dates:
        daily_results = [daily_entry if item["date"] == daily_entry["date"] else item for item in daily_results]
    else:
        daily_results.append(daily_entry)
    daily_results.sort(key=lambda item: item["date"])
    return {"daily_results": daily_results}


def generate_html(
    target_date: date,
    summary: dict[str, Any],
    verified: list[dict[str, Any]],
    stadium_stats: list[dict[str, Any]],
    rank_stats: dict[str, Any],
    cumulative: dict[str, Any],
) -> str:
    date_str = target_date.strftime("%Y/%m/%d")
    weekday_jp = "月火水木金土日"[target_date.weekday()]
    daily = cumulative.get("daily_results", [])
    cumulative_roi = _cumulative_roi(daily)
    sign = "+" if summary["profit"] >= 0 else ""
    lines = [
        f"<h2>{date_str}（{weekday_jp}）のAI予測結果</h2>",
        "<h3>本日の回収率</h3>",
        (
            f"<p><strong>{summary['roi']:.1f}%</strong>"
            f"（投資 {summary['investment']:,}円 → 払戻 {summary['total_payout']:,}円 / 損益 {sign}{summary['profit']:,}円）</p>"
        ),
    ]
    if daily:
        lines.append("<h3>累積回収率</h3>")
        lines.append(f"<p><strong>{cumulative_roi:.1f}%</strong>（{len(daily)}日分）</p>")

    hit_entries = [item for item in verified if item["hit"]]
    lines.append("<h2>的中した買い目</h2>")
    if hit_entries:
        lines.append("<table>")
        lines.append("  <tr><th>場</th><th>R</th><th>買い目</th><th>配当</th><th>回収率</th></tr>")
        for entry in sorted(hit_entries, key=lambda item: -int(item["payout"])):
            lines.append(
                f"  <tr><td>{entry['stadium_name']}</td>"
                f"<td>{entry['race_number']}R</td>"
                f"<td>{entry['combination']}</td>"
                f"<td>{int(entry['payout']):,}円</td>"
                f"<td>{int(entry['payout']) / BET_UNIT_YEN * 100:.0f}%</td></tr>"
            )
        lines.append("</table>")
    else:
        lines.append("<p>本日は的中なし。期待回収率基準で買い目を絞っているため、的中率より ROI を優先しています。</p>")

    lines.append("<h2>場別回収率</h2>")
    lines.append("<table>")
    lines.append("  <tr><th>場</th><th>回収率</th><th>損益</th><th>的中</th><th>買い目数</th></tr>")
    for entry in sorted(stadium_stats, key=lambda item: -float(item["roi"])):
        profit = int(entry["payout"]) - int(entry["investment"])
        sign = "+" if profit >= 0 else ""
        lines.append(
            f"  <tr><td>{entry['stadium_name']}</td>"
            f"<td>{float(entry['roi']):.1f}%</td>"
            f"<td>{sign}{profit:,}円</td>"
            f"<td>{entry['hits']}/{entry['bets']}</td>"
            f"<td>{entry['bets']}</td></tr>"
        )
    lines.append("</table>")

    if rank_stats["total"] > 0:
        lines.append("<h2>1着予測の参考精度</h2>")
        lines.append(
            f"<p>1着的中 {rank_stats['top1_hit']}/{rank_stats['total']} "
            f"({rank_stats['top1_hit'] / rank_stats['total'] * 100:.1f}%) / "
            f"3着内 {rank_stats['top3_hit']}/{rank_stats['total']} "
            f"({rank_stats['top3_hit'] / rank_stats['total'] * 100:.1f}%)</p>"
        )

    return "\n".join(lines)


def generate_title(
    target_date: date,
    summary: dict[str, Any],
    cumulative: dict[str, Any],
) -> str:
    date_str = target_date.strftime("%m/%d")
    weekday_jp = "月火水木金土日"[target_date.weekday()]
    daily = cumulative.get("daily_results", [])
    if daily:
        return f"【{date_str}({weekday_jp})結果】回収率{summary['roi']:.0f}% 累積{_cumulative_roi(daily):.0f}% | ボートレースAI分析"
    return f"【{date_str}({weekday_jp})結果】回収率{summary['roi']:.0f}% | ボートレースAI分析"


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
    actual_order = "-".join(str(lane) for lane in finish_order[:3]) if len(finish_order) >= 3 else ""
    return {
        "finish_order": finish_order,
        "actual_order": actual_order,
        "payoff_map": payoff_map,
    }


def _race_date_from_key(race_key: str) -> str:
    date_key = race_key.split("_", 1)[0]
    return f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:]}"


def _cumulative_roi(daily: list[dict[str, Any]]) -> float:
    investment = sum(int(item.get("investment", 0)) for item in daily)
    payout = sum(int(item.get("payout", 0)) for item in daily)
    return (payout / investment * 100) if investment else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
