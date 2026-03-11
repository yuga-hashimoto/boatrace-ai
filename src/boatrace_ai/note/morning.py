"""Generate a morning note-style article from prediction output."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

from boatrace_ai.note import STADIUMS


BET_UNIT_YEN = 100


def generate_morning_note(
    *,
    race_date: str,
    prediction_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    target_date = datetime.strptime(race_date, "%Y-%m-%d").date()
    prediction_path = find_prediction_file(prediction_dir, target_date)
    predictions = _load_json(prediction_path)
    cumulative = load_cumulative(output_dir)

    html = generate_html(target_date, predictions, cumulative)
    title = generate_title(target_date, predictions, cumulative)

    output_dir.mkdir(parents=True, exist_ok=True)
    date_key = target_date.strftime("%Y%m%d")
    html_path = output_dir / f"morning_{date_key}.html"
    title_path = output_dir / f"morning_{date_key}_title.txt"
    html_path.write_text(html, encoding="utf-8")
    title_path.write_text(title, encoding="utf-8")

    return {
        "prediction_path": str(prediction_path),
        "html_path": str(html_path),
        "title_path": str(title_path),
        "title": title,
        "recommendations": len(predictions.get("recommendations", [])),
    }


def find_prediction_file(prediction_dir: Path, target_date: date) -> Path:
    date_key = target_date.strftime("%Y%m%d")
    candidates = sorted(prediction_dir.glob(f"predictions_{date_key}_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No prediction file found for {target_date.isoformat()} in {prediction_dir}")
    return candidates[-1]


def load_cumulative(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "cumulative_results.json"
    if not path.exists():
        return {"daily_results": []}
    return _load_json(path)


def generate_html(
    target_date: date,
    predictions: dict[str, Any],
    cumulative: dict[str, Any],
) -> str:
    date_str = target_date.strftime("%Y/%m/%d")
    weekday_jp = "月火水木金土日"[target_date.weekday()]
    recommendations = predictions.get("recommendations", [])
    race_predictions = predictions.get("race_predictions", [])
    model_metrics = predictions.get("model_metrics") or {}
    top1_hit_rate = model_metrics.get("top1_hit_rate")

    daily = cumulative.get("daily_results", [])
    cumulative_roi = _cumulative_roi(daily)
    lines: list[str] = [f"<h2>{date_str}（{weekday_jp}）のAI予測</h2>"]

    if daily:
        lines.append("<h3>累積回収率</h3>")
        lines.append(f"<p><strong>{cumulative_roi:.1f}%</strong>（{len(daily)}日分の実績）</p>")
    else:
        lines.append("<p>本日が初回配信です。夜に検証記事を出して累積回収率を更新します。</p>")

    lines.append("<h2>本日の分析概要</h2>")
    lines.append(f"<p>対象: <strong>{_count_stadiums(race_predictions)}場 / {len(race_predictions)}レース</strong></p>")
    lines.append(f"<p>推奨買い目: <strong>{len(recommendations)}件</strong> / 想定投資 {len(recommendations) * BET_UNIT_YEN:,}円</p>")
    if top1_hit_rate is not None:
        lines.append(f"<p>直近ホールドアウトの1着的中率: {float(top1_hit_rate) * 100:.1f}%</p>")

    upsets = find_upset_predictions(race_predictions, top_n=3)
    if upsets:
        lines.append("<h2>注目の穴予測</h2>")
        for upset in upsets:
            lines.append(
                f"<p>{upset['stadium_name']}{upset['race_number']}R: "
                f"<strong>{upset['boat_number']}号艇 {upset['racer_name']}</strong> "
                f"を1着予測（{upset['win_prob']:.1f}%）</p>"
            )

    lines.append("<hr/>")
    lines.append("<p>有料パートでは期待回収率の高い順に買い目を掲載しています。</p>")
    lines.append("<!-- NOTE_PAYWALL_BOUNDARY -->")

    sorted_recommendations = sorted(
        recommendations,
        key=lambda item: (-float(item.get("expected_value", 0)), -float(item.get("probability_ratio", 0))),
    )
    top20 = sorted_recommendations[:20]
    if top20:
        lines.append("<h2>期待回収率TOP20</h2>")
        lines.append("<table>")
        lines.append("  <tr><th>順位</th><th>場</th><th>R</th><th>買い目</th><th>予測的中率</th><th>平均配当</th><th>期待回収率</th></tr>")
        for index, recommendation in enumerate(top20, start=1):
            lines.append(
                f"  <tr><td>{index}</td>"
                f"<td>{recommendation.get('stadium_name', '?')}</td>"
                f"<td>{int(recommendation.get('race_number', 0))}R</td>"
                f"<td>{recommendation.get('combination', '-')}</td>"
                f"<td>{float(recommendation.get('probability', 0)):.2f}%</td>"
                f"<td>{int(float(recommendation.get('avg_payout', 0))):,}円</td>"
                f"<td>{float(recommendation.get('expected_value', 0)) * 100:.0f}%</td></tr>"
            )
        lines.append("</table>")

    lines.append("<h2>場別の本命</h2>")
    for stadium_id, races in sorted(_group_by_stadium(race_predictions).items()):
        lines.append(f"<h3>{STADIUMS.get(stadium_id, f'場{stadium_id}')}</h3>")
        for race in races:
            top_boat = _top_boat(race)
            if top_boat is None:
                continue
            lines.append(
                f"<p>{race.get('race_number', 0)}R: "
                f"{top_boat['boat_number']}号艇 {top_boat.get('racer_name', '不明')} "
                f"（{float(top_boat.get('win_prob', 0)):.1f}%）</p>"
            )

    return "\n".join(lines)


def generate_title(
    target_date: date,
    predictions: dict[str, Any],
    cumulative: dict[str, Any],
) -> str:
    date_str = target_date.strftime("%m/%d")
    weekday_jp = "月火水木金土日"[target_date.weekday()]
    daily = cumulative.get("daily_results", [])
    recommendations = predictions.get("recommendations", [])
    if daily:
        return f"【{date_str}({weekday_jp})予測】累積回収率{_cumulative_roi(daily):.0f}%のAI | 厳選{len(recommendations)}買い目"
    return f"【{date_str}({weekday_jp})予測】ボートレースAI分析 | 厳選{len(recommendations)}買い目"


def find_upset_predictions(race_predictions: list[dict[str, Any]], top_n: int = 3) -> list[dict[str, Any]]:
    upsets: list[dict[str, Any]] = []
    for race in race_predictions:
        top_boat = _top_boat(race)
        if not top_boat or int(top_boat.get("boat_number", 1)) == 1:
            continue
        upsets.append(
            {
                "stadium_name": race.get("stadium_name", "?"),
                "race_number": int(race.get("race_number", 0)),
                "boat_number": int(top_boat.get("boat_number", 0)),
                "racer_name": top_boat.get("racer_name", "不明"),
                "win_prob": float(top_boat.get("win_prob", 0)),
            }
        )
    upsets.sort(key=lambda item: -item["win_prob"])
    return upsets[:top_n]


def _count_stadiums(race_predictions: list[dict[str, Any]]) -> int:
    return len({int(race.get("stadium", 0)) for race in race_predictions})


def _group_by_stadium(race_predictions: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for race in race_predictions:
        grouped[int(race.get("stadium", 0))].append(race)
    for races in grouped.values():
        races.sort(key=lambda item: int(item.get("race_number", 0)))
    return grouped


def _top_boat(race_prediction: dict[str, Any]) -> dict[str, Any] | None:
    boats = race_prediction.get("boats", [])
    if not boats:
        return None
    ranked = sorted(
        boats,
        key=lambda item: (int(item.get("predicted_rank", 99)), -float(item.get("win_prob", 0))),
    )
    return ranked[0]


def _cumulative_roi(daily: list[dict[str, Any]]) -> float:
    investment = sum(int(item.get("investment", 0)) for item in daily)
    payout = sum(int(item.get("payout", 0)) for item in daily)
    return (payout / investment * 100) if investment else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
