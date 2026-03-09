#!/usr/bin/env python3
"""
ボートレース夕方検証スクリプト
当日のレース結果を取得し、朝の予測と突合して的中率・回収率を算出。
note.com無料記事用HTMLを生成する。

Usage:
    python boatrace_note_evening.py [--date YYYY-MM-DD]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, date
from collections import defaultdict
from pathlib import Path

import httpx

# ── 定数 ──────────────────────────────────────────────
STADIUMS = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川",
    6: "浜名湖", 7: "蒲郡", 8: "常滑", 9: "津", 10: "三国",
    11: "びわこ", 12: "住之江", 13: "尼崎", 14: "鳴門", 15: "丸亀",
    16: "児島", 17: "宮島", 18: "徳山", 19: "下関", 20: "若松",
    21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}

BASE_DIR = Path("output")
FILES_DIR = Path(".")
RESULTS_API_PRIMARY = "https://boatraceopenapi.github.io/api/v2/results/{date_str}"
RESULTS_API_SECONDARY = "https://boatrace-api.because-and.co.jp/api/v1"
BET_UNIT = 100  # 均等ベット金額

MAX_RETRIES = 2
RETRY_DELAY = 2  # seconds


# ── ユーティリティ ─────────────────────────────────────
def ensure_dirs():
    """出力ディレクトリを作成"""
    (BASE_DIR / "note").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)


def _fetch_from_primary_api(target_date: date) -> list:
    """Primary API (boatraceopenapi) からレース結果を取得"""
    date_str = target_date.strftime("%Y%m%d")
    url = RESULTS_API_PRIMARY.format(date_str=date_str)
    print(f"[INFO] Primary API取得: {url}")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    print(f"[INFO] Primary API: {len(data)} レース結果を取得")
                    return data
                elif isinstance(data, dict):
                    results = data.get("results", data.get("data", []))
                    if isinstance(results, list):
                        print(f"[INFO] Primary API: {len(results)} レース結果を取得")
                        return results
                    return [data] if data else []
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            msg = f"HTTP {e.response.status_code}" if hasattr(e, 'response') else str(e)
            print(f"[WARN] Primary API: {msg} (試行 {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return []


def _fetch_from_secondary_api(target_date: date) -> list:
    """Secondary API (because-and.co.jp) から場ごと・レースごとに結果を取得"""
    date_str = target_date.strftime("%Y-%m-%d")
    date_str_nohyphen = target_date.strftime("%Y%m%d")
    print(f"[INFO] Secondary API取得 ({RESULTS_API_SECONDARY})")
    collected = []
    try:
        with httpx.Client(timeout=15.0) as client:
            for jcd_int in range(1, 25):
                jcd = f"{jcd_int:02d}"
                found_any = False
                for race_no in range(1, 13):
                    url = f"{RESULTS_API_SECONDARY}/stadiums/{jcd}/races/{race_no}/result?hd={date_str}"
                    try:
                        resp = client.get(url)
                        if resp.status_code != 200:
                            continue
                        data = resp.json()
                        body = data.get("body", {})
                        if not body:
                            continue
                        found_any = True
                        # Secondary API format -> unified format
                        results_list = body.get("results", body.get("result", []))
                        finish_order = []
                        if isinstance(results_list, list):
                            sorted_results = sorted(
                                [r for r in results_list if r.get("rank") or r.get("arrival") or r.get("order")],
                                key=lambda r: int(r.get("rank", r.get("arrival", r.get("order", 99))))
                            )
                            for r in sorted_results:
                                boat = r.get("waku", r.get("boat_no", r.get("lane", 0)))
                                finish_order.append(int(boat))
                        payoffs_raw = body.get("payoffs", body.get("payoff", []))
                        trifecta_payout = 0
                        if isinstance(payoffs_raw, list):
                            for p in payoffs_raw:
                                if "3連単" in str(p.get("type", "")):
                                    trifecta_payout = int(p.get("amount", p.get("payout", 0)))
                                    break
                        collected.append({
                            "venue_code": jcd,
                            "venue_name": STADIUMS.get(jcd_int, f"場{jcd}"),
                            "race_no": race_no,
                            "result": finish_order,
                            "trifecta_payout": trifecta_payout,
                        })
                    except Exception:
                        continue
                if found_any:
                    print(f"  {STADIUMS.get(jcd_int, jcd)}: OK")
    except Exception as e:
        print(f"[WARN] Secondary API接続エラー: {e}")
    if collected:
        print(f"[INFO] Secondary API: {len(collected)} レース結果を取得")
    return collected


def _fetch_from_local_file(target_date: date) -> list:
    """ローカルファイル (data/today_results.json) からフォールバック読み込み"""
    local_path = FILES_DIR / "data" / "today_results.json"
    print(f"[INFO] ローカルファイル検索: {local_path}")
    if not local_path.exists():
        print(f"[WARN] ローカル結果ファイルなし")
        return []
    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    file_date = data.get("date", "").replace("/", "-")
    target_iso = target_date.isoformat()
    if file_date != target_iso:
        print(f"[WARN] ローカルファイルの日付 ({file_date}) が対象日 ({target_iso}) と一致しません")
        return []
    races = data.get("races", [])
    if races:
        print(f"[INFO] ローカルファイル: {len(races)} レース結果を読み込み")
    return races


def fetch_results(target_date: date) -> list:
    """レース結果を取得（複数ソースをフォールバック）"""
    # 1) Primary API
    results = _fetch_from_primary_api(target_date)
    if results:
        return results

    # 2) Secondary API
    results = _fetch_from_secondary_api(target_date)
    if results:
        return results

    # 3) Local file fallback
    results = _fetch_from_local_file(target_date)
    if results:
        return results

    print("[ERROR] 全ソースから結果取得に失敗しました")
    return []


def load_predictions(target_date: date) -> dict:
    """朝の予測JSONを読み込む"""
    date_str = target_date.strftime("%Y%m%d")
    pred_path = BASE_DIR / "data" / f"predictions_{date_str}.json"
    print(f"[INFO] 予測ファイル読み込み: {pred_path}")

    if not pred_path.exists():
        print(f"[ERROR] 予測ファイルが見つかりません: {pred_path}")
        return {}

    with open(pred_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cumulative() -> dict:
    """累積成績ファイルを読み込む（なければ新規）"""
    cum_path = BASE_DIR / "data" / "cumulative_results.json"
    if cum_path.exists():
        with open(cum_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"daily_results": []}


def save_cumulative(data: dict):
    """累積成績ファイルを保存"""
    cum_path = BASE_DIR / "data" / "cumulative_results.json"
    with open(cum_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 累積成績保存: {cum_path}")


# ── 結果パース ─────────────────────────────────────────
def build_result_map(results: list, target_date: date) -> dict:
    """
    結果をrace_keyでインデックス化。
    複数のデータフォーマットに対応:
      - Primary API形式: race_stadium_number, race_number, race_result_boats, race_result_payoffs
      - ローカルファイル形式: venue_code, race_no, result (着順配列), trifecta_payout
    returns: {race_key: {"finish_order": [1着艇番, 2着艇番, ...], "payoffs": [...]}}
    """
    result_map = {}
    date_str = target_date.strftime("%Y-%m-%d")

    for race in results:
        stadium = 0
        race_num = 0
        finish_order = []
        payoffs = []

        # ── Format A: Primary API (race_stadium_number / race_result_boats) ──
        if "race_stadium_number" in race:
            try:
                stadium = int(race.get("race_stadium_number", 0))
                race_num = int(race.get("race_number", 0))
            except (ValueError, TypeError):
                continue

            boats = race.get("race_result_boats", [])
            if not boats:
                continue
            try:
                sorted_boats = sorted(
                    [b for b in boats if b.get("place_number") is not None and b.get("place_number") > 0],
                    key=lambda b: b["place_number"]
                )
            except (KeyError, TypeError):
                continue
            if not sorted_boats:
                continue
            finish_order = [int(b["boat_number"]) for b in sorted_boats]
            payoffs = race.get("race_result_payoffs", [])

        # ── Format B: Local file / Secondary API (venue_code / race_no / result) ──
        elif "venue_code" in race or "race_no" in race:
            try:
                venue_code = race.get("venue_code", "0")
                stadium = int(venue_code)
                race_num = int(race.get("race_no", 0))
            except (ValueError, TypeError):
                continue

            result_arr = race.get("result", [])
            if not result_arr or not isinstance(result_arr, list):
                continue
            finish_order = [int(x) for x in result_arr]

            # 払戻: trifecta_payout -> payoffs形式に変換
            tri_payout = race.get("trifecta_payout", 0)
            if tri_payout and len(finish_order) >= 3:
                combo = f"{finish_order[0]}-{finish_order[1]}-{finish_order[2]}"
                payoffs = [{
                    "payoff_type": "3連単",
                    "combination": combo,
                    "payoff_amount": int(tri_payout),
                }]

        else:
            continue

        if stadium == 0 or race_num == 0:
            continue
        if not finish_order:
            continue

        race_key = f"{date_str}_{stadium}_{race_num}"
        result_map[race_key] = {
            "finish_order": finish_order,
            "payoffs": payoffs,
            "stadium": stadium,
            "race_number": race_num,
        }

    print(f"[INFO] {len(result_map)} レースの結果をパース")
    return result_map


def get_payoff(payoffs: list, bet_type: str, combination: str) -> int:
    """払戻リストから対応する配当を取得"""
    for p in payoffs:
        p_type = p.get("payoff_type", "")
        p_combo = p.get("combination", "")
        if p_type == bet_type and p_combo == combination:
            return int(p.get("payoff_amount", 0))
    return 0


# ── 的中判定 ───────────────────────────────────────────
def verify_recommendations(recommendations: list, result_map: dict) -> list:
    """各推奨買い目の的中判定を行う"""
    verified = []

    for rec in recommendations:
        race_key = rec.get("race_key", "")
        bet_type = rec.get("bet_type", "3連単")
        combination = rec.get("combination", "")
        stadium = rec.get("stadium", 0)
        stadium_name = rec.get("stadium_name", STADIUMS.get(stadium, "?"))

        entry = {
            "race_key": race_key,
            "stadium": stadium,
            "stadium_name": stadium_name,
            "race_number": rec.get("race_number", 0),
            "bet_type": bet_type,
            "combination": combination,
            "probability": rec.get("probability", 0),
            "expected_value": rec.get("expected_value", 0),
            "avg_payout": rec.get("avg_payout", 0),
            "hit": False,
            "payout": 0,
            "actual_order": "",
            "race_found": False,
        }

        if race_key not in result_map:
            verified.append(entry)
            continue

        entry["race_found"] = True
        result = result_map[race_key]
        finish_order = result["finish_order"]

        # 実際の着順文字列
        if bet_type == "3連単" and len(finish_order) >= 3:
            actual = f"{finish_order[0]}-{finish_order[1]}-{finish_order[2]}"
        elif bet_type == "2連単" and len(finish_order) >= 2:
            actual = f"{finish_order[0]}-{finish_order[1]}"
        else:
            actual = "-".join(str(x) for x in finish_order[:3])

        entry["actual_order"] = actual

        # 的中判定
        if combination == actual:
            entry["hit"] = True
            payout = get_payoff(result["payoffs"], bet_type, combination)
            entry["payout"] = payout

        verified.append(entry)

    return verified


def verify_rank_predictions(race_predictions: list, result_map: dict) -> dict:
    """1着予測の精度を検証"""
    stats = {
        "total": 0,
        "top1_hit": 0,
        "top2_hit": 0,
        "top3_hit": 0,
        "details": [],
    }

    for pred in race_predictions:
        race_key = pred.get("race_key", "")
        if race_key not in result_map:
            continue

        boats = pred.get("boats", [])
        if not boats:
            continue

        # AI予測1位の艇番号を取得
        predicted_first = None
        for b in boats:
            if b.get("predicted_rank") == 1:
                predicted_first = b["boat_number"]
                break

        if predicted_first is None:
            # predicted_rankがない場合、win_probが最大の艇
            predicted_first = max(boats, key=lambda x: x.get("win_prob", 0))["boat_number"]

        result = result_map[race_key]
        finish_order = result["finish_order"]

        if not finish_order:
            continue

        stats["total"] += 1

        # 実際に何着だったか
        try:
            actual_place = finish_order.index(predicted_first) + 1
        except ValueError:
            actual_place = 99  # 見つからない場合

        detail = {
            "race_key": race_key,
            "predicted_first": predicted_first,
            "actual_place": actual_place,
        }
        stats["details"].append(detail)

        if actual_place == 1:
            stats["top1_hit"] += 1
        if actual_place <= 2:
            stats["top2_hit"] += 1
        if actual_place <= 3:
            stats["top3_hit"] += 1

    return stats


# ── 集計 ──────────────────────────────────────────────
def compute_summary(verified: list) -> dict:
    """全体サマリーを計算"""
    total_bets = len(verified)
    hits = sum(1 for v in verified if v["hit"])
    investment = total_bets * BET_UNIT
    total_payout = sum(v["payout"] for v in verified)
    hit_rate = (hits / total_bets * 100) if total_bets > 0 else 0.0
    roi = (total_payout / investment * 100) if investment > 0 else 0.0
    profit = total_payout - investment

    return {
        "total_bets": total_bets,
        "hits": hits,
        "hit_rate": hit_rate,
        "investment": investment,
        "total_payout": total_payout,
        "roi": roi,
        "profit": profit,
    }


def compute_stadium_stats(verified: list) -> list:
    """場別成績を集計"""
    stadium_data = defaultdict(lambda: {
        "bets": 0, "hits": 0, "investment": 0, "payout": 0
    })

    for v in verified:
        sid = v["stadium"]
        sname = v["stadium_name"]
        key = (sid, sname)
        stadium_data[key]["bets"] += 1
        stadium_data[key]["investment"] += BET_UNIT
        if v["hit"]:
            stadium_data[key]["hits"] += 1
            stadium_data[key]["payout"] += v["payout"]

    result = []
    for (sid, sname), st in sorted(stadium_data.items()):
        roi = (st["payout"] / st["investment"] * 100) if st["investment"] > 0 else 0.0
        result.append({
            "stadium": sid,
            "stadium_name": sname,
            "bets": st["bets"],
            "hits": st["hits"],
            "investment": st["investment"],
            "payout": st["payout"],
            "roi": roi,
        })
    return result


# ── HTML生成 ──────────────────────────────────────────
def format_yen(amount: int) -> str:
    """金額をカンマ区切りに"""
    return f"{amount:,}"


def generate_html(
    target_date: date,
    summary: dict,
    verified: list,
    stadium_stats: list,
    rank_stats: dict,
    cumulative: dict,
) -> str:
    """note.com無料記事用HTMLを生成（回収率を最大訴求ポイントに）"""
    date_str = target_date.strftime("%Y/%m/%d")
    weekday_jp = "月火水木金土日"[target_date.weekday()]
    lines = []

    # ── 累積回収率の事前計算 ──
    daily = cumulative.get("daily_results", [])
    cum_roi = 0.0
    cum_n_days = 0
    if daily:
        cum_n_days = len(daily)
        cum_total_inv = sum(d["investment"] for d in daily)
        cum_total_pay = sum(d["payout"] for d in daily)
        cum_roi = (cum_total_pay / cum_total_inv * 100) if cum_total_inv > 0 else 0.0

    # ── ヒーローセクション：回収率を大きく表示 ──
    roi = summary["roi"]
    profit = summary["profit"]
    sign = "+" if profit >= 0 else ""
    lines.append(f"<h2>{date_str}（{weekday_jp}）のAI予測結果</h2>")
    lines.append("<h3>本日の回収率</h3>")
    lines.append(f"<p><strong>{roi:.1f}%</strong>（投資 {format_yen(summary['investment'])}円 → 払戻 {format_yen(summary['total_payout'])}円 / 損益 {sign}{format_yen(profit)}円）</p>")
    if cum_n_days > 0:
        lines.append("<h3>累積回収率</h3>")
        lines.append(f"<p><strong>{cum_roi:.1f}%</strong>（過去{cum_n_days}日間）</p>")
    lines.append(f"<p>推奨買い目 {summary['total_bets']}件 / 的中 {summary['hits']}件（的中率 {summary['hit_rate']:.1f}%）</p>")

    # ── 回収率推移（累積成績を記事上部に配置） ──
    if daily and len(daily) > 1:
        lines.append("")
        lines.append("<h2>回収率推移</h2>")
        lines.append("<table>")
        lines.append("  <tr><th>日付</th><th>回収率</th><th>損益</th><th>的中</th><th>買い目数</th></tr>")
        for d in reversed(daily):  # 新しい日付を上に
            d_roi = (d["payout"] / d["investment"] * 100) if d["investment"] > 0 else 0.0
            d_profit = d["payout"] - d["investment"]
            d_sign = "+" if d_profit >= 0 else ""
            lines.append(
                f'  <tr><td>{d["date"]}</td>'
                f'<td>{d_roi:.1f}%</td>'
                f'<td>{d_sign}{format_yen(d_profit)}円</td>'
                f'<td>{d["hits"]}/{d["total_bets"]}</td>'
                f'<td>{d["total_bets"]}</td></tr>'
            )
        lines.append("</table>")
        # 累積合計行
        cum_total_bets = sum(d["total_bets"] for d in daily)
        cum_total_hits = sum(d["hits"] for d in daily)
        cum_total_inv = sum(d["investment"] for d in daily)
        cum_total_pay = sum(d["payout"] for d in daily)
        cum_profit = cum_total_pay - cum_total_inv
        cum_sign = "+" if cum_profit >= 0 else ""
        lines.append(f"<p><strong>累積: 回収率 {cum_roi:.1f}% / 損益 {cum_sign}{format_yen(cum_profit)}円 / 的中 {cum_total_hits}/{cum_total_bets}</strong></p>")
    elif daily:
        lines.append("")
        lines.append("<h2>累積成績</h2>")
        lines.append("<p>本日が初日です。明日以降、回収率の推移を表示します。</p>")

    # ── 的中した買い目 ──
    lines.append("")
    lines.append("<h2>的中した買い目</h2>")
    hit_entries = [v for v in verified if v["hit"]]
    if hit_entries:
        lines.append("<table>")
        lines.append("  <tr><th>場</th><th>R</th><th>買い目</th><th>種別</th><th>配当</th><th>回収率</th></tr>")
        for h in sorted(hit_entries, key=lambda x: -x["payout"]):
            h_roi = h["payout"] / BET_UNIT * 100
            lines.append(
                f'  <tr><td>{h["stadium_name"]}</td>'
                f'<td>{h["race_number"]}R</td>'
                f'<td>{h["combination"]}</td>'
                f'<td>{h["bet_type"]}</td>'
                f'<td>{format_yen(h["payout"])}円</td>'
                f'<td>{h_roi:.0f}%</td></tr>'
            )
        lines.append("</table>")
    else:
        lines.append("<p>本日は的中なし。3連単は的中率が低い分、当たれば高配当。長期の回収率で勝負するスタイルです。</p>")

    # ── 場別回収率 ──
    lines.append("")
    lines.append("<h2>場別回収率</h2>")
    lines.append("<table>")
    lines.append("  <tr><th>場</th><th>回収率</th><th>損益</th><th>的中</th><th>買い目数</th></tr>")
    for s in sorted(stadium_stats, key=lambda x: -x["roi"]):
        s_profit = s["payout"] - s["investment"]
        s_sign = "+" if s_profit >= 0 else ""
        lines.append(
            f'  <tr><td>{s["stadium_name"]}</td>'
            f'<td>{s["roi"]:.1f}%</td>'
            f'<td>{s_sign}{format_yen(s_profit)}円</td>'
            f'<td>{s["hits"]}/{s["bets"]}</td>'
            f'<td>{s["bets"]}</td></tr>'
        )
    lines.append("</table>")

    # ── 不的中一覧 ──
    miss_entries = [v for v in verified if not v["hit"] and v["race_found"]]
    if miss_entries:
        lines.append("")
        lines.append("<h3>不的中の買い目（結果判明分）</h3>")
        lines.append("<table>")
        lines.append("  <tr><th>場</th><th>R</th><th>予測</th><th>結果</th><th>種別</th></tr>")
        for m in sorted(miss_entries, key=lambda x: (x["stadium"], x["race_number"])):
            lines.append(
                f'  <tr><td>{m["stadium_name"]}</td>'
                f'<td>{m["race_number"]}R</td>'
                f'<td>{m["combination"]}</td>'
                f'<td>{m["actual_order"]}</td>'
                f'<td>{m["bet_type"]}</td></tr>'
            )
        lines.append("</table>")

    # ── 1着予測の精度（補足情報） ──
    lines.append("")
    lines.append("<h2>AI予測の精度（参考）</h2>")
    lines.append("<p>各レースのAI予測1位の艇が実際に何着だったか:</p>")
    rt = rank_stats["total"]
    if rt > 0:
        pct1 = rank_stats["top1_hit"] / rt * 100
        pct3 = rank_stats["top3_hit"] / rt * 100
        lines.append(f"<p>1着的中 {rank_stats['top1_hit']}/{rt}（{pct1:.1f}%） / 3着以内 {rank_stats['top3_hit']}/{rt}（{pct3:.1f}%）</p>")
    else:
        lines.append("<p>結果が取得できたレースがないため、精度計算はできませんでした。</p>")

    # ── フッター ──
    lines.append("")
    lines.append("<hr/>")
    lines.append("<p><strong>明日の予測は有料記事で毎朝9時に公開中！ フォローして回収率の推移を追いかけてください！</strong></p>")

    return "\n".join(lines)


def generate_title(target_date: date, summary: dict, cumulative: dict = None) -> str:
    """記事タイトルを生成（回収率を常に表示）"""
    date_str = target_date.strftime("%m/%d")
    weekday_jp = "月火水木金土日"[target_date.weekday()]
    roi = summary["roi"]

    # 累積回収率があれば併記
    cum_part = ""
    if cumulative:
        daily = cumulative.get("daily_results", [])
        if daily:
            cum_inv = sum(d["investment"] for d in daily)
            cum_pay = sum(d["payout"] for d in daily)
            if cum_inv > 0:
                cum_roi = cum_pay / cum_inv * 100
                cum_part = f" 累積{cum_roi:.0f}%"

    return f"【{date_str}({weekday_jp})結果】回収率{roi:.0f}%{cum_part} | ボートレースAI分析"


# ── メイン ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ボートレース夕方検証: 結果取得 → 予測突合 → note記事HTML生成"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="対象日 (YYYY-MM-DD形式、デフォルト: 今日)",
    )
    args = parser.parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = date.today()

    date_yyyymmdd = target_date.strftime("%Y%m%d")
    date_iso = target_date.isoformat()
    print(f"=== ボートレース夕方検証: {date_iso} ===")

    ensure_dirs()

    # 1. 予測読み込み
    predictions = load_predictions(target_date)
    if not predictions:
        print("[ERROR] 予測データなし。終了します。")
        sys.exit(1)

    recommendations = predictions.get("recommendations", [])
    race_predictions = predictions.get("race_predictions", [])
    print(f"[INFO] 推奨買い目: {len(recommendations)}件, レース予測: {len(race_predictions)}件")

    # 2. 結果取得
    results = fetch_results(target_date)
    if not results:
        print("[ERROR] レース結果が取得できませんでした。")
        sys.exit(1)

    # 3. 結果マップ構築
    result_map = build_result_map(results, target_date)

    # 4. 的中判定
    verified = verify_recommendations(recommendations, result_map)

    # 5. 1着予測精度
    rank_stats = verify_rank_predictions(race_predictions, result_map)

    # 6. サマリー計算
    summary = compute_summary(verified)
    stadium_stats = compute_stadium_stats(verified)

    print(f"\n{'='*50}")
    print(f"  的中: {summary['hits']}/{summary['total_bets']} ({summary['hit_rate']:.1f}%)")
    print(f"  投資: {format_yen(summary['investment'])}円")
    print(f"  払戻: {format_yen(summary['total_payout'])}円")
    print(f"  回収率: {summary['roi']:.1f}%")
    profit_sign = '+' if summary['profit'] >= 0 else ''
    print(f"  損益: {profit_sign}{format_yen(summary['profit'])}円")
    print(f"{'='*50}")

    if rank_stats["total"] > 0:
        rt = rank_stats["total"]
        print(f"  1着予測精度: {rank_stats['top1_hit']}/{rt} ({rank_stats['top1_hit']/rt*100:.1f}%)")

    # 7. 累積成績更新
    cumulative = load_cumulative()
    daily_entry = {
        "date": date_iso,
        "total_bets": summary["total_bets"],
        "hits": summary["hits"],
        "investment": summary["investment"],
        "payout": summary["total_payout"],
        "roi": summary["roi"],
    }
    # 同日のデータがあれば上書き
    existing_dates = {d["date"] for d in cumulative["daily_results"]}
    if date_iso in existing_dates:
        cumulative["daily_results"] = [
            d if d["date"] != date_iso else daily_entry
            for d in cumulative["daily_results"]
        ]
    else:
        cumulative["daily_results"].append(daily_entry)
    # 日付順にソート
    cumulative["daily_results"].sort(key=lambda d: d["date"])
    save_cumulative(cumulative)

    # 8. HTML生成
    html = generate_html(target_date, summary, verified, stadium_stats, rank_stats, cumulative)
    title = generate_title(target_date, summary, cumulative)

    # 9. ファイル出力
    html_path = BASE_DIR / "note" / f"evening_{date_yyyymmdd}.html"
    title_path = BASE_DIR / "note" / f"evening_{date_yyyymmdd}_title.txt"
    verify_path = BASE_DIR / "data" / f"verification_{date_yyyymmdd}.json"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] HTML出力: {html_path}")

    with open(title_path, "w", encoding="utf-8") as f:
        f.write(title)
    print(f"[INFO] タイトル出力: {title_path}")

    verification_data = {
        "date": date_iso,
        "summary": summary,
        "verified": verified,
        "stadium_stats": stadium_stats,
        "rank_stats": rank_stats,
    }
    with open(verify_path, "w", encoding="utf-8") as f:
        json.dump(verification_data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 検証データ出力: {verify_path}")

    print(f"\n[DONE] 記事タイトル: {title}")
    print(f"[DONE] 全処理完了")


if __name__ == "__main__":
    main()
