#!/usr/bin/env python3
"""
ボートレース朝予測スクリプト
予測データを読み込み、回収率を中心訴求ポイントとしたnote.com有料記事HTMLを生成する。

Usage:
    python boatrace_note_morning.py [--date YYYY-MM-DD]
"""

import argparse
import json
import sys
from datetime import datetime, date
from pathlib import Path
from collections import defaultdict

# ── 定数 ──────────────────────────────────────────────
STADIUMS = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川",
    6: "浜名湖", 7: "蒲郡", 8: "常滑", 9: "津", 10: "三国",
    11: "びわこ", 12: "住之江", 13: "尼崎", 14: "鳴門", 15: "丸亀",
    16: "児島", 17: "宮島", 18: "徳山", 19: "下関", 20: "若松",
    21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}

BASE_DIR = Path("output")
BET_UNIT = 100


# ── ユーティリティ ─────────────────────────────────────
def ensure_dirs():
    """出力ディレクトリを作成"""
    (BASE_DIR / "note").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)


def format_yen(amount) -> str:
    """金額をカンマ区切りに"""
    return f"{int(amount):,}"


def load_predictions(target_date: date) -> dict:
    """予測JSONを読み込む"""
    date_str = target_date.strftime("%Y%m%d")
    path = BASE_DIR / "data" / f"predictions_{date_str}.json"
    if not path.exists():
        print(f"[ERROR] 予測ファイルが見つかりません: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] 予測ファイル読み込み: {path}")
    return data


def load_cumulative() -> dict:
    """累積成績JSONを読み込む（存在しない場合は空dictを返す）"""
    path = BASE_DIR / "data" / "cumulative_results.json"
    if not path.exists():
        print("[INFO] 累積成績ファイルなし（初回実行）")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    daily = data.get("daily_results", [])
    print(f"[INFO] 累積成績読み込み: {len(daily)}日分")
    return data


# ── 分析ヘルパー ──────────────────────────────────────
def find_upset_predictions(race_predictions: list, top_n: int = 3) -> list:
    """AIが1号艇以外を1位予測しているレースを抽出（穴予測）"""
    upsets = []
    for rp in race_predictions:
        boats = rp.get("boats", [])
        if not boats:
            continue
        # predicted_rank == 1 の艇を探す
        top_boat = None
        for b in boats:
            if b.get("predicted_rank") == 1:
                top_boat = b
                break
        if top_boat is None:
            # predicted_rank がない場合は win_prob 最大を使う
            top_boat = max(boats, key=lambda x: x.get("win_prob", 0))
        if top_boat and top_boat.get("boat_number", 1) != 1:
            stadium_id = rp.get("stadium", 0)
            upsets.append({
                "stadium": stadium_id,
                "stadium_name": STADIUMS.get(stadium_id, f"場{stadium_id}"),
                "race_number": rp.get("race_number", 0),
                "boat_number": top_boat["boat_number"],
                "racer_name": top_boat.get("racer_name", "不明"),
                "win_prob": top_boat.get("win_prob", 0),
            })
    # win_prob降順でソートしてトップNを返す
    upsets.sort(key=lambda x: -x["win_prob"])
    return upsets[:top_n]


def group_recommendations_by_type(recommendations: list) -> dict:
    """買い目を種別ごとにカウント"""
    counts = defaultdict(int)
    for r in recommendations:
        counts[r.get("bet_type", "不明")] += 1
    return dict(counts)


def group_race_predictions_by_stadium(race_predictions: list) -> dict:
    """レース予測を場ごとにグループ化"""
    groups = defaultdict(list)
    for rp in race_predictions:
        stadium_id = rp.get("stadium", 0)
        groups[stadium_id].append(rp)
    # 各グループをレース番号順にソート
    for sid in groups:
        groups[sid].sort(key=lambda x: x.get("race_number", 0))
    return dict(groups)


def count_unique_stadiums(recommendations: list, race_predictions: list) -> int:
    """分析対象の場数を算出"""
    stadiums = set()
    for r in recommendations:
        stadiums.add(r.get("stadium", 0))
    for rp in race_predictions:
        stadiums.add(rp.get("stadium", 0))
    return len(stadiums)


def count_unique_races(race_predictions: list) -> int:
    """分析対象のレース数を算出"""
    return len(race_predictions)


# ── HTML生成 ──────────────────────────────────────────
def generate_html(
    target_date: date,
    predictions: dict,
    cumulative: dict,
) -> str:
    """note.com有料記事用HTMLを生成（回収率を最大訴求ポイントに）"""
    date_str = target_date.strftime("%Y/%m/%d")
    weekday_jp = "月火水木金土日"[target_date.weekday()]

    recommendations = predictions.get("recommendations", [])
    race_predictions = predictions.get("race_predictions", [])
    model_accuracy = predictions.get("model_cv_accuracy", 0)
    total_train = predictions.get("total_train_races", 0)

    # model_cv_accuracy が 0-1 なら % に変換
    if 0 < model_accuracy < 1:
        model_accuracy_pct = model_accuracy * 100
    else:
        model_accuracy_pct = model_accuracy

    lines = []

    # ── 累積回収率の事前計算 ──
    daily = cumulative.get("daily_results", [])
    cum_roi = 0.0
    cum_n_days = 0
    cum_total_inv = 0
    cum_total_pay = 0
    cum_total_bets = 0
    cum_total_hits = 0
    if daily:
        cum_n_days = len(daily)
        cum_total_inv = sum(d["investment"] for d in daily)
        cum_total_pay = sum(d["payout"] for d in daily)
        cum_total_bets = sum(d["total_bets"] for d in daily)
        cum_total_hits = sum(d["hits"] for d in daily)
        cum_roi = (cum_total_pay / cum_total_inv * 100) if cum_total_inv > 0 else 0.0

    # ── 集計 ──
    n_stadiums = count_unique_stadiums(recommendations, race_predictions)
    n_races = count_unique_races(race_predictions)
    n_recs = len(recommendations)
    type_counts = group_recommendations_by_type(recommendations)
    type_str = "、".join(f"{k} {v}件" for k, v in type_counts.items())

    # ====================================================================
    # 無料パート（ペイウォール前）
    # ====================================================================

    # ── ヒーローセクション：累積回収率を大きく表示 ──
    lines.append(f"<h2>{date_str}（{weekday_jp}）のAI予測</h2>")

    if cum_n_days > 0 and cum_total_inv > 0:
        lines.append("<h3>累積回収率</h3>")
        lines.append(f"<p><strong>{cum_roi:.1f}%</strong>（過去{cum_n_days}日間の実績）</p>")

        # 累積ROI推移テーブル（複数日ある場合）
        if cum_n_days > 1:
            lines.append("")
            lines.append("<h3>回収率推移</h3>")
            lines.append("<table>")
            lines.append("  <tr><th>日付</th><th>回収率</th><th>損益</th><th>的中</th></tr>")
            for d in reversed(daily):  # 新しい日付を上に
                d_roi = (d["payout"] / d["investment"] * 100) if d["investment"] > 0 else 0.0
                d_profit = d["payout"] - d["investment"]
                d_sign = "+" if d_profit >= 0 else ""
                lines.append(
                    f'  <tr><td>{d["date"]}</td>'
                    f'<td>{d_roi:.1f}%</td>'
                    f'<td>{d_sign}{format_yen(d_profit)}円</td>'
                    f'<td>{d["hits"]}/{d["total_bets"]}</td></tr>'
                )
            lines.append("</table>")

        # 累積合計行
        cum_profit = cum_total_pay - cum_total_inv
        cum_sign = "+" if cum_profit >= 0 else ""
        lines.append(f"<p><strong>累積: 回収率 {cum_roi:.1f}% / 損益 {cum_sign}{format_yen(cum_profit)}円 / 的中 {cum_total_hits}/{cum_total_bets}</strong></p>")
    else:
        lines.append("<p>本日が初回の予測配信です。毎晩21時に実績検証レポート（無料）を公開します。</p>")

    # ── 本日の分析サマリー ──
    lines.append("")
    lines.append("<h2>本日の分析概要</h2>")
    lines.append(f"<p>分析対象: <strong>{n_stadiums}場 / {n_races}レース</strong></p>")
    lines.append(f"<p>学習データ: 過去30日間 {format_yen(total_train)}レース（モデル精度 {model_accuracy_pct:.1f}%）</p>")
    lines.append(f"<p>推奨買い目: <strong>{n_recs}件</strong>（{type_str}）</p>")
    lines.append(f"<p>想定投資額: {format_yen(n_recs * BET_UNIT)}円（1買い目{BET_UNIT}円均等）</p>")

    # ── 注目の穴予測（トップ3） ──
    upsets = find_upset_predictions(race_predictions, top_n=3)
    if upsets:
        lines.append("")
        lines.append("<h2>注目の穴予測</h2>")
        lines.append("<p>AIが1号艇（イン）以外を1着と予測したレース:</p>")
        for u in upsets:
            lines.append(
                f"<p>{u['stadium_name']}{u['race_number']}R: "
                f"<strong>{u['boat_number']}号艇 {u['racer_name']}</strong> "
                f"がAI予測1位（{u['win_prob']:.1f}%）</p>"
            )

    # ── CTA ──
    lines.append("")
    lines.append("<hr/>")
    lines.append("<p>この先の有料パートでは、<strong>期待回収率TOP20の具体的な買い目・全場の詳細分析</strong>をお届けします。</p>")
    lines.append("<p>毎晩の実績検証レポートは無料で公開中！</p>")

    # ── ペイウォール境界 ──
    lines.append("")
    lines.append("<!-- NOTE_PAYWALL_BOUNDARY -->")
    lines.append("")

    # ====================================================================
    # 有料パート（ペイウォール後）
    # ====================================================================

    # ── 期待回収率TOP20 推奨買い目 ──
    sorted_recs = sorted(recommendations, key=lambda x: -x.get("expected_value", 0))
    top20 = sorted_recs[:20]
    rest = sorted_recs[20:]

    lines.append("<h2>期待回収率TOP20 推奨買い目</h2>")
    lines.append("<table>")
    lines.append("  <tr><th>順位</th><th>場</th><th>R</th><th>買い目</th><th>種別</th><th>予測的中率</th><th>平均配当</th><th>期待回収率</th></tr>")
    for i, r in enumerate(top20, 1):
        ev_pct = r.get("expected_value", 0) * 100
        lines.append(
            f'  <tr><td>{i}</td>'
            f'<td>{r.get("stadium_name", STADIUMS.get(r.get("stadium", 0), "?"))}</td>'
            f'<td>{r.get("race_number", 0)}R</td>'
            f'<td>{r.get("combination", "-")}</td>'
            f'<td>{r.get("bet_type", "-")}</td>'
            f'<td>{r.get("probability", 0):.2f}%</td>'
            f'<td>{format_yen(r.get("avg_payout", 0))}円</td>'
            f'<td>{ev_pct:.0f}%</td></tr>'
        )
    lines.append("</table>")

    # ── 全推奨買い目（TOP20以外） ──
    if rest:
        lines.append("")
        lines.append("<h2>その他の推奨買い目</h2>")
        lines.append("<table>")
        lines.append("  <tr><th>場</th><th>R</th><th>買い目</th><th>種別</th><th>期待回収率</th></tr>")
        for r in rest:
            ev_pct = r.get("expected_value", 0) * 100
            lines.append(
                f'  <tr><td>{r.get("stadium_name", STADIUMS.get(r.get("stadium", 0), "?"))}</td>'
                f'<td>{r.get("race_number", 0)}R</td>'
                f'<td>{r.get("combination", "-")}</td>'
                f'<td>{r.get("bet_type", "-")}</td>'
                f'<td>{ev_pct:.0f}%</td></tr>'
            )
        lines.append("</table>")

    # ── 場別詳細分析 ──
    lines.append("")
    lines.append("<h2>場別詳細分析</h2>")
    stadium_groups = group_race_predictions_by_stadium(race_predictions)
    for sid in sorted(stadium_groups.keys()):
        s_name = STADIUMS.get(sid, f"場{sid}")
        races = stadium_groups[sid]
        lines.append(f"<h3>{s_name}</h3>")
        for rp in races:
            boats = rp.get("boats", [])
            # AI予測1位の艇を探す
            top_boat = None
            for b in boats:
                if b.get("predicted_rank") == 1:
                    top_boat = b
                    break
            if top_boat is None and boats:
                top_boat = max(boats, key=lambda x: x.get("win_prob", 0))
            if top_boat:
                lines.append(
                    f"<p>{rp.get('race_number', 0)}R: "
                    f"本命 <strong>{top_boat['boat_number']}号艇 {top_boat.get('racer_name', '不明')}</strong>"
                    f"（{top_boat.get('win_prob', 0):.1f}%）</p>"
                )

    # ── 買い方ガイド ──
    lines.append("")
    lines.append("<h2>買い方ガイド</h2>")
    lines.append(f"<p><strong>均等買い戦略</strong>: 各買い目に{BET_UNIT}円ずつ均等にベットします。</p>")
    lines.append(f"<p>本日の推奨{n_recs}件すべてに{BET_UNIT}円ずつ購入した場合の投資額は <strong>{format_yen(n_recs * BET_UNIT)}円</strong> です。</p>")
    lines.append("<p>3連単は的中率が低い（通常1-5%程度）ですが、的中時の配当が大きいため、<strong>長期的にプラスの回収率</strong>を目指す戦略です。</p>")
    lines.append("<p>1日の結果に一喜一憂せず、1週間・1ヶ月単位で回収率をトラッキングしてください。</p>")

    # ── フッター CTA ──
    lines.append("")
    lines.append("<hr/>")
    lines.append("<p><strong>毎晩21時に実績検証レポートを無料公開中！ フォローして回収率の推移を確認してください！</strong></p>")

    return "\n".join(lines)


def generate_title(target_date: date, predictions: dict, cumulative: dict) -> str:
    """記事タイトルを生成（累積回収率があれば表示）"""
    date_str = target_date.strftime("%m/%d")
    weekday_jp = "月火水木金土日"[target_date.weekday()]

    daily = cumulative.get("daily_results", [])
    n_recs = len(predictions.get("recommendations", []))

    if daily:
        cum_inv = sum(d["investment"] for d in daily)
        cum_pay = sum(d["payout"] for d in daily)
        if cum_inv > 0:
            cum_roi = cum_pay / cum_inv * 100
            return f"【{date_str}({weekday_jp})予測】累積回収率{cum_roi:.0f}%のAI | ボートレースAI分析"

    # 累積データなし
    return f"【{date_str}({weekday_jp})予測】ボートレースAI分析 | 厳選{n_recs}買い目"


# ── メイン ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ボートレース朝予測: 予測データからnote記事HTML生成"
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
    print(f"=== ボートレース朝予測記事生成: {date_iso} ===")

    ensure_dirs()

    # 1. 予測読み込み
    predictions = load_predictions(target_date)
    if not predictions:
        print("[ERROR] 予測データなし。終了します。")
        sys.exit(1)

    recommendations = predictions.get("recommendations", [])
    race_predictions = predictions.get("race_predictions", [])
    print(f"[INFO] 推奨買い目: {len(recommendations)}件, レース予測: {len(race_predictions)}件")

    # 2. 累積成績読み込み（存在しなくてもOK）
    cumulative = load_cumulative()

    # 3. HTML生成
    print("[INFO] 記事HTML生成中...")
    html = generate_html(target_date, predictions, cumulative)
    title = generate_title(target_date, predictions, cumulative)

    # 4. ファイル出力
    html_path = BASE_DIR / "note" / f"morning_{date_yyyymmdd}.html"
    title_path = BASE_DIR / "note" / f"morning_{date_yyyymmdd}_title.txt"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] HTML出力: {html_path}")

    with open(title_path, "w", encoding="utf-8") as f:
        f.write(title)
    print(f"[INFO] タイトル出力: {title_path}")

    print(f"\n=== 完了 ===")
    print(f"タイトル: {title}")
    print(f"HTML: {html_path} ({len(html)} bytes)")
    print(f"タイトル: {title_path}")


if __name__ == "__main__":
    main()
