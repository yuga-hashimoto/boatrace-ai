# 毎日の自動改善プロンプト

このファイルを毎日Claudeに渡して実行させる。

---

## プロンプト本文

```
あなたはboatrace-aiリポジトリの改善エージェントです。
本日の日付: {今日の日付 (YYYY-MM-DD)}
作業ディレクトリ: /Users/yu-ga/Documents/GitHub/boatrace-ai

## 目標
- 本日のバックテストを実施し、昨日の recommendation_roi を超えることを目指す
- 結果をroi_history.jsonに記録して履歴を保持する

## 手順

### Step 1: 履歴確認
`artifacts/backtests/roi_history.json` を読み込んで昨日のROIを確認せよ。
最新エントリのrecommendation_roiを「ベースラインROI」とする。

### Step 2: 現在設定でベースラインバックテスト
現在の `configs/base.json` の betting設定でバックテストを実行せよ。

```bash
cd /Users/yu-ga/Documents/GitHub/boatrace-ai
python -m boatrace_ai backtest \
  --betting-preset monthly-roi \
  --output-path artifacts/backtests/today_baseline.json
```

`today_baseline.json` から以下を読み取れ:
- metrics.recommendation_roi
- metrics.recommendation_bets
- metrics.recommendation_hits
- metrics.recommendation_hit_rate

### Step 3: パラメータ探索（最大5パターン試す）
ベースラインROIを超えることを目指して、以下のパラメータを変えてバックテストを試せ。
試すパラメータは `configs/base.json` の `betting` セクションを参考に選べ。

**試すべき変動パラメータ（例）:**
- `--min-market-odds` (現在値 ±10〜20)
- `--max-market-odds` (現在値 ±10〜20)
- `--min-probability` (0.15〜0.30の範囲)
- `--candidate-pool-size` (6〜20の範囲)
- `--min-expected-value` (0.0〜2.0の範囲)

各パターンは以下のように実行せよ:
```bash
python -m boatrace_ai backtest \
  --betting-preset monthly-roi \
  --min-market-odds <値> \
  --max-market-odds <値> \
  --output-path artifacts/backtests/today_trial_N.json
```

**ただし以下の制約を守れ:**
- recommendation_bets が 5 未満のパターンは無効（サンプル不足）
- 過去の roi_history.json で試行済みの組み合わせは避ける
- overfittingを避けるため、betsが非常に少なくROIが極端に高いパターン（bets < 10 かつ ROI > 5.0）は採用しない

### Step 4: 最良パターンの選定
試したパターンの中でrecommendation_roi が最も高く、かつ制約を満たすものを選ぶ。

**採用条件:**
- recommendation_roi > ベースラインROI（roi_history.json の最新エントリ）
- recommendation_bets >= 5

採用パターンがあれば Step 5 へ。なければ Step 6（変化なし）へ。

### Step 5: 設定更新とコミット
採用パターンのパラメータで `configs/base.json` の `betting` セクションを更新せよ。

更新後、再度バックテストを実行して確認:
```bash
python -m boatrace_ai backtest \
  --betting-preset monthly-roi \
  --output-path artifacts/backtests/today_best.json
```

`roi_history.json` に本日のエントリを追加せよ（以下のフォーマット）:
```json
{
  "date": "YYYY-MM-DD",
  "recommendation_roi": <値>,
  "recommendation_bets": <値>,
  "recommendation_hits": <値>,
  "recommendation_hit_rate": <値>,
  "recommendation_return": <値>,
  "recommendation_profit": <値>,
  "betting_preset": "monthly-roi",
  "backtest_file": "today_best.json",
  "params_changed": {
    "<変更したパラメータ名>": {"from": <旧値>, "to": <新値>}
  },
  "notes": "改善内容の簡潔な説明"
}
```

gitコミット:
```bash
git add configs/base.json artifacts/backtests/roi_history.json artifacts/backtests/today_best.json
git commit -m "perf: improve backtest ROI <旧ROI> → <新ROI> (<日付>)"
```

### Step 6: 変化なしの場合
改善パターンが見つからなかった場合も roi_history.json に記録せよ:
```json
{
  "date": "YYYY-MM-DD",
  "recommendation_roi": <ベースラインと同じ>,
  "recommendation_bets": <値>,
  "recommendation_hits": <値>,
  "recommendation_hit_rate": <値>,
  "recommendation_return": <値>,
  "recommendation_profit": <値>,
  "betting_preset": "monthly-roi",
  "backtest_file": "today_baseline.json",
  "notes": "本日は改善なし。試行パターン: <試したパラメータの概要>"
}
```

gitコミット:
```bash
git add artifacts/backtests/roi_history.json
git commit -m "chore: daily backtest log (<日付>) - no improvement found"
```

### Step 7: 最終レポート
以下の形式で本日の結果をまとめよ:

---
## 本日の改善レポート (YYYY-MM-DD)

| 指標 | 昨日 | 本日 | 変化 |
|------|------|------|------|
| recommendation_roi | X.XX | X.XX | +/-X.XX |
| recommendation_bets | N | N | +/-N |
| recommendation_hit_rate | X.XX% | X.XX% | +/-X.XX% |

**変更内容:** （なければ「変更なし」）
**試行したパラメータ:** （一覧）
---
```

---

## 使い方

毎日、以下のコマンドでClaudeを起動してこのプロンプトを渡す:

```bash
# 毎朝このコマンドを実行（例: cronで自動化）
cd /Users/yu-ga/Documents/GitHub/boatrace-ai
claude "$(cat DAILY_IMPROVEMENT_PROMPT.md | sed 's/{今日の日付 (YYYY-MM-DD)}/'"$(date +%Y-%m-%d)"'/g')"
```

または `cron` で自動実行:
```
# 毎朝9時に実行
0 9 * * * cd /Users/yu-ga/Documents/GitHub/boatrace-ai && claude "$(cat DAILY_IMPROVEMENT_PROMPT.md | sed 's/{今日の日付 (YYYY-MM-DD)}/'"$(date +%Y-%m-%d)"'/g')" >> logs/daily_improvement.log 2>&1
```
