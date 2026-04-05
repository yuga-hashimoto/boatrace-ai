#!/usr/bin/env bash
# 毎日の自動改善スクリプト
# Usage: bash scripts/daily_improve.sh
# または cron: 0 9 * * * bash /path/to/scripts/daily_improve.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TODAY=$(date +%Y-%m-%d)
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

echo "[$(date)] Starting daily improvement for $TODAY" | tee -a "$LOG_DIR/daily_improvement.log"

PROMPT="あなたはboatrace-aiリポジトリの改善エージェントです。
本日の日付: $TODAY
作業ディレクトリ: $REPO_DIR

## 目標
- 本日のバックテストを実施し、昨日の recommendation_roi を超えることを目指す
- 結果をroi_history.jsonに記録して履歴を保持する

## 手順

### Step 1: 履歴確認
\`artifacts/backtests/roi_history.json\` を読み込んで昨日のROIを確認せよ。
最新エントリのrecommendation_roiを「ベースラインROI」とする。

### Step 2: 現在設定でベースラインバックテスト
現在の configs/base.json の betting設定でバックテストを実行せよ。

\`\`\`bash
cd $REPO_DIR
python -m boatrace_ai backtest \\
  --betting-preset monthly-roi \\
  --output-path artifacts/backtests/today_baseline.json
\`\`\`

### Step 3: パラメータ探索（最大5パターン試す）
ベースラインROIを超えることを目指して、以下のパラメータを変えてバックテストを試せ。

試すべき変動パラメータ（例）:
- --min-market-odds (現在値 ±10〜20)
- --max-market-odds (現在値 ±10〜20)
- --min-probability (0.15〜0.30の範囲)
- --candidate-pool-size (6〜20の範囲)
- --min-expected-value (0.0〜2.0の範囲)

制約:
- recommendation_bets が 5 未満のパターンは無効
- bets < 10 かつ ROI > 5.0 のパターンは overfitting の可能性があるため採用しない

### Step 4: 最良パターンの選定
recommendation_roi が最も高く制約を満たすパターンを選ぶ。
採用条件: recommendation_roi > ベースラインROI かつ recommendation_bets >= 5

### Step 5: 設定更新（改善があった場合）
採用パターンで configs/base.json の betting セクションを更新し、
artifacts/backtests/roi_history.json に本日エントリを追加し、
git commit せよ:
  git add configs/base.json artifacts/backtests/roi_history.json artifacts/backtests/today_best.json
  git commit -m 'perf: improve backtest ROI <旧ROI> → <新ROI> ($TODAY)'

### Step 6: 変化なしの場合
roi_history.json に記録し git commit せよ:
  git add artifacts/backtests/roi_history.json
  git commit -m 'chore: daily backtest log ($TODAY) - no improvement found'

### Step 7: 最終レポート
昨日と今日のROI比較表を出力せよ。

roi_history.json のフォーマット（エントリ例）:
{
  \"date\": \"$TODAY\",
  \"recommendation_roi\": <値>,
  \"recommendation_bets\": <値>,
  \"recommendation_hits\": <値>,
  \"recommendation_hit_rate\": <値>,
  \"recommendation_return\": <値>,
  \"recommendation_profit\": <値>,
  \"betting_preset\": \"monthly-roi\",
  \"backtest_file\": \"today_best.json\",
  \"params_changed\": {\"<パラメータ名>\": {\"from\": <旧値>, \"to\": <新値>}},
  \"notes\": \"改善内容の説明\"
}"

cd "$REPO_DIR"
claude "$PROMPT" 2>&1 | tee -a "$LOG_DIR/daily_improvement.log"

echo "[$(date)] Done." | tee -a "$LOG_DIR/daily_improvement.log"
