# boatrace-ai

ボートレースの予測・分析パイプライン。データ収集からモデル学習、予測、note.com記事自動生成までを一気通貫で行う。

## Features

- **データ収集** (`collect`): レースデータをスクレイピング・API取得
- **特徴量生成** (`build-dataset`): 生データからモデル用テーブルを構築
- **モデル学習** (`train`): LightGBMベースの予測モデルを訓練
- **予測** (`predict`): 指定日のレース結果を予測
- **note記事生成 (朝)** (`note-morning`): 予測データから有料記事HTMLを生成（回収率中心訴求）
- **note記事生成 (夜)** (`note-evening`): 結果APIと突合して無料実績記事HTMLを生成（回収率推移付き）

## Quick Start

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## CLI Usage

```bash
# パイプライン（スタブ）
boatrace-ai collect --config configs/base.json
boatrace-ai build-dataset
boatrace-ai train
boatrace-ai predict --race-date 2026-03-09

# note記事生成
boatrace-ai note-morning --race-date 2026-03-09   # 朝の有料予測記事
boatrace-ai note-evening --race-date 2026-03-09   # 夜の無料実績記事
```

### note-morning (朝の有料記事)

予測JSONを読み込み、回収率を最大の訴求ポイントとした記事HTMLを生成:
- **無料パート**: 累積回収率ヒーロー表示 → 回収率推移テーブル → 分析サマリー → 注目の穴予測 → CTA
- **有料パート**: 期待回収率TOP20テーブル → 全買い目 → 場別詳細分析 → 買い方ガイド

### note-evening (夜の無料記事)

レース結果APIから当日結果を取得し、朝の予測と突合:
- 回収率ヒーローセクション（本日 + 累積）
- 回収率推移テーブル（日別一覧）
- 的中買い目テーブル（配当・回収率付き）
- 場別回収率（ROI降順）
- 不的中一覧・AI予測精度

## Project Layout

```
boatrace-ai/
├── configs/
│   └── base.json
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
├── src/
│   └── boatrace_ai/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py              # CLI entrypoint
│       ├── collect/            # データ収集
│       ├── evaluate/           # 評価
│       ├── features/           # 特徴量生成
│       ├── note/               # note.com記事生成
│       │   ├── __init__.py
│       │   ├── morning.py      # 朝の有料予測記事
│       │   └── evening.py      # 夜の無料実績記事
│       ├── predict/            # 予測
│       └── train/              # モデル学習
├── tests/
│   └── test_cli.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Output

記事生成コマンドは `output/` ディレクトリに以下を出力:

| ファイル | 説明 |
|----------|------|
| `output/note/morning_YYYYMMDD.html` | 朝の有料記事HTML |
| `output/note/morning_YYYYMMDD_title.txt` | 朝の記事タイトル |
| `output/note/evening_YYYYMMDD.html` | 夜の無料記事HTML |
| `output/note/evening_YYYYMMDD_title.txt` | 夜の記事タイトル |
| `output/data/predictions_YYYYMMDD.json` | 予測データ |
| `output/data/verification_YYYYMMDD.json` | 検証データ |
| `output/data/cumulative_results.json` | 累積成績 |

## License

Private
