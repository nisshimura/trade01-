# t02-nakagawa

部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略の再現実装。

**論文**: Nakagawa et al. (2026) "Lead-lag strategies for Japanese and U.S. sectors using subspace regularization PCA" (SIG-FIN-036-13)

## セットアップ

```bash
uv sync
```

## 実行

### PCA ベースライン（論文再現）
```bash
uv run python -m src.main
```

### Mamba3 性能改善サイクル
```bash
PYTHONUNBUFFERED=1 uv run python -m src.pipeline.experiment
```

## プロジェクト構成

```
├── data/               # データ (raw / processed / external)
├── src/
│   ├── components/     # データローダー、モデル、バックテスト
│   ├── config/         # 設定 (ティッカー、ハイパーパラメータ)
│   ├── pipeline/       # 学習・実験パイプライン
│   ├── utils/          # ヘルパー関数
│   └── main.py         # PCA戦略 エントリポイント
├── models/             # 学習済みモデル重み
├── results/            # 評価結果・図表
└── tests/              # テスト
```

## データソース

**Stooq** (無料) を使用。Yahoo Finance は OHLC 価格データの不整合、分割/配当調整の矛盾、タイムゾーン問題のため使用しない。

- 米国: Select Sector SPDR ETF (11セクター)
- 日本: NEXT FUNDS TOPIX-17 業種別 ETF (17セクター)
