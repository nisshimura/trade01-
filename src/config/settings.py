"""
論文パラメータおよびティッカー定義
Nakagawa et al. (2026) SIG-FIN-036-13
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── 米国 Select Sector SPDR ETF (11 sectors) ──
US_TICKERS = [
    "XLB",   # Materials
    "XLC",   # Communication Services
    "XLE",   # Energy
    "XLF",   # Financials
    "XLI",   # Industrials
    "XLK",   # Information Technology
    "XLP",   # Consumer Staples
    "XLRE",  # Real Estate
    "XLU",   # Utilities
    "XLV",   # Health Care
    "XLY",   # Consumer Discretionary
]

# ── 日本 NEXT FUNDS TOPIX-17 業種別 ETF (17 sectors) ──
JP_TICKERS = [
    "1617",  # 食品
    "1618",  # エネルギー資源
    "1619",  # 建設・資材
    "1620",  # 素材・化学
    "1621",  # 医薬品
    "1622",  # 自動車・輸送機
    "1623",  # 鉄鋼・非鉄
    "1624",  # 機械
    "1625",  # 電機・精密
    "1626",  # 情報通信・サービスその他
    "1627",  # 電力・ガス
    "1628",  # 運輸・物流
    "1629",  # 商社・卸売
    "1630",  # 小売
    "1631",  # 銀行
    "1632",  # 金融（除く銀行）
    "1633",  # 不動産
]

# ── シクリカル / ディフェンシブ ラベル ──
US_CYCLICAL = ["XLB", "XLE", "XLF", "XLRE"]
US_DEFENSIVE = ["XLK", "XLP", "XLU", "XLV"]
JP_CYCLICAL = ["1618", "1625", "1629", "1631"]
JP_DEFENSIVE = ["1617", "1621", "1627", "1630"]

# ── モデルハイパーパラメータ ──
WINDOW_LENGTH = 60
LAMBDA_REG = 0.9
K_COMPONENTS = 3
K0_PRIOR = 3
QUANTILE_Q = 0.3

# ── データ期間 ──
SAMPLE_START = "2010-01-01"
SAMPLE_END = "2025-12-31"
CFULL_START = "2010-01-01"
CFULL_END = "2014-12-31"

# ── データソース ──
DATA_SOURCE = "stooq"
CACHE_DIR = str(PROJECT_ROOT / "data" / "processed")

# ── 出力先 ──
MODELS_DIR = str(PROJECT_ROOT / "models")
RESULTS_DIR = str(PROJECT_ROOT / "results")
