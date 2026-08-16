"""OBS000006 Phase 2: トレンド強度指標の閾値をデータ駆動で導出 (結果を見る前に確定).

OBS000006 2026-08-14追記が事前登録した式をそのまま適用する:

    ADX_threshold(pair) = percentile(ADX_dist(pair), 70)

この式を、OBS000006追記3が差し戻した3つのトレンド強度指標候補すべてに
同一のパーセンタイル(70)で適用する（候補ごとに異なるパーセンタイルを
選ぶと、それ自体が新たなHARKingの入口になるため、既存の式を使い回す）。

候補:
    C1_ADXPercentile70    : 現行ADX(非Wilder, 14, D1) の70%ile
    C2_WilderADXPercentile70: Wilder標準ADX(14, D1) の70%ile
    C3_MASpreadATRPercentile70: |SMA20-SMA50|/ATR(14,D1) の70%ile

出力: research/EXP-FX000001/10-result/trend_strength_thresholds.json
      (この値は run_train_val_test.py の C1/C2/C3 プリセットにそのまま
      ハードコードし、結果を見てから変更しない)

Usage:
    python scripts/derive_trend_strength_thresholds.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from minmax_fx_dt.strategy.indicators import adx as adx_ind
from minmax_fx_dt.strategy.indicators import adx_wilder as adx_wilder_ind
from minmax_fx_dt.strategy.indicators import ma_spread_atr_strength
from minmax_fx_dt.strategy.indicators import sma as sma_ind

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
PERCENTILE = 70  # OBS000006 2026-08-14追記で事前登録済み、変更しない
LT_SMA_SHORT, LT_SMA_LONG = 20, 50  # A1_A2_combined 系プリセットと揃える


def load_m5(pair: str) -> pd.DataFrame:
    ds1_path = ROOT / "data" / "curated" / "ds-1.json"
    with ds1_path.open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]


def to_h4(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "open": m5["open"].resample("4h").first(),
        "high": m5["high"].resample("4h").max(),
        "low": m5["low"].resample("4h").min(),
        "close": m5["close"].resample("4h").last(),
    }).dropna()


def to_d1(h4: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "open": h4["open"].resample("D").first(),
        "high": h4["high"].resample("D").max(),
        "low": h4["low"].resample("D").min(),
        "close": h4["close"].resample("D").last(),
    }).dropna()


def main() -> int:
    print(f"=== OBS000006 Phase 2: トレンド強度閾値の導出 (percentile={PERCENTILE}, 事前登録済み) ===\n")
    results = {}
    for pair in PAIRS:
        m5 = load_m5(pair)
        h4 = to_h4(m5)
        d1 = to_d1(h4)
        s20 = sma_ind(d1["close"], LT_SMA_SHORT)
        s50 = sma_ind(d1["close"], LT_SMA_LONG)

        c1 = adx_ind(d1["high"], d1["low"], d1["close"], length=14)["ADX_14"].dropna()
        c2 = adx_wilder_ind(d1["high"], d1["low"], d1["close"], length=14)["ADX_14"].dropna()
        c3 = ma_spread_atr_strength(d1["close"], s20, s50, d1["high"], d1["low"], atr_length=14).dropna()

        thresholds = {
            "C1_ADXPercentile70": round(float(np.percentile(c1, PERCENTILE)), 2),
            "C2_WilderADXPercentile70": round(float(np.percentile(c2, PERCENTILE)), 2),
            "C3_MASpreadATRPercentile70": round(float(np.percentile(c3, PERCENTILE)), 4),
        }
        results[pair] = thresholds
        print(f"[{pair}]  C1(現行ADX)={thresholds['C1_ADXPercentile70']}  "
              f"C2(Wilder ADX)={thresholds['C2_WilderADXPercentile70']}  "
              f"C3(MA乖離/ATR)={thresholds['C3_MASpreadATRPercentile70']}")

    out_dir = ROOT / "research" / "EXP-FX000001" / "10-result"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trend_strength_thresholds.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "percentile": PERCENTILE,
            "rule_registered_at": "2026-08-14 (OBS000006追記)、対象拡大は2026-08-16 (OBS000006追記3)",
            "_note": "この値は run_train_val_test.py の C1/C2/C3 プリセットにそのままハードコードし、"
                     "Train/Validation/Testいずれの結果を見た後も変更しない",
            "pairs": results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
