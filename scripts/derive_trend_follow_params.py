"""SYS-FX008 (EXP-FX000002) パラメータ導出: MA期間・トレンド強度閾値.

00-spec.md §パラメータ空間の「導出方針」を実施する。結果を見る前に確定した
ルールをそのまま適用する（HARKing防止）。

導出ルール（本スクリプト作成時点で確定、結果を見て変更しない）:
    trend_duration_median(pair) = D1のZigZag転換点間隔の中央値
        (ZigZag閾値2.0xATRはOBS000006追記6でSYS-FX007のDonchian期間導出に
        使った値をそのまま流用。指標が変わっても閾値を変えると、それ自体が
        新たなHARKingの入口になるため)
    MA_long(pair)  = round_to_standard(trend_duration_median(pair), [20,50,100,150,200])
    MA_short(pair) = round_to_standard(trend_duration_median(pair)/3, [10,20,30,50])
        (短期MAは1トレンドレグの約1/3で反応し、長期MAは1レグ全体を捉える
        という古典的なMAクロス設計比を採用)
    trend_strength_threshold(pair) = percentile(ADX(14,D1)_dist(pair), 70)
        (OBS000006 2026-08-14追記で確定したルールをそのまま再利用。
        SYS-FX007のLTレイヤと同じD1・ADX(14)・percentile=70のため、
        既存の research/EXP-FX000001/10-result/trend_strength_thresholds.json
        のC1_ADXPercentile70列をそのまま転用する)

出力: research/EXP-FX000002/10-result/trend_follow_params.json

Usage:
    python scripts/derive_trend_follow_params.py
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
from minmax_fx_dt.strategy.indicators import atr as atr_ind
from minmax_fx_dt.strategy.support_resistance import zigzag_pivot_indices

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
ZIGZAG_THRESHOLD_ATR = 2.0
MA_LONG_CANDIDATES = [20, 50, 100, 150, 200]
MA_SHORT_CANDIDATES = [10, 20, 30, 50]


def load_m5(pair: str) -> pd.DataFrame:
    ds1_path = ROOT / "data" / "curated" / "ds-1.json"
    with ds1_path.open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]


def to_d1(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "open": m5["open"].resample("D").first(),
        "high": m5["high"].resample("D").max(),
        "low": m5["low"].resample("D").min(),
        "close": m5["close"].resample("D").last(),
    }).dropna()


def round_to_standard(value: float, candidates: list[int]) -> int:
    return min(candidates, key=lambda c: abs(c - value))


def load_adx_thresholds() -> dict[str, float]:
    path = ROOT / "research" / "EXP-FX000001" / "10-result" / "trend_strength_thresholds.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {pair: v["C1_ADXPercentile70"] for pair, v in data["pairs"].items()}


def main() -> int:
    print("=== EXP-FX000002 パラメータ導出: MA期間・トレンド強度閾値 ===\n")
    adx_thresholds = load_adx_thresholds()
    results = {}

    for pair in PAIRS:
        m5 = load_m5(pair)
        d1 = to_d1(m5)
        atr_d1 = atr_ind(d1["high"], d1["low"], d1["close"], length=14)

        pivots = zigzag_pivot_indices(d1["high"], d1["low"], atr_d1, ZIGZAG_THRESHOLD_ATR)
        if len(pivots) >= 2:
            gaps = np.diff(pivots)
            duration_median = float(np.median(gaps))
        else:
            duration_median = float("nan")

        ma_long = round_to_standard(duration_median, MA_LONG_CANDIDATES) if not np.isnan(duration_median) else 50
        ma_short = round_to_standard(duration_median / 3, MA_SHORT_CANDIDATES) if not np.isnan(duration_median) else 20

        results[pair] = {
            "trend_duration_median_days": round(duration_median, 2) if not np.isnan(duration_median) else None,
            "n_pivots": len(pivots),
            "ma_long": ma_long,
            "ma_short": ma_short,
            "adx_threshold": adx_thresholds[pair],
        }
        print(f"[{pair}] トレンド持続期間中央値={duration_median:.1f}日 (n_pivots={len(pivots)}) "
              f"→ MA_short={ma_short} / MA_long={ma_long}  ADX閾値={adx_thresholds[pair]}")

    out_dir = ROOT / "research" / "EXP-FX000002" / "10-result"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trend_follow_params.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "derivation_rule": {
                "zigzag_threshold_atr": ZIGZAG_THRESHOLD_ATR,
                "ma_long_candidates": MA_LONG_CANDIDATES,
                "ma_short_candidates": MA_SHORT_CANDIDATES,
                "adx_threshold_source": "EXP-FX000001/10-result/trend_strength_thresholds.json (C1_ADXPercentile70)",
            },
            "_note": "この値はTrain/Validation/Testいずれの結果を見た後も変更しない",
            "pairs": results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
