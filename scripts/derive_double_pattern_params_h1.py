"""EXP-FX000003 派生検討: ダブルトップ/ボトム検出パラメータをH1で導出.

背景: 司令塔から「H4の代表例(継続文脈)がレンジの中の値動きにも見え、綺麗な
ダブルボトムと言い切れない」という指摘を受けた。あわせて「H4→H1に変更して
サンプル数nを増やし、綺麗なダブルボトムを検出する」との依頼。

本スクリプトは `derive_double_pattern_params.py`(H4版)と全く同じ導出方法論
(ZigZag閾値2.0xATR固定・許容誤差はp25・ストップバッファはp25・遅延はp90)を、
時間軸をH1に変えて再適用する。H4→H1でバー粒度が4倍になるため、実時間で同じ
窓を保つべきパラメータ(ネックライン割れ探索の打ち切り・トレーリング幅算出の
MFE観測窓)は本数を4倍にスケールする。それ以外の閾値(百分位・ZigZag閾値2.0)
はH4版から変更しない(結果を見る前に固定)。

Usage:
    python scripts/derive_double_pattern_params_h1.py

出力: research/EXP-FX000003/10-result/double_pattern_params_h1.json
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

from minmax_fx_dt.strategy.indicators import atr as atr_ind
from minmax_fx_dt.strategy.support_resistance import zigzag_pivots_typed

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"

# 事前登録 (結果を見る前に確定。H4版から変更しない値はそのまま踏襲)
ZIGZAG_THRESHOLD_ATR = 2.0
TOLERANCE_PERCENTILE = 25
BUFFER_PERCENTILE = 25
STALENESS_PERCENTILE = 90
BREAK_SEARCH_CAP_BARS = 60 * 4  # H4の60本(10日)と同じ実時間窓をH1本数で確保
STALENESS_CANDIDATES = [10, 15, 20, 30, 40, 60, 80, 120]  # H1粒度に合わせ候補を拡張
TRAIL_MFE_HORIZON_BARS = 30 * 4  # H4の30本(5日)と同じ実時間窓
LT_SMA_SHORT, LT_SMA_LONG = 10, 20


def load_m5(pair: str) -> pd.DataFrame:
    ds1_path = ROOT / "data" / "curated" / "ds-1.json"
    with ds1_path.open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]


def to_h1(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "open": m5["open"].resample("1h").first(),
        "high": m5["high"].resample("1h").max(),
        "low": m5["low"].resample("1h").min(),
        "close": m5["close"].resample("1h").last(),
    }).dropna()


def to_d1(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "open": m5["open"].resample("D").first(),
        "high": m5["high"].resample("D").max(),
        "low": m5["low"].resample("D").min(),
        "close": m5["close"].resample("D").last(),
    }).dropna()


def lt_direction_series(d1: pd.DataFrame) -> pd.Series:
    sma_short = d1["close"].rolling(LT_SMA_SHORT, min_periods=LT_SMA_SHORT).mean()
    sma_long = d1["close"].rolling(LT_SMA_LONG, min_periods=LT_SMA_LONG).mean()
    direction = pd.Series("NONE", index=d1.index, dtype="object")
    direction = direction.mask(sma_short > sma_long, "UP")
    direction = direction.mask(sma_short < sma_long, "DOWN")
    return direction


def round_to_standard(value: float, candidates: list[int]) -> int:
    return min(candidates, key=lambda c: abs(c - value))


def alternating_triplets(pivots: list[tuple[int, str]]) -> list[tuple[int, str, int, str, int, str]]:
    triplets = []
    for i in range(2, len(pivots)):
        idx1, kind1 = pivots[i - 2]
        idx2, kind2 = pivots[i - 1]
        idx3, kind3 = pivots[i]
        if kind1 == kind3 and kind1 != kind2:
            triplets.append((idx1, kind1, idx2, kind2, idx3, kind3))
    return triplets


def main() -> int:
    print("=== EXP-FX000003派生: ダブルパターン検出パラメータ導出 (H1版) ===\n")
    print(f"事前登録: zigzag_threshold_atr={ZIGZAG_THRESHOLD_ATR}(H4版と同一), "
          f"tolerance_percentile=p{TOLERANCE_PERCENTILE}, buffer_percentile=p{BUFFER_PERCENTILE}, "
          f"staleness_percentile=p{STALENESS_PERCENTILE}, "
          f"break_search_cap={BREAK_SEARCH_CAP_BARS}本(H4の60本と同じ実時間), "
          f"trail_mfe_horizon={TRAIL_MFE_HORIZON_BARS}本(H4の30本と同じ実時間)\n")

    pooled_delta_atr: list[float] = []
    pooled_bar_range_atr: list[float] = []
    pair_stats: dict[str, dict] = {}
    pivots_cache: dict[str, list] = {}
    h1_cache: dict[str, pd.DataFrame] = {}
    atr_cache: dict[str, pd.Series] = {}

    for pair in PAIRS:
        m5 = load_m5(pair)
        h1 = to_h1(m5)
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        h1_cache[pair] = h1
        atr_cache[pair] = atr_h1

        pivots = zigzag_pivots_typed(h1["high"], h1["low"], atr_h1, ZIGZAG_THRESHOLD_ATR)
        pivots_cache[pair] = pivots
        triplets = alternating_triplets(pivots)

        pair_delta_atr: list[float] = []
        for idx1, kind1, idx2, _kind2, idx3, _kind3 in triplets:
            atr_neckline = atr_h1.iloc[idx2]
            if pd.isna(atr_neckline) or atr_neckline <= 0:
                continue
            p1 = float(h1["high" if kind1 == "HIGH" else "low"].iloc[idx1])
            p2 = float(h1["high" if kind1 == "HIGH" else "low"].iloc[idx3])
            pair_delta_atr.append(abs(p1 - p2) / float(atr_neckline))

        bar_range_atr = ((h1["high"] - h1["low"]) / atr_h1).replace([np.inf, -np.inf], np.nan).dropna()

        pair_stats[pair] = {
            "n_pivots": len(pivots),
            "n_alternating_triplets": len(triplets),
            "delta_atr_median": round(float(np.median(pair_delta_atr)), 3) if pair_delta_atr else None,
        }
        print(f"[{pair}] ピボット数={len(pivots)}  交互3点組={len(triplets)}件  "
              f"ΔP/ATR中央値={pair_stats[pair]['delta_atr_median']}")

        pooled_delta_atr.extend(pair_delta_atr)
        pooled_bar_range_atr.extend(bar_range_atr.tolist())

    pattern_tolerance_atr = round(float(np.percentile(pooled_delta_atr, TOLERANCE_PERCENTILE)), 3)
    stop_buffer_atr = round(float(np.percentile(pooled_bar_range_atr, BUFFER_PERCENTILE)), 3)
    print(f"\npooled n_triplets={len(pooled_delta_atr)}  "
          f"pattern_tolerance_atr(p{TOLERANCE_PERCENTILE})={pattern_tolerance_atr}")
    print(f"pooled n_bars={len(pooled_bar_range_atr)}  "
          f"stop_buffer_atr(p{BUFFER_PERCENTILE})={stop_buffer_atr}")

    pooled_lags: list[int] = []
    pooled_mfe: list[float] = []
    for pair in PAIRS:
        h1 = h1_cache[pair]
        atr_h1 = atr_cache[pair]
        m5 = load_m5(pair)
        d1 = to_d1(m5)
        lt_dir = lt_direction_series(d1)
        triplets = alternating_triplets(pivots_cache[pair])

        pair_mfe: list[float] = []
        for idx1, kind1, idx2, _kind2, idx3, _kind3 in triplets:
            atr_neckline = atr_h1.iloc[idx2]
            if pd.isna(atr_neckline) or atr_neckline <= 0:
                continue
            p1 = float(h1["high" if kind1 == "HIGH" else "low"].iloc[idx1])
            p2 = float(h1["high" if kind1 == "HIGH" else "low"].iloc[idx3])
            if abs(p1 - p2) / float(atr_neckline) > pattern_tolerance_atr:
                continue
            neckline = float(h1["low" if kind1 == "HIGH" else "high"].iloc[idx2])
            required_lt = "DOWN" if kind1 == "HIGH" else "UP"
            search_end = min(idx3 + 1 + BREAK_SEARCH_CAP_BARS, len(h1))
            for j in range(idx3 + 1, search_end):
                broke = (
                    (kind1 == "HIGH" and float(h1["low"].iloc[j]) < neckline)
                    or (kind1 == "LOW" and float(h1["high"].iloc[j]) > neckline)
                )
                if not broke:
                    continue
                pooled_lags.append(j - idx3)
                lt_at_break = lt_dir.asof(h1.index[j])
                atr_entry = atr_h1.iloc[j]
                if lt_at_break != required_lt or pd.isna(atr_entry) or atr_entry <= 0:
                    break
                entry_price = float(h1["close"].iloc[j])
                window_high = h1["high"].iloc[j + 1: j + 1 + TRAIL_MFE_HORIZON_BARS]
                window_low = h1["low"].iloc[j + 1: j + 1 + TRAIL_MFE_HORIZON_BARS]
                if len(window_high) == 0:
                    break
                if kind1 == "HIGH":
                    mfe = (entry_price - float(window_low.min())) / float(atr_entry)
                else:
                    mfe = (float(window_high.max()) - entry_price) / float(atr_entry)
                pair_mfe.append(mfe)
                break

        if pair_mfe:
            print(f"[{pair}] LT一致シグナル={len(pair_mfe)}件  MFE中央値={np.median(pair_mfe):.2f}xATR")
        pooled_mfe.extend(pair_mfe)

    staleness_raw = float(np.percentile(pooled_lags, STALENESS_PERCENTILE)) if pooled_lags else 80.0
    max_bars_since_second_pivot = round_to_standard(staleness_raw, STALENESS_CANDIDATES)
    print(f"\nネックライン割れまでの遅延 (許容誤差内で一致した{len(pooled_lags)}件、"
          f"打ち切り{BREAK_SEARCH_CAP_BARS}本): p{STALENESS_PERCENTILE}={staleness_raw:.1f}本 "
          f"→ max_bars_since_second_pivot={max_bars_since_second_pivot}")

    atr_trail_multiplier = round(float(np.median(pooled_mfe)), 2) if pooled_mfe else 2.0
    print(f"\npooled LT一致シグナル={len(pooled_mfe)}件  MFE中央値(pooled、horizon={TRAIL_MFE_HORIZON_BARS}本)="
          f"{atr_trail_multiplier}xATR → atr_trail_multiplier = {atr_trail_multiplier}")

    out_dir = ROOT / "research" / "EXP-FX000003" / "10-result"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "double_pattern_params_h1.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "timeframe": "H1",
            "zigzag_threshold_atr": ZIGZAG_THRESHOLD_ATR,
            "pattern_tolerance_atr": pattern_tolerance_atr,
            "stop_buffer_atr": stop_buffer_atr,
            "max_bars_since_second_pivot": max_bars_since_second_pivot,
            "staleness_raw_p90_bars": round(staleness_raw, 2),
            "break_search_cap_bars": BREAK_SEARCH_CAP_BARS,
            "atr_trail_multiplier": atr_trail_multiplier,
            "trail_mfe_horizon_bars": TRAIL_MFE_HORIZON_BARS,
            "lt_sma_short": LT_SMA_SHORT,
            "lt_sma_long": LT_SMA_LONG,
            "pooled_n_triplets": len(pooled_delta_atr),
            "pooled_n_break_lags": len(pooled_lags),
            "pooled_n_lt_matched_signals": len(pooled_mfe),
            "_note": "H4版(double_pattern_params.json)と同じ導出方法論をH1に適用したもの。この値はTrain/Validation/Testいずれの結果を見た後も変更しない",
            "pairs": pair_stats,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
