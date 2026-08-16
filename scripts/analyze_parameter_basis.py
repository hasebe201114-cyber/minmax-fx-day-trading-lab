"""OBS000006 Phase 0: パラメータ基準値のデータ駆動根拠づけ.

差し戻し1「通貨ペア別・時間軸別の基礎統計を算出するスクリプトを新設」への対応。
Train データ（spec v2.2 期間: 2023-11-01〜2025-03-31）のみを対象に、以下を算出する:

- ATR の実効 pips 分布（H4, パーセンタイル、通貨ペア別）
- Donchian レンジ幅の分布（長さ 10/20/30/50、H4、パーセンタイル）
- ADX(14) の分布（D1, パーセンタイル） — 現行閾値 20 が各ペアの分布の何%ileに
  相当するかを算出し、「GBP/JPYではADX30は上位5%にしか該当せず」型の
  スケールミスマッチを事前検知する
- M5 バーの典型値動き幅（ノイズ床）と ATR(H4) の比率（EUR/USD 型のスケール
  ミスマッチの再発防止チェック）

出力: research/EXP-FX000001/10-result/parameter_basis.json

Usage:
    python scripts/analyze_parameter_basis.py
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
from minmax_fx_dt.strategy.indicators import donchian as donchian_ind
from minmax_fx_dt.strategy.indicators import sma as sma_ind

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
DONCHIAN_LENGTHS = [10, 20, 30, 50]
PERCENTILES = [10, 25, 50, 75, 90]
ADX_CANDIDATE_THRESHOLDS = [15, 20, 25, 30, 35, 40, 45, 50]

# 現行 (教科書慣例値) パラメータ — 比較用
CURRENT_PARAMS = {
    "atr_length": 14,
    "adx_length": 14,
    "adx_threshold": 20.0,  # A1_A2_combined 等で使用
    "donchian_length": 50,  # A1_A2_combined で使用
    "lt_sma_short": 20,
    "lt_sma_long": 50,
}


def is_jpy_pair(pair: str) -> bool:
    return "JPY" in pair


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


def percentile_dict(values: pd.Series) -> dict:
    v = values.dropna()
    if len(v) == 0:
        return {}
    return {f"p{p}": round(float(np.percentile(v, p)), 4) for p in PERCENTILES}


def analyze_lt_classification_sensitivity(d1: pd.DataFrame) -> dict:
    """classify_lt_direction() が実際に使う SMA順序条件と ADX 条件を分離して、
    ADX 閾値の変化が LT 方向判定（UP/DOWN 発生率）にどれだけ効くかを検証する。

    単純な ADX の周辺分布（%ile）だけでは、「SMA 条件と AND を取った実際の
    判定にどれだけ効くか」は分からない（両条件は相関するため）。この関数は
    その相関を踏まえた限界効果を計測する。
    """
    sma_short = sma_ind(d1["close"], CURRENT_PARAMS["lt_sma_short"])
    sma_long = sma_ind(d1["close"], CURRENT_PARAMS["lt_sma_long"])
    adx_val = adx_ind(d1["high"], d1["low"], d1["close"], length=CURRENT_PARAMS["adx_length"])[
        f"ADX_{CURRENT_PARAMS['adx_length']}"
    ]

    above = (d1["close"] > sma_short) & (sma_short > sma_long)
    below = (d1["close"] < sma_short) & (sma_short < sma_long)
    sma_dir = above | below
    valid = sma_long.notna() & adx_val.notna()
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {}

    sma_only_pct = round(100 * float((sma_dir & valid).sum()) / n_valid, 1)
    by_threshold = {}
    for t in ADX_CANDIDATE_THRESHOLDS:
        combined = sma_dir & (adx_val >= t) & valid
        by_threshold[str(t)] = round(100 * float(combined.sum()) / n_valid, 1)

    return {
        "n_valid_bars": n_valid,
        "sma_order_only_pct": sma_only_pct,
        "combined_pct_by_adx_threshold": by_threshold,
    }


def analyze_pair(pair: str) -> dict:
    pip = 0.01 if is_jpy_pair(pair) else 0.0001
    m5 = load_m5(pair)
    h4 = to_h4(m5)
    d1 = to_d1(h4)

    # ATR(14) on H4, in pips
    atr_h4 = atr_ind(h4["high"], h4["low"], h4["close"], length=CURRENT_PARAMS["atr_length"]) / pip

    # Donchian range width (DCU-DCL) at each candidate length, in pips and in ATR multiples
    donchian_stats = {}
    for length in DONCHIAN_LENGTHS:
        dc = donchian_ind(h4["high"], h4["low"], lower_length=length, upper_length=length)
        width_pips = (dc["DCU"] - dc["DCL"]) / pip
        width_in_atr = width_pips / atr_h4.reindex(width_pips.index)
        donchian_stats[str(length)] = {
            "width_pips": percentile_dict(width_pips),
            "width_in_atr_multiples": percentile_dict(width_in_atr),
        }

    # ADX(14) on D1
    adx_d1 = adx_ind(d1["high"], d1["low"], d1["close"], length=CURRENT_PARAMS["adx_length"])[
        f"ADX_{CURRENT_PARAMS['adx_length']}"
    ]
    adx_valid = adx_d1.dropna()
    current_threshold_percentile = (
        float((adx_valid < CURRENT_PARAMS["adx_threshold"]).mean() * 100) if len(adx_valid) > 0 else None
    )

    # M5 ノイズ床 (バーの高安値幅) と ATR(H4) の比率
    m5_range_pips = (m5["high"] - m5["low"]) / pip
    m5_noise_median = float(m5_range_pips.median())
    atr_h4_median = float(atr_h4.dropna().median()) if atr_h4.notna().any() else float("nan")
    noise_to_atr_ratio = m5_noise_median / atr_h4_median if atr_h4_median else float("nan")

    # LT分類 (classify_lt_direction が実際に使う SMA×ADX AND条件) への ADX 閾値の限界効果
    lt_sensitivity = analyze_lt_classification_sensitivity(d1)

    return {
        "n_h4_bars": len(h4),
        "n_d1_bars": len(d1),
        "atr_h4_pips": percentile_dict(atr_h4),
        "donchian_h4": donchian_stats,
        "adx_d1": percentile_dict(adx_d1),
        "adx_current_threshold": CURRENT_PARAMS["adx_threshold"],
        "adx_current_threshold_percentile": (
            round(current_threshold_percentile, 1) if current_threshold_percentile is not None else None
        ),
        "m5_noise_floor_pips_median": round(m5_noise_median, 3),
        "atr_h4_pips_median": round(atr_h4_median, 3) if not np.isnan(atr_h4_median) else None,
        "m5_noise_to_atr_h4_ratio": round(noise_to_atr_ratio, 4) if not np.isnan(noise_to_atr_ratio) else None,
        "lt_classification_sensitivity": lt_sensitivity,
    }


def main() -> int:
    print(f"=== OBS000006 Phase 0: パラメータ基準値の基礎統計 (Train {TRAIN_START}〜{TRAIN_END}) ===\n")
    results = {}
    for pair in PAIRS:
        print(f"[{pair}]")
        stats = analyze_pair(pair)
        results[pair] = stats

        print(f"  ATR(14,H4) pips:      p10={stats['atr_h4_pips'].get('p10')}  p50={stats['atr_h4_pips'].get('p50')}  p90={stats['atr_h4_pips'].get('p90')}")
        print(f"  Donchian(50,H4) 幅:   p50={stats['donchian_h4']['50']['width_pips'].get('p50')}pips ({stats['donchian_h4']['50']['width_in_atr_multiples'].get('p50')}xATR)")
        print(f"  ADX(14,D1):           p10={stats['adx_d1'].get('p10')}  p50={stats['adx_d1'].get('p50')}  p90={stats['adx_d1'].get('p90')}")
        print(f"  現行ADX閾値(20)の位置: 下位 {stats['adx_current_threshold_percentile']}%ile")
        print(f"  M5ノイズ床/ATR(H4)比: {stats['m5_noise_to_atr_h4_ratio']}")
        sens = stats["lt_classification_sensitivity"]
        if sens:
            print(f"  LT方向判定 (D1) への限界効果: SMA順序のみ={sens['sma_order_only_pct']}%  " +
                  "  ".join(f"th{t}={sens['combined_pct_by_adx_threshold'][str(t)]}%" for t in ADX_CANDIDATE_THRESHOLDS))
        print()

    out_dir = ROOT / "research" / "EXP-FX000001" / "10-result"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parameter_basis.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "current_params_reference": CURRENT_PARAMS,
            "pairs": results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
