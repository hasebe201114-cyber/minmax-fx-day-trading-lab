"""OBS000006 Phase 2 (SL/TP/Donchian軸): データ駆動でパラメータを導出.

2026-08-14追記の式をそのまま適用する (結果を見る前に確定):

    SL_pips(pair)          = max(k1 * ATR_median(pair), k2 * noise_p95(pair))
    Donchian_period(pair)  = round_to_standard(median_range_duration(pair))
    TP_pips(pair)          = SL_pips(pair) * RR   # RR は全ペア共通、フォロースルー分布から1つだけ決める

k1・k2・RR の数値化方針 (本スクリプト実行前に確定、Phase 1統計や過去のバック
テスト結果を見て後から変更しない):
    - k1 = 1.5: OBS000006追記1で「1.5×ATRはEUR/USDを含む全5ペアでM15ノイズp95
      の2.4〜2.6倍相当であり妥当」と既に検証済みの値をそのまま流用 (今回の分析
      結果を見て新たに選んだ値ではない)
    - k2 = 2.5: 上記の実測倍率(2.44〜2.58)の中央付近を採用
    - RR = follow-through 分布(本スクリプトで新規算出)の pooled 中央値から決定
    - Donchian_period: 候補 [10, 20, 30, 50] のうち、ZigZag(ATR比例閾値
      2.0倍で反転を確認、Williams Fractalでは窓幅5の機械的制約でノイズに
      支配されると判明したため採用)の転換点間隔中央値に最も近いものを採用

出力: research/EXP-FX000001/10-result/sltp_donchian_params.json
      (この値は Train/Validation/Test いずれの結果を見た後も変更しない)

Usage:
    python scripts/derive_sltp_donchian_params.py
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
from minmax_fx_dt.strategy.support_resistance import zigzag_pivot_indices

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
DONCHIAN_CANDIDATES = [10, 20, 30, 50]
FOLLOW_THROUGH_HORIZON_BARS = 20  # ブレイク後にフォロースルーを測る窓 (H4本数)

# 事前登録 (結果を見る前に確定)
K1 = 1.5
K2 = 2.5
ZIGZAG_THRESHOLD_ATR = 2.0  # スイングと認定する最小反転幅 (ATR倍数)


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


def round_to_standard(value: float, candidates: list[int]) -> int:
    return min(candidates, key=lambda c: abs(c - value))


def range_duration_median(h4: pd.DataFrame, atr_h4: pd.Series) -> float:
    """ZigZag転換点の間隔(バー数)の中央値 = レンジ持続期間の代理指標."""
    pivots = zigzag_pivot_indices(h4["high"], h4["low"], atr_h4, ZIGZAG_THRESHOLD_ATR)
    if len(pivots) < 2:
        return float("nan")
    gaps = np.diff(pivots)
    return float(np.median(gaps))


def follow_through_multiples(h4: pd.DataFrame, atr_h4: pd.Series, donchian_period: int) -> np.ndarray:
    """Donchian(donchian_period)ブレイク後、次horizon本でのMFE(ATR倍数)を算出."""
    high, low = h4["high"], h4["low"]
    roll_max = high.shift(1).rolling(donchian_period, min_periods=donchian_period).max()
    roll_min = low.shift(1).rolling(donchian_period, min_periods=donchian_period).min()
    breakout_up = high > roll_max
    breakout_down = low < roll_min

    n = len(h4)
    results = []
    for i in range(n - FOLLOW_THROUGH_HORIZON_BARS):
        atr_i = atr_h4.iloc[i]
        if pd.isna(atr_i) or atr_i <= 0:
            continue
        window_high = high.iloc[i + 1: i + 1 + FOLLOW_THROUGH_HORIZON_BARS]
        window_low = low.iloc[i + 1: i + 1 + FOLLOW_THROUGH_HORIZON_BARS]
        entry_price = h4["close"].iloc[i]
        if breakout_up.iloc[i]:
            mfe = (window_high.max() - entry_price) / atr_i
            results.append(mfe)
        elif breakout_down.iloc[i]:
            mfe = (entry_price - window_low.min()) / atr_i
            results.append(mfe)
    return np.array(results)


def main() -> int:
    print("=== OBS000006 Phase 2 (SL/TP/Donchian軸): パラメータ導出 ===\n")
    print(f"事前登録: k1={K1}, k2={K2} (OBS000006追記1で既に検証済みの倍率を流用)\n")

    pair_stats = {}
    all_follow_through: dict[int, list[float]] = {c: [] for c in DONCHIAN_CANDIDATES}

    for pair in PAIRS:
        pip = 0.01 if is_jpy_pair(pair) else 0.0001
        m5 = load_m5(pair)
        h4 = to_h4(m5)

        atr_h4 = atr_ind(h4["high"], h4["low"], h4["close"], length=14)
        atr_median_pips = float((atr_h4 / pip).dropna().median())

        m5_range_pips = (m5["high"] - m5["low"]) / pip
        noise_p95_pips = float(np.percentile(m5_range_pips.dropna(), 95))

        med_gap = range_duration_median(h4, atr_h4)
        donchian_period = round_to_standard(med_gap, DONCHIAN_CANDIDATES) if not np.isnan(med_gap) else 50

        sl_pips = max(K1 * atr_median_pips, K2 * noise_p95_pips)

        pair_stats[pair] = {
            "atr_median_pips": round(atr_median_pips, 3),
            "noise_p95_pips": round(noise_p95_pips, 3),
            "fractal_gap_median_bars": round(med_gap, 2) if not np.isnan(med_gap) else None,
            "donchian_period": donchian_period,
            "sl_pips": round(sl_pips, 3),
        }
        print(f"[{pair}] ATR中央値={atr_median_pips:.1f}pips  ノイズp95={noise_p95_pips:.1f}pips  "
              f"フラクタル間隔中央値={med_gap:.1f}本→Donchian={donchian_period}  SL={sl_pips:.1f}pips")

        # フォロースルー分布 (各候補Donchian期間で算出、pooledのRR算出用)
        for c in DONCHIAN_CANDIDATES:
            ft = follow_through_multiples(h4, atr_h4, c)
            all_follow_through[c].extend(ft.tolist())

    # 各ペアで実際に採用されたDonchian期間のフォロースルー分布をpoolして中央値→RR
    pooled_ft = []
    for pair, stats in pair_stats.items():
        pooled_ft.extend(all_follow_through[stats["donchian_period"]])
    rr = round(float(np.median(pooled_ft)), 1) if pooled_ft else 2.0

    print(f"\nフォロースルー分布 (採用Donchian期間別、pooled、n={len(pooled_ft)}): 中央値={rr}xATR → RR={rr}")

    for pair, stats in pair_stats.items():
        stats["tp_pips"] = round(stats["sl_pips"] * rr, 3)

    out_dir = ROOT / "research" / "EXP-FX000001" / "10-result"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sltp_donchian_params.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "formula": "SL_pips=max(k1*ATR_median,k2*noise_p95); Donchian=round_to_standard(fractal_gap_median); TP_pips=SL_pips*RR",
            "k1": K1,
            "k2": K2,
            "rr": rr,
            "_note": "この値は Train/Validation/Test いずれの結果を見た後も変更しない",
            "pairs": pair_stats,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
