"""SYS-FX008 (EXP-FX000002) 追加試行 E1: ATRトレーリング幅のデータ駆動導出.

初期プリセット(atr_trail_multiplier=2.0固定、データ駆動導出していなかった)の
Train結果で、EUR/USDのペイオフレシオが0.61(K4m基準1.5を大幅未達)と突出して
悪く、EUR/JPYも1.37と基準未達だった。この2ペアは平均利益が平均損失より
小さい、または僅差という逆転が起きている。

atr_trail_multiplierを、実際のLT+MT確定シグナル後のフォロースルー分布
(次15 D1バーでのMFE、ATR倍数)から導出し直す。現行の2.0は固定値で
データ駆動導出していなかった (00-spec.md未記載の暗黙のデフォルト)。

出力: research/EXP-FX000002/10-result/trail_multiplier.json
      (この値はTrain再評価の結果を見た後も変更しない)

Usage:
    python scripts/derive_trend_follow_trail_multiplier.py
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

from minmax_fx_dt.strategy.trend_following import TrendFollowConfig, evaluate_trend_follow

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
FOLLOW_THROUGH_HORIZON_D1_BARS = 15  # シグナル後のフォロースルーを測る窓 (D1本数)


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


def to_h4(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "open": m5["open"].resample("4h").first(),
        "high": m5["high"].resample("4h").max(),
        "low": m5["low"].resample("4h").min(),
        "close": m5["close"].resample("4h").last(),
    }).dropna()


def load_trend_follow_params() -> dict:
    path = ROOT / "research" / "EXP-FX000002" / "10-result" / "trend_follow_params.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)["pairs"]


def main() -> int:
    print("=== SYS-FX008 追加試行 E1: ATRトレーリング幅の導出 ===\n")
    params = load_trend_follow_params()
    all_mfe: list[float] = []

    for pair in PAIRS:
        p = params[pair]
        cfg = TrendFollowConfig(lt_sma_short=p["ma_short"], lt_sma_long=p["ma_long"],
                                 lt_trend_strength_threshold=p["adx_threshold"])
        m5 = load_m5(pair)
        d1 = to_d1(m5)
        h4 = to_h4(m5)

        from minmax_fx_dt.strategy.indicators import atr as atr_ind
        atr_d1 = atr_ind(d1["high"], d1["low"], d1["close"], length=cfg.atr_length)

        pair_mfe: list[float] = []
        min_len = max(cfg.lt_sma_long, cfg.atr_length) + 1
        for i in range(min_len, len(d1) - FOLLOW_THROUGH_HORIZON_D1_BARS):
            d1_slice = d1.iloc[: i + 1]
            confirmed_d1_ts = d1_slice.index[-1]
            h4_slice = h4.loc[h4.index <= confirmed_d1_ts]
            if len(h4_slice) < cfg.mt_confirm_sma_length:
                continue
            sig = evaluate_trend_follow(
                lt_high=d1_slice["high"], lt_low=d1_slice["low"], lt_close=d1_slice["close"],
                mt_close=h4_slice["close"], config=cfg,
            )
            if sig.direction == "NONE" or sig.atr_value <= 0:
                continue
            entry_price = d1["close"].iloc[i]
            window_high = d1["high"].iloc[i + 1: i + 1 + FOLLOW_THROUGH_HORIZON_D1_BARS]
            window_low = d1["low"].iloc[i + 1: i + 1 + FOLLOW_THROUGH_HORIZON_D1_BARS]
            if sig.direction == "UP":
                mfe = (window_high.max() - entry_price) / sig.atr_value
            else:
                mfe = (entry_price - window_low.min()) / sig.atr_value
            pair_mfe.append(mfe)

        if pair_mfe:
            print(f"[{pair}] n_signals={len(pair_mfe)}  MFE中央値={np.median(pair_mfe):.2f}xATR")
            all_mfe.extend(pair_mfe)
        else:
            print(f"[{pair}] シグナルなし")

    trail_multiplier = round(float(np.median(all_mfe)), 2) if all_mfe else 2.0
    print(f"\npooled n_signals={len(all_mfe)}  MFE中央値(pooled)={trail_multiplier}xATR")
    print(f"→ atr_trail_multiplier = {trail_multiplier} (現行2.0固定値から更新)")

    out_dir = ROOT / "research" / "EXP-FX000002" / "10-result"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trail_multiplier.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "horizon_d1_bars": FOLLOW_THROUGH_HORIZON_D1_BARS,
            "pooled_n_signals": len(all_mfe),
            "atr_trail_multiplier": trail_multiplier,
            "_note": "この値はTrain再評価の結果を見た後も変更しない",
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
