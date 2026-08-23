"""EXP-FX000009: M3エントリー層向けstop_buffer_atr_m3の再導出.

M5版の導出方法論(`backtest_vol_breakout_dow_theory.py`の
`stop_buffer_atr_m5 = round(percentile(pooled_bar_range_atr_m5, 25), 3)`)と
完全に同一のロジックをM3リサンプルバーに適用する。事前登録
(`research/EXP-FX000009/00-spec.md`): 4通貨プールのM3バーレンジ/ATR(M3,14)比の
p25パーセンタイルを機械的に採用する(選定の余地なし)。
atr_trail_multiplier_m3 = stop_buffer_atr_m3 × 1.0 (T-02公式をそのまま適用)。

出力: research/method-notes/m3_entry_params_trainonly.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

PAIRS = SELECTED_PAIRS
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
BUFFER_PERCENTILE = 25
M3_JSON_PATH = ROOT / "data" / "curated" / "ds-1-m3-train-4pairs.json"


def load_m3(pair: str) -> pd.DataFrame:
    with M3_JSON_PATH.open(encoding="utf-8") as f:
        ds = json.load(f)
    df = pd.DataFrame(ds["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]


def main() -> int:
    print("=== EXP-FX000009: stop_buffer_atr_m3 再導出 (Train期間、4通貨プール) ===\n")

    pooled_bar_range_atr_m3: list[float] = []
    for pair in PAIRS:
        m3 = load_m3(pair)
        atr_m3 = atr_ind(m3["high"], m3["low"], m3["close"], length=14)
        bar_range_atr = ((m3["high"] - m3["low"]) / atr_m3).replace([np.inf, -np.inf], np.nan).dropna()
        pooled_bar_range_atr_m3.extend(bar_range_atr.tolist())
        print(f"[{pair}] M3バー数={len(m3)}  レンジ/ATR比 p25={bar_range_atr.quantile(0.25):.3f}")

    stop_buffer_atr_m3 = round(float(np.percentile(pooled_bar_range_atr_m3, BUFFER_PERCENTILE)), 3)
    atr_trail_multiplier_m3 = round(stop_buffer_atr_m3 * 1.0, 3)

    print(f"\npooled M3バーレンジ/ATR比 n={len(pooled_bar_range_atr_m3)}件の"
          f"p{BUFFER_PERCENTILE} = {stop_buffer_atr_m3}")
    print(f"stop_buffer_atr_m3 = {stop_buffer_atr_m3} (参考: M5版=0.703)")
    print(f"atr_trail_multiplier_m3 = stop_buffer_atr_m3 × 1.0 = {atr_trail_multiplier_m3} (参考: M5版=0.703)")

    out_path = ROOT / "research" / "method-notes" / "m3_entry_params_trainonly.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "purpose": "EXP-FX000009: M3エントリー層向けstop_buffer_atr_m3・atr_trail_multiplier_m3の再導出",
            "pairs": PAIRS,
            "train_period": [TRAIN_START, TRAIN_END],
            "pooled_n_bars_for_buffer": len(pooled_bar_range_atr_m3),
            "stop_buffer_atr_m3": stop_buffer_atr_m3,
            "stop_buffer_atr_m3_percentile": BUFFER_PERCENTILE,
            "atr_trail_multiplier_m3": atr_trail_multiplier_m3,
            "stop_buffer_atr_m5_reference": 0.703,
            "_note": (
                "M5版(backtest_vol_breakout_dow_theory.pyのstop_buffer_atr_m5導出)と同一方法論"
                "(pooled 4通貨バーレンジ/ATR比のp25)をM3リサンプルバーに適用。"
                "atr_trail_multiplier_m3はT-02で確立済みの公式(stop_buffer_atr×1.0)をそのまま適用。"
                "zigzag_threshold_atr_m3は再導出せず1.0のまま据え置く(M5版も暫定固定値のため"
                "再導出の方法論が存在しない、EXP-FX000009 00-prescreen.md参照)。"
            ),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
