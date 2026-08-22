"""SYS-FX012検討: N_BREAKOUTで検出したエントリーが、H1のダウ理論トレンド
方向と順行/逆行のどちらだったかで、件数・勝率がどう変わるかを見る.

司令塔の質問「N_BREAKOUTのときには、トレンドと逆行した方向へのエントリーと、
トレンド方向にエントリーした場合の、エントリー数と勝率を教えて下さい」への
回答用。「トレンド」の基準は「H1におけるダウ[理論]を検知できているかどうか」
(司令塔確認済み)。

H1のダウ理論トレンド方向の判定方法:
`zigzag_pivots_typed()`(既存関数、`support_resistance.py`)をH1に適用し、
直近確定2つのHIGHピボット・2つのLOWピボットを比較する。
  - 直近HIGH > 前回HIGH かつ 直近LOW > 前回LOW → UP(Higher High/Higher Low)
  - 直近HIGH < 前回HIGH かつ 直近LOW < 前回LOW → DOWN(Lower High/Lower Low)
  - それ以外(混在パターン・ピボット不足) → 判定不能
`threshold_atr=2.0`は新規調整ではなく、SYS-FX009のH1ダブルトップ/ボトム検出
(`derive_double_pattern_params_h1.py`)で既に確立済みの「意味あるスイング」
閾値をそのまま流用する。

先読み排除: 各トレンドイベントのブレイク確定バー(`break_time`)までのH1データ
のみを使ってピボットを再計算する(そのイベント以降のデータは一切参照しない)。

正式プロトコル外の探索的診断。00-spec.md等は変更しない。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from backtest_vol_breakout_dow_theory import (  # noqa: E402
    select_non_overlapping_breakout_events, simulate_dow_theory_trend,
)
from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402
from backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd import (  # noqa: E402
    ATR_TRAIL_MULTIPLIER_M5, BREAKEVEN_TRIGGER_R, PERIODS, STOP_BUFFER_ATR_M5,
    TP_LEVELS_TRAILONLY, load_m5_period,
)
from derive_vol_breakout_entry_params import N_BREAKOUT, to_h1  # noqa: E402
from price_shock_filter import make_price_shock_check  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402
from minmax_fx_dt.strategy.support_resistance import zigzag_pivots_typed  # noqa: E402

ZIGZAG_THRESHOLD_ATR_H1 = 2.0  # SYS-FX009 H1版(derive_double_pattern_params_h1.py)と同一


def h1_dow_trend_direction(h1: pd.DataFrame, atr_h1: pd.Series, up_to_idx: int) -> str | None:
    """up_to_idx(含む)までのH1データのみでダウ理論トレンド方向を判定(先読みなし)。"""
    sub_high = h1["high"].iloc[: up_to_idx + 1]
    sub_low = h1["low"].iloc[: up_to_idx + 1]
    sub_atr = atr_h1.iloc[: up_to_idx + 1]
    pivots = zigzag_pivots_typed(sub_high, sub_low, sub_atr, ZIGZAG_THRESHOLD_ATR_H1)
    highs = [i for i, k in pivots if k == "HIGH"]
    lows = [i for i, k in pivots if k == "LOW"]
    if len(highs) < 2 or len(lows) < 2:
        return None
    h_last, h_prev = sub_high.iloc[highs[-1]], sub_high.iloc[highs[-2]]
    l_last, l_prev = sub_low.iloc[lows[-1]], sub_low.iloc[lows[-2]]
    if h_last > h_prev and l_last > l_prev:
        return "UP"
    if h_last < h_prev and l_last < l_prev:
        return "DOWN"
    return None


def summarize(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0}
    r = np.array([t["r"] for t in trades])
    win_rate = float((r > 0).mean())
    mean_r = float(r.mean())
    wins, losses = r[r > 0], r[r < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) else None
    return {"n": n, "win_rate": round(win_rate, 4), "mean_r_gross": round(mean_r, 4),
            "profit_factor": round(pf, 3) if pf else None}


def main() -> int:
    start, end = PERIODS["train"]
    m5_by_pair, h1_by_pair, atr_h1_by_pair = {}, {}, {}
    for pair in SELECTED_PAIRS:
        m5 = load_m5_period(pair, start, end)
        m5_by_pair[pair] = m5
        h1_by_pair[pair] = to_h1(m5)
        atr_h1_by_pair[pair] = atr_ind(h1_by_pair[pair]["high"], h1_by_pair[pair]["low"],
                                        h1_by_pair[pair]["close"], length=14)
    shock_check = make_price_shock_check(h1_by_pair, atr_h1_by_pair)

    aligned, counter, undetermined = [], [], []
    per_pair: dict[str, dict] = {}
    for pair, m5 in m5_by_pair.items():
        h1, atr_h1 = h1_by_pair[pair], atr_h1_by_pair[pair]
        atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)

        ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
        idxs = np.where(ratio.values >= N_BREAKOUT)[0]
        positions = [h1.index.get_loc(ratio.index[i]) for i in idxs]
        directions = ["UP" if h1.iloc[pos]["close"] > h1.iloc[pos]["open"] else "DOWN" for pos in positions]
        dedup_positions = select_non_overlapping_breakout_events(h1.index, positions, directions)
        dedup_directions = {pos: d for pos, d in zip(positions, directions)}

        pair_aligned, pair_counter, pair_undetermined = [], [], []
        for pos in dedup_positions:
            direction = dedup_directions[pos]
            h1_trend = h1_dow_trend_direction(h1, atr_h1, pos)
            trades = simulate_dow_theory_trend(
                m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER_M5,
                blackout_check=shock_check, tp_levels=TP_LEVELS_TRAILONLY, skip_first_entry=False,
                atr_trail_series=atr_m5, m5_exit=True, breakeven_trigger_r=BREAKEVEN_TRIGGER_R)
            for t in trades:
                if h1_trend is None:
                    undetermined.append(t)
                    pair_undetermined.append(t)
                elif t["direction"] == h1_trend:
                    aligned.append(t)
                    pair_aligned.append(t)
                else:
                    counter.append(t)
                    pair_counter.append(t)
        per_pair[pair] = {
            "aligned(順行)": summarize(pair_aligned),
            "counter(逆行)": summarize(pair_counter),
            "undetermined(判定不能)": summarize(pair_undetermined),
        }
        print(f"{pair}: 順行={summarize(pair_aligned)}  逆行={summarize(pair_counter)}  "
              f"判定不能={summarize(pair_undetermined)}")

    print("\n=== プール全体(Train、4通貨) ===")
    print(f"順行(トレンド方向へのエントリー): {summarize(aligned)}")
    print(f"逆行(トレンドと逆方向へのエントリー): {summarize(counter)}")
    print(f"判定不能(ピボット不足・混在パターン): {summarize(undetermined)}")

    result = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "purpose": "N_BREAKOUTエントリーをH1ダウ理論トレンド方向との順行/逆行で分類し、件数・勝率を比較",
        "method": {
            "h1_trend_definition": "zigzag_pivots_typed(H1, threshold_atr=2.0)の直近2 HIGH・2 LOWピボットで"
                                    "HH+HL→UP、LH+LL→DOWN、それ以外は判定不能。ブレイク確定バーまでのデータのみ使用(先読みなし)",
            "zigzag_threshold_atr_h1": ZIGZAG_THRESHOLD_ATR_H1,
            "threshold_source": "SYS-FX009 H1版(derive_double_pattern_params_h1.py)の既存確立値をそのまま流用",
        },
        "caveat": "正式プロトコル外の探索的診断。00-spec.md等は変更しない。r_gross(コスト控除前)ベース",
        "per_pair": per_pair,
        "pooled": {
            "aligned(順行)": summarize(aligned),
            "counter(逆行)": summarize(counter),
            "undetermined(判定不能)": summarize(undetermined),
        },
    }
    out_path = ROOT / "research" / "method-notes" / "n_breakout_h1_dow_trend_alignment.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
