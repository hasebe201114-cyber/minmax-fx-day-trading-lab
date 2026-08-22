"""SYS-FX012検討: Donchian(20)継続+CALM_RATIO質フィルターで拾った「N_BREAKOUT
に含まれない新規追加分」のエントリーを、H1ダウ理論トレンド方向との順行/逆行で
分類する(N_BREAKOUT側の同種分析`analyze_n_breakout_h1_dow_trend_alignment.py`
との直接比較用)。

対象母集団の定義: Donchian(20)ブレイク かつ range/ATR>=CALM_RATIO(2.0) だが
range/ATR>=N_BREAKOUT(3.5)ではない(=既存のN_BREAKOUT検出に含まれない、
ハイブリッド設計が新規に追加する分のみ)バー。この「純粋な追加分」が、
Trainベースライン評価でトレード当たりの質を希釈した主犯であるため、その質を
H1トレンド順行/逆行で分解できるかを見る。

H1ダウ理論トレンド方向の判定方法はN_BREAKOUT側の分析と完全に同一
(zigzag_pivots_typed、threshold_atr=2.0、先読みなし)。

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

from analyze_n_breakout_h1_dow_trend_alignment import (  # noqa: E402
    ZIGZAG_THRESHOLD_ATR_H1, h1_dow_trend_direction, summarize,
)
from backtest_vol_breakout_dow_theory import (  # noqa: E402
    select_non_overlapping_breakout_events, simulate_dow_theory_trend,
)
from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402
from backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd import (  # noqa: E402
    ATR_TRAIL_MULTIPLIER_M5, BREAKEVEN_TRIGGER_R, PERIODS, STOP_BUFFER_ATR_M5,
    TP_LEVELS_TRAILONLY, load_m5_period,
)
from backtest_vol_continuation_hybrid_4pairs_trainonly import DONCHIAN_LENGTH  # noqa: E402
from derive_vol_breakout_entry_params import N_BREAKOUT, to_h1  # noqa: E402
from price_shock_filter import CALM_RATIO, make_price_shock_check  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402
from minmax_fx_dt.strategy.indicators import donchian  # noqa: E402


def detect_continuation_only_events(h1: pd.DataFrame, atr_h1: pd.Series) -> tuple[list[int], list[str]]:
    """Donchian継続+CALM_RATIOは満たすがN_BREAKOUTスパイクではない(=新規追加分のみ)バーを検出。"""
    dc = donchian(h1["high"], h1["low"], DONCHIAN_LENGTH, DONCHIAN_LENGTH).shift(1)
    ratio = (h1["high"] - h1["low"]) / atr_h1
    close = h1["close"]

    is_spike = ratio >= N_BREAKOUT
    is_cont_up = (close > dc["DCU"]) & (ratio >= CALM_RATIO) & (~is_spike)
    is_cont_down = (close < dc["DCL"]) & (ratio >= CALM_RATIO) & (~is_spike)
    up = is_cont_up.fillna(False)
    down = is_cont_down.fillna(False)

    positions, directions = [], []
    for i in range(len(h1)):
        if bool(up.iloc[i]):
            positions.append(i)
            directions.append("UP")
        elif bool(down.iloc[i]):
            positions.append(i)
            directions.append("DOWN")
    return positions, directions


def main() -> int:
    print(f"検出対象: Donchian({DONCHIAN_LENGTH})継続 AND range/ATR>=CALM_RATIO({CALM_RATIO}) "
          f"AND NOT(range/ATR>=N_BREAKOUT({N_BREAKOUT}))  ※N_BREAKOUTに含まれない新規追加分のみ")
    print(f"H1トレンド判定: zigzag threshold_atr={ZIGZAG_THRESHOLD_ATR_H1}(SYS-FX009 H1版と同一)\n")

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
    n_events_raw_total = n_events_dedup_total = 0
    for pair, m5 in m5_by_pair.items():
        h1, atr_h1 = h1_by_pair[pair], atr_h1_by_pair[pair]
        atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)

        positions, directions = detect_continuation_only_events(h1, atr_h1)
        n_events_raw_total += len(positions)
        dedup_positions = select_non_overlapping_breakout_events(h1.index, positions, directions)
        n_events_dedup_total += len(dedup_positions)
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
            "n_events_raw": len(positions), "n_events_dedup": len(dedup_positions),
            "aligned(順行)": summarize(pair_aligned),
            "counter(逆行)": summarize(pair_counter),
            "undetermined(判定不能)": summarize(pair_undetermined),
        }
        print(f"{pair}: イベント={len(positions)}件(dedup後{len(dedup_positions)})  "
              f"順行={summarize(pair_aligned)}  逆行={summarize(pair_counter)}  "
              f"判定不能={summarize(pair_undetermined)}")

    print(f"\n=== プール全体(Train、4通貨、Donchian継続の新規追加分のみ) ===")
    print(f"イベント総数={n_events_raw_total}(dedup後{n_events_dedup_total})")
    print(f"順行(H1トレンド方向へのエントリー): {summarize(aligned)}")
    print(f"逆行(H1トレンドと逆方向へのエントリー): {summarize(counter)}")
    print(f"判定不能(ピボット不足・混在パターン): {summarize(undetermined)}")

    result = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "purpose": "Donchian継続+CALM_RATIOで拾った、N_BREAKOUTに含まれない新規追加分のエントリーを"
                   "H1ダウ理論トレンド方向との順行/逆行で分類し、N_BREAKOUT側の分析と比較する",
        "method": {
            "detection": f"Donchian({DONCHIAN_LENGTH})継続 AND range/ATR>=CALM_RATIO({CALM_RATIO}) "
                         f"AND NOT(range/ATR>=N_BREAKOUT({N_BREAKOUT}))",
            "h1_trend_definition": "analyze_n_breakout_h1_dow_trend_alignment.pyと完全に同一",
        },
        "caveat": "正式プロトコル外の探索的診断。00-spec.md等は変更しない。r_gross(コスト控除前)ベース",
        "n_events_raw": n_events_raw_total, "n_events_dedup": n_events_dedup_total,
        "per_pair": per_pair,
        "pooled": {
            "aligned(順行)": summarize(aligned),
            "counter(逆行)": summarize(counter),
            "undetermined(判定不能)": summarize(undetermined),
        },
        "comparison_source": "research/method-notes/n_breakout_h1_dow_trend_alignment.json",
    }
    out_path = ROOT / "research" / "method-notes" / "donchian_continuation_h1_dow_trend_alignment.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
