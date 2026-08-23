"""EXP-FX000009 追加分析(2026-08-23、司令塔依頼): M3再導出版がH1版に対して
mean_r_net低下・permutation_p非有意化を起こした理由を、トレンドイベント単位で
M5版とM3版を突き合わせて特定する。「M5では拾いきれなかったエントリー機会」
(機会面)と、そこで生まれたトレードの収益性(収益面)を分離して評価する。

**正式なEXP-FX000009の判定基準ではない(追加の原因分析)。フォワードテスト中の
凍結設計には一切影響しない。**

方法: 検出層・トレンド判定フィルター(H1、両パイプライン共通・不変)で得られる
85件のトレンドイベントそれぞれについて、M5版パラメータ(stop_buffer_atr_m5=0.703)
とM3版再導出パラメータ(stop_buffer_atr_m3=0.7)でsimulate_dow_theory_trend()を
individually実行し、`break_time`(イベント識別子)でトレードを対応付ける。
`res`の生出力に含まれる`entry_seq`・`break_time`をそのまま利用する(既存の
最終出力JSONでは欠落しているフィールド)。

出力: research/method-notes/m3_vs_m5_entry_opportunities.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from analyze_n_breakout_h1_dow_trend_alignment import h1_dow_trend_direction  # noqa: E402
from backtest_vol_breakout_dow_theory import (  # noqa: E402
    select_non_overlapping_breakout_events, simulate_dow_theory_trend,
)
from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402
from backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd import (  # noqa: E402
    BREAKEVEN_TRIGGER_R, COMMISSION_RATE_ROUND_TRIP, N_BREAKOUT, PERIODS,
    SLIPPAGE_PIPS_MARKET_LEG, SLIPPAGE_PIPS_STOP_TRIGGERED, SPREAD_PIPS,
    STOP_BUFFER_ATR_M5, TP_LEVELS_TRAILONLY, load_m5_period, pip_size, to_h1,
)
from backtest_m3_rederived_params_trainonly import (  # noqa: E402
    ATR_TRAIL_MULTIPLIER_M3, STOP_BUFFER_ATR_M3,
)
from derive_vol_breakout_entry_params import N_BREAKOUT as _NB  # noqa: E402,F401
from explore_m3_entry_trainonly import load_m3_period  # noqa: E402
from price_shock_filter import make_price_shock_check  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

ATR_TRAIL_MULTIPLIER_M5_LOCAL = STOP_BUFFER_ATR_M5 * 1.0


def compute_r_net(sim: dict, pair: str) -> float:
    spread = SPREAD_PIPS.get(pair, 0.5)
    pip = pip_size(pair)
    fraction_via_tp = 0.0  # TP_LEVELS_TRAILONLY=[]なので常に0
    fraction_remaining = 1.0
    remaining_is_market = sim["exit_reason"] in ("WEEKEND_NO_TP", "TP_THEN_WEEKEND", "MAX_HOLD")
    remaining_is_stop_triggered = sim["exit_reason"] in ("SL_INITIAL_NO_TP", "TP_THEN_SL_TRAIL")
    entry_pips = spread + SLIPPAGE_PIPS_MARKET_LEG
    if remaining_is_market:
        exit_slippage = fraction_remaining * SLIPPAGE_PIPS_MARKET_LEG
    elif remaining_is_stop_triggered:
        exit_slippage = fraction_remaining * SLIPPAGE_PIPS_STOP_TRIGGERED
    else:
        exit_slippage = 0.0
    exit_pips = spread + exit_slippage
    cost_price = (entry_pips + exit_pips) * pip
    cost_r = cost_price / sim["initial_risk"]
    leverage_ratio = sim["entry_price"] / sim["initial_risk"]
    commission_r = COMMISSION_RATE_ROUND_TRIP * leverage_ratio
    return sim["r"] - cost_r - commission_r


def main() -> int:
    start, end = PERIODS["train"]
    print("=== EXP-FX000009追加分析: M5版 vs M3版をトレンドイベント単位で突き合わせ ===\n")

    events_summary: dict[str, dict] = {}
    exit_reason_counts_m5: dict[str, int] = defaultdict(int)
    exit_reason_counts_m3: dict[str, int] = defaultdict(int)

    h1_by_pair, atr_h1_by_pair = {}, {}
    m5_by_pair, m3_by_pair = {}, {}
    for pair in SELECTED_PAIRS:
        m5 = load_m5_period(pair, start, end)
        m3 = load_m3_period(pair, start, end)
        if len(m5) < 1000 or len(m3) < 1000:
            continue
        m5_by_pair[pair] = m5
        m3_by_pair[pair] = m3
        h1 = to_h1(m5)
        h1_by_pair[pair] = h1
        atr_h1_by_pair[pair] = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    shock_check = make_price_shock_check(h1_by_pair, atr_h1_by_pair)

    n_events_total = 0
    for pair in m5_by_pair:
        m5, m3 = m5_by_pair[pair], m3_by_pair[pair]
        h1, atr_h1 = h1_by_pair[pair], atr_h1_by_pair[pair]
        atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)
        atr_m3 = atr_ind(m3["high"], m3["low"], m3["close"], length=14)

        ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
        idxs = np.where(ratio.values >= N_BREAKOUT)[0]
        positions = [h1.index.get_loc(ratio.index[i]) for i in idxs]
        directions = ["UP" if h1.iloc[pos]["close"] > h1.iloc[pos]["open"] else "DOWN" for pos in positions]
        dedup_positions = select_non_overlapping_breakout_events(h1.index, positions, directions)
        dedup_directions = {pos: d for pos, d in zip(positions, directions)}

        for pos in dedup_positions:
            trend = h1_dow_trend_direction(h1, atr_h1, pos)
            if trend is None:
                continue
            direction = dedup_directions[pos]
            n_events_total += 1

            trades_m5_raw = simulate_dow_theory_trend(
                m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER_M5_LOCAL,
                blackout_check=shock_check, tp_levels=TP_LEVELS_TRAILONLY, skip_first_entry=False,
                atr_trail_series=atr_m5, m5_exit=True, breakeven_trigger_r=BREAKEVEN_TRIGGER_R)
            trades_m3_raw = simulate_dow_theory_trend(
                m3, atr_m3, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M3, ATR_TRAIL_MULTIPLIER_M3,
                blackout_check=shock_check, tp_levels=TP_LEVELS_TRAILONLY, skip_first_entry=False,
                atr_trail_series=atr_m3, m5_exit=True, breakeven_trigger_r=BREAKEVEN_TRIGGER_R)

            break_time = h1.index[pos]
            event_key = f"{pair}__{break_time.isoformat()}"

            r_net_m5_list = [compute_r_net(t, pair) for t in trades_m5_raw]
            r_net_m3_list = [compute_r_net(t, pair) for t in trades_m3_raw]
            for t in trades_m5_raw:
                exit_reason_counts_m5[t["exit_reason"]] += 1
            for t in trades_m3_raw:
                exit_reason_counts_m3[t["exit_reason"]] += 1

            events_summary[event_key] = {
                "pair": pair, "direction": direction, "break_time": str(break_time),
                "n_trades_m5": len(trades_m5_raw), "n_trades_m3": len(trades_m3_raw),
                "sum_r_net_m5": round(sum(r_net_m5_list), 4),
                "sum_r_net_m3": round(sum(r_net_m3_list), 4),
                "first_entry_time_m5": trades_m5_raw[0]["entry_time"] if trades_m5_raw else None,
                "first_entry_time_m3": trades_m3_raw[0]["entry_time"] if trades_m3_raw else None,
                "first_entry_price_m5": trades_m5_raw[0]["entry_price"] if trades_m5_raw else None,
                "first_entry_price_m3": trades_m3_raw[0]["entry_price"] if trades_m3_raw else None,
            }
        print(f"[{pair}] 処理完了")

    # 分類集計
    more_in_m3 = [e for e in events_summary.values() if e["n_trades_m3"] > e["n_trades_m5"]]
    fewer_in_m3 = [e for e in events_summary.values() if e["n_trades_m3"] < e["n_trades_m5"]]
    same_count = [e for e in events_summary.values() if e["n_trades_m3"] == e["n_trades_m5"]]
    m5_zero_m3_positive = [e for e in events_summary.values() if e["n_trades_m5"] == 0 and e["n_trades_m3"] > 0]
    m3_zero_m5_positive = [e for e in events_summary.values() if e["n_trades_m3"] == 0 and e["n_trades_m5"] > 0]

    total_n_m5 = sum(e["n_trades_m5"] for e in events_summary.values())
    total_n_m3 = sum(e["n_trades_m3"] for e in events_summary.values())
    total_r_net_m5 = sum(e["sum_r_net_m5"] for e in events_summary.values())
    total_r_net_m3 = sum(e["sum_r_net_m3"] for e in events_summary.values())

    def _summarize_group(group: list[dict], label: str) -> dict:
        diffs = [e["sum_r_net_m3"] - e["sum_r_net_m5"] for e in group]
        return {
            "label": label, "n_events": len(group),
            "sum_r_net_diff_total": round(sum(diffs), 4),
            "mean_r_net_diff_per_event": round(float(np.mean(diffs)), 4) if diffs else None,
            "n_diff_positive": sum(1 for d in diffs if d > 0),
            "n_diff_negative": sum(1 for d in diffs if d < 0),
        }

    print(f"\n=== 集計(全{n_events_total}イベント) ===")
    print(f"M5合計トレード数: {total_n_m5}  M3合計トレード数: {total_n_m3}  (差分+{total_n_m3-total_n_m5})")
    print(f"M5合計r_net: {round(total_r_net_m5,3)}  M3合計r_net: {round(total_r_net_m3,3)}  "
          f"(差分{round(total_r_net_m3-total_r_net_m5,3)})")
    print(f"\nM3の方がトレード数が多いイベント: {len(more_in_m3)}件")
    print(f"M3の方がトレード数が少ないイベント: {len(fewer_in_m3)}件")
    print(f"トレード数が同じイベント: {len(same_count)}件")
    print(f"M5で0件・M3で1件以上(M5が完全に見逃した機会): {len(m5_zero_m3_positive)}件")
    print(f"M3で0件・M5で1件以上(M3が完全に見逃した機会): {len(m3_zero_m5_positive)}件")

    print("\nExit理由分布 M5:", dict(exit_reason_counts_m5))
    print("Exit理由分布 M3:", dict(exit_reason_counts_m3))

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "purpose": "EXP-FX000009追加分析。M3再導出版がH1(M5)版よりmean_r_netが低下し"
                  "permutation_pが非有意化した理由を、トレンドイベント単位でM5版とM3版を"
                  "突き合わせて特定する。正式な判定基準ではない、追加の原因分析",
        "n_events_total": n_events_total,
        "total_n_trades_m5": total_n_m5,
        "total_n_trades_m3": total_n_m3,
        "total_r_net_m5": round(total_r_net_m5, 4),
        "total_r_net_m3": round(total_r_net_m3, 4),
        "total_r_net_diff": round(total_r_net_m3 - total_r_net_m5, 4),
        "group_more_trades_in_m3": _summarize_group(more_in_m3, "M3の方がトレード数が多い"),
        "group_fewer_trades_in_m3": _summarize_group(fewer_in_m3, "M3の方がトレード数が少ない"),
        "group_same_trade_count": _summarize_group(same_count, "トレード数が同じ"),
        "n_m5_zero_m3_positive": len(m5_zero_m3_positive),
        "n_m3_zero_m5_positive": len(m3_zero_m5_positive),
        "exit_reason_counts_m5": dict(exit_reason_counts_m5),
        "exit_reason_counts_m3": dict(exit_reason_counts_m3),
        "events": events_summary,
    }
    out_path = ROOT / "research" / "method-notes" / "m3_vs_m5_entry_opportunities.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
