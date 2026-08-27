"""EXP-FX000017(SYS-FX023): ニュース窓(BOJ/FOMC)限定でのSYS-FX011凍結ロジック
評価(Train単独、正式KPIパイプライン).

事前登録(`research/EXP-FX000017/00-spec.md`): 候補①(BOJ+FOMC窓、既存
economic_calendar.build_blackout_windows()の既定)・候補②(BOJ単独窓)の2通りを
比較する。検出層(N_BREAKOUT=3.5)・エントリー層(M5ダウ理論連続追跡)・出口
(トレール専業)・コストモデルはSYS-FX011のT-13確定版と完全に同一。価格反応型
ショック抑制フィルターは目的が矛盾するため適用しない。新規パラメータの導入は
一切なし(窓によるイベントフィルタリングのみが新規要素)。

出力: research/method-notes/sysfx023_calendar_window_trainonly_backtest.json
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
    ATR_TRAIL_MULTIPLIER_M5, BREAKEVEN_TRIGGER_R, COMMISSION_RATE_ROUND_TRIP,
    INITIAL_CAPITAL_USD, MAX_LEVERAGE, N_BREAKOUT, PERIODS, RISK_PCT_PER_TRADE,
    SLIPPAGE_PIPS_MARKET_LEG, SLIPPAGE_PIPS_STOP_TRIGGERED, SPREAD_PIPS,
    STOP_BUFFER_ATR_M5, TP_CUM_FRACTION, TP_LEVELS_TRAILONLY, load_m5_period,
    pip_size, to_h1,
)
from economic_calendar import BOJ_MEETINGS, build_blackout_windows  # noqa: E402
from evaluate_vol_breakout_dow_theory_kpi import evaluate_period  # noqa: E402
from minmax_fx_dt.backtest.permutation import permutation_test_block  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

CANDIDATES = {
    "candidate1_boj_fomc": None,          # None = build_blackout_windows()既定(BOJ+FOMC)
    "candidate2_boj_only": BOJ_MEETINGS,  # BOJ単独
}


def in_any_window(t: pd.Timestamp, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> bool:
    t = t.tz_localize("Asia/Tokyo") if t.tzinfo is None else t
    return any(w0 <= t <= w1 for w0, w1 in windows)


def find_trades(pair: str, m5: pd.DataFrame, windows) -> list[dict]:
    h1 = to_h1(m5)
    atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)

    ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
    idxs = np.where(ratio.values >= N_BREAKOUT)[0]
    positions = [h1.index.get_loc(ratio.index[i]) for i in idxs]
    directions = ["UP" if h1.iloc[pos]["close"] > h1.iloc[pos]["open"] else "DOWN" for pos in positions]
    dedup_positions = select_non_overlapping_breakout_events(h1.index, positions, directions)
    dedup_directions = {pos: d for pos, d in zip(positions, directions)}

    trades = []
    for pos in dedup_positions:
        if not in_any_window(h1.index[pos], windows):
            continue
        direction = dedup_directions[pos]
        trades.extend(simulate_dow_theory_trend(
            m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER_M5,
            blackout_check=None, tp_levels=TP_LEVELS_TRAILONLY, skip_first_entry=False,
            atr_trail_series=atr_m5, m5_exit=True, breakeven_trigger_r=BREAKEVEN_TRIGGER_R))
    return trades


def run_period(windows, start: str, end: str) -> dict:
    all_trades: list[dict] = []
    for pair in SELECTED_PAIRS:
        m5 = load_m5_period(pair, start, end)
        if len(m5) < 1000:
            continue
        trades = find_trades(pair, m5, windows)
        spread = SPREAD_PIPS.get(pair, 0.5)
        pip = pip_size(pair)
        for sim in trades:
            fraction_via_tp = TP_CUM_FRACTION[sim["n_levels_hit"]]
            fraction_remaining = 1.0 - fraction_via_tp
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
            r_net = sim["r"] - cost_r - commission_r
            all_trades.append({
                "pair": pair, "direction": sim["direction"],
                "entry_time": pd.Timestamp(sim["entry_time"]), "exit_time": sim["exit_time"],
                "entry_price": sim["entry_price"], "initial_risk": sim["initial_risk"],
                "exit_reason": sim["exit_reason"], "n_levels_hit": sim["n_levels_hit"],
                "fraction_via_tp": fraction_via_tp,
                "r_gross": sim["r"], "cost_r": cost_r, "commission_r": commission_r,
                "r_net": r_net, "leverage_ratio": leverage_ratio,
            })

    all_trades.sort(key=lambda t: t["entry_time"])
    events = []
    for idx, t in enumerate(all_trades):
        events.append((t["entry_time"], 0, idx, "ENTRY"))
        events.append((t["exit_time"], 1, idx, "EXIT"))
    events.sort(key=lambda e: (e[0], e[1]))

    balance = INITIAL_CAPITAL_USD
    ruined = False
    equity_curve = [{"time": str(pd.Timestamp(start)), "balance": balance}]
    for time_, _order, idx, kind in events:
        t = all_trades[idx]
        if kind == "ENTRY":
            if ruined:
                t["risk_dollars"] = 0.0
                t["skipped_ruin"] = True
            else:
                max_risk_pct = MAX_LEVERAGE / t["leverage_ratio"] if t["leverage_ratio"] > 0 else RISK_PCT_PER_TRADE
                effective_risk_pct = min(RISK_PCT_PER_TRADE, max_risk_pct)
                t["risk_dollars"] = balance * effective_risk_pct
                t["effective_risk_pct"] = effective_risk_pct
                t["skipped_ruin"] = False
        else:
            if t.get("skipped_ruin"):
                t["dollar_pnl"] = 0.0
            else:
                t["dollar_pnl"] = t["r_net"] * t["risk_dollars"]
                balance += t["dollar_pnl"]
                if balance <= 0:
                    balance = 0.0
                    ruined = True
            t["balance_after"] = balance
            equity_curve.append({"time": str(time_), "balance": balance})

    n = len(all_trades)
    r_values = [t["r_net"] for t in all_trades if not t.get("skipped_ruin")]
    day_clusters_for_perm = [t["entry_time"].strftime("%Y-%m-%d") for t in all_trades if not t.get("skipped_ruin")]
    perm_result_block = permutation_test_block(r_values, day_clusters_for_perm, seed=42) if len(r_values) >= 4 else None
    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r < 0]
    profit_factor_val = (sum(wins) / abs(sum(losses))) if losses else None
    payoff_val = (float(np.mean(wins)) / abs(float(np.mean(losses)))) if wins and losses else None
    win_rate = (len(wins) / len(r_values)) if r_values else None
    mean_r_net = float(np.mean(r_values)) if r_values else None

    return {
        "period": "train", "start": start, "end": end,
        "n_trades": n, "win_rate": round(win_rate, 4) if win_rate else None,
        "mean_r_net": round(mean_r_net, 4) if mean_r_net else None,
        "profit_factor": round(profit_factor_val, 3) if profit_factor_val else None,
        "payoff_ratio": round(payoff_val, 3) if payoff_val else None,
        "perm_p_block": round(perm_result_block.p_value, 4) if perm_result_block else None,
        "trades": [
            {k: (str(v) if isinstance(v, pd.Timestamp) else (round(v, 6) if isinstance(v, float) else v))
             for k, v in t.items()}
            for t in all_trades
        ],
        "equity_curve": equity_curve,
    }


def main() -> int:
    start, end = PERIODS["train"]
    print(f"=== EXP-FX000017(SYS-FX023): ニュース窓限定でのSYS-FX011凍結ロジック評価 (Train単独) ===\n")

    results = {}
    kpis = {}
    for name, meetings in CANDIDATES.items():
        windows = build_blackout_windows(meetings=meetings) if meetings is not None else build_blackout_windows()
        print(f"--- {name} (窓数={len(windows)}) ---")
        result = run_period(windows, start, end)
        kpi = evaluate_period("train", result, perm_p_field="perm_p_block",
                               apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)
        results[name] = result
        kpis[name] = kpi
        print(f"  トレード数={result['n_trades']}  ペイオフ={kpi['payoff_ratio']}  "
              f"KPI={kpi['kpi_required_pass_count']}  実効n={kpi['n_trades_effective']}  "
              f"perm_p={kpi['permutation_p_clustered']}\n")

    # 選定ルール(spec事前登録): 実効n>=300を満たす候補のうちKPI達成数最大(同数ならペイオフ最大)
    eligible = [(name, k) for name, k in kpis.items() if k["n_trades_effective"] >= 300]
    if eligible:
        def sort_key(item):
            name, k = item
            return (int(k["kpi_required_pass_count"].split("/")[0]), k["payoff_ratio"] or 0)
        selected = max(eligible, key=sort_key)
    else:
        selected = None

    print("=== サマリ ===")
    print(f"{'candidate':<24}{'n_trades':>10}{'ペイオフ':>10}{'KPI':>8}{'実効n':>8}{'perm_p':>10}")
    for name in CANDIDATES:
        k = kpis[name]
        print(f"{name:<24}{results[name]['n_trades']:>10}{k['payoff_ratio']:>10}{k['kpi_required_pass_count']:>8}"
              f"{k['n_trades_effective']:>8}{k['permutation_p_clustered']:>10}")

    if selected:
        print(f"\n[選定] {selected[0]} (KPI={selected[1]['kpi_required_pass_count']}、"
              f"ペイオフ={selected[1]['payoff_ratio']})")
    else:
        print("\n[選定] 該当候補なし(実効n>=300を満たす候補が存在しない) → "
              "正式な採否判断は不可、実効n不足として記録")

    out_path = ROOT / "research" / "method-notes" / "sysfx023_calendar_window_trainonly_backtest.json"
    out_path.write_text(json.dumps({
        "generated_at": pd.Timestamp.now().isoformat(),
        "purpose": "EXP-FX000017(SYS-FX023): ニュース窓(BOJ/FOMC)限定でのSYS-FX011凍結ロジック評価",
        "candidates": list(CANDIDATES.keys()),
        "selected": selected[0] if selected else None,
        "results": results,
        "kpis": kpis,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
