"""EXP-FX000015(SYS-FX021): 継続確認条件を「新高値/新安値」から「型崩れ水準を
下回らない」に緩和したTrain評価(H1版・H4版の2パターン).

司令塔提案「H1の高値更新を判断の起点にするのではなく、H1の下値を更新せずに
M5のトレンドが上昇したことを判断する」を受けて検証する。SYS-FX020(H4継続確認)
の不採用原因(新高値条件が粗い時間軸ほど厳しくなる)が、条件緩和で解消するかを
確認する。

検出層(H1、N_BREAKOUT=3.5)・トレンド判定不能除外フィルターは一切変更しない。
`simulate_dow_theory_trend()`の`confirm_mode="no_reversal"`を使う。

出力: research/method-notes/sysfx021_no_reversal_confirm_trainonly_backtest.json
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

from analyze_n_breakout_h1_dow_trend_alignment import h1_dow_trend_direction  # noqa: E402
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
from backtest_sysfx020_h4_confirm_trainonly import to_h4  # noqa: E402
from evaluate_vol_breakout_dow_theory_kpi import evaluate_period  # noqa: E402
from price_shock_filter import make_price_shock_check  # noqa: E402
from minmax_fx_dt.backtest.permutation import permutation_test_block  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402


def find_trades(pair: str, m5: pd.DataFrame, shock_check, confirm_bars: pd.DataFrame | None):
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
        if h1_dow_trend_direction(h1, atr_h1, pos) is None:
            continue
        direction = dedup_directions[pos]
        trades.extend(simulate_dow_theory_trend(
            m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER_M5,
            blackout_check=shock_check, tp_levels=TP_LEVELS_TRAILONLY, skip_first_entry=False,
            atr_trail_series=atr_m5, m5_exit=True, breakeven_trigger_r=BREAKEVEN_TRIGGER_R,
            confirm_bars=confirm_bars, confirm_mode="no_reversal"))
    return trades


def run_period(start: str, end: str, use_h4: bool) -> dict:
    m5_by_pair, h1_by_pair, atr_h1_by_pair, confirm_by_pair = {}, {}, {}, {}
    for pair in SELECTED_PAIRS:
        m5 = load_m5_period(pair, start, end)
        if len(m5) < 1000:
            continue
        m5_by_pair[pair] = m5
        h1_by_pair[pair] = to_h1(m5)
        atr_h1_by_pair[pair] = atr_ind(h1_by_pair[pair]["high"], h1_by_pair[pair]["low"],
                                        h1_by_pair[pair]["close"], length=14)
        confirm_by_pair[pair] = to_h4(m5) if use_h4 else None
    shock_check = make_price_shock_check(h1_by_pair, atr_h1_by_pair)

    all_trades: list[dict] = []
    for pair, m5 in m5_by_pair.items():
        trades = find_trades(pair, m5, shock_check, confirm_by_pair[pair])
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
        "period": "train", "start": start, "end": end, "use_h4_confirm": use_h4,
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
    print("=== EXP-FX000015(SYS-FX021): 継続確認条件を「型崩れ水準を下回らない」に緩和 Train評価 ===\n")

    with (ROOT / "research" / "method-notes" / "candidate3_cost_ratio_filter_trainonly_backtest.json").open(
        encoding="utf-8"
    ) as f:
        c1_result = json.load(f)
    candidate1_kpi = c1_result["candidate1_reference"]

    results = {}
    for label, use_h4 in [("h1_no_reversal", False), ("h4_no_reversal", True)]:
        print(f"--- {label} ---")
        result = run_period(start, end, use_h4)
        kpi = evaluate_period("train", result, perm_p_field="perm_p_block",
                               apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)
        results[label] = {"backtest": result, "kpi": kpi}
        print(f"  トレード数={result['n_trades']}  ペイオフ={kpi['payoff_ratio']}  KPI={kpi['kpi_required_pass_count']}  "
              f"実効n={kpi['n_trades_effective']}  Sharpe={kpi['monthly_sharpe']}  DD={kpi['max_dd_pct']}%  "
              f"perm_p={kpi['permutation_p_clustered']}\n")

    print("=== H1版candidate①(基準) vs H1+no_reversal vs H4+no_reversal、Train ===")
    print(f"{'指標':<20}{'H1版基準':>12}{'H1+no_rev':>12}{'H4+no_rev':>12}")
    for k, label in [
        ("n_trades_effective", "実効n"), ("monthly_sharpe", "月次シャープ"),
        ("profit_factor", "PF"), ("payoff_ratio", "ペイオフ"),
        ("max_dd_pct", "最大DD%"), ("permutation_p_clustered", "perm_p"),
    ]:
        print(f"{label:<20}{str(candidate1_kpi.get(k)):>12}{str(results['h1_no_reversal']['kpi'].get(k)):>12}"
              f"{str(results['h4_no_reversal']['kpi'].get(k)):>12}")
    print(f"{'KPI達成(必須)':<20}{candidate1_kpi['kpi_required_pass_count']:>12}"
          f"{results['h1_no_reversal']['kpi']['kpi_required_pass_count']:>12}"
          f"{results['h4_no_reversal']['kpi']['kpi_required_pass_count']:>12}")

    out_path = ROOT / "research" / "method-notes" / "sysfx021_no_reversal_confirm_trainonly_backtest.json"
    out_path.write_text(json.dumps({
        "generated_at": pd.Timestamp.now().isoformat(),
        "design": "候補①の継続確認条件をconfirm_mode='no_reversal'(型崩れ水準を下回らなければ再開)に緩和。"
                  "h1_no_reversal=confirmバーH1のまま条件のみ緩和、h4_no_reversal=confirmバーH4+条件緩和",
        "candidate1_reference": candidate1_kpi,
        "results": results,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
