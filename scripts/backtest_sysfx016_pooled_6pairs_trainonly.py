"""SYS-FX016(EXP-FX000010) 第2段階: 個別Train評価で正だったCAD_JPY・CHF_JPYを
既存4通貨に統合した6通貨プールでTrain評価する(NZD_JPYは個別評価で負のため除外、
00-spec.mdの事前登録ルールに従う).

検出層・M5層・出口設計・コストモデルの構造・検定方式はSYS-FX012の凍結済み
最良候補から一切変更しない。変えるのは対象通貨のみ。

出力: research/method-notes/sysfx016_pooled_6pairs_trainonly_backtest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from backtest_sysfx016_new_jpy_pairs_trainonly import NEW_PAIR_SPREAD_PIPS  # noqa: E402
from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402
from backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd import (  # noqa: E402
    COMMISSION_RATE_ROUND_TRIP, INITIAL_CAPITAL_USD, MAX_LEVERAGE, PERIODS,
    RISK_PCT_PER_TRADE, SLIPPAGE_PIPS_MARKET_LEG, SLIPPAGE_PIPS_STOP_TRIGGERED,
    SPREAD_PIPS, TP_CUM_FRACTION, TP_LEVELS_TRAILONLY, load_m5_period, pip_size,
    to_h1,
)
from backtest_vol_continuation_candidates_trendfilter_4pairs_trainonly import (  # noqa: E402
    detect_candidate1, find_trades_trendfiltered,
)
from evaluate_vol_breakout_dow_theory_kpi import evaluate_period  # noqa: E402
from price_shock_filter import make_price_shock_check  # noqa: E402
from minmax_fx_dt.backtest.permutation import (  # noqa: E402
    permutation_test_block, permutation_test_clustered,
)
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402
import numpy as np  # noqa: E402

# 個別Train評価(sysfx016_new_jpy_pairs)でmean_r_net>0だった通貨のみ追加(NZD_JPYは除外)
POOLED_PAIRS = list(SELECTED_PAIRS) + ["CAD_JPY", "CHF_JPY"]
ALL_SPREAD_PIPS = {**SPREAD_PIPS, **NEW_PAIR_SPREAD_PIPS}


def run_pooled_period(start: str, end: str) -> dict:
    m5_by_pair, h1_by_pair, atr_h1_by_pair = {}, {}, {}
    for pair in POOLED_PAIRS:
        m5 = load_m5_period(pair, start, end)
        if len(m5) < 1000:
            continue
        m5_by_pair[pair] = m5
        h1_by_pair[pair] = to_h1(m5)
        atr_h1_by_pair[pair] = atr_ind(h1_by_pair[pair]["high"], h1_by_pair[pair]["low"],
                                        h1_by_pair[pair]["close"], length=14)
    shock_check = make_price_shock_check(h1_by_pair, atr_h1_by_pair)

    all_trades: list[dict] = []
    n_raw_total = n_dedup_total = n_trendfiltered_total = 0
    for pair, m5 in m5_by_pair.items():
        trades, n_raw, n_dedup, n_trendfiltered = find_trades_trendfiltered(
            pair, m5, shock_check, detect_candidate1)
        n_raw_total += n_raw
        n_dedup_total += n_dedup
        n_trendfiltered_total += n_trendfiltered
        spread = ALL_SPREAD_PIPS.get(pair, 0.5)
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
    pairs_for_perm = [t["pair"] for t in all_trades if not t.get("skipped_ruin")]
    day_clusters_for_perm = [t["entry_time"].strftime("%Y-%m-%d") for t in all_trades if not t.get("skipped_ruin")]
    # perm_p_clustered(通貨ペア相関考慮版)はPAIR_CORRELATION_MATRIXに新規通貨(CAD/CHF_JPY)の
    # 実測相関エントリがなくKeyErrorになるため計算しない。KPI判定に使うのはperm_p_block
    # (日次クラスタベース、evaluate_period呼び出し時にperm_p_field="perm_p_block"指定)のみ。
    perm_result = None
    perm_result_block = permutation_test_block(r_values, day_clusters_for_perm, seed=42) if len(r_values) >= 4 else None
    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r < 0]
    profit_factor_val = (sum(wins) / abs(sum(losses))) if losses else None
    payoff_val = (float(np.mean(wins)) / abs(float(np.mean(losses)))) if wins and losses else None
    win_rate = (len(wins) / len(r_values)) if r_values else None
    mean_r_net = float(np.mean(r_values)) if r_values else None

    return {
        "period": "train", "start": start, "end": end, "pairs": POOLED_PAIRS,
        "n_events_raw": n_raw_total, "n_events_dedup": n_dedup_total,
        "n_events_trendfiltered": n_trendfiltered_total,
        "n_trades": n, "win_rate": round(win_rate, 4) if win_rate else None,
        "mean_r_net": round(mean_r_net, 4) if mean_r_net else None,
        "profit_factor": round(profit_factor_val, 3) if profit_factor_val else None,
        "payoff_ratio": round(payoff_val, 3) if payoff_val else None,
        "perm_p_clustered": round(perm_result.p_value, 4) if perm_result else None,
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
    print(f"=== SYS-FX016(EXP-FX000010) 6通貨プール({POOLED_PAIRS}) Train評価 ===\n")
    result = run_pooled_period(start, end)
    kpi = evaluate_period("train", result, perm_p_field="perm_p_block",
                           apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)

    print(f"イベント: raw={result['n_events_raw']} dedup={result['n_events_dedup']} "
          f"判定不能除外後={result['n_events_trendfiltered']}")
    print(f"トレード数={result['n_trades']}  勝率={result['win_rate']}  PF={result['profit_factor']}  "
          f"ペイオフ={result['payoff_ratio']}")
    print(f"KPI: {kpi['kpi_required_pass_count']}  実効n={kpi['n_trades_effective']}  "
          f"Sharpe={kpi['monthly_sharpe']}  DD={kpi['max_dd_pct']}%  perm_p={kpi['permutation_p_clustered']}")

    with (ROOT / "research" / "method-notes" / "candidate3_cost_ratio_filter_trainonly_backtest.json").open(
        encoding="utf-8"
    ) as f:
        c1_result = json.load(f)
    candidate1_kpi = c1_result["candidate1_reference"]

    print("\n=== 候補①(4通貨、基準) vs 6通貨プール、Train ===")
    print(f"{'指標':<20}{'4通貨':>14}{'6通貨':>14}")
    for k, label in [
        ("n_trades_effective", "実効n"), ("monthly_sharpe", "月次シャープ"),
        ("profit_factor", "PF"), ("payoff_ratio", "ペイオフ"),
        ("max_dd_pct", "最大DD%"), ("spread_cost_multiplier", "スプレッド倍率"),
        ("permutation_p_clustered", "perm_p"),
    ]:
        print(f"{label:<20}{str(candidate1_kpi.get(k)):>14}{str(kpi.get(k)):>14}")
    print(f"{'KPI達成(必須)':<20}{candidate1_kpi['kpi_required_pass_count']:>14}{kpi['kpi_required_pass_count']:>14}")

    out_path = ROOT / "research" / "method-notes" / "sysfx016_pooled_6pairs_trainonly_backtest.json"
    out_path.write_text(json.dumps({
        "generated_at": pd.Timestamp.now().isoformat(),
        "design": f"候補①(N_BREAKOUT=3.5+H1トレンド判定不能除外フィルター)を{POOLED_PAIRS}(6通貨)へ適用。"
                  "NZD_JPYは個別Train評価でmean_r_net負のため除外(事前登録ルール)",
        "candidate1_reference": candidate1_kpi,
        "kpi": kpi,
        "backtest": result,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
