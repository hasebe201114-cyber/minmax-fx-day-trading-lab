"""SYS-FX013(EXP-FX000007) 第1段階: GBP_USD・AUD_USD・NZD_USDを個別に
Train評価し、エッジの有無を確認する.

検出層・M5層・検定はSYS-FX012の凍結済み最良候補（候補①=N_BREAKOUT単独+
H1トレンド判定不能除外フィルター）を一切変更せず流用する。パラメータの
再導出は行わない。コストモデルの新規通貨スプレッドは`00-spec.md`記載の
暫定値（GMOコイン公式に未掲載のため業界一般水準を参考にした保守的な仮定）
を使用する。

各通貨単体でmean_r_net・勝率を確認し、EUR_USD検証と同じ形式で
エッジの有無を判定する(00-spec.mdの検証プロトコル)。

出力: research/method-notes/sysfx013_new_pairs_trainonly_backtest.json
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

from backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd import (  # noqa: E402
    ATR_TRAIL_MULTIPLIER_M5, BREAKEVEN_TRIGGER_R, COMMISSION_RATE_ROUND_TRIP,
    PERIODS, SLIPPAGE_PIPS_MARKET_LEG, SLIPPAGE_PIPS_STOP_TRIGGERED,
    STOP_BUFFER_ATR_M5, TP_CUM_FRACTION, TP_LEVELS_TRAILONLY, load_m5_period,
    pip_size,
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

# 00-spec.md「コストモデル」節: GMOコイン公式に未掲載のため保守的な暫定値
NEW_PAIR_SPREAD_PIPS = {"GBP_USD": 1.0, "AUD_USD": 0.8, "NZD_USD": 1.5}
INITIAL_CAPITAL_USD = 1000.0
RISK_PCT_PER_TRADE = 0.01
MAX_LEVERAGE = 25.0


def evaluate_single_pair(pair: str, start: str, end: str) -> dict:
    m5 = load_m5_period(pair, start, end)
    h1_by_pair = {pair: None}
    atr_h1_by_pair = {pair: None}
    from derive_vol_breakout_entry_params import to_h1  # noqa: E402
    h1 = to_h1(m5)
    atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    h1_by_pair[pair] = h1
    atr_h1_by_pair[pair] = atr_h1
    shock_check = make_price_shock_check(h1_by_pair, atr_h1_by_pair)

    trades, n_raw, n_dedup, n_trendfiltered = find_trades_trendfiltered(
        pair, m5, shock_check, detect_candidate1)

    spread = NEW_PAIR_SPREAD_PIPS[pair]
    pip = pip_size(pair)
    all_trades = []
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
                t["skipped_ruin"] = False
            balance_snapshot = balance
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

    r_net_values = np.array([t["r_net"] for t in all_trades])
    r_gross_values = np.array([t["r_gross"] for t in all_trades])

    result = {
        "period": "train", "pair": pair, "start": start, "end": end,
        "n_events_raw": n_raw, "n_events_dedup": n_dedup, "n_events_trendfiltered": n_trendfiltered,
        "n_trades": len(all_trades),
        "mean_r_net": round(float(r_net_values.mean()), 4) if len(r_net_values) else None,
        "mean_r_gross": round(float(r_gross_values.mean()), 4) if len(r_gross_values) else None,
        "win_rate": round(float((r_net_values > 0).mean()), 4) if len(r_net_values) else None,
        "spread_pips_used": spread,
        "trades": [
            {k: (str(v) if isinstance(v, pd.Timestamp) else (round(v, 6) if isinstance(v, float) else v))
             for k, v in t.items()}
            for t in all_trades
        ],
        "equity_curve": equity_curve,
    }
    return result


def main() -> int:
    start, end = PERIODS["train"]
    print(f"=== SYS-FX013(EXP-FX000007) 第1段階: 非JPY通貨個別Train評価 ===")
    print(f"検出層・M5層: SYS-FX012の凍結済み最良候補(候補①+判定不能除外フィルター)をそのまま使用\n")

    results = {}
    for pair in ["GBP_USD", "AUD_USD", "NZD_USD"]:
        print(f"--- {pair} ---")
        r = evaluate_single_pair(pair, start, end)
        results[pair] = r
        print(f"  イベント={r['n_events_raw']}件(dedup後{r['n_events_dedup']}、判定不能除外後{r['n_events_trendfiltered']})")
        print(f"  トレード数={r['n_trades']}  mean_r_net={r['mean_r_net']}  mean_r_gross={r['mean_r_gross']}  "
              f"勝率={r['win_rate']}  (スプレッド={r['spread_pips_used']}pips使用)\n")

    print("=== サマリ ===")
    for pair, r in results.items():
        verdict = "エッジあり(正)" if (r['mean_r_net'] or 0) > 0 else "エッジなし(負またはゼロ)"
        print(f"  {pair}: n={r['n_trades']}  mean_r_net={r['mean_r_net']}  勝率={r['win_rate']}  → {verdict}")

    out_path = ROOT / "research" / "method-notes" / "sysfx013_new_pairs_trainonly_backtest.json"
    out_path.write_text(json.dumps({
        "generated_at": pd.Timestamp.now().isoformat(),
        "design": "SYS-FX012の凍結済み最良候補(候補①=N_BREAKOUT単独+H1トレンド判定不能除外フィルター)を"
                  "GBP_USD・AUD_USD・NZD_USDに個別適用(通貨プールせず単体評価)",
        "note": "コストモデルのスプレッドはGMOコイン公式未掲載のため00-spec.md記載の保守的暫定値を使用",
        "results": results,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
