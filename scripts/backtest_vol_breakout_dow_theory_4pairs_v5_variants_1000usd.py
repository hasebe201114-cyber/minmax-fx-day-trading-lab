"""EXP-FX000005 改善ループ第5試行: TP3=4Rと初回エントリー除外を切り分けて評価.

背景: 両者を組み合わせたv5(`backtest_vol_breakout_dow_theory_4pairs_v5_1000usd.py`)
はTest期間で明確に悪化した(正式KPI達成数6/10→3/10、mean_r_net 0.17→0.03、
最大連続損失5→8)。どちらの変更が悪化の主因かを切り分けるため、TP3=4Rのみ・
初回エントリー除外のみ・両方(参考として再掲)の3バリアントを同一パイプラインで
比較する。

出力: research/method-notes/vol_breakout_dow_theory_4pairs_v5_variants_1000usd_backtest.json
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

from backtest_vol_breakout_dow_theory import simulate_dow_theory_trend  # noqa: E402
from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402
from derive_vol_breakout_entry_params import N_BREAKOUT, to_h1  # noqa: E402
from economic_calendar import is_blackout  # noqa: E402
from minmax_fx_dt.backtest.permutation import permutation_test_clustered  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_train.json").open(encoding="utf-8") as f:
    TRAIN_RESULT = json.load(f)
STOP_BUFFER_ATR_M5 = TRAIN_RESULT["params"]["stop_buffer_atr_m5"]
ATR_TRAIL_MULTIPLIER = TRAIN_RESULT["params"]["atr_trail_multiplier"]

PERIODS = {
    "train":      ("2023-11-01", "2025-03-31"),
    "validation": ("2025-04-01", "2025-11-30"),
    "test":       ("2025-12-01", "2026-08-15"),
}

VARIANTS = {
    "baseline(第3試行)": {"tp_levels": None, "skip_first_entry": False},
    "TP3=4Rのみ": {"tp_levels": [(1.0, 0.40), (2.0, 0.35), (4.0, 0.25)], "skip_first_entry": False},
    "初回除外のみ": {"tp_levels": None, "skip_first_entry": True},
    "両方(第5試行)": {"tp_levels": [(1.0, 0.40), (2.0, 0.35), (4.0, 0.25)], "skip_first_entry": True},
}

SPREAD_PIPS = {"USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6, "EUR_USD": 0.3}
SLIPPAGE_PIPS_MARKET_LEG = 0.5
COMMISSION_RATE_ROUND_TRIP = 0.00004
RISK_PCT_PER_TRADE = 0.01
MAX_LEVERAGE = 25.0
INITIAL_CAPITAL_USD = 1000.0


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def tp_cum_fraction(tp_levels) -> list[float]:
    levels = tp_levels if tp_levels is not None else [(1.0, 0.40), (2.0, 0.35), (3.0, 0.25)]
    cum = 0.0
    out = [0.0]
    for _r, frac in levels:
        cum += frac
        out.append(cum)
    return out


def load_m5_period(pair: str, start: str, end: str) -> pd.DataFrame:
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= start) & (df.index <= end)]


def run_variant(variant_name: str, period_name: str, start: str, end: str,
                 tp_levels, skip_first_entry: bool) -> dict:
    cum_fraction = tp_cum_fraction(tp_levels)
    all_trades: list[dict] = []
    for pair in SELECTED_PAIRS:
        m5 = load_m5_period(pair, start, end)
        if len(m5) < 1000:
            continue
        h1 = to_h1(m5)
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)
        ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
        idxs = np.where(ratio.values >= N_BREAKOUT)[0]

        trades = []
        for i in idxs:
            pos = h1.index.get_loc(ratio.index[i])
            bar = h1.iloc[pos]
            direction = "UP" if bar["close"] > bar["open"] else "DOWN"
            trades.extend(simulate_dow_theory_trend(
                m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER,
                blackout_check=is_blackout, tp_levels=tp_levels, skip_first_entry=skip_first_entry,
            ))

        spread = SPREAD_PIPS.get(pair, 0.5)
        pip = pip_size(pair)
        for sim in trades:
            fraction_via_tp = cum_fraction[sim["n_levels_hit"]]
            fraction_remaining = 1.0 - fraction_via_tp
            remaining_is_market = sim["exit_reason"] in ("WEEKEND_NO_TP", "TP_THEN_WEEKEND", "MAX_HOLD")
            entry_pips = spread + SLIPPAGE_PIPS_MARKET_LEG
            exit_pips = spread + (fraction_remaining * SLIPPAGE_PIPS_MARKET_LEG if remaining_is_market else 0.0)
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
    win_rate = float(np.mean([r > 0 for r in r_values])) if r_values else None
    mean_r_net = float(np.mean(r_values)) if r_values else None
    total_return_pct = (balance / INITIAL_CAPITAL_USD - 1.0) * 100.0
    balances = [pt["balance"] for pt in equity_curve]
    running_max = np.maximum.accumulate(balances) if balances else np.array([INITIAL_CAPITAL_USD])
    drawdowns = [(b - m) / m * 100.0 if m > 0 else 0.0 for b, m in zip(balances, running_max)]
    max_dd_pct = min(drawdowns) if drawdowns else 0.0
    perm_result = permutation_test_clustered(r_values, pairs_for_perm, seed=42) if len(r_values) >= 4 else None
    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else None

    # 最大連続損失(トレード単位、時系列順)
    worst = cur = 0
    for t in all_trades:
        if t.get("skipped_ruin"):
            continue
        if t["dollar_pnl"] < 0:
            cur += 1
            worst = max(worst, cur)
        else:
            cur = 0

    return {
        "variant": variant_name, "period": period_name, "n_trades": n,
        "final_balance_usd": round(balance, 2), "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "mean_r_net": round(mean_r_net, 4) if mean_r_net is not None else None,
        "profit_factor": round(pf, 3) if pf else None,
        "max_consecutive_losses": worst,
        "perm_p_clustered": round(perm_result.p_value, 4) if perm_result else None,
    }


def main() -> int:
    print("=== EXP-FX000005 TP3=4R / 初回エントリー除外 の切り分け比較 ===\n")
    results = {}
    for variant_name, cfg in VARIANTS.items():
        results[variant_name] = {}
        print(f"--- {variant_name} ---")
        for period_name, (start, end) in PERIODS.items():
            r = run_variant(variant_name, period_name, start, end, cfg["tp_levels"], cfg["skip_first_entry"])
            results[variant_name][period_name] = r
            print(f"  {period_name}: n={r['n_trades']} final=${r['final_balance_usd']} "
                  f"return={r['total_return_pct']}% maxDD={r['max_drawdown_pct']}% "
                  f"win_rate={r['win_rate']} mean_r_net={r['mean_r_net']} PF={r['profit_factor']} "
                  f"max_loss_streak={r['max_consecutive_losses']} perm_p={r['perm_p_clustered']}")
        print()

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_v5_variants_1000usd_backtest.json"
    out_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(),
        "selected_pairs": SELECTED_PAIRS,
        "variants": {k: v for k, v in VARIANTS.items()},
        "results": results,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
