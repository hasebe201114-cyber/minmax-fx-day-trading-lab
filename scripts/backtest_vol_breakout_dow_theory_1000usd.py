"""EXP-FX000005 改善ループ第1試行(M5ダウ理論連続押し目買い版)の
Train/Validation/Test 3期間・$1,000初期資金・実運用コスト込みバックテスト.

司令塔依頼「投資額1000ドルとしてバックテスト、収益率も」を受け、SYS-FX009の
`backtest_h1_tighter_stop_1000usd.py`と同一の方法論(複利ポジションサイジング・
実運用コスト・イベント駆動エクイティカーブ)をダウ理論連続押し目買い版に適用する。

パラメータ(N=3.5, zigzag_threshold_atr_m5=1.0, stop_buffer_atr_m5=0.701,
atr_trail_multiplier=3.23)はTrainデータから導出済みの値をそのまま使用し
(`vol_breakout_dow_theory_train.json`)、Validation/Testでも再導出しない
(TVT分離の既存方針)。各期間は独立に$1,000からスタートする(複利は期間を
跨がない)。

コスト・サイジング設計(事前登録、`backtest_h1_tighter_stop_1000usd.py`と同一):
    - ポジションサイジング: エントリー時点の口座残高の1%をリスク額とする
    - コスト: スプレッド往復(通貨別SPREAD_PIPS) + スリッページ往復1.0pip +
      手数料往復0.004%。Rマルチプル換算してからドルP&Lへ変換
    - 複数ポジションの同時保有はイベント駆動(エントリー/エグジット時刻順)で
      処理。残高が0以下になったら以降の新規エントリーをスキップ(破産ガード)
    - 統計的有意性: permutation_test_clustered()を各期間で実行

出力: research/method-notes/vol_breakout_dow_theory_1000usd_backtest.json
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

from backtest_vol_breakout_dow_theory import is_weekend_close_time, track_dow_theory_pullbacks  # noqa: E402
from derive_vol_breakout_entry_params import N_BREAKOUT, PAIRS, to_h1  # noqa: E402
from minmax_fx_dt.backtest.permutation import permutation_test_clustered  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_train.json").open(encoding="utf-8") as f:
    TRAIN_RESULT = json.load(f)
TRAIN_PARAMS = TRAIN_RESULT["params"]
STOP_BUFFER_ATR_M5 = TRAIN_PARAMS["stop_buffer_atr_m5"]
ATR_TRAIL_MULTIPLIER = TRAIN_PARAMS["atr_trail_multiplier"]

PERIODS = {
    "train":      ("2023-11-01", "2025-03-31"),
    "validation": ("2025-04-01", "2025-11-30"),
    "test":       ("2025-12-01", "2026-08-15"),
}

TP_LEVELS = [(1.0, 0.40), (2.0, 0.35), (3.0, 0.25)]
MAX_HOLD_BARS = 24 * 10  # H1、10日相当(暴走ループ防止の安全上限、既存スクリプト踏襲)

SPREAD_PIPS = {"USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6, "EUR_USD": 0.3}
SLIPPAGE_PIPS_ROUND_TRIP = 1.0
COMMISSION_RATE_ROUND_TRIP = 0.00004

RISK_PCT_PER_TRADE = 0.01
INITIAL_CAPITAL_USD = 1000.0


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def load_m5_period(pair: str, start: str, end: str) -> pd.DataFrame:
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= start) & (df.index <= end)]


def find_entries_for_period(pair: str, m5: pd.DataFrame) -> tuple[list[dict], pd.DataFrame, pd.Series]:
    h1 = to_h1(m5)
    atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)

    ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
    idxs = np.where(ratio.values >= N_BREAKOUT)[0]

    entries = []
    for i in idxs:
        pos = h1.index.get_loc(ratio.index[i])
        bar = h1.iloc[pos]
        direction = "UP" if bar["close"] > bar["open"] else "DOWN"
        pullbacks = track_dow_theory_pullbacks(m5, atr_m5, h1, pos, direction)
        for pb in pullbacks:
            buffer = STOP_BUFFER_ATR_M5 * pb["pivot_atr"]
            stop0 = pb["pivot_price"] - buffer if direction == "UP" else pb["pivot_price"] + buffer
            entry_price = pb["confirm_price"]
            initial_risk = abs(entry_price - stop0)
            if initial_risk <= 0:
                continue
            entry_h1_idx = int(h1.index.searchsorted(pb["confirm_time"], side="right") - 1)
            if entry_h1_idx < 0 or entry_h1_idx >= len(h1):
                continue
            entries.append(dict(pair=pair, direction=direction, entry_idx=entry_h1_idx,
                                 entry_time=pb["confirm_time"], entry_price=entry_price,
                                 initial_risk=initial_risk))
    return entries, h1, atr_h1


def simulate_trade(h1: pd.DataFrame, atr_h1: pd.Series, entry: dict, trail_mult: float) -> dict:
    """段階利確(40/35/25%)、1R到達後BE+ATRトレーリング。exit_timeも返す。"""
    direction = entry["direction"]
    entry_price = entry["entry_price"]
    risk = entry["initial_risk"]
    stop = entry_price - risk if direction == "UP" else entry_price + risk
    levels = [(r, frac, entry_price + r * risk if direction == "UP" else entry_price - r * risk, False)
              for r, frac in TP_LEVELS]
    remaining_fraction = 1.0
    realized_r = 0.0
    be_moved = False
    n = len(h1)
    start = entry["entry_idx"] + 1
    end = min(n, start + MAX_HOLD_BARS)
    for i in range(start, end):
        ts = h1.index[i]
        o, h, low, c = float(h1["open"].iloc[i]), float(h1["high"].iloc[i]), float(h1["low"].iloc[i]), float(h1["close"].iloc[i])
        n_levels_hit = sum(1 for lv in levels if lv[3])
        if is_weekend_close_time(ts):
            exit_r = (c - entry_price) / risk if direction == "UP" else (entry_price - c) / risk
            reason = "WEEKEND_NO_TP" if n_levels_hit == 0 else "TP_THEN_WEEKEND"
            return {"r": realized_r + remaining_fraction * exit_r, "exit_reason": reason,
                    "n_levels_hit": n_levels_hit, "exit_time": ts}
        stop_hit = (low <= stop) if direction == "UP" else (h >= stop)
        if stop_hit:
            exit_r = (stop - entry_price) / risk if direction == "UP" else (entry_price - stop) / risk
            reason = "SL_INITIAL_NO_TP" if n_levels_hit == 0 else "TP_THEN_SL_TRAIL"
            return {"r": realized_r + remaining_fraction * exit_r, "exit_reason": reason,
                    "n_levels_hit": n_levels_hit, "exit_time": ts}
        for idx_lv, (r_level, frac, price_level, hit) in enumerate(levels):
            if hit or remaining_fraction <= 0:
                continue
            reached = (h >= price_level) if direction == "UP" else (low <= price_level)
            if reached:
                realized_r += frac * r_level
                remaining_fraction -= frac
                levels[idx_lv] = (r_level, frac, price_level, True)
                if not be_moved:
                    stop = max(stop, entry_price) if direction == "UP" else min(stop, entry_price)
                    be_moved = True
        if be_moved and remaining_fraction > 0:
            atr_i = atr_h1.asof(ts)
            if pd.notna(atr_i) and atr_i > 0:
                if direction == "UP":
                    stop = max(stop, o - trail_mult * float(atr_i))
                else:
                    stop = min(stop, o + trail_mult * float(atr_i))
        if remaining_fraction <= 1e-9:
            return {"r": realized_r, "exit_reason": "TP_FULL", "n_levels_hit": 3, "exit_time": ts}
    ts_last = h1.index[end - 1]
    c = float(h1["close"].iloc[end - 1])
    exit_r = (c - entry_price) / risk if direction == "UP" else (entry_price - c) / risk
    n_levels_hit = sum(1 for lv in levels if lv[3])
    return {"r": realized_r + remaining_fraction * exit_r, "exit_reason": "MAX_HOLD",
            "n_levels_hit": n_levels_hit, "exit_time": ts_last}


def run_period(period_name: str, start: str, end: str) -> dict:
    print(f"\n=== {period_name}: {start} 〜 {end} ===")
    all_trades: list[dict] = []
    for pair in PAIRS:
        m5 = load_m5_period(pair, start, end)
        if len(m5) < 1000:
            print(f"  [{pair}] データ不足 ({len(m5)}bars)、スキップ")
            continue
        entries, h1, atr_h1 = find_entries_for_period(pair, m5)
        spread = SPREAD_PIPS.get(pair, 0.5)
        pip = pip_size(pair)
        cost_price = (2 * spread + SLIPPAGE_PIPS_ROUND_TRIP) * pip
        for e in entries:
            sim = simulate_trade(h1, atr_h1, e, ATR_TRAIL_MULTIPLIER)
            cost_r = cost_price / e["initial_risk"]
            leverage_ratio = e["entry_price"] / e["initial_risk"]
            commission_r = COMMISSION_RATE_ROUND_TRIP * leverage_ratio
            r_net = sim["r"] - cost_r - commission_r
            all_trades.append({
                "pair": pair, "direction": e["direction"],
                "entry_time": e["entry_time"], "exit_time": sim["exit_time"],
                "entry_price": e["entry_price"], "initial_risk": e["initial_risk"],
                "exit_reason": sim["exit_reason"], "n_levels_hit": sim["n_levels_hit"],
                "r_gross": sim["r"], "cost_r": cost_r, "commission_r": commission_r,
                "r_net": r_net, "leverage_ratio": leverage_ratio,
            })
        print(f"  [{pair}] トレード={len(entries)}件")

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
                t["risk_dollars"] = balance * RISK_PCT_PER_TRADE
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
    n_effective_trades = sum(1 for t in all_trades if not t.get("skipped_ruin"))
    r_values = [t["r_net"] for t in all_trades if not t.get("skipped_ruin")]
    pairs_for_perm = [t["pair"] for t in all_trades if not t.get("skipped_ruin")]

    n_wins = sum(1 for r in r_values if r > 0)
    win_rate = n_wins / len(r_values) if r_values else None
    mean_r_net = float(np.mean(r_values)) if r_values else None
    final_balance = balance
    total_return_pct = (final_balance / INITIAL_CAPITAL_USD - 1.0) * 100.0

    balances = [pt["balance"] for pt in equity_curve]
    running_max = np.maximum.accumulate(balances) if balances else np.array([INITIAL_CAPITAL_USD])
    drawdowns = [(b - m) / m * 100.0 if m > 0 else 0.0 for b, m in zip(balances, running_max)]
    max_dd_pct = min(drawdowns) if drawdowns else 0.0

    perm_result = None
    if len(r_values) >= 4:
        perm_result = permutation_test_clustered(r_values, pairs_for_perm, seed=42)

    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r < 0]
    profit_factor_val = (sum(wins) / abs(sum(losses))) if losses else None
    payoff_val = (float(np.mean(wins)) / abs(float(np.mean(losses)))) if wins and losses else None

    print(f"  トレード数={n} (破産ガードでスキップ={n - n_effective_trades})  最終残高=${final_balance:.2f}  "
          f"総リターン={total_return_pct:.1f}%  最大DD={max_dd_pct:.1f}%")
    print(f"  勝率={win_rate:.3f}  平均r_net={mean_r_net:.4f}  PF={profit_factor_val}  ペイオフ={payoff_val}"
          f"{f'  perm_p(cluster)={perm_result.p_value:.4f}' if perm_result else ''}")

    return {
        "period": period_name, "start": start, "end": end,
        "n_trades": n, "n_effective_trades": n_effective_trades,
        "final_balance_usd": round(final_balance, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "mean_r_net": round(mean_r_net, 4) if mean_r_net is not None else None,
        "profit_factor": round(profit_factor_val, 3) if profit_factor_val else None,
        "payoff_ratio": round(payoff_val, 3) if payoff_val else None,
        "perm_p_clustered": round(perm_result.p_value, 4) if perm_result else None,
        "leverage_ratio_stats": {
            "median": round(float(np.median([t["leverage_ratio"] for t in all_trades])), 1) if all_trades else None,
            "max": round(float(np.max([t["leverage_ratio"] for t in all_trades])), 1) if all_trades else None,
        },
        "trades": [
            {k: (str(v) if isinstance(v, pd.Timestamp) else (round(v, 6) if isinstance(v, float) else v))
             for k, v in t.items()}
            for t in all_trades
        ],
        "equity_curve": equity_curve,
    }


def main() -> int:
    print("=== SYS-FX011 ダウ理論連続押し目買い版 Train/Validation/Test $1,000バックテスト ===")
    print(f"事前登録(Train導出値をそのまま使用): N={N_BREAKOUT}, "
          f"zigzag_threshold_atr_m5={TRAIN_PARAMS['zigzag_threshold_atr_m5']}, "
          f"stop_buffer_atr_m5={STOP_BUFFER_ATR_M5}, atr_trail_multiplier={ATR_TRAIL_MULTIPLIER}, "
          f"risk_pct_per_trade={RISK_PCT_PER_TRADE}, initial_capital=${INITIAL_CAPITAL_USD}")

    period_results = {}
    for period_name, (start, end) in PERIODS.items():
        period_results[period_name] = run_period(period_name, start, end)

    print(f"\n=== サマリ ===")
    print(f"{'期間':<12}{'取引数':>8}{'最終残高':>12}{'総リターン':>10}{'最大DD':>8}{'勝率':>8}{'PF':>8}{'perm_p':>8}")
    for name, r in period_results.items():
        print(f"{name:<12}{r['n_trades']:>8}{'$'+str(r['final_balance_usd']):>12}"
              f"{str(r['total_return_pct'])+'%':>10}{str(r['max_drawdown_pct'])+'%':>8}"
              f"{r['win_rate']:>8}{r['profit_factor']:>8}{r['perm_p_clustered']:>8}")

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_1000usd_backtest.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "design": {
                "n_breakout": N_BREAKOUT, "zigzag_threshold_atr_m5": TRAIN_PARAMS["zigzag_threshold_atr_m5"],
                "stop_buffer_atr_m5": STOP_BUFFER_ATR_M5, "atr_trail_multiplier": ATR_TRAIL_MULTIPLIER,
                "risk_pct_per_trade": RISK_PCT_PER_TRADE, "initial_capital_usd": INITIAL_CAPITAL_USD,
                "tp_levels": TP_LEVELS, "spread_pips": SPREAD_PIPS,
                "slippage_pips_round_trip": SLIPPAGE_PIPS_ROUND_TRIP,
                "commission_rate_round_trip": COMMISSION_RATE_ROUND_TRIP,
            },
            "periods": period_results,
            "_note": (
                "ダウ理論連続押し目買い版を、Train/Validation/Test 3期間・$1,000初期資金・"
                "実運用コスト込みで評価。パラメータはTrain導出値をそのまま使用(再学習なし)。"
                "各期間は独立に$1,000からリスタート(TVT分離の既存方針)。"
            ),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
