"""EXP-FX000006(SYS-FX012) Trainベースライン評価.

検出層をSYS-FX011のN_BREAKOUT単独から、OR合成(N_BREAKOUT既存条件 OR
Donchian(20)継続+CALM_RATIO質フィルター)へ変更する(`00-spec.md`「検出層の
設計」節で事前登録、結果を見る前に固定)。それ以外(M5ダウ理論連続追跡・
トレール専業出口・コストモデル・重複除去・価格反応型ショック抑制・検定
方式)はSYS-FX011の最終候補(trailonly版、T-01〜T-13適用後)から一切変更
しない。

検証プロトコル(spec記載): Train単独でまず選定基準を満たすかを確認する。
Validation/Testは本スクリプトでは計算しない(HARKing防止、Trainで一定の
見込みが立った場合にのみ次の段階でValidationを参照する)。

出力: research/method-notes/vol_continuation_hybrid_4pairs_trainonly_backtest.json
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
    INITIAL_CAPITAL_USD, MAX_LEVERAGE, PERIODS, RISK_PCT_PER_TRADE,
    SLIPPAGE_PIPS_MARKET_LEG, SLIPPAGE_PIPS_STOP_TRIGGERED, SPREAD_PIPS,
    STOP_BUFFER_ATR_M5, TP_CUM_FRACTION, TP_LEVELS_TRAILONLY, load_m5_period,
    pip_size,
)
from derive_vol_breakout_entry_params import N_BREAKOUT, to_h1  # noqa: E402
from price_shock_filter import CALM_RATIO, make_price_shock_check  # noqa: E402
from minmax_fx_dt.backtest.permutation import (  # noqa: E402
    permutation_test_block, permutation_test_clustered,
)
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402
from minmax_fx_dt.strategy.indicators import donchian  # noqa: E402

DONCHIAN_LENGTH = 20  # indicators.py donchian()の既定値をそのまま採用(新規調整なし)


def detect_hybrid_events(h1: pd.DataFrame, atr_h1: pd.Series) -> tuple[list[int], list[str]]:
    """00-spec.md「検出層の設計」節のOR合成ルール。先読みなし(Donchianはshift(1))。"""
    dc = donchian(h1["high"], h1["low"], DONCHIAN_LENGTH, DONCHIAN_LENGTH).shift(1)
    ratio = (h1["high"] - h1["low"]) / atr_h1
    close, open_ = h1["close"], h1["open"]

    is_spike_up = (ratio >= N_BREAKOUT) & (close > open_)
    is_spike_down = (ratio >= N_BREAKOUT) & (close < open_)
    is_cont_up = (close > dc["DCU"]) & (ratio >= CALM_RATIO)
    is_cont_down = (close < dc["DCL"]) & (ratio >= CALM_RATIO)
    up = (is_spike_up | is_cont_up).fillna(False)
    down = (is_spike_down | is_cont_down).fillna(False)

    positions, directions = [], []
    for i in range(len(h1)):
        if bool(up.iloc[i]):
            positions.append(i)
            directions.append("UP")
        elif bool(down.iloc[i]):
            positions.append(i)
            directions.append("DOWN")
    return positions, directions


def find_trades_for_period(pair: str, m5: pd.DataFrame, shock_check) -> list[dict]:
    h1 = to_h1(m5)
    atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)

    positions, directions = detect_hybrid_events(h1, atr_h1)
    dedup_positions = select_non_overlapping_breakout_events(h1.index, positions, directions)
    dedup_directions = {pos: d for pos, d in zip(positions, directions)}

    trades = []
    for pos in dedup_positions:
        direction = dedup_directions[pos]
        trades.extend(simulate_dow_theory_trend(
            m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER_M5,
            blackout_check=shock_check, tp_levels=TP_LEVELS_TRAILONLY, skip_first_entry=False,
            atr_trail_series=atr_m5, m5_exit=True, breakeven_trigger_r=BREAKEVEN_TRIGGER_R))
    return trades, len(positions), len(dedup_positions)


def run_period(period_name: str, start: str, end: str) -> dict:
    print(f"\n=== {period_name}: {start} 〜 {end} ===")
    m5_by_pair, h1_by_pair, atr_h1_by_pair = {}, {}, {}
    for pair in SELECTED_PAIRS:
        m5 = load_m5_period(pair, start, end)
        if len(m5) < 1000:
            print(f"  [{pair}] データ不足 ({len(m5)}bars)、スキップ")
            continue
        m5_by_pair[pair] = m5
        h1_by_pair[pair] = to_h1(m5)
        atr_h1_by_pair[pair] = atr_ind(h1_by_pair[pair]["high"], h1_by_pair[pair]["low"],
                                        h1_by_pair[pair]["close"], length=14)
    shock_check = make_price_shock_check(h1_by_pair, atr_h1_by_pair)

    all_trades: list[dict] = []
    n_events_raw, n_events_dedup = 0, 0
    for pair, m5 in m5_by_pair.items():
        trades, n_raw, n_dedup = find_trades_for_period(pair, m5, shock_check)
        n_events_raw += n_raw
        n_events_dedup += n_dedup
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
        print(f"  [{pair}] イベント={n_raw}件(dedup後{n_dedup})  トレード={len(trades)}件")

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
                t["leverage_capped"] = effective_risk_pct < RISK_PCT_PER_TRADE
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
    day_clusters_for_perm = [t["entry_time"].strftime("%Y-%m-%d") for t in all_trades if not t.get("skipped_ruin")]

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
    perm_result_block = None
    if len(r_values) >= 4:
        perm_result = permutation_test_clustered(r_values, pairs_for_perm, seed=42)
        perm_result_block = permutation_test_block(r_values, day_clusters_for_perm, seed=42)

    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r < 0]
    profit_factor_val = (sum(wins) / abs(sum(losses))) if losses else None
    payoff_val = (float(np.mean(wins)) / abs(float(np.mean(losses)))) if wins and losses else None

    print(f"  イベント総数={n_events_raw}(dedup後{n_events_dedup})  トレード数={n} "
          f"(破産ガードでスキップ={n - n_effective_trades})  最終残高=${final_balance:.2f}  "
          f"総リターン={total_return_pct:.1f}%  最大DD(初期資金比)={max_dd_pct:.1f}%")
    print(f"  勝率={win_rate:.3f}  平均r_net={mean_r_net:.4f}  PF={profit_factor_val}  ペイオフ={payoff_val}"
          f"{f'  perm_p(pair-cluster,旧)={perm_result.p_value:.4f}' if perm_result else ''}"
          f"{f'  perm_p(day-block,T-06)={perm_result_block.p_value:.4f}' if perm_result_block else ''}")

    return {
        "period": period_name, "start": start, "end": end,
        "n_events_raw": n_events_raw, "n_events_dedup": n_events_dedup,
        "n_trades": n, "n_effective_trades": n_effective_trades,
        "final_balance_usd": round(final_balance, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "mean_r_net": round(mean_r_net, 4) if mean_r_net is not None else None,
        "profit_factor": round(profit_factor_val, 3) if profit_factor_val else None,
        "payoff_ratio": round(payoff_val, 3) if payoff_val else None,
        "perm_p_clustered": round(perm_result.p_value, 4) if perm_result else None,
        "perm_p_block": round(perm_result_block.p_value, 4) if perm_result_block else None,
        "perm_p_block_method": perm_result_block.method if perm_result_block else None,
        "leverage_ratio_stats": {
            "median": round(float(np.median([t["leverage_ratio"] for t in all_trades])), 1) if all_trades else None,
            "max": round(float(np.max([t["leverage_ratio"] for t in all_trades])), 1) if all_trades else None,
        },
        "n_leverage_capped": sum(1 for t in all_trades if t.get("leverage_capped")),
        "trades": [
            {k: (str(v) if isinstance(v, pd.Timestamp) else (round(v, 6) if isinstance(v, float) else v))
             for k, v in t.items()}
            for t in all_trades
        ],
        "equity_curve": equity_curve,
    }


def main() -> int:
    print("=== EXP-FX000006(SYS-FX012) Trainベースライン評価 ===")
    print(f"検出層: N_BREAKOUT={N_BREAKOUT} OR (Donchian({DONCHIAN_LENGTH})継続 AND range/ATR>=CALM_RATIO={CALM_RATIO})")

    start, end = PERIODS["train"]
    result = run_period("train", start, end)

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "selected_pairs": SELECTED_PAIRS,
        "design": {
            "detection_layer": f"N_BREAKOUT({N_BREAKOUT}) OR (Donchian({DONCHIAN_LENGTH}) continuation AND "
                                f"range/ATR>=CALM_RATIO({CALM_RATIO}))",
            "note": "検出層のみSYS-FX011から変更。M5エントリー/出口/コスト/重複除去/ショック抑制は完全に同一",
        },
        "periods": {"train": result},
        "_note": "00-spec.mdの検証プロトコルに従いTrain単独のみを計算。Validation/Testは未実施(HARKing防止)",
    }
    out_path = ROOT / "research" / "method-notes" / "vol_continuation_hybrid_4pairs_trainonly_backtest.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
