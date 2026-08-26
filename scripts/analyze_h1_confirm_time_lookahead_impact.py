"""N_BREAKOUT値動き分析(`analyze_n_breakout_price_movement.py`)で発見した
「h1.index[break_idx]がH1バーの開始時刻であり確定時刻ではない」という問題が、
SYS-FX012の現行凍結設計(候補①=N_BREAKOUT単独+H1トレンド判定不能除外フィルター、
フォワードテスト実施中)のTrain評価に実際どの程度影響するかを定量化する.

## 問題の要約(詳細はanalyze_n_breakout_price_movement.pyのdocstring参照)

pandas `resample('1h')`の既定ラベル付け(label='left')により、`h1.index[i]`は
H1バーの**開始時刻**(例: ラベル10:00のバーはM5データ[10:00,11:00)を集約)である。
`backtest_vol_breakout_dow_theory.py`の`simulate_dow_theory_trend()`は
`break_time = h1.index[break_idx]`をそのまま「バー確定時刻」として扱い、
そこから`start_time = break_time + 30分`(準備期間の終端)を計算しているため、
`start_time`は実際にはバーがまだ形成中(バー開始から30分後、確定=開始+60分より
前)の時点になる。同様に`select_non_overlapping_breakout_events()`の追跡窓
(72時間)も、`price_shock_filter.py`のショック抑制判定も、すべてこの
バー開始時刻を基準にしている。

## 検証方法(自然実験、共有本番コードは一切変更しない)

`h1`(および`atr_h1`)のインデックスラベルを一律+1時間シフトしたコピーを作り、
既存の公式関数(`select_non_overlapping_breakout_events`・`h1_dow_trend_direction`・
`simulate_dow_theory_trend`・`make_price_shock_check`・`detect_candidate1`)に
そのまま渡す。OHLC/ATRの値・順序は不変(シフトはラベルのみ)、m5(エントリー/
イグジット判定に使う実時刻データ)は一切変更しない。これにより「H1バーの
ラベルが開始時刻でなく確定(終了)時刻を正しく表していたら」という反実仮想を、
本番コードのロジックを一切変更せずに再現できる(h1_dow_trend_direction等の
純粋に位置ベースの関数は本質的に無関係、日付演算を行う関数のみ影響を受ける)。

正式プロトコル外の探索的診断。00-spec.md等は変更しない。KPI判定・採否判断には
反映しない(反映するかどうかは司令塔判断)。

出力: research/method-notes/h1_confirm_time_lookahead_impact.json
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
    INITIAL_CAPITAL_USD, MAX_LEVERAGE, PERIODS, RISK_PCT_PER_TRADE,
    SLIPPAGE_PIPS_MARKET_LEG, SLIPPAGE_PIPS_STOP_TRIGGERED, SPREAD_PIPS,
    STOP_BUFFER_ATR_M5, TP_CUM_FRACTION, TP_LEVELS_TRAILONLY, load_m5_period,
    pip_size,
)
from backtest_vol_continuation_candidates_trendfilter_4pairs_trainonly import detect_candidate1  # noqa: E402
from derive_vol_breakout_entry_params import to_h1  # noqa: E402
from evaluate_vol_breakout_dow_theory_kpi import evaluate_period  # noqa: E402
from price_shock_filter import make_price_shock_check  # noqa: E402
from minmax_fx_dt.backtest.permutation import permutation_test_block, permutation_test_clustered  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

CONFIRM_SHIFT = pd.Timedelta(hours=1)  # H1バー長(既定resample('1h'))と一致させる


def shifted(h1: pd.DataFrame, atr_h1: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """h1/atr_h1のインデックスラベルを+1時間シフトしたコピーを返す(値・順序は不変)."""
    h1_c = h1.copy()
    h1_c.index = h1.index + CONFIRM_SHIFT
    atr_c = atr_h1.copy()
    atr_c.index = atr_h1.index + CONFIRM_SHIFT
    return h1_c, atr_c


def find_trades_variant(pair: str, m5: pd.DataFrame, shock_check, use_confirm_time: bool) -> tuple[list[dict], int, int, int]:
    h1_raw = to_h1(m5)
    atr_h1_raw = atr_ind(h1_raw["high"], h1_raw["low"], h1_raw["close"], length=14)
    atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)

    if use_confirm_time:
        h1, atr_h1 = shifted(h1_raw, atr_h1_raw)
    else:
        h1, atr_h1 = h1_raw, atr_h1_raw

    up, down = detect_candidate1(h1, atr_h1)
    positions, directions = [], []
    for i in range(len(h1)):
        if bool(up.iloc[i]):
            positions.append(i)
            directions.append("UP")
        elif bool(down.iloc[i]):
            positions.append(i)
            directions.append("DOWN")

    dedup_positions = select_non_overlapping_breakout_events(h1.index, positions, directions)
    dedup_directions = dict(zip(positions, directions, strict=True))

    n_events_dedup = len(dedup_positions)
    n_events_trendfiltered = 0
    trades = []
    n_entries_within_forming_bar = 0  # 診断用: エントリーがブレイクバー確定(=+60分)より前に発生した件数
    for pos in dedup_positions:
        h1_trend = h1_dow_trend_direction(h1, atr_h1, pos)
        if h1_trend is None:
            continue
        n_events_trendfiltered += 1
        direction = dedup_directions[pos]
        break_time = h1.index[pos]
        new_trades = simulate_dow_theory_trend(
            m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER_M5,
            blackout_check=shock_check, tp_levels=TP_LEVELS_TRAILONLY, skip_first_entry=False,
            atr_trail_series=atr_m5, m5_exit=True, breakeven_trigger_r=BREAKEVEN_TRIGGER_R)
        if not use_confirm_time:
            for t in new_trades:
                delay_min = (pd.Timestamp(t["entry_time"]) - break_time).total_seconds() / 60
                if delay_min < 60:
                    n_entries_within_forming_bar += 1
        trades.extend(new_trades)
    return trades, len(positions), n_events_dedup, n_events_trendfiltered, n_entries_within_forming_bar


def run_period_variant(start: str, end: str, use_confirm_time: bool) -> dict:
    label = "confirm_time_fixed" if use_confirm_time else "original(bug)"
    print(f"\n--- candidate1_n_breakout_only [{label}] ---")
    m5_by_pair, h1_by_pair, atr_h1_by_pair = {}, {}, {}
    for pair in SELECTED_PAIRS:
        m5 = load_m5_period(pair, start, end)
        if len(m5) < 1000:
            continue
        m5_by_pair[pair] = m5
        h1_raw = to_h1(m5)
        atr_h1_raw = atr_ind(h1_raw["high"], h1_raw["low"], h1_raw["close"], length=14)
        if use_confirm_time:
            h1_by_pair[pair], atr_h1_by_pair[pair] = shifted(h1_raw, atr_h1_raw)
        else:
            h1_by_pair[pair], atr_h1_by_pair[pair] = h1_raw, atr_h1_raw
    shock_check = make_price_shock_check(h1_by_pair, atr_h1_by_pair)

    all_trades: list[dict] = []
    n_raw_total = n_dedup_total = n_trendfiltered_total = n_entries_within_forming_bar_total = 0
    for pair, m5 in m5_by_pair.items():
        trades, n_raw, n_dedup, n_trendfiltered, n_forming = find_trades_variant(pair, m5, shock_check, use_confirm_time)
        n_raw_total += n_raw
        n_dedup_total += n_dedup
        n_trendfiltered_total += n_trendfiltered
        n_entries_within_forming_bar_total += n_forming
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
    perm_result = permutation_test_clustered(r_values, pairs_for_perm, seed=42) if len(r_values) >= 4 else None
    perm_result_block = permutation_test_block(r_values, day_clusters_for_perm, seed=42) if len(r_values) >= 4 else None
    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r < 0]
    profit_factor_val = (sum(wins) / abs(sum(losses))) if losses else None
    payoff_val = (float(np.mean(wins)) / abs(float(np.mean(losses)))) if wins and losses else None
    win_rate = (len(wins) / len(r_values)) if r_values else None
    mean_r_net = float(np.mean(r_values)) if r_values else None

    print(f"  イベント総数={n_raw_total}(dedup後{n_dedup_total}、判定不能除外後{n_trendfiltered_total})  "
          f"トレード数={n}(うちブレイクバー形成中エントリー{n_entries_within_forming_bar_total}件)  "
          f"勝率={win_rate}  PF={profit_factor_val}  ペイオフ={payoff_val}  "
          f"perm_p(block)={perm_result_block.p_value if perm_result_block else None}")

    return {
        "variant": label,
        "period": "train", "start": start, "end": end,
        "n_events_raw": n_raw_total, "n_events_dedup": n_dedup_total,
        "n_events_trendfiltered": n_trendfiltered_total,
        "n_entries_within_forming_bar": n_entries_within_forming_bar_total,
        "n_trades": n, "win_rate": round(win_rate, 4) if win_rate else None,
        "mean_r_net": round(mean_r_net, 4) if mean_r_net else None,
        "profit_factor": round(profit_factor_val, 3) if profit_factor_val else None,
        "payoff_ratio": round(payoff_val, 3) if payoff_val else None,
        "perm_p_clustered": round(perm_result.p_value, 4) if perm_result else None,
        "perm_p_block": round(perm_result_block.p_value, 4) if perm_result_block else None,
        "final_balance": balance,
        "total_return_pct": round((balance / INITIAL_CAPITAL_USD - 1) * 100, 2),
        "trades": [
            {k: (str(v) if isinstance(v, pd.Timestamp) else (round(v, 6) if isinstance(v, float) else v))
             for k, v in t.items()}
            for t in all_trades
        ],
        "equity_curve": equity_curve,
    }


def main() -> int:
    start, end = PERIODS["train"]
    print("=== N_BREAKOUT確定時刻の先読み問題: SYS-FX012候補①(現行凍結設計)Trainへの影響 ===")

    result_bug = run_period_variant(start, end, use_confirm_time=False)
    result_fixed = run_period_variant(start, end, use_confirm_time=True)

    kpi_bug = evaluate_period("train", result_bug, perm_p_field="perm_p_block",
                               apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)
    kpi_fixed = evaluate_period("train", result_fixed, perm_p_field="perm_p_block",
                                 apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)

    print("\n=== 比較サマリ ===")
    print(f"{'項目':<24}{'既存(バグ)':>14}{'確定時刻修正後':>16}")
    for key, label in [("n_events_raw", "検出イベント数(生)"), ("n_events_dedup", "dedup後"),
                        ("n_events_trendfiltered", "判定不能除外後"), ("n_trades", "トレード数"),
                        ("win_rate", "勝率"), ("mean_r_net", "平均R(net)"),
                        ("profit_factor", "Profit Factor"), ("payoff_ratio", "ペイオフレシオ"),
                        ("perm_p_block", "permutation p(block)"), ("total_return_pct", "総リターン%")]:
        print(f"{label:<24}{str(result_bug.get(key)):>14}{str(result_fixed.get(key)):>16}")
    print(f"{'必須KPI達成数':<24}{kpi_bug.get('kpi_required_pass_count'):>14}"
          f"{kpi_fixed.get('kpi_required_pass_count'):>16}")

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "purpose": (
            "N_BREAKOUT値動き分析で発見した「h1.index[break_idx]をバー確定時刻として"
            "扱っているが実際はバー開始時刻」という問題が、SYS-FX012の現行凍結設計"
            "(候補①、フォワードテスト実施中)のTrain評価に与える影響を定量化する"
        ),
        "method": (
            "h1/atr_h1のインデックスラベルを一律+1時間(H1バー長)シフトしたコピーを"
            "既存の公式関数群(simulate_dow_theory_trend等)にそのまま渡す自然実験。"
            "共有本番コード(backtest_vol_breakout_dow_theory.py等)は一切変更していない。"
            "m5(実時刻の値動きデータ)は不変。"
        ),
        "caveat": (
            "正式プロトコル外の探索的診断。stop_buffer_atr_m5・atr_trail_multiplier_m5等の"
            "パラメータはTrain(バグあり版)から導出された既存値をそのまま流用しており、"
            "確定時刻修正後に再導出はしていない(再導出すれば影響はさらに変わりうる)。"
            "00-spec.md等は変更せず、KPI判定・採否判断への反映は司令塔判断に委ねる。"
        ),
        "original_bug": {k: v for k, v in result_bug.items() if k not in ("trades", "equity_curve")},
        "confirm_time_fixed": {k: v for k, v in result_fixed.items() if k not in ("trades", "equity_curve")},
        "kpi_original_bug": kpi_bug,
        "kpi_confirm_time_fixed": kpi_fixed,
    }
    out_path = ROOT / "research" / "method-notes" / "h1_confirm_time_lookahead_impact.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
