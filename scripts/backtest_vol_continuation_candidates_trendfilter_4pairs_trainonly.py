"""SYS-FX012検討: 候補①②③それぞれに「H1ダウ理論トレンド判定不能を除外する
フィルター」を重ねがけした場合のTrainベースライン評価(6パターン中、未計算の
3パターン=①'②'③'を計算する。①②③自体は既存の公式KPI評価を再利用)。

司令塔の選択「①②③の3つ全てにそれぞれ適用して比較」を受けて実施。

フィルター方法: 各トレンドイベントの検出バー(break_idx)時点で、H1ダウ理論
トレンド方向(zigzag threshold_atr=2.0、先読みなし、
`analyze_n_breakout_h1_dow_trend_alignment.py`と完全に同一の判定関数)が
「判定不能」(HH+HLでもLH+LLでもない、またはピボット不足)だったイベントは、
検出の時点でスキップする(トレードを生成しない)。これによりコストモデル・
複利のエクイティカーブを事後的にフィルタリングし直す必要がなく、通常の
バックテストと同じ手順で正しい複利計算ができる。

正式プロトコル外の探索的な比較試算。00-spec.md等は変更しない。

出力: research/method-notes/vol_continuation_candidates_trendfilter_4pairs_trainonly_backtest.json
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
from backtest_vol_continuation_hybrid_4pairs_trainonly import DONCHIAN_LENGTH  # noqa: E402
from derive_vol_breakout_entry_params import N_BREAKOUT, to_h1  # noqa: E402
from evaluate_vol_breakout_dow_theory_kpi import evaluate_period  # noqa: E402
from price_shock_filter import CALM_RATIO, make_price_shock_check  # noqa: E402
from minmax_fx_dt.backtest.permutation import (  # noqa: E402
    permutation_test_block, permutation_test_clustered,
)
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402
from minmax_fx_dt.strategy.indicators import donchian  # noqa: E402


def detect_candidate1(h1, atr_h1):
    """候補①: N_BREAKOUT単独。"""
    ratio = (h1["high"] - h1["low"]) / atr_h1
    close, open_ = h1["close"], h1["open"]
    up = ((ratio >= N_BREAKOUT) & (close > open_)).fillna(False)
    down = ((ratio >= N_BREAKOUT) & (close < open_)).fillna(False)
    return up, down


def detect_candidate2(h1, atr_h1):
    """候補②: N_BREAKOUT OR Donchian(CALM_RATIOなし)。"""
    dc = donchian(h1["high"], h1["low"], DONCHIAN_LENGTH, DONCHIAN_LENGTH).shift(1)
    ratio = (h1["high"] - h1["low"]) / atr_h1
    close, open_ = h1["close"], h1["open"]
    is_spike_up = (ratio >= N_BREAKOUT) & (close > open_)
    is_spike_down = (ratio >= N_BREAKOUT) & (close < open_)
    up = (is_spike_up | (close > dc["DCU"])).fillna(False)
    down = (is_spike_down | (close < dc["DCL"])).fillna(False)
    return up, down


def detect_candidate3(h1, atr_h1):
    """候補③: N_BREAKOUT OR (Donchian AND CALM_RATIO)。00-spec.mdの事前登録設計。"""
    dc = donchian(h1["high"], h1["low"], DONCHIAN_LENGTH, DONCHIAN_LENGTH).shift(1)
    ratio = (h1["high"] - h1["low"]) / atr_h1
    close, open_ = h1["close"], h1["open"]
    is_spike_up = (ratio >= N_BREAKOUT) & (close > open_)
    is_spike_down = (ratio >= N_BREAKOUT) & (close < open_)
    is_cont_up = (close > dc["DCU"]) & (ratio >= CALM_RATIO)
    is_cont_down = (close < dc["DCL"]) & (ratio >= CALM_RATIO)
    up = (is_spike_up | is_cont_up).fillna(False)
    down = (is_spike_down | is_cont_down).fillna(False)
    return up, down


CANDIDATES = {
    "candidate1_n_breakout_only": detect_candidate1,
    "candidate2_or_donchian_nocalm": detect_candidate2,
    "candidate3_or_donchian_calm": detect_candidate3,
}


def find_trades_trendfiltered(pair: str, m5: pd.DataFrame, shock_check, detect_fn,
                               cost_ratio_max: float | None = None,
                               max_entry_seq: int | None = None) -> tuple[list[dict], int, int, int]:
    """cost_ratio_max: 改善ループ第5試行(2026-08-22)で追加。Noneなら既存挙動と完全に
    同一(後方互換)。数値を指定すると、エントリー時点で見積もった往復コスト
    (spread+slippage+commission、ストップ決済想定)がinitial_risk(1R)に対して
    この比率を超えるエントリーを見送る(候補③のValidation DD悪化の原因分析で
    判明した「閑散相場でSL幅が極端に狭くなりコスト比率が肥大化する」ケースへの
    対策、詳細はresearch/EXP-FX000006/00-spec.md参照)。

    max_entry_seq: EXP-FX000011(SYS-FX017)で追加(2026-08-23)。Noneなら既存挙動と
    完全に同一(後方互換)。指定時はsimulate_dow_theory_trend()へそのまま渡す。"""
    h1 = to_h1(m5)
    atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)

    up, down = detect_fn(h1, atr_h1)
    positions, directions = [], []
    for i in range(len(h1)):
        if bool(up.iloc[i]):
            positions.append(i)
            directions.append("UP")
        elif bool(down.iloc[i]):
            positions.append(i)
            directions.append("DOWN")

    dedup_positions = select_non_overlapping_breakout_events(h1.index, positions, directions)
    dedup_directions = {pos: d for pos, d in zip(positions, directions)}

    cost_ratio_check = None
    if cost_ratio_max is not None:
        spread = SPREAD_PIPS.get(pair, 0.5)
        pip = pip_size(pair)

        def cost_ratio_check(entry_price: float, initial_risk: float) -> bool:
            # ストップ決済(SL_INITIAL_NO_TP)を想定した保守的な見積もり。
            # TP_LEVELS_TRAILONLY=[]のため実際の往復コストもこの計算式と一致する
            # (n_levels_hit常に0、fraction_via_tp常に0)。
            entry_pips_est = spread + SLIPPAGE_PIPS_MARKET_LEG
            exit_pips_est = spread + SLIPPAGE_PIPS_STOP_TRIGGERED
            cost_price_est = (entry_pips_est + exit_pips_est) * pip
            cost_r_est = cost_price_est / initial_risk
            leverage_ratio_est = entry_price / initial_risk
            commission_r_est = COMMISSION_RATE_ROUND_TRIP * leverage_ratio_est
            return (cost_r_est + commission_r_est) > cost_ratio_max

    n_events_dedup = len(dedup_positions)
    n_events_trendfiltered = 0
    trades = []
    for pos in dedup_positions:
        h1_trend = h1_dow_trend_direction(h1, atr_h1, pos)
        if h1_trend is None:
            continue  # 判定不能イベントはスキップ(トレード生成しない)
        n_events_trendfiltered += 1
        direction = dedup_directions[pos]
        trades.extend(simulate_dow_theory_trend(
            m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER_M5,
            blackout_check=shock_check, tp_levels=TP_LEVELS_TRAILONLY, skip_first_entry=False,
            atr_trail_series=atr_m5, m5_exit=True, breakeven_trigger_r=BREAKEVEN_TRIGGER_R,
            cost_ratio_check=cost_ratio_check, max_entry_seq=max_entry_seq))
    return trades, len(positions), n_events_dedup, n_events_trendfiltered


def run_period(candidate_name: str, detect_fn, start: str, end: str,
                cost_ratio_max: float | None = None,
                max_entry_seq: int | None = None) -> dict:
    print(f"\n--- {candidate_name} (判定不能除外) ---")
    m5_by_pair, h1_by_pair, atr_h1_by_pair = {}, {}, {}
    for pair in SELECTED_PAIRS:
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
            pair, m5, shock_check, detect_fn, cost_ratio_max=cost_ratio_max, max_entry_seq=max_entry_seq)
        n_raw_total += n_raw
        n_dedup_total += n_dedup
        n_trendfiltered_total += n_trendfiltered
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
          f"トレード数={n}  勝率={win_rate}  PF={profit_factor_val}  ペイオフ={payoff_val}")

    return {
        "period": "train", "start": start, "end": end,
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
    results = {}
    kpi_results = {}
    for name, detect_fn in CANDIDATES.items():
        period_result = run_period(name, detect_fn, start, end)
        results[name] = period_result
        kpi = evaluate_period("train", period_result, perm_p_field="perm_p_block",
                               apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)
        kpi_results[name] = kpi
        print(f"  KPI: {kpi['kpi_required_pass_count']}  実効n={kpi['n_trades_effective']}  "
              f"Sharpe={kpi['monthly_sharpe']}  DD={kpi['max_dd_pct']}%  perm_p={kpi['permutation_p_clustered']}")

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "purpose": "候補①②③それぞれにH1ダウ理論判定不能イベント除外フィルターを重ねた場合のTrain評価",
        "backtest": results,
        "kpi": kpi_results,
    }
    out_path = ROOT / "research" / "method-notes" / "vol_continuation_candidates_trendfilter_4pairs_trainonly_backtest.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
