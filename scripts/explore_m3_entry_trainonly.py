"""非公式の探索的診断(2026-08-23、司令塔の質問「トレンドは1Hのまま、エントリーだけ
3Mにした場合」への回答): 候補①(N_BREAKOUT単独+H1トレンド判定不能除外フィルター)の
「エントリー層」をM5からM3(3分足)へ差し替えた場合にTrainでどう変化するかを確認する。

**正式なEXP-FX000006改善ループの試行としてはカウントしない**(上限5回は既に消化
済み)。**フォワードテスト中の凍結設計(候補①、M5エントリーベース)には一切影響
しない**探索的診断。

トレンド判定層(H1、N_BREAKOUT=3.5・zigzag閾値=2.0)は一切変更せず、既存ds-1.json
(M5)から`to_h1()`で構築した本来のH1バーをそのまま使う。変更するのはエントリー層
のみ: `simulate_dow_theory_trend()`の第1・第2引数(旧`m5`/`atr_m5`)にM3バーを渡す
(GMO公開API(interval=1min)から新規取得しM3リサンプルした`ds-1-m3-train-4pairs.json`、
`scripts/data/fetch_m1_resample_m3_train_4pairs.py`参照)。

パラメータは一切再導出しない(stop_buffer_atr_m5=0.703・zigzag_threshold_atr_m5=1.0
等、すべてM5版の値をそのまま流用する「純粋なエントリー粒度差し替え」、M30診断と
同じ設計方針)。M3はM5よりバー数が約1.67倍細かいため、ZigZag型崩れ判定・ATRトレール
の効き方がM5版と変わる点に注意。

出力: research/method-notes/explore_m3_entry_trainonly.json
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
from evaluate_vol_breakout_dow_theory_kpi import evaluate_period  # noqa: E402
from price_shock_filter import make_price_shock_check  # noqa: E402
from minmax_fx_dt.backtest.permutation import (  # noqa: E402
    permutation_test_block, permutation_test_clustered,
)
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

M3_JSON_PATH = ROOT / "data" / "curated" / "ds-1-m3-train-4pairs.json"


def load_m3_period(pair: str, start: str, end: str) -> pd.DataFrame:
    with M3_JSON_PATH.open(encoding="utf-8") as f:
        ds = json.load(f)
    df = pd.DataFrame(ds["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= start) & (df.index <= end)]


def find_trades_m3(pair: str, m5: pd.DataFrame, m3: pd.DataFrame, shock_check):
    """検出層・トレンド判定不能除外フィルターはH1(m5から構築、完全に不変)のまま、
    エントリー層のみM3バーに差し替える。"""
    h1 = to_h1(m5)
    atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    atr_m3 = atr_ind(m3["high"], m3["low"], m3["close"], length=14)

    ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
    idxs = np.where(ratio.values >= N_BREAKOUT)[0]
    positions = [h1.index.get_loc(ratio.index[i]) for i in idxs]
    directions = ["UP" if h1.iloc[pos]["close"] > h1.iloc[pos]["open"] else "DOWN" for pos in positions]

    dedup_positions = select_non_overlapping_breakout_events(h1.index, positions, directions)
    dedup_directions = {pos: d for pos, d in zip(positions, directions)}

    n_events_dedup = len(dedup_positions)
    n_events_trendfiltered = 0
    trades = []
    for pos in dedup_positions:
        trend = h1_dow_trend_direction(h1, atr_h1, pos)
        if trend is None:
            continue
        n_events_trendfiltered += 1
        direction = dedup_directions[pos]
        trades.extend(simulate_dow_theory_trend(
            m3, atr_m3, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER_M5,
            blackout_check=shock_check, tp_levels=TP_LEVELS_TRAILONLY, skip_first_entry=False,
            atr_trail_series=atr_m3, m5_exit=True, breakeven_trigger_r=BREAKEVEN_TRIGGER_R))
    return trades, len(positions), n_events_dedup, n_events_trendfiltered


def run_period_m3(start: str, end: str) -> dict:
    m5_by_pair, m3_by_pair, h1_by_pair, atr_h1_by_pair = {}, {}, {}, {}
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
    # 検出層はH1(m5由来)のまま不変のため、価格反応型ショック抑制フィルターもH1基準
    shock_check = make_price_shock_check(h1_by_pair, atr_h1_by_pair)

    all_trades: list[dict] = []
    n_raw_total = n_dedup_total = n_trendfiltered_total = 0
    for pair in m5_by_pair:
        trades, n_raw, n_dedup, n_trendfiltered = find_trades_m3(
            pair, m5_by_pair[pair], m3_by_pair[pair], shock_check)
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
    print("=== 非公式探索的診断: エントリー層をM5→M3に差し替え(トレンド判定はH1不変)、Train評価 ===\n")
    result = run_period_m3(start, end)
    kpi = evaluate_period("train", result, perm_p_field="perm_p_block",
                           apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)

    print(f"イベント: raw={result['n_events_raw']} dedup={result['n_events_dedup']} "
          f"判定不能除外後={result['n_events_trendfiltered']}")
    print(f"トレード数={result['n_trades']}  勝率={result['win_rate']}  PF={result['profit_factor']}  "
          f"ペイオフ={result['payoff_ratio']}")
    print(f"KPI: {kpi['kpi_required_pass_count']}  実効n={kpi['n_trades_effective']}  "
          f"Sharpe={kpi['monthly_sharpe']}  DD={kpi['max_dd_pct']}%  perm_p={kpi['permutation_p_clustered']}")

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "purpose": "非公式探索的診断。EXP-FX000006の正式な改善ループ試行としてはカウントしない。"
                  "フォワードテスト中の凍結設計(候補①、M5エントリーベース)には一切影響しない",
        "design": "候補①(N_BREAKOUT=3.5単独+ダウ理論判定不能除外フィルター、zigzag閾値2.0)を"
                  "トレンド判定層(H1)は不変のまま、エントリー層のみM5からM3へ純粋に粒度差し替え。"
                  "パラメータは一切再導出していない",
        "backtest": result,
        "kpi": kpi,
    }
    out_path = ROOT / "research" / "method-notes" / "explore_m3_entry_trainonly.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
