"""SYS-FX012(EXP-FX000006) フォワードテスト・サイクル実行.

司令塔の明示指示(2026-08-22、「このまま進める」)により、Train+Validation
合計で必須KPI13/18(実効n・ペイオフレシオ・permutation有意性が未達)・C品質
チーム未レビューの状態のまま、実際の値動きに対するフォワードテスト(ペーパー
トレード、実発注なし)を開始する。

設計は改善ループ第2試行で確立した最良候補を完全凍結して使う(新規パラメータ
なし): detect_candidate1(N_BREAKOUT=3.5単独) + H1ダウ理論トレンド判定不能
イベント除外フィルター(zigzag threshold_atr_h1=2.0)。対象は4通貨(USD/JPY・
EUR/JPY・GBP/JPY・AUD/JPY)。エントリー層・出口・コストモデルはT-13確定版を
そのまま流用。

cutoff = 2026-08-15 06:00 JST (既存data/curated/ds-1.jsonの4通貨全ての最終
バーの直後)。cutoff以前のH1バーはATR/zigzagの文脈計算にのみ使い、新規
トレンドイベントの検出には使わない(Train/Validation/Testいずれとも非重複)。

データは`data/curated/ds-1-forward.json`(既存ds-1.jsonとは別ファイル、
`scripts/data/fetch_ds1_ohlcv.py`で取得)を使う。本スクリプトは毎回、
cutoff〜最新取得バーまでの全期間を再計算する(差分ではなく全期間再計算、
状態管理を単純化するため)。

出力:
  research/method-notes/sysfx012_forward_test_ledger.json
  (毎回上書き。cutoff以降の全トレード・エクイティカーブ・可能ならKPI評価を含む)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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
    INITIAL_CAPITAL_USD, MAX_LEVERAGE, RISK_PCT_PER_TRADE,
    SLIPPAGE_PIPS_MARKET_LEG, SLIPPAGE_PIPS_STOP_TRIGGERED, SPREAD_PIPS,
    STOP_BUFFER_ATR_M5, TP_CUM_FRACTION, TP_LEVELS_TRAILONLY, pip_size,
)
from backtest_vol_continuation_candidates_trendfilter_4pairs_trainonly import (  # noqa: E402
    detect_candidate1,
)
from derive_vol_breakout_entry_params import to_h1  # noqa: E402
from evaluate_vol_breakout_dow_theory_kpi import evaluate_period  # noqa: E402
from price_shock_filter import make_price_shock_check  # noqa: E402
from minmax_fx_dt.backtest.permutation import (  # noqa: E402
    permutation_test_block, permutation_test_clustered,
)
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

CUTOFF = pd.Timestamp("2026-08-15 06:00:00")  # JST、ds-1.jsonの4通貨最終バー(05:55)の直後
DS1_JSON = ROOT / "data" / "curated" / "ds-1.json"
DS1_FORWARD_JSON = ROOT / "data" / "curated" / "ds-1-forward.json"
OUT_PATH = ROOT / "research" / "method-notes" / "sysfx012_forward_test_ledger.json"


def load_m5_forward(pair: str) -> pd.DataFrame:
    """既存ds-1.json(凍結、warmup文脈用)+ds-1-forward.json(新規)を結合。

    どちらのファイルにも触れず(書き込みしない)、in-memoryで結合するのみ。
    """
    frames = []
    with DS1_JSON.open(encoding="utf-8") as f:
        ds1 = json.load(f)
    if pair in ds1["pairs"]:
        df = pd.DataFrame(ds1["pairs"][pair]["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        frames.append(df.set_index("timestamp"))

    if DS1_FORWARD_JSON.exists():
        with DS1_FORWARD_JSON.open(encoding="utf-8") as f:
            dsf = json.load(f)
        if pair in dsf["pairs"]:
            df = pd.DataFrame(dsf["pairs"][pair]["data"])
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            frames.append(df.set_index("timestamp"))

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def find_forward_trades(pair: str, m5: pd.DataFrame, shock_check, cutoff: pd.Timestamp):
    """凍結済み候補①+判定不能除外フィルターを、cutoff以降のイベントのみに適用。

    cutoff以前のバーはATR(H1)・zigzagピボットの文脈計算にのみ使う(先読みなし、
    Validationでのwarmupと同じ扱い)。新規トレンドイベントの検出(=positions)は
    cutoff以降のH1バーに限定する。
    """
    h1 = to_h1(m5)
    atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)

    up, down = detect_candidate1(h1, atr_h1)
    positions, directions = [], []
    for i in range(len(h1)):
        if h1.index[i] < cutoff:
            continue
        if bool(up.iloc[i]):
            positions.append(i)
            directions.append("UP")
        elif bool(down.iloc[i]):
            positions.append(i)
            directions.append("DOWN")

    dedup_positions = select_non_overlapping_breakout_events(h1.index, positions, directions)
    dedup_directions = {pos: d for pos, d in zip(positions, directions)}

    n_events_dedup = len(dedup_positions)
    n_events_trendfiltered = 0
    trades = []
    for pos in dedup_positions:
        h1_trend = h1_dow_trend_direction(h1, atr_h1, pos)
        if h1_trend is None:
            continue
        n_events_trendfiltered += 1
        direction = dedup_directions[pos]
        trades.extend(simulate_dow_theory_trend(
            m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER_M5,
            blackout_check=shock_check, tp_levels=TP_LEVELS_TRAILONLY, skip_first_entry=False,
            atr_trail_series=atr_m5, m5_exit=True, breakeven_trigger_r=BREAKEVEN_TRIGGER_R))
    return trades, len(positions), n_events_dedup, n_events_trendfiltered


def run_forward_cycle() -> dict:
    m5_by_pair, h1_by_pair, atr_h1_by_pair = {}, {}, {}
    latest_bar_by_pair = {}
    for pair in SELECTED_PAIRS:
        m5 = load_m5_forward(pair)
        m5_by_pair[pair] = m5
        h1_by_pair[pair] = to_h1(m5)
        atr_h1_by_pair[pair] = atr_ind(h1_by_pair[pair]["high"], h1_by_pair[pair]["low"],
                                        h1_by_pair[pair]["close"], length=14)
        latest_bar_by_pair[pair] = str(m5.index.max())
    shock_check = make_price_shock_check(h1_by_pair, atr_h1_by_pair)

    all_trades: list[dict] = []
    n_raw_total = n_dedup_total = n_trendfiltered_total = 0
    for pair, m5 in m5_by_pair.items():
        trades, n_raw, n_dedup, n_trendfiltered = find_forward_trades(pair, m5, shock_check, CUTOFF)
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
    equity_curve = [{"time": str(CUTOFF), "balance": balance}]
    open_positions = []
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
            open_positions.append(idx)
        else:
            if idx in open_positions:
                open_positions.remove(idx)
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
    n_closed = sum(1 for t in all_trades if "dollar_pnl" in t)
    n_open = n - n_closed
    r_values = [t["r_net"] for t in all_trades if not t.get("skipped_ruin") and "dollar_pnl" in t]
    pairs_for_perm = [t["pair"] for t in all_trades if not t.get("skipped_ruin") and "dollar_pnl" in t]
    day_clusters_for_perm = [t["entry_time"].strftime("%Y-%m-%d") for t in all_trades
                              if not t.get("skipped_ruin") and "dollar_pnl" in t]
    perm_result = permutation_test_clustered(r_values, pairs_for_perm, seed=42) if len(r_values) >= 4 else None
    perm_result_block = permutation_test_block(r_values, day_clusters_for_perm, seed=42) if len(r_values) >= 4 else None
    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r < 0]
    profit_factor_val = (sum(wins) / abs(sum(losses))) if losses else None
    payoff_val = (float(np.mean(wins)) / abs(float(np.mean(losses)))) if wins and losses else None
    win_rate = (len(wins) / len(r_values)) if r_values else None
    mean_r_net = float(np.mean(r_values)) if r_values else None

    period_result = {
        "period": "forward_test", "cutoff": str(CUTOFF),
        "n_events_raw": n_raw_total, "n_events_dedup": n_dedup_total,
        "n_events_trendfiltered": n_trendfiltered_total,
        "n_trades_total": n, "n_trades_closed": n_closed, "n_trades_open": n_open,
        "win_rate": round(win_rate, 4) if win_rate else None,
        "mean_r_net": round(mean_r_net, 4) if mean_r_net else None,
        "profit_factor": round(profit_factor_val, 3) if profit_factor_val else None,
        "payoff_ratio": round(payoff_val, 3) if payoff_val else None,
        "perm_p_clustered": round(perm_result.p_value, 4) if perm_result else None,
        "perm_p_block": round(perm_result_block.p_value, 4) if perm_result_block else None,
        "final_balance": round(balance, 2),
        "trades": [
            {k: (str(v) if isinstance(v, pd.Timestamp) else (round(v, 6) if isinstance(v, float) else v))
             for k, v in t.items()}
            for t in all_trades
        ],
        "equity_curve": equity_curve,
    }

    kpi = None
    closed_trades_for_kpi = [t for t in period_result["trades"] if "dollar_pnl" in t]
    if len(closed_trades_for_kpi) > 0:
        kpi_input = dict(period_result)
        kpi_input["trades"] = closed_trades_for_kpi
        kpi = evaluate_period("forward_test", kpi_input, perm_p_field="perm_p_block",
                               apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "design": "SYS-FX012凍結最良候補(候補①N_BREAKOUT単独+H1トレンド判定不能除外フィルター、4通貨)。"
                  "00-spec.md「フォワードテスト仕様」節に事前登録済み。新規パラメータなし",
        "note": "司令塔の明示指示(2026-08-22)により、必須KPI13/18未達・C品質チーム未レビューのまま"
                "フォワードテスト(ペーパートレード、実発注なし)を実施中。cutoff以降の全期間を毎回再計算する",
        "cutoff": str(CUTOFF),
        "latest_bar_by_pair": latest_bar_by_pair,
        "backtest": period_result,
        "kpi": kpi,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def main() -> int:
    print("=== SYS-FX012 フォワードテスト・サイクル実行 ===")
    print(f"cutoff = {CUTOFF} (JST)")
    out = run_forward_cycle()
    p = out["backtest"]
    print(f"\n最新バー: {out['latest_bar_by_pair']}")
    print(f"イベント={p['n_events_raw']}件(dedup後{p['n_events_dedup']}、判定不能除外後{p['n_events_trendfiltered']})")
    print(f"トレード数={p['n_trades_total']}(決済済み{p['n_trades_closed']}、保有中{p['n_trades_open']})")
    print(f"最終残高=${p['final_balance']}  勝率={p['win_rate']}  PF={p['profit_factor']}  "
          f"ペイオフ={p['payoff_ratio']}  perm_p(block)={p['perm_p_block']}")
    if out["kpi"]:
        print(f"KPI: {out['kpi']['kpi_required_pass_count']}")
    else:
        print("KPI: 決済済みトレードなし、評価不可(想定通り。母数がまだ小さい初期段階)")
    print(f"\n[出力]: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
