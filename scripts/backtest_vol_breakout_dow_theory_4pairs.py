"""EXP-FX000005 改善ループ第2試行: 対象通貨をEUR/USDを除く4通貨に絞り込んだ
Trainベースライン再評価.

背景: `vol_breakout_dow_theory_1000usd_backtest.json`の通貨別内訳可視化で、
EUR/USDがTrain単独でも唯一マイナス(-$394)だったことが判明。EUR/USDは本戦略の
対象5通貨の中で唯一JPY建てでない通貨ペアでもあり(価格スケール・スプレッド
構造が他4通貨と異なる)、データ根拠(Train単独結果)と構造的理由の両方から
除外対象として妥当と判断(司令塔確認済み)。

事前登録: HARKing防止のため、選定基準はTrainデータのみに基づく
(Validation/Testの結果は選定に一切使用しない)。stop_buffer_atr_m5は
4通貨プールから再導出する(5通貨プールでの値0.701とは異なりうる)。
N=3.5・zigzag_threshold_atr_m5=1.0・atr_trail_multiplier=3.23は既存の
5通貨版導出値をそのまま踏襲する(これらは通貨横断的な検出閾値であり、
1通貨除外による再導出は行わない)。

エントリー・イグジットロジックは`backtest_vol_breakout_dow_theory.py`の
`simulate_dow_theory_trend()`(1通貨1ポジション制約+H1継続確認再開ロジック
込み)をそのまま再利用する。

出力: research/method-notes/vol_breakout_dow_theory_4pairs_train.json
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

from backtest_vol_breakout_dow_theory import ATR_TRAIL_MULTIPLIER, simulate_dow_theory_trend
from derive_vol_breakout_entry_params import N_BREAKOUT, load_m5, to_h1
from minmax_fx_dt.backtest.permutation import permutation_test_clustered
from minmax_fx_dt.decision.criteria import compute_n_trades_effective
from minmax_fx_dt.strategy.indicators import atr as atr_ind

TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"

# 事前登録: EUR/USDを除く4通貨(司令塔確認済み、Train単独結果+構造的理由)
SELECTED_PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
ZIGZAG_THRESHOLD_ATR_M5 = 1.0
BUFFER_PERCENTILE = 25


def main() -> int:
    print("=== EXP-FX000005 改善ループ第2試行: 対象4通貨(EUR/USD除外)Trainベースライン ===\n")
    print(f"対象通貨: {SELECTED_PAIRS}\n")

    pooled_bar_range_atr_m5: list[float] = []
    h1_cache: dict[str, pd.DataFrame] = {}
    atr_h1_cache: dict[str, pd.Series] = {}
    m5_cache: dict[str, pd.DataFrame] = {}
    atr_m5_cache: dict[str, pd.Series] = {}
    break_events: dict[str, list[tuple[int, str]]] = {}

    for pair in SELECTED_PAIRS:
        m5 = load_m5(pair)
        h1 = to_h1(m5)
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)
        h1_cache[pair], atr_h1_cache[pair] = h1, atr_h1
        m5_cache[pair], atr_m5_cache[pair] = m5, atr_m5

        bar_range_atr_m5 = ((m5["high"] - m5["low"]) / atr_m5).replace([np.inf, -np.inf], np.nan).dropna()
        pooled_bar_range_atr_m5.extend(bar_range_atr_m5.tolist())

        ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
        idxs = np.where(ratio.values >= N_BREAKOUT)[0]
        events = []
        for i in idxs:
            pos = h1.index.get_loc(ratio.index[i])
            bar = h1.iloc[pos]
            direction = "UP" if bar["close"] > bar["open"] else "DOWN"
            events.append((pos, direction))
        break_events[pair] = events

    stop_buffer_atr_m5 = round(float(np.percentile(pooled_bar_range_atr_m5, BUFFER_PERCENTILE)), 3)
    print(f"stop_buffer_atr_m5(4通貨プール再導出): pooled M5バーレンジ/ATR比 n={len(pooled_bar_range_atr_m5)}件の"
          f"p{BUFFER_PERCENTILE} = {stop_buffer_atr_m5}\n")

    all_results: list[dict] = []
    trades_per_currency: dict[str, int] = {}
    n_trend_events = 0

    for pair in SELECTED_PAIRS:
        h1, atr_h1 = h1_cache[pair], atr_h1_cache[pair]
        m5, atr_m5 = m5_cache[pair], atr_m5_cache[pair]
        pair_results = []
        for break_idx, direction in break_events[pair]:
            n_trend_events += 1
            trades = simulate_dow_theory_trend(m5, atr_m5, h1, atr_h1, break_idx, direction,
                                                stop_buffer_atr_m5, ATR_TRAIL_MULTIPLIER)
            for t in trades:
                t["pair"] = pair
            pair_results.extend(trades)
        all_results.extend(pair_results)
        trades_per_currency[pair] = len(pair_results)
        mean_r = float(np.mean([r["r"] for r in pair_results])) if pair_results else None
        print(f"[{pair}] トレンドイベント={len(break_events[pair])}件  押し目買いトレード={len(pair_results)}件" +
              (f"  mean_R={mean_r:.4f}" if pair_results else ""))

    n_total = len(all_results)
    print(f"\n全体: トレンドイベント={n_trend_events}件  トレード={n_total}件  "
          f"平均トレード/イベント={n_total / n_trend_events:.2f}" if n_trend_events else "")

    rs = [r["r"] for r in all_results]
    exit_reason_counts: dict[str, int] = {}
    for r in all_results:
        exit_reason_counts[r["exit_reason"]] = exit_reason_counts.get(r["exit_reason"], 0) + 1
    print(f"\nプール(4通貨) n={n_total}  mean_R={np.mean(rs):.4f}  win_rate={np.mean([r>0 for r in rs]):.3f}")
    print(f"イグジット内訳: {exit_reason_counts}")

    n_eff = compute_n_trades_effective(trades_per_currency, n_total)
    pairs_flat = [r["pair"] for r in all_results]
    perm = permutation_test_clustered(rs, pairs_flat, n_permutations=20000, seed=42) if n_total >= 4 else None
    perm_p = perm.p_value if perm else None
    print(f"実効トレード数: {n_eff:.1f}  min_n_trades(>=300)={'PASS' if n_eff >= 300 else 'FAIL'}  "
          f"permutation_p(clustered)={perm_p}")

    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    profit_factor_val = (sum(wins) / abs(sum(losses))) if losses else None
    payoff_val = (float(np.mean(wins)) / abs(float(np.mean(losses)))) if wins and losses else None
    print(f"PF(Rベース)={profit_factor_val:.3f}" if profit_factor_val else "PF: 算出不可")
    print(f"ペイオフ比(Rベース)={payoff_val:.3f}" if payoff_val else "ペイオフ比: 算出不可")

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_train.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "selected_pairs": SELECTED_PAIRS,
            "excluded_pair": "EUR_USD",
            "exclusion_reason": "Train単独でmean_r_net/dollar_pnlがマイナス(唯一)。かつ唯一のJPY建てでない通貨ペア",
            "params": {"n_breakout": N_BREAKOUT, "zigzag_threshold_atr_m5": ZIGZAG_THRESHOLD_ATR_M5,
                       "stop_buffer_atr_m5": stop_buffer_atr_m5, "atr_trail_multiplier": ATR_TRAIL_MULTIPLIER},
            "n_trend_events": n_trend_events,
            "n_trades_total": n_total,
            "trades_per_currency": trades_per_currency,
            "mean_r": round(float(np.mean(rs)), 4) if rs else None,
            "win_rate": round(float(np.mean([r > 0 for r in rs])), 3) if rs else None,
            "profit_factor_r": round(profit_factor_val, 3) if profit_factor_val else None,
            "payoff_ratio_r": round(payoff_val, 3) if payoff_val else None,
            "exit_reason_counts": exit_reason_counts,
            "n_trades_effective": n_eff,
            "min_n_trades_pass": n_eff >= 300 if rs else False,
            "permutation_p_clustered": perm_p,
            "_note": "改善ループ第2試行(00-spec.md参照)。EUR/USDを除く4通貨版。HARKing防止のためTrain単独結果で選定",
            "trades": all_results,
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
