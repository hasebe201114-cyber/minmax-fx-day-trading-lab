"""EXP-FX000005 改善ループ第2試行(4通貨版)のValidation確認.

Train導出パラメータ(4通貨プールで再導出したstop_buffer_atr_m5=0.703、他は
5通貨版から踏襲)を再学習せずValidation期間へそのまま適用する(過学習検出
プロトコル踏襲)。

出力: research/method-notes/vol_breakout_dow_theory_4pairs_validation.json
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

import derive_vol_breakout_entry_params as base  # noqa: E402

VALIDATION_START, VALIDATION_END = "2025-04-01", "2025-11-30"
base.TRAIN_START, base.TRAIN_END = VALIDATION_START, VALIDATION_END

from backtest_vol_breakout_dow_theory import simulate_dow_theory_trend  # noqa: E402
from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402
from derive_vol_breakout_entry_params import N_BREAKOUT, load_m5, to_h1  # noqa: E402
from minmax_fx_dt.backtest.permutation import permutation_test_clustered  # noqa: E402
from minmax_fx_dt.decision.criteria import compute_n_trades_effective  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_train.json").open(encoding="utf-8") as f:
    TRAIN_RESULT = json.load(f)
TRAIN_PARAMS = TRAIN_RESULT["params"]
STOP_BUFFER_ATR_M5 = TRAIN_PARAMS["stop_buffer_atr_m5"]
ATR_TRAIL_MULTIPLIER = TRAIN_PARAMS["atr_trail_multiplier"]


def main() -> int:
    print("=== EXP-FX000005 改善ループ第2試行(4通貨版) Validation確認 ===\n")
    print(f"対象通貨: {SELECTED_PAIRS}")
    print(f"Train導出パラメータをそのまま使用: N={N_BREAKOUT}, stop_buffer_atr_m5={STOP_BUFFER_ATR_M5}, "
          f"atr_trail_multiplier={ATR_TRAIL_MULTIPLIER}\n")

    all_results: list[dict] = []
    trades_per_currency: dict[str, int] = {}
    n_trend_events = 0

    for pair in SELECTED_PAIRS:
        m5 = load_m5(pair)
        h1 = to_h1(m5)
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)

        ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
        idxs = np.where(ratio.values >= N_BREAKOUT)[0]
        events = []
        for i in idxs:
            pos = h1.index.get_loc(ratio.index[i])
            bar = h1.iloc[pos]
            direction = "UP" if bar["close"] > bar["open"] else "DOWN"
            events.append((pos, direction))
        n_trend_events += len(events)

        pair_results = []
        for break_idx, direction in events:
            trades = simulate_dow_theory_trend(m5, atr_m5, h1, atr_h1, break_idx, direction,
                                                STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER)
            for t in trades:
                t["pair"] = pair
            pair_results.extend(trades)
        all_results.extend(pair_results)
        trades_per_currency[pair] = len(pair_results)
        mean_r = float(np.mean([r["r"] for r in pair_results])) if pair_results else None
        print(f"[{pair}] トレンドイベント={len(events)}件  押し目買いトレード={len(pair_results)}件" +
              (f"  mean_R={mean_r:.4f}" if pair_results else ""))

    n_total = len(all_results)
    print(f"\n全体: トレンドイベント={n_trend_events}件  トレード={n_total}件")

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

    train_mean_r = TRAIN_RESULT["mean_r"]
    print(f"\n=== Train比較 ===  Train mean_R={train_mean_r}  Validation mean_R={round(float(np.mean(rs)), 4) if rs else None}")
    if rs and train_mean_r is not None:
        if float(np.mean(rs)) > 0 and train_mean_r > 0:
            print("符号は維持(Train/Validationともにプラス) — 過学習の兆候は見られない")
        elif float(np.mean(rs)) <= 0 <= train_mean_r:
            print("符号が反転 — 過学習の可能性")

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_validation.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "validation_period": [VALIDATION_START, VALIDATION_END],
            "selected_pairs": SELECTED_PAIRS,
            "params_from_train": TRAIN_PARAMS,
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
            "train_mean_r_for_comparison": train_mean_r,
            "trades": all_results,
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
