"""EXP-FX000005 フェーズゲート2 第4段階: SYS-FX011(ボラティリティ・ブレイク戦略)の
Train期間ベースライン評価.

エントリー検出(`derive_vol_breakout_entry_params.py`の`find_entry()`、確定済みの
N=3.5・探索窓30分〜3時間・M15・retrace_ratio=0.55)と、SYS-FX009 H1版で確立済みの
段階利確イグジット(`analyze_scaled_exit_diagnostic.py`の`simulate_scaled_scheme()`、
1R=40%/2R=35%/3R=25%、1R到達後BE+ATRトレーリング)をそのまま結線する。

SL配置(spec記載): 戻り局面の極値(retrace_extreme) ∓ ATR(H1,14)×stop_buffer_atr
(導出値0.629)。ATRはブレイクバー確定時点のATR(atr_at_break)を使用(先読み無し)。
atr_trail_multiplierは導出値3.23を使用。

事前登録: 本スクリプトはTrain期間のみで評価する(HARKing防止、Validation/Testは
別途)。SYS-FX007/008/009のフェーズゲート2初回Train評価と同じ精度(R マルチプル・
勝率・pooled n・permutation_p)で判定材料を出す段階であり、実運用コスト込みの
$建て評価(KPI表全項目)はTrain選定基準を通過した後の深掘りで行う(SYS-FX009の
scaled_exit_diagnosticから$1000バックテストへ進んだ流れを踏襲)。

出力: research/method-notes/vol_breakout_train.json
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

from analyze_scaled_exit_diagnostic import simulate_scaled_scheme
from derive_vol_breakout_entry_params import (
    N_BREAKOUT, load_m5, to_h1, to_m15, find_entry, PAIRS as ALL_PAIRS,
)
from minmax_fx_dt.backtest.permutation import permutation_test_clustered
from minmax_fx_dt.decision.criteria import compute_n_trades_effective
from minmax_fx_dt.strategy.indicators import atr as atr_ind

TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"

with (ROOT / "research" / "EXP-FX000005" / "10-result" / "vol_breakout_entry_params.json").open(encoding="utf-8") as f:
    PARAMS = json.load(f)
RETRACE_RATIO = PARAMS["retrace_ratio"]
STOP_BUFFER_ATR = PARAMS["stop_buffer_atr"]
ATR_TRAIL_MULTIPLIER = PARAMS["atr_trail_multiplier"]


def build_entries_for_pair(pair: str) -> tuple[list[dict], pd.DataFrame, pd.Series]:
    m5 = load_m5(pair)
    h1 = to_h1(m5)
    m15 = to_m15(m5)
    atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
    idxs = np.where(ratio.values >= N_BREAKOUT)[0]

    sim_entries = []
    for i in idxs:
        pos = h1.index.get_loc(ratio.index[i])
        bar = h1.iloc[pos]
        direction = "UP" if bar["close"] > bar["open"] else "DOWN"
        result = find_entry(h1, m15, pos, direction, RETRACE_RATIO)
        if result is None or result["outcome"] != "ENTRY":
            continue
        entry_time = result["entry_time"]
        entry_price = result["entry_price"]
        retrace_extreme = result["retrace_extreme"]
        atr_at_break = float(atr_h1.iloc[pos])
        if atr_at_break <= 0:
            continue
        buffer = STOP_BUFFER_ATR * atr_at_break
        stop0 = retrace_extreme - buffer if direction == "UP" else retrace_extreme + buffer
        initial_risk = abs(entry_price - stop0)
        if initial_risk <= 0:
            continue
        entry_h1_idx = int(h1.index.searchsorted(entry_time, side="right") - 1)
        if entry_h1_idx < 0 or entry_h1_idx >= len(h1):
            continue
        sim_entries.append(dict(
            pair=pair, direction=direction, entry_idx=entry_h1_idx, entry_time=str(entry_time),
            entry_price=entry_price, stop0=stop0, initial_risk=initial_risk,
        ))
    return sim_entries, h1, atr_h1


def main() -> int:
    print("=== EXP-FX000005: SYS-FX011 Trainベースライン評価 ===\n")
    print(f"パラメータ: N={N_BREAKOUT}, retrace_ratio={RETRACE_RATIO}, "
          f"stop_buffer_atr={STOP_BUFFER_ATR}, atr_trail_multiplier={ATR_TRAIL_MULTIPLIER}\n")

    all_results: list[dict] = []
    trades_per_currency: dict[str, int] = {}
    pnl_per_currency: dict[str, list[float]] = {}

    for pair in ALL_PAIRS:
        sim_entries, h1, atr_h1 = build_entries_for_pair(pair)
        pair_results = []
        for e in sim_entries:
            res = simulate_scaled_scheme(h1, atr_h1, e, ATR_TRAIL_MULTIPLIER)
            res["pair"] = pair
            res["direction"] = e["direction"]
            res["entry_time"] = e["entry_time"]
            pair_results.append(res)
        all_results.extend(pair_results)
        trades_per_currency[pair] = len(pair_results)
        pnl_per_currency[pair] = [r["r"] for r in pair_results]
        mean_r = float(np.mean(pnl_per_currency[pair])) if pair_results else None
        print(f"[{pair}] n={len(pair_results)}  mean_R={mean_r:.4f}" if pair_results
              else f"[{pair}] n=0")

    n_total = len(all_results)
    rs = [r["r"] for r in all_results]
    exit_reason_counts: dict[str, int] = {}
    for r in all_results:
        exit_reason_counts[r["exit_reason"]] = exit_reason_counts.get(r["exit_reason"], 0) + 1
    print(f"\nプール(5通貨) n={n_total}  mean_R={np.mean(rs):.4f}  win_rate={np.mean([r>0 for r in rs]):.3f}")
    print(f"イグジット内訳: {exit_reason_counts}")

    n_eff = compute_n_trades_effective(trades_per_currency, n_total)
    pairs_flat = [r["pair"] for r in all_results]
    perm = permutation_test_clustered(rs, pairs_flat, n_permutations=20000, seed=42) if n_total >= 4 else None
    perm_p = perm.p_value if perm else None
    print(f"実効トレード数: {n_eff:.1f}  min_n_trades(>=300)={'PASS' if n_eff >= 300 else 'FAIL'}  "
          f"permutation_p(clustered)={perm_p}")

    profit_factor_val = (sum(r for r in rs if r > 0) / abs(sum(r for r in rs if r < 0))
                          if any(r < 0 for r in rs) else None)
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    payoff_val = (float(np.mean(wins)) / abs(float(np.mean(losses)))
                  if wins and losses else None)

    print(f"PF(Rベース)={profit_factor_val:.3f}" if profit_factor_val else "PF: 算出不可(負けトレード無し)")
    print(f"ペイオフ比(Rベース)={payoff_val:.3f}" if payoff_val else "ペイオフ比: 算出不可")

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_train.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "params": {"n_breakout": N_BREAKOUT, "retrace_ratio": RETRACE_RATIO,
                       "stop_buffer_atr": STOP_BUFFER_ATR, "atr_trail_multiplier": ATR_TRAIL_MULTIPLIER},
            "n_trades_total": n_total,
            "trades_per_currency": trades_per_currency,
            "mean_r": round(float(np.mean(rs)), 4) if rs else None,
            "win_rate": round(float(np.mean([r > 0 for r in rs])), 3) if rs else None,
            "profit_factor_r": round(profit_factor_val, 3) if profit_factor_val else None,
            "payoff_ratio_r": round(payoff_val, 3) if payoff_val else None,
            "exit_reason_counts": exit_reason_counts,
            "n_trades_effective": n_eff,
            "min_n_trades_pass": n_eff >= 300,
            "permutation_p_clustered": perm_p,
            "_note": (
                "R マルチプル単位の簡易評価(実運用コスト・$建て複利サイジング未反映)。"
                "SYS-FX009のscaled_exit_diagnostic→$1000バックテストと同じ2段階アプローチの第1段階。"
                "この段階でmin_n_trades等が未達でも即REJECTとはせず、Validation拡張やコスト込み評価の要否を判断する材料とする"
            ),
            "trades": all_results,
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
