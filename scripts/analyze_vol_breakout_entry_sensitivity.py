"""EXP-FX000005 改善ループ第4試行: エントリー機会拡大の感度分析
(ブレイク検出閾値N・M5 ZigZag閾値・経済指標カレンダー窓).

背景: 経済指標カレンダーフィルター追加後も実効サンプル数(min_n_trades=300)への
未達が全期間・全試行を通じて最大のボトルネックとして残っている
(4通貨+カレンダー版: Train113.0/Validation35.0/Test53.4)。司令塔へ提示した
4つの拡大案のうち、司令塔選択「③カレンダー窓の縮小、①N閾値の緩和、
②ZigZag閾値の緩和」を受け、3方向をTrain(4通貨プール、EUR/USD除外は既存の
選定を踏襲)のみで感度分析する。HARKing防止のため選定はTrainのみで行い、
Validation/Testは選定確定後に1回だけ確認する(既存の改善ループプロトコル踏襲)。

事前登録(結果を見る前に固定):
- Part A (①ブレイク検出閾値N ②M5 ZigZag閾値): カレンダーフィルター無しで、
  N∈{3.5(現行), 3.0, 2.5} × zigzag_threshold_atr_m5∈{1.0(現行), 0.8, 0.6}
  の9通りをTrain(4通貨プール)で評価する。stop_buffer_atr_m5はM5バーレンジ/ATR比
  のp25として1回だけ導出し(N・zigzag非依存のため)全候補で共通使用する。
- Part B (③カレンダー窓): 現行のN=3.5・zigzag_threshold_atr_m5=1.0を固定し、
  ブラックアウトバッファ∈{24h(現行), 12h, 6h} × 対象会合∈{BOJ+FOMC(現行),
  BOJのみ} の6通りをTrain(4通貨プール)で評価する。
- 選定ルール: 各パートで「トレード数が現行(N=3.5/zigzag=1.0、または
  24h+BOJ+FOMC)より増加し、かつpooled mean_Rが正を維持する」候補のうち、
  実効トレード数(n_trades_effective)が最大のものを次点候補とする。該当候補が
  無い場合は、その方向での拡大は不採用と結論する(強制的にどれかを選ばない)。

出力: research/method-notes/vol_breakout_entry_sensitivity.json
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
from derive_vol_breakout_entry_params import load_m5, to_h1
from economic_calendar import BOJ_MEETINGS, FOMC_MEETINGS, make_blackout_check
from minmax_fx_dt.backtest.permutation import permutation_test_clustered
from minmax_fx_dt.decision.criteria import compute_n_trades_effective
from minmax_fx_dt.strategy.indicators import atr as atr_ind

TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"

SELECTED_PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
BUFFER_PERCENTILE = 25
CURRENT_N = 3.5
CURRENT_ZIGZAG = 1.0
CURRENT_BUFFER_HOURS = 24
N_CANDIDATES = [3.5, 3.0, 2.5]
ZIGZAG_CANDIDATES = [1.0, 0.8, 0.6]
BUFFER_CANDIDATES = [24, 12, 6]


def detect_break_events(h1: pd.DataFrame, atr_h1: pd.Series, n_breakout: float) -> list[tuple[int, str]]:
    ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
    idxs = np.where(ratio.values >= n_breakout)[0]
    events = []
    for i in idxs:
        pos = h1.index.get_loc(ratio.index[i])
        bar = h1.iloc[pos]
        direction = "UP" if bar["close"] > bar["open"] else "DOWN"
        events.append((pos, direction))
    return events


def evaluate(
    pairs: list[str],
    h1_cache: dict, atr_h1_cache: dict, m5_cache: dict, atr_m5_cache: dict,
    break_events_by_pair: dict[str, list[tuple[int, str]]],
    stop_buffer_atr_m5: float, zigzag_threshold: float,
    blackout_check=None,
) -> dict:
    all_results: list[dict] = []
    trades_per_currency: dict[str, int] = {}
    n_trend_events = 0
    for pair in pairs:
        h1, atr_h1 = h1_cache[pair], atr_h1_cache[pair]
        m5, atr_m5 = m5_cache[pair], atr_m5_cache[pair]
        pair_results = []
        for break_idx, direction in break_events_by_pair[pair]:
            n_trend_events += 1
            trades = simulate_dow_theory_trend(
                m5, atr_m5, h1, atr_h1, break_idx, direction,
                stop_buffer_atr_m5, ATR_TRAIL_MULTIPLIER,
                blackout_check=blackout_check, zigzag_threshold_atr_m5=zigzag_threshold,
            )
            for t in trades:
                t["pair"] = pair
            pair_results.extend(trades)
        all_results.extend(pair_results)
        trades_per_currency[pair] = len(pair_results)

    n_total = len(all_results)
    if n_total == 0:
        return {"n_trend_events": n_trend_events, "n_trades_total": 0, "trades_per_currency": trades_per_currency,
                "mean_r": None, "win_rate": None, "profit_factor_r": None, "n_trades_effective": 0.0,
                "permutation_p_clustered": None}

    rs = [r["r"] for r in all_results]
    n_eff = compute_n_trades_effective(trades_per_currency, n_total)
    pairs_flat = [r["pair"] for r in all_results]
    perm = permutation_test_clustered(rs, pairs_flat, n_permutations=20000, seed=42) if n_total >= 4 else None
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else None
    return {
        "n_trend_events": n_trend_events,
        "n_trades_total": n_total,
        "trades_per_currency": trades_per_currency,
        "mean_r": round(float(np.mean(rs)), 4),
        "win_rate": round(float(np.mean([r > 0 for r in rs])), 3),
        "profit_factor_r": round(pf, 3) if pf else None,
        "n_trades_effective": round(n_eff, 1),
        "permutation_p_clustered": perm.p_value if perm else None,
    }


def main() -> int:
    print("=== EXP-FX000005 改善ループ第4試行: エントリー機会拡大の感度分析(N/ZigZag/カレンダー) ===\n")

    h1_cache: dict[str, pd.DataFrame] = {}
    atr_h1_cache: dict[str, pd.Series] = {}
    m5_cache: dict[str, pd.DataFrame] = {}
    atr_m5_cache: dict[str, pd.Series] = {}
    pooled_bar_range_atr_m5: list[float] = []

    for pair in SELECTED_PAIRS:
        m5 = load_m5(pair)
        h1 = to_h1(m5)
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)
        h1_cache[pair], atr_h1_cache[pair] = h1, atr_h1
        m5_cache[pair], atr_m5_cache[pair] = m5, atr_m5
        bar_range = ((m5["high"] - m5["low"]) / atr_m5).replace([np.inf, -np.inf], np.nan).dropna()
        pooled_bar_range_atr_m5.extend(bar_range.tolist())

    stop_buffer_atr_m5 = round(float(np.percentile(pooled_bar_range_atr_m5, BUFFER_PERCENTILE)), 3)
    print(f"stop_buffer_atr_m5(4通貨プール、N/zigzag非依存): {stop_buffer_atr_m5}\n")

    # --- Part A: N x zigzag_threshold グリッド(カレンダーフィルター無し) ---
    print("--- Part A: ブレイク検出閾値N x M5 ZigZag閾値 ---\n")
    part_a: list[dict] = []
    for n_val in N_CANDIDATES:
        break_events_by_pair = {pair: detect_break_events(h1_cache[pair], atr_h1_cache[pair], n_val)
                                 for pair in SELECTED_PAIRS}
        for zz in ZIGZAG_CANDIDATES:
            r = evaluate(SELECTED_PAIRS, h1_cache, atr_h1_cache, m5_cache, atr_m5_cache,
                         break_events_by_pair, stop_buffer_atr_m5, zz)
            r["n_breakout"] = n_val
            r["zigzag_threshold_atr_m5"] = zz
            part_a.append(r)
            print(f"  N={n_val} zigzag={zz}: events={r['n_trend_events']} trades={r['n_trades_total']} "
                  f"mean_R={r['mean_r']} PF={r['profit_factor_r']} n_eff={r['n_trades_effective']} "
                  f"perm_p={r['permutation_p_clustered']}")

    baseline_a = next(r for r in part_a if r["n_breakout"] == CURRENT_N and r["zigzag_threshold_atr_m5"] == CURRENT_ZIGZAG)
    candidates_a = [r for r in part_a
                    if r["n_trades_total"] > baseline_a["n_trades_total"] and (r["mean_r"] or 0) > 0
                    and not (r["n_breakout"] == CURRENT_N and r["zigzag_threshold_atr_m5"] == CURRENT_ZIGZAG)]
    best_a = max(candidates_a, key=lambda r: r["n_trades_effective"]) if candidates_a else None
    print(f"\n  現行(N={CURRENT_N}, zigzag={CURRENT_ZIGZAG}): trades={baseline_a['n_trades_total']} "
          f"mean_R={baseline_a['mean_r']} n_eff={baseline_a['n_trades_effective']}")
    if best_a:
        print(f"  [選定] N={best_a['n_breakout']}, zigzag={best_a['zigzag_threshold_atr_m5']}: "
              f"trades={best_a['n_trades_total']} mean_R={best_a['mean_r']} n_eff={best_a['n_trades_effective']}")
    else:
        print("  [選定] 該当候補なし(トレード数増加かつmean_R>0を満たす組み合わせが存在しない)")

    # --- Part B: カレンダー窓バッファ x 対象会合(N=3.5, zigzag=1.0固定) ---
    print("\n--- Part B: 経済指標カレンダー窓の縮小(N=3.5, zigzag=1.0固定) ---\n")
    break_events_current = {pair: detect_break_events(h1_cache[pair], atr_h1_cache[pair], CURRENT_N)
                             for pair in SELECTED_PAIRS}
    meeting_sets = {"boj_fomc": None, "boj_only": BOJ_MEETINGS}
    part_b: list[dict] = []
    for buf in BUFFER_CANDIDATES:
        for scope_name, meetings in meeting_sets.items():
            bc = make_blackout_check(buffer_hours=buf, meetings=meetings)
            r = evaluate(SELECTED_PAIRS, h1_cache, atr_h1_cache, m5_cache, atr_m5_cache,
                         break_events_current, stop_buffer_atr_m5, CURRENT_ZIGZAG, blackout_check=bc)
            r["buffer_hours"] = buf
            r["meeting_scope"] = scope_name
            part_b.append(r)
            print(f"  buffer={buf}h scope={scope_name}: trades={r['n_trades_total']} mean_R={r['mean_r']} "
                  f"PF={r['profit_factor_r']} n_eff={r['n_trades_effective']} perm_p={r['permutation_p_clustered']}")

    baseline_b = next(r for r in part_b if r["buffer_hours"] == CURRENT_BUFFER_HOURS and r["meeting_scope"] == "boj_fomc")
    candidates_b = [r for r in part_b
                    if r["n_trades_total"] > baseline_b["n_trades_total"] and (r["mean_r"] or 0) > 0
                    and not (r["buffer_hours"] == CURRENT_BUFFER_HOURS and r["meeting_scope"] == "boj_fomc")]
    best_b = max(candidates_b, key=lambda r: r["n_trades_effective"]) if candidates_b else None
    print(f"\n  現行(buffer={CURRENT_BUFFER_HOURS}h, boj_fomc): trades={baseline_b['n_trades_total']} "
          f"mean_R={baseline_b['mean_r']} n_eff={baseline_b['n_trades_effective']}")
    if best_b:
        print(f"  [選定] buffer={best_b['buffer_hours']}h, scope={best_b['meeting_scope']}: "
              f"trades={best_b['n_trades_total']} mean_R={best_b['mean_r']} n_eff={best_b['n_trades_effective']}")
    else:
        print("  [選定] 該当候補なし(トレード数増加かつmean_R>0を満たす組み合わせが存在しない)")

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_entry_sensitivity.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "selected_pairs": SELECTED_PAIRS,
            "stop_buffer_atr_m5": stop_buffer_atr_m5,
            "current_params": {"n_breakout": CURRENT_N, "zigzag_threshold_atr_m5": CURRENT_ZIGZAG,
                                "buffer_hours": CURRENT_BUFFER_HOURS, "meeting_scope": "boj_fomc"},
            "part_a_grid": part_a,
            "part_a_selected": best_a,
            "part_b_grid": part_b,
            "part_b_selected": best_b,
            "selection_rule": (
                "各パートで現行よりトレード数が増加し、かつpooled mean_Rが正を維持する候補のうち、"
                "n_trades_effectiveが最大のものを選定。該当なしの場合は拡大不採用。"
                "選定はTrainのみに基づく(HARKing防止)。Validation確認は別途1回だけ実施する。"
            ),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
