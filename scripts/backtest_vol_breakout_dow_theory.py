"""EXP-FX000005 改善ループ第1試行: M5ダウ理論に基づく連続押し目買い版.

背景: `backtest_vol_breakout_train.py`(単発戻り確認エントリー)のTrainベースラインが
5通貨すべてマイナス(mean_R=-0.242)だったことを受け、司令塔からの設計変更案
「検知後は5分足のダウ理論に則って押し目買いを繰り返す。一度トレンドが終わったら終了」
を実装する。事前登録の詳細は`00-spec.md`§改善ループ第1試行を参照。

検出層(H1、N=3.5)は変更しない。エントリー層のみ、単発の戻り確認から、M5の
ZigZag(閾値1.0×ATR(M5)、暫定固定値)によるダウ理論スイング追跡へ変更する:
  - UP方向: 新しい安値ピボットが直前確定安値より高ければ(Higher Low)押し目買い
    エントリー、低ければ(Lower Low)トレンド終了(型崩れ)として以降の新規エントリー
    を停止
  - 追跡起点はブレイクバー自身の安値/高値
  - 安全上限: ブレイク確定後72時間
SL/TP構造(stop_buffer_atr_m5×ATR(M5)、SYS-FX009段階利確方式、atr_trail_multiplier
=3.23をH1バーで判定)は`backtest_vol_breakout_train.py`と同一方法論を踏襲。
各エントリーは独立トレードとして評価する(司令塔確認済み)。

出力: research/method-notes/vol_breakout_dow_theory_train.json
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
from derive_vol_breakout_entry_params import N_BREAKOUT, load_m5, to_h1, PAIRS as ALL_PAIRS
from minmax_fx_dt.backtest.permutation import permutation_test_clustered
from minmax_fx_dt.decision.criteria import compute_n_trades_effective
from minmax_fx_dt.strategy.indicators import atr as atr_ind

TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"

# 事前登録 (00-spec.md §改善ループ第1試行、結果を見る前に固定)
ZIGZAG_THRESHOLD_ATR_M5 = 1.0
MAX_TREND_HOURS = 72
WINDOW_START_MIN = 30
BUFFER_PERCENTILE = 25

with (ROOT / "research" / "EXP-FX000005" / "10-result" / "vol_breakout_entry_params.json").open(encoding="utf-8") as f:
    H1_PARAMS = json.load(f)
ATR_TRAIL_MULTIPLIER = H1_PARAMS["atr_trail_multiplier"]


def is_weekend_close_time(ts: pd.Timestamp) -> bool:
    return ts.weekday() == 5 and ts.hour >= 6


def track_dow_theory_pullbacks(m5: pd.DataFrame, atr_m5: pd.Series, h1: pd.DataFrame,
                                break_idx: int, direction: str) -> list[dict]:
    """M5ダウ理論スイング追跡で押し目買い(UP)/戻り売り(DOWN)エントリー候補を列挙する."""
    break_bar = h1.iloc[break_idx]
    break_time = h1.index[break_idx]
    start_time = break_time + pd.Timedelta(minutes=WINDOW_START_MIN)
    end_time = break_time + pd.Timedelta(hours=MAX_TREND_HOURS)

    start_pos = m5.index.searchsorted(start_time, side="right")
    end_pos = m5.index.searchsorted(end_time, side="right")
    if start_pos >= len(m5) or start_pos >= end_pos:
        return []

    entries: list[dict] = []
    last_confirmed_extreme = float(break_bar["low"]) if direction == "UP" else float(break_bar["high"])
    # 初回は直前のブレイク方向への動きを探索中("SEARCHING_HIGH"=UP方向の場合)
    state = "SEARCHING_HIGH" if direction == "UP" else "SEARCHING_LOW"
    running_extreme = float(m5["high"].iloc[start_pos]) if direction == "UP" else float(m5["low"].iloc[start_pos])
    running_extreme_idx = start_pos

    for i in range(start_pos, end_pos):
        ts = m5.index[i]
        if is_weekend_close_time(ts):
            break
        atr_i = atr_m5.iloc[i]
        if pd.isna(atr_i) or atr_i <= 0:
            continue
        thresh = ZIGZAG_THRESHOLD_ATR_M5 * float(atr_i)
        h_i, l_i, c_i = float(m5["high"].iloc[i]), float(m5["low"].iloc[i]), float(m5["close"].iloc[i])

        if direction == "UP":
            if state == "SEARCHING_HIGH":
                if h_i > running_extreme:
                    running_extreme, running_extreme_idx = h_i, i
                if running_extreme - l_i >= thresh:
                    state = "SEARCHING_LOW"
                    running_extreme, running_extreme_idx = l_i, i
            else:  # SEARCHING_LOW
                if l_i < running_extreme:
                    running_extreme, running_extreme_idx = l_i, i
                if h_i - running_extreme >= thresh:
                    pivot_low = running_extreme
                    if pivot_low > last_confirmed_extreme:
                        entries.append({"confirm_idx": i, "confirm_time": ts, "confirm_price": c_i,
                                         "pivot_price": pivot_low, "pivot_atr": float(atr_i)})
                        last_confirmed_extreme = pivot_low
                        state = "SEARCHING_HIGH"
                        running_extreme, running_extreme_idx = h_i, i
                    else:
                        break  # Lower Low: トレンド終了
        else:  # DOWN
            if state == "SEARCHING_LOW":
                if l_i < running_extreme:
                    running_extreme, running_extreme_idx = l_i, i
                if h_i - running_extreme >= thresh:
                    state = "SEARCHING_HIGH"
                    running_extreme, running_extreme_idx = h_i, i
            else:  # SEARCHING_HIGH
                if h_i > running_extreme:
                    running_extreme, running_extreme_idx = h_i, i
                if running_extreme - l_i >= thresh:
                    pivot_high = running_extreme
                    if pivot_high < last_confirmed_extreme:
                        entries.append({"confirm_idx": i, "confirm_time": ts, "confirm_price": c_i,
                                         "pivot_price": pivot_high, "pivot_atr": float(atr_i)})
                        last_confirmed_extreme = pivot_high
                        state = "SEARCHING_LOW"
                        running_extreme, running_extreme_idx = l_i, i
                    else:
                        break  # Higher High: トレンド終了

    return entries


def main() -> int:
    print("=== EXP-FX000005 改善ループ第1試行: M5ダウ理論連続押し目買い Trainベースライン ===\n")
    print(f"事前登録: zigzag_threshold_atr_m5={ZIGZAG_THRESHOLD_ATR_M5}, "
          f"max_trend_hours={MAX_TREND_HOURS}, atr_trail_multiplier(H1流用)={ATR_TRAIL_MULTIPLIER}\n")

    pooled_bar_range_atr_m5: list[float] = []
    h1_cache: dict[str, pd.DataFrame] = {}
    atr_h1_cache: dict[str, pd.Series] = {}
    m5_cache: dict[str, pd.DataFrame] = {}
    atr_m5_cache: dict[str, pd.Series] = {}
    break_events: dict[str, list[tuple[int, str]]] = {}

    for pair in ALL_PAIRS:
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
    print(f"stop_buffer_atr_m5: pooled M5バーレンジ/ATR比 n={len(pooled_bar_range_atr_m5)}件の"
          f"p{BUFFER_PERCENTILE} = {stop_buffer_atr_m5}\n")

    all_results: list[dict] = []
    trades_per_currency: dict[str, int] = {}
    n_trend_events = 0
    n_pullback_candidates = 0

    for pair in ALL_PAIRS:
        h1, atr_h1 = h1_cache[pair], atr_h1_cache[pair]
        m5, atr_m5 = m5_cache[pair], atr_m5_cache[pair]
        pair_results = []
        for break_idx, direction in break_events[pair]:
            n_trend_events += 1
            pullbacks = track_dow_theory_pullbacks(m5, atr_m5, h1, break_idx, direction)
            n_pullback_candidates += len(pullbacks)
            for pb in pullbacks:
                buffer = stop_buffer_atr_m5 * pb["pivot_atr"]
                stop0 = pb["pivot_price"] - buffer if direction == "UP" else pb["pivot_price"] + buffer
                entry_price = pb["confirm_price"]
                initial_risk = abs(entry_price - stop0)
                if initial_risk <= 0:
                    continue
                entry_h1_idx = int(h1.index.searchsorted(pb["confirm_time"], side="right") - 1)
                if entry_h1_idx < 0 or entry_h1_idx >= len(h1):
                    continue
                entry = dict(pair=pair, direction=direction, entry_idx=entry_h1_idx,
                             entry_time=str(pb["confirm_time"]), entry_price=entry_price,
                             stop0=stop0, initial_risk=initial_risk)
                res = simulate_scaled_scheme(h1, atr_h1, entry, ATR_TRAIL_MULTIPLIER)
                res["pair"] = pair
                res["direction"] = direction
                res["entry_time"] = entry["entry_time"]
                pair_results.append(res)
        all_results.extend(pair_results)
        trades_per_currency[pair] = len(pair_results)
        mean_r = float(np.mean([r["r"] for r in pair_results])) if pair_results else None
        print(f"[{pair}] トレンドイベント={len(break_events[pair])}件  押し目買いトレード={len(pair_results)}件" +
              (f"  mean_R={mean_r:.4f}" if pair_results else ""))

    n_total = len(all_results)
    print(f"\n全体: トレンドイベント={n_trend_events}件  押し目買い候補={n_pullback_candidates}件  "
          f"平均押し目/イベント={n_pullback_candidates / n_trend_events:.2f}" if n_trend_events else "")

    if n_total == 0:
        print("\n有効トレードなし")
        rs, exit_reason_counts, n_eff, perm_p = [], {}, 0.0, None
        profit_factor_val = payoff_val = None
    else:
        rs = [r["r"] for r in all_results]
        exit_reason_counts = {}
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

        wins = [r for r in rs if r > 0]
        losses = [r for r in rs if r < 0]
        profit_factor_val = (sum(wins) / abs(sum(losses))) if losses else None
        payoff_val = (float(np.mean(wins)) / abs(float(np.mean(losses)))) if wins and losses else None
        print(f"PF(Rベース)={profit_factor_val:.3f}" if profit_factor_val else "PF: 算出不可")
        print(f"ペイオフ比(Rベース)={payoff_val:.3f}" if payoff_val else "ペイオフ比: 算出不可")

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_train.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "params": {"n_breakout": N_BREAKOUT, "zigzag_threshold_atr_m5": ZIGZAG_THRESHOLD_ATR_M5,
                       "max_trend_hours": MAX_TREND_HOURS, "stop_buffer_atr_m5": stop_buffer_atr_m5,
                       "atr_trail_multiplier": ATR_TRAIL_MULTIPLIER},
            "n_trend_events": n_trend_events,
            "n_pullback_candidates": n_pullback_candidates,
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
            "_note": (
                "改善ループ第1試行(00-spec.md参照)。M5ダウ理論スイングによる連続押し目買い版。"
                "検出層(H1,N=3.5)・SL/TP構造(段階利確)は単発版と同一、エントリー層のみ変更。"
                "各押し目買いは独立トレードとして評価"
            ),
            "trades": all_results,
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
