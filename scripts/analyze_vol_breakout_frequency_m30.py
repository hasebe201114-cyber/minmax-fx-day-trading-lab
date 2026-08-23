"""EXP-FX000008: M30向けN_BREAKOUT再導出 第1段階: 高ボラブレイク検出のN(ATR倍率)候補と
発生頻度の実測(M30版).

`analyze_vol_breakout_frequency.py`(H1版)と完全に同一の方法論をM30リサンプルバーに
適用する。対象通貨はSYS-FX012凍結設計と同一の4通貨(JPYクロス)に揃える。

事前登録(`research/EXP-FX000008/00-spec.md`): プール発生頻度が週1回程度(1.0
events/week に最も近い値)となる候補Nを採用する。本スクリプトは実測のみを行い、
最終選定はこのJSON出力を見た上で(選定ロジック自体は本スクリプトの実行前に固定済み)
後続スクリプトで機械的に行う。

出力: research/method-notes/vol_breakout_frequency_m30.json
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

from minmax_fx_dt.strategy.indicators import atr as atr_ind

from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402

PAIRS = SELECTED_PAIRS
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
N_CANDIDATES = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def load_m5(pair: str) -> pd.DataFrame:
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]


def to_m30(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: m5[c].resample("30min").agg(a) for c, a in
                          [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]}).dropna()


def retracement_fraction(m30: pd.DataFrame, m5: pd.DataFrame, break_idx: int, direction: str,
                          max_wait_hours: int = 48) -> float | None:
    """ブレイクバー確定後、M5でブレイク方向と逆行した最大値幅 ÷ ブレイクバーのレンジ を返す."""
    break_bar = m30.iloc[break_idx]
    break_range = float(break_bar["high"] - break_bar["low"])
    if break_range <= 0:
        return None
    break_time = m30.index[break_idx]
    window_end = break_time + pd.Timedelta(hours=max_wait_hours)
    m5_after = m5[(m5.index > break_time) & (m5.index <= window_end)]
    if len(m5_after) == 0:
        return None
    if direction == "UP":
        worst = float(m5_after["low"].min())
        retrace = break_bar["high"] - worst
    else:
        worst = float(m5_after["high"].max())
        retrace = worst - break_bar["low"]
    return max(0.0, retrace / break_range)


def main() -> int:
    print("=== EXP-FX000008: 高ボラブレイク検出の頻度・戻り幅 基礎統計 (M30版, Train期間) ===\n")

    all_ratios = []
    events_by_n: dict[float, int] = {n: 0 for n in N_CANDIDATES}
    retrace_fracs_by_n: dict[float, list[float]] = {n: [] for n in N_CANDIDATES}
    n_weeks_total = 0

    for pair in PAIRS:
        m5 = load_m5(pair)
        m30 = to_m30(m5)
        atr_m30 = atr_ind(m30["high"], m30["low"], m30["close"], length=14)
        ranges = m30["high"] - m30["low"]
        ratio = (ranges / atr_m30).dropna()
        all_ratios.extend(ratio.tolist())
        n_weeks_total += len(m30) / (48 * 7)

        for n in N_CANDIDATES:
            idxs = np.where(ratio.values >= n)[0]
            events_by_n[n] += len(idxs)
            for i in idxs:
                pos = m30.index.get_loc(ratio.index[i])
                bar = m30.iloc[pos]
                direction = "UP" if bar["close"] > bar["open"] else "DOWN"
                frac = retracement_fraction(m30, m5, pos, direction)
                if frac is not None:
                    retrace_fracs_by_n[n].append(frac)

        print(f"[{pair}] M30バー数={len(m30)}  レンジ/ATR比 p90={ratio.quantile(0.90):.2f} "
              f"p95={ratio.quantile(0.95):.2f} p97={ratio.quantile(0.97):.2f} p99={ratio.quantile(0.99):.2f}")

    all_ratios_arr = np.array(all_ratios)
    print(f"\n全体プール(4通貨): n={len(all_ratios_arr)}バー  約{n_weeks_total:.0f}週相当")
    print(f"レンジ/ATR比の分布: p90={np.percentile(all_ratios_arr,90):.2f}  p95={np.percentile(all_ratios_arr,95):.2f}  "
          f"p97={np.percentile(all_ratios_arr,97):.2f}  p99={np.percentile(all_ratios_arr,99):.2f}\n")

    print(f"{'N候補':<8}{'総イベント数':>10}{'週あたり(4通貨)':>16}{'戻り幅中央値':>12}{'戻り幅p25':>10}{'戻り幅p75':>10}")
    results = {}
    best_n = None
    best_diff = None
    for n in N_CANDIDATES:
        n_events = events_by_n[n]
        per_week = n_events / n_weeks_total if n_weeks_total else 0
        fracs = retrace_fracs_by_n[n]
        med = float(np.median(fracs)) if fracs else None
        p25 = float(np.percentile(fracs, 25)) if fracs else None
        p75 = float(np.percentile(fracs, 75)) if fracs else None
        results[str(n)] = {"n_events": n_events, "events_per_week": round(per_week, 3),
                            "retrace_frac_median": round(med, 3) if med else None,
                            "retrace_frac_p25": round(p25, 3) if p25 else None,
                            "retrace_frac_p75": round(p75, 3) if p75 else None,
                            "n_retrace_samples": len(fracs)}
        diff = abs(per_week - 1.0)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_n = n
        print(f"{n:<8}{n_events:>10}{per_week:>16.3f}"
              f"{med if med else float('nan'):>12.3f}{p25 if p25 else float('nan'):>10.3f}{p75 if p75 else float('nan'):>10.3f}")

    print(f"\n[選定] 週1回程度(1.0 events/week)に最も近い候補: N={best_n} "
          f"(events_per_week={results[str(best_n)]['events_per_week']})")

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_frequency_m30.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "purpose": "EXP-FX000008: M30向けN_BREAKOUT再導出(頻度ベース、週1回程度を目安に選定)",
            "pairs": PAIRS,
            "train_period": [TRAIN_START, TRAIN_END],
            "n_weeks_total_pooled": round(n_weeks_total, 1),
            "range_atr_ratio_percentiles": {
                "p90": round(float(np.percentile(all_ratios_arr, 90)), 3),
                "p95": round(float(np.percentile(all_ratios_arr, 95)), 3),
                "p97": round(float(np.percentile(all_ratios_arr, 97)), 3),
                "p99": round(float(np.percentile(all_ratios_arr, 99)), 3),
            },
            "n_candidates": results,
            "selected_n_breakout": best_n,
            "selection_rule": "プール発生頻度(events/week)が1.0に最も近い候補を選定(spec事前登録)",
            "_note": (
                "H1版(vol_breakout_frequency.json)と同一方法論をM30リサンプルバーに適用した"
                "頻度実測。対象は4通貨(JPYクロス、SYS-FX012凍結設計と同一)。"
            ),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
