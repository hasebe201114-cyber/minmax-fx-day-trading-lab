"""EXP-FX000005 フェーズゲート2 第1段階: 高ボラブレイク検出のN(ATR倍率)候補と
発生頻度の実測、および戻り幅分布の基礎統計.

spec(`00-spec.md`)で確定した検出定義(H1バーのレンジ/ATR(H1,14,Wilder)比)を
Trainデータで実測し、「週1回程度」に近い頻度になる候補Nを検討する材料を出す。
あわせて、暫定的にN候補ごとのブレイク後M5戻り幅(ブレイク方向と逆行した最大値幅
÷ブレイクバーのレンジ)の分布も測定し、retrace_ratio導出の材料とする。

事前登録: 本スクリプトは基礎統計の実測のみを行い、パラメータの最終決定は行わない
(spec記載の通り、頻度とのバランスを見て複数候補を比較検討する)。

出力: research/method-notes/vol_breakout_frequency.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from minmax_fx_dt.strategy.indicators import atr as atr_ind

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
N_CANDIDATES = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def load_m5(pair: str) -> pd.DataFrame:
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]


def to_h1(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: m5[c].resample("1h").agg(a) for c, a in
                          [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]}).dropna()


def retracement_fraction(h1: pd.DataFrame, m5: pd.DataFrame, break_idx: int, direction: str,
                          max_wait_bars: int = 48) -> float | None:
    """ブレイクバー確定後、M5でブレイク方向と逆行した最大値幅 ÷ ブレイクバーのレンジ を返す."""
    break_bar = h1.iloc[break_idx]
    break_range = float(break_bar["high"] - break_bar["low"])
    if break_range <= 0:
        return None
    break_time = h1.index[break_idx]
    next_time = h1.index[break_idx + 1] if break_idx + 1 < len(h1) else None
    window_end = break_time + pd.Timedelta(hours=max_wait_bars)
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
    print("=== EXP-FX000005: 高ボラブレイク検出の頻度・戻り幅 基礎統計 (Train期間) ===\n")

    all_ratios = []
    events_by_n: dict[float, int] = {n: 0 for n in N_CANDIDATES}
    retrace_fracs_by_n: dict[float, list[float]] = {n: [] for n in N_CANDIDATES}
    n_weeks_total = 0

    for pair in PAIRS:
        m5 = load_m5(pair)
        h1 = to_h1(m5)
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        ranges = h1["high"] - h1["low"]
        ratio = (ranges / atr_h1).dropna()
        all_ratios.extend(ratio.tolist())
        n_weeks_total += len(h1) / (24 * 7)

        for n in N_CANDIDATES:
            idxs = np.where(ratio.values >= n)[0]
            events_by_n[n] += len(idxs)
            for i in idxs:
                pos = h1.index.get_loc(ratio.index[i])
                bar = h1.iloc[pos]
                direction = "UP" if bar["close"] > bar["open"] else "DOWN"
                frac = retracement_fraction(h1, m5, pos, direction)
                if frac is not None:
                    retrace_fracs_by_n[n].append(frac)

        print(f"[{pair}] H1バー数={len(h1)}  レンジ/ATR比 p90={ratio.quantile(0.90):.2f} "
              f"p95={ratio.quantile(0.95):.2f} p97={ratio.quantile(0.97):.2f} p99={ratio.quantile(0.99):.2f}")

    all_ratios_arr = np.array(all_ratios)
    print(f"\n全体プール(5通貨): n={len(all_ratios_arr)}バー  約{n_weeks_total:.0f}週相当")
    print(f"レンジ/ATR比の分布: p90={np.percentile(all_ratios_arr,90):.2f}  p95={np.percentile(all_ratios_arr,95):.2f}  "
          f"p97={np.percentile(all_ratios_arr,97):.2f}  p99={np.percentile(all_ratios_arr,99):.2f}\n")

    print(f"{'N候補':<8}{'総イベント数':>10}{'週あたり(全5通貨)':>16}{'戻り幅中央値':>12}{'戻り幅p25':>10}{'戻り幅p75':>10}")
    results = {}
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
        print(f"{n:<8}{n_events:>10}{per_week:>16.3f}"
              f"{med if med else float('nan'):>12.3f}{p25 if p25 else float('nan'):>10.3f}{p75 if p75 else float('nan'):>10.3f}")

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_frequency.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "n_weeks_total_pooled": round(n_weeks_total, 1),
            "range_atr_ratio_percentiles": {
                "p90": round(float(np.percentile(all_ratios_arr, 90)), 3),
                "p95": round(float(np.percentile(all_ratios_arr, 95)), 3),
                "p97": round(float(np.percentile(all_ratios_arr, 97)), 3),
                "p99": round(float(np.percentile(all_ratios_arr, 99)), 3),
            },
            "n_candidates": results,
            "_note": (
                "高ボラブレイク検出(H1レンジ/ATR比)のN候補ごとの発生頻度と、"
                "ブレイク後M5での戻り幅(ブレイクバーのレンジに対する比率)の分布。"
                "パラメータ最終決定の材料であり、この段階では確定しない。"
            ),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
