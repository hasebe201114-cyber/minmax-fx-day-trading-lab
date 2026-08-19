"""EXP-FX000005 フェーズゲート2 第2段階: 戻り確認の探索窓(M)を確定し、
その窓内での戻り幅分布(retrace_ratio導出の材料)を実測する.

## 背景・経緯(HARKing防止のため事前登録の変遷を明記)

`analyze_vol_breakout_frequency.py` の初回実測は「ブレイクバー確定後48H1バー
(=2日間)」を窓としたが、これは司令塔の実際の意図(「初動30分で準備、確定後
30分〜3時間だけトレードする」)とは異なる時間軸だった。2日窓では戻り幅中央値が
ほぼ1.0(=ブレイクバーの逆側まで完全に戻ることが大半)という結果になり、
「戻ってから継続」という設計仮説を否定するように見えたが、これは窓が長すぎた
ことによるアーティファクトだった。

司令塔との確認により、探索窓Mは以下の通り確定した:
- 起点: ブレイクバー(H1)が**確定した**時刻(バーオープン時ではない)
- 終点: 起点から3時間後
- ただし起点から30分は「準備期間」として判定に含めない(実質的な判定窓は
  確定後30分〜3時間)
- 判定粒度: M5の代わりにM15を使用可(データ取得容易性とのバランス、司令塔承認済み)

事前登録: 本スクリプトは上記の確定済み窓定義で戻り幅分布を実測するのみ。
retrace_ratio自体の最終値はこの実測結果を見た上で別途フェーズゲート2の
後続ステップで確定する(本スクリプトはその材料出しに留める)。

N(高ボラ判定のATR倍率)は `vol_breakout_frequency.json` の実測で
「週1回程度」に最も近い N=3.5 (pooled 1.013 events/week, n=261) を暫定採用する。

出力: research/method-notes/vol_breakout_retrace_window.json
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
N_BREAKOUT = 3.5
WINDOW_START_MIN = 30
WINDOW_END_HOURS = 3


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


def to_m15(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: m5[c].resample("15min").agg(a) for c, a in
                          [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]}).dropna()


def measure_event(h1: pd.DataFrame, m15: pd.DataFrame, break_idx: int, direction: str) -> dict | None:
    """確定後30分〜3時間の窓(M15)で、戻り幅・完全戻り・完全反転を測定する."""
    break_bar = h1.iloc[break_idx]
    break_range = float(break_bar["high"] - break_bar["low"])
    if break_range <= 0:
        return None
    break_time = h1.index[break_idx]

    window_start = break_time + pd.Timedelta(minutes=WINDOW_START_MIN)
    window_end = break_time + pd.Timedelta(hours=WINDOW_END_HOURS)
    m15_win = m15[(m15.index > window_start) & (m15.index <= window_end)]
    if len(m15_win) == 0:
        return None

    pre_break = h1.iloc[max(0, break_idx - 5):break_idx]
    if len(pre_break) == 0:
        return None

    if direction == "UP":
        worst = float(m15_win["low"].min())
        retrace = break_bar["high"] - worst
        full_retrace = worst <= break_bar["open"]
        pre_break_level = float(pre_break["low"].min())
        full_reverse = worst <= pre_break_level
    else:
        worst = float(m15_win["high"].max())
        retrace = worst - break_bar["low"]
        full_retrace = worst >= break_bar["open"]
        pre_break_level = float(pre_break["high"].max())
        full_reverse = worst >= pre_break_level

    frac = max(0.0, retrace / break_range)
    return {"retrace_frac": frac, "full_retrace": bool(full_retrace), "full_reverse": bool(full_reverse)}


def main() -> int:
    print("=== EXP-FX000005: 戻り確認探索窓(確定後30分〜3時間, M15) 戻り幅分布実測 (Train期間) ===\n")
    print(f"検出条件: H1レンジ/ATR(H1,14,Wilder) >= N={N_BREAKOUT}")
    print(f"窓: ブレイクバー確定後 {WINDOW_START_MIN}分 〜 {WINDOW_END_HOURS}時間 (M15データ)\n")

    fracs: list[float] = []
    full_retrace_count = 0
    full_reverse_count = 0
    n_events_total = 0
    per_pair = {}

    for pair in PAIRS:
        m5 = load_m5(pair)
        h1 = to_h1(m5)
        m15 = to_m15(m5)
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        ranges = h1["high"] - h1["low"]
        ratio = (ranges / atr_h1).dropna()

        idxs = np.where(ratio.values >= N_BREAKOUT)[0]
        pair_fracs = []
        pair_retrace = 0
        pair_reverse = 0
        for i in idxs:
            pos = h1.index.get_loc(ratio.index[i])
            bar = h1.iloc[pos]
            direction = "UP" if bar["close"] > bar["open"] else "DOWN"
            result = measure_event(h1, m15, pos, direction)
            if result is None:
                continue
            pair_fracs.append(result["retrace_frac"])
            pair_retrace += int(result["full_retrace"])
            pair_reverse += int(result["full_reverse"])

        fracs.extend(pair_fracs)
        full_retrace_count += pair_retrace
        full_reverse_count += pair_reverse
        n_events_total += len(pair_fracs)
        per_pair[pair] = {
            "n_events": len(pair_fracs),
            "full_retrace": pair_retrace,
            "full_reverse": pair_reverse,
            "retrace_frac_median": round(float(np.median(pair_fracs)), 3) if pair_fracs else None,
        }
        print(f"[{pair}] n={len(pair_fracs)}  完全戻り={pair_retrace}  完全反転={pair_reverse}  "
              f"戻り幅中央値={np.median(pair_fracs) if pair_fracs else float('nan'):.3f}")

    fracs_arr = np.array(fracs)
    median = float(np.median(fracs_arr))
    p25 = float(np.percentile(fracs_arr, 25))
    p75 = float(np.percentile(fracs_arr, 75))
    mean = float(np.mean(fracs_arr))
    retrace_rate = full_retrace_count / n_events_total
    reverse_rate = full_reverse_count / n_events_total

    print(f"\n=== プール(5通貨) n={n_events_total} ===")
    print(f"戻り幅: 中央値={median:.3f}  p25={p25:.3f}  p75={p75:.3f}  平均={mean:.3f}")
    print(f"始値まで完全に戻った割合: {full_retrace_count}/{n_events_total} ({retrace_rate:.1%})")
    print(f"ブレイク前(直近5H1バー)水準まで完全に巻き戻った割合: "
          f"{full_reverse_count}/{n_events_total} ({reverse_rate:.1%})")

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_retrace_window.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "n_breakout_threshold": N_BREAKOUT,
            "window_definition": {
                "start_minutes_after_h1_close": WINDOW_START_MIN,
                "end_hours_after_h1_close": WINDOW_END_HOURS,
                "granularity": "M15",
            },
            "n_events_pooled": n_events_total,
            "retrace_frac": {"median": round(median, 3), "p25": round(p25, 3),
                              "p75": round(p75, 3), "mean": round(mean, 3)},
            "full_retrace_rate": round(retrace_rate, 3),
            "full_reverse_rate": round(reverse_rate, 3),
            "per_pair": per_pair,
            "_note": (
                "確定後30分〜3時間(M15)という司令塔確認済みの探索窓での戻り幅分布。"
                "2日窓での初回実測(vol_breakout_frequency.json)は窓が長すぎるアーティファクトで"
                "あったため、本ファイルの数値をspec Mパラメータ・retrace_ratio導出の正式な材料とする。"
            ),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
