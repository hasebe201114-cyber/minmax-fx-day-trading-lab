"""EXP-FX000005 フェーズゲート2 第3段階: retrace_ratio・stop_buffer_atr・
atr_trail_multiplierをTrainデータから導出する.

## 事前登録(結果を見る前に確定する導出方法)

- **retrace_ratio**: `vol_breakout_retrace_window.json`(確定済みの探索窓=確定後
  30分〜3時間・M15)で既に測定済みの戻り幅分布の**中央値**を0.05刻みに丸めた値。
  SYS-FX009の`atr_trail_multiplier`が「フォロースルー分布の中央値」から導出された
  前例(`derive_double_pattern_params_h1.py`)を踏襲し、retrace_ratioも中央値ベース
  とする(中央値=0.528 → 0.5)。
- **エントリー確定ロジック**(spec記載の「M5で反転確認」をM15で代替、司令塔承認済み):
  探索窓(確定後30分〜3時間)のM15バーを時系列に走査し、
  1. まだ閾値未達なら、そのバーの安値(UP)/高値(DOWN)がretrace_ratio水準に達したかを判定
  2. 閾値到達後、そのバー以降で終値がブレイク方向へ転じた(直前M15終値からの上昇/下落)
     M15バーが出た時点のその終値でエントリー確定
  3. 閾値到達後、確定前にブレイク方向と逆側へブレイクバーの始値を超えて進んだら、
     そのイベントは見送り(無効化、spec記載の条件をそのまま適用)
  4. 窓内に確定・無効化のいずれも起きなければタイムアウト
- **stop_buffer_atr**: SYS-FX009 H1版と同一方法論(`derive_double_pattern_params_h1.py`)
  を踏襲。H1バーレンジ/ATR(H1,14,Wilder)の分布のp25(=「典型的な1本分のノイズ幅」)。
  この分布はパターン非依存の汎用統計量のため、SYS-FX009 H1版と同じ値になることが
  期待されるが、本戦略専用の成果物として同一Train期間・5通貨で再計算する。
- **atr_trail_multiplier**: SYS-FX009 H1版と同一方法論(確定エントリー後のフォロー
  スルーMFEの中央値、ATR単位)を踏襲するが、観測窓はSYS-FX009(H4由来の週単位保有、
  120H1本=5日)ではなく、**本戦略の想定保有期間(spec記載「数時間〜数日」)の上限に
  合わせ72H1本(3日)**とする(H1足へのスケーリングという方法論は同じだが、窓の長さ
  自体は保有期間想定の違いを反映して意図的に変更する)。

出力: research/EXP-FX000005/10-result/vol_breakout_entry_params.json
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
BUFFER_PERCENTILE = 25
TRAIL_HORIZON_HOURS = 72


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


def to_m30(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: m5[c].resample("30min").agg(a) for c, a in
                          [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]}).dropna()


def find_entry(h1: pd.DataFrame, m15: pd.DataFrame, break_idx: int, direction: str,
               retrace_ratio: float) -> dict | None:
    """探索窓内で閾値到達→反転確認→エントリー確定 or 無効化/タイムアウトを判定する."""
    break_bar = h1.iloc[break_idx]
    break_range = float(break_bar["high"] - break_bar["low"])
    if break_range <= 0:
        return None
    break_time = h1.index[break_idx]
    window_start = break_time + pd.Timedelta(minutes=WINDOW_START_MIN)
    window_end = break_time + pd.Timedelta(hours=WINDOW_END_HOURS)

    m15_before = m15[m15.index <= window_start]
    if len(m15_before) == 0:
        return None
    prev_close = float(m15_before["close"].iloc[-1])

    m15_win = m15[(m15.index > window_start) & (m15.index <= window_end)]
    if len(m15_win) == 0:
        return None

    if direction == "UP":
        threshold_price = float(break_bar["high"] - retrace_ratio * break_range)
        invalid_price = float(break_bar["open"])
    else:
        threshold_price = float(break_bar["low"] + retrace_ratio * break_range)
        invalid_price = float(break_bar["open"])

    triggered = False
    retrace_extreme = float(break_bar["high"]) if direction == "UP" else float(break_bar["low"])
    for ts, bar in m15_win.iterrows():
        if not triggered:
            reached = (bar["low"] <= threshold_price) if direction == "UP" else (bar["high"] >= threshold_price)
            if reached:
                triggered = True
        if triggered:
            retrace_extreme = min(retrace_extreme, float(bar["low"])) if direction == "UP" \
                else max(retrace_extreme, float(bar["high"]))
            invalidated = (bar["low"] <= invalid_price) if direction == "UP" else (bar["high"] >= invalid_price)
            if invalidated:
                return {"outcome": "INVALIDATED"}
            reversed_ = (bar["close"] > prev_close) if direction == "UP" else (bar["close"] < prev_close)
            if reversed_:
                return {"outcome": "ENTRY", "entry_time": ts, "entry_price": float(bar["close"]),
                         "retrace_extreme": retrace_extreme}
        prev_close = float(bar["close"])

    return {"outcome": "TIMEOUT"} if triggered else {"outcome": "NO_TRIGGER"}


def main() -> int:
    print("=== EXP-FX000005: retrace_ratio・stop_buffer_atr・atr_trail_multiplier 導出 (Train期間) ===\n")

    with (ROOT / "research" / "method-notes" / "vol_breakout_retrace_window.json").open(encoding="utf-8") as f:
        window_stats = json.load(f)
    median_frac = window_stats["retrace_frac"]["median"]
    retrace_ratio = round(round(median_frac / 0.05) * 0.05, 2)
    print(f"retrace_ratio: 戻り幅分布の中央値={median_frac} → 0.05刻みに丸め = {retrace_ratio}\n")

    outcome_counts: dict[str, int] = {}
    entries: list[dict] = []
    pooled_bar_range_atr: list[float] = []

    h1_cache: dict[str, pd.DataFrame] = {}
    atr_cache: dict[str, pd.Series] = {}

    for pair in PAIRS:
        m5 = load_m5(pair)
        h1 = to_h1(m5)
        m15 = to_m15(m5)
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        h1_cache[pair] = h1
        atr_cache[pair] = atr_h1

        bar_range_atr = ((h1["high"] - h1["low"]) / atr_h1).replace([np.inf, -np.inf], np.nan).dropna()
        pooled_bar_range_atr.extend(bar_range_atr.tolist())

        ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
        idxs = np.where(ratio.values >= N_BREAKOUT)[0]
        pair_outcomes: dict[str, int] = {}
        for i in idxs:
            pos = h1.index.get_loc(ratio.index[i])
            bar = h1.iloc[pos]
            direction = "UP" if bar["close"] > bar["open"] else "DOWN"
            result = find_entry(h1, m15, pos, direction, retrace_ratio)
            if result is None:
                continue
            outcome = result["outcome"]
            pair_outcomes[outcome] = pair_outcomes.get(outcome, 0) + 1
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            if outcome == "ENTRY":
                entries.append({
                    "pair": pair, "break_idx": pos, "direction": direction,
                    "entry_time": result["entry_time"], "entry_price": result["entry_price"],
                    "atr_at_break": float(atr_h1.iloc[pos]),
                })
        print(f"[{pair}] {pair_outcomes}")

    print(f"\n=== プール(5通貨) N={N_BREAKOUT}イベント総数={sum(outcome_counts.values())} ===")
    for k, v in outcome_counts.items():
        print(f"  {k}: {v}")
    n_entries = outcome_counts.get("ENTRY", 0)
    print(f"\n有効エントリー数(ENTRY): {n_entries}")

    stop_buffer_atr = round(float(np.percentile(pooled_bar_range_atr, BUFFER_PERCENTILE)), 3)
    print(f"\nstop_buffer_atr: pooled H1バーレンジ/ATR比 n={len(pooled_bar_range_atr)}件の"
          f"p{BUFFER_PERCENTILE} = {stop_buffer_atr}")

    mfe_list: list[float] = []
    for e in entries:
        h1 = h1_cache[e["pair"]]
        entry_time = e["entry_time"]
        window_h1 = h1[(h1.index > entry_time) & (h1.index <= entry_time + pd.Timedelta(hours=TRAIL_HORIZON_HOURS))]
        if len(window_h1) == 0 or e["atr_at_break"] <= 0:
            continue
        if e["direction"] == "UP":
            mfe = (float(window_h1["high"].max()) - e["entry_price"]) / e["atr_at_break"]
        else:
            mfe = (e["entry_price"] - float(window_h1["low"].min())) / e["atr_at_break"]
        mfe_list.append(mfe)

    atr_trail_multiplier = round(float(np.median(mfe_list)), 2) if mfe_list else None
    print(f"\natr_trail_multiplier: 有効エントリー後{TRAIL_HORIZON_HOURS}時間のMFE(ATR単位) "
          f"n={len(mfe_list)}件の中央値 = {atr_trail_multiplier}")

    out_dir = ROOT / "research" / "EXP-FX000005" / "10-result"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vol_breakout_entry_params.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "n_breakout_threshold": N_BREAKOUT,
            "window_definition": {"start_minutes_after_h1_close": WINDOW_START_MIN,
                                   "end_hours_after_h1_close": WINDOW_END_HOURS, "granularity": "M15"},
            "retrace_ratio": retrace_ratio,
            "retrace_ratio_source_median": median_frac,
            "outcome_counts": outcome_counts,
            "n_valid_entries": n_entries,
            "stop_buffer_atr": stop_buffer_atr,
            "stop_buffer_atr_percentile": BUFFER_PERCENTILE,
            "pooled_n_bars_for_buffer": len(pooled_bar_range_atr),
            "atr_trail_multiplier": atr_trail_multiplier,
            "trail_horizon_hours": TRAIL_HORIZON_HOURS,
            "pooled_n_mfe_samples": len(mfe_list),
            "_note": (
                "retrace_ratioは戻り幅分布(vol_breakout_retrace_window.json)の中央値を0.05刻みに丸めた値。"
                "stop_buffer_atr/atr_trail_multiplierはSYS-FX009 H1版(derive_double_pattern_params_h1.py)と"
                "同一方法論。atr_trail_multiplierの観測窓のみ、本戦略の想定保有期間(数時間〜数日)に"
                "合わせ72時間(3日)に変更。この値はTrain/Validation/Testいずれの結果を見た後も変更しない"
            ),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
