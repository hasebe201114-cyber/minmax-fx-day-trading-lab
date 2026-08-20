"""EXP-FX000005 高ボラ判定(H1ブレイク検出)の精度確認: エントリー機会を逃した
トレンドイベントの抽出・分析.

司令塔依頼「過去データにより高ボラ判定の精度を確認したい。エントリー機会を
逃した取引を抽出して分析することは出来ますか？」への対応。

「高ボラ判定」= H1レンジ/ATR比 ≥ N_BREAKOUT(3.5) によるブレイク検出。この判定
自体は多くのイベントを正しく拾っている前提で、**検出はされたがエントリーに
至らなかったイベント**(=機会損失の候補)を抽出し、実際にその後どれだけ
値動きがあったか(逃した含み益の大きさ)を定量化する。加えて、機会損失の
原因(①カレンダーブラックアウトによる抑制 ②M5構造上、継続ピボットが一度も
確定しなかった)を切り分ける。

方法: 各ブレイクイベントについて、現行採用candidate(N=3.5・zigzag=1.0・
カレンダーフィルターあり)で`simulate_dow_theory_trend()`を実行しトレード数を
確認。トレード数=0のイベントについて、
  (a) カレンダーフィルターを外して再実行し、トレードが発生するようになれば
      「ブラックアウトによる機会損失」と分類
  (b) それでも0件のままなら「M5構造上、継続ピボットが一度も確定しなかった」
      (=クリーンに伸びきってしまい、押し目/戻りが形成されなかった)と分類
分類後、各イベントの追跡窓内の生の値動き(stop_buffer_atr_m5×ATR(M5,ブレイク時)
を疑似リスク単位とした最大順行幅=MFE、終端時点の純移動量)を計測し、実際に
成立したトレードのr_gross分布と比較する。

出力: research/method-notes/missed_entry_opportunities.json
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

from backtest_vol_breakout_dow_theory import (  # noqa: E402
    ATR_TRAIL_MULTIPLIER, MAX_TREND_HOURS, WINDOW_START_MIN, is_weekend_close_time,
    simulate_dow_theory_trend,
)
from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402
from derive_vol_breakout_entry_params import N_BREAKOUT, to_h1  # noqa: E402
from economic_calendar import is_blackout  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_train.json").open(encoding="utf-8") as f:
    TRAIN_RESULT = json.load(f)
STOP_BUFFER_ATR_M5 = TRAIN_RESULT["params"]["stop_buffer_atr_m5"]

PERIODS = {
    "train":      ("2023-11-01", "2025-03-31"),
    "validation": ("2025-04-01", "2025-11-30"),
    "test":       ("2025-12-01", "2026-08-15"),
}
TP3_R = 4.0
TP_LEVELS = [(1.0, 0.40), (2.0, 0.35), (TP3_R, 0.25)]


def load_m5_period(pair: str, start: str, end: str) -> pd.DataFrame:
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= start) & (df.index <= end)]


def window_price_stats(m5: pd.DataFrame, break_idx_time: pd.Timestamp, direction: str,
                        pseudo_risk: float, break_price_ref: float) -> dict | None:
    """追跡窓[break_time+30min, break_time+72h](週末クローズがあればそこまで)の
    生の値動きを疑似リスク単位(pseudo_risk)で計測する。"""
    start_time = break_idx_time + pd.Timedelta(minutes=WINDOW_START_MIN)
    end_time = break_idx_time + pd.Timedelta(hours=MAX_TREND_HOURS)
    start_pos = m5.index.searchsorted(start_time, side="right")
    end_pos = m5.index.searchsorted(end_time, side="right")
    if start_pos >= len(m5) or start_pos >= end_pos:
        return None
    # 週末クローズで打ち切り
    actual_end = start_pos
    for i in range(start_pos, end_pos):
        if is_weekend_close_time(m5.index[i]):
            break
        actual_end = i + 1
    if actual_end <= start_pos:
        return None
    seg = m5.iloc[start_pos:actual_end]
    if pseudo_risk <= 0:
        return None
    if direction == "UP":
        mfe = (float(seg["high"].max()) - break_price_ref) / pseudo_risk
        mae = (break_price_ref - float(seg["low"].min())) / pseudo_risk
        net = (float(seg["close"].iloc[-1]) - break_price_ref) / pseudo_risk
    else:
        mfe = (break_price_ref - float(seg["low"].min())) / pseudo_risk
        mae = (float(seg["high"].max()) - break_price_ref) / pseudo_risk
        net = (break_price_ref - float(seg["close"].iloc[-1])) / pseudo_risk
    return {"mfe_pseudo_r": round(mfe, 3), "mae_pseudo_r": round(mae, 3), "net_move_pseudo_r": round(net, 3),
            "n_bars": int(actual_end - start_pos)}


def main() -> int:
    print("=== EXP-FX000005 高ボラ判定の精度確認: 機会損失イベントの抽出 ===\n")

    all_events = []
    converted_rs = []
    for period_name, (start, end) in PERIODS.items():
        for pair in SELECTED_PAIRS:
            m5 = load_m5_period(pair, start, end)
            if len(m5) < 1000:
                continue
            h1 = to_h1(m5)
            atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
            atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)
            ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
            idxs = np.where(ratio.values >= N_BREAKOUT)[0]

            for i in idxs:
                pos = h1.index.get_loc(ratio.index[i])
                bar = h1.iloc[pos]
                direction = "UP" if bar["close"] > bar["open"] else "DOWN"
                break_time = h1.index[pos]

                trades_with_blackout = simulate_dow_theory_trend(
                    m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER,
                    blackout_check=is_blackout, tp_levels=TP_LEVELS,
                )
                n_with = len(trades_with_blackout)

                if n_with > 0:
                    converted_rs.extend(t["r"] for t in trades_with_blackout)
                    all_events.append({
                        "period": period_name, "pair": pair, "direction": direction,
                        "break_time": str(break_time), "outcome": "converted",
                        "n_trades": n_with, "reason": None,
                    })
                    continue

                # トレード0件: ブラックアウト無しで再実行して原因を切り分け
                trades_no_blackout = simulate_dow_theory_trend(
                    m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER,
                    blackout_check=None, tp_levels=TP_LEVELS,
                )
                reason = "blackout_suppressed" if len(trades_no_blackout) > 0 else "no_continuation_pivot"

                atr_at_break = float(atr_h1.iloc[pos]) if pd.notna(atr_h1.iloc[pos]) else None
                pseudo_risk = STOP_BUFFER_ATR_M5 * atr_at_break if atr_at_break else None
                stats = window_price_stats(m5, break_time, direction, pseudo_risk, float(bar["close"])) if pseudo_risk else None

                all_events.append({
                    "period": period_name, "pair": pair, "direction": direction,
                    "break_time": str(break_time), "outcome": "missed",
                    "n_trades": 0, "reason": reason,
                    "n_would_be_trades_without_blackout": len(trades_no_blackout),
                    **({} if stats is None else stats),
                })

    df = pd.DataFrame(all_events)
    n_total = len(df)
    n_converted = int((df["outcome"] == "converted").sum())
    n_missed = int((df["outcome"] == "missed").sum())
    print(f"全ブレイクイベント数: {n_total}  変換(≥1トレード): {n_converted}  機会損失(0トレード): {n_missed}")
    print(f"機会損失率: {n_missed / n_total:.3f}\n")

    missed = df[df["outcome"] == "missed"].copy()
    reason_counts = missed["reason"].value_counts().to_dict()
    print("--- 機会損失の内訳(理由別) ---")
    for reason, cnt in reason_counts.items():
        sub = missed[missed["reason"] == reason]
        mfe = sub["mfe_pseudo_r"].dropna()
        net = sub["net_move_pseudo_r"].dropna()
        print(f"  {reason}: n={cnt}"
              + (f"  MFE(疑似R)中央値={mfe.median():.2f} p75={mfe.quantile(0.75):.2f}" if len(mfe) else "")
              + (f"  純移動(疑似R)中央値={net.median():.2f}" if len(net) else ""))

    print(f"\n--- 参考: 実際に成立したトレードのr_gross分布(n={len(converted_rs)}) ---")
    if converted_rs:
        arr = np.array(converted_rs)
        print(f"  mean={arr.mean():.3f}  median={np.median(arr):.3f}  p25={np.percentile(arr,25):.3f}  p75={np.percentile(arr,75):.3f}")

    no_pivot = missed[missed["reason"] == "no_continuation_pivot"]
    mfe_no_pivot = no_pivot["mfe_pseudo_r"].dropna()
    print(f"\n--- 「継続ピボット未確定」イベント(n={len(no_pivot)})のMFE(疑似R)分布 ---")
    if len(mfe_no_pivot):
        for p in [10, 25, 50, 75, 90]:
            print(f"  p{p}={np.percentile(mfe_no_pivot, p):.2f}")
        print(f"  MFE>=1.0R(TP1相当)の割合: {(mfe_no_pivot >= 1.0).mean():.3f}")
        print(f"  MFE>=3.0Rの割合: {(mfe_no_pivot >= 3.0).mean():.3f}")

    out_path = ROOT / "research" / "method-notes" / "missed_entry_opportunities.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "n_total_events": n_total,
            "n_converted": n_converted,
            "n_missed": n_missed,
            "missed_rate": round(n_missed / n_total, 4) if n_total else None,
            "reason_counts": reason_counts,
            "converted_trades_r_gross_summary": {
                "n": len(converted_rs),
                "mean": round(float(np.mean(converted_rs)), 4) if converted_rs else None,
                "median": round(float(np.median(converted_rs)), 4) if converted_rs else None,
            },
            "events": all_events,
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
