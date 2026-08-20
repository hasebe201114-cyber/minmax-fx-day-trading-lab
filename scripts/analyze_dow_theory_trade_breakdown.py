"""EXP-FX000005 補助分析: ダウ理論連続押し目買い版の内訳集計.

司令塔依頼: 銘柄別・方向(買い/売り)別のトレード内訳、1つの初動(トレンドイベント)
あたりの平均トレード数、初動確定からトレードエントリーまでの経過時間(平均・最大)。

`backtest_vol_breakout_dow_theory.py`の`simulate_dow_theory_trend()`(1通貨1ポジション
制約 + M5型崩れ後のH1継続確認による再開ロジック込み、2026-08-20修正版)をそのまま
再利用し、Train・Validation両期間について集計する(新たなバックテストではなく
既存結果の内訳を出す集計スクリプト、KPI判定は行わない)。

出力: research/method-notes/vol_breakout_dow_theory_breakdown.json
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

from backtest_vol_breakout_dow_theory import simulate_dow_theory_trend  # noqa: E402
from derive_vol_breakout_entry_params import N_BREAKOUT, to_h1, PAIRS as ALL_PAIRS  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_train.json").open(encoding="utf-8") as f:
    TRAIN_RESULT = json.load(f)
STOP_BUFFER_ATR_M5 = TRAIN_RESULT["params"]["stop_buffer_atr_m5"]
ATR_TRAIL_MULTIPLIER = TRAIN_RESULT["params"]["atr_trail_multiplier"]

PERIODS = {
    "train": ("2023-11-01", "2025-03-31"),
    "validation": ("2025-04-01", "2025-11-30"),
}


def collect_events(period_start: str, period_end: str) -> list[dict]:
    base.TRAIN_START, base.TRAIN_END = period_start, period_end
    from derive_vol_breakout_entry_params import load_m5  # noqa: PLC0415 (再バインド後に取り直す)

    records: list[dict] = []
    for pair in ALL_PAIRS:
        m5 = load_m5(pair)
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
            trades = simulate_dow_theory_trend(m5, atr_m5, h1, atr_h1, pos, direction,
                                                STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER)
            elapsed_hours = [
                (pd.Timestamp(t["entry_time"]) - break_time).total_seconds() / 3600.0 for t in trades
            ]
            records.append({
                "pair": pair, "direction": direction, "break_time": str(break_time),
                "n_trades": len(trades), "elapsed_hours": elapsed_hours,
            })
    return records


def summarize(records: list[dict], group_key) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(group_key(r), []).append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        n_events = len(rs)
        n_trades = sum(r["n_trades"] for r in rs)
        all_elapsed = [h for r in rs for h in r["elapsed_hours"]]
        out[k] = {
            "n_events": n_events,
            "n_trades": n_trades,
            "trades_per_event": round(n_trades / n_events, 3) if n_events else None,
            "elapsed_hours_mean": round(float(np.mean(all_elapsed)), 2) if all_elapsed else None,
            "elapsed_hours_median": round(float(np.median(all_elapsed)), 2) if all_elapsed else None,
            "elapsed_hours_max": round(float(np.max(all_elapsed)), 2) if all_elapsed else None,
        }
    return out


def print_table(title: str, table: dict) -> None:
    print(f"\n--- {title} ---")
    print(f"{'':<10}{'events':>8}{'trades':>8}{'trades/event':>14}{'経過(平均h)':>12}{'経過(中央値h)':>14}{'経過(最大h)':>12}")
    for k, v in table.items():
        print(f"{k:<10}{v['n_events']:>8}{v['n_trades']:>8}{v['trades_per_event']:>14}"
              f"{v['elapsed_hours_mean']:>12}{v['elapsed_hours_median']:>14}{v['elapsed_hours_max']:>12}")


def main() -> int:
    print("=== EXP-FX000005: ダウ理論連続押し目買い版 内訳集計(Train + Validation) ===")

    all_output = {}
    for period_name, (start, end) in PERIODS.items():
        print(f"\n\n########## {period_name.upper()} ({start} 〜 {end}) ##########")
        records = collect_events(start, end)

        by_pair = summarize(records, lambda r: r["pair"])
        print_table("銘柄別", by_pair)

        by_direction = summarize(records, lambda r: r["direction"])
        print_table("方向別(買い=UP/売り=DOWN)", by_direction)

        by_pair_direction = summarize(records, lambda r: f"{r['pair']}_{r['direction']}")
        print_table("銘柄×方向別", by_pair_direction)

        overall = summarize(records, lambda r: "ALL")
        print_table("全体", overall)

        all_output[period_name] = {
            "by_pair": by_pair,
            "by_direction": by_direction,
            "by_pair_direction": by_pair_direction,
            "overall": overall,
        }

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_breakdown.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "stop_buffer_atr_m5": STOP_BUFFER_ATR_M5,
            "n_breakout_threshold": N_BREAKOUT,
            "periods": {k: list(v) for k, v in PERIODS.items()},
            "results": all_output,
            "_note": "トレンドイベント(初動)単位・押し目買いトレード単位の内訳集計。KPI判定は行わない",
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
