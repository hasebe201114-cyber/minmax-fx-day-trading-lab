"""DS-4 セッション境界 + ボラ集計 (DS-1 から自前計算).

DATA-FX001 spec §DS-4: 東京/ロンドン/NY セッションの境界時刻、各セッションの
典型ボラ (出来高は GMO 公開 API に volume 列がないため対象外)。

セッション定義 (JST、DST は簡略化し年間固定とする — 本PJの用途は「エントリー
禁止/推奨の目安」であり、分単位の精度は不要なため):
    東京:     09:00 - 18:00
    ロンドン: 17:00 - 02:00 (翌日)
    NY:       22:00 - 07:00 (翌日)
(セッションは重複しうる。1 本の M5 バーが複数セッションに属してよい)

Usage:
    python scripts/data/build_ds4_sessions.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]

# JST 時刻区間 (開始時, 終了時)。終了 < 開始なら日をまたぐ。
SESSIONS = {
    "TOKYO": (9, 18),
    "LONDON": (17, 2),
    "NEW_YORK": (22, 7),
}


def in_session(hour: int, start: int, end: int) -> bool:
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def load_m5(pair: str) -> pd.DataFrame:
    ds1_path = ROOT / "data" / "curated" / "ds-1.json"
    with ds1_path.open(encoding="utf-8") as f:
        ds1 = json.load(f)
    records = ds1["pairs"][pair]["data"]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def is_jpy_pair(pair: str) -> bool:
    return "JPY" in pair


def main() -> int:
    out = {
        "metadata": {
            "id": "DS-4",
            "name": "セッション境界 + ボラ集計 (DS-1 から自前計算)",
            "generated_at": datetime.now().isoformat(),
            "session_definition_jst": SESSIONS,
            "_note": "出来高は GMO 公開 klines API に volume 列がないため対象外。"
                     "ボラは M5 バーの高安値幅 (pips) の平均・中央値。"
                     "セッション時刻は DST を簡略化した年間固定値 (エントリー可否の"
                     "目安用途であり分単位の精度は不要なため)。",
        },
        "pairs": {},
    }

    for pair in PAIRS:
        print(f"[{pair}] 集計中...")
        df = load_m5(pair)
        pip_size = 0.01 if is_jpy_pair(pair) else 0.0001
        range_pips = (df["high"] - df["low"]) / pip_size
        hour = df.index.hour

        # 時間帯別 (0-23 JST) 集計
        hourly = {}
        for h in range(24):
            mask = hour == h
            if mask.sum() == 0:
                continue
            vals = range_pips[mask]
            hourly[str(h)] = {
                "n_bars": int(mask.sum()),
                "mean_range_pips": round(float(vals.mean()), 3),
                "median_range_pips": round(float(vals.median()), 3),
            }

        # セッション別集計 (重複あり)
        session_stats = {}
        for name, (start, end) in SESSIONS.items():
            mask = np.array([in_session(h, start, end) for h in hour])
            vals = range_pips[mask]
            session_stats[name] = {
                "jst_hours": f"{start:02d}:00-{end:02d}:00",
                "n_bars": int(mask.sum()),
                "mean_range_pips": round(float(vals.mean()), 3) if mask.sum() > 0 else 0.0,
                "median_range_pips": round(float(vals.median()), 3) if mask.sum() > 0 else 0.0,
            }

        out["pairs"][pair] = {
            "n_bars_total": len(df),
            "period_start": str(df.index[0]),
            "period_end": str(df.index[-1]),
            "hourly_jst": hourly,
            "sessions": session_stats,
        }
        for name, s in session_stats.items():
            print(f"  {name:<10} {s['jst_hours']:<12} mean={s['mean_range_pips']:>6.2f}pips  n={s['n_bars']}")

    out_path = ROOT / "data" / "curated" / "ds-4.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
