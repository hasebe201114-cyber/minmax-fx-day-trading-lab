"""Dukascopy M5 の取得漏れ日を検出して補填する（EXP-FX000020 ゲート1）.

## 背景（2026-08-28）

`fetch_dukascopy_m5.py` は `fetch_day()` がリトライ上限まで失敗すると **None を返して
その日を黙って捨てる**設計だった。公開フィードは 503 を返すことがあり、実測では
USD/JPY の 2021-11〜2023-10 で **週末以外に 9日連続を含む複数の欠損**が発生していた
（2023-03-18→03-27 で219時間、2022-06-20→06-27 で168時間 など）。

欠損日があると、バックテストエンジンは隣接する2本のM5バーを連続とみなすため、
**実際には9日離れた価格が「5分間の値動き」に見える**。ATR・ブレイク判定・ストップ
到達のいずれもが壊れるため、この状態で評価してはいけない。

さらに悪いことに、当初の品質ゲート（`extended_data.coverage_report`）は「1日あたりの
バー数の中央値」を見ていたため、**日がまるごと欠けているケースを検出できなかった**
（存在する日はすべて288本なので「密度100%」と報告されてしまう）。

## 本スクリプトの動作

1. 既存CSVを読み、取得対象期間のUTC平日（土曜を除く）のうち **バーが1本も無い日** を列挙
2. その日だけを再取得（リトライを増やし、間隔を空ける）
3. 取得できた分をマージして書き戻す
4. 収束するまで（または改善が止まるまで）繰り返す

Usage:
    python scripts/data/repair_dukascopy_m5.py --tag 2021-11_2023-10
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "dukascopy"

import pandas as pd  # noqa: E402

from fetch_dukascopy_m5 import POINT_DIVISOR, SYMBOL_MAP, decode_day, fetch_day  # noqa: E402

MAX_ROUNDS = 4
PAUSE_BETWEEN = 0.4


def utc_dates(tag: str) -> list[date]:
    start_s, end_s = tag.split("_")
    y0, m0 = map(int, start_s.split("-"))
    y1, m1 = map(int, end_s.split("-"))
    d = date(y0, m0, 1)
    last = date(y1 + (1 if m1 == 12 else 0), 1 if m1 == 12 else m1 + 1, 1) - timedelta(days=1)
    out = []
    while d <= last:
        if d.weekday() != 5:  # 土曜は市場が完全に閉じておりファイルが存在しない
            out.append(d)
        d += timedelta(days=1)
    return out


def _as_jst_index(df: pd.DataFrame) -> pd.DataFrame:
    """concat 後にインデックスの型/tz が崩れることがあるため、毎回正規化する."""
    idx = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True)).tz_convert("Asia/Tokyo")
    out = df.copy()
    out.index = idx
    out.index.name = "timestamp"
    return out.sort_index()


def missing_utc_days(df: pd.DataFrame, targets: list[date]) -> list[date]:
    if df.empty:
        return list(targets)
    have = set(pd.DatetimeIndex(df.index).tz_convert("UTC").date)
    return [d for d in targets if d not in have]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="2021-11_2023-10")
    ap.add_argument("--pairs", nargs="*", default=list(SYMBOL_MAP.keys()))
    args = ap.parse_args()

    targets = utc_dates(args.tag)
    print(f"=== Dukascopy M5 取得漏れの補填: {args.tag}（対象UTC平日 {len(targets)}日）===\n")

    for pair in args.pairs:
        path = OUT_DIR / f"ohlcv_{pair}_5min_{args.tag}.csv"
        if not path.exists():
            print(f"[{pair}] ファイルなし、スキップ: {path.name}")
            continue
        df = _as_jst_index(pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp"))
        miss = missing_utc_days(df, targets)
        print(f"[{pair}] 現在 {len(df):,}本  欠損UTC平日 {len(miss)}日")
        if not miss:
            print("    欠損なし\n")
            continue

        for round_no in range(1, MAX_ROUNDS + 1):
            recovered, rows = 0, []
            for d in miss:
                raw = fetch_day(SYMBOL_MAP[pair], d.year, d.month, d.day)
                got = decode_day(raw, d.year, d.month, d.day, POINT_DIVISOR[pair])
                if got:
                    rows.extend(got)
                    recovered += 1
                time.sleep(PAUSE_BETWEEN)
            if rows:
                m1 = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
                m1["timestamp"] = pd.to_datetime(m1["timestamp"], utc=True).dt.tz_convert("Asia/Tokyo")
                m1 = m1.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
                agg = [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]
                add = pd.DataFrame({c: m1[c].resample("5min").agg(a) for c, a in agg}).dropna()
                df = _as_jst_index(pd.concat([df, add]))
                df = df[~df.index.duplicated(keep="first")]
            before = len(miss)
            miss = missing_utc_days(df, targets)
            print(f"    round{round_no}: {recovered}/{before}日を回収  残り欠損 {len(miss)}日  "
                  f"累計 {len(df):,}本", flush=True)
            if not miss or recovered == 0:
                break

        df.index.name = "timestamp"
        df.to_csv(path)
        status = "完全" if not miss else f"**残り{len(miss)}日欠損**"
        print(f"    -> {path.name} ({len(df):,}行) {status}")
        if miss:
            print(f"       欠損日: {[str(d) for d in miss[:10]]}{' ...' if len(miss) > 10 else ''}")
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
