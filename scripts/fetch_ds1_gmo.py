"""GMO コイン外国為替FX API から 5 通貨 × M5 足を取得し data/raw/ds-1/ に CSV 保存.

差し戻し 3 (train/val/test 分離) の Test 期間用データ取得.
- Test: 2024-07-01 ~ 2025-12-31 (GMO API)
- Train/Val (2020-01-01 ~ 2024-06-30) は scripts/fetch_ds1_dukascopy.py で取得済

GMO API の制約:
- get_klines は 1 日単位 (date=YYYYMMDD)
- interval: 5min
- rate limit あり (0.5秒 sleep で対応)

Usage:
    python scripts/fetch_ds1_gmo.py --start 2024-07-01 --end 2025-12-31
    python scripts/fetch_ds1_gmo.py --pair USD_JPY --start 2024-07-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from minmax_fx_dt.data import GMOClient

# 通貨ペア一覧 (GMO 形式 = USD_JPY)
PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]

RAW_DIR = ROOT / "data" / "raw" / "ds-1"


def fetch_one_period(client: GMOClient, pair: str, date: datetime) -> list[dict]:
    """1 通貨 × 1 日分を取得."""
    date_str = date.strftime("%Y%m%d")
    try:
        klines = client.get_klines(pair, "5min", date_str)
        if not klines:
            return []
        records = []
        for k in klines:
            # GMO 形式: openTime (Unix ms), open, high, low, close
            ts_ms = int(k.get("openTime", 0))
            if ts_ms == 0:
                continue
            # JST (UTC+9) の pandas Timestamp に変換
            ts = pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert("Asia/Tokyo")
            records.append({
                "timestamp": ts,
                "open": float(k.get("open", 0)),
                "high": float(k.get("high", 0)),
                "low": float(k.get("low", 0)),
                "close": float(k.get("close", 0)),
                "volume": float(k.get("volume", 0)),
            })
        return records
    except Exception as e:
        print(f"    [NG] {pair} {date_str}: {e}")
        return []


def fetch_one(
    client: GMOClient,
    pair: str,
    start: datetime,
    end: datetime,
    sleep_sec: float = 0.5,
) -> Path | None:
    """1 通貨ペアを 1 日ずつ取得して 1 つの CSV に保存."""
    print(f"[{pair}] GMO API から取得中: {start.date()} - {end.date()}")
    t0 = time.time()
    all_records: list[dict] = []
    cur = start
    n_days = 0
    n_empty = 0
    while cur <= end:
        recs = fetch_one_period(client, pair, cur)
        if recs:
            all_records.extend(recs)
        else:
            n_empty += 1
        n_days += 1
        cur += timedelta(days=1)
        time.sleep(sleep_sec)
    elapsed = time.time() - t0
    print(f"  {n_days} 日リクエスト, {len(all_records)} bars 取得 ({n_empty} 日 empty), {elapsed:.1f}秒")

    if not all_records:
        print(f"  [NG] データなし")
        return None

    df = pd.DataFrame(all_records)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"ohlcv_{pair}_5min_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_gmo.csv"
    df.to_csv(out_path, index=False)
    print(f"  [OK] {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"    first: {df.iloc[0].to_dict()}")
    print(f"    last:  {df.iloc[-1].to_dict()}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="GMO API から 5 通貨 M5 足取得")
    parser.add_argument("--start", default="2024-07-01", help="取得開始日 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="取得終了日 (YYYY-MM-DD)")
    parser.add_argument("--pair", choices=PAIRS, help="1 通貨だけ取得")
    parser.add_argument("--sleep", type=float, default=0.5, help="API リクエスト間 sleep 秒数")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    targets = [args.pair] if args.pair else PAIRS
    n_days = (end - start).days + 1
    total_requests = len(targets) * n_days
    est_seconds = total_requests * args.sleep
    print(f"=== GMO API データ取得 ===")
    print(f"期間: {start.date()} - {end.date()} ({n_days} 日)")
    print(f"対象通貨: {targets}")
    print(f"推定リクエスト: {total_requests}, 推定時間: {est_seconds / 60:.1f} 分 (sleep={args.sleep}秒)")
    print(f"出力先: {RAW_DIR}")
    print()

    client = GMOClient("", "")  # Public API (認証不要)

    total_t0 = time.time()
    succeeded = []
    failed = []
    for pair in targets:
        try:
            path = fetch_one(client, pair, start, end, sleep_sec=args.sleep)
            if path:
                succeeded.append(pair)
        except Exception as e:
            failed.append((pair, str(e)[:200]))
            print(f"  [NG] {pair}: {e}")
        print()

    total_elapsed = time.time() - total_t0
    print(f"=== 完了 ===")
    print(f"成功: {len(succeeded)}/{len(targets)} 通貨 ({succeeded})")
    if failed:
        print(f"失敗: {len(failed)} 通貨")
        for p, e in failed:
            print(f"  - {p}: {e}")
    print(f"総時間: {total_elapsed:.1f}秒 ({total_elapsed / 60:.1f}分)")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
