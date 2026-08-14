"""dukascopy から 5 通貨 × M5 足を取得し data/raw/ds-1/ に CSV 保存.

差し戻し 3 (train/val/test 分離) のためのデータ取得.
- Train: 2020-01-01 ~ 2022-12-31 (dukascopy)
- Validation: 2023-01-01 ~ 2024-06-30 (dukascopy)
- Test: 2024-07-01 ~ 2025-12-31 (GMO API で別途取得 → scripts/fetch_ds1_gmo.py)

Usage:
    python scripts/fetch_ds1_dukascopy.py --start 2020-01-01 --end 2024-06-30
    python scripts/fetch_ds1_dukascopy.py --pair USD_JPY
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import dukascopy_python
from dukascopy_python.instruments import (
    INSTRUMENT_FX_CROSSES_AUD_JPY,
    INSTRUMENT_FX_CROSSES_EUR_JPY,
    INSTRUMENT_FX_CROSSES_GBP_JPY,
    INSTRUMENT_FX_MAJORS_EUR_USD,
    INSTRUMENT_FX_MAJORS_USD_JPY,
)

# 通貨マッピング (USD_JPY 等は aggregate_ds1.py の命名規則と一致)
PAIRS: dict[str, str] = {
    "USD_JPY": "INSTRUMENT_FX_MAJORS_USD_JPY",
    "EUR_JPY": "INSTRUMENT_FX_CROSSES_EUR_JPY",
    "GBP_JPY": "INSTRUMENT_FX_CROSSES_GBP_JPY",
    "AUD_JPY": "INSTRUMENT_FX_CROSSES_AUD_JPY",
    "EUR_USD": "INSTRUMENT_FX_MAJORS_EUR_USD",
}

INSTRUMENT_MAP = {
    "USD_JPY": INSTRUMENT_FX_MAJORS_USD_JPY,
    "EUR_JPY": INSTRUMENT_FX_CROSSES_EUR_JPY,
    "GBP_JPY": INSTRUMENT_FX_CROSSES_GBP_JPY,
    "AUD_JPY": INSTRUMENT_FX_CROSSES_AUD_JPY,
    "EUR_USD": INSTRUMENT_FX_MAJORS_EUR_USD,
}

# 既存 aggregate_ds1.py の命名規則: ohlcv_<PAIR>_<interval>_<start>_<end>.csv
# 開始日・終了日は YYYYMMDD 形式
RAW_DIR = ROOT / "data" / "raw" / "ds-1"


def fetch_one(pair: str, start: datetime, end: datetime, offer_side: str = "BID") -> Path:
    """1 通貨ペアの OHLCV を dukascopy から取得して CSV 保存."""
    if pair not in INSTRUMENT_MAP:
        raise ValueError(f"Unknown pair: {pair}")

    instrument = INSTRUMENT_MAP[pair]
    offer = (
        dukascopy_python.OFFER_SIDE_BID
        if offer_side == "BID"
        else dukascopy_python.OFFER_SIDE_ASK
    )

    print(f"[{pair}] dukascopy から取得中: {start.date()} - {end.date()} (offer={offer_side})")
    t0 = time.time()
    df = dukascopy_python.fetch(
        instrument=instrument,
        interval=dukascopy_python.INTERVAL_MIN_5,
        offer_side=offer,
        start=start,
        end=end,
    )
    elapsed = time.time() - t0
    print(f"  {len(df)} bars ({elapsed:.1f}秒)")

    if len(df) == 0:
        print(f"  [NG] データなし")
        return None

    # JST (UTC+9) に変換して aggregate_ds1.py の既存形式に揃える
    df.index = df.index.tz_convert("Asia/Tokyo")
    df = df.reset_index()
    # timestamp カラム名に統一
    df = df.rename(columns={"timestamp": "timestamp"})

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"ohlcv_{pair}_5min_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_dukascopy.csv"
    df.to_csv(out_path, index=False)
    print(f"  [OK] {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"    first: {df.iloc[0].to_dict()}")
    print(f"    last:  {df.iloc[-1].to_dict()}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="dukascopy から 5 通貨 M5 足取得")
    parser.add_argument("--start", default="2020-01-01", help="取得開始日 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-06-30", help="取得終了日 (YYYY-MM-DD)")
    parser.add_argument(
        "--pair",
        choices=list(PAIRS.keys()),
        help="1 通貨だけ取得（指定なしで 5 通貨全部）",
    )
    parser.add_argument(
        "--offer-side",
        default="BID",
        choices=["BID", "ASK"],
        help="BID/ASK (デフォルト BID)",
    )
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    targets = [args.pair] if args.pair else list(PAIRS.keys())

    print(f"=== dukascopy データ取得 ===")
    print(f"期間: {start.date()} - {end.date()}")
    print(f"対象通貨: {targets}")
    print(f"出力先: {RAW_DIR}")
    print()

    total_t0 = time.time()
    succeeded = []
    failed = []
    for pair in targets:
        try:
            path = fetch_one(pair, start, end, offer_side=args.offer_side)
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
    print(f"総時間: {total_elapsed:.1f}秒")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
