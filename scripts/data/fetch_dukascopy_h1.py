"""Dukascopy公開ヒストリカルフィードからH1(1時間足)OHLCVを取得する.

背景: GMOコイン外国為替FX公開klines APIには2023-10-27頃より前のデータが
原理的に存在しない(データ保持期間の壁、PJ000003 Q6で確認済み)。本スクリプト
は、GMOの壁より前の期間で「方向性の予測力なし」という結論(2026-08-17、
`analyze_signal_ic.py`等)が別データソースでも成立するかを確認する頑健性
チェック専用に、Dukascopy(認証不要の公開ヒストリカルフィード)からH1足を
取得する。

**この取得データは正式なTrain/Validation/Test評価には使わない**
(コストモデルがGMO専用に較正されているため、KPI評価には使えない — CLAUDE.md
「検証ステップ」参照)。あくまで統計的性質(IC・分散比・自己相関)の頑健性
チェック専用。

エンドポイント: https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM}/BID_candles_hour_1.bi5
    - {MM}は0始まり(01月=00)
    - 1ファイルが1ヶ月分のH1バーを含む(月初からの秒オフセット×24レコード/日)
    - レコード形式: >IIIIIf (offset_sec, open, close, low, high, volume)、
      価格は整数(pair毎のpoint値で除算)、LZMA圧縮

Usage:
    python scripts/data/fetch_dukascopy_h1.py --start 2018-11 --end 2023-10
"""

from __future__ import annotations

import argparse
import csv
import lzma
import struct
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "dukascopy"

# minmax_fx_dt側のペア表記 -> Dukascopy銘柄コード
SYMBOL_MAP = {
    "USD_JPY": "USDJPY",
    "EUR_JPY": "EURJPY",
    "GBP_JPY": "GBPJPY",
    "AUD_JPY": "AUDJPY",
    "EUR_USD": "EURUSD",
}

# 価格の除算値 (JPYクォート=3桁小数、それ以外=5桁小数)
POINT_DIVISOR = {
    "USD_JPY": 1000.0,
    "EUR_JPY": 1000.0,
    "GBP_JPY": 1000.0,
    "AUD_JPY": 1000.0,
    "EUR_USD": 100000.0,
}

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
MAX_RETRIES = 5


def month_range(start: str, end: str):
    """'YYYY-MM' 形式の開始・終了(両端含む)を年月タプルで列挙."""
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def fetch_month(symbol: str, year: int, month: int) -> bytes | None:
    """1ヶ月分のH1 bi5を取得。データ不存在(404)はNoneを返す."""
    url = f"{BASE_URL}/{symbol}/{year}/{month - 1:02d}/BID_candles_hour_1.bi5"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionResetError):
            time.sleep(2 ** attempt)
    print(f"    [WARN] {symbol} {year}-{month:02d}: {MAX_RETRIES}回リトライ失敗、スキップ")
    return None


def decode_month(raw: bytes, year: int, month: int, divisor: float) -> list[tuple]:
    """bi5バイト列を(timestamp, open, high, low, close, volume)のリストへ."""
    if not raw:
        return []
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError:
        return []
    month_start = datetime(year, month, 1, tzinfo=timezone.utc)
    n = len(data) // 24
    rows = []
    for i in range(n):
        rec = data[i * 24:(i + 1) * 24]
        offset_sec, o, c, lo, hi, vol = struct.unpack(">IIIIIf", rec)
        if vol <= 0.0:
            continue  # 出来高0 = 非取引時間(週末・祝日)の停滞バー、除外
        ts = month_start + timedelta(seconds=offset_sec)
        rows.append((ts.strftime("%Y-%m-%dT%H:%M:%S"), o / divisor, hi / divisor,
                      lo / divisor, c / divisor, vol))
    return rows


def fetch_pair(pair: str, start: str, end: str) -> list[tuple]:
    symbol = SYMBOL_MAP[pair]
    divisor = POINT_DIVISOR[pair]
    all_rows: list[tuple] = []
    months = list(month_range(start, end))
    for idx, (y, m) in enumerate(months):
        raw = fetch_month(symbol, y, m)
        rows = decode_month(raw, y, m, divisor)
        all_rows.extend(rows)
        if (idx + 1) % 12 == 0 or idx == len(months) - 1:
            print(f"    {pair}: {idx + 1}/{len(months)}ヶ月取得済み (累計{len(all_rows)}バー)")
        time.sleep(0.15)  # 公開フィードへの配慮
    return all_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM (両端含む)")
    ap.add_argument("--end", required=True, help="YYYY-MM (両端含む)")
    ap.add_argument("--pairs", nargs="*", default=list(SYMBOL_MAP.keys()))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Dukascopy H1取得: {args.start} 〜 {args.end} ===")
    print("※ 頑健性チェック専用データ。正式KPI評価には使用しない (コストモデルはGMO専用較正)\n")

    for pair in args.pairs:
        print(f"[{pair}]")
        rows = fetch_pair(pair, args.start, args.end)
        out_path = OUT_DIR / f"ohlcv_{pair}_h1_{args.start}_{args.end}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            writer.writerows(rows)
        print(f"    -> {out_path} ({len(rows)}行)\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
