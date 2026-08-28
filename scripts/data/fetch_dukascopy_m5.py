"""Dukascopy公開ヒストリカルフィードからM1を取得し、M5(5分足)へ再構成する.

## 背景（2026-08-28、司令塔判断により方針変更）

`fetch_dukascopy_h1.py` は「頑健性チェック専用、正式KPI評価には使わない」という
方針で作られた。しかし SYS-FX007〜024 の検証を通じて、**実効トレード数(min_n_trades
= 300)の未達が、戦略の優劣とは無関係に5戦略以上の判断を止めてきた**ことが判明した
（OBS000012 §1）。本PJのスコープ「中期スイング・保有期間 数日〜数週間」と
「Train 17ヶ月 × 4通貨で実効n 300」は構造的に両立しない。

n を増やす方向として通貨拡大は SYS-FX016/019 で失敗済み（質が希薄化する）。
残る正しい方向は**時間軸の拡張**であり、GMO の壁(2023-10-27)より前を埋められる
のは Dukascopy だけである。

司令塔判断「使ってよい（保守的コスト仮定を条件に）」を受け、**本スクリプトが取得
するデータは正式な Train 評価に使用する**。ただし当時のスプレッドは現在より広い
はずであり、コスト仮定は必ず保守側に倒し感度分析を併記すること（EXP-FX000020
の spec で事前登録）。

## 取得方式

エンドポイント: {BASE}/{SYMBOL}/{YYYY}/{MM-1:02d}/{DD:02d}/BID_candles_min_1.bi5
    - {MM} は0始まり（01月=00）、{DD} は0始まりではなく実日付-1 の0始まり
    - 1ファイルが1日分のM1バーを含む（日初(UTC)からの秒オフセット）
    - レコード形式: >IIIIIf (offset_sec, open, close, low, high, volume)、LZMA圧縮
    - volume<=0 は非取引時間の停滞バーとして除外（H1版と同一の扱い）

出力は DS-1 と同一形式（timestamp,open,high,low,close、JST tz-aware）に揃え、
既存ローダ（`grid_portfolio_engine.load_m5` 等）がそのまま読めるようにする。

Usage:
    python scripts/data/fetch_dukascopy_m5.py --start 2018-11 --end 2023-10
    python scripts/data/fetch_dukascopy_m5.py --start 2018-11 --end 2018-11 --pairs USD_JPY  # スモーク
"""

from __future__ import annotations

import argparse
import calendar
import lzma
import struct
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "dukascopy"

import pandas as pd  # noqa: E402

SYMBOL_MAP = {
    "USD_JPY": "USDJPY", "EUR_JPY": "EURJPY", "GBP_JPY": "GBPJPY",
    "AUD_JPY": "AUDJPY", "EUR_USD": "EURUSD",
}
POINT_DIVISOR = {
    "USD_JPY": 1000.0, "EUR_JPY": 1000.0, "GBP_JPY": 1000.0,
    "AUD_JPY": 1000.0, "EUR_USD": 100000.0,
}

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
MAX_RETRIES = 5
WORKERS = 8           # 公開フィードへの同時接続数 (16 は実測でむしろ悪化)
REQUEST_PAUSE = 0.0
CHUNK_MONTHS = 6      # メモリ節約のため数ヶ月ずつ取得してM5へ畳む


def month_range(start: str, end: str):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        m = 1 if m == 12 else m + 1
        if m == 1:
            y += 1


def fetch_day(symbol: str, year: int, month: int, day: int) -> bytes | None:
    """1日分のM1 bi5を取得。データ不存在(404)はNoneを返す."""
    url = f"{BASE_URL}/{symbol}/{year}/{month - 1:02d}/{day - 1:02d}/BID_candles_min_1.bi5"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError):
            time.sleep(1.5 ** attempt)
    return None


def decode_day(raw: bytes | None, year: int, month: int, day: int, divisor: float) -> list[tuple]:
    if not raw:
        return []
    # Dukascopy は LZMA-alone 形式 (先頭 0x5D)。xz (0xFD'7zXZ') ではない点に注意。
    # 先頭バイトで分岐すると decompress を丸ごと飛ばして無意味なバイト列を読むことになる
    # (2026-08-28 に実際に踏んだ)。無条件に decompress し、失敗時のみ生バイト列とみなす。
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError:
        data = raw
    day_start = datetime(year, month, day, tzinfo=timezone.utc)
    rows = []
    for i in range(len(data) // 24):
        offset_sec, o, c, lo, hi, vol = struct.unpack(">IIIIIf", data[i * 24:(i + 1) * 24])
        if vol <= 0.0:
            continue  # 非取引時間の停滞バー (H1版と同一の扱い)
        ts = day_start + timedelta(seconds=offset_sec)
        rows.append((ts, o / divisor, hi / divisor, lo / divisor, c / divisor))
    return rows


def fetch_chunk_m5(pair: str, ym_list: list[tuple[int, int]]) -> pd.DataFrame:
    """複数ヶ月分のM1を並列取得し、M5へ再構成して返す (JST tz-aware)."""
    symbol, divisor = SYMBOL_MAP[pair], POINT_DIVISOR[pair]
    tasks: list[tuple[int, int, int]] = []
    for year, month in ym_list:
        for d in range(1, calendar.monthrange(year, month)[1] + 1):
            # 土曜は市場が完全に閉じており必ず空ファイルになるため、リクエスト自体を送らない
            # (日曜は NY 22:00 UTC の週明けオープンを含むため除外しない)
            if datetime(year, month, d).weekday() == 5:
                continue
            tasks.append((year, month, d))

    def work(t: tuple[int, int, int]) -> list[tuple]:
        year, month, day = t
        raw = fetch_day(symbol, year, month, day)
        time.sleep(REQUEST_PAUSE)
        return decode_day(raw, year, month, day, divisor)

    rows: list[tuple] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for chunk in ex.map(work, tasks):
            rows.extend(chunk)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    m1 = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    m1["timestamp"] = pd.to_datetime(m1["timestamp"], utc=True).dt.tz_convert("Asia/Tokyo")
    m1 = m1.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    agg = [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]
    return pd.DataFrame({c: m1[c].resample("5min").agg(a) for c, a in agg}).dropna()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM (両端含む)")
    ap.add_argument("--end", required=True, help="YYYY-MM (両端含む)")
    ap.add_argument("--pairs", nargs="*", default=["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    months = list(month_range(args.start, args.end))
    print(f"=== Dukascopy M1→M5 取得: {args.start} 〜 {args.end} ({len(months)}ヶ月) ===")
    print("※ 司令塔判断(2026-08-28)により、本データは正式なTrain評価に使用する。")
    print("   ただしコスト仮定は保守側に倒すこと (EXP-FX000020 spec で事前登録)\n")

    for pair in args.pairs:
        out_path = OUT_DIR / f"ohlcv_{pair}_5min_{args.start}_{args.end}.csv"
        if out_path.exists():
            print(f"[{pair}] 既存ファイルあり、スキップ: {out_path.name}")
            continue
        print(f"[{pair}] 取得開始")
        frames = []
        t0 = time.time()
        chunks = [months[i:i + CHUNK_MONTHS] for i in range(0, len(months), CHUNK_MONTHS)]
        done = 0
        for chunk in chunks:
            frames.append(fetch_chunk_m5(pair, chunk))
            done += len(chunk)
            total = sum(len(f) for f in frames)
            el = time.time() - t0
            print(f"    {pair}: {done}/{len(months)}ヶ月  累計M5={total:,}本  経過{el/60:.1f}分", flush=True)
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="first")]
        out.index.name = "timestamp"
        out.to_csv(out_path)
        print(f"    -> {out_path.name} ({len(out):,}行, {out.index[0]} 〜 {out.index[-1]})\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
