"""非公式探索的診断向け: エントリー層をM5→M3にした場合の検証データ取得.

司令塔の質問「トレンドは1Hのまま、エントリーだけ3Mにした場合」への回答材料として、
既存ds-1.json(M5)には無いM3粒度のデータをGMO公開API(interval=1min)から新規取得し、
その場でM3へリサンプルして保存する(生M1データは容量が大きいため永続化しない)。

対象: SYS-FX012凍結設計と同一の4通貨(JPYクロス)、Train期間のみ(2023-11-01〜
2025-03-31)。**正式なDS登録は行わない**(本スクリプトはSYS-FX012の改善ループ・
EXP-FX000008の枠組みとは無関係な、単発の探索的診断用データ取得)。

出力: data/curated/ds-1-m3-train-4pairs.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd

from minmax_fx_dt.data import GMOClient, GMOClientError

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
RATE_LIMIT_SLEEP = 0.1


def fetch_one_day(client: GMOClient, symbol: str, date: pd.Timestamp) -> list[dict]:
    date_str = date.strftime("%Y%m%d")
    try:
        return client.get_klines(symbol, "1min", date_str)
    except GMOClientError as e:
        print(f"  [NG] {symbol} {date_str}: {e}")
        return []


def klines_to_df(klines: list[dict]) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame()
    records = [{
        "timestamp": pd.to_datetime(int(k["openTime"]), unit="ms", utc=True).tz_convert("Asia/Tokyo"),
        "open": float(k["open"]), "high": float(k["high"]),
        "low": float(k["low"]), "close": float(k["close"]),
    } for k in klines]
    return pd.DataFrame(records).set_index("timestamp").sort_index()


def to_m3(m1: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: m1[c].resample("3min").agg(a) for c, a in
                          [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]}).dropna()


def fetch_pair_m3(client: GMOClient, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    all_days = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")
    print(f"[{symbol}] {len(all_days)}日分をM1で取得しM3へリサンプル開始")
    dfs: list[pd.DataFrame] = []
    success = fail = no_data = 0
    for i, day in enumerate(all_days):
        klines = fetch_one_day(client, symbol, day)
        if not klines:
            no_data += 1
        else:
            df = klines_to_df(klines)
            if not df.empty:
                dfs.append(df)
                success += 1
            else:
                fail += 1
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(all_days)}] {symbol} {day.date()} (取得{success}日 失敗{fail}日 休場{no_data}日)")
        time.sleep(RATE_LIMIT_SLEEP)

    if not dfs:
        print(f"  [NG] {symbol}: データ取得できず")
        return pd.DataFrame()

    m1 = pd.concat(dfs).sort_index()
    m1 = m1[~m1.index.duplicated(keep="first")]
    m3 = to_m3(m1)
    print(f"  [OK] {symbol}: M1 {len(m1)}本 → M3 {len(m3)}本")
    return m3


def main() -> int:
    client = GMOClient("", "")
    start, end = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    out: dict = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "interval": "3min",
        "_note": (
            "非公式探索的診断用データ。GMO公開API(interval=1min)から取得したM1をその場でM3へ"
            "リサンプルしたもの。正式なDS登録はしていない。SYS-FX012凍結設計・EXP-FX000008とは"
            "無関係な単発取得。対象4通貨・Train期間のみ。"
        ),
        "pairs": {},
    }
    for pair in PAIRS:
        m3 = fetch_pair_m3(client, pair, start, end)
        if m3.empty:
            continue
        out["pairs"][pair] = {
            "n_bars": len(m3),
            "start": m3.index[0].isoformat(),
            "end": m3.index[-1].isoformat(),
            "columns": list(m3.columns),
            "data": m3.reset_index().to_dict(orient="records"),
        }

    out_path = ROOT / "data" / "curated" / "ds-1-m3-train-4pairs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    total = sum(v["n_bars"] for v in out["pairs"].values())
    print(f"\n[出力]: {out_path} (合計{total}本、{len(out['pairs'])}通貨)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
