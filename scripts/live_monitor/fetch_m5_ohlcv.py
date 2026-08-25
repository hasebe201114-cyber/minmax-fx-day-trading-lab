"""SYS-FX012 フォワードテスト用 M5 OHLCV のライブ取得.

GMO Coin 外国為替FX 公開 API (認証不要) から過去 24-48 時間 + 当日 (現在時刻まで) の
M5 OHLCV を取得し、`data/curated/ds-1-forward.json` に追記マージする。

- 既存 ds-1-forward.json の最新バー timestamp 以降のみ追加 (重複防止)
- 4 通貨 (USD_JPY, EUR_JPY, GBP_JPY, AUD_JPY) 凍結設計
- rate limit: 100ms sleep (既存 scripts/data/fetch_ds1_ohlcv.py と同じ)
- 公開 API のみ使用、API キー不要 (CLAUDE.md「APIキー・.env.localは読まない」に抵触しない)

【cycle failure (2026-08-26 朝 07:00 JST 確認) 修正】
- 原因: live-ticker-poll.yml は bid/ask 気配値のみ取得しており、M5 OHLCV は
  data/curated/ds-1-forward.json に反映されていなかった
- 結果: sysfx012-fx-forward-cycle.yml の latest_bar_by_pair が 2026-08-24 16:25 JST で
  止まり、フォワードテスト集計が古い
- 修正: 本スクリプトを毎時実行する workflow update-ds1-forward.yml を追加
- 想定: 毎時 5 分に GMO API から過去 2 日分取得 → 既存 JSON に追記 → cycle 実行時に最新反映

Usage:
    PYTHONPATH=src python3 scripts/live_monitor/fetch_m5_ohlcv.py --all-pairs
    PYTHONPATH=src python3 scripts/live_monitor/fetch_m5_ohlcv.py --pair USD_JPY --lookback-days 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from minmax_fx_dt.data.gmo_fx_client import GMOClient  # noqa: E402

# SYS-FX012 凍結設計と同一の 4 通貨
TARGET_PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
INTERVAL = "5min"
RATE_LIMIT_SLEEP = 0.1
FORWARD_JSON = ROOT / "data" / "curated" / "ds-1-forward.json"
JST = timezone(timedelta(hours=9))


def klines_to_records(klines: list[dict]) -> list[dict]:
    """klines API レスポンスを ds-1-forward.json 互換の record list に変換.

    戻り値の timestamp は ISO 8601 形式の JST タイムゾーン付き文字列。
    """
    records = []
    for k in klines:
        ts = pd.to_datetime(int(k["openTime"]), unit="ms", utc=True).tz_convert(JST)
        records.append({
            "timestamp": ts.isoformat(),
            "open": float(k["open"]),
            "high": float(k["high"]),
            "low": float(k["low"]),
            "close": float(k["close"]),
        })
    return records


def fetch_one_day(client: GMOClient, symbol: str, date: pd.Timestamp) -> list[dict]:
    """1 日分の klines を取得."""
    date_str = date.strftime("%Y%m%d")
    try:
        return client.get_klines(symbol, INTERVAL, date_str)
    except Exception as e:
        print(f"  [NG] {symbol} {date_str}: {e}")
        return []


def load_existing() -> dict:
    """既存 ds-1-forward.json を読み込み (存在しなければ初期化)."""
    if FORWARD_JSON.exists():
        with FORWARD_JSON.open(encoding="utf-8") as f:
            return json.load(f)
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(JST).isoformat(),
        "interval": INTERVAL,
        "pairs": {},
    }


def merge_pair(existing_pair: dict, new_records: list[dict]) -> tuple[dict, int]:
    """1 通貨分の records を既存 pair にマージ. 戻り値: (新 pair dict, 追加件数)."""
    if not new_records:
        return existing_pair, 0
    existing_ts = {r["timestamp"] for r in existing_pair.get("data", [])}
    added = [r for r in new_records if r["timestamp"] not in existing_ts]
    if not added:
        return existing_pair, 0
    merged = existing_pair.get("data", []) + added
    merged.sort(key=lambda r: r["timestamp"])
    new_pair = dict(existing_pair)
    new_pair["data"] = merged
    new_pair["n_bars"] = len(merged)
    if merged:
        new_pair["start"] = merged[0]["timestamp"]
        new_pair["end"] = merged[-1]["timestamp"]
    new_pair["columns"] = ["open", "high", "low", "close"]
    new_pair["source"] = (
        "GMO Coin 外国為替FX ライブ取得 (cutoff 2026-08-15 06:00 JST 以降、"
        "scripts/live_monitor/fetch_m5_ohlcv.py)"
    )
    return new_pair, len(added)


def fetch_and_merge(pairs: list[str], lookback_days: int = 2) -> tuple[dict, int]:
    """pair ごとに fetch → 既存 JSON にマージ → 返す."""
    client = GMOClient("", "")  # 公開エンドポイントのみ
    out = load_existing()
    today_jst = datetime.now(JST).date()
    days = [today_jst - timedelta(days=i) for i in range(lookback_days)]
    days.reverse()  # 古い→新しい順 (重複排除のため)

    total_added = 0
    for symbol in pairs:
        print(f"\n[{symbol}] 取得開始 (lookback={lookback_days} 日: {days[0]} 〜 {days[-1]})")
        all_records = []
        for d in days:
            klines = fetch_one_day(client, symbol, pd.Timestamp(d))
            all_records.extend(klines_to_records(klines))
            time.sleep(RATE_LIMIT_SLEEP)
        # 重複排除 + ソート
        seen: set[str] = set()
        uniq = []
        for r in all_records:
            if r["timestamp"] not in seen:
                seen.add(r["timestamp"])
                uniq.append(r)
        uniq.sort(key=lambda r: r["timestamp"])
        existing_pair = out.get("pairs", {}).get(symbol, {"data": []})
        new_pair, added = merge_pair(existing_pair, uniq)
        out.setdefault("pairs", {})[symbol] = new_pair
        total_added += added
        n_bars = new_pair.get("n_bars", 0)
        end = new_pair.get("end", "(empty)")
        print(f"  [{symbol}] 取得 {len(uniq)} 件 / 既存に追加 {added} 件 / 累計 n_bars={n_bars} / end={end}")
    out["generated_at"] = datetime.now(JST).isoformat()
    return out, total_added


def main() -> int:
    parser = argparse.ArgumentParser(description="SYS-FX012 forward M5 OHLCV live update")
    parser.add_argument("--all-pairs", action="store_true", help="4 通貨全て (default)")
    parser.add_argument("--pair", help="通貨ペア (例: USD_JPY)")
    parser.add_argument(
        "--lookback-days", type=int, default=2,
        help="取得日数 (default: 2 = 昨日 + 今日、土日も含む)"
    )
    args = parser.parse_args()

    if args.all_pairs or (not args.all_pairs and not args.pair):
        targets = TARGET_PAIRS
    else:
        targets = [args.pair]

    out, added = fetch_and_merge(targets, args.lookback_days)
    FORWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    FORWARD_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    size = FORWARD_JSON.stat().st_size
    print(f"\n[OK] {FORWARD_JSON}")
    print(f"     size: {size / 1024:.1f} KB  ({size:,} bytes)")
    print(f"     追加: {added} bars (4 通貨合計)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
