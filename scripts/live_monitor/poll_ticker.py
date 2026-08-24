"""EXP-FX000016 Stage 1・コンポーネントA: ライブ気配値の記録.

GMO公開Ticker API(`GMOClient.get_ticker()`、認証不要)を1回呼び出し、
SYS-FX012対象4通貨のbid/ask/実測スプレッドを記録する。

**実発注は一切行わない**。認証情報も使わない(GMOClient("", "")で公開
エンドポイントのみ呼び出す、CLAUDE.md「APIキー・.env.localは読まない」に抵触しない)。

想定される起動方法: Routine(毎時)から本スクリプトを直接実行。
1回の呼び出しはAPI 1コールのみで軽量。

出力: data/raw/live-ticker/YYYY-MM.csv (月次ローテーション、追記)
列: polled_at, pair, bid, ask, spread_pips, api_timestamp, market_status

Usage:
    PYTHONPATH=src python3 scripts/live_monitor/poll_ticker.py
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from minmax_fx_dt.data.gmo_fx_client import GMOClient  # noqa: E402

# SYS-FX012凍結設計と同一の4通貨
TARGET_PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]

OUT_DIR = ROOT / "data" / "raw" / "live-ticker"


def pip_size(pair: str) -> float:
    """JPYクォート=0.01(2桁目)、それ以外=0.0001(4桁目)。プロジェクト共通の定義に合わせる。"""
    return 0.01 if pair.endswith("JPY") else 0.0001


def poll_once() -> list[dict]:
    client = GMOClient("", "")  # 認証不要の公開エンドポイントのみ使用
    resp = client.get_ticker()
    by_pair = {row["symbol"]: row for row in resp.get("data", [])}

    polled_at = datetime.now(timezone.utc).isoformat()
    records = []
    for pair in TARGET_PAIRS:
        row = by_pair.get(pair)
        if row is None:
            continue
        bid, ask = float(row["bid"]), float(row["ask"])
        spread_pips = (ask - bid) / pip_size(pair)
        records.append({
            "polled_at": polled_at,
            "pair": pair,
            "bid": bid,
            "ask": ask,
            "spread_pips": round(spread_pips, 4),
            "api_timestamp": row.get("timestamp"),
            "market_status": row.get("status"),
        })
    return records


def append_records(records: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    month_tag = datetime.now(timezone.utc).strftime("%Y-%m")
    out_path = OUT_DIR / f"{month_tag}.csv"
    is_new = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "polled_at", "pair", "bid", "ask", "spread_pips", "api_timestamp", "market_status"])
        if is_new:
            writer.writeheader()
        writer.writerows(records)
    return out_path


def main() -> int:
    records = poll_once()
    if not records:
        print("[WARN] 取得できたレコードが0件でした(API応答に対象通貨が含まれない)")
        return 1
    out_path = append_records(records)
    for r in records:
        print(f"{r['pair']}: bid={r['bid']} ask={r['ask']} spread={r['spread_pips']}pips "
              f"status={r['market_status']}")
    print(f"[出力]: {out_path} に{len(records)}件追記")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
