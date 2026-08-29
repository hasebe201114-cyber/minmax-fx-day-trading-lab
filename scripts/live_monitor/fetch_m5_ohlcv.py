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

【フォワードデータ消失バグ (2026-08-29 発見) 修正】
- 原因: ds-1-forward.json は .gitignore 対象で、GitHub Actions runner は毎回
  使い捨てのため、毎時 workflow が「追記マージ」した結果が次回実行に一切残らない
  (毎回 lookback 2 日分だけの JSON が作られ、runner 終了とともに消える)
- 結果: 週次 cycle が読めるフォワード区間は「実行直前の 2 日分」のみ。
  cutoff (2026-08-15 06:00 JST) 〜 実行 2 日前までが恒久的な空白となり、
  ledger は n_events_raw=0 (真値は 6) という誤った値を報告し続けていた
- 修正: バーの正本を git 管理の追記型 CSV `data/raw/ds-1-forward/*.csv` に移し、
  ds-1-forward.json はそこから再生成される派生物として扱う (既存 data/raw/ds-1/
  と同じ「raw CSV が正本・curated JSON は派生」構造に揃えた)。JSON が無い runner
  でも CSV から全期間を復元できる

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
# バーの正本 (git 管理・追記型)。ds-1-forward.json はここからの派生物。
FORWARD_CSV_DIR = ROOT / "data" / "raw" / "ds-1-forward"
CSV_COLUMNS = ["timestamp", "open", "high", "low", "close"]
JST = timezone(timedelta(hours=9))

FORWARD_SOURCE_NOTE = (
    "GMO Coin 外国為替FX ライブ取得 (cutoff 2026-08-15 06:00 JST 以降、"
    "scripts/live_monitor/fetch_m5_ohlcv.py)"
)


def csv_path(pair: str) -> Path:
    """通貨ペアごとの追記型 CSV パス."""
    return FORWARD_CSV_DIR / f"ohlcv_{pair}_5min_forward.csv"


def load_forward_csv(pair: str) -> list[dict]:
    """git 管理の追記型 CSV から 1 通貨分の records を読む (無ければ空)."""
    path = csv_path(pair)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    # JSON 側 (ds-1-forward.json) の timestamp は isoformat 文字列。
    # CSV は "2026-08-15 06:00:00+09:00" 形式で保存されるため、突き合わせ前に揃える。
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(JST).map(lambda t: t.isoformat())
    )
    return [
        {
            "timestamp": str(row.timestamp),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        for row in df.itertuples(index=False)
    ]


def write_forward_csv(pair: str, records: list[dict]) -> None:
    """1 通貨分の records を CSV へ書き出す (timestamp 昇順・重複排除済み前提).

    書式は `scripts/data/fetch_ds1_ohlcv.py` の出力と揃える (timestamp を
    Asia/Tokyo の tz 付き index にして出力)。同ディレクトリの日付範囲別 CSV と
    混在しても `aggregate_to_json()` が symbol 単位で結合・重複排除できる。
    """
    if not records:
        return
    FORWARD_CSV_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=CSV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Tokyo")
    df.set_index("timestamp").sort_index().to_csv(csv_path(pair))


def pair_dict_from_records(records: list[dict]) -> dict:
    """records から ds-1-forward.json の pair エントリを組み立てる."""
    return {
        "data": records,
        "n_bars": len(records),
        "start": records[0]["timestamp"] if records else None,
        "end": records[-1]["timestamp"] if records else None,
        "columns": ["open", "high", "low", "close"],
        "source": FORWARD_SOURCE_NOTE,
    }


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


def load_existing(pairs: list[str] | None = None) -> dict:
    """既存バーを読み込む.

    正本は git 管理の `data/raw/ds-1-forward/*.csv`。使い捨て runner では
    ds-1-forward.json (.gitignore 対象) が存在しないため、CSV から全期間を
    復元する。JSON が手元にある場合はそこにも入っているバーを取り込み、
    両者の和集合を返す (どちらか一方しか無い環境でも欠落しない)。
    """
    out = {
        "schema_version": "1.0",
        "generated_at": datetime.now(JST).isoformat(),
        "interval": INTERVAL,
        "pairs": {},
    }
    if FORWARD_JSON.exists():
        with FORWARD_JSON.open(encoding="utf-8") as f:
            out = json.load(f)
            out.setdefault("pairs", {})

    for pair in pairs if pairs is not None else TARGET_PAIRS:
        csv_records = load_forward_csv(pair)
        if not csv_records:
            continue
        existing_pair = out["pairs"].get(pair, {"data": []})
        merged_pair, _ = merge_pair(existing_pair, csv_records)
        out["pairs"][pair] = merged_pair
    return out


def merge_pair(existing_pair: dict, new_records: list[dict]) -> tuple[dict, int]:
    """1 通貨分の records を既存 pair にマージ. 戻り値: (新 pair dict, 追加件数)."""
    existing_records = existing_pair.get("data", [])
    existing_ts = {r["timestamp"] for r in existing_records}
    added = [r for r in new_records if r["timestamp"] not in existing_ts]
    merged = existing_records + added
    merged.sort(key=lambda r: r["timestamp"])
    return pair_dict_from_records(merged), len(added)


def fetch_and_merge(pairs: list[str], lookback_days: int = 2) -> tuple[dict, int]:
    """pair ごとに fetch → 既存バー (正本 CSV) にマージ → 返す."""
    client = GMOClient("", "")  # 公開エンドポイントのみ
    out = load_existing(pairs)
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

    # 正本 (git 管理・追記型 CSV) を先に更新する。
    # ds-1-forward.json は .gitignore 対象で使い捨て runner とともに消えるため、
    # ここを書き損ねると次回実行でバーが失われる (2026-08-29 に修正した消失バグ)。
    for symbol in targets:
        records = out.get("pairs", {}).get(symbol, {}).get("data", [])
        write_forward_csv(symbol, records)
        print(f"  [CSV] {csv_path(symbol).relative_to(ROOT)}: {len(records)} bars")

    FORWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    FORWARD_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    size = FORWARD_JSON.stat().st_size
    print(f"\n[OK] {FORWARD_JSON}")
    print(f"     size: {size / 1024:.1f} KB  ({size:,} bytes)")
    print(f"     追加: {added} bars (4 通貨合計)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
