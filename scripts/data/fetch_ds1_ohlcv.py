"""DS-1 OHLCV 取得スクリプト (GMO Coin 外国為替FX).

5 通貨 × 5 年 × M5 足 = 約 9,000 リクエスト。
Public API なので API キー不要だが、GMO 公式ドキュメントの rate limit に注意。

Usage:
    python scripts/data/fetch_ds1_ohlcv.py --pair USD_JPY --start 2024-01-01 --end 2024-12-31
    python scripts/data/fetch_ds1_ohlcv.py --all-pairs --start 2020-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# プロジェクトルートを sys.path に追加
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd

from minmax_fx_dt.data import GMOClient, GMOClientError


# OBS0002 案 A の通貨ペア
DEFAULT_PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
INTERVAL = "5min"
RATE_LIMIT_SLEEP = 0.1  # 100ms


def fetch_one_day(
    client: GMOClient,
    symbol: str,
    date: pd.Timestamp,
    interval: str = INTERVAL,
) -> list[dict]:
    """1日分の klines を取得."""
    date_str = date.strftime("%Y%m%d")
    try:
        return client.get_klines(symbol, interval, date_str)
    except GMOClientError as e:
        print(f"  [NG] {symbol} {date_str}: {e}")
        return []


def klines_to_df(klines: list[dict]) -> pd.DataFrame:
    """klines API レスポンスを DataFrame に変換."""
    if not klines:
        return pd.DataFrame()
    records = []
    for k in klines:
        records.append({
            "timestamp": pd.to_datetime(int(k["openTime"]), unit="ms", utc=True).tz_convert("Asia/Tokyo"),
            "open": float(k["open"]),
            "high": float(k["high"]),
            "low": float(k["low"]),
            "close": float(k["close"]),
        })
    df = pd.DataFrame(records)
    df = df.set_index("timestamp").sort_index()
    return df


def fetch_pair(
    client: GMOClient,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    out_dir: Path,
    interval: str = INTERVAL,
) -> dict:
    """1 通貨ペアの全期間取得."""
    all_days = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")
    print(f"[{symbol}] {len(all_days)} 日分を {interval} で取得開始 (推定 {len(all_days) * RATE_LIMIT_SLEEP:.0f} 秒)")

    dfs: list[pd.DataFrame] = []
    success_days = 0
    fail_days = 0
    no_data_days = 0

    for i, day in enumerate(all_days):
        klines = fetch_one_day(client, symbol, day, interval)
        if not klines:
            no_data_days += 1
        else:
            df = klines_to_df(klines)
            if not df.empty:
                dfs.append(df)
                success_days += 1
            else:
                fail_days += 1
        # 進捗表示 (10 日毎)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(all_days)}] {symbol} {day.date()} (取得 {success_days} 日, 失敗 {fail_days} 日, 休場 {no_data_days} 日)")
        time.sleep(RATE_LIMIT_SLEEP)

    if not dfs:
        print(f"  [NG] {symbol}: データ取得できず")
        return {
            "symbol": symbol,
            "n_days_success": 0,
            "n_days_fail": fail_days,
            "n_days_no_data": no_data_days,
            "n_bars": 0,
            "file": None,
        }

    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    # ファイル保存
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ohlcv_{symbol}_{interval}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    combined.to_csv(out_file)
    print(f"  [OK] {symbol}: {len(combined)} 本を {out_file.name} に保存")

    return {
        "symbol": symbol,
        "n_days_success": success_days,
        "n_days_fail": fail_days,
        "n_days_no_data": no_data_days,
        "n_bars": len(combined),
        "start": combined.index[0].isoformat(),
        "end": combined.index[-1].isoformat(),
        "file": str(out_file.relative_to(ROOT)),
    }


def aggregate_to_json(raw_dir: Path, out_json: Path, *, allow_shrink: bool = False) -> dict:
    """CSV 群を 1 つの JSON に集約.

    OBS000007 (2026-08-15 独立監査 B4) で発覚したバグの修正:
    旧実装は同一 symbol の複数 CSV (例: 異なる日付範囲で複数回取得した場合) を
    `pairs_data[symbol] = df` で単純に上書きしており、ソート順で「最後のファイルが勝つ」
    ため、範囲の狭い (壊れた/古い) CSV が全期間 CSV を silently 上書きしうる。
    これにより過去に ds-1.json の USD_JPY が 74,232 本 → 852 本に縮小する事故が発生した。
    修正: 同一 symbol の全 CSV を結合し、タイムスタンプで重複排除する。
    さらに、既存 ds-1.json と比べて件数が減る場合は allow_shrink=True でない限り
    エラーにして書き込みを拒否する (縮小事故の再発防止)。
    """
    files = sorted(raw_dir.glob("ohlcv_*.csv"))
    frames_by_symbol: dict[str, list[pd.DataFrame]] = {}
    for f in files:
        df = pd.read_csv(f, index_col="timestamp", parse_dates=True)
        # ファイル名から symbol 抽出 (ohlcv_USD_JPY_...)
        name = f.stem
        parts = name.split("_")
        symbol = "_".join(parts[1:3])  # USD_JPY
        frames_by_symbol.setdefault(symbol, []).append(df)

    pairs_data: dict[str, pd.DataFrame] = {}
    for symbol, frames in frames_by_symbol.items():
        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        pairs_data[symbol] = combined

    # 縮小ガード: 既存 ds-1.json があれば、通貨ペアごとの本数が減っていないか確認
    if out_json.exists() and not allow_shrink:
        try:
            existing = json.loads(out_json.read_text(encoding="utf-8"))
            existing_pairs = existing.get("pairs", {})
        except (json.JSONDecodeError, OSError):
            existing_pairs = {}
        shrink_errors = []
        for symbol, df in pairs_data.items():
            prev_n = existing_pairs.get(symbol, {}).get("n_bars")
            if prev_n is not None and len(df) < prev_n:
                shrink_errors.append(f"{symbol}: {prev_n} 本 → {len(df)} 本に縮小")
        if shrink_errors:
            raise RuntimeError(
                "ds-1.json の書き込みを中止しました (件数が縮小するため、--allow-shrink で明示的に許可しない限り拒否):\n  "
                + "\n  ".join(shrink_errors)
            )

    out: dict = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "interval": INTERVAL,
        "pairs": {},
    }
    for symbol, df in pairs_data.items():
        out["pairs"][symbol] = {
            "n_bars": len(df),
            "start": df.index[0].isoformat(),
            "end": df.index[-1].isoformat(),
            "columns": list(df.columns),
            "data": df.reset_index().to_dict(orient="records"),
        }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return {"n_pairs": len(pairs_data), "total_bars": sum(len(df) for df in pairs_data.values()), "json": str(out_json)}


def main() -> int:
    parser = argparse.ArgumentParser(description="DS-1 OHLCV 取得 (GMO FX API)")
    parser.add_argument("--pair", help="通貨ペア (例: USD_JPY)")
    parser.add_argument("--all-pairs", action="store_true", help="OBS0002 案 A の 5 通貨全て")
    parser.add_argument("--start", default="2024-01-01", help="開始日 YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="終了日 YYYY-MM-DD")
    parser.add_argument("--interval", default=INTERVAL, help=f"足種 (1min/5min/15min/30min/1hour, default {INTERVAL})")
    parser.add_argument("--out-dir", default="data/raw/ds-1", help="生データ出力ディレクトリ")
    parser.add_argument("--out-json", default="data/curated/ds-1.json", help="集約 JSON 出力先")
    parser.add_argument(
        "--allow-shrink",
        action="store_true",
        help="集約結果が既存 ds-1.json より件数が少なくても書き込みを許可する (通常は不要、事故防止のためデフォルト拒否)",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    out_dir = ROOT / args.out_dir
    out_json = ROOT / args.out_json

    if args.all_pairs:
        pairs = DEFAULT_PAIRS
    elif args.pair:
        pairs = [args.pair]
    else:
        print("エラー: --pair または --all-pairs を指定してください")
        return 1

    client = GMOClient("", "")  # Public API は API キー不要

    # 通貨ペア毎に取得
    results = []
    for symbol in pairs:
        result = fetch_pair(client, symbol, start, end, out_dir, args.interval)
        results.append(result)

    # JSON 集約
    print("\n=== JSON 集約 ===")
    try:
        agg_result = aggregate_to_json(out_dir, out_json, allow_shrink=args.allow_shrink)
    except RuntimeError as e:
        print(f"  [NG] {e}")
        return 1
    print(f"  {agg_result['n_pairs']} 通貨, 合計 {agg_result['total_bars']} 本")
    print(f"  保存先: {agg_result['json']}")

    # 結果サマリ
    print("\n=== 結果サマリ ===")
    total_bars = 0
    for r in results:
        total_bars += r["n_bars"]
        print(f"  {r['symbol']:<10} 取得 {r['n_days_success']:>4} 日, 失敗 {r['n_days_fail']:>3} 日, 休場 {r['n_days_no_data']:>3} 日, {r['n_bars']:>6} 本")
    print(f"  合計: {total_bars:,} 本")

    # MANIFEST.json 更新
    manifest_path = ROOT / "data" / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": "1.0", "datasets": []}

    # 既存の DS-1 を削除して再追加
    manifest["datasets"] = [d for d in manifest.get("datasets", []) if d.get("id") != "DS-1"]
    manifest["datasets"].append({
        "id": "DS-1",
        "name": "OHLCV",
        "source": "GMO Coin 外国為替FX Public API (/v1/klines)",
        "fetched_at": datetime.now().isoformat(),
        "interval": args.interval,
        "n_pairs": len(results),
        "total_bars": total_bars,
        "files": [r["file"] for r in results if r.get("file")],
    })
    manifest["generated_at"] = datetime.now().isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMANIFEST.json 更新: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
