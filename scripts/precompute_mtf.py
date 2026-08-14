"""MTF 集約を事前計算してキャッシュ (parquet 優先, pickle フォールバック).

差し戻し 3 対応: train/val/test 各セルで同じ MTF 集約を 3 回繰り返す無駄を排除.
5 通貨 × 4 TF (M5, M15, H4, D1) を一度だけ集約して保存し、各セルは parquet/pickle
読み込み + 期間フィルタ + バックテストのみにする.

Usage:
  python scripts/precompute_mtf.py
  python scripts/precompute_mtf.py --symbols USD_JPY EUR_JPY
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd

CACHE_DIR = ROOT / "data" / "curated" / "mtf_cache"
DS1_PATH = ROOT / "data" / "curated" / "ds-1.json"

# 仕様で固定された通貨 (OBS000002 案 A)
DEFAULT_SYMBOLS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]


def _try_parquet():
    """pyarrow が利用可能なら True."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def aggregate_to_mtf(m5_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """M5 → M5 / M15 / H4 / D1 集約.

    戻り値: {"M5": ..., "M15": ..., "H4": ..., "D1": ...}
    """
    m15 = pd.DataFrame({
        "open": m5_df["open"].resample("15min").first(),
        "high": m5_df["high"].resample("15min").max(),
        "low": m5_df["low"].resample("15min").min(),
        "close": m5_df["close"].resample("15min").last(),
    }).dropna()
    h4 = pd.DataFrame({
        "open": m5_df["open"].resample("4h").first(),
        "high": m5_df["high"].resample("4h").max(),
        "low": m5_df["low"].resample("4h").min(),
        "close": m5_df["close"].resample("4h").last(),
    }).dropna()
    d1 = pd.DataFrame({
        "open": h4["open"].resample("D").first(),
        "high": h4["high"].resample("D").max(),
        "low": h4["low"].resample("D").min(),
        "close": h4["close"].resample("D").last(),
    }).dropna()
    return {"M5": m5_df, "M15": m15, "H4": h4, "D1": d1}


def load_ohlcv_from_ds1(ds1: dict, symbol: str) -> pd.DataFrame:
    """DS-1 JSON dict から指定通貨の OHLCV を読み込み."""
    if symbol not in ds1["pairs"]:
        raise ValueError(f"DS-1 に {symbol} がありません: {list(ds1['pairs'].keys())}")
    records = ds1["pairs"][symbol]["data"]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def save_cache(cache: dict[str, pd.DataFrame], path: Path, use_parquet: bool) -> None:
    """キャッシュ保存 (parquet 優先, pickle フォールバック)."""
    if use_parquet:
        # 4 TF を別ファイルで保存
        for tf, df in cache.items():
            df.to_parquet(path.with_name(f"{path.stem}_{tf}.parquet"))
    else:
        # 単一 pickle に 4 TF まとめて保存
        with path.with_suffix(".pkl").open("wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> int:
    parser = argparse.ArgumentParser(description="MTF 集約を事前計算してキャッシュ")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS,
                        help="対象通貨 (デフォルト: 5 通貨全て)")
    parser.add_argument("--force", action="store_true", help="キャッシュがあっても再生成")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    use_parquet = _try_parquet()
    suffix = ".parquet (per TF)" if use_parquet else ".pkl (4 TF まとめ)"
    print(f"=== MTF 事前計算 ===")
    print(f"キャッシュ先: {CACHE_DIR}")
    print(f"保存形式: {suffix}")
    print(f"対象通貨: {args.symbols}")
    print()

    t0 = time.time()
    print(f"DS-1 読み込み中: {DS1_PATH} (444 MB, 30-60 秒)")
    with DS1_PATH.open(encoding="utf-8") as f:
        ds1 = json.load(f)
    print(f"  → {time.time() - t0:.1f}秒")
    print()

    total = len(args.symbols)
    for i, symbol in enumerate(args.symbols, 1):
        cache_path = CACHE_DIR / f"{symbol}.pkl"
        parquet_marker = CACHE_DIR / f"{symbol}_D1.parquet"
        if not args.force and (parquet_marker.exists() if use_parquet else cache_path.exists()):
            print(f"[{i}/{total}] {symbol}: キャッシュ既存、スキップ")
            continue

        t0 = time.time()
        print(f"[{i}/{total}] {symbol}: データ読み込み中...")
        m5 = load_ohlcv_from_ds1(ds1, symbol)
        print(f"  M5 bars: {len(m5)} ({m5.index[0].date()} - {m5.index[-1].date()})")

        t1 = time.time()
        print(f"  MTF 集約中...")
        cache = aggregate_to_mtf(m5)
        for tf, df in cache.items():
            print(f"    {tf}: {len(df)} bars")

        t2 = time.time()
        save_cache(cache, cache_path, use_parquet)
        size_mb = sum(
            (cache_path.with_name(f"{cache_path.stem}_{tf}.parquet").stat().st_size
             if use_parquet
             else cache_path.with_suffix(".pkl").stat().st_size)
            for tf in cache
        ) / 1_000_000
        print(f"  → 集約 {t2 - t1:.1f}秒, 保存 {(time.time() - t2):.1f}秒, "
              f"合計 {time.time() - t0:.1f}秒 ({size_mb:.1f} MB)")

    print()
    print(f"=== 完了: 総時間 {time.time() - t0:.1f}秒 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
