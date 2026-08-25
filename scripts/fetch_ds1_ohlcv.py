"""data/raw/ds-1/ 配下の全 CSV を読み込み、data/curated/ds-1.json に集約.

(.aggregate_ds1.py を 2026-08-26 に fetch_ds1_ohlcv.py として関数化、ROOT ハードコード除去、allow_shrink チェック追加)

- ソース優先順位: dukascopy > gmo (期間重複時)
- ファイル名規則:
  ohlcv_<PAIR>_<interval>_<start>_<end>.csv                  (旧 GMO 形式)
  ohlcv_<PAIR>_<interval>_<start>_<end>_dukascopy.csv       (新規 dukascopy)
  ohlcv_<PAIR>_<interval>_<start>_<end>_gmo.csv              (新規 GMO)

cycle failure (2026-08-25 10:00 JST run 32795838404) の修正:
- 元の .aggregate_ds1.py は ROOT を Windows ハードコードしていて Actions (Linux) で動かない
- さらに workflow (sysfx012-fx-forward-cycle.yml) は `from fetch_ds1_ohlcv import aggregate_to_json` を期待
- このファイルで `aggregate_to_json(raw_dir, out_json, allow_shrink=False)` を提供して workflow と整合させる
"""
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ソース優先順位 (小さいほど優先)
SOURCE_RANK = {"dukascopy": 0, "gmo": 1, "unknown": 2}


def parse_csv_meta(path: Path) -> dict:
    """ファイル名から symbol / start / end / source を抽出."""
    name = path.stem
    parts = name.split("_")
    # ohlcv_USD_JPY_5min_20200101_20240630_dukascopy
    # 0=ohlcv, 1=USD, 2=JPY, 3=interval, 4=start, 5=end, 6=source(optional)
    if len(parts) >= 7 and parts[6] in ("dukascopy", "gmo"):
        source = parts[6]
    else:
        source = "unknown"  # 旧 GMO 形式
    symbol = f"{parts[1]}_{parts[2]}"
    return {
        "path": path,
        "symbol": symbol,
        "start": parts[4],
        "end": parts[5],
        "source": source,
    }


def aggregate_to_json(
    raw_dir: Path = Path("data/raw/ds-1"),
    out_json: Path = Path("data/curated/ds-1.json"),
    allow_shrink: bool = False,
) -> int:
    """data/raw/ds-1/ 配下の CSV を集約して data/curated/ds-1.json に出力.

    Parameters
    ----------
    raw_dir : Path
        ohlcv_<PAIR>_<interval>_<start>_<end>[_<source>].csv 群のディレクトリ
    out_json : Path
        出力先 JSON パス
    allow_shrink : bool
        False (デフォルト) の場合、既存 ds-1.json よりも n_bars が減少するなら sys.exit(1)。
        True なら無条件に書き出し。

    Returns
    -------
    int
        出力ファイルサイズ (bytes)
    """
    if not raw_dir.exists():
        print(f"[NG] {raw_dir} が存在しない")
        sys.exit(1)

    # 全ファイル解析
    all_files = sorted(raw_dir.glob("ohlcv_*.csv"))
    print(f"[INFO] raw_dir={raw_dir}  ファイル: {len(all_files)} 個")
    metas = [parse_csv_meta(f) for f in all_files]
    for m in metas:
        print(f"  {m['path'].name}  symbol={m['symbol']}  source={m['source']}  period={m['start']}-{m['end']}")

    # 通貨ごとにグループ化
    by_symbol: dict[str, list[dict]] = {}
    for m in metas:
        by_symbol.setdefault(m["symbol"], []).append(m)

    # 通貨ごとに集約
    all_pairs: dict[str, pd.DataFrame] = {}
    for symbol, ms in by_symbol.items():
        # ソース優先順位でソート (dukascopy 優先)
        ms.sort(key=lambda m: SOURCE_RANK[m["source"]])
        print(f"\n[{symbol}] ソース優先順:")
        for m in ms:
            print(f"  {m['source']:10} {m['start']}-{m['end']}  ({m['path'].name})")

        # 読み込み + 重複処理 (重複時はソース優先順位が高い方を残す)
        df_combined: Optional[pd.DataFrame] = None
        for m in ms:
            df = pd.read_csv(m["path"], parse_dates=["timestamp"])
            df = df.set_index("timestamp").sort_index()
            # タイムゾーンの正規化 (tz-aware JST に統一)
            if df.index.tz is None:
                df.index = df.index.tz_localize("Asia/Tokyo")
            else:
                df.index = df.index.tz_convert("Asia/Tokyo")
            if df_combined is None:
                df_combined = df
            else:
                # 高優先度 (後で concat) が上書きされるよう keep="last"
                df_combined = pd.concat([df_combined, df]).sort_index()
                df_combined = df_combined[~df_combined.index.duplicated(keep="last")]

        if df_combined is not None:
            # タイムゾーンを外して ISO 文字列化 (ds-1.json との互換性)
            df_combined.index = df_combined.index.tz_localize(None)
            all_pairs[symbol] = df_combined
            print(f"  最終: {len(df_combined)} bars, {df_combined.index[0]} - {df_combined.index[-1]}")

    # 集約後の n_bars 集計
    new_n_bars = sum(len(df) for df in all_pairs.values())
    print(f"\n[INFO] 新規 n_bars 合計: {new_n_bars}")

    # 既存 ds-1.json との shrinkage チェック (allow_shrink=False)
    if not allow_shrink and out_json.exists():
        try:
            existing = json.loads(out_json.read_text(encoding="utf-8"))
            existing_n_bars = sum(p.get("n_bars", 0) for p in existing.get("pairs", {}).values())
            existing_existing = existing.get("generated_at", "unknown")
            print(f"[INFO] 既存 n_bars 合計: {existing_n_bars}  (generated_at: {existing_existing})")
            if new_n_bars < existing_n_bars:
                print(
                    f"[NG] shrinkage detected: existing={existing_n_bars} > new={new_n_bars}. "
                    f"差分 {existing_n_bars - new_n_bars} bars 消失を検出。"
                    f"--allow-shrink で明示的に許可する場合のみ上書きされます。"
                )
                sys.exit(1)
            print(f"[OK] shrinkage check passed: existing={existing_n_bars} <= new={new_n_bars}")
        except Exception as e:
            print(f"[WARN] 既存 {out_json} のパースに失敗 ({e})、shrinkage チェックをスキップ")
    elif allow_shrink:
        print("[WARN] allow_shrink=True: shrinkage チェックをスキップ")
    else:
        print(f"[INFO] 既存 {out_json} 不在、新規作成")

    # 集約 JSON 構築
    out = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "interval": "5min",
        "pairs": {},
    }
    for symbol, df in all_pairs.items():
        out["pairs"][symbol] = {
            "n_bars": int(len(df)),
            "start": df.index[0].isoformat(),
            "end": df.index[-1].isoformat(),
            "columns": list(df.columns),
            "source": "dukascopy + GMO Coin 外国為替FX (ソース優先: dukascopy > gmo)",
            "data": df.reset_index().to_dict(orient="records"),
        }
        print(f"  {symbol:<10} n_bars={len(df):>6} start={df.index[0].date()} end={df.index[-1].date()}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    size = out_json.stat().st_size
    print(f"\n[OK] {out_json}")
    print(f"     size: {size / 1024 / 1024:.1f} MB  ({size:,} bytes)")
    return size


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="data/raw/ds-1/*.csv → data/curated/ds-1.json")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ds-1"))
    parser.add_argument("--out-json", type=Path, default=Path("data/curated/ds-1.json"))
    parser.add_argument("--allow-shrink", action="store_true", default=False)
    args = parser.parse_args()
    aggregate_to_json(
        raw_dir=args.raw_dir,
        out_json=args.out_json,
        allow_shrink=args.allow_shrink,
    )
