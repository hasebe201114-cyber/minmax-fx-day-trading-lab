"""SYS-FX012 フォワードテスト用 M5 OHLCV のライブ取得.

GMO Coin 外国為替FX 公開 API (認証不要) から過去 24-48 時間 + 当日 (現在時刻まで) の
M5 OHLCV を取得し、`data/raw/ds-1-forward/*.csv` (git 管理・永続) に追記マージする。

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

【2026-08-27 追加修正 (OBS000009 不具合2)】
- 上記修正後も、本スクリプトの永続先は `data/curated/ds-1-forward.json` のみで、
  これは `.gitignore` 対象。GitHub Actions のランナーは実行ごとにまっさらな
  環境のため、「既存 JSON に追記マージ」する設計はワークフロー実行をまたいで
  一切永続せず、cutoff (2026-08-15 06:00 JST) 以降のデータが実質「直近
  lookback_days 日分のローリング窓」でしか蓄積されていなかった
  (2026-08-19 21:00 JSTの3件のN_BREAKOUTイベントを取りこぼしていたことを
  `scripts/analyze_post_breakout_trend_visual_check.py` の副産物で発見)。
- 修正: 永続先を `data/raw/ds-1-forward/ohlcv_{symbol}_5min_live.csv`
  (git 管理下、ds-1 と同じ CSV スキーマ) に変更。実行のたびに既存 CSV を
  読み込み・新規取得分とマージ・重複排除して書き戻す。`data/curated/
  ds-1-forward.json` は、CSV 全量から毎回再構築する派生物として維持する
  (同一実行内でこのファイルを直接読む既存コードとの後方互換のため)。

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
RAW_DIR = ROOT / "data" / "raw" / "ds-1-forward"
FORWARD_JSON = ROOT / "data" / "curated" / "ds-1-forward.json"
JST = timezone(timedelta(hours=9))


def klines_to_df(klines: list[dict]) -> pd.DataFrame:
    """klines API レスポンスを DataFrame (JST tz-aware index) に変換."""
    if not klines:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    records = []
    for k in klines:
        ts = pd.to_datetime(int(k["openTime"]), unit="ms", utc=True).tz_convert(JST)
        records.append({
            "timestamp": ts,
            "open": float(k["open"]),
            "high": float(k["high"]),
            "low": float(k["low"]),
            "close": float(k["close"]),
        })
    return pd.DataFrame(records).set_index("timestamp").sort_index()


def fetch_one_day(client: GMOClient, symbol: str, date: pd.Timestamp) -> list[dict]:
    """1 日分の klines を取得."""
    date_str = date.strftime("%Y%m%d")
    try:
        return client.get_klines(symbol, INTERVAL, date_str)
    except Exception as e:
        print(f"  [NG] {symbol} {date_str}: {e}")
        return []


def raw_csv_path(symbol: str) -> Path:
    """本スクリプトが継続的に追記する、1通貨1本の永続CSVパス.

    `data/raw/ds-1-forward/` には過去に手動で取得した日付範囲付きファイル
    (`ohlcv_{symbol}_5min_{start}_{end}.csv`) も存在するが、本スクリプトが
    書くのは常にこの `_live.csv` サフィックスのファイル。
    `load_m5_forward()` 側は `ohlcv_{pair}_5min_*.csv` で両方を glob するため、
    どちらも透過的に結合される。
    """
    return RAW_DIR / f"ohlcv_{symbol}_5min_live.csv"


def load_existing_raw(symbol: str) -> pd.DataFrame:
    """既存の全 raw CSV (手動分含む) を結合して読み込む (存在しなければ空)."""
    files = sorted(RAW_DIR.glob(f"ohlcv_{symbol}_5min_*.csv"))
    if not files:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    frames = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["timestamp"])
        frames.append(df.set_index("timestamp"))
    combined = pd.concat(frames).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def merge_and_write_raw(symbol: str, new_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """新規取得分を `_live.csv` にマージして書き戻す. 戻り値: (全量df, 追加件数)."""
    existing_live = pd.DataFrame(columns=["open", "high", "low", "close"])
    path = raw_csv_path(symbol)
    if path.exists():
        existing_live = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")

    before_n = len(existing_live)
    merged_live = pd.concat([existing_live, new_df]).sort_index()
    merged_live = merged_live[~merged_live.index.duplicated(keep="last")]
    merged_live.index.name = "timestamp"  # 空DataFrameとのconcatでindex名が落ちるため明示
    added = len(merged_live) - before_n

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    merged_live.to_csv(path)

    # 手動分の過去ファイルも含めた全量 (curated JSON 再構築・カバレッジ表示用)
    all_files_df = load_existing_raw(symbol)
    return all_files_df, added


def fetch_and_persist(pairs: list[str], lookback_days: int = 2) -> tuple[dict, int]:
    """pair ごとに fetch → raw CSV にマージ → curated JSON を全量から再構築."""
    client = GMOClient("", "")  # 公開エンドポイントのみ
    today_jst = datetime.now(JST).date()
    days = [today_jst - timedelta(days=i) for i in range(lookback_days)]
    days.reverse()  # 古い→新しい順

    out = {
        "schema_version": "1.0",
        "generated_at": datetime.now(JST).isoformat(),
        "interval": INTERVAL,
        "pairs": {},
    }
    total_added = 0
    for symbol in pairs:
        print(f"\n[{symbol}] 取得開始 (lookback={lookback_days} 日: {days[0]} 〜 {days[-1]})")
        dfs = []
        for d in days:
            klines = fetch_one_day(client, symbol, pd.Timestamp(d))
            dfs.append(klines_to_df(klines))
            time.sleep(RATE_LIMIT_SLEEP)
        new_df = pd.concat(dfs).sort_index() if dfs else pd.DataFrame(columns=["open", "high", "low", "close"])
        new_df = new_df[~new_df.index.duplicated(keep="last")]

        all_df, added = merge_and_write_raw(symbol, new_df)
        total_added += added

        if len(all_df) > 0:
            out["pairs"][symbol] = {
                "n_bars": len(all_df),
                "start": all_df.index[0].isoformat(),
                "end": all_df.index[-1].isoformat(),
                "columns": list(all_df.columns),
                "source": (
                    "GMO Coin 外国為替FX ライブ取得 (cutoff 2026-08-15 06:00 JST 以降、"
                    "data/raw/ds-1-forward/*.csv から再構築、scripts/live_monitor/fetch_m5_ohlcv.py)"
                ),
                "data": all_df.reset_index().to_dict(orient="records"),
            }
            end = out["pairs"][symbol]["end"]
        else:
            end = "(empty)"
        print(f"  [{symbol}] 取得 {len(new_df)} 件 / raw CSV 新規追加 {added} 件 / "
              f"累計 n_bars={len(all_df)} / end={end}")
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

    out, added = fetch_and_persist(targets, args.lookback_days)
    FORWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    FORWARD_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    size = FORWARD_JSON.stat().st_size
    print(f"\n[OK] raw CSV 永続先: {RAW_DIR}")
    print(f"[OK] {FORWARD_JSON} (CSV全量からの再構築、参考用)")
    print(f"     size: {size / 1024:.1f} KB  ({size:,} bytes)")
    print(f"     raw CSV への新規追加: {added} bars (4 通貨合計)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
