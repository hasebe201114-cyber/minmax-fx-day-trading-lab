"""SYS-FX012(EXP-FX000006) フォワードテストの週次サイクル一括実行.

週次Routine(毎週月曜09:00 JST)から呼ばれる単一エントリポイント。
1. `data/curated/ds-1-forward.json`の既存最終バーを見て新規取得範囲を決定
   (初回はcutoff=2026-08-15から)
2. `scripts/data/fetch_ds1_ohlcv.py`の関数を再利用してGMO公開APIから新規データを取得し、
   data/raw/ds-1-forward/・data/curated/ds-1-forward.json に追記(既存ds-1.jsonには一切触れない)
3. `run_forward_test_cycle.py`のrun_forward_cycle()でcutoff以降の全期間を再計算し、
   research/method-notes/sysfx012_forward_test_ledger.json を更新

Usage: PYTHONPATH=src:scripts python3 scripts/forward_test/weekly_cycle.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from data.fetch_ds1_ohlcv import DEFAULT_PAIRS, GMOClient, aggregate_to_json, fetch_pair  # noqa: E402
from forward_test.run_forward_test_cycle import CUTOFF, run_forward_cycle  # noqa: E402

FORWARD_PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]  # SYS-FX012凍結設計の4通貨のみ(EUR_USD等は不要)
RAW_DIR = ROOT / "data" / "raw" / "ds-1-forward"
CURATED_JSON = ROOT / "data" / "curated" / "ds-1-forward.json"


def determine_fetch_start() -> pd.Timestamp:
    if CURATED_JSON.exists():
        with CURATED_JSON.open(encoding="utf-8") as f:
            existing = json.load(f)
        ends = [pd.Timestamp(v["end"]).tz_localize(None) for v in existing.get("pairs", {}).values()]
        if ends:
            return min(ends).normalize()  # 最も古いペアに合わせて再取得(1日分の重複は許容、dedupされる)
    return CUTOFF.normalize()


def main() -> int:
    start = determine_fetch_start()
    end = pd.Timestamp.now().normalize()
    print(f"=== 週次フォワードテスト・サイクル: 取得範囲 {start.date()} 〜 {end.date()} ===")

    if start > end:
        print("新規取得範囲なし(前回実行から日付が進んでいない)、既存データのまま再計算のみ実施")
    else:
        client = GMOClient("", "")
        for symbol in FORWARD_PAIRS:
            fetch_pair(client, symbol, start, end, RAW_DIR)
        agg = aggregate_to_json(RAW_DIR, CURATED_JSON, allow_shrink=False)
        print(f"[取得完了] {agg['n_pairs']}通貨・{agg['total_bars']}本 → {agg['json']}")

    print("\n=== フォワードテスト再計算 ===")
    out = run_forward_cycle()
    p = out["backtest"]
    print(f"最新バー: {out['latest_bar_by_pair']}")
    print(f"イベント={p['n_events_raw']}件(dedup後{p['n_events_dedup']}、判定不能除外後{p['n_events_trendfiltered']})")
    print(f"トレード数={p['n_trades_total']}(決済済み{p['n_trades_closed']}、保有中{p['n_trades_open']})  "
          f"最終残高=${p['final_balance']}")
    if out["kpi"]:
        print(f"KPI: {out['kpi']['kpi_required_pass_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
