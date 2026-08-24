"""EXP-FX000016 Stage 1・コンポーネントB/C: バックテストとの乖離レコンサイル.

`poll_ticker.py`で蓄積したライブ気配値と、SYS-FX012フォワードテスト台帳
(`sysfx012_forward_test_ledger.json`)の各トレードを時刻で突き合わせ、以下を算出する:

  B. 実測スプレッド vs コストモデル定数(SPREAD_PIPS)の乖離
  C. エントリー/エグジット時刻のmarket_statusが常にOPENか(市場休止中の
     誤検出が無いかの整合性チェック)

各トレードについて、entry_time/exit_timeに時間的に最も近いライブ気配値
レコードを探す。**許容範囲(既定90分)を超えて近いレコードが無い場合は
「判定不能」として明示し、無理に埋めない**(spec §Stage 1受け入れ基準)。

ライブ記録は2026-08-24開始のため、それ以前のトレードは全件「判定不能」
になるのが正常(想定通り)。記録が蓄積されるにつれ判定可能なトレードが
増えていく。

Usage:
    PYTHONPATH=src python3 scripts/live_monitor/reconcile_divergence.py

出力: research/method-notes/live_cost_divergence.json
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd import SPREAD_PIPS  # noqa: E402

LEDGER_PATH = ROOT / "research" / "method-notes" / "sysfx012_forward_test_ledger.json"
LIVE_TICKER_DIR = ROOT / "data" / "raw" / "live-ticker"
OUT_PATH = ROOT / "research" / "method-notes" / "live_cost_divergence.json"

MATCH_TOLERANCE = timedelta(minutes=90)  # 毎時ポーリングなので±90分を許容範囲とする


def load_live_ticker_records() -> list[dict]:
    records = []
    if not LIVE_TICKER_DIR.exists():
        return records
    for csv_path in sorted(LIVE_TICKER_DIR.glob("*.csv")):
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["polled_at"] = datetime.fromisoformat(row["polled_at"])
                row["spread_pips"] = float(row["spread_pips"])
                records.append(row)
    return records


def nearest_record(records_by_pair: dict[str, list[dict]], pair: str, ts: datetime) -> dict | None:
    candidates = records_by_pair.get(pair, [])
    if not candidates:
        return None
    best, best_delta = None, None
    for r in candidates:
        delta = abs(r["polled_at"] - ts)
        if best_delta is None or delta < best_delta:
            best, best_delta = r, delta
    if best is None or best_delta > MATCH_TOLERANCE:
        return None
    return best


def main() -> int:
    if not LEDGER_PATH.exists():
        print(f"[ERROR] {LEDGER_PATH} が見つかりません。先にフォワードテストを実行してください")
        return 1

    with LEDGER_PATH.open(encoding="utf-8") as f:
        ledger = json.load(f)
    trades = ledger.get("backtest", {}).get("trades", [])

    live_records = load_live_ticker_records()
    records_by_pair: dict[str, list[dict]] = {}
    for r in live_records:
        records_by_pair.setdefault(r["pair"], []).append(r)

    print(f"ライブ気配値レコード: {len(live_records)}件（{LIVE_TICKER_DIR}）")
    print(f"フォワードテスト・トレード: {len(trades)}件\n")

    reconciled, insufficient_data = [], 0
    status_anomalies = []
    for t in trades:
        pair = t["pair"]
        modeled_spread = SPREAD_PIPS.get(pair, 0.5)
        entry_ts = datetime.fromisoformat(t["entry_time"]).replace(tzinfo=timezone.utc)
        rec = nearest_record(records_by_pair, pair, entry_ts)
        if rec is None:
            insufficient_data += 1
            continue
        realized_spread = rec["spread_pips"]
        divergence_ratio = realized_spread / modeled_spread if modeled_spread else None
        if rec["market_status"] != "OPEN":
            status_anomalies.append({
                "pair": pair, "entry_time": t["entry_time"], "market_status": rec["market_status"],
            })
        reconciled.append({
            "pair": pair, "entry_time": t["entry_time"],
            "modeled_spread_pips": modeled_spread, "realized_spread_pips": realized_spread,
            "divergence_ratio": round(divergence_ratio, 3) if divergence_ratio else None,
            "matched_record_delta_min": round(
                abs(rec["polled_at"] - entry_ts).total_seconds() / 60, 1),
        })

    print(f"突き合わせ成立: {len(reconciled)}件 / 判定不能(記録密度不足): {insufficient_data}件\n")

    by_pair_summary = {}
    for pair in sorted({t["pair"] for t in trades}):
        pair_rows = [r for r in reconciled if r["pair"] == pair]
        if not pair_rows:
            by_pair_summary[pair] = {"n": 0, "note": "突き合わせ可能なトレードなし"}
            continue
        ratios = [r["divergence_ratio"] for r in pair_rows if r["divergence_ratio"] is not None]
        by_pair_summary[pair] = {
            "n": len(pair_rows),
            "mean_divergence_ratio": round(sum(ratios) / len(ratios), 3) if ratios else None,
            "note": "実測スプレッド ÷ モデル想定スプレッド。1.0=モデルと一致、>1.0=モデルより実測が広い",
        }
        print(f"{pair}: n={len(pair_rows)} 平均乖離倍率={by_pair_summary[pair]['mean_divergence_ratio']}")

    if status_anomalies:
        print(f"\n[要確認] エントリー時刻のmarket_statusがOPEN以外のレコード: {len(status_anomalies)}件")
    else:
        print("\n[OK] 突き合わせできた範囲では、市場休止中の誤検出は無し")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "EXP-FX000016 Stage 1: バックテストのコストモデル仮定とライブ実測スプレッドの乖離レコンサイル",
        "match_tolerance_minutes": MATCH_TOLERANCE.total_seconds() / 60,
        "n_trades_total": len(trades),
        "n_reconciled": len(reconciled),
        "n_insufficient_data": insufficient_data,
        "note": "ライブ記録は2026-08-24開始のため、それ以前のトレードは判定不能になるのが正常。"
                "記録の蓄積とともにn_reconciledが増えていく想定",
        "by_pair_summary": by_pair_summary,
        "market_status_anomalies": status_anomalies,
        "reconciled_trades": reconciled,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[出力]: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
