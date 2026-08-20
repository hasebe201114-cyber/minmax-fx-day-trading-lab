"""EXP-FX000005 改善ループ第5試行(TP3=4R + 初回エントリー除外)の正式KPI判定.

`evaluate_vol_breakout_dow_theory_kpi.py`(ピーク比DD採用済み)と同一のロジックを
再利用し、対象を改善ループ第5試行の$1,000バックテスト結果に差し替える。

出力: research/method-notes/vol_breakout_dow_theory_4pairs_v5_kpi_evaluation.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_vol_breakout_dow_theory_kpi import KPI_THRESHOLDS, evaluate_period  # noqa: E402


def main() -> int:
    print("=== EXP-FX000005 改善ループ第5試行(TP3=4R + 初回エントリー除外) 正式KPI判定 ===\n")

    with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_v5_1000usd_backtest.json").open(
        encoding="utf-8"
    ) as f:
        backtest = json.load(f)

    results = {}
    for period_name in ["train", "validation", "test"]:
        r = evaluate_period(period_name, backtest["periods"][period_name])
        results[period_name] = r
        print(f"--- {period_name} ---")
        print(f"  月次シャープ={r['monthly_sharpe']}  最大DD(ピーク比)={r['max_dd_pct']}%(月間{r['max_dd_monthly_pct']}%)  "
              f"PF={r['profit_factor']}  ペイオフ={r['payoff_ratio']}  最大連敗={r['max_consecutive_losses']}")
        print(f"  スプレッドコスト倍率={r['spread_cost_multiplier']}  実効n={r['n_trades_effective']}  "
              f"perm_p={r['permutation_p_clustered']}")
        print(f"  KPI: {r['kpi_pass_count']} 達成  {r['kpi_pass']}\n")

    print("=== サマリ ===")
    print(f"{'期間':<12}{'Sharpe':>8}{'最大DD':>8}{'PF':>7}{'ペイオフ':>9}{'実効n':>8}{'perm_p':>8}{'KPI達成':>9}")
    for name in ["train", "validation", "test"]:
        r = results[name]
        print(f"{name:<12}{r['monthly_sharpe']:>8}{r['max_dd_pct']:>7.2f}%{r['profit_factor']:>7}"
              f"{r['payoff_ratio']:>9}{r['n_trades_effective']:>8}{str(r['permutation_p_clustered']):>8}"
              f"{r['kpi_pass_count']:>9}")

    all_pass = all(r["kpi_pass_count"].split("/")[0] == r["kpi_pass_count"].split("/")[1] for r in results.values())
    print(f"\n3期間すべてで全KPI達成: {'はい' if all_pass else 'いいえ'}")

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_v5_kpi_evaluation.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "selected_pairs": backtest["selected_pairs"],
            "kpi_thresholds": KPI_THRESHOLDS,
            "periods": results,
            "all_periods_all_kpi_pass": all_pass,
            "_note": "改善ループ第5試行(TP3=4R+初回エントリー除外)。ピーク比DD採用済みの既存版と同一ロジックで判定。",
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
