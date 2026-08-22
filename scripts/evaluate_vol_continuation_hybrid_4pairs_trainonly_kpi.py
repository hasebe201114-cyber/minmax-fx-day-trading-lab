"""EXP-FX000006(SYS-FX012) Trainベースラインの正式KPI判定.

SYS-FX011の最終確定評価設定(perm_p_field="perm_p_block"、
apply_n_correlation_discount=False、apply_k3m_scale_invariant=True)を
そのまま踏襲する(`00-spec.md`「KPI閾値」節、SYS-FX011と完全に同一枠組み)。

出力: research/method-notes/vol_continuation_hybrid_4pairs_trainonly_kpi_evaluation.json
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
    print("=== EXP-FX000006(SYS-FX012) Trainベースライン 正式KPI判定 ===\n")

    with (ROOT / "research" / "method-notes" / "vol_continuation_hybrid_4pairs_trainonly_backtest.json").open(
        encoding="utf-8"
    ) as f:
        backtest = json.load(f)

    r = evaluate_period("train", backtest["periods"]["train"],
                         perm_p_field="perm_p_block", apply_n_correlation_discount=False,
                         apply_k3m_scale_invariant=True)

    print("--- train ---")
    print(f"  月次シャープ={r['monthly_sharpe']}  最大DD(ピーク比)={r['max_dd_pct']}%(月間{r['max_dd_monthly_pct']}%)  "
          f"PF={r['profit_factor']}  ペイオフ={r['payoff_ratio']}  最大連敗={r['max_consecutive_losses']}"
          f"(K3m判定={'PASS' if r['kpi_pass']['max_consecutive_losses'] else 'FAIL'})")
    if r["k3m_scale_invariant"]:
        k3m = r["k3m_scale_invariant"]
        print(f"    i.i.d.帰無分布パーセンタイル={k3m['observed_percentile_in_null']}"
              f" (n={k3m['n_trades']}, win_rate={k3m['win_rate']}, alpha={k3m['alpha']})")
    print(f"  スプレッドコスト倍率={r['spread_cost_multiplier']}  実効n={r['n_trades_effective']}  "
          f"perm_p={r['permutation_p_clustered']}")
    print(f"  KPI: {r['kpi_pass_count']} 達成(必須{r['kpi_required_pass_count']})  {r['kpi_pass']}\n")

    # SYS-FX011ベースライン(trailonly、T-08適用後)との直接比較
    with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_v7_trailonly_t08fix_kpi_evaluation.json").open(
        encoding="utf-8"
    ) as f:
        baseline = json.load(f)
    b = baseline["periods"]["train"]
    print("=== SYS-FX011ベースライン(N_BREAKOUT単独)との比較 ===")
    print(f"{'指標':<20}{'ベースライン':>14}{'ハイブリッド':>14}")
    for k, label in [
        ("n_trades_effective", "実効n"), ("monthly_sharpe", "月次シャープ"),
        ("profit_factor", "PF"), ("payoff_ratio", "ペイオフ"),
        ("max_dd_pct", "最大DD%"), ("spread_cost_multiplier", "スプレッド倍率"),
        ("permutation_p_clustered", "perm_p"),
    ]:
        print(f"{label:<20}{str(b.get(k)):>14}{str(r.get(k)):>14}")
    print(f"{'KPI達成(必須)':<20}{b['kpi_required_pass_count']:>14}{r['kpi_required_pass_count']:>14}")

    out_path = ROOT / "research" / "method-notes" / "vol_continuation_hybrid_4pairs_trainonly_kpi_evaluation.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "selected_pairs": backtest["selected_pairs"],
            "design": backtest["design"],
            "kpi_thresholds": KPI_THRESHOLDS,
            "train": r,
            "baseline_comparison_source": "vol_breakout_dow_theory_4pairs_v7_trailonly_t08fix_kpi_evaluation.json",
            "_note": "00-spec.mdの検証プロトコルに従いTrain単独のみの正式判定。Validation/Testは未実施",
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
