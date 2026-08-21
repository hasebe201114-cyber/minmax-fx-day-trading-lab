"""EXP-FX000005 T-08(K3mのスケール不変な再定義)適用後の正式KPI判定.

T-16再査読(2026-08-21)が新規発見: 外部レビュー(2026-08-20)が指摘したT-08
(「最大連続損失≤5」は絶対件数のためnに依存し、観測勝率のi.i.d.帰無仮説でも
通過率が約6割前後になる=ノイズと区別がつかない)が、優先順位トラッキングから
漏れたまま一度も対応されていなかった。

`evaluate_vol_breakout_dow_theory_kpi.py`の`evaluate_period()`を
`apply_k3m_scale_invariant=True`で呼び出し、K3mを絶対件数閾値から
`decision.criteria.compute_k3m_scale_invariant()`によるi.i.d.帰無分布
パーセンタイル判定に切り替える。対象データ・他のKPI設定(perm_p_field=
"perm_p_block"・apply_n_correlation_discount=False)はtrailonly版(T-01〜T-13+
重複バグ修正まで)から変更しない。

00-spec.md「Test凍結宣言」により、Testの数値もここで算出・記録するが、
結論・意思決定にはTrain+Validationのみを用いる。

出力: research/method-notes/vol_breakout_dow_theory_4pairs_v7_trailonly_t08fix_kpi_evaluation.json
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
    print("=== EXP-FX000005 T-08(K3mスケール不変再定義)適用後 正式KPI判定 ===\n")

    with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd_backtest.json").open(
        encoding="utf-8"
    ) as f:
        backtest = json.load(f)

    results = {}
    for period_name in ["train", "validation", "test"]:
        r = evaluate_period(period_name, backtest["periods"][period_name],
                             perm_p_field="perm_p_block", apply_n_correlation_discount=False,
                             apply_k3m_scale_invariant=True)
        results[period_name] = r
        print(f"--- {period_name} ---")
        print(f"  月次シャープ={r['monthly_sharpe']}  最大DD(ピーク比)={r['max_dd_pct']}%(月間{r['max_dd_monthly_pct']}%)  "
              f"PF={r['profit_factor']}  ペイオフ={r['payoff_ratio']}  最大連敗={r['max_consecutive_losses']}"
              f"(K3m判定={'PASS' if r['kpi_pass']['max_consecutive_losses'] else 'FAIL'})")
        if r["k3m_scale_invariant"]:
            k3m = r["k3m_scale_invariant"]
            print(f"    i.i.d.帰無分布パーセンタイル={k3m['observed_percentile_in_null']}"
                  f" (n={k3m['n_trades']}, win_rate={k3m['win_rate']}, alpha={k3m['alpha']})")
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

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_v7_trailonly_t08fix_kpi_evaluation.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "selected_pairs": backtest["selected_pairs"],
            "kpi_thresholds": KPI_THRESHOLDS,
            "periods": results,
            "all_periods_all_kpi_pass": all_pass,
            "_note": (
                "T-01〜T-09+重複トレード生成バグ修正+T-13(トレール専業化)に加え、"
                "T-08(K3mをi.i.d.帰無分布パーセンタイル判定へ再定義)を適用した正式KPI判定。"
                "対象バックテストデータはtrailonly版から変更なし(K3mの判定ロジックのみ変更)。"
                "Test凍結宣言によりTestの数値も算出するが、結論・意思決定にはTrain+Validation"
                "のみを用いる。"
            ),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
