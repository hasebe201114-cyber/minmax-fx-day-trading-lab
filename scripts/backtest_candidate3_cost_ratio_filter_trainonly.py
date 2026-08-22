"""SYS-FX012改善ループ第5試行(最終、上限5回のうち最後の1回): 候補③(CALM_RATIO=2.0)
+H1トレンド判定不能除外フィルターに、コスト比率フィルターを追加した場合のTrain評価.

候補③のValidation DD悪化(27.27%)のトレードレベル分析で、ATR(M5)が閑散相場で
極端に縮小するとSL幅(=1R)も比例して縮小し、固定額のスプレッド・スリッページ
コストがR単位に対して肥大化するケース(コスト膨張トレード)が集中していたと
判明した(2026-08-22)。検出層・エントリー層・出口設計は一切変更せず、エントリー
時点で見積もり往復コストがinitial_risk(1R)の1/3(=既存K5m閾値の単一トレード版)
を超える場合はエントリーを見送るフィルターのみを追加する。

事前登録: research/EXP-FX000006/00-spec.md「改善ループ第5試行」節を参照。
新規自由パラメータはcost_ratio_max=1/3のみ(K5m閾値からの転用、恣意的選定でない)。

出力: research/method-notes/candidate3_cost_ratio_filter_trainonly_backtest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from backtest_candidate3_calmratio_sweep_trainonly import make_detect_candidate3  # noqa: E402
from backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd import PERIODS  # noqa: E402
from backtest_vol_continuation_candidates_trendfilter_4pairs_trainonly import run_period  # noqa: E402
from evaluate_vol_breakout_dow_theory_kpi import evaluate_period  # noqa: E402

CALM_RATIO = 2.0
COST_RATIO_MAX = 1.0 / 3.0  # K5m(スプレッドコスト倍率≥3倍)の単一トレード版、恣意的パラメータではない


def main() -> int:
    start, end = PERIODS["train"]
    print("=== SYS-FX012改善ループ第5試行(最終) 候補③(CALM_RATIO=2.0)+コスト比率フィルター Train評価 ===\n")
    print(f"cost_ratio_max = {COST_RATIO_MAX:.4f} (K5m閾値≥3倍の単一トレード版として転用)\n")

    detect_fn = make_detect_candidate3(CALM_RATIO)
    result = run_period(f"candidate3_calm{CALM_RATIO}_costfilter", detect_fn, start, end,
                         cost_ratio_max=COST_RATIO_MAX)
    kpi = evaluate_period("train", result, perm_p_field="perm_p_block",
                           apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)

    print(f"\nKPI: {kpi['kpi_required_pass_count']}  実効n={kpi['n_trades_effective']}  "
          f"Sharpe={kpi['monthly_sharpe']}  PF={kpi['profit_factor']}  ペイオフ={kpi['payoff_ratio']}  "
          f"DD={kpi['max_dd_pct']}%  perm_p={kpi['permutation_p_clustered']}")
    print(f"kpi_pass: {kpi['kpi_pass']}")

    # 比較用: 候補③(フィルターなし)のTrain基準、候補①(最良)のTrain基準
    with (ROOT / "research" / "method-notes" / "candidate3_calmratio_sweep_trainonly_backtest.json").open(
        encoding="utf-8"
    ) as f:
        sweep = json.load(f)
    candidate3_baseline_kpi = sweep["kpi"][f"calm_ratio_{CALM_RATIO}"]

    with (ROOT / "research" / "method-notes" /
          "vol_continuation_candidates_trendfilter_4pairs_trainonly_backtest.json").open(encoding="utf-8") as f:
        c1_result = json.load(f)
    candidate1_kpi = c1_result["kpi"]["candidate1_n_breakout_only"]

    print("\n=== 候補①(最良、フィルター無し) vs 候補③(フィルター無し) vs 候補③+コスト比率フィルター、Train ===")
    print(f"{'指標':<20}{'候補①':>14}{'候補③素':>14}{'候補③+filter':>16}")
    for k, label in [
        ("n_trades_effective", "実効n"), ("monthly_sharpe", "月次シャープ"),
        ("profit_factor", "PF"), ("payoff_ratio", "ペイオフ"),
        ("max_dd_pct", "最大DD%"), ("spread_cost_multiplier", "スプレッド倍率"),
        ("permutation_p_clustered", "perm_p"),
    ]:
        print(f"{label:<20}{str(candidate1_kpi.get(k)):>14}{str(candidate3_baseline_kpi.get(k)):>14}"
              f"{str(kpi.get(k)):>16}")
    print(f"{'KPI達成(必須)':<20}{candidate1_kpi['kpi_required_pass_count']:>14}"
          f"{candidate3_baseline_kpi['kpi_required_pass_count']:>14}{kpi['kpi_required_pass_count']:>16}")

    out_path = ROOT / "research" / "method-notes" / "candidate3_cost_ratio_filter_trainonly_backtest.json"
    out_path.write_text(json.dumps({
        "generated_at": pd.Timestamp.now().isoformat(),
        "design": f"N_BREAKOUT OR (Donchian(20) AND CALM_RATIO>={CALM_RATIO}) + H1トレンド判定不能除外フィルター "
                  f"+ コスト比率フィルター(cost_ratio_max={COST_RATIO_MAX:.4f})",
        "note": "改善ループ第5試行(最終、上限5回のうち最後の1回)。候補③のValidation DD悪化の原因分析(閑散相場での"
                "SL幅極端縮小によるコスト比率肥大化)を受けた対策。検出層・エントリー層・出口設計は変更しない",
        "cost_ratio_max": COST_RATIO_MAX,
        "candidate1_reference": candidate1_kpi,
        "candidate3_baseline_reference": candidate3_baseline_kpi,
        "kpi": kpi,
        "backtest": result,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
