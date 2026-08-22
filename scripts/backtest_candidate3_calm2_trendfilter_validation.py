"""SYS-FX012改善ループ第3試行の候補「候補③(CALM_RATIO=2.0、既存値)+H1トレンド
判定不能除外フィルター」を、Trainで導出したパラメータのまま(再学習せず)
Validationへ1回だけ適用する.

CALM_RATIOグリッドサーチ(1.5〜3.5)ではTrain必須KPIが2.0と2.5でともに6/9の
プラトーを形成したが、①+判定不能除外フィルター(n=300, 7/9)には届かなかった。
本試行はn=300より頑健(実効nに余裕がある)候補の代表として、既存値である
CALM_RATIO=2.0(グリッドサーチ前から`price_shock_filter.py`で確立済みの値、
恣意的に選んだ値ではない)をValidationで1回だけ確認する。CALM_RATIO=2.5等
グリッド内の他の値はValidationでは評価しない(多重比較を避ける事前登録済み
運用方針)。

出力: research/method-notes/candidate3_calm2_trendfilter_validation_backtest.json
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


def main() -> int:
    start, end = PERIODS["validation"]
    print("=== SYS-FX012改善ループ第3試行 候補③(CALM_RATIO=2.0)+判定不能除外 Validation確認 ===\n")

    detect_fn = make_detect_candidate3(CALM_RATIO)
    result = run_period(f"candidate3_calm{CALM_RATIO}_trendfilter", detect_fn, start, end)
    kpi = evaluate_period("validation", result, perm_p_field="perm_p_block",
                           apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)

    print(f"\nKPI: {kpi['kpi_required_pass_count']}  実効n={kpi['n_trades_effective']}  "
          f"Sharpe={kpi['monthly_sharpe']}  PF={kpi['profit_factor']}  ペイオフ={kpi['payoff_ratio']}  "
          f"DD={kpi['max_dd_pct']}%  perm_p={kpi['permutation_p_clustered']}")
    print(f"kpi_pass: {kpi['kpi_pass']}")

    # Trainとの比較用
    with (ROOT / "research" / "method-notes" / "candidate3_calmratio_sweep_trainonly_backtest.json").open(
        encoding="utf-8"
    ) as f:
        sweep = json.load(f)
    train_kpi = sweep["kpi"][f"calm_ratio_{CALM_RATIO}"]

    print("\n=== Train(判定不能除外後) vs Validation(判定不能除外後)、CALM_RATIO=2.0 ===")
    print(f"{'指標':<20}{'Train':>14}{'Validation':>14}")
    for k, label in [
        ("n_trades_effective", "実効n"), ("monthly_sharpe", "月次シャープ"),
        ("profit_factor", "PF"), ("payoff_ratio", "ペイオフ"),
        ("max_dd_pct", "最大DD%"), ("spread_cost_multiplier", "スプレッド倍率"),
        ("permutation_p_clustered", "perm_p"),
    ]:
        print(f"{label:<20}{str(train_kpi.get(k)):>14}{str(kpi.get(k)):>14}")
    print(f"{'KPI達成(必須)':<20}{train_kpi['kpi_required_pass_count']:>14}{kpi['kpi_required_pass_count']:>14}")

    out_path = ROOT / "research" / "method-notes" / "candidate3_calm2_trendfilter_validation_backtest.json"
    out_path.write_text(json.dumps({
        "generated_at": pd.Timestamp.now().isoformat(),
        "design": f"N_BREAKOUT OR (Donchian(20) AND CALM_RATIO>={CALM_RATIO}) + H1トレンド判定不能除外フィルター",
        "note": "CALM_RATIOグリッドサーチ(1.5-3.5)のうち、既存値2.0(恣意的選定でない)を代表として"
                "Validationで1回だけ確認する(多重比較を避けるための事前登録済み運用方針)",
        "train_reference": train_kpi,
        "validation": kpi,
        "backtest": result,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
