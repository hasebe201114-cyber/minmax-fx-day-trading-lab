"""SYS-FX012改善ループ第2試行の最良候補「N_BREAKOUT単独+H1トレンド判定不能
除外フィルター」を、Trainで導出したパラメータのまま(再学習せず)Validation
へ1回だけ適用する.

00-spec.md「検証プロトコル」節: 「Validationは、Trainで一定の見込みが立った
場合にのみ参照する（HARKing防止）」。Trainで必須KPI7/9(6パターン中最良、
元のベースライン6/9を上回る)という見込みが立ったため、司令塔の指示
「validation検証を進めたい」を受けて実施する。

対象は6パターンのうち最良だった1パターンのみ(候補①+判定不能除外フィルター)。
他の5パターンはValidationでは評価しない(多重比較・選択バイアスを避けるため、
Trainで最良だった1候補のみを1回だけ確認するという事前登録済みの運用方針)。

検出層・フィルター・M5エントリー/出口/コスト/検定はTrain評価から一切変更
しない。

出力: research/method-notes/vol_breakout_trendfilter_candidate1_validation_backtest.json
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

from backtest_vol_continuation_candidates_trendfilter_4pairs_trainonly import (  # noqa: E402
    detect_candidate1, run_period,
)
from backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd import PERIODS  # noqa: E402
from evaluate_vol_breakout_dow_theory_kpi import evaluate_period  # noqa: E402


def main() -> int:
    start, end = PERIODS["validation"]
    print("=== SYS-FX012改善ループ第2試行 最良候補のValidation確認 ===")
    print("候補: N_BREAKOUT単独 + H1トレンド判定不能除外フィルター（Trainパラメータをそのまま適用、再学習なし）\n")

    result = run_period("candidate1_n_breakout_only_trendfilter", detect_candidate1, start, end)
    kpi = evaluate_period("validation", result, perm_p_field="perm_p_block",
                           apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)

    print(f"\nKPI: {kpi['kpi_required_pass_count']}  実効n={kpi['n_trades_effective']}  "
          f"Sharpe={kpi['monthly_sharpe']}  PF={kpi['profit_factor']}  ペイオフ={kpi['payoff_ratio']}  "
          f"DD={kpi['max_dd_pct']}%  perm_p={kpi['permutation_p_clustered']}")
    print(f"kpi_pass: {kpi['kpi_pass']}")

    # Trainとの比較用にTrain側の結果も読み込む
    with (ROOT / "research" / "method-notes" / "vol_continuation_candidates_trendfilter_4pairs_trainonly_backtest.json").open(
        encoding="utf-8"
    ) as f:
        train_all = json.load(f)
    train_kpi = train_all["kpi"]["candidate1_n_breakout_only"]

    print("\n=== Train(判定不能除外後) vs Validation(判定不能除外後) ===")
    print(f"{'指標':<20}{'Train':>14}{'Validation':>14}")
    for k, label in [
        ("n_trades_effective", "実効n"), ("monthly_sharpe", "月次シャープ"),
        ("profit_factor", "PF"), ("payoff_ratio", "ペイオフ"),
        ("max_dd_pct", "最大DD%"), ("spread_cost_multiplier", "スプレッド倍率"),
        ("permutation_p_clustered", "perm_p"),
    ]:
        print(f"{label:<20}{str(train_kpi.get(k)):>14}{str(kpi.get(k)):>14}")
    print(f"{'KPI達成(必須)':<20}{train_kpi['kpi_required_pass_count']:>14}{kpi['kpi_required_pass_count']:>14}")

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_trendfilter_candidate1_validation_backtest.json"
    out_path.write_text(json.dumps({
        "generated_at": pd.Timestamp.now().isoformat(),
        "design": "N_BREAKOUT単独 + H1ダウ理論トレンド判定不能除外フィルター(zigzag threshold_atr=2.0)",
        "note": "Trainで導出・選定したパラメータをそのまま適用(再学習なし)。6パターン中Trainで最良だった"
                "この1候補のみをValidationで1回だけ確認する(多重比較を避けるための事前登録済み運用方針)",
        "train_reference": train_kpi,
        "validation": kpi,
        "backtest": result,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
