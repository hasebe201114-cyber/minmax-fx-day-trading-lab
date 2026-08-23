"""EXP-FX000011(SYS-FX017): 同一イベント内の再エントリー回数上限(max_entry_seq=3)
を候補①に適用したTrain評価.

事前登録(`research/EXP-FX000011/00-spec.md`): max_entry_seq=3は「同一方向への
建て増しは最大3回まで」という一般的なリスク管理慣行に基づく唯一の事前登録値。
簡易プローブ(EXP-FX000009の議論で提示した粗い試算)の結果を選定根拠にしない。
検出層・トレンド判定フィルター・出口設計・コストモデルは完全凍結(候補①と同一)。

出力: research/method-notes/sysfx017_max_entry_seq_trainonly_backtest.json
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

from backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd import PERIODS  # noqa: E402
from backtest_vol_continuation_candidates_trendfilter_4pairs_trainonly import (  # noqa: E402
    detect_candidate1, run_period,
)
from evaluate_vol_breakout_dow_theory_kpi import evaluate_period  # noqa: E402

MAX_ENTRY_SEQ = 3  # 事前登録値(一般的なピラミッディング上限慣行、結果を見て選んでいない)


def main() -> int:
    start, end = PERIODS["train"]
    print(f"=== EXP-FX000011(SYS-FX017): max_entry_seq={MAX_ENTRY_SEQ} でTrain評価 ===\n")

    result = run_period("candidate1_max_entry_seq3", detect_candidate1, start, end,
                         max_entry_seq=MAX_ENTRY_SEQ)
    kpi = evaluate_period("train", result, perm_p_field="perm_p_block",
                           apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)

    print(f"\nKPI: {kpi['kpi_required_pass_count']}  実効n={kpi['n_trades_effective']}  "
          f"Sharpe={kpi['monthly_sharpe']}  PF={kpi['profit_factor']}  ペイオフ={kpi['payoff_ratio']}  "
          f"DD={kpi['max_dd_pct']}%  perm_p={kpi['permutation_p_clustered']}")
    print(f"kpi_pass: {kpi['kpi_pass']}")

    with (ROOT / "research" / "method-notes" / "candidate3_cost_ratio_filter_trainonly_backtest.json").open(
        encoding="utf-8"
    ) as f:
        c1_result = json.load(f)
    candidate1_kpi = c1_result["candidate1_reference"]

    print("\n=== 候補①(基準、キャップ無し) vs 候補①+max_entry_seq=3、Train ===")
    print(f"{'指標':<20}{'候補①':>14}{'+cap3':>14}")
    for k, label in [
        ("n_trades_effective", "実効n"), ("monthly_sharpe", "月次シャープ"),
        ("profit_factor", "PF"), ("payoff_ratio", "ペイオフ"),
        ("max_dd_pct", "最大DD%"), ("spread_cost_multiplier", "スプレッド倍率"),
        ("permutation_p_clustered", "perm_p"),
    ]:
        print(f"{label:<20}{str(candidate1_kpi.get(k)):>14}{str(kpi.get(k)):>14}")
    print(f"{'KPI達成(必須)':<20}{candidate1_kpi['kpi_required_pass_count']:>14}{kpi['kpi_required_pass_count']:>14}")

    out_path = ROOT / "research" / "method-notes" / "sysfx017_max_entry_seq_trainonly_backtest.json"
    out_path.write_text(json.dumps({
        "generated_at": pd.Timestamp.now().isoformat(),
        "design": f"候補①(N_BREAKOUT=3.5+H1トレンド判定不能除外フィルター) + max_entry_seq={MAX_ENTRY_SEQ}"
                  "(同一イベント内の再エントリー回数上限、一般的なピラミッディング制限慣行に基づく事前登録値)",
        "note": "EXP-FX000009の追加分析(トレンド判定H1不変・エントリーM3化の原因分析)で「同一イベント内の"
                "再エントリーが多いほど平均r_netが低下する」と判明したことを受けた対策。検出層・トレンド判定"
                "フィルター・出口設計・コストモデルは変更しない",
        "max_entry_seq": MAX_ENTRY_SEQ,
        "candidate1_reference": candidate1_kpi,
        "kpi": kpi,
        "backtest": result,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
