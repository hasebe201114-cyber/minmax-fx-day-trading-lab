"""SYS-FX012改善ループ第3試行: 候補③(N_BREAKOUT OR (Donchian AND CALM_RATIO))
+H1トレンド判定不能除外フィルターをベースに、CALM_RATIOの閾値をTrain単独で
グリッドサーチする.

事前登録(結果を見る前に固定): CALM_RATIOの候補値は[1.5, 2.0, 2.5, 3.0, 3.5]
の5点とする。2.0(既存の`price_shock_filter.py`の値)を含む対称的な範囲を機械的に
設定し、結果を見てから範囲を追加・変更しない。3.5はN_BREAKOUT自体と同値になる
点(この場合Donchian分岐が実質無効化される、参考の境界値)。

検出条件: range/ATR>=N_BREAKOUT(3.5) OR (Donchian(20)ブレイク AND
range/ATR>=CALM_RATIO(可変)) 、かつH1ダウ理論トレンド判定不能イベントは除外
(改善ループ第2試行で確立した設計をそのまま踏襲)。

正式プロトコル外の探索的な比較試算(グリッドサーチはT-18の数え方に準拠し
改善ループ1試行としてカウント)。00-spec.md等は変更しない(グリッド結果は
別途仕様書へ反映する)。

出力: research/method-notes/candidate3_calmratio_sweep_trainonly_backtest.json
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
    run_period,
)
from backtest_vol_continuation_hybrid_4pairs_trainonly import DONCHIAN_LENGTH  # noqa: E402
from derive_vol_breakout_entry_params import N_BREAKOUT  # noqa: E402
from evaluate_vol_breakout_dow_theory_kpi import evaluate_period  # noqa: E402
from minmax_fx_dt.strategy.indicators import donchian  # noqa: E402

CALM_RATIO_GRID = [1.5, 2.0, 2.5, 3.0, 3.5]  # 事前登録、結果を見る前に固定


def make_detect_candidate3(calm_ratio: float):
    def detect(h1, atr_h1):
        dc = donchian(h1["high"], h1["low"], DONCHIAN_LENGTH, DONCHIAN_LENGTH).shift(1)
        ratio = (h1["high"] - h1["low"]) / atr_h1
        close, open_ = h1["close"], h1["open"]
        is_spike_up = (ratio >= N_BREAKOUT) & (close > open_)
        is_spike_down = (ratio >= N_BREAKOUT) & (close < open_)
        is_cont_up = (close > dc["DCU"]) & (ratio >= calm_ratio)
        is_cont_down = (close < dc["DCL"]) & (ratio >= calm_ratio)
        up = (is_spike_up | is_cont_up).fillna(False)
        down = (is_spike_down | is_cont_down).fillna(False)
        return up, down
    return detect


def main() -> int:
    start, end = PERIODS["train"]
    print(f"=== 候補③+判定不能除外フィルター CALM_RATIOグリッドサーチ(Train) ===")
    print(f"グリッド: {CALM_RATIO_GRID}\n")

    results, kpis = {}, {}
    for calm_ratio in CALM_RATIO_GRID:
        name = f"calm_ratio_{calm_ratio}"
        detect_fn = make_detect_candidate3(calm_ratio)
        period_result = run_period(name, detect_fn, start, end)
        kpi = evaluate_period("train", period_result, perm_p_field="perm_p_block",
                               apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)
        results[name] = period_result
        kpis[name] = kpi
        print(f"  CALM_RATIO={calm_ratio}: n={kpi['n_trades_effective']}  "
              f"KPI={kpi['kpi_required_pass_count']}  Sharpe={kpi['monthly_sharpe']}  "
              f"PF={kpi['profit_factor']}  ペイオフ={kpi['payoff_ratio']}  DD={kpi['max_dd_pct']}%  "
              f"perm_p={kpi['permutation_p_clustered']}\n")

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "purpose": "候補③+H1トレンド判定不能除外フィルターをベースにCALM_RATIOをTrain単独でグリッドサーチ",
        "calm_ratio_grid": CALM_RATIO_GRID,
        "backtest": results,
        "kpi": kpis,
    }
    out_path = ROOT / "research" / "method-notes" / "candidate3_calmratio_sweep_trainonly_backtest.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
