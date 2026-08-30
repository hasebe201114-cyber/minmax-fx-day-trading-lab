"""統計検定モジュール.

起源:
    minmax-fx-eval-framework (v0.2/v0.3 フレームワーク) の DSR 実装を
    親 PJ (minmax-fx-day-trading-lab) へ Phase 1/2 マージ (2026-08-29/30).
    Phase 1: 破壊的変更なし・参考値扱い.
    Phase 2: v0.3 必須ゲート化.
"""

from .dsr import (
    DeflatedSharpeRatioResult,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    expected_max_sharpe_ratio,
    DSR_REQUIRED_THRESHOLD,
    EULER_MASCHERONI,
    EULER_E,
)
from .n_trials_counter import (
    NTrialsBreakdown,
    count_n_trials,
    KNOWN_STRATEGY_N_TRIALS,
)

__all__ = [
    # DSR
    "DeflatedSharpeRatioResult",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe_ratio",
    "DSR_REQUIRED_THRESHOLD",
    "EULER_MASCHERONI",
    "EULER_E",
    # n_trials
    "NTrialsBreakdown",
    "count_n_trials",
    "KNOWN_STRATEGY_N_TRIALS",
]
