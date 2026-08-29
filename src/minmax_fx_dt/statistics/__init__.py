"""統計検定モジュール.

起源:
    minmax-fx-eval-framework (v0.2 フレームワーク) の DSR 実装を
    親 PJ (minmax-fx-day-trading-lab) へ Phase 1 マージ (2026-08-29).
    破壊的変更なし・参考値扱い.
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

__all__ = [
    # DSR
    "DeflatedSharpeRatioResult",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe_ratio",
    "DSR_REQUIRED_THRESHOLD",
    "EULER_MASCHERONI",
    "EULER_E",
]
