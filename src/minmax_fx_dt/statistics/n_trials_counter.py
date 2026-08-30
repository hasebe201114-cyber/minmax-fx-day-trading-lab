"""n_trials 厳密カウントヘルパー (v0.3 M-R2 対応).

起源:
    minmax-fx-eval-framework v0.3 から Phase 2 マージ (2026-08-30).
    詳細: research/メソッドフレームワーク再設計/00-spec-v0.3.md §5.

v0.2 では「改善ループ試行数」のみカウントしていたが、v0.3 では以下を含める:
    - 改善ループ各試行
    - グリッドサーチ・ablations（パラメータの自由度）
    - 通貨選択（"5 通貨"/"4 通貨" のような選択の自由度）
    - 期間選択（Train/Val/Test 期間変更の自由度）
    - 閾値選択（ADX 等パラメータ閾値変更の自由度）

bug fix や statistical correction（permutation 検定手法の是正等）は
自由探索ではないためカウントしない。

使用例:
    >>> n = count_n_trials(
    ...     n_improvement_loops=7,
    ...     n_currency_choices=2,  # 5 通貨 → 4 通貨
    ...     n_period_choices=1,
    ...     n_threshold_choices=2,  # N=3.5 / N=4.0 候補
    ... )
    >>> print(n)  # 7 * 2 * 1 * 2 = 28
    28
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NTrialsBreakdown:
    """n_trials 内訳の記録用."""

    n_improvement_loops: int = 1
    n_grid_search_combinations: int = 1
    n_currency_choices: int = 1
    n_period_choices: int = 1
    n_threshold_choices: int = 1
    n_bug_fixes: int = 0  # カウントしない（参考情報）
    notes: str = ""

    @property
    def conservative(self) -> int:
        """conservative カウント: 全ての自由度の積."""
        return (
            self.n_improvement_loops
            * self.n_grid_search_combinations
            * self.n_currency_choices
            * self.n_period_choices
            * self.n_threshold_choices
        )

    @property
    def liberal(self) -> int:
        """liberal カウント: 全ての自由度の和（保守的に多い）."""
        return (
            self.n_improvement_loops
            + self.n_grid_search_combinations
            + self.n_currency_choices
            + self.n_period_choices
            + self.n_threshold_choices
        )

    def to_dict(self) -> dict:
        return {
            "n_improvement_loops": self.n_improvement_loops,
            "n_grid_search_combinations": self.n_grid_search_combinations,
            "n_currency_choices": self.n_currency_choices,
            "n_period_choices": self.n_period_choices,
            "n_threshold_choices": self.n_threshold_choices,
            "n_bug_fixes_excluded": self.n_bug_fixes,
            "n_trials_conservative": self.conservative,
            "n_trials_liberal": self.liberal,
            "notes": self.notes,
        }


def count_n_trials(
    n_improvement_loops: int = 1,
    *,
    n_grid_search_combinations: int = 1,
    n_currency_choices: int = 1,
    n_period_choices: int = 1,
    n_threshold_choices: int = 1,
    n_bug_fixes: int = 0,
    notes: str = "",
) -> int:
    """n_trials を厳密にカウントする便利関数.

    Args:
        n_improvement_loops: 改善ループの試行数（基本 1、Q49〜Q53 など）。
        n_grid_search_combinations: グリッドサーチ・ablations の組み合わせ数。
        n_currency_choices: 通貨選択の自由度（"5 通貨"/"4 通貨" など）。
        n_period_choices: 期間選択の自由度（Train/Val/Test 期間変更）。
        n_threshold_choices: 閾値選択の自由度（ADX 20/25、N=3.5/4.0 等）。
        n_bug_fixes: バグ修正数（カウントしない・参考情報）。
        notes: 自由度の説明（"5→4 通貨" 等）。

    Returns:
        n_trials の conservative カウント（全自由度の積）。
    """
    breakdown = NTrialsBreakdown(
        n_improvement_loops=n_improvement_loops,
        n_grid_search_combinations=n_grid_search_combinations,
        n_currency_choices=n_currency_choices,
        n_period_choices=n_period_choices,
        n_threshold_choices=n_threshold_choices,
        n_bug_fixes=n_bug_fixes,
        notes=notes,
    )
    return breakdown.conservative


# ============================================================
# 既知戦略のプリセット（参考）
# ============================================================

KNOWN_STRATEGY_N_TRIALS: dict[str, NTrialsBreakdown] = {
    "SYS-FX007": NTrialsBreakdown(
        n_improvement_loops=1,
        n_grid_search_combinations=6,  # 6 プリセット (Base〜A4)
        n_currency_choices=1,  # USD/JPY のみ
        n_period_choices=1,
        n_threshold_choices=1,
        notes="6 ablations, USD/JPY only",
    ),
    "SYS-FX008": NTrialsBreakdown(
        n_improvement_loops=3,
        n_grid_search_combinations=1,
        n_currency_choices=1,  # USD/JPY のみ（本 PJ では）
        n_period_choices=1,
        n_threshold_choices=1,
        n_bug_fixes=1,  # 週末クローズ修正（カウントしない）
        notes="3 loops, USD/JPY",
    ),
    "SYS-FX009 v2": NTrialsBreakdown(
        n_improvement_loops=1,
        n_grid_search_combinations=1,
        n_currency_choices=1,  # USD/JPY のみ
        n_period_choices=1,
        n_threshold_choices=1,
        n_bug_fixes=1,  # 陳腐化シグナル回転売買バグ
        notes="1 loop, USD/JPY, bug fix excluded",
    ),
    "SYS-FX011 v7": NTrialsBreakdown(
        n_improvement_loops=7,
        n_grid_search_combinations=1,
        n_currency_choices=2,  # 5 → 4 通貨
        n_period_choices=1,
        n_threshold_choices=2,  # N=3.5 / N=4.0 候補
        n_bug_fixes=2,  # 週末クローズ・重複トレード生成バグ（カウントしない）
        notes="7 loops, 5→4 currencies, N=3.5/4.0",
    ),
    "SYS-FX011 T-13": NTrialsBreakdown(
        n_improvement_loops=7,
        n_grid_search_combinations=1,
        n_currency_choices=2,  # 5 → 4 通貨
        n_period_choices=1,
        n_threshold_choices=2,  # N=3.5 / N=4.0
        n_bug_fixes=2,
        notes="7 loops, 5→4 currencies, N=3.5/4.0 (T-13: trailonly)",
    ),
    "SYS-FX012": NTrialsBreakdown(
        n_improvement_loops=5,  # 改善ループ上限 5
        n_grid_search_combinations=1,
        n_currency_choices=1,  # 4 通貨固定
        n_period_choices=1,
        n_threshold_choices=1,
        notes="5 loops, 4 currencies fixed",
    ),
    # SYS-FX010: キャリー戦略（no-stop 系 5 試行程度の改善）
    "SYS-FX010": NTrialsBreakdown(
        n_improvement_loops=5,  # no-stop 系の 5 バリアント
        n_grid_search_combinations=1,
        n_currency_choices=1,  # USD/JPY のみ
        n_period_choices=1,
        n_threshold_choices=1,
        notes="5 no-stop variants, USD/JPY (synthetic returns)",
    ),
}


__all__ = [
    "NTrialsBreakdown",
    "count_n_trials",
    "KNOWN_STRATEGY_N_TRIALS",
]
