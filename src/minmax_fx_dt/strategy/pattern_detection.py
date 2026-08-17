"""SYS-FX009 v2 ダブルトップ/ダブルボトム検出 (EXP-FX000003).

ZigZag転換点（`support_resistance.zigzag_pivots_typed`）を使い、直近3点の
連続転換点（山→谷→山 / 谷→山→谷）が「ほぼ同水準の2つの山（谷）」を形成して
いるかを判定する。「上位足の流れに乗る」設計のため、ダブルトップはLT方向が
DOWNの時のみ、ダブルボトムはUPの時のみ有効なシグナルとして扱う
（トレンド反転の天井/底取りではなく、トレンド継続中の戻り高値/安値失敗と
してのセットアップ）。

RangeBreakoutEngine のような永続状態は持たない。各バーで確定済みMTデータ
から毎回ステートレスに評価する（trend_following.py と同じ設計思想）。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from minmax_fx_dt.strategy.support_resistance import zigzag_pivots_typed


@dataclass
class DoublePatternConfig:
    """パターン検出パラメータ."""

    zigzag_threshold_atr: float = 2.0  # スイングと認定する最小反転幅 (ATR倍数)
    pattern_tolerance_atr: float = 0.5  # 2つの山/谷が「ほぼ同水準」とみなす許容誤差 (ATR倍数)
    stop_buffer_atr: float = 0.1  # パターン無効化ストップに乗せる余白 (ATR倍数)
    max_bars_since_second_pivot: int = 20  # 2本目の山/谷確定からこのバー数を超えたら陳腐化とみなす


@dataclass
class DoublePatternSignal:
    """評価結果."""

    direction: str  # "UP" / "DOWN" / "NONE"
    neckline_price: float = 0.0
    pattern_extreme_price: float = 0.0  # ストップ算出の基準 (2つの山の高い方 / 谷の低い方)
    second_pivot_idx: int = -1


def evaluate_double_pattern_signal(
    mt_high: pd.Series,
    mt_low: pd.Series,
    atr_series: pd.Series,
    current_price: float,
    lt_direction: str,
    config: DoublePatternConfig | None = None,
) -> DoublePatternSignal:
    """直近の確定済みMT(H4)ピボット3点からダブルトップ/ボトムシグナルを評価する.

    Args:
        mt_high/mt_low: 中期足 (H4) の高値/安値。呼び出し側は確定済みバーのみ渡すこと。
        atr_series: mt_high/mt_lowと同じインデックスのATR系列。
        current_price: 現在値 (ネックラインブレイクの判定に使う、ST粒度で良い)。
        lt_direction: LTレイヤの方向判定 ("UP" / "DOWN" / "RANGE")。
        config: パターン検出設定。

    Returns:
        DoublePatternSignal。
    """
    if config is None:
        config = DoublePatternConfig()

    pivots = zigzag_pivots_typed(mt_high, mt_low, atr_series, config.zigzag_threshold_atr)
    if len(pivots) < 3:
        return DoublePatternSignal(direction="NONE")

    idx1, kind1 = pivots[-3]
    idx2, kind2 = pivots[-2]
    idx3, kind3 = pivots[-1]
    if kind1 != kind3 or kind1 == kind2:
        return DoublePatternSignal(direction="NONE")

    # 2本目の転換点(ネックラインを挟む谷/山)から現在までのバー数が離れすぎていたら陳腐化
    bars_since = (len(mt_high) - 1) - idx3
    if bars_since > config.max_bars_since_second_pivot:
        return DoublePatternSignal(direction="NONE")

    atr_at_neckline = atr_series.iloc[idx2]
    if pd.isna(atr_at_neckline) or atr_at_neckline <= 0:
        return DoublePatternSignal(direction="NONE")
    tolerance = config.pattern_tolerance_atr * float(atr_at_neckline)

    if kind1 == "HIGH" and lt_direction == "DOWN":
        p1 = float(mt_high.iloc[idx1])
        neckline = float(mt_low.iloc[idx2])
        p2 = float(mt_high.iloc[idx3])
        if abs(p1 - p2) <= tolerance and current_price < neckline:
            return DoublePatternSignal(
                direction="DOWN",
                neckline_price=neckline,
                pattern_extreme_price=max(p1, p2),
                second_pivot_idx=idx3,
            )

    if kind1 == "LOW" and lt_direction == "UP":
        p1 = float(mt_low.iloc[idx1])
        neckline = float(mt_high.iloc[idx2])
        p2 = float(mt_low.iloc[idx3])
        if abs(p1 - p2) <= tolerance and current_price > neckline:
            return DoublePatternSignal(
                direction="UP",
                neckline_price=neckline,
                pattern_extreme_price=min(p1, p2),
                second_pivot_idx=idx3,
            )

    return DoublePatternSignal(direction="NONE")


__all__ = ["DoublePatternConfig", "DoublePatternSignal", "evaluate_double_pattern_signal"]
