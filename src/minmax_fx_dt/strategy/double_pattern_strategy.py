"""SYS-FX009 v2 上位足トレンド + ダブルトップ/ボトム戦略 (EXP-FX000003).

LT (D1) の方向判定 (SMAクロスのみ、SYS-FX008 `trend_following.lt_direction_only()` と
同じロジック) と MT (H4) のダブルトップ/ボトム検出 (`pattern_detection.evaluate_double_pattern_signal`)
を組み合わせ、エントリー価格・初期ストップ・初期リスク幅 (1:1ターゲット算出用) を
一括評価する。

trend_following.py と同じくステートレス設計。ただし「1:1到達後にブレイクイーブン+
ATRトレーリングへ切替」はポジションの含み状態 (エントリー以降にターゲットへ到達済みか)
を要するため、ここでは扱わずランナー側 (backtest/double_pattern_runner.py) で管理する。
このモジュールは「新規エントリーすべきか」の判定にのみ責務を持つ。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from minmax_fx_dt.strategy.indicators import atr as atr_ind
from minmax_fx_dt.strategy.indicators import sma as sma_ind
from minmax_fx_dt.strategy.pattern_detection import (
    DoublePatternConfig,
    evaluate_double_pattern_signal,
)


@dataclass
class DoublePatternStrategyConfig:
    """SYS-FX009 v2 パラメータ (00-spec.md、Trainデータから導出予定)."""

    lt_sma_short: int = 10
    lt_sma_long: int = 20
    pattern: DoublePatternConfig = field(default_factory=DoublePatternConfig)
    atr_length: int = 14  # MT(H4) ATR期間。ストップ算出・1:1到達後のトレーリング共通
    atr_trail_multiplier: float = 2.0  # 1:1到達後のトレーリングストップ幅 (ATR倍数)


@dataclass
class DoublePatternEntrySignal:
    """評価結果 (新規エントリー可否 + エントリー時のストップ算出情報)."""

    direction: str  # "UP" / "DOWN" / "NONE"
    lt_direction: str  # "UP" / "DOWN" / "NONE" (LTレイヤ単体の判定)
    entry_price: float = 0.0  # 現在値 (成行想定、ネックラインブレイク済み)
    stop_loss: float = 0.0  # パターン無効化水準 + バッファ
    initial_risk: float = 0.0  # |entry_price - stop_loss| (1:1ターゲット算出用)
    atr_value: float = 0.0  # MT ATR 直近値 (1:1到達後のトレーリング幅算出用)


def lt_direction_only(lt_close: pd.Series, config: DoublePatternStrategyConfig) -> str:
    """SMAクロスのみによるLT方向判定 (トレンド強度フィルターなし).

    パターン確認自体がエントリー精度のフィルターとして機能する設計のため
    (00-prescreen.md)、ADXによるトレンド強度フィルターは重ねない。
    """
    sma_short = sma_ind(lt_close, config.lt_sma_short)
    sma_long = sma_ind(lt_close, config.lt_sma_long)
    if sma_short.empty or pd.isna(sma_short.iloc[-1]) or pd.isna(sma_long.iloc[-1]):
        return "NONE"
    if sma_short.iloc[-1] > sma_long.iloc[-1]:
        return "UP"
    if sma_short.iloc[-1] < sma_long.iloc[-1]:
        return "DOWN"
    return "NONE"


def evaluate_double_pattern_entry(
    lt_close: pd.Series,
    mt_high: pd.Series,
    mt_low: pd.Series,
    mt_close: pd.Series,
    current_price: float,
    config: DoublePatternStrategyConfig | None = None,
) -> DoublePatternEntrySignal:
    """LT (D1) + MT (H4) の確定済みバーからダブルトップ/ボトム・エントリーシグナルを評価する.

    Args:
        lt_close: 長期足 (D1) の終値。呼び出し側は確定済みバーのみ渡すこと。
        mt_high/low/close: 中期足 (H4) の OHLC。同上。
        current_price: 現在値 (ネックラインブレイクの判定に使う、ST粒度で良い)。
        config: パラメータ設定 (None ならデフォルト)。

    Returns:
        DoublePatternEntrySignal。
    """
    if config is None:
        config = DoublePatternStrategyConfig()

    lt_dir = lt_direction_only(lt_close, config)
    if lt_dir not in ("UP", "DOWN"):
        return DoublePatternEntrySignal(direction="NONE", lt_direction=lt_dir)

    atr_series = atr_ind(mt_high, mt_low, mt_close, length=config.atr_length)
    if atr_series.empty or pd.isna(atr_series.iloc[-1]) or atr_series.iloc[-1] <= 0:
        return DoublePatternEntrySignal(direction="NONE", lt_direction=lt_dir)
    atr_value = float(atr_series.iloc[-1])

    pattern_signal = evaluate_double_pattern_signal(
        mt_high, mt_low, atr_series, current_price, lt_dir, config.pattern
    )
    if pattern_signal.direction == "NONE":
        return DoublePatternEntrySignal(direction="NONE", lt_direction=lt_dir, atr_value=atr_value)

    buffer = config.pattern.stop_buffer_atr * atr_value
    if pattern_signal.direction == "DOWN":
        stop_loss = pattern_signal.pattern_extreme_price + buffer
    else:
        stop_loss = pattern_signal.pattern_extreme_price - buffer
    initial_risk = abs(current_price - stop_loss)
    if initial_risk <= 0:
        return DoublePatternEntrySignal(direction="NONE", lt_direction=lt_dir, atr_value=atr_value)

    return DoublePatternEntrySignal(
        direction=pattern_signal.direction,
        lt_direction=lt_dir,
        entry_price=current_price,
        stop_loss=stop_loss,
        initial_risk=initial_risk,
        atr_value=atr_value,
    )


__all__ = [
    "DoublePatternStrategyConfig",
    "DoublePatternEntrySignal",
    "lt_direction_only",
    "evaluate_double_pattern_entry",
]
