"""SYS-FX008 トレンドフォロー・MAクロス戦略 (EXP-FX000002).

SYS-FX007 (レンジブレイク・プルバック) より単純な2レイヤ構造:
    LT (D1): SMA短期/長期クロス + トレンド強度フィルター (ADX) で方向判定
    MT (H4): LT方向とH4終値の位置関係で継続確認 (ノイズによる単発クロスの排除)

RangeBreakoutEngine のような状態機械は使わない。各バーの確定済みLT/MTデータ
だけから毎回ステートレスに評価する (`classify_lt_direction` と同じ設計思想)。
「待機中の状態」を持たないため、SYS-FX007で問題になった「エンジン状態が
バーをまたいで永続化されない」「同一確定バーへの重複処理」という類のバグは
構造的に発生しない。

決済はトレーリングストップ (ATR比例) または LT 方向反転で行う。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from minmax_fx_dt.strategy.indicators import adx as adx_ind
from minmax_fx_dt.strategy.indicators import atr as atr_ind
from minmax_fx_dt.strategy.indicators import classify_lt_direction
from minmax_fx_dt.strategy.indicators import sma as sma_ind


@dataclass
class TrendFollowConfig:
    """SYS-FX008 パラメータ (00-spec.md v1.1、Trainデータから導出済み)."""

    lt_sma_short: int = 10
    lt_sma_long: int = 20
    lt_trend_strength_threshold: float = 40.0  # ADX(14, D1) 閾値、通貨ペア別に導出
    mt_confirm_sma_length: int = 20  # H4継続確認用SMA (固定値、追加のデータ駆動導出はしない)
    atr_length: int = 14
    atr_trail_multiplier: float = 2.0  # トレーリングストップ幅 (ATR倍数)


@dataclass
class TrendFollowSignal:
    """評価結果."""

    direction: str  # "UP" / "DOWN" / "NONE" (エントリー方向、MT確認込み)
    lt_direction: str  # "UP" / "DOWN" / "RANGE" (LTレイヤ単体の判定)
    mt_confirmed_direction: str  # "UP" / "DOWN" / "NONE" (MTレイヤ単体の判定)
    atr_value: float  # LT (D1) ATR の直近値 (トレーリングストップ幅の算出用)


def evaluate_trend_follow(
    lt_high: pd.Series,
    lt_low: pd.Series,
    lt_close: pd.Series,
    mt_close: pd.Series,
    config: TrendFollowConfig | None = None,
) -> TrendFollowSignal:
    """LT (D1) + MT (H4) の確定済みバーからトレンドフォローシグナルを評価する.

    Args:
        lt_high/low/close: 長期足 (D1) の OHLC。呼び出し側は確定済みバーのみ渡すこと。
        mt_close: 中期足 (H4) の終値。同上。
        config: パラメータ設定 (None ならデフォルト)。

    Returns:
        TrendFollowSignal。
    """
    if config is None:
        config = TrendFollowConfig()

    sma_short = sma_ind(lt_close, config.lt_sma_short)
    sma_long = sma_ind(lt_close, config.lt_sma_long)
    adx_df = adx_ind(lt_high, lt_low, lt_close, length=14)
    adx_value = adx_df.iloc[:, 0]
    lt_direction_series = classify_lt_direction(
        lt_close, sma_short, sma_long, adx_value, config.lt_trend_strength_threshold
    )
    lt_direction = str(lt_direction_series.iloc[-1])

    mt_sma = sma_ind(mt_close, config.mt_confirm_sma_length)
    mt_confirmed_direction = "NONE"
    if not mt_sma.empty and not pd.isna(mt_sma.iloc[-1]):
        if mt_close.iloc[-1] > mt_sma.iloc[-1]:
            mt_confirmed_direction = "UP"
        elif mt_close.iloc[-1] < mt_sma.iloc[-1]:
            mt_confirmed_direction = "DOWN"

    direction = "NONE"
    if lt_direction == "UP" and mt_confirmed_direction == "UP":
        direction = "UP"
    elif lt_direction == "DOWN" and mt_confirmed_direction == "DOWN":
        direction = "DOWN"

    atr_series = atr_ind(lt_high, lt_low, lt_close, length=config.atr_length)
    atr_value = (
        float(atr_series.iloc[-1])
        if not atr_series.empty and not pd.isna(atr_series.iloc[-1])
        else 0.0
    )

    return TrendFollowSignal(
        direction=direction,
        lt_direction=lt_direction,
        mt_confirmed_direction=mt_confirmed_direction,
        atr_value=atr_value,
    )


def lt_direction_only(
    lt_close: pd.Series,
    config: TrendFollowConfig | None = None,
) -> str:
    """トレンド強度フィルターを無視した、SMAクロスのみによる方向判定.

    ポジション保有中の「反転クロスでの手仕舞い」判定に使う (spec §決済)。
    フィルター込みのlt_directionでは "RANGE" に落ちた瞬間に手仕舞いしてしまい
    トレーリングストップより先に大半のトレードが終わってしまうため、
    決済判定には純粋なSMAクロスの向きだけを見る。
    """
    if config is None:
        config = TrendFollowConfig()
    sma_short = sma_ind(lt_close, config.lt_sma_short)
    sma_long = sma_ind(lt_close, config.lt_sma_long)
    if sma_short.empty or pd.isna(sma_short.iloc[-1]) or pd.isna(sma_long.iloc[-1]):
        return "NONE"
    if sma_short.iloc[-1] > sma_long.iloc[-1]:
        return "UP"
    if sma_short.iloc[-1] < sma_long.iloc[-1]:
        return "DOWN"
    return "NONE"


__all__ = [
    "TrendFollowConfig",
    "TrendFollowSignal",
    "evaluate_trend_follow",
    "lt_direction_only",
]
