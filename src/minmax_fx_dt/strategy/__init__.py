"""戦略ロジック.

SYS-FX007 レンジブレイク・プルバック戦略 v2 の構成:
    indicators           - テクニカル指標 (手書き実装、pandas-ta 不要)
    support_resistance   - レジサポライン検出 (v2 中核、Williams Fractal + クラスタリング)
    range_breakout       - レンジブレイク・ステートマシン (5 条件 AND エントリー判定)
    multi_timeframe      - 3 レイヤ MTF フレームワーク (LT/MT/ST 統合評価)
"""

from minmax_fx_dt.strategy.indicators import (
    adx,
    atr,
    bbands,
    classify_lt_direction,
    donchian,
    ema,
    rsi,
    sma,
)
from minmax_fx_dt.strategy.support_resistance import (
    SRLevel,
    cluster_fractals,
    count_touches,
    detect_fractals,
    detect_support_resistance,
    filter_by_strength,
    find_nearest_sr,
)
from minmax_fx_dt.strategy.range_breakout import (
    EntrySignal,
    OrderBookSignal,
    PullbackSignal,
    RangeBreakoutEngine,
    State,
    hedged_position_state,
)
from minmax_fx_dt.strategy.multi_timeframe import MTFConfig, MTFResult, evaluate_mtf

__all__ = [
    # indicators (pandas-ta 不要、手書き実装)
    "sma", "ema", "atr", "donchian", "bbands", "adx", "rsi",
    "classify_lt_direction",
    # support_resistance
    "SRLevel", "detect_fractals", "cluster_fractals", "count_touches",
    "detect_support_resistance", "find_nearest_sr", "filter_by_strength",
    # range_breakout
    "State", "RangeBreakoutEngine", "EntrySignal", "PullbackSignal",
    "OrderBookSignal", "hedged_position_state",
    # multi_timeframe
    "MTFConfig", "MTFResult", "evaluate_mtf",
]
