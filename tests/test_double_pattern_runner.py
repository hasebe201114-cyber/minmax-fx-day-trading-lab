"""SYS-FX009 v2 (EXP-FX000003) ダブルパターン・ランナーの回帰テスト."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from minmax_fx_dt.backtest.double_pattern_runner import run_double_pattern_backtest
from minmax_fx_dt.backtest.simulator import SimulatorConfig
from minmax_fx_dt.strategy.double_pattern_strategy import DoublePatternStrategyConfig
from minmax_fx_dt.strategy.pattern_detection import DoublePatternConfig

# ダブルトップ形状 (山1=152 -> ネックライン=150 -> 山2=152 -> ネックライン割れ→147) を
# 経過時間(時間単位)の区分線形関数として定義する。H4/M5 双方をこの関数から生成することで、
# 粒度が違っても同じ形状を正確に追跡させる。
_SEG_BOUNDARIES_H = [0, 160, 240, 320, 380]  # 40本x4h, 20本x4h, 20本x4h, 15本x4h
_SEG_PRICES_TOP = [150.0, 152.0, 150.0, 152.0, 147.0]
_SEG_PRICES_BOTTOM = [150.0, 148.0, 150.0, 148.0, 153.0]


def _piecewise_price(elapsed_hours: float, seg_prices: list[float]) -> float:
    bounds = _SEG_BOUNDARIES_H
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if elapsed_hours <= hi:
            frac = (elapsed_hours - lo) / (hi - lo) if hi > lo else 0.0
            return seg_prices[i] + (seg_prices[i + 1] - seg_prices[i]) * frac
    return seg_prices[-1]


def _build_ohlcv_from_prices(prices: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "open": prices,
        "high": prices + 0.05,
        "low": prices - 0.05,
        "close": prices,
    }, index=index)


def _pattern_dataset(seg_prices: list[float], lt_direction: str, start: datetime):
    total_hours = _SEG_BOUNDARIES_H[-1]
    n_days = int(total_hours / 24) + 1

    d1_idx = pd.date_range(start=start, periods=n_days, freq="D")
    if lt_direction == "DOWN":
        d1_prices = 150.0 - np.linspace(0, 6, n_days)
    else:
        d1_prices = 150.0 + np.linspace(0, 6, n_days)
    lt = _build_ohlcv_from_prices(d1_prices, d1_idx)

    n_h4 = total_hours // 4
    h4_idx = pd.date_range(start=start, periods=n_h4, freq="4h")
    h4_prices = np.array([_piecewise_price(i * 4, seg_prices) for i in range(n_h4)])
    mt = _build_ohlcv_from_prices(h4_prices, h4_idx)

    n_m5 = total_hours * 12
    m5_idx = pd.date_range(start=start, periods=n_m5, freq="5min")
    m5_prices = np.array([_piecewise_price(i * 5 / 60, seg_prices) for i in range(n_m5)])
    st = _build_ohlcv_from_prices(m5_prices, m5_idx)

    return lt, mt, st


def _dp_config() -> DoublePatternStrategyConfig:
    return DoublePatternStrategyConfig(
        lt_sma_short=5,
        lt_sma_long=10,
        pattern=DoublePatternConfig(
            zigzag_threshold_atr=2.0,
            pattern_tolerance_atr=0.5,
            stop_buffer_atr=0.1,
            max_bars_since_second_pivot=20,
        ),
        atr_length=14,
        atr_trail_multiplier=2.0,
    )


def _sim_config() -> SimulatorConfig:
    return SimulatorConfig(
        initial_cash_jpy=1_000_000.0, lot_size=1000, spread_pips=0.3,
        slippage_pips=0.5, is_jpy_pair=True, weekend_close=True,
        max_dd_pause_threshold_pct=50.0,
    )


def _assert_no_weekend_entries(trade_history) -> None:
    for t in trade_history:
        entry_ts = t.entry_time
        assert not (entry_ts.dayofweek == 5 and entry_ts.hour < 6), (
            f"週末クローズ窓内でエントリーが発生: {entry_ts}"
        )


def test_double_top_pattern_produces_short_entry() -> None:
    """LT=DOWN + ダブルトップ(山1≒山2)のネックライン割れで SHORT エントリーが発生する."""
    lt, mt, st = _pattern_dataset(_SEG_PRICES_TOP, "DOWN", datetime(2026, 1, 5))
    result = run_double_pattern_backtest(
        lt, mt, st, pair="USD_JPY", sim_config=_sim_config(), dp_config=_dp_config()
    )
    assert result.state.total_trades > 0
    assert any(t.side == "SELL" for t in result.state.trade_history)
    _assert_no_weekend_entries(result.state.trade_history)


def test_double_bottom_pattern_produces_long_entry() -> None:
    """LT=UP + ダブルボトム(谷1≒谷2)のネックライン上抜けで LONG エントリーが発生する."""
    lt, mt, st = _pattern_dataset(_SEG_PRICES_BOTTOM, "UP", datetime(2026, 1, 5))
    result = run_double_pattern_backtest(
        lt, mt, st, pair="USD_JPY", sim_config=_sim_config(), dp_config=_dp_config()
    )
    assert result.state.total_trades > 0
    assert any(t.side == "BUY" for t in result.state.trade_history)
    _assert_no_weekend_entries(result.state.trade_history)


def test_stale_signal_does_not_churn_after_intrabar_crash() -> None:
    """H4確定バーが変わらない間に急落してcached_signal.stop_lossを下抜けた場合、
    陳腐化したシグナルでの新規エントリーを繰り返さないことを確認する.

    実データ(2024-07-11 BOJ介入時のUSD/JPY等)で発見された回帰バグ: cached_signal
    は次のH4確定バーまで再計算されない(絶対価格のstop_lossを保持したまま)ため、
    同一確定バー内で価格が急落してstop_lossを下抜けると、「建値時点で既にストップ
    状態」のポジションを毎M5バールから繰り返し開いてしまい、そのたびに(逆説的に)
    有利な価格でストップに掛かる回転売買が発生していた(3ペアで計84トレード、
    最大288,367円相当の見せかけの利益を生んでいた)。
    """
    lt, mt, st = _pattern_dataset(_SEG_PRICES_BOTTOM, "UP", datetime(2026, 1, 5))
    # 直近の確定H4バー(16:00)以降、価格がstop_loss(147.91付近)を大きく下回ったまま
    # 保持される「フラッシュクラッシュ」を、同一未確定バー内に注入する。
    crash_mask = st.index >= "2026-01-20 16:15:00"
    st.loc[crash_mask, ["open", "high", "low", "close"]] = [140.0, 140.05, 139.95, 140.0]

    result = run_double_pattern_backtest(
        lt, mt, st, pair="USD_JPY", sim_config=_sim_config(), dp_config=_dp_config()
    )
    # クラッシュ後の陳腐化シグナルでの回転売買がなければ、少数のトレードで収まるはず
    # (回帰前は同一ウィンドウ内で48本近くの繰り返しエントリーが発生していた)。
    assert result.state.total_trades <= 3


def test_target_reached_moves_stop_to_breakeven_or_better() -> None:
    """1:1ターゲット到達後は、ストップがブレイクイーブン以上に有利な水準へ切り替わる.

    ダブルトップ形状の最終区間 (ネックライン割れ後の下落) を延長し、1:1ターゲット
    到達後もトレンドが継続するデータで検証する。到達済みトレードは、エグジット価格
    が建値を超えて不利になっていない (ブレイクイーブン以上) ことを確認する。
    """
    extended_top = [150.0, 152.0, 150.0, 152.0, 140.0]  # ネックライン割れ後、大きく下落を継続
    lt, mt, st = _pattern_dataset(extended_top, "DOWN", datetime(2026, 1, 5))
    result = run_double_pattern_backtest(
        lt, mt, st, pair="USD_JPY", sim_config=_sim_config(), dp_config=_dp_config()
    )
    short_trades = [t for t in result.state.trade_history if t.side == "SELL"]
    assert len(short_trades) > 0
    target_reached_trades = [t for t in short_trades if t.target_reached]
    assert len(target_reached_trades) > 0
    for t in target_reached_trades:
        assert t.exit_price is not None
        assert t.exit_price <= t.entry_price, (
            f"1:1到達後のエグジットが建値より不利: entry={t.entry_price} exit={t.exit_price}"
        )
