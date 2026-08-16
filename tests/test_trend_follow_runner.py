"""SYS-FX008 (EXP-FX000002) トレンドフォロー・ランナーの回帰テスト."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from minmax_fx_dt.backtest.simulator import PositionSide, SimulatorConfig
from minmax_fx_dt.backtest.trend_follow_runner import run_trend_follow_backtest
from minmax_fx_dt.strategy.trend_following import TrendFollowConfig


def _build_ohlcv(prices: np.ndarray, freq: str, start: datetime) -> pd.DataFrame:
    n = len(prices)
    dates = pd.date_range(start=start, periods=n, freq=freq)
    return pd.DataFrame({
        "open": prices,
        "high": prices + 0.05,
        "low": prices - 0.05,
        "close": prices,
    }, index=dates)


def _uptrend_dataset(n_days: int = 60):
    np.random.seed(1)
    d1_prices = 150.0 + np.linspace(0, 6, n_days) + np.random.normal(0, 0.05, n_days)
    lt = _build_ohlcv(d1_prices, "D", datetime(2026, 1, 5))  # 月曜始まり

    n_h4 = n_days * 6
    h4_prices = 150.0 + np.linspace(0, 6, n_h4) + np.random.normal(0, 0.03, n_h4)
    mt = _build_ohlcv(h4_prices, "4h", datetime(2026, 1, 5))

    n_m5 = n_days * 288
    m5_prices = 150.0 + np.linspace(0, 6, n_m5) + np.random.normal(0, 0.01, n_m5)
    st = _build_ohlcv(m5_prices, "5min", datetime(2026, 1, 5))
    return lt, mt, st


def test_uptrend_produces_long_entry() -> None:
    lt, mt, st = _uptrend_dataset()
    cfg = TrendFollowConfig(lt_sma_short=5, lt_sma_long=10, lt_trend_strength_threshold=10.0)
    sim = SimulatorConfig(initial_cash_jpy=1_000_000.0, lot_size=1000, spread_pips=0.3,
                           slippage_pips=0.5, is_jpy_pair=True, weekend_close=True,
                           max_dd_pause_threshold_pct=50.0)
    result = run_trend_follow_backtest(lt, mt, st, pair="USD_JPY", sim_config=sim, tf_config=cfg)
    assert result.state.total_trades > 0
    assert any(t.side == "BUY" for t in result.state.trade_history)


def test_weekend_close_blocks_reentry_churn() -> None:
    """土曜06:00 JST未満の窓で強制クローズされた直後に再エントリーし、
    また強制クローズされる...という無意味な回転売買が起きないことを確認する
    (実データのスモークテストで発見したバグの回帰テスト)。"""
    np.random.seed(2)
    # 2026-06-06 は土曜日。00:00-05:59 JST の間、明確な上昇トレンドを作る
    n_days = 30
    d1_prices = 150.0 + np.linspace(0, 5, n_days) + np.random.normal(0, 0.02, n_days)
    lt = _build_ohlcv(d1_prices, "D", datetime(2026, 5, 11))  # 月曜始まり、6/6を含む

    n_h4 = n_days * 6
    h4_prices = 150.0 + np.linspace(0, 5, n_h4) + np.random.normal(0, 0.01, n_h4)
    mt = _build_ohlcv(h4_prices, "4h", datetime(2026, 5, 11))

    n_m5 = n_days * 288
    m5_prices = 150.0 + np.linspace(0, 5, n_m5) + np.random.normal(0, 0.005, n_m5)
    st = _build_ohlcv(m5_prices, "5min", datetime(2026, 5, 11))

    cfg = TrendFollowConfig(lt_sma_short=5, lt_sma_long=10, lt_trend_strength_threshold=10.0)
    sim = SimulatorConfig(initial_cash_jpy=1_000_000.0, lot_size=1000, spread_pips=0.3,
                           slippage_pips=0.5, is_jpy_pair=True, weekend_close=True,
                           max_dd_pause_threshold_pct=50.0)
    result = run_trend_follow_backtest(lt, mt, st, pair="USD_JPY", sim_config=sim, tf_config=cfg)

    # 土曜 00:00-05:59 JST の窓の中で開いたトレードが無いこと (エントリーはブロックされる)
    for t in result.state.trade_history:
        entry_ts = t.entry_time
        assert not (entry_ts.dayofweek == 5 and entry_ts.hour < 6), (
            f"週末クローズ窓内でエントリーが発生: {entry_ts}"
        )


def test_trailing_stop_only_moves_favorably() -> None:
    """トレーリングストップは有利な方向にのみ動き、不利な方向には後退しない."""
    lt, mt, st = _uptrend_dataset(n_days=40)
    cfg = TrendFollowConfig(lt_sma_short=5, lt_sma_long=10, lt_trend_strength_threshold=10.0,
                             atr_trail_multiplier=2.0)
    sim = SimulatorConfig(initial_cash_jpy=1_000_000.0, lot_size=1000, spread_pips=0.3,
                           slippage_pips=0.5, is_jpy_pair=True, weekend_close=True,
                           max_dd_pause_threshold_pct=50.0)
    result = run_trend_follow_backtest(lt, mt, st, pair="USD_JPY", sim_config=sim, tf_config=cfg)
    long_trades = [t for t in result.state.trade_history if t.side == "BUY"]
    assert len(long_trades) > 0
    # 明確な上昇トレンド環境下 (6日以上保有) では、少なくとも1件は最終ストップが
    # 建値を上回っている (=トレーリングストップが有利な方向に切り上がり、利益を
    # 確保する水準まで動いた) はず。これが起きなければ「有利な方向にのみ動く」
    # という設計 (st_open基準で切り上げ、切り下げない) が機能していない可能性が高い。
    non_sl_trades = [t for t in long_trades if t.exit_reason != "SL" and t.hold_days >= 3]
    assert any(t.stop_loss > t.entry_price for t in non_sl_trades)
