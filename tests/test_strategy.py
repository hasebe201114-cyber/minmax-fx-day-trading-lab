"""SYS-FX007 戦略モジュールのスモークテスト.

合成データで S/R 検出 / ステートマシン / MTF 評価が動作することを確認。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from minmax_fx_dt.strategy.support_resistance import (
    SRLevel,
    cluster_fractals,
    count_touches,
    detect_fractals,
    detect_support_resistance,
    find_nearest_sr,
    zigzag_pivot_indices,
)
from minmax_fx_dt.strategy.range_breakout import (
    OrderBookSignal,
    PullbackSignal,
    RangeBreakoutEngine,
    State,
    hedged_position_state,
)
from minmax_fx_dt.strategy.multi_timeframe import MTFConfig, evaluate_mtf


# ---- フィクスチャ ----

@pytest.fixture
def synthetic_h4() -> tuple[pd.Series, pd.Series, pd.Series]:
    """H4 足風の合成データ (レンジ → 上限ブレイク)."""
    n = 200
    idx = pd.date_range(end=datetime(2025, 1, 1), periods=n, freq="4h")
    base = 100.0
    # 前半 150 本は 100 ± 2 のレンジ、後半 50 本は 102 に向かって上昇
    close = np.concatenate([
        base + 2 * np.sin(np.linspace(0, 6 * np.pi, 150)) + np.random.normal(0, 0.2, 150),
        np.linspace(102, 104, 50) + np.random.normal(0, 0.2, 50),
    ])
    high = close + np.abs(np.random.normal(0.3, 0.1, n))
    low = close - np.abs(np.random.normal(0.3, 0.1, n))
    return (
        pd.Series(high, index=idx),
        pd.Series(low, index=idx),
        pd.Series(close, index=idx),
    )


# ---- support_resistance テスト ----

def test_detect_fractals_basic() -> None:
    """フラクタル検出の基本動作."""
    idx = pd.date_range("2025-01-01", periods=10, freq="D")
    high = pd.Series([1, 2, 3, 5, 4, 3, 2, 4, 5, 3], index=idx)
    low = pd.Series([1, 1, 2, 3, 2, 1, 0, 2, 3, 1], index=idx)
    fractal_highs, fractal_lows = detect_fractals(high, low, window=5)
    # index 3 (value=5) が high の極大
    assert fractal_highs.iloc[3] is True or fractal_highs.iloc[3] == True
    # index 6 (value=0) が low の極小
    assert fractal_lows.iloc[6] is True or fractal_lows.iloc[6] == True


def test_count_touches() -> None:
    """接触カウントの基本動作."""
    idx = pd.date_range("2025-01-01", periods=10, freq="D")
    high = pd.Series([100, 105, 110, 102, 108, 100, 105, 110, 102, 108], index=idx)
    low = pd.Series([95, 100, 105, 95, 100, 95, 100, 105, 95, 100], index=idx)
    touches = count_touches(high, low, 105.0, tolerance_pct=0.01)
    # 105 ± 1.05 (99-111) の範囲で接触
    assert touches >= 4


def test_detect_support_resistance_returns_list(synthetic_h4: tuple) -> None:
    """S/R 検出がリストを返す."""
    high, low, close = synthetic_h4
    levels = detect_support_resistance(high, low, close, min_touches=2, fractal_window=5)
    assert isinstance(levels, list)
    for lv in levels:
        assert isinstance(lv, SRLevel)
        assert lv.kind in ("RESISTANCE", "SUPPORT")
        assert lv.strength >= 0.0
        assert lv.strength <= 1.0


def test_count_touches_tolerance_pct_is_a_fraction_not_a_percent_number() -> None:
    """count_touches() の tolerance_pct は「比率」(0.005=0.5%) を受け取る規約であり、
    cluster_threshold_pct と同じ「% 数値」(0.5=0.5%) を直接渡すと 100 倍広い許容誤差
    になることの確認 (OBS000007 追記8 で発見したバグの再現用)。"""
    idx = pd.date_range("2025-01-01", periods=300, freq="4h")
    # 100 から 200 へ単調に離れていく価格系列 (104.0 付近には序盤の数本だけが接触する)
    prices = np.linspace(100.0, 200.0, 300)
    high = pd.Series(prices + 0.3, index=idx)
    low = pd.Series(prices - 0.3, index=idx)

    correct_touches = count_touches(high, low, 104.0, tolerance_pct=0.5 / 100.0)
    buggy_touches = count_touches(high, low, 104.0, tolerance_pct=0.5)

    assert correct_touches < 10
    assert buggy_touches > 100  # 価格の 50% (52〜156) が全て「接触」扱いになる旧バグ
    assert correct_touches < buggy_touches / 10


def test_detect_support_resistance_does_not_inflate_touch_counts() -> None:
    """detect_support_resistance() が touch_tolerance_pct を count_touches() へ渡す際に
    比率へ変換 (/100) することの回帰テスト (OBS000007 追記8)。

    修正前は変換していなかったため、価格の 50% という許容誤差になり、ほぼ全バーが
    「接触」判定され touches が期間内バー数近くまで膨張していた
    (実測: USD/JPY Train 期間で全 32 ラインが一律 1609 回)。
    """
    idx = pd.date_range("2025-01-01", periods=500, freq="4h")
    rng = np.random.default_rng(42)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.3, 500))
    high = pd.Series(prices + np.abs(rng.normal(0.2, 0.05, 500)), index=idx)
    low = pd.Series(prices - np.abs(rng.normal(0.2, 0.05, 500)), index=idx)
    close = pd.Series(prices, index=idx)

    levels = detect_support_resistance(
        high, low, close,
        min_touches=1, fractal_window=5,
        cluster_threshold_pct=0.5, touch_tolerance_pct=0.5,
    )
    assert len(levels) > 0
    for lv in levels:
        # バグ時は touches が期間内バー数の大半 (500本中数百本) に達していた
        assert lv.touches < len(high) * 0.5


def test_find_nearest_sr() -> None:
    """最も近い S/R ラインの検索."""
    levels = [
        SRLevel(price=100.0, kind="SUPPORT", touches=5, first_touch=pd.Timestamp("2024-01-01"),
                last_touch=pd.Timestamp("2024-12-01"), strength=0.7, fractal_count=2),
        SRLevel(price=110.0, kind="RESISTANCE", touches=4, first_touch=pd.Timestamp("2024-02-01"),
                last_touch=pd.Timestamp("2024-11-01"), strength=0.6, fractal_count=2),
    ]
    nearest = find_nearest_sr(105.0, levels)
    assert nearest is not None
    assert nearest.price in (100.0, 110.0)


def test_zigzag_pivot_indices_ignores_noise_below_threshold() -> None:
    """OBS000006追記6: 閾値未満の微小反転はピボットとして検出されない."""
    n = 50
    # ATR=1.0 固定、閾値2.0xATR=2.0 未満の上下動のみ (ノイズ)
    high = pd.Series([100.0 + 0.3 * ((-1) ** i) for i in range(n)])
    low = high - 0.2
    atr = pd.Series([1.0] * n)
    pivots = zigzag_pivot_indices(high, low, atr, threshold_atr=2.0)
    assert pivots == []


def test_zigzag_pivot_indices_detects_swing_above_threshold() -> None:
    """閾値以上の明確なスイング (上昇→下降→上昇) はピボットとして検出される."""
    # 0->10 上昇 (10 pt) -> 10->0 下落 (10 pt) -> 0->10 上昇 (10 pt)、閾値 2.0 を大きく上回る
    up1 = list(range(0, 11))
    down1 = list(range(9, -1, -1))
    up2 = list(range(1, 11))
    prices = up1 + down1 + up2
    high = pd.Series([float(p) for p in prices])
    low = high - 0.1
    atr = pd.Series([1.0] * len(high))
    pivots = zigzag_pivot_indices(high, low, atr, threshold_atr=2.0)
    assert len(pivots) >= 2  # 少なくとも山と谷を検出


def test_zigzag_pivot_indices_empty_input() -> None:
    empty = pd.Series([], dtype=float)
    assert zigzag_pivot_indices(empty, empty, empty, threshold_atr=2.0) == []


# ---- range_breakout テスト ----

def test_engine_initial_state() -> None:
    """エンジンの初期ステート."""
    engine = RangeBreakoutEngine()
    assert engine.state == State.RANGE_FORMING


def test_engine_breakout_up() -> None:
    """上限ブレイクのステート遷移."""
    engine = RangeBreakoutEngine()
    engine.update_range(upper=105.0, lower=95.0, atr=0.5, ts=pd.Timestamp("2025-01-01"))
    engine.process_h4_bar(close=106.0, timestamp=pd.Timestamp("2025-01-02"))
    assert engine.state == State.RANGE_BREAKOUT_UP
    assert engine.last_breakout_level == 105.0
    assert engine.last_breakout_side == "UP"


def test_engine_breakout_down() -> None:
    """下限ブレイクのステート遷移."""
    engine = RangeBreakoutEngine()
    engine.update_range(upper=105.0, lower=95.0, atr=0.5, ts=pd.Timestamp("2025-01-01"))
    engine.process_h4_bar(close=94.0, timestamp=pd.Timestamp("2025-01-02"))
    assert engine.state == State.RANGE_BREAKOUT_DOWN
    assert engine.last_breakout_level == 95.0
    assert engine.last_breakout_side == "DOWN"


def test_engine_5_conditions_all_pass() -> None:
    """5 条件 AND すべて満たすとエントリー可."""
    engine = RangeBreakoutEngine()
    engine.update_range(upper=105.0, lower=95.0, atr=0.5, ts=pd.Timestamp("2025-01-01"))
    engine.update_sr_levels([
        SRLevel(
            price=105.0, kind="RESISTANCE", touches=5,
            first_touch=pd.Timestamp("2024-01-01"), last_touch=pd.Timestamp("2024-12-01"),
            strength=0.8, fractal_count=2,
        ),
    ])
    engine.update_lt_direction("UP")
    engine.update_pullback(
        PullbackSignal(confirmed=True, pattern="PIN_BAR", timestamp=pd.Timestamp("2025-01-03"))
    )
    engine.update_orderbook(
        OrderBookSignal(available=False, buy_thickness_ratio=1.0, sentiment_long_pct=50.0, passes=True)
    )
    # ブレイク
    engine.process_h4_bar(close=106.0, timestamp=pd.Timestamp("2025-01-02"))
    # エントリー判定
    sig = engine.check_entry_conditions(close=105.5, timestamp=pd.Timestamp("2025-01-03"))
    assert sig.should_enter is True
    assert sig.side == "BUY"
    assert sig.conditions_passed["1_lt_direction"] is True
    assert sig.conditions_passed["2_mt1_breakout"] is True
    assert sig.conditions_passed["3_mt2_sr_line"] is True
    assert sig.conditions_passed["4_mt3_orderbook"] is True
    assert sig.conditions_passed["5_st_pullback"] is True
    assert sig.stop_loss < sig.entry_price
    assert sig.take_profit > sig.entry_price


def test_engine_5_conditions_lt_mismatch() -> None:
    """LT 方向不一致でエントリー不可."""
    engine = RangeBreakoutEngine()
    engine.update_range(upper=105.0, lower=95.0, atr=0.5, ts=pd.Timestamp("2025-01-01"))
    engine.update_lt_direction("DOWN")  # ブレイクは UP だが LT は DOWN
    engine.update_pullback(
        PullbackSignal(confirmed=True, pattern="PIN_BAR", timestamp=pd.Timestamp("2025-01-03"))
    )
    engine.process_h4_bar(close=106.0, timestamp=pd.Timestamp("2025-01-02"))
    sig = engine.check_entry_conditions(close=105.5, timestamp=pd.Timestamp("2025-01-03"))
    assert sig.should_enter is False
    assert sig.conditions_passed["1_lt_direction"] is False


def test_hedged_position_margin_overflow() -> None:
    """証拠金消費率超過で両方クローズ."""
    result = hedged_position_state(
        long_entry=100.0,
        short_entry=100.0,
        current_price=101.0,
        margin_usage_pct=35.0,  # 上限 30% 超過
        margin_limit_pct=30.0,
    )
    assert result["should_close_long"] is True
    assert result["should_close_short"] is True
    assert "証拠金消費率" in str(result["reason"])


# ---- multi_timeframe テスト ----

def test_evaluate_mtf_synthetic() -> None:
    """合成 MTF データで evaluate_mtf が動作."""
    np.random.seed(42)
    n_d1 = 250
    n_h4 = n_d1 * 6
    n_m15 = n_h4 * 16

    d1_idx = pd.date_range(end=datetime(2025, 1, 1), periods=n_d1, freq="D")
    h4_idx = pd.date_range(end=datetime(2025, 1, 1), periods=n_h4, freq="4h")
    m15_idx = pd.date_range(end=datetime(2025, 1, 1), periods=n_m15, freq="15min")

    d1_close = pd.Series(100.0 + np.cumsum(np.random.normal(0, 0.5, n_d1)), index=d1_idx)
    h4_close = pd.Series(100.0 + np.cumsum(np.random.normal(0, 0.2, n_h4)), index=h4_idx)
    m15_close = pd.Series(100.0 + np.cumsum(np.random.normal(0, 0.1, n_m15)), index=m15_idx)

    h4_high = h4_close + np.abs(np.random.normal(0.3, 0.1, n_h4))
    h4_low = h4_close - np.abs(np.random.normal(0.3, 0.1, n_h4))
    m15_high = m15_close + np.abs(np.random.normal(0.15, 0.05, n_m15))
    m15_low = m15_close - np.abs(np.random.normal(0.15, 0.05, n_m15))
    d1_high = d1_close + np.abs(np.random.normal(0.5, 0.2, n_d1))
    d1_low = d1_close - np.abs(np.random.normal(0.5, 0.2, n_d1))

    result = evaluate_mtf(
        lt_high=pd.Series(d1_high, index=d1_idx),
        lt_low=pd.Series(d1_low, index=d1_idx),
        lt_close=d1_close,
        mt_high=pd.Series(h4_high, index=h4_idx),
        mt_low=pd.Series(h4_low, index=h4_idx),
        mt_close=h4_close,
        st_high=pd.Series(m15_high, index=m15_idx),
        st_low=pd.Series(m15_low, index=m15_idx),
        st_close=m15_close,
    )
    assert result.lt_direction in ("UP", "DOWN", "RANGE")
    assert isinstance(result.sr_levels_count, int)
    assert result.engine.state in [s for s in State]
    assert isinstance(result.entry_signal.conditions_passed, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
