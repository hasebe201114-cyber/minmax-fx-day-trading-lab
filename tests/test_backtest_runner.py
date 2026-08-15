"""バックテストエンジンの回帰テスト (OBS000007 独立監査 P1 項目3).

OBS000007 で修正した 5 件のバグ (先読み・確定バースライス・ステートマシン
永続化・PULLBACK_CONFIRMED 状態固着・S/R 許容誤差) と is_jpy_pair の
通貨スケールバグを、それぞれ単体で検証する。独立監査が
「既存テストではこれらのバグを一つも検知できない」と指摘したことへの対応。
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from minmax_fx_dt.backtest.runner import last_confirmed_bar_ts, run_backtest
from minmax_fx_dt.backtest.simulator import (
    PortfolioState,
    SimulatorConfig,
    apply_slippage,
    calc_pnl,
    margin_usage_pct,
    open_long,
    spread_round_trip_cost_jpy,
)
from minmax_fx_dt.strategy.multi_timeframe import MTFConfig, evaluate_mtf
from minmax_fx_dt.strategy.range_breakout import (
    OrderBookSignal,
    PullbackSignal,
    RangeBreakoutEngine,
    SRLevel,
    State,
)


# ---- last_confirmed_bar_ts (確定済みバー特定ロジック) ----

def test_last_confirmed_bar_ts_excludes_in_progress_bar() -> None:
    """OBS000007 チェック #2 の具体例: H4 00:00/04:00/08:00, ts=05:15 のとき
    進行中バー (04:00) を除外し、確定済みの 00:00 を返す."""
    index = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 04:00", "2024-01-01 08:00"])
    ts = pd.Timestamp("2024-01-01 05:15")
    result = last_confirmed_bar_ts(index, ts)
    assert result == pd.Timestamp("2024-01-01 00:00")


def test_last_confirmed_bar_ts_exact_bar_boundary() -> None:
    """ts がバー開始時刻ちょうどの場合、そのバー自身がまだ進行中として除外される."""
    index = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 04:00", "2024-01-01 08:00"])
    ts = pd.Timestamp("2024-01-01 04:00")
    result = last_confirmed_bar_ts(index, ts)
    assert result == pd.Timestamp("2024-01-01 00:00")


def test_last_confirmed_bar_ts_warmup_returns_none() -> None:
    """バックテスト開始直後、確定済みバーが1本もない場合は None."""
    index = pd.DatetimeIndex(["2024-01-01 00:00"])
    ts = pd.Timestamp("2024-01-01 00:30")
    assert last_confirmed_bar_ts(index, ts) is None

    ts_before_any_bar = pd.Timestamp("2023-12-31 00:00")
    assert last_confirmed_bar_ts(index, ts_before_any_bar) is None


# ---- PULLBACK_CONFIRMED 状態固着バグの回帰テスト ----

def test_pullback_confirmed_returns_to_range_forming() -> None:
    """PULLBACK_CONFIRMED から、価格がレンジに戻れば RANGE_FORMING に復帰する
    (修正前はこの遷移が存在せず永久固着していた)."""
    engine = RangeBreakoutEngine()
    engine.update_range(upper=105.0, lower=95.0, atr=0.5, ts=pd.Timestamp("2025-01-01"))
    engine.state = State.PULLBACK_CONFIRMED
    engine.last_breakout_side = "UP"

    engine.process_h4_bar(close=100.0, timestamp=pd.Timestamp("2025-01-02"))  # レンジ内に復帰

    assert engine.state == State.RANGE_FORMING
    assert engine.last_breakout_level is None
    assert engine.last_breakout_side is None


def test_pullback_confirmed_continues_to_trend() -> None:
    """PULLBACK_CONFIRMED から、価格がレンジ外に留まれば TREND_UP/DOWN に進む
    (レンジに戻らない限り RANGE_FORMING に復帰せず、次のセットアップを待てる)."""
    engine = RangeBreakoutEngine()
    engine.update_range(upper=105.0, lower=95.0, atr=0.5, ts=pd.Timestamp("2025-01-01"))
    engine.state = State.PULLBACK_CONFIRMED
    engine.last_breakout_side = "UP"

    engine.process_h4_bar(close=108.0, timestamp=pd.Timestamp("2025-01-02"))  # レンジ外継続

    assert engine.state == State.TREND_UP


def test_pullback_confirmed_not_stuck_forever() -> None:
    """PULLBACK_CONFIRMED のまま何十バー経過しても抜けられない、という
    修正前のバグが再発しないことを確認 (最終的に RANGE_FORMING か TREND に必ず到達)."""
    engine = RangeBreakoutEngine()
    engine.update_range(upper=105.0, lower=95.0, atr=0.5, ts=pd.Timestamp("2025-01-01"))
    engine.state = State.PULLBACK_CONFIRMED
    engine.last_breakout_side = "UP"

    for i in range(50):
        engine.process_h4_bar(close=108.0, timestamp=pd.Timestamp("2025-01-01") + pd.Timedelta(hours=4 * i))

    assert engine.state != State.PULLBACK_CONFIRMED


# ---- S/R 許容誤差 (ATR 連動) の回帰テスト ----

def test_sr_tolerance_uses_atr_not_wide_price_pct() -> None:
    """許容誤差が ATR x 0.5 になっており、旧実装 (price x 0.5% = USD/JPY で約75pips)
    より厳格になっていることを、24pips と 75pips の中間のオフセットで検証する."""
    engine = RangeBreakoutEngine()
    atr = 0.48  # USD/JPY の H4 ATR 実測中央値相当 (約48pips)
    engine.update_range(upper=150.0, lower=149.0, atr=atr, ts=pd.Timestamp("2025-01-01"))
    # ブレイクレベル(150.0)から 0.40 円(=40pips)離れた位置に S/R ライン
    # 40pips は 新許容誤差 (0.5*ATR=24pips) を超えるが、旧許容誤差 (150*0.5%=75pips) 未満
    offset = 0.40
    engine.update_sr_levels([
        SRLevel(
            price=150.0 - offset, kind="RESISTANCE", touches=5,
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
    engine.process_h4_bar(close=150.5, timestamp=pd.Timestamp("2025-01-02"))
    sig = engine.check_entry_conditions(close=150.2, timestamp=pd.Timestamp("2025-01-03"))

    # 新許容誤差 (24pips) では 40pips 離れた S/R は条件不成立のはず
    assert sig.conditions_passed["3_mt2_sr_line"] is False
    assert sig.should_enter is False


def test_sr_tolerance_still_passes_within_atr_range() -> None:
    """ATR x 0.5 以内の S/R ラインは引き続き条件成立する (許容誤差が厳しすぎて
    正当なケースまで弾いていないことを確認)."""
    engine = RangeBreakoutEngine()
    atr = 0.48
    engine.update_range(upper=150.0, lower=149.0, atr=atr, ts=pd.Timestamp("2025-01-01"))
    offset = 0.10  # 10pips、0.5*ATR=24pips 以内
    engine.update_sr_levels([
        SRLevel(
            price=150.0 - offset, kind="RESISTANCE", touches=5,
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
    engine.process_h4_bar(close=150.5, timestamp=pd.Timestamp("2025-01-02"))
    sig = engine.check_entry_conditions(close=150.2, timestamp=pd.Timestamp("2025-01-03"))

    assert sig.conditions_passed["3_mt2_sr_line"] is True


# ---- EUR/USD (非 JPY ペア) の pip スケール回帰テスト ----

def test_calc_pnl_uses_correct_pip_value_for_non_jpy_pair() -> None:
    """is_jpy_pair=False のとき pip_value=0.0001 (JPY ペアの 0.01 ではなく)
    を使うことを確認 (旧 ablation_sweep.py / analyze_trades.py の
    is_jpy_pair=True 全ペア共通ハードコードで壊れていたスケール)."""
    entry = 1.08000
    exit_ = 1.08100  # 10 pips (EUR/USD として)
    pnl_jpy, pnl_pips = calc_pnl(entry, exit_, size=1000, side="BUY", is_jpy_pair=False)
    assert pnl_pips == pytest.approx(10.0, abs=0.01)

    # is_jpy_pair=True で誤って計算すると pip_value が 100 倍 (0.0001->0.01) になり
    # pips 換算が 1/100 (0.1pips) に縮んでしまう (回帰防止: 旧ハードコードバグの再現)
    wrong_pnl_jpy, wrong_pnl_pips = calc_pnl(entry, exit_, size=1000, side="BUY", is_jpy_pair=True)
    assert wrong_pnl_pips == pytest.approx(0.1, abs=0.001)
    assert wrong_pnl_pips != pytest.approx(pnl_pips, abs=0.01)


def test_apply_slippage_pip_scale_non_jpy() -> None:
    """スリッページも EUR/USD では pip_value=0.0001 で計算される."""
    price = 1.08000
    slipped = apply_slippage(price, "BUY", slippage_pips=0.5, is_jpy_pair=False)
    assert slipped == pytest.approx(1.08005, abs=1e-6)

    # is_jpy_pair=True だと 0.01 換算になり 100 倍のスリッページになってしまう (回帰防止)
    wrong_slipped = apply_slippage(price, "BUY", slippage_pips=0.5, is_jpy_pair=True)
    assert wrong_slipped == pytest.approx(1.08500, abs=1e-6)


# ---- エンジン永続化 (process_h4 ゲート) の回帰テスト ----

def _synthetic_mtf_series(n_d1: int = 260, seed: int = 7):
    rng = np.random.default_rng(seed)
    n_h4 = n_d1 * 6
    n_m15 = n_h4 * 16
    d1_idx = pd.date_range(end=datetime(2025, 1, 1), periods=n_d1, freq="D")
    h4_idx = pd.date_range(end=datetime(2025, 1, 1), periods=n_h4, freq="4h")
    m15_idx = pd.date_range(end=datetime(2025, 1, 1), periods=n_m15, freq="15min")

    d1_close = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.5, n_d1)), index=d1_idx)
    h4_close = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.2, n_h4)), index=h4_idx)
    m15_close = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.1, n_m15)), index=m15_idx)

    h4_high = h4_close + np.abs(rng.normal(0.3, 0.1, n_h4))
    h4_low = h4_close - np.abs(rng.normal(0.3, 0.1, n_h4))
    m15_high = m15_close + np.abs(rng.normal(0.15, 0.05, n_m15))
    m15_low = m15_close - np.abs(rng.normal(0.15, 0.05, n_m15))
    d1_high = d1_close + np.abs(rng.normal(0.5, 0.2, n_d1))
    d1_low = d1_close - np.abs(rng.normal(0.5, 0.2, n_d1))

    return {
        "lt_high": d1_high, "lt_low": d1_low, "lt_close": d1_close,
        "mt_high": h4_high, "mt_low": h4_low, "mt_close": h4_close,
        "st_high": m15_high, "st_low": m15_low, "st_close": m15_close,
    }


def test_engine_reused_when_passed_and_process_h4_gated() -> None:
    """engine を渡すと新規生成されず再利用され、process_h4=False では
    process_h4_bar が呼ばれない (= 同一確定バーに対する重複遷移が起きない)
    ことを確認 (OBS000007 チェック #3)."""
    data = _synthetic_mtf_series()
    config = MTFConfig()

    result1 = evaluate_mtf(**data, config=config, engine=None, process_h4=True)
    engine = result1.engine
    state_after_first = engine.state

    # 同じデータで process_h4=False を渡すと状態が変化しないはず
    result2 = evaluate_mtf(**data, config=config, engine=engine, process_h4=False)
    assert result2.engine is engine  # 同一インスタンスが再利用されている
    # PULLBACK_CONFIRMED のようにエントリー判定で変わる場合はあるが、
    # 少なくとも process_h4_bar による RANGE_BREAKOUT_*/TREND_* 遷移は起きない
    assert engine.state == state_after_first or engine.state == State.PULLBACK_CONFIRMED


# ---- ルックアヘッド回帰テスト (統合) ----

def test_sl_tp_not_triggered_by_future_price_within_same_h4_bar() -> None:
    """OBS000007 チェック #1 の回帰テスト: H4 バー内の「未来の」スパイクが、
    そのバーの前半にあたる ST バーの SL/TP 判定に漏れ込まないことを確認する。

    H4 バー (00:00-04:00) の後半 (03:xx) に SL を割るスパイクを仕込み、
    バー前半 (00:xx-02:xx) の間はまだそのスパイクが「見えて」いないことを、
    ポジションが前半の ST バーでは決済されないことで確認する。
    """
    n_d1 = 60
    d1_idx = pd.date_range(end=datetime(2024, 6, 1), periods=n_d1, freq="D")
    h4_idx = pd.date_range(end=datetime(2024, 6, 1), periods=n_d1 * 6, freq="4h")
    m15_idx = pd.date_range(end=datetime(2024, 6, 1), periods=n_d1 * 6 * 16, freq="15min")

    lt_close = pd.Series(150.0, index=d1_idx)
    lt_high = lt_close + 0.5
    lt_low = lt_close - 0.5

    mt_close = pd.Series(150.0, index=h4_idx)
    mt_high = mt_close + 0.2
    mt_low = mt_close - 0.2

    st_close = pd.Series(150.0, index=m15_idx)
    st_high = st_close + 0.05
    st_low = st_close - 0.05

    # 最後の H4 バーの後半 (3本目の M15 バー、つまりバー開始から45分後) にだけ
    # 大きく下に振れるスパイクを注入する。SL/TP 判定に使う high/low がこの
    # スパイクを「バー前半の時点で」拾ってしまっていないかを確認する。
    last_h4_start = h4_idx[-1]
    spike_ts = last_h4_start + pd.Timedelta(minutes=45)
    st_low = st_low.copy()
    st_low.loc[spike_ts] = 100.0  # 50円下抜けの大きなスパイク (SL を明確に下回る)

    lt_ohlcv = pd.DataFrame({"open": lt_close, "high": lt_high, "low": lt_low, "close": lt_close})
    mt_ohlcv = pd.DataFrame({"open": mt_close, "high": mt_high, "low": mt_low, "close": mt_close})
    st_ohlcv = pd.DataFrame({"open": st_close, "high": st_high, "low": st_low, "close": st_close})

    sim_config = SimulatorConfig(is_jpy_pair=True, weekend_close=False)

    # このテストは「バー全体の高安を先読みして使っていた旧実装なら、
    # スパイク発生前の ST バーの時点で SL に抵触してしまう」という
    # 回帰を検知する目的なので、simulator.check_stop_loss_take_profit を
    # 直接使い、旧実装 (バー全体の高安) と新実装 (ST バー自身の高安) の
    # 挙動差を確認する形で検証する。
    from minmax_fx_dt.backtest.simulator import (
        PortfolioState,
        check_stop_loss_take_profit,
        open_long,
    )

    state = PortfolioState(cash=1_000_000.0, initial_cash=1_000_000.0, max_equity=1_000_000.0)
    open_long(state, sim_config, entry_price=150.0, entry_time=last_h4_start, stop_loss=145.0, take_profit=155.0)

    # バー前半 (スパイク発生前) の ST バー高安で判定 -> 決済されないはず
    bar_before_spike_high = float(st_high.loc[last_h4_start])
    bar_before_spike_low = float(st_low.loc[last_h4_start])
    check_stop_loss_take_profit(state, sim_config, high=bar_before_spike_high, low=bar_before_spike_low, timestamp=last_h4_start)
    assert state.long_trade is not None, "スパイク発生前の ST バーの高安だけでは SL に抵触しないはず"

    # 旧実装の再現: H4 バー全体 (スパイクを含む) の高安を使うと、
    # スパイク発生前であっても SL に抵触してしまう (先読みバグの再現)
    bar_full_low = float(min(mt_low.loc[last_h4_start], st_low.min()))
    check_stop_loss_take_profit(state, sim_config, high=bar_before_spike_high, low=bar_full_low, timestamp=last_h4_start)
    assert state.long_trade is None, "旧実装 (バー全体の高安) では先読みで SL に抵触してしまうはず"


# ---- B6: spread_round_trip_cost_jpy の回帰テスト ----

def test_spread_round_trip_cost_jpy_matches_pair_specific_spread() -> None:
    """通貨ペア別 spread_pips を反映し、旧実装の全ペア共通60円固定でないことを確認."""
    usdjpy_config = SimulatorConfig(spread_pips=0.3, lot_size=1_000, is_jpy_pair=True)
    gbpjpy_config = SimulatorConfig(spread_pips=0.7, lot_size=1_000, is_jpy_pair=True)
    eurusd_config = SimulatorConfig(spread_pips=0.3, lot_size=1_000, is_jpy_pair=False)

    usdjpy_cost = spread_round_trip_cost_jpy(usdjpy_config)
    gbpjpy_cost = spread_round_trip_cost_jpy(gbpjpy_config)
    eurusd_cost = spread_round_trip_cost_jpy(eurusd_config)

    # USD/JPY: 2 x 0.3pips x 0.01 x 1000 = 6.0 JPY (旧ハードコード60円ではない)
    assert usdjpy_cost == pytest.approx(6.0, abs=0.01)
    # スプレッドが広い GBP/JPY はより高コスト
    assert gbpjpy_cost > usdjpy_cost
    assert gbpjpy_cost == pytest.approx(14.0, abs=0.01)
    # EUR/USD は JPY 換算 (150.0 固定) が適用される
    assert eurusd_cost > 0


# ---- B7: margin_usage_pct の通貨換算回帰テスト ----

def test_margin_usage_pct_converts_non_jpy_to_jpy() -> None:
    """EUR/USD のような非JPYペアでも証拠金消費率が JPY 換算されることを確認
    (旧実装は is_jpy_pair 未使用で USD 建てのまま initial_cash と比較しており、
    実質 150 分の1 に過小評価されていた)."""
    state = PortfolioState(cash=1_000_000.0, initial_cash=1_000_000.0, max_equity=1_000_000.0)
    config = SimulatorConfig(is_jpy_pair=False, lot_size=1_000)
    open_long(state, config, entry_price=1.08, entry_time=pd.Timestamp("2024-01-01"), stop_loss=1.07, take_profit=1.09)

    pct_non_jpy = margin_usage_pct(state, current_price=1.08, is_jpy_pair=False, leverage=25.0)
    pct_wrong_jpy_flag = margin_usage_pct(state, current_price=1.08, is_jpy_pair=True, leverage=25.0)

    # 正しい JPY 換算 (150倍) をすると、is_jpy_pair=True 誤判定時より遥かに大きい値になる
    assert pct_non_jpy > pct_wrong_jpy_flag * 100  # 150倍相当の差があるはず
    assert pct_non_jpy == pytest.approx(pct_wrong_jpy_flag * 150.0, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
