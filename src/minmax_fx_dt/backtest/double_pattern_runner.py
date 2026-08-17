"""バックテスト・ランナー (SYS-FX009 v2 上位足トレンド+ダブルトップ/ボトム戦略、EXP-FX000003).

`double_pattern_strategy.evaluate_double_pattern_entry()` はステートレスなため、
trend_follow_runner.py と同じく「エンジン状態がバーをまたいで永続化されない」
という類のバグは構造的に発生しない。以下は runner.py / trend_follow_runner.py と
同じ設計を踏襲する:

    - 確定済みバーのみを使う (last_confirmed_bar_ts で進行中バーを除外)
    - SL/1:1ターゲット到達チェックはこのバー自身の高安のみを使う (先読み回避)
    - トレーリングストップの更新はこのバーの Open のみを使う (先読み回避)

決済ルール (00-prescreen.md §戦略設計):
    1. 初期ストップ (パターン無効化水準 + バッファ) はエントリーから固定
    2. |entry - stop| と同じ値幅 (1:1) に到達したら、ストップをブレイクイーブンへ
       移動し、以降は ATR トレーリングストップに切り替える
    3. 週末強制クローズ (土曜 06:00 JST)

SYS-FX008 と異なり LT 反転での手仕舞いは行わない (パターン自体が明確な無効化
水準を持つため、反転手仕舞いを重ねる設計にしていない。00-prescreen.md 参照)。

データ要件:
    LT: D1 終値 (方向判定用、OHLCは不要)
    MT: H4 OHLCV (パターン検出・ATR算出用)
    ST: M5 OHLCV
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from minmax_fx_dt.backtest.metrics import BacktestMetrics, compute_metrics
from minmax_fx_dt.backtest.runner import _slice_by_period, last_confirmed_bar_ts
from minmax_fx_dt.backtest.simulator import (
    PortfolioState,
    PositionSide,
    SimulatorConfig,
    check_stop_loss_take_profit,
    close_long,
    close_short,
    is_dd_paused,
    is_weekend_close_time,
    margin_usage_pct,
    maybe_force_weekend_close,
    open_long,
    open_short,
    spread_round_trip_cost_jpy,
    update_equity,
)
from minmax_fx_dt.strategy.double_pattern_strategy import (
    DoublePatternEntrySignal,
    DoublePatternStrategyConfig,
    evaluate_double_pattern_entry,
)

# 1:1到達前は take_profit を「到達しえない値」として置き、シミュレータ組み込みの
# TP即決済ではなく、このランナー自身の1:1判定 (ブレイクイーブン+トレーリング切替)
# で決済ロジックを制御する (trend_follow_runner.py と同じ手法)。
_TP_DISABLE_MULTIPLIER = 1000.0


@dataclass
class DoublePatternBacktestResult:
    """バックテスト結果."""

    metrics: BacktestMetrics
    state: PortfolioState
    config: SimulatorConfig
    pair: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp


def _apply_target_and_trailing(
    state: PortfolioState,
    dp_config: DoublePatternStrategyConfig,
    st_open: float,
    st_high: float,
    st_low: float,
    atr_value: float,
) -> None:
    """1:1ターゲット到達判定 (高安ベース) とトレーリングストップ更新 (Openベース)."""
    if state.long_trade is not None:
        trade = state.long_trade
        if not trade.target_reached and trade.initial_risk > 0:
            target_price = trade.entry_price + trade.initial_risk
            if st_high >= target_price:
                trade.target_reached = True
                trade.stop_loss = max(trade.stop_loss, trade.entry_price)
        if trade.target_reached and atr_value > 0:
            new_stop = st_open - dp_config.atr_trail_multiplier * atr_value
            if new_stop > trade.stop_loss:
                trade.stop_loss = new_stop
    if state.short_trade is not None:
        trade = state.short_trade
        if not trade.target_reached and trade.initial_risk > 0:
            target_price = trade.entry_price - trade.initial_risk
            if st_low <= target_price:
                trade.target_reached = True
                trade.stop_loss = min(trade.stop_loss, trade.entry_price)
        if trade.target_reached and atr_value > 0:
            new_stop = st_open + dp_config.atr_trail_multiplier * atr_value
            if new_stop < trade.stop_loss:
                trade.stop_loss = new_stop


def run_double_pattern_backtest(
    lt_ohlcv: pd.DataFrame,
    mt_ohlcv: pd.DataFrame,
    st_ohlcv: pd.DataFrame,
    *,
    pair: str = "USDJPY",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    sim_config: Optional[SimulatorConfig] = None,
    dp_config: Optional[DoublePatternStrategyConfig] = None,
) -> DoublePatternBacktestResult:
    """バックテストを実行.

    Args:
        lt_ohlcv: 長期足 (D1) の OHLCV (close のみ使用).
        mt_ohlcv: 中期足 (H4) の OHLCV.
        st_ohlcv: 短期足 (M5) の OHLCV.
        pair: 通貨ペア名.
        start/end: 期間フィルタ.
        sim_config: シミュレータ設定.
        dp_config: ダブルパターン戦略設定.

    Returns:
        DoublePatternBacktestResult.
    """
    if sim_config is None:
        sim_config = SimulatorConfig()
    if dp_config is None:
        dp_config = DoublePatternStrategyConfig()

    if start is not None:
        lt_ohlcv = _slice_by_period(lt_ohlcv, start, end or lt_ohlcv.index[-1])
        mt_ohlcv = _slice_by_period(mt_ohlcv, start, end or mt_ohlcv.index[-1])
        st_ohlcv = _slice_by_period(st_ohlcv, start, end or st_ohlcv.index[-1])

    if not isinstance(lt_ohlcv.index, pd.DatetimeIndex):
        raise ValueError("lt_ohlcv.index must be DatetimeIndex")
    if not isinstance(mt_ohlcv.index, pd.DatetimeIndex):
        raise ValueError("mt_ohlcv.index must be DatetimeIndex")

    state = PortfolioState(
        cash=sim_config.initial_cash_jpy,
        initial_cash=sim_config.initial_cash_jpy,
        max_equity=sim_config.initial_cash_jpy,
    )
    max_margin_pct = 0.0

    min_lt_bars = max(dp_config.lt_sma_short, dp_config.lt_sma_long)
    min_mt_bars = max(dp_config.atr_length, dp_config.pattern.max_bars_since_second_pivot) + 5

    last_processed_d1_ts: Optional[pd.Timestamp] = None
    last_processed_h4_ts: Optional[pd.Timestamp] = None
    cached_signal: Optional[DoublePatternEntrySignal] = None

    for st_ts, st_row in st_ohlcv.iterrows():
        st_open = float(st_row["open"])
        st_high = float(st_row["high"])
        st_low = float(st_row["low"])
        st_close = float(st_row["close"])

        # 1. 週末強制クローズ
        maybe_force_weekend_close(state, sim_config, st_ts, st_close)

        # 2. 1:1ターゲット到達判定 + トレーリングストップ更新 (先読み回避、SLは有利な方向にのみ動かす)
        atr_for_trail = cached_signal.atr_value if cached_signal is not None else 0.0
        _apply_target_and_trailing(state, dp_config, st_open, st_high, st_low, atr_for_trail)

        # 3. SL チェック (このバー自身の高安のみ、先読みなし。TPは実質無効化済み)
        check_stop_loss_take_profit(state, sim_config, high=st_high, low=st_low, timestamp=st_ts)

        # 4. DD 停止判定
        if is_dd_paused(state, sim_config):
            update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
            continue

        # 5. 確定済みバーのみでシグナル評価 (walk-forward、未確定バーの除外)
        confirmed_d1_ts = last_confirmed_bar_ts(lt_ohlcv.index, st_ts)
        confirmed_h4_ts = last_confirmed_bar_ts(mt_ohlcv.index, st_ts)
        if confirmed_d1_ts is None or confirmed_h4_ts is None:
            update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
            continue

        lt_slice = lt_ohlcv.loc[lt_ohlcv.index <= confirmed_d1_ts]
        mt_slice = mt_ohlcv.loc[mt_ohlcv.index <= confirmed_h4_ts]
        if len(lt_slice) < min_lt_bars or len(mt_slice) < min_mt_bars:
            update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
            continue

        is_new_bar = (confirmed_d1_ts != last_processed_d1_ts) or (confirmed_h4_ts != last_processed_h4_ts)
        if is_new_bar or cached_signal is None:
            try:
                cached_signal = evaluate_double_pattern_entry(
                    lt_close=lt_slice["close"],
                    mt_high=mt_slice["high"], mt_low=mt_slice["low"], mt_close=mt_slice["close"],
                    current_price=st_close,
                    config=dp_config,
                )
            except Exception:
                update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
                continue
            last_processed_d1_ts = confirmed_d1_ts
            last_processed_h4_ts = confirmed_h4_ts

        # 6. エントリー判定 (フラットの場合のみ)
        # 週末クローズ窓 (土曜 06:00 JST 未満) では新規エントリーしない (回転売買防止、SYS-FX008と同じ理由)。
        # cached_signal は次の確定バーまで再計算されない (絶対価格のstop_lossを含む) ため、
        # 同一確定バー内で価格が急変(フラッシュクラッシュ等)すると、現在値が既に
        # stop_loss を超えた「建値時点で即ストップ状態」のポジションを繰り返し開いてしまう
        # 回転売買バグが起きる (2024-07-11 BOJ介入時のUSD/JPY等で実測、98/268トレードが該当)。
        # エントリー方向と整合する側に現在値があるかを確認し、既にstop_loss側へ price
        # が抜けていれば陳腐化したシグナルとしてエントリーしない。
        signal_still_valid = (
            cached_signal is not None
            and cached_signal.direction == "UP"
            and st_close > cached_signal.stop_loss
        ) or (
            cached_signal is not None
            and cached_signal.direction == "DOWN"
            and st_close < cached_signal.stop_loss
        )
        if (
            state.position_side == PositionSide.FLAT
            and cached_signal is not None
            and cached_signal.direction != "NONE"
            and signal_still_valid
            and not (sim_config.weekend_close and is_weekend_close_time(st_ts))
        ):
            if cached_signal.direction == "UP":
                trade = open_long(
                    state, sim_config,
                    entry_price=st_close, entry_time=st_ts,
                    stop_loss=cached_signal.stop_loss,
                    take_profit=st_close + cached_signal.initial_risk * _TP_DISABLE_MULTIPLIER,
                )
            else:
                trade = open_short(
                    state, sim_config,
                    entry_price=st_close, entry_time=st_ts,
                    stop_loss=cached_signal.stop_loss,
                    take_profit=st_close - cached_signal.initial_risk * _TP_DISABLE_MULTIPLIER,
                )
            trade.lt_direction = cached_signal.lt_direction
            # 約定価格(スプレッド/スリッページ込み)ベースで初期リスク幅を再計算 (1:1ターゲットの基準)
            trade.initial_risk = abs(trade.entry_price - trade.stop_loss)

        # 7. 証拠金消費率追跡
        m_pct = margin_usage_pct(state, st_close, sim_config.is_jpy_pair)
        max_margin_pct = max(max_margin_pct, m_pct)

        # 8. エクイティ更新
        update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)

    # 期間末尾で全決済
    if state.long_trade is not None:
        close_long(state, sim_config, st_ohlcv["close"].iloc[-1], st_ohlcv.index[-1], "END")
    if state.short_trade is not None:
        close_short(state, sim_config, st_ohlcv["close"].iloc[-1], st_ohlcv.index[-1], "END")

    metrics = compute_metrics(
        state,
        spread_round_trip_jpy=spread_round_trip_cost_jpy(sim_config),
        max_margin_usage_pct=max_margin_pct,
    )

    return DoublePatternBacktestResult(
        metrics=metrics,
        state=state,
        config=sim_config,
        pair=pair,
        period_start=st_ohlcv.index[0],
        period_end=st_ohlcv.index[-1],
    )


__all__ = ["DoublePatternBacktestResult", "run_double_pattern_backtest"]
