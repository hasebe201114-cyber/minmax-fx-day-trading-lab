"""バックテスト・メインランナー (SYS-FX007 レンジブレイク・プルバック戦略).

3 レイヤ MTF データを入力として受け取り、戦略エンジン (multi_timeframe) と
シミュレータ (simulator) を駆動してトレードを実行、KPI (metrics) を計算する。

データ要件:
    LT: D1 / W1 OHLCV
    MT: H4 / H1 OHLCV
    ST: M15 / M5 OHLCV

実行:
    python scripts/run_backtest.py --pair USDJPY --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from minmax_fx_dt.backtest.metrics import BacktestMetrics, compute_metrics, to_dict
from minmax_fx_dt.backtest.simulator import (
    PortfolioState,
    PositionSide,
    SimulatorConfig,
    check_stop_loss_take_profit,
    close_long,
    close_short,
    is_dd_paused,
    margin_usage_pct,
    maybe_force_weekend_close,
    open_long,
    open_short,
    update_equity,
)
from minmax_fx_dt.strategy.range_breakout import State


@dataclass
class BacktestResult:
    """バックテスト結果."""

    metrics: BacktestMetrics
    state: PortfolioState
    config: SimulatorConfig
    pair: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp


def _slice_by_period(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """期間で df を絞り込み."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    return df.loc[(df.index >= start) & (df.index <= end)]


def run_backtest(
    lt_ohlcv: pd.DataFrame,
    mt_ohlcv: pd.DataFrame,
    st_ohlcv: pd.DataFrame,
    *,
    pair: str = "USDJPY",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    sim_config: Optional[SimulatorConfig] = None,
    mtf_config = None,
) -> BacktestResult:
    """バックテストを実行.

    Args:
        lt_ohlcv: 長期足 (D1/W1) の OHLCV (columns: open, high, low, close, volume).
        mt_ohlcv: 中期足 (H4/H1) の OHLCV.
        st_ohlcv: 短期足 (M15/M5) の OHLCV.
        pair: 通貨ペア名.
        start: 開始日.
        end: 終了日.
        sim_config: シミュレータ設定.
        mtf_config: MTF 設定.

    Returns:
        BacktestResult.
    """
    if sim_config is None:
        sim_config = SimulatorConfig()

    # 期間フィルタ
    if start is not None:
        lt_ohlcv = _slice_by_period(lt_ohlcv, start, end or lt_ohlcv.index[-1])
        mt_ohlcv = _slice_by_period(mt_ohlcv, start, end or mt_ohlcv.index[-1])
        st_ohlcv = _slice_by_period(st_ohlcv, start, end or st_ohlcv.index[-1])

    # MTFConfig は pandas-ta が必要なので、遅延インポート
    from minmax_fx_dt.strategy.multi_timeframe import MTFConfig, evaluate_mtf

    if mtf_config is None:
        mtf_config = MTFConfig()

    state = PortfolioState(
        cash=sim_config.initial_cash_jpy,
        initial_cash=sim_config.initial_cash_jpy,
        max_equity=sim_config.initial_cash_jpy,
    )

    weak_breakout_signals = 0
    weak_breakout_blocked = 0
    max_margin_pct = 0.0

    # ST バーを時系列で処理
    last_mt_bar_ts: Optional[pd.Timestamp] = None
    last_mt_result = None
    last_h4_index: Optional[pd.Timestamp] = None
    last_h4_high = last_h4_low = last_h4_close = None

    # 簡略化のため、ST の各バーで:
    # 1. 最新の H4 バーに更新があれば RangeBreakoutEngine を駆動
    # 2. 直近 N 本の ST でプルバック確認
    # 3. エントリー判定 → シミュレータ
    # 4. 週末強制クローズ
    # 5. エクイティ更新

    # H4 → ST のマッピング (H4 バーが属する時刻範囲の ST バーを紐付け)
    mt_ohlcv = mt_ohlcv.copy()
    if not isinstance(mt_ohlcv.index, pd.DatetimeIndex):
        raise ValueError("mt_ohlcv.index must be DatetimeIndex")
    mt_ohlcv["bar_id"] = mt_ohlcv.index

    for st_ts, st_row in st_ohlcv.iterrows():
        st_high = float(st_row["high"])
        st_low = float(st_row["low"])
        st_close = float(st_row["close"])

        # 1. 対応する H4 バーを特定 (直近の H4)
        current_h4_idx = mt_ohlcv.index[mt_ohlcv.index <= st_ts]
        if len(current_h4_idx) == 0:
            update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
            continue
        current_h4_ts = current_h4_idx[-1]
        if current_h4_ts != last_h4_index:
            # H4 バー切替
            last_h4_index = current_h4_ts
            h4_row = mt_ohlcv.loc[current_h4_ts]
            last_h4_high = float(h4_row["high"])
            last_h4_low = float(h4_row["low"])
            last_h4_close = float(h4_row["close"])

        # 2. 週末強制クローズ
        maybe_force_weekend_close(state, sim_config, st_ts, st_close)

        # 3. SL / TP チェック
        if last_h4_high is not None and last_h4_low is not None:
            check_stop_loss_take_profit(
                state, sim_config,
                high=last_h4_high, low=last_h4_low,
                timestamp=st_ts,
            )

        # 4. DD 停止判定
        if is_dd_paused(state, sim_config):
            update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
            continue

        # 5. MTF 評価: その時点までのデータにスライスして渡す (walk-forward)
        # これが無いと Donchian upper がトレンド最高値に張り付き、ブレイクが検出されない
        try:
            lt_slice = lt_ohlcv.loc[lt_ohlcv.index <= st_ts]
            mt_slice = mt_ohlcv.loc[mt_ohlcv.index <= st_ts]
            st_slice = st_ohlcv.loc[st_ohlcv.index <= st_ts]
            if len(lt_slice) < mtf_config.lt_sma_long or len(mt_slice) < mtf_config.mt_donchian_length:
                update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
                continue
            result = evaluate_mtf(
                lt_high=lt_slice["high"],
                lt_low=lt_slice["low"],
                lt_close=lt_slice["close"],
                mt_high=mt_slice["high"],
                mt_low=mt_slice["low"],
                mt_close=mt_slice["close"],
                st_high=st_slice["high"],
                st_low=st_slice["low"],
                st_close=st_slice["close"],
                config=mtf_config,
            )
        except Exception:
            update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
            continue

        last_mt_result = result
        sig = result.entry_signal

        # 6. エントリー判定
        # S/R ライン価格 (条件 3 で参照したライン)
        sr_level_price = 0.0
        if result.engine.last_breakout_level is not None and result.engine.sr_levels:
            from minmax_fx_dt.strategy.support_resistance import find_nearest_sr
            nearest = find_nearest_sr(result.engine.last_breakout_level, result.engine.sr_levels)
            if nearest is not None:
                sr_level_price = float(nearest.price)

        if sig.should_enter and state.position_side in (PositionSide.FLAT,):
            if sig.side == "BUY":
                trade = open_long(
                    state, sim_config,
                    entry_price=sig.entry_price,
                    entry_time=st_ts,
                    stop_loss=sig.stop_loss,
                    take_profit=sig.take_profit,
                )
                trade.entry_conditions = dict(sig.conditions_passed)
                trade.lt_direction = result.engine.lt_direction
                trade.sr_level_price = sr_level_price
            elif sig.side == "SELL":
                trade = open_short(
                    state, sim_config,
                    entry_price=sig.entry_price,
                    entry_time=st_ts,
                    stop_loss=sig.stop_loss,
                    take_profit=sig.take_profit,
                )
                trade.entry_conditions = dict(sig.conditions_passed)
                trade.lt_direction = result.engine.lt_direction
                trade.sr_level_price = sr_level_price
        elif sig.should_enter and state.position_side == PositionSide.SHORT and sig.side == "BUY":
            # ショート中に買いシグナル → ショートクローズ + ロング
            close_short(state, sim_config, sig.entry_price, st_ts, "REVERSE")
            trade = open_long(
                state, sim_config,
                entry_price=sig.entry_price,
                entry_time=st_ts,
                stop_loss=sig.stop_loss,
                take_profit=sig.take_profit,
            )
            trade.entry_conditions = dict(sig.conditions_passed)
            trade.lt_direction = result.engine.lt_direction
            trade.sr_level_price = sr_level_price
        elif sig.should_enter and state.position_side == PositionSide.LONG and sig.side == "SELL":
            close_long(state, sim_config, sig.entry_price, st_ts, "REVERSE")
            trade = open_short(
                state, sim_config,
                entry_price=sig.entry_price,
                entry_time=st_ts,
                stop_loss=sig.stop_loss,
                take_profit=sig.take_profit,
            )
            trade.entry_conditions = dict(sig.conditions_passed)
            trade.lt_direction = result.engine.lt_direction
            trade.sr_level_price = sr_level_price
        else:
            # エントリー不可だが、シグナル有りの場合は記録
            if result.engine.state in (State.RANGE_BREAKOUT_UP, State.RANGE_BREAKOUT_DOWN):
                weak_breakout_signals += 1
                if not sig.should_enter:
                    weak_breakout_blocked += 1

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
        spread_round_trip_jpy=60.0,
        weak_breakout_signals=weak_breakout_signals,
        weak_breakout_blocked=weak_breakout_blocked,
        max_margin_usage_pct=max_margin_pct,
    )

    return BacktestResult(
        metrics=metrics,
        state=state,
        config=sim_config,
        pair=pair,
        period_start=st_ohlcv.index[0],
        period_end=st_ohlcv.index[-1],
    )


__all__ = ["BacktestResult", "run_backtest", "to_dict"]
