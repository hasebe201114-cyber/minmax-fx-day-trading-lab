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
    last_mt_result = None
    engine = None  # RangeBreakoutEngine をバーをまたいで永続化する (OBS000007 チェック #3)
    last_processed_h4_ts: Optional[pd.Timestamp] = None  # 直近に process_h4_bar 済みの確定 H4 バー

    # 各 ST バーで:
    # 1. 週末強制クローズ
    # 2. SL / TP チェック (このバー自身の高安のみを使用、先読みなし)
    # 3. 確定済みバーのみで MTF 評価 (未確定の進行中 H4/D1 バーは除外)
    # 4. 新たに 1 本確定した H4 バーがあれば、その回のみステートマシンを駆動
    # 5. エントリー判定 → シミュレータ
    # 6. エクイティ更新

    mt_ohlcv = mt_ohlcv.copy()
    if not isinstance(mt_ohlcv.index, pd.DatetimeIndex):
        raise ValueError("mt_ohlcv.index must be DatetimeIndex")
    if not isinstance(lt_ohlcv.index, pd.DatetimeIndex):
        raise ValueError("lt_ohlcv.index must be DatetimeIndex")

    for st_ts, st_row in st_ohlcv.iterrows():
        st_high = float(st_row["high"])
        st_low = float(st_row["low"])
        st_close = float(st_row["close"])

        # 1. 週末強制クローズ
        maybe_force_weekend_close(state, sim_config, st_ts, st_close)

        # 2. SL / TP チェック: 進行中 H4 バーの確定高安 (最大 4 時間先の情報) ではなく、
        # このバー自身の高安のみを使う (OBS000007 チェック #1、先読みバグの修正)
        check_stop_loss_take_profit(
            state, sim_config,
            high=st_high, low=st_low,
            timestamp=st_ts,
        )

        # 3. DD 停止判定
        if is_dd_paused(state, sim_config):
            update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
            continue

        # 4. 確定済みバーのみで MTF 評価 (walk-forward、未確定バーの除外)
        # OBS000007 チェック #2: st_ts が含まれる H4/D1 バーはまだ確定していないため、
        # LT/MT の判定には「1 本前まで」の確定済みバーのみを使う。
        current_h4_idx = mt_ohlcv.index[mt_ohlcv.index <= st_ts]
        current_d1_idx = lt_ohlcv.index[lt_ohlcv.index <= st_ts]
        if len(current_h4_idx) == 0 or len(current_d1_idx) == 0:
            update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
            continue
        confirmed_h4_idx = mt_ohlcv.index[mt_ohlcv.index < current_h4_idx[-1]]
        confirmed_d1_idx = lt_ohlcv.index[lt_ohlcv.index < current_d1_idx[-1]]
        if len(confirmed_h4_idx) == 0 or len(confirmed_d1_idx) == 0:
            # まだ 1 本も確定済みバーがない (バックテスト開始直後)
            update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
            continue
        confirmed_h4_ts = confirmed_h4_idx[-1]
        confirmed_d1_ts = confirmed_d1_idx[-1]

        try:
            lt_slice = lt_ohlcv.loc[lt_ohlcv.index <= confirmed_d1_ts]
            mt_slice = mt_ohlcv.loc[mt_ohlcv.index <= confirmed_h4_ts]
            st_slice = st_ohlcv.loc[st_ohlcv.index <= st_ts]
            if len(lt_slice) < mtf_config.lt_sma_long or len(mt_slice) < mtf_config.mt_donchian_length:
                update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
                continue

            # 新たに 1 本確定した H4 バーがある場合のみステートマシンを駆動する
            # (同じ確定バーに対して繰り返し process_h4_bar すると、実際には新しい
            # 確定バーがないのに RANGE_BREAKOUT_* → TREND_* へ誤って進んでしまう)
            is_new_h4_bar = confirmed_h4_ts != last_processed_h4_ts

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
                engine=engine,
                process_h4=is_new_h4_bar,
            )
        except Exception:
            update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
            continue

        engine = result.engine  # ステートマシンを次のバーへ永続化
        if is_new_h4_bar:
            last_processed_h4_ts = confirmed_h4_ts

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
