"""バックテスト・ランナー (SYS-FX008 トレンドフォロー・MAクロス戦略、EXP-FX000002).

`trend_following.evaluate_trend_follow()` はステートレス（RangeBreakoutEngineの
ような待機状態を持たない）なため、SYS-FX007のrunner.pyで問題になった
「エンジン状態がバーをまたいで永続化されない」「同一確定バーへの重複処理」
という類のバグは構造的に発生しない。ただし以下は runner.py と同じ設計を踏襲する:

    - 確定済みバーのみを使う (last_confirmed_bar_ts で進行中バーを除外)
    - SL/TPチェックはこのバー自身の高安のみを使う (先読み回避)
    - トレーリングストップの更新はこのバーの Open のみを使う (Close/High/Lowは
      使わない。同一バー内でOpenを見て決めた損切り幅を、同じバーのHigh/Lowで
      判定するのはやや楽観的だが、これは実運用でも「寄り付き後すぐにストップ
      を調整する」動作に相当し許容範囲とする)

データ要件:
    LT: D1 OHLCV
    MT: H4 終値 (継続確認用、OHLCは不要)
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
from minmax_fx_dt.strategy.trend_following import (
    TrendFollowConfig,
    TrendFollowSignal,
    evaluate_trend_follow,
    lt_direction_only,
)

# トレーリングストップに対する take_profit の実質無効化倍率。
# 決済はトレーリングストップまたはLT反転クロスのみで行う設計 (00-spec.md §決済) のため、
# take_profit 自体は「到達しえない値」として置くだけで、実際の決済判定には使わない。
_TP_DISABLE_MULTIPLIER = 1000.0


@dataclass
class TrendFollowBacktestResult:
    """バックテスト結果."""

    metrics: BacktestMetrics
    state: PortfolioState
    config: SimulatorConfig
    pair: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp


def run_trend_follow_backtest(
    lt_ohlcv: pd.DataFrame,
    mt_ohlcv: pd.DataFrame,
    st_ohlcv: pd.DataFrame,
    *,
    pair: str = "USDJPY",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    sim_config: Optional[SimulatorConfig] = None,
    tf_config: Optional[TrendFollowConfig] = None,
) -> TrendFollowBacktestResult:
    """バックテストを実行.

    Args:
        lt_ohlcv: 長期足 (D1) の OHLCV.
        mt_ohlcv: 中期足 (H4) の OHLCV (close のみ使用).
        st_ohlcv: 短期足 (M5) の OHLCV.
        pair: 通貨ペア名.
        start/end: 期間フィルタ.
        sim_config: シミュレータ設定.
        tf_config: トレンドフォロー戦略設定.

    Returns:
        TrendFollowBacktestResult.
    """
    if sim_config is None:
        sim_config = SimulatorConfig()
    if tf_config is None:
        tf_config = TrendFollowConfig()

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

    # シグナルのキャッシュ (新しい確定 D1/H4 バーが出た時だけ再計算、効率化のため)
    last_processed_d1_ts: Optional[pd.Timestamp] = None
    last_processed_h4_ts: Optional[pd.Timestamp] = None
    cached_signal: Optional[TrendFollowSignal] = None
    cached_lt_dir_only = "NONE"

    for st_ts, st_row in st_ohlcv.iterrows():
        st_open = float(st_row["open"])
        st_high = float(st_row["high"])
        st_low = float(st_row["low"])
        st_close = float(st_row["close"])

        # 1. 週末強制クローズ
        maybe_force_weekend_close(state, sim_config, st_ts, st_close)

        # 2. トレーリングストップ更新 (このバーの Open のみ使用、先読み回避。SLは有利な方向にのみ動かす)
        if cached_signal is not None and cached_signal.atr_value > 0:
            trail = tf_config.atr_trail_multiplier * cached_signal.atr_value
            if state.long_trade is not None:
                new_stop = st_open - trail
                if new_stop > state.long_trade.stop_loss:
                    state.long_trade.stop_loss = new_stop
            if state.short_trade is not None:
                new_stop = st_open + trail
                if new_stop < state.short_trade.stop_loss:
                    state.short_trade.stop_loss = new_stop

        # 3. SL チェック (このバー自身の高安のみ、先読みなし)
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
        if len(lt_slice) < tf_config.lt_sma_long or len(mt_slice) < tf_config.mt_confirm_sma_length:
            update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
            continue

        is_new_bar = (confirmed_d1_ts != last_processed_d1_ts) or (confirmed_h4_ts != last_processed_h4_ts)
        if is_new_bar:
            try:
                cached_signal = evaluate_trend_follow(
                    lt_high=lt_slice["high"], lt_low=lt_slice["low"], lt_close=lt_slice["close"],
                    mt_close=mt_slice["close"], config=tf_config,
                )
                cached_lt_dir_only = lt_direction_only(lt_slice["close"], tf_config)
            except Exception:
                update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
                continue
            last_processed_d1_ts = confirmed_d1_ts
            last_processed_h4_ts = confirmed_h4_ts

        if cached_signal is None:
            update_equity(state, st_close, st_ts, sim_config.is_jpy_pair)
            continue

        # 6. 反転クロスでの手仕舞い (トレンド強度フィルター抜き、純粋なSMAクロスの向きで判定。
        #    フィルター込みの方向だと "RANGE" に落ちた瞬間に手仕舞いしてしまい、
        #    トレーリングストップより先に大半のトレードが終わってしまうため)
        if state.position_side == PositionSide.LONG and cached_lt_dir_only == "DOWN":
            close_long(state, sim_config, st_close, st_ts, "REVERSE")
        elif state.position_side == PositionSide.SHORT and cached_lt_dir_only == "UP":
            close_short(state, sim_config, st_close, st_ts, "REVERSE")

        # 7. エントリー判定 (フラットの場合のみ)
        # 週末クローズ窓 (土曜 06:00 JST 未満) では新規エントリーしない。
        # このバーで強制クローズされた直後に再エントリーし、次のバーでまた強制
        # クローズされる、を繰り返す無意味な回転売買を防ぐ (スモークテストで実測)。
        if (
            state.position_side == PositionSide.FLAT
            and cached_signal.direction != "NONE"
            and cached_signal.atr_value > 0
            and not (sim_config.weekend_close and is_weekend_close_time(st_ts))
        ):
            trail = tf_config.atr_trail_multiplier * cached_signal.atr_value
            if cached_signal.direction == "UP":
                trade = open_long(
                    state, sim_config,
                    entry_price=st_close, entry_time=st_ts,
                    stop_loss=st_close - trail,
                    take_profit=st_close + trail * _TP_DISABLE_MULTIPLIER,
                )
                trade.lt_direction = cached_signal.lt_direction
            elif cached_signal.direction == "DOWN":
                trade = open_short(
                    state, sim_config,
                    entry_price=st_close, entry_time=st_ts,
                    stop_loss=st_close + trail,
                    take_profit=st_close - trail * _TP_DISABLE_MULTIPLIER,
                )
                trade.lt_direction = cached_signal.lt_direction

        # 8. 証拠金消費率追跡
        m_pct = margin_usage_pct(state, st_close, sim_config.is_jpy_pair)
        max_margin_pct = max(max_margin_pct, m_pct)

        # 9. エクイティ更新
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

    return TrendFollowBacktestResult(
        metrics=metrics,
        state=state,
        config=sim_config,
        pair=pair,
        period_start=st_ohlcv.index[0],
        period_end=st_ohlcv.index[-1],
    )


__all__ = ["TrendFollowBacktestResult", "run_trend_follow_backtest"]
