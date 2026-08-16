"""マルチタイムフレーム統合 (SYS-FX007 v2).

3 レイヤ MTF 構造:
    LT (D1 / W1): 長期方向
    MT (H4 / H1): 中期レンジ・ブレイク判定 (MT-1, MT-2, MT-3)
    ST (M15 / M5): 戻り確認

各レイヤの OHLCV を受け取り、RangeBreakoutEngine を駆動してエントリーシグナルを生成する。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from minmax_fx_dt.strategy.indicators import (
    adx,
    adx_wilder,
    atr,
    bbands,
    classify_lt_direction,
    donchian,
    ma_spread_atr_strength,
    sma,
)
from minmax_fx_dt.strategy.range_breakout import (
    EntrySignal,
    OrderBookSignal,
    PullbackSignal,
    RangeBreakoutEngine,
    State,
)
from minmax_fx_dt.strategy.support_resistance import (
    detect_support_resistance,
    find_nearest_sr,
)


@dataclass
class MTFConfig:
    """MTF 設定."""

    # LT パラメータ
    lt_sma_short: int = 50
    lt_sma_long: int = 200
    lt_adx_threshold: float = 25.0
    # OBS000006 Phase 1 (追記3): トレンド強度指標の選択。
    # "adx" (既定・非Wilder) / "adx_wilder" (RMA平滑) / "ma_spread_atr" (SMA乖離/ATR)
    # lt_adx_threshold は選択した指標の閾値として共用する。
    lt_trend_strength_method: str = "adx"

    # MT パラメータ
    mt_donchian_length: int = 20
    mt_atr_length: int = 14
    mt_min_range_atr_multiplier: float = 1.0

    # S/R パラメータ
    sr_fractal_window: int = 5
    sr_min_touches: int = 3
    sr_cluster_threshold_pct: float = 0.5
    sr_lookback_days: int = 365

    # エンジン閾値
    sr_strength_threshold: float = 0.5
    atr_stop_multiplier: float = 1.0
    reward_risk_ratio: float = 2.0
    # プルバック深さ (A4): breakout_level から min_depth_atr × ATR 以上離れたら「深いプルバック」と判定
    # 0.0 = 制約なし (デフォルト、後方互換)
    pb_min_depth_atr: float = 0.0


def detect_pullback(
    st_high: pd.Series,
    st_low: pd.Series,
    st_close: pd.Series,
    breakout_level: float,
    direction: str,
    atr_value: float,
    min_depth_atr: float = 0.0,
) -> PullbackSignal:
    """ST レイヤで戻り確認シグナルを検出 (M15/M5).

    シンプルな実装: 直近 N 本で breakout level まで戻って反発したことを確認.
    A4 拡張: `min_depth_atr` オプションで「浅いプルバック (= ノイズ)」を排除.
    breakout_level から min_depth_atr × ATR 以上離れたら「深いプルバック」と判定.

    Args:
        st_high: ST 高値系列.
        st_low: ST 安値系列.
        st_close: ST 終値系列.
        breakout_level: ブレイクレベル.
        direction: "UP" / "DOWN".
        atr_value: ATR (H4 基準).
        min_depth_atr: 最小プルバック深さ (単位: ATR). 0.0 = 制約なし.

    Returns:
        PullbackSignal.
    """
    if len(st_close) < 3:
        return PullbackSignal(confirmed=False, pattern="NONE", timestamp=st_close.index[-1])

    last_n = 3
    recent_close = st_close.iloc[-last_n:]
    recent_high = st_high.iloc[-last_n:]
    recent_low = st_low.iloc[-last_n:]

    pin_threshold = atr_value * 0.5
    depth_threshold = min_depth_atr * atr_value  # 0.0 なら制約なし

    if direction == "UP":
        # ブル・ピンバー: 直近のいずれかで breakout level に戻り、長い下髭
        returned_to_level = (recent_low <= breakout_level + pin_threshold).any()
        # 深いプルバック: breakout_level から下に depth_threshold 以上離れた安値が存在
        deep_pullback = (
            (recent_low <= breakout_level - depth_threshold).any()
            if depth_threshold > 0
            else True
        )
        last_bar = st_close.iloc[-1]
        last_low = st_low.iloc[-1]
        last_high = st_high.iloc[-1]
        body_size = abs(last_bar - recent_close.iloc[-2]) if len(recent_close) >= 2 else 0
        lower_wick = min(last_bar, recent_close.iloc[-2] if len(recent_close) >= 2 else last_bar) - last_low
        pin_bar = lower_wick > body_size and lower_wick > pin_threshold

        if returned_to_level and deep_pullback and last_bar > breakout_level:
            return PullbackSignal(
                confirmed=True,
                pattern="PIN_BAR" if pin_bar else "RETURN_OK",
                timestamp=st_close.index[-1],
            )

    elif direction == "DOWN":
        returned_to_level = (recent_high >= breakout_level - pin_threshold).any()
        # 深いプルバック: breakout_level から上に depth_threshold 以上離れた高値が存在
        deep_pullback = (
            (recent_high >= breakout_level + depth_threshold).any()
            if depth_threshold > 0
            else True
        )
        last_bar = st_close.iloc[-1]
        last_high = st_high.iloc[-1]
        body_size = abs(last_bar - recent_close.iloc[-2]) if len(recent_close) >= 2 else 0
        upper_wick = last_high - max(last_bar, recent_close.iloc[-2] if len(recent_close) >= 2 else last_bar)
        pin_bar = upper_wick > body_size and upper_wick > pin_threshold

        if returned_to_level and deep_pullback and last_bar < breakout_level:
            return PullbackSignal(
                confirmed=True,
                pattern="PIN_BAR" if pin_bar else "RETURN_OK",
                timestamp=st_close.index[-1],
            )

    return PullbackSignal(confirmed=False, pattern="NONE", timestamp=st_close.index[-1])


@dataclass
class MTFResult:
    """MTF 評価結果."""

    engine: RangeBreakoutEngine
    entry_signal: EntrySignal
    lt_direction: str
    sr_levels_count: int
    timestamp: pd.Timestamp


def evaluate_mtf(
    lt_high: pd.Series,
    lt_low: pd.Series,
    lt_close: pd.Series,
    mt_high: pd.Series,
    mt_low: pd.Series,
    mt_close: pd.Series,
    st_high: pd.Series,
    st_low: pd.Series,
    st_close: pd.Series,
    *,
    orderbook_signal: OrderBookSignal | None = None,
    config: MTFConfig | None = None,
    engine: RangeBreakoutEngine | None = None,
    process_h4: bool = True,
) -> MTFResult:
    """マルチタイムフレーム評価を実行.

    Args:
        lt_high/low/close: 長期足 (D1 / W1) の OHLC。呼び出し側は確定済み
            (未確定の進行中バーを含まない) バーのみを渡すこと (OBS000007 チェック #2)。
        mt_high/low/close: 中期足 (H4 / H1) の OHLC。同上、確定済みバーのみ。
        st_high/low/close: 短期足 (M15 / M5) の OHLC。
        orderbook_signal: 注文板 / センチメントシグナル (オプション).
        config: MTF 設定 (None ならデフォルト).
        engine: 既存の RangeBreakoutEngine (バーをまたいで永続化する場合に渡す)。
            None の場合は新規生成する (後方互換、単発評価用)。
            OBS000007 チェック #3: 呼び出しのたびに新規生成するとステートマシンが
            機能しない (RANGE_BREAKOUT_UP 等の状態が保持されない) ため、
            バックテストループ側で 1 つの engine を使い回すことを想定する。
        process_h4: True の場合のみ engine.process_h4_bar() を呼ぶ。
            確定済み H4 バーが新たに1本確定したタイミングでのみ True にし、
            同一の確定バーに対して重複して状態遷移させないようにする。

    Returns:
        MTFResult (エンジン、エントリーシグナル、LT 方向、S/R 本数、タイムスタンプ).
    """
    if config is None:
        config = MTFConfig()

    # 1. LT 方向判定
    sma_short = sma(lt_close, config.lt_sma_short)
    sma_long = sma(lt_close, config.lt_sma_long)
    if config.lt_trend_strength_method == "adx_wilder":
        trend_strength = adx_wilder(lt_high, lt_low, lt_close, length=14).iloc[:, 0]
    elif config.lt_trend_strength_method == "ma_spread_atr":
        trend_strength = ma_spread_atr_strength(
            lt_close, sma_short, sma_long, lt_high, lt_low, atr_length=14
        )
    else:
        trend_strength = adx(lt_high, lt_low, lt_close, length=14).iloc[:, 0]
    lt_direction_series = classify_lt_direction(
        lt_close, sma_short, sma_long, trend_strength, config.lt_adx_threshold
    )
    lt_direction = str(lt_direction_series.iloc[-1])

    # 2. MT レンジ + S/R
    dc = donchian(mt_high, mt_low, lower_length=config.mt_donchian_length, upper_length=config.mt_donchian_length)
    dc_upper = dc.iloc[:, 2] if dc.shape[1] >= 3 else dc.iloc[:, -1]  # DCU
    dc_lower = dc.iloc[:, 0]  # DCL
    mt_atr = atr(mt_high, mt_low, mt_close, length=config.mt_atr_length)

    # shift(1) で前バーまでの max/min を使う (ブレイク判定用)
    # これをしないと「現在バーの high = upper」になって close > upper が永遠に成立しない
    dc_upper_prior = dc_upper.shift(1)
    dc_lower_prior = dc_lower.shift(1)

    last_dc_upper = float(dc_upper_prior.iloc[-1]) if not pd.isna(dc_upper_prior.iloc[-1]) else float(dc_upper.iloc[-1])
    last_dc_lower = float(dc_lower_prior.iloc[-1]) if not pd.isna(dc_lower_prior.iloc[-1]) else float(dc_lower.iloc[-1])
    last_atr = float(mt_atr.iloc[-1])
    last_mt_ts = mt_close.index[-1]

    sr_levels = detect_support_resistance(
        mt_high,
        mt_low,
        mt_close,
        fractal_window=config.sr_fractal_window,
        min_touches=config.sr_min_touches,
        cluster_threshold_pct=config.sr_cluster_threshold_pct,
        lookback_days=config.sr_lookback_days,
    )

    # 3. エンジン初期化 (初回のみ) または既存エンジンの更新
    # OBS000007 チェック #3: engine を毎回新規生成するとステートマシンが機能しない。
    # engine が渡されていればそれを再利用し、S/R・LT方向・レンジ情報だけ更新する。
    if engine is None:
        engine = RangeBreakoutEngine(
            sr_levels=sr_levels,
            lt_direction=lt_direction,
            sr_strength_threshold=config.sr_strength_threshold,
            atr_stop_multiplier=config.atr_stop_multiplier,
            reward_risk_ratio=config.reward_risk_ratio,
        )
    else:
        engine.sr_levels = sr_levels
        engine.lt_direction = lt_direction
    engine.update_range(last_dc_upper, last_dc_lower, last_atr, last_mt_ts)
    engine.update_orderbook(orderbook_signal or OrderBookSignal(
        available=False, buy_thickness_ratio=1.0, sentiment_long_pct=50.0, passes=True
    ))

    # 4. 確定済み H4 バーが新たに1本確定したタイミングでのみステートマシン駆動。
    # OBS000007 チェック #3: 同一の確定バーに対して process_h4_bar を繰り返し呼ぶと、
    # RANGE_BREAKOUT_UP/DOWN が (実際には新しい確定バーがないのに) TREND_UP/DOWN へ
    # 誤って進んでしまう。process_h4=False の場合は呼び出し側が既に処理済みとみなす。
    last_mt_close = float(mt_close.iloc[-1])
    if process_h4:
        engine.process_h4_bar(close=last_mt_close, timestamp=last_mt_ts)

    # 5. ST 戻り確認
    if engine.last_breakout_level is not None and engine.last_breakout_side is not None:
        direction = "UP" if engine.last_breakout_side == "UP" else "DOWN"
        pullback = detect_pullback(
            st_high, st_low, st_close,
            breakout_level=engine.last_breakout_level,
            direction=direction,
            atr_value=last_atr,
            min_depth_atr=config.pb_min_depth_atr,
        )
        engine.update_pullback(pullback)

    # 6. エントリー判定
    last_st_close = float(st_close.iloc[-1])
    last_st_ts = st_close.index[-1]
    entry_signal = engine.check_entry_conditions(close=last_st_close, timestamp=last_st_ts)

    return MTFResult(
        engine=engine,
        entry_signal=entry_signal,
        lt_direction=lt_direction,
        sr_levels_count=len(sr_levels),
        timestamp=last_st_ts,
    )


if __name__ == "__main__":
    # サンプル実行: 合成 MTF データ
    import numpy as np

    np.random.seed(42)
    n_d1 = 250
    n_h4 = n_d1 * 6
    n_m15 = n_h4 * 16

    d1_idx = pd.date_range(end=datetime(2025, 1, 1), periods=n_d1, freq="D")
    h4_idx = pd.date_range(end=datetime(2025, 1, 1), periods=n_h4, freq="4h")
    m15_idx = pd.date_range(end=datetime(2025, 1, 1), periods=n_m15, freq="15min")

    base = 100.0
    d1_close = pd.Series(base + np.cumsum(np.random.normal(0, 0.5, n_d1)), index=d1_idx)
    h4_close = pd.Series(base + np.cumsum(np.random.normal(0, 0.2, n_h4)), index=h4_idx)
    m15_close = pd.Series(base + np.cumsum(np.random.normal(0, 0.1, n_m15)), index=m15_idx)

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
    print(f"LT 方向: {result.lt_direction}")
    print(f"S/R ライン: {result.sr_levels_count} 本")
    print(f"ステート: {result.engine.state.value}")
    print(f"エントリー判定: {result.entry_signal.should_enter} ({result.entry_signal.side})")
    print(f"  条件: {result.entry_signal.conditions_passed}")
