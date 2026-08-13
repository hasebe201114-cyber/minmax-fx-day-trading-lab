"""レンジブレイク・ステートマシン (SYS-FX007 v2 の中核).

3 レイヤ MTF 構造でレンジ形成・ブレイク・戻りを管理し、両建て運用に対応する。

ステート:
    RANGE_FORMING       - レンジ形成中
    RANGE_BREAKOUT_UP   - 上限ブレイク検出 (戻り待ち)
    RANGE_BREAKOUT_DOWN - 下限ブレイク検出 (戻り待ち)
    PULLBACK_CONFIRMED  - 戻り確認 (エントリー可)
    TREND_UP / TREND_DOWN - トレンド継続中
    EXIT                - ポジション解消 (レンジ再形成待ち)

5 条件 AND (MT-1 + MT-2 + MT-3 + LT + ST) でエントリー判定:
    1. LT 方向一致 (D1/W1 の MA / ADX)
    2. MT-1 レンジブレイク (H4 Donchian/BB)
    3. MT-2 レジサポライン強度 (S/R ライン直近 1 年 3 回以上接触)
    4. MT-3 注文板 / センチメント (G フィルタ、後段で取得)
    5. ST 戻り確認 (M15/M5 ピンバー / 包み線)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from minmax_fx_dt.strategy.support_resistance import SRLevel, find_nearest_sr


class State(str, Enum):
    """戦略ステート."""

    RANGE_FORMING = "RANGE_FORMING"
    RANGE_BREAKOUT_UP = "RANGE_BREAKOUT_UP"
    RANGE_BREAKOUT_DOWN = "RANGE_BREAKOUT_DOWN"
    PULLBACK_CONFIRMED = "PULLBACK_CONFIRMED"
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    EXIT = "EXIT"


@dataclass
class RangeInfo:
    """レンジ情報."""

    upper: float          # レンジ上限
    lower: float          # レンジ下限
    mid: float            # レンジ中央
    atr: float            # レンジ幅 (ATR)
    last_update: pd.Timestamp


@dataclass
class PullbackSignal:
    """戻り確認シグナル."""

    confirmed: bool        # 確認されたか
    pattern: str           # パターン ("PIN_BAR" / "ENGULFING" / "NONE")
    timestamp: pd.Timestamp
    note: str = ""


@dataclass
class OrderBookSignal:
    """注文板 / センチメントシグナル (MT-3)."""

    available: bool                # データが利用可能か
    buy_thickness_ratio: float     # buy 厚み / 平均 (1.0 = 平均)
    sentiment_long_pct: float      # long % (50 = 中立)
    passes: bool                   # 閾値を満たすか
    note: str = ""


@dataclass
class EntrySignal:
    """エントリーシグナル."""

    should_enter: bool
    side: str                # "BUY" / "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    state: State
    conditions_passed: dict[str, bool]  # 5 条件それぞれの pass 状況
    timestamp: pd.Timestamp
    note: str = ""


@dataclass
class RangeBreakoutEngine:
    """レンジブレイク・ステートマシン.

    Attributes:
        state: 現在のステート.
        range_info: レンジ情報.
        sr_levels: 検出済み S/R ライン.
        last_breakout_level: 最後にブレイクした価格.
        pullback_signal: 戻り確認シグナル.
        orderbook_signal: 注文板 / センチメントシグナル.
        lt_direction: LT 方向 ("UP" / "DOWN" / "RANGE").
    """

    state: State = State.RANGE_FORMING
    range_info: Optional[RangeInfo] = None
    sr_levels: list[SRLevel] = field(default_factory=list)
    last_breakout_level: Optional[float] = None
    last_breakout_side: Optional[str] = None
    pullback_signal: Optional[PullbackSignal] = None
    orderbook_signal: Optional[OrderBookSignal] = None
    lt_direction: str = "RANGE"
    # 閾値 (spec パラメータ空間から)
    sr_strength_threshold: float = 0.5
    buy_thickness_threshold: float = 1.5
    sentiment_long_threshold: float = 55.0
    atr_stop_multiplier: float = 1.0
    reward_risk_ratio: float = 2.0

    def update_range(self, upper: float, lower: float, atr: float, ts: pd.Timestamp) -> None:
        """レンジを更新 (H4 クローズ時)."""
        self.range_info = RangeInfo(
            upper=upper,
            lower=lower,
            mid=(upper + lower) / 2.0,
            atr=atr,
            last_update=ts,
        )

    def update_sr_levels(self, levels: list[SRLevel]) -> None:
        """S/R ラインを更新."""
        self.sr_levels = levels

    def update_pullback(self, signal: PullbackSignal) -> None:
        """戻り確認シグナルを更新."""
        self.pullback_signal = signal

    def update_orderbook(self, signal: OrderBookSignal) -> None:
        """注文板シグナルを更新."""
        self.orderbook_signal = signal

    def update_lt_direction(self, direction: str) -> None:
        """LT 方向を更新."""
        self.lt_direction = direction

    def process_h4_bar(
        self,
        close: float,
        timestamp: pd.Timestamp,
    ) -> None:
        """H4 バーのクロージングで状態を更新 (MT-1 評価).

        Args:
            close: H4 終値.
            timestamp: バー時刻.
        """
        if self.range_info is None:
            return

        prev_state = self.state

        if self.state == State.RANGE_FORMING:
            # 上限ブレイク
            if close > self.range_info.upper:
                self.state = State.RANGE_BREAKOUT_UP
                self.last_breakout_level = self.range_info.upper
                self.last_breakout_side = "UP"
            # 下限ブレイク
            elif close < self.range_info.lower:
                self.state = State.RANGE_BREAKOUT_DOWN
                self.last_breakout_level = self.range_info.lower
                self.last_breakout_side = "DOWN"

        elif self.state in (State.RANGE_BREAKOUT_UP, State.RANGE_BREAKOUT_DOWN):
            # レンジ内に戻った場合は RANGE_FORMING に戻す
            if self.range_info.lower <= close <= self.range_info.upper:
                self.state = State.RANGE_FORMING
                self.last_breakout_level = None
                self.last_breakout_side = None
                self.pullback_signal = None
            else:
                # トレンド方向に進展
                if self.state == State.RANGE_BREAKOUT_UP:
                    self.state = State.TREND_UP
                else:
                    self.state = State.TREND_DOWN

        elif self.state in (State.TREND_UP, State.TREND_DOWN):
            # レンジ内に戻る = レンジ再形成
            if self.range_info.lower <= close <= self.range_info.upper:
                self.state = State.RANGE_FORMING
                self.last_breakout_level = None
                self.last_breakout_side = None
                self.pullback_signal = None
                self.orderbook_signal = None

        # 変化なしなら no-op
        if self.state != prev_state:
            # 状態変化時のログ用
            pass

    def check_entry_conditions(
        self,
        close: float,
        timestamp: pd.Timestamp,
    ) -> EntrySignal:
        """5 条件 AND チェック → エントリー可否判定.

        条件:
            1. LT 方向一致 (D1/W1)
            2. MT-1 レンジブレイク (H4 close > upper or < lower)
            3. MT-2 S/R ライン強度 (直近ブレイクレベルが主要 S/R)
            4. MT-3 注文板 / センチメント (G フィルタ)
            5. ST 戻り確認 (M15/M5 ピンバー / 包み線)

        Args:
            close: 現在価格.
            timestamp: 評価時刻.

        Returns:
            EntrySignal.
        """
        conditions = {
            "1_lt_direction": False,
            "2_mt1_breakout": False,
            "3_mt2_sr_line": False,
            "4_mt3_orderbook": False,
            "5_st_pullback": False,
        }

        side = ""
        if self.state == State.RANGE_BREAKOUT_UP:
            side = "BUY"
        elif self.state == State.RANGE_BREAKOUT_DOWN:
            side = "SELL"
        else:
            return EntrySignal(
                should_enter=False,
                side="",
                entry_price=close,
                stop_loss=0.0,
                take_profit=0.0,
                state=self.state,
                conditions_passed=conditions,
                timestamp=timestamp,
                note="ステートがブレイク系ではない",
            )

        # 条件 1: LT 方向一致
        conditions["1_lt_direction"] = (
            (side == "BUY" and self.lt_direction == "UP")
            or (side == "SELL" and self.lt_direction == "DOWN")
        )

        # 条件 2: MT-1 レンジブレイク (ステートマシンが既に判定済み)
        conditions["2_mt1_breakout"] = self.state in (
            State.RANGE_BREAKOUT_UP,
            State.RANGE_BREAKOUT_DOWN,
        )

        # 条件 3: MT-2 S/R ライン
        if self.last_breakout_level is not None and self.sr_levels:
            nearest = find_nearest_sr(self.last_breakout_level, self.sr_levels)
            if nearest is not None:
                tolerance = nearest.price * 0.005  # 0.5% 以内
                conditions["3_mt2_sr_line"] = (
                    abs(nearest.price - self.last_breakout_level) <= tolerance
                    and nearest.strength >= self.sr_strength_threshold
                )

        # 条件 4: MT-3 注文板 / センチメント
        if self.orderbook_signal is not None and self.orderbook_signal.available:
            conditions["4_mt3_orderbook"] = self.orderbook_signal.passes
        else:
            # フォールバックモード: データなし → 条件 4 は pass とみなす
            conditions["4_mt3_orderbook"] = True

        # 条件 5: ST 戻り確認
        if self.pullback_signal is not None:
            conditions["5_st_pullback"] = self.pullback_signal.confirmed

        should_enter = all(conditions.values())

        # 損切り / 利食い計算
        if should_enter and self.range_info is not None:
            atr = self.range_info.atr
            if side == "BUY":
                stop_loss = close - atr * self.atr_stop_multiplier
                take_profit = close + atr * self.atr_stop_multiplier * self.reward_risk_ratio
            else:
                stop_loss = close + atr * self.atr_stop_multiplier
                take_profit = close - atr * self.atr_stop_multiplier * self.reward_risk_ratio
        else:
            stop_loss = 0.0
            take_profit = 0.0

        if should_enter:
            self.state = State.PULLBACK_CONFIRMED

        return EntrySignal(
            should_enter=should_enter,
            side=side,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            state=self.state,
            conditions_passed=conditions,
            timestamp=timestamp,
            note="5 条件 AND すべて満たす" if should_enter else "条件未充足",
        )


def hedged_position_state(
    long_entry: float | None,
    short_entry: float | None,
    current_price: float,
    max_hold_days: int = 20,
    held_days: int = 0,
    margin_usage_pct: float = 0.0,
    margin_limit_pct: float = 30.0,
) -> dict[str, object]:
    """両建てポジションの状態を評価.

    Args:
        long_entry: ロング建値 (None なら建玉なし).
        short_entry: ショート建値 (None なら建玉なし).
        current_price: 現在価格.
        max_hold_days: 最大保有日数.
        held_days: 現在の保有日数.
        margin_usage_pct: 証拠金消費率 (% of account).
        margin_limit_pct: 証拠金消費率の上限.

    Returns:
        評価結果 dict:
        - should_close_long: ロングをクローズすべきか
        - should_close_short: ショートをクローズすべきか
        - reason: 理由
    """
    result: dict[str, object] = {
        "should_close_long": False,
        "should_close_short": False,
        "reason": "",
    }

    if long_entry is None and short_entry is None:
        result["reason"] = "両建てなし"
        return result

    if margin_usage_pct > margin_limit_pct:
        result["should_close_long"] = long_entry is not None
        result["should_close_short"] = short_entry is not None
        result["reason"] = f"証拠金消費率 {margin_usage_pct:.1f}% > 上限 {margin_limit_pct:.1f}%"
        return result

    if held_days >= max_hold_days:
        result["should_close_long"] = long_entry is not None
        result["should_close_short"] = short_entry is not None
        result["reason"] = f"保有日数 {held_days} >= 上限 {max_hold_days}"
        return result

    if long_entry is not None and short_entry is not None:
        if current_price > short_entry * 1.01:
            result["should_close_short"] = True
            result["reason"] = f"ショート建値 {short_entry:.3f} から 1% 超不利"
        elif current_price < long_entry * 0.99:
            result["should_close_long"] = True
            result["reason"] = f"ロング建値 {long_entry:.3f} から 1% 超不利"

    return result


if __name__ == "__main__":
    # サンプル実行: ステートマシン
    from datetime import datetime

    engine = RangeBreakoutEngine()
    engine.update_range(upper=150.5, lower=149.5, atr=0.5, ts=pd.Timestamp(datetime(2025, 1, 1)))
    engine.update_sr_levels(
        [
            SRLevel(
                price=150.5,
                kind="RESISTANCE",
                touches=4,
                first_touch=pd.Timestamp("2024-01-01"),
                last_touch=pd.Timestamp("2024-12-01"),
                strength=0.7,
                fractal_count=2,
            ),
        ]
    )
    engine.update_lt_direction("UP")

    # H4 ブレイク
    engine.process_h4_bar(close=151.0, timestamp=pd.Timestamp(datetime(2025, 1, 2)))
    print(f"ブレイク後ステート: {engine.state.value}")

    # 戻り確認
    engine.update_pullback(
        PullbackSignal(confirmed=True, pattern="PIN_BAR", timestamp=pd.Timestamp(datetime(2025, 1, 3)))
    )
    # 注文板 (フォールバックで pass)
    engine.update_orderbook(
        OrderBookSignal(
            available=False,
            buy_thickness_ratio=1.0,
            sentiment_long_pct=50.0,
            passes=True,
        )
    )

    # エントリー判定
    sig = engine.check_entry_conditions(close=150.7, timestamp=pd.Timestamp(datetime(2025, 1, 3)))
    print(f"エントリー判定: should_enter={sig.should_enter} side={sig.side}")
    print(f"  条件: {sig.conditions_passed}")
    print(f"  損切り: {sig.stop_loss:.3f}, 利食い: {sig.take_profit:.3f}")
