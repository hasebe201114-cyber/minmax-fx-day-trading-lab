"""トレード・シミュレータ (SYS-FX007 レンジブレイク・プルバック戦略向け).

両建て状態を含むポジション管理、エントリー/エグジット、損益計算。

主な責務:
    - 両建て対応ポジション管理 (FLAT / LONG / SHORT / BOTH)
    - スプレッド + スリッページのコスト適用
    - 週末強制クローズ制約
    - 証拠金消費率の追跡
    - 取引履歴の記録
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd

from minmax_fx_dt.strategy.range_breakout import EntrySignal


class PositionSide(str, Enum):
    """ポジション種別."""

    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


@dataclass
class Trade:
    """個別の取引記録."""

    side: str              # "BUY" / "SELL"
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    size: int = 0          # 通貨単位
    stop_loss: float = 0.0
    take_profit: float = 0.0
    pnl: float = 0.0        # 確定損益 (価格差分 + スワップ)
    pnl_pips: float = 0.0  # 損益 (pips, 価格差分のみ)
    swap_pnl: float = 0.0  # スワップ損益 (JPY, v2.1 OBS000004 差し戻し 2 で追加)
    hold_days: int = 0     # 保有日数 (スワップ計算用)
    exit_reason: str = ""  # "TP" / "SL" / "REVERSE" / "WEEKEND" / "END"
    entry_conditions: Optional[dict[str, bool]] = None  # 4 条件 AND の成立状況 (v2.1 で MT-3 恒久フォールバック)
    lt_direction: str = ""  # エントリー時の LT 方向
    sr_level_price: float = 0.0  # エントリー時の S/R ライン価格（あれば）
    initial_risk: float = 0.0  # |entry_price - 初期ストップ| (SYS-FX009 1:1ターゲット算出用)
    target_reached: bool = False  # 1:1ターゲット到達済みか (SYS-FX009、到達後はブレイクイーブン+トレーリングへ切替)


@dataclass
class PortfolioState:
    """ポートフォリオ状態."""

    cash: float            # 現金残高 (JPY)
    initial_cash: float    # 初期残高
    position_side: PositionSide = PositionSide.FLAT
    long_trade: Optional[Trade] = None
    short_trade: Optional[Trade] = None
    total_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    max_equity: float = 0.0
    max_dd_jpy: float = 0.0
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    trade_history: list[Trade] = field(default_factory=list)


def apply_slippage(price: float, side: str, slippage_pips: float, is_jpy_pair: bool) -> float:
    """スリッページを適用.

    Args:
        price: 約定価格.
        side: "BUY" / "SELL".
        slippage_pips: スリッページ (pips).
        is_jpy_pair: JPY ペアか (0.01 が 1 pip).

    Returns:
        スリッページ適用後価格.
    """
    pip_value = 0.01 if is_jpy_pair else 0.0001
    slip_amount = slippage_pips * pip_value
    if side == "BUY":
        return price + slip_amount
    return price - slip_amount


def calc_pnl(
    entry_price: float,
    exit_price: float,
    size: int,
    side: str,
    is_jpy_pair: bool,
) -> tuple[float, float]:
    """損益を計算 (価格差分のみ、スワップは含まない).

    Args:
        entry_price: エントリー価格.
        exit_price: エグジット価格.
        size: 通貨単位.
        side: "BUY" / "SELL".
        is_jpy_pair: JPY ペアか.

    Returns:
        (pnl_jpy, pnl_pips).
    """
    pip_value = 0.01 if is_jpy_pair else 0.0001
    if side == "BUY":
        diff = exit_price - entry_price
    else:
        diff = entry_price - exit_price
    pnl_pips = diff / pip_value
    # JPY 換算: 通貨単位 × 価格差 (JPY ペア) または × pip_value × 100 (非 JPY ペア)
    if is_jpy_pair:
        pnl_jpy = diff * size
    else:
        # USD 換算 → JPY 換算 (USD/JPY 想定)。簡略化のため 150.0 固定
        pnl_jpy = diff * size * 150.0
    return pnl_jpy, pnl_pips


def calc_swap(
    side: str,
    size: int,
    lot_size: int,
    hold_days: int,
    swap_long_jpy_per_lot_per_day: float,
    swap_short_jpy_per_lot_per_day: float,
) -> float:
    """スワップ損益を計算 (JPY).

    Args:
        side: "BUY" / "SELL".
        size: 通貨単位.
        lot_size: 1 ロットあたりの通貨単位.
        hold_days: 保有日数.
        swap_long_jpy_per_lot_per_day: ロング 1 ロット 1 日あたりスワップ (JPY, 受取が正).
        swap_short_jpy_per_lot_per_day: ショート 1 ロット 1 日あたりスワップ (JPY, 受取が正).

    Returns:
        スワップ損益 (JPY).
    """
    if hold_days <= 0:
        return 0.0
    lots = size / lot_size if lot_size > 0 else 0
    if side == "BUY":
        per_day = swap_long_jpy_per_lot_per_day
    else:
        per_day = swap_short_jpy_per_lot_per_day
    return per_day * lots * hold_days


@dataclass
class SimulatorConfig:
    """シミュレータ設定."""

    initial_cash_jpy: float = 1_000_000.0  # 100 万円
    lot_size: int = 1_000                   # 1,000 通貨
    spread_pips: float = 0.3                # USD/JPY 想定 (銭 → 0.3 pips)
    slippage_pips: float = 0.5
    is_jpy_pair: bool = True
    weekend_close: bool = True              # 土曜 06:00 JST までに全決済
    max_hold_days: int = 20
    max_dd_pause_threshold_pct: float = 50.0  # DD が証拠金の 50% で停止
    # スワップ (v2.1 OBS000004 差し戻し 2 で追加)
    # 0.0 = スワップなし (デフォルト、後方互換)
    swap_long_jpy_per_lot_per_day: float = 0.0
    swap_short_jpy_per_lot_per_day: float = 0.0


def spread_round_trip_cost_jpy(config: SimulatorConfig) -> float:
    """スプレッド往復コスト (JPY / 1トレード) を SimulatorConfig の実際の設定値から算出.

    OBS000007 独立監査 B6: 旧実装は `spread_round_trip_jpy=60.0` を
    全通貨ペア共通でハードコードしており (metrics.py の呼び出し元)、
    is_jpy_pair 修正で spread_pips を通貨別にしたにもかかわらず、
    K5m (スプレッド往復コスト比較) はその変更を反映していなかった。
    open_long/open_short/close_long/close_short の実装 (エントリー・
    エグジットの双方で spread_pips 全額を課金) と整合する値を返す。
    """
    pip_value = 0.01 if config.is_jpy_pair else 0.0001
    jpy_rate = 1.0 if config.is_jpy_pair else 150.0
    # エントリー時 + エグジット時に spread_pips 全額ずつ課金される実装のため、
    # 往復コスト = 2 x spread_pips
    return 2.0 * config.spread_pips * pip_value * config.lot_size * jpy_rate


def open_long(
    state: PortfolioState,
    config: SimulatorConfig,
    entry_price: float,
    entry_time: pd.Timestamp,
    stop_loss: float,
    take_profit: float,
) -> Trade:
    """ロング・ポジションを開く.

    スプレッド・スリッページを適用後、state.long_trade を更新.
    """
    fill_price = apply_slippage(entry_price, "BUY", config.slippage_pips, config.is_jpy_pair)
    fill_price += config.spread_pips * (0.01 if config.is_jpy_pair else 0.0001)  # ask で約定
    trade = Trade(
        side="BUY",
        entry_time=entry_time,
        entry_price=fill_price,
        size=config.lot_size,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    state.long_trade = trade
    if state.position_side == PositionSide.SHORT:
        state.position_side = PositionSide.BOTH
    else:
        state.position_side = PositionSide.LONG
    return trade


def open_short(
    state: PortfolioState,
    config: SimulatorConfig,
    entry_price: float,
    entry_time: pd.Timestamp,
    stop_loss: float,
    take_profit: float,
) -> Trade:
    """ショート・ポジションを開く."""
    fill_price = apply_slippage(entry_price, "SELL", config.slippage_pips, config.is_jpy_pair)
    fill_price -= config.spread_pips * (0.01 if config.is_jpy_pair else 0.0001)  # bid で約定
    trade = Trade(
        side="SELL",
        entry_time=entry_time,
        entry_price=fill_price,
        size=config.lot_size,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    state.short_trade = trade
    if state.position_side == PositionSide.LONG:
        state.position_side = PositionSide.BOTH
    else:
        state.position_side = PositionSide.SHORT
    return trade


def close_long(
    state: PortfolioState,
    config: SimulatorConfig,
    exit_price: float,
    exit_time: pd.Timestamp,
    reason: str,
) -> Optional[Trade]:
    """ロングを決済."""
    if state.long_trade is None:
        return None
    fill_price = apply_slippage(exit_price, "SELL", config.slippage_pips, config.is_jpy_pair)
    fill_price -= config.spread_pips * (0.01 if config.is_jpy_pair else 0.0001)  # bid で決済
    pnl_jpy, pnl_pips = calc_pnl(
        state.long_trade.entry_price, fill_price,
        state.long_trade.size, "BUY", config.is_jpy_pair,
    )
    state.long_trade.exit_time = exit_time
    state.long_trade.exit_price = fill_price
    state.long_trade.pnl_pips = pnl_pips
    state.long_trade.exit_reason = reason
    # v2.1: スワップ計算
    hold_days = max(1, (exit_time - state.long_trade.entry_time).days) if state.long_trade.entry_time is not None else 0
    state.long_trade.hold_days = hold_days
    swap_pnl = calc_swap(
        side="BUY",
        size=state.long_trade.size,
        lot_size=config.lot_size,
        hold_days=hold_days,
        swap_long_jpy_per_lot_per_day=config.swap_long_jpy_per_lot_per_day,
        swap_short_jpy_per_lot_per_day=config.swap_short_jpy_per_lot_per_day,
    )
    state.long_trade.swap_pnl = swap_pnl
    total_pnl = pnl_jpy + swap_pnl
    state.long_trade.pnl = total_pnl
    state.cash += total_pnl
    state.total_trades += 1
    if pnl_jpy > 0:
        state.win_trades += 1
        state.consecutive_losses = 0
    else:
        state.loss_trades += 1
        state.consecutive_losses += 1
        state.max_consecutive_losses = max(state.max_consecutive_losses, state.consecutive_losses)
    state.trade_history.append(state.long_trade)
    state.long_trade = None
    if state.position_side == PositionSide.BOTH:
        state.position_side = PositionSide.SHORT
    else:
        state.position_side = PositionSide.FLAT
    return state.trade_history[-1]


def close_short(
    state: PortfolioState,
    config: SimulatorConfig,
    exit_price: float,
    exit_time: pd.Timestamp,
    reason: str,
) -> Optional[Trade]:
    """ショートを決済."""
    if state.short_trade is None:
        return None
    fill_price = apply_slippage(exit_price, "BUY", config.slippage_pips, config.is_jpy_pair)
    fill_price += config.spread_pips * (0.01 if config.is_jpy_pair else 0.0001)
    pnl_jpy, pnl_pips = calc_pnl(
        state.short_trade.entry_price, fill_price,
        state.short_trade.size, "SELL", config.is_jpy_pair,
    )
    state.short_trade.exit_time = exit_time
    state.short_trade.exit_price = fill_price
    state.short_trade.pnl_pips = pnl_pips
    state.short_trade.exit_reason = reason
    # v2.1: スワップ計算
    hold_days = max(1, (exit_time - state.short_trade.entry_time).days) if state.short_trade.entry_time is not None else 0
    state.short_trade.hold_days = hold_days
    swap_pnl = calc_swap(
        side="SELL",
        size=state.short_trade.size,
        lot_size=config.lot_size,
        hold_days=hold_days,
        swap_long_jpy_per_lot_per_day=config.swap_long_jpy_per_lot_per_day,
        swap_short_jpy_per_lot_per_day=config.swap_short_jpy_per_lot_per_day,
    )
    state.short_trade.swap_pnl = swap_pnl
    total_pnl = pnl_jpy + swap_pnl
    state.short_trade.pnl = total_pnl
    state.cash += total_pnl
    state.total_trades += 1
    if pnl_jpy > 0:
        state.win_trades += 1
        state.consecutive_losses = 0
    else:
        state.loss_trades += 1
        state.consecutive_losses += 1
        state.max_consecutive_losses = max(state.max_consecutive_losses, state.consecutive_losses)
    state.trade_history.append(state.short_trade)
    state.short_trade = None
    if state.position_side == PositionSide.BOTH:
        state.position_side = PositionSide.LONG
    else:
        state.position_side = PositionSide.FLAT
    return state.trade_history[-1]


def unrealized_pnl(state: PortfolioState, current_price: float, is_jpy_pair: bool) -> float:
    """含み損益を計算."""
    pnl = 0.0
    if state.long_trade is not None:
        _, pips = calc_pnl(
            state.long_trade.entry_price, current_price,
            state.long_trade.size, "BUY", is_jpy_pair,
        )
        pnl += pips * (0.01 if is_jpy_pair else 0.0001) * state.long_trade.size * (1 if is_jpy_pair else 150.0)
    if state.short_trade is not None:
        _, pips = calc_pnl(
            state.short_trade.entry_price, current_price,
            state.short_trade.size, "SELL", is_jpy_pair,
        )
        pnl += pips * (0.01 if is_jpy_pair else 0.0001) * state.short_trade.size * (1 if is_jpy_pair else 150.0)
    return pnl


def margin_usage_pct(
    state: PortfolioState,
    current_price: float,
    is_jpy_pair: bool,
    leverage: float = 25.0,
) -> float:
    """証拠金消費率 (% of initial cash).

    必要証拠金 = 通貨単位 × 価格 / レバレッジ (JPY 換算).
    両建て時は max(必要証拠金_long, 必要証拠金_short) で計算。

    OBS000007 独立監査 B7: 旧実装は is_jpy_pair を受け取りながら未使用で、
    非 JPY ペア (例: EUR/USD) では `size × price` が USD 建てのまま
    `initial_cash` (JPY) と比較されており、証拠金消費率が実質 150 分の1
    程度に過小評価されていた。calc_pnl と同じ簡略化 (150.0 固定) で
    JPY 換算する。
    """
    jpy_rate = 1.0 if is_jpy_pair else 150.0
    long_margin = 0.0
    short_margin = 0.0
    if state.long_trade is not None:
        long_margin = state.long_trade.size * current_price / leverage * jpy_rate
    if state.short_trade is not None:
        short_margin = state.short_trade.size * current_price / leverage * jpy_rate
    used = max(long_margin, short_margin) if (state.long_trade and state.short_trade) else (long_margin + short_margin)
    return (used / state.initial_cash) * 100.0


def update_equity(
    state: PortfolioState,
    current_price: float,
    timestamp: pd.Timestamp,
    is_jpy_pair: bool,
) -> None:
    """エクイティを更新し、DD を追跡."""
    unrealized = unrealized_pnl(state, current_price, is_jpy_pair)
    equity = state.cash + unrealized
    state.equity_curve.append((timestamp, equity))
    state.max_equity = max(state.max_equity, equity)
    dd = state.max_equity - equity
    state.max_dd_jpy = max(state.max_dd_jpy, dd)


def is_weekend_close_time(timestamp: pd.Timestamp) -> bool:
    """土曜 06:00 JST かどうか (週末強制クローズ判定)."""
    if not isinstance(timestamp, pd.Timestamp):
        return False
    # 土曜日 = dayofweek 5, 06:00 未満
    return timestamp.dayofweek == 5 and timestamp.hour < 6


def maybe_force_weekend_close(
    state: PortfolioState,
    config: SimulatorConfig,
    timestamp: pd.Timestamp,
    current_price: float,
) -> None:
    """週末強制クローズ (Q7 制約)."""
    if not config.weekend_close:
        return
    if is_weekend_close_time(timestamp):
        if state.long_trade is not None:
            close_long(state, config, current_price, timestamp, "WEEKEND")
        if state.short_trade is not None:
            close_short(state, config, current_price, timestamp, "WEEKEND")


def check_stop_loss_take_profit(
    state: PortfolioState,
    config: SimulatorConfig,
    high: float,
    low: float,
    timestamp: pd.Timestamp,
) -> list[str]:
    """SL / TP チェック.

    Returns:
        決済された理由のリスト.
    """
    reasons: list[str] = []
    # ロング
    if state.long_trade is not None:
        if low <= state.long_trade.stop_loss:
            close_long(state, config, state.long_trade.stop_loss, timestamp, "SL")
            reasons.append("LONG_SL")
        elif high >= state.long_trade.take_profit:
            close_long(state, config, state.long_trade.take_profit, timestamp, "TP")
            reasons.append("LONG_TP")
    # ショート
    if state.short_trade is not None:
        if high >= state.short_trade.stop_loss:
            close_short(state, config, state.short_trade.stop_loss, timestamp, "SL")
            reasons.append("SHORT_SL")
        elif low <= state.short_trade.take_profit:
            close_short(state, config, state.short_trade.take_profit, timestamp, "TP")
            reasons.append("SHORT_TP")
    return reasons


def is_dd_paused(state: PortfolioState, config: SimulatorConfig) -> bool:
    """最大 DD が閾値を超えて停止状態か."""
    dd_pct = (state.max_dd_jpy / state.initial_cash) * 100.0
    return dd_pct >= config.max_dd_pause_threshold_pct


__all__ = [
    "PositionSide",
    "Trade",
    "PortfolioState",
    "SimulatorConfig",
    "spread_round_trip_cost_jpy",
    "apply_slippage",
    "calc_pnl",
    "open_long",
    "open_short",
    "close_long",
    "close_short",
    "unrealized_pnl",
    "margin_usage_pct",
    "update_equity",
    "is_weekend_close_time",
    "maybe_force_weekend_close",
    "check_stop_loss_take_profit",
    "is_dd_paused",
]
