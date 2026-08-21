"""バックテスト KPI 計算 (SYS-FX007 レンジブレイク・プルバック戦略向け).

K1m〜K7m および v2 拡張 KPI を計算する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from minmax_fx_dt.backtest.simulator import PortfolioState, Trade


@dataclass
class BacktestMetrics:
    """バックテスト結果の KPI."""

    # K1m: 月次シャープレシオ
    sharpe_monthly: float
    sharpe_yearly: float

    # K1m: 月次 Profit Factor
    profit_factor_monthly: float
    profit_factor_yearly: float

    # 期待値 (JPY / トレード)
    expectancy_jpy: float

    # K2m: 最大 DD
    max_dd_jpy: float
    max_dd_monthly_pct: float      # 月間 (証拠金 %)
    max_dd_yearly_pct: float       # 年間 (証拠金 %)

    # K3m: 最大連続損失
    max_consecutive_losses: int

    # K4m: ペイオフレシオ
    payoff_ratio: float

    # K5m: 1トレード期待値 / スプレッド往復
    spread_round_trip_jpy: float
    edge_per_trade_jpy: float

    # K6m: バックテスト ↔ フォワード乖離率 (フォワードテスト時のみ計算)
    backtest_forward_divergence_pct: float

    # K7m: 両建て証拠金消費率
    max_margin_usage_pct: float

    # v2 拡張
    weak_breakout_exclusion_pct: float  # 弱いブレイクの排除率
    # OBS000007 独立監査 B2: エンジンが should_enter=True と判定したが、
    # 既に同方向ポジションを保有中で実際には建てられなかった (=シグナルが
    # 黙って破棄された) 件数。エンジンの状態はこの時点で PULLBACK_CONFIRMED
    # に進んでしまうため、保有期間が長期化すると見えない機会損失になりうる。
    signals_dropped_position_open: int
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float

    # メタ
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    equity_curve: pd.DataFrame  # columns: timestamp, equity


def compute_returns(equity_curve: pd.DataFrame) -> pd.Series:
    """エクイティカーブから日次リターンを計算."""
    if len(equity_curve) < 2:
        return pd.Series(dtype=float)
    eq = equity_curve.set_index("timestamp")["equity"]
    return eq.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """年率シャープレシオを計算.

    Args:
        returns: 期間リターン系列.
        periods_per_year: 1 年あたりの期間数 (デフォルト 252 営業日).
        risk_free: リスクフリーレート (期間あたり).

    Returns:
        年率シャープレシオ.
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free
    if excess.std() == 0 or np.isnan(excess.std()):
        return 0.0
    return float(excess.mean() / excess.std() * math.sqrt(periods_per_year))


def monthly_sharpe(equity_curve: pd.DataFrame) -> float:
    """月次シャープレシオ (K1m)."""
    if len(equity_curve) < 2:
        return 0.0
    eq = equity_curve.set_index("timestamp")["equity"].resample("ME").last()
    monthly_returns = eq.pct_change().dropna()
    if len(monthly_returns) < 2:
        return 0.0
    if monthly_returns.std() == 0:
        return 0.0
    return float(monthly_returns.mean() / monthly_returns.std() * math.sqrt(12))


def profit_factor(trade_pnls: list[float]) -> float:
    """Profit Factor = 総利益 / 総損失 (絶対値)."""
    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def max_drawdown(equity_curve: pd.DataFrame) -> tuple[float, float]:
    """最大 DD を計算 (JPY と、初期資金比の%).

    DD定義ルール(2026-08-21、外部レビューT-10対応。恒久ルール、
    `PJ000004-基本データ層と検証プロセス定義.md`「Q9」参照): **固定ロット
    サイジング(残高に依存せず1トレードのリスク量が一定)の戦略にのみ使う。**
    複利サイジング(残高の一定%をリスクに充てる)の戦略はリスク量自体が残高に
    比例して変動するため、この関数ではなく`peak_relative_max_dd_pct()`を使うこと。
    SYS-FX007〜010は固定ロット方式のためこの関数が正しい定義（過去判定の
    遡及的な再評価は不要と2026-08-21に確認済み）。"""
    if len(equity_curve) < 2:
        return 0.0, 0.0
    eq = equity_curve["equity"].to_numpy()
    running_max = np.maximum.accumulate(eq)
    drawdown = running_max - eq
    max_dd_jpy = float(drawdown.max()) if len(drawdown) > 0 else 0.0
    initial = eq[0] if len(eq) > 0 else 1.0
    max_dd_pct = (max_dd_jpy / initial) * 100.0 if initial > 0 else 0.0
    return max_dd_jpy, max_dd_pct


def monthly_max_dd_pct(equity_curve: pd.DataFrame, initial_cash: float) -> float:
    """月次の最大 DD (初期資金比の%)。`max_drawdown()`と同じ適用条件
    (固定ロットサイジングのみ、DD定義ルールはそちらのdocstring参照)。"""
    if len(equity_curve) < 2:
        return 0.0
    eq = equity_curve.set_index("timestamp")["equity"]
    monthly_max = eq.resample("ME").max()
    monthly_dd = []
    for month_end, mmax in monthly_max.items():
        # 当月末までの累積最大
        prev_max = eq.loc[:month_end].max()
        dd = (prev_max - mmax) / initial_cash * 100.0
        monthly_dd.append(dd)
    return float(max(monthly_dd)) if monthly_dd else 0.0


def peak_relative_max_dd_pct(equity_curve: pd.DataFrame) -> float:
    """最大DD(直近ピーク比の%)。複利サイジング戦略向け(DD定義ルールは
    `max_drawdown()`のdocstring参照)。SYS-FX011(EXP-FX000005)で
    `risk_pct_per_trade`(残高の1%)による複利サイジングを採用した際、初期
    資金比では複利成長後の変動を実態より過大に見せてしまう問題が発覚し
    (2026-08-20、司令塔指摘)、`scripts/evaluate_vol_breakout_dow_theory_kpi.py`
    に一時的にローカル実装されていたものを、T-10対応(2026-08-21)で全SYS共通
    のこのモジュールへ昇格した。

    `equity_curve`は`timestamp`・`equity`列を持つDataFrameを想定する。"""
    if len(equity_curve) < 2:
        return 0.0
    eq = equity_curve.set_index("timestamp")["equity"]
    running_max = eq.cummax()
    dd = (running_max - eq) / running_max * 100.0
    return float(dd.max()) if len(dd) > 0 else 0.0


def peak_relative_monthly_max_dd_pct(equity_curve: pd.DataFrame) -> float:
    """月次の最大DD(直近ピーク比の%)。複利サイジング戦略向け(`peak_relative_max_dd_pct()`
    参照)。"""
    if len(equity_curve) < 2:
        return 0.0
    eq = equity_curve.set_index("timestamp")["equity"]
    running_max = eq.cummax()
    dd = (running_max - eq) / running_max * 100.0
    monthly_max = dd.resample("ME").max().dropna()
    return float(monthly_max.max()) if len(monthly_max) > 0 else 0.0


def payoff_ratio(trade_pnls: list[float]) -> float:
    """ペイオフレシオ = 平均利益 / 平均損失 (絶対値)."""
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    if not wins or not losses:
        return 0.0
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))
    if avg_loss == 0:
        return float("inf")
    return avg_win / avg_loss


def compute_metrics(
    state: PortfolioState,
    *,
    spread_round_trip_jpy: float = 60.0,  # 1pip × 1000通貨 = 10円/片道 → 20円/往復 (USD/JPY 想定)
    weak_breakout_signals: int = 0,
    weak_breakout_blocked: int = 0,
    backtest_forward_divergence_pct: float = 0.0,
    max_margin_usage_pct: float = 0.0,
    signals_dropped_position_open: int = 0,
) -> BacktestMetrics:
    """バックテストの総合 KPI を計算.

    Args:
        state: ポートフォリオ状態.
        spread_round_trip_jpy: スプレッド往復コスト (JPY / 1,000通貨 / 1トレード).
        weak_breakout_signals: 弱いブレイクのシグナル総数 (デバッグ用).
        weak_breakout_blocked: 弱いブレイクでブロックされた数.
        backtest_forward_divergence_pct: BT/FT 乖離率 (フォワード時のみ).
        max_margin_usage_pct: 両建て状態の最大証拠金消費率.
        signals_dropped_position_open: 既存ポジションと同方向のため破棄されたシグナル数 (OBS000007 B2)。

    Returns:
        BacktestMetrics.
    """
    # エクイティカーブを DataFrame に
    if not state.equity_curve:
        # 空の場合
        eq_df = pd.DataFrame(columns=["timestamp", "equity"])
    else:
        ts_list, eq_list = zip(*state.equity_curve)
        eq_df = pd.DataFrame({"timestamp": list(ts_list), "equity": list(eq_list)})

    # 取引 PnL
    pnls = [t.pnl for t in state.trade_history]
    pips = [t.pnl_pips for t in state.trade_history]

    # K1m: シャープレシオ
    sharpe_m = monthly_sharpe(eq_df)
    eq_returns = compute_returns(eq_df)
    sharpe_y = sharpe_ratio(eq_returns, periods_per_year=252)

    # K1m: Profit Factor
    pf_m = profit_factor(pnls)
    pf_y = pf_m  # 累積なので同じ

    # 期待値
    expectancy = sum(pnls) / len(pnls) if pnls else 0.0

    # K2m: 最大 DD
    max_dd_jpy, max_dd_total_pct = max_drawdown(eq_df)
    max_dd_monthly = monthly_max_dd_pct(eq_df, state.initial_cash)
    max_dd_yearly = max_dd_monthly  # 簡略化: 月次の max を流用

    # K3m: 最大連続損失
    max_consec = state.max_consecutive_losses

    # K4m: ペイオフレシオ
    pr = payoff_ratio(pnls)

    # K5m: 1トレード期待値 / スプレッド往復
    edge_per_trade = expectancy

    # K7m: 両建て証拠金消費率
    margin_pct = max_margin_usage_pct

    # v2 拡張: 弱いブレイクの排除率
    if weak_breakout_signals > 0:
        we = (weak_breakout_blocked / weak_breakout_signals) * 100.0
    else:
        we = 0.0

    # 期間
    if not state.equity_curve:
        period_start = pd.Timestamp("2000-01-01")
        period_end = pd.Timestamp("2000-01-01")
    else:
        period_start = state.equity_curve[0][0]
        period_end = state.equity_curve[-1][0]

    return BacktestMetrics(
        sharpe_monthly=sharpe_m,
        sharpe_yearly=sharpe_y,
        profit_factor_monthly=pf_m,
        profit_factor_yearly=pf_y,
        expectancy_jpy=expectancy,
        max_dd_jpy=max_dd_jpy,
        max_dd_monthly_pct=max_dd_monthly,
        max_dd_yearly_pct=max_dd_yearly,
        max_consecutive_losses=max_consec,
        payoff_ratio=pr,
        spread_round_trip_jpy=spread_round_trip_jpy,
        edge_per_trade_jpy=edge_per_trade,
        backtest_forward_divergence_pct=backtest_forward_divergence_pct,
        max_margin_usage_pct=margin_pct,
        weak_breakout_exclusion_pct=we,
        signals_dropped_position_open=signals_dropped_position_open,
        n_trades=len(pnls),
        n_wins=state.win_trades,
        n_losses=state.loss_trades,
        win_rate=state.win_trades / max(1, state.total_trades) * 100.0,
        period_start=period_start,
        period_end=period_end,
        equity_curve=eq_df,
    )


def to_dict(m: BacktestMetrics) -> dict:
    """KPI を dict 形式に変換 (JSON シリアライズ可能)."""
    return {
        "sharpe_monthly": round(m.sharpe_monthly, 4),
        "sharpe_yearly": round(m.sharpe_yearly, 4),
        "profit_factor_monthly": round(m.profit_factor_monthly, 4),
        "profit_factor_yearly": round(m.profit_factor_yearly, 4),
        "expectancy_jpy": round(m.expectancy_jpy, 2),
        "max_dd_jpy": round(m.max_dd_jpy, 2),
        "max_dd_monthly_pct": round(m.max_dd_monthly_pct, 4),
        "max_dd_yearly_pct": round(m.max_dd_yearly_pct, 4),
        "max_consecutive_losses": m.max_consecutive_losses,
        "payoff_ratio": round(m.payoff_ratio, 4),
        "spread_round_trip_jpy": round(m.spread_round_trip_jpy, 2),
        "edge_per_trade_jpy": round(m.edge_per_trade_jpy, 2),
        "backtest_forward_divergence_pct": round(m.backtest_forward_divergence_pct, 4),
        "max_margin_usage_pct": round(m.max_margin_usage_pct, 4),
        "weak_breakout_exclusion_pct": round(m.weak_breakout_exclusion_pct, 4),
        "signals_dropped_position_open": m.signals_dropped_position_open,
        "n_trades": m.n_trades,
        "n_wins": m.n_wins,
        "n_losses": m.n_losses,
        "win_rate": round(m.win_rate, 4),
        "period_start": m.period_start.isoformat() if hasattr(m.period_start, "isoformat") else str(m.period_start),
        "period_end": m.period_end.isoformat() if hasattr(m.period_end, "isoformat") else str(m.period_end),
    }


__all__ = [
    "BacktestMetrics",
    "compute_returns",
    "sharpe_ratio",
    "monthly_sharpe",
    "profit_factor",
    "max_drawdown",
    "monthly_max_dd_pct",
    "peak_relative_max_dd_pct",
    "peak_relative_monthly_max_dd_pct",
    "payoff_ratio",
    "compute_metrics",
    "to_dict",
]
