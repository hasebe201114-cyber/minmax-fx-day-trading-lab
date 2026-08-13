"""バックテストエンジン.

SYS-FX007 レンジブレイク・プルバック戦略の検証用:
    simulator  - ポジション管理 (両建て対応、スリッページ、週末クローズ)
    metrics    - KPI 計算 (K1m〜K7m、v2 拡張)
    runner     - メインランナー (MTF 評価 + シミュレータ + KPI)
"""

from minmax_fx_dt.backtest.simulator import (
    PortfolioState,
    PositionSide,
    SimulatorConfig,
    Trade,
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
from minmax_fx_dt.backtest.metrics import (
    BacktestMetrics,
    compute_metrics,
    max_drawdown,
    monthly_sharpe,
    payoff_ratio,
    profit_factor,
    to_dict,
)
from minmax_fx_dt.backtest.runner import BacktestResult, run_backtest

__all__ = [
    # simulator
    "Trade", "PortfolioState", "PositionSide", "SimulatorConfig",
    "open_long", "open_short", "close_long", "close_short",
    "update_equity", "margin_usage_pct", "maybe_force_weekend_close",
    "check_stop_loss_take_profit", "is_dd_paused",
    # metrics
    "BacktestMetrics", "compute_metrics", "to_dict",
    "monthly_sharpe", "profit_factor", "max_drawdown", "payoff_ratio",
    # runner
    "BacktestResult", "run_backtest",
]
