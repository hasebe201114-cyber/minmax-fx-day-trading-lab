"""EXP-FX000018 / SYS-FX024: グリッド戦略用 正式KPI判定パイプライン.

`research/EXP-FX000018/00-spec.md` §6 で事前登録したKPIゲート(必須10項目・参考2項目)
をそのまま実装する。既存の`evaluate_vol_breakout_dow_theory_kpi.evaluate_period()`は
シングルポジション・確定損益ベースのエクイティカーブを前提としており流用できないため、
以下の点を新規に実装した:

- エクイティカーブは`grid_portfolio_engine`が出力する**MTM(時価評価)系列**を使う
  (含み損を抱えたまま積み増す構造のDDを過小評価しないため、spec §4.3)
- **K7m(両建て証拠金消費率、合算方式・MTM equity比)を必須ゲートに追加**
- **K4mをPF≥1.5(勝率中立版)へ置換**(spec §6.1、グリッドは構造上ペイオフ<1.5)
- permutation testのクラスタキーを**ISO年-週**にする(spec §6.2、エピソード内依存の吸収)
- **独立グリッドエピソード数**を別ゲートとして追加

DD(ピーク比)・月次シャープ・PF・ペイオフ・K3mスケール不変判定は、既存の
`minmax_fx_dt.backtest.metrics` / `decision.criteria` の共通実装をそのまま再利用する。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from minmax_fx_dt.backtest.metrics import (
    max_drawdown, monthly_max_dd_pct, monthly_sharpe, payoff_ratio,
    peak_relative_max_dd_pct, peak_relative_monthly_max_dd_pct, profit_factor,
)
from minmax_fx_dt.backtest.permutation import permutation_test_block
from minmax_fx_dt.decision.criteria import compute_k3m_scale_invariant, compute_n_trades_effective

# spec §6 の閾値 (結果を見る前に固定)
KPI_THRESHOLDS = {
    "monthly_sharpe": 0.4,
    "profit_factor": 1.2,
    "monthly_expectancy_positive": True,
    "max_dd_monthly_pct": 10.0,
    "max_dd_yearly_pct": 20.0,
    "profit_factor_k4m_alt": 1.5,
    "spread_cost_multiplier": 3.0,
    "min_n_trades_effective": 300,
    "min_n_episodes": 100,
    "permutation_p_value": 0.05,
    "max_margin_usage_pct": 30.0,
    "max_consecutive_losses_alpha": 0.05,
}

KPI_GATE_TIER = {
    "monthly_sharpe": "必須",
    "profit_factor": "必須",
    "monthly_expectancy_positive": "必須",
    "max_dd_monthly_pct": "必須",
    "max_dd_yearly_pct": "必須",
    "profit_factor_k4m_alt": "必須",
    "spread_cost_multiplier": "必須",
    "min_n_trades_effective": "必須",
    "min_n_episodes": "必須",
    "permutation_p_value": "必須",
    "max_margin_usage_pct": "必須",
    "max_consecutive_losses": "参考",
}
INITIAL_CAPITAL_USD = 1000.0


def max_consecutive_losses(trades: list[dict]) -> int:
    worst = cur = 0
    for t in sorted(trades, key=lambda x: x["exit_time"]):
        if t["dollar_pnl"] < 0:
            cur += 1
            worst = max(worst, cur)
        else:
            cur = 0
    return worst


def evaluate_grid_period(period_name: str, sim: dict, *, seed: int = 42) -> dict:
    """`grid_portfolio_engine.simulate()`の出力を spec §6 のゲートで判定する."""
    trades = sim["trades"]
    eq = pd.DataFrame(sim["equity_curve"])
    eq["timestamp"] = pd.to_datetime(eq["time"], format="mixed", utc=True).dt.tz_localize(None)
    eq["equity"] = eq["balance"]                    # MTM (時価評価) — 判定に使う
    eq_realized = eq.copy()
    eq_realized["equity"] = eq_realized["realized_balance"]  # 確定損益のみ (参考)

    m_sharpe = monthly_sharpe(eq)
    dd_pct = peak_relative_max_dd_pct(eq)
    dd_monthly_pct = peak_relative_monthly_max_dd_pct(eq)
    _dd_usd, dd_pct_of_initial = max_drawdown(eq)
    dd_monthly_of_initial = monthly_max_dd_pct(eq, INITIAL_CAPITAL_USD)
    dd_pct_realized = peak_relative_max_dd_pct(eq_realized)
    dd_monthly_pct_realized = peak_relative_monthly_max_dd_pct(eq_realized)

    pnls = [t["dollar_pnl"] for t in trades]
    n_trades = len(trades)
    pf = profit_factor(pnls) if n_trades else 0.0
    payoff = payoff_ratio(pnls) if n_trades else 0.0
    max_losses = max_consecutive_losses(trades)
    win_rate = (sum(1 for p in pnls if p > 0) / n_trades) if n_trades else 0.0
    k3m = compute_k3m_scale_invariant(n_trades, win_rate, max_losses) if n_trades else None

    eq_i = eq.set_index("timestamp")
    monthly_pnl = eq_i["equity"].resample("ME").last().diff().dropna()
    monthly_expectancy_positive = bool(monthly_pnl.mean() > 0) if len(monthly_pnl) else False

    # K5m: 平均粗エッジ(R) / 平均コスト(R)
    if n_trades:
        mean_r_gross = float(np.mean([t["r_gross"] for t in trades]))
        mean_cost = float(np.mean([t["cost_r"] + t["commission_r"] for t in trades]))
        spread_cost_multiplier = mean_r_gross / mean_cost if mean_cost > 0 else None
    else:
        mean_r_gross = mean_cost = 0.0
        spread_cost_multiplier = None

    trades_per_currency: dict[str, int] = {}
    for t in trades:
        trades_per_currency[t["pair"]] = trades_per_currency.get(t["pair"], 0) + 1
    # spec §6.2: 依存構造は週ブロック順列で直接捕捉するため、n側の相関割引は行わない (T-07規約)
    n_eff = compute_n_trades_effective(trades_per_currency, n_trades, apply_correlation_discount=False)

    # 独立グリッドエピソード数 = 実際にトレードが発生した (通貨, 世代) の組み合わせ数
    n_episodes = len({(t["pair"], t["gen_id"]) for t in trades})

    # spec §6.2: 主検定 = ISO年-週ブロック (全通貨共通)、参考 = 日ブロック
    perm_week = perm_day = None
    if n_trades >= 4:
        entry_ts = [pd.Timestamp(t["entry_time"]) for t in trades]
        week_keys = [f"{ts.isocalendar()[0]}-W{ts.isocalendar()[1]:02d}" for ts in entry_ts]
        day_keys = [ts.strftime("%Y-%m-%d") for ts in entry_ts]
        perm_week = permutation_test_block(pnls, week_keys, seed=seed)
        perm_day = permutation_test_block(pnls, day_keys, seed=seed)

    perm_p = perm_week.p_value if perm_week else None
    k7m = sim["max_margin_sum_pct"]

    kpi_pass = {
        "monthly_sharpe": m_sharpe >= KPI_THRESHOLDS["monthly_sharpe"],
        "profit_factor": pf >= KPI_THRESHOLDS["profit_factor"],
        "monthly_expectancy_positive": monthly_expectancy_positive,
        "max_dd_monthly_pct": dd_monthly_pct <= KPI_THRESHOLDS["max_dd_monthly_pct"],
        "max_dd_yearly_pct": abs(dd_pct) <= KPI_THRESHOLDS["max_dd_yearly_pct"],
        "profit_factor_k4m_alt": pf >= KPI_THRESHOLDS["profit_factor_k4m_alt"],
        "spread_cost_multiplier": (spread_cost_multiplier or 0) >= KPI_THRESHOLDS["spread_cost_multiplier"],
        "min_n_trades_effective": n_eff >= KPI_THRESHOLDS["min_n_trades_effective"],
        "min_n_episodes": n_episodes >= KPI_THRESHOLDS["min_n_episodes"],
        "permutation_p_value": (perm_p is not None) and (perm_p < KPI_THRESHOLDS["permutation_p_value"]),
        "max_margin_usage_pct": k7m <= KPI_THRESHOLDS["max_margin_usage_pct"],
        "max_consecutive_losses": (k3m["pass_"] if k3m else False),
    }
    required = {k: v for k, v in kpi_pass.items() if KPI_GATE_TIER[k] == "必須"}
    reference = {k: v for k, v in kpi_pass.items() if KPI_GATE_TIER[k] == "参考"}

    outcome_breakdown: dict[str, dict] = {}
    for outcome in ("TP", "STOP", "WEEKEND", "MARK", "PERIOD_END"):
        sub = [t for t in trades if t["outcome"] == outcome]
        if sub:
            vals = [t["dollar_pnl"] for t in sub]
            outcome_breakdown[outcome] = {
                "n": len(sub), "share": round(len(sub) / n_trades, 4),
                "sum_usd": round(float(np.sum(vals)), 2),
                "mean_usd": round(float(np.mean(vals)), 4),
            }

    swap_total = float(np.sum([t["swap_usd"] for t in trades])) if n_trades else 0.0

    return {
        "period": period_name,
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "final_balance_usd": round(sim["final_balance_usd"], 2),
        "total_return_pct": round((sim["final_balance_usd"] / INITIAL_CAPITAL_USD - 1) * 100, 2),
        "ruined": sim["ruined"],
        "monthly_sharpe": round(m_sharpe, 3),
        "max_dd_pct": round(dd_pct, 2),
        "max_dd_monthly_pct": round(dd_monthly_pct, 2),
        "max_dd_pct_realized_only_reference": round(dd_pct_realized, 2),
        "max_dd_monthly_pct_realized_only_reference": round(dd_monthly_pct_realized, 2),
        "max_dd_pct_of_initial_capital_reference": round(dd_pct_of_initial, 2),
        "max_dd_monthly_pct_of_initial_capital_reference": round(dd_monthly_of_initial, 2),
        "profit_factor": round(pf, 3),
        "payoff_ratio_reference": round(payoff, 3),
        "max_consecutive_losses": max_losses,
        "k3m_scale_invariant": k3m,
        "monthly_expectancy_positive": monthly_expectancy_positive,
        "spread_cost_multiplier": round(spread_cost_multiplier, 2) if spread_cost_multiplier else None,
        "mean_r_gross": round(mean_r_gross, 5),
        "mean_cost_r": round(mean_cost, 5),
        "n_trades_effective": round(n_eff, 1),
        "n_episodes": n_episodes,
        "n_generations": sim["n_generations"],
        "permutation_p_week_block": round(perm_week.p_value, 4) if perm_week else None,
        "permutation_p_week_method": perm_week.method if perm_week else None,
        "permutation_p_day_block_reference": round(perm_day.p_value, 4) if perm_day else None,
        "k7m_margin_sum_pct": round(k7m, 2),
        "k7m_margin_max_method_reference": round(sim["max_margin_max_pct"], 2),
        "max_concurrent_positions": sim["max_concurrent_positions"],
        "guard_block_rate": round(sim["guard_block_rate"], 4),
        "n_entry_attempts": sim["n_entry_attempts"],
        "n_entry_blocked_margin": sim["n_entry_blocked_margin"],
        "n_entry_blocked_cap": sim["n_entry_blocked_cap"],
        "both_side_stop_events": sim["both_side_stop_events"],
        "swap_total_usd": round(swap_total, 2),
        "outcome_breakdown": outcome_breakdown,
        "trades_per_currency": trades_per_currency,
        "kpi_pass": kpi_pass,
        "kpi_required_pass_count": f"{sum(required.values())}/{len(required)}",
        "kpi_required_all_pass": all(required.values()),
        "kpi_reference_pass_count": f"{sum(reference.values())}/{len(reference)}",
    }


def print_period(r: dict) -> None:
    print(f"  トレード数={r['n_trades']}  勝率={r['win_rate']}  最終残高=${r['final_balance_usd']} "
          f"({r['total_return_pct']:+.1f}%)  破産={r['ruined']}")
    print(f"  月次シャープ={r['monthly_sharpe']}  最大DD(MTM・ピーク比)={r['max_dd_pct']}% "
          f"(月間{r['max_dd_monthly_pct']}%)  [参考:確定損益のみ={r['max_dd_pct_realized_only_reference']}%]")
    print(f"  PF={r['profit_factor']}  ペイオフ(参考)={r['payoff_ratio_reference']}  "
          f"最大連敗={r['max_consecutive_losses']}  コスト倍率={r['spread_cost_multiplier']}")
    print(f"  実効n={r['n_trades_effective']}  エピソード数={r['n_episodes']}/{r['n_generations']}  "
          f"perm_p(週)={r['permutation_p_week_block']} [参考 日={r['permutation_p_day_block_reference']}]")
    print(f"  K7m(合算)={r['k7m_margin_sum_pct']}%  [参考 MAX方式={r['k7m_margin_max_method_reference']}%]  "
          f"最大同時保有={r['max_concurrent_positions']}  ガード発動率={r['guard_block_rate']}")
    print(f"  スワップ合計=${r['swap_total_usd']}  両側同時ストップ={r['both_side_stop_events']}件")
    print(f"  outcome内訳: " + "  ".join(
        f"{k}:n={v['n']}({v['share']:.1%}) 平均${v['mean_usd']:+.2f}" for k, v in r["outcome_breakdown"].items()))
    ng = [k for k, v in r["kpi_pass"].items() if not v and KPI_GATE_TIER[k] == "必須"]
    print(f"  **必須KPI: {r['kpi_required_pass_count']}**  未達={ng if ng else 'なし'}")
