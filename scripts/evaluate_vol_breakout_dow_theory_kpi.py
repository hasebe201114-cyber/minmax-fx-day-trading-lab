"""EXP-FX000005: ダウ理論連続押し目買い版(確定コストモデル)の正式KPI判定.

`backtest_vol_breakout_dow_theory_1000usd.py`の出力(1通貨1ポジション制約+
H1継続確認再開ロジック、エントリー=成行/SL・トレーリング・TP=指値/週末・
MAX_HOLD=成行という確定済みコストモデル)を対象に、`00-spec.md`のK1m〜K7m閾値
+実効サンプル数+permutation有意性の全ゲートで正式判定する。

出力: research/method-notes/vol_breakout_dow_theory_kpi_evaluation.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd

from minmax_fx_dt.backtest.metrics import (
    max_drawdown, monthly_max_dd_pct, monthly_sharpe, payoff_ratio, profit_factor,
)
from minmax_fx_dt.decision.criteria import compute_n_trades_effective

# 00-spec.md のKPI閾値
KPI_THRESHOLDS = {
    "monthly_sharpe": 0.4,
    "profit_factor": 1.2,
    "monthly_expectancy_positive": True,
    "max_dd_monthly_pct": 10.0,
    "max_dd_yearly_pct": 20.0,
    "max_consecutive_losses": 5,
    "payoff_ratio": 1.5,
    "spread_cost_multiplier": 3.0,
    "min_n_trades_effective": 300,
    "permutation_p_value": 0.05,
}
INITIAL_CAPITAL_USD = 1000.0


def max_consecutive_losses(trades: list[dict]) -> int:
    trades_sorted = sorted(trades, key=lambda t: t["entry_time"])
    worst = cur = 0
    for t in trades_sorted:
        if t["dollar_pnl"] < 0:
            cur += 1
            worst = max(worst, cur)
        else:
            cur = 0
    return worst


def evaluate_period(period_name: str, p: dict) -> dict:
    eq_curve = pd.DataFrame(p["equity_curve"])
    eq_curve["timestamp"] = pd.to_datetime(eq_curve["time"], format="mixed", utc=True).dt.tz_localize(None)
    eq_curve["equity"] = eq_curve["balance"]

    m_sharpe = monthly_sharpe(eq_curve)
    _dd_usd, dd_pct = max_drawdown(eq_curve)
    dd_monthly_pct = monthly_max_dd_pct(eq_curve, INITIAL_CAPITAL_USD)

    dollar_pnls = [t["dollar_pnl"] for t in p["trades"]]
    pf = profit_factor(dollar_pnls)
    payoff = payoff_ratio(dollar_pnls)
    max_losses = max_consecutive_losses(p["trades"])

    # 月次期待値: 月ごとのドルP&L平均が正か
    eq_curve_i = eq_curve.set_index("timestamp")
    monthly_pnl = eq_curve_i["equity"].resample("ME").last().diff().dropna()
    monthly_expectancy_positive = bool(monthly_pnl.mean() > 0) if len(monthly_pnl) > 0 else False

    # K5m: スプレッドコスト倍率 (平均粗エッジ ÷ 平均コスト)
    mean_r_gross = sum(t["r_gross"] for t in p["trades"]) / len(p["trades"])
    mean_cost = sum(t["cost_r"] + t["commission_r"] for t in p["trades"]) / len(p["trades"])
    spread_cost_multiplier = mean_r_gross / mean_cost if mean_cost > 0 else None

    trades_per_currency: dict[str, int] = {}
    for t in p["trades"]:
        trades_per_currency[t["pair"]] = trades_per_currency.get(t["pair"], 0) + 1
    n_eff = compute_n_trades_effective(trades_per_currency, len(p["trades"]))

    perm_p = p["perm_p_clustered"]

    kpi_pass = {
        "monthly_sharpe": m_sharpe >= KPI_THRESHOLDS["monthly_sharpe"],
        "profit_factor": pf >= KPI_THRESHOLDS["profit_factor"],
        "monthly_expectancy_positive": monthly_expectancy_positive,
        "max_dd_monthly_pct": dd_monthly_pct <= KPI_THRESHOLDS["max_dd_monthly_pct"],
        "max_dd_yearly_pct": abs(dd_pct) <= KPI_THRESHOLDS["max_dd_yearly_pct"],
        "max_consecutive_losses": max_losses <= KPI_THRESHOLDS["max_consecutive_losses"],
        "payoff_ratio": payoff >= KPI_THRESHOLDS["payoff_ratio"],
        "spread_cost_multiplier": (spread_cost_multiplier or 0) >= KPI_THRESHOLDS["spread_cost_multiplier"],
        "min_n_trades_effective": n_eff >= KPI_THRESHOLDS["min_n_trades_effective"],
        "permutation_p_value": (perm_p is not None) and (perm_p < KPI_THRESHOLDS["permutation_p_value"]),
    }

    return {
        "period": period_name,
        "monthly_sharpe": round(m_sharpe, 3),
        "max_dd_pct": round(dd_pct, 2),
        "max_dd_monthly_pct": round(dd_monthly_pct, 2),
        "profit_factor": round(pf, 3),
        "payoff_ratio": round(payoff, 3),
        "max_consecutive_losses": max_losses,
        "monthly_expectancy_positive": monthly_expectancy_positive,
        "spread_cost_multiplier": round(spread_cost_multiplier, 2) if spread_cost_multiplier else None,
        "n_trades_effective": round(n_eff, 1),
        "permutation_p_clustered": perm_p,
        "kpi_pass": kpi_pass,
        "kpi_pass_count": f"{sum(kpi_pass.values())}/{len(kpi_pass)}",
    }


def main() -> int:
    print("=== EXP-FX000005: SYS-FX011 ダウ理論連続押し目買い版(確定コストモデル) 正式KPI判定 ===\n")

    with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_1000usd_backtest.json").open(encoding="utf-8") as f:
        backtest = json.load(f)

    results = {}
    for period_name in ["train", "validation", "test"]:
        r = evaluate_period(period_name, backtest["periods"][period_name])
        results[period_name] = r
        print(f"--- {period_name} ---")
        print(f"  月次シャープ={r['monthly_sharpe']}  最大DD={r['max_dd_pct']}%(月間{r['max_dd_monthly_pct']}%)  "
              f"PF={r['profit_factor']}  ペイオフ={r['payoff_ratio']}  最大連敗={r['max_consecutive_losses']}")
        print(f"  スプレッドコスト倍率={r['spread_cost_multiplier']}  実効n={r['n_trades_effective']}  "
              f"perm_p={r['permutation_p_clustered']}")
        print(f"  KPI: {r['kpi_pass_count']} 達成  {r['kpi_pass']}\n")

    print("=== サマリ ===")
    print(f"{'期間':<12}{'Sharpe':>8}{'最大DD':>8}{'PF':>7}{'ペイオフ':>9}{'実効n':>8}{'perm_p':>8}{'KPI達成':>9}")
    for name in ["train", "validation", "test"]:
        r = results[name]
        print(f"{name:<12}{r['monthly_sharpe']:>8}{r['max_dd_pct']:>7.2f}%{r['profit_factor']:>7}"
              f"{r['payoff_ratio']:>9}{r['n_trades_effective']:>8}{str(r['permutation_p_clustered']):>8}"
              f"{r['kpi_pass_count']:>9}")

    all_pass = all(r["kpi_pass_count"].split("/")[0] == r["kpi_pass_count"].split("/")[1] for r in results.values())
    print(f"\n3期間すべてで全KPI達成: {'はい' if all_pass else 'いいえ'}")

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_kpi_evaluation.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "kpi_thresholds": KPI_THRESHOLDS,
            "periods": results,
            "all_periods_all_kpi_pass": all_pass,
            "_note": (
                "確定コストモデル(エントリー=成行/SL・トレーリング・TP=指値/週末・MAX_HOLD=成行)、"
                "$1,000初期資金・1%リスクサイジングでのTVT結果を対象にK1m〜K7m相当+実効n+"
                "permutation有意性の全ゲートで正式判定。K6m(フォワード比較)・K7m(両建て証拠金)"
                "は本戦略の現行設計では判定対象外。"
            ),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
