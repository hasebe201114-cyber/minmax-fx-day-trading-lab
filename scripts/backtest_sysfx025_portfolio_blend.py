"""EXP-FX000019 / SYS-FX025: SYS-FX011 × SYS-FX024 の等ウェイト合成の Validation 再現性検証.

`00-spec.md` §2〜§4 で Validation を見る前に固定した設計をそのまま実行する:

  - 配分は **等ウェイト 50:50・日次リバランス**（結果を見てから変えない）
  - 構成要素は**いずれも凍結済み設計**。SYS-FX024 R-A のパラメータ（N=3・k=1.72×ATR・
    R=24・ER≤0.2013・rel_gap_p99）は Train 導出値をそのまま Validation へ適用し、
    再導出しない
  - 合成トレード列は両戦略のトレードを結合し `dollar_pnl` を 0.5 倍
  - permutation test のクラスタキーは ISO年-週（保守側）
  - 必須KPIゲートは単独戦略と同一（緩めない、PJ000004 Q13）

判定は spec §4 の3段階（再現した / 部分的に再現 / 再現せず）で機械的に行う。

出力: research/method-notes/sysfx025_portfolio_blend.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from backtest_sysfx024_grid_trainonly import jsonable  # noqa: E402
from grid_portfolio_engine import simulate  # noqa: E402
from minmax_fx_dt.backtest.metrics import (  # noqa: E402
    monthly_sharpe, payoff_ratio, peak_relative_max_dd_pct,
    peak_relative_monthly_max_dd_pct, profit_factor,
)
from minmax_fx_dt.backtest.permutation import permutation_test_block  # noqa: E402
from minmax_fx_dt.decision.criteria import (  # noqa: E402
    compute_k3m_scale_invariant, compute_n_trades_effective,
)

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
PERIODS = {
    "train": ("2023-11-01", "2025-03-31"),
    "validation": ("2025-04-01", "2025-11-30"),
}
FX011_JSON = ROOT / "research" / "method-notes" / \
    "vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd_backtest.json"
RESULT_DIR = ROOT / "research" / "EXP-FX000018" / "10-result"

KPI_THRESHOLDS = {
    "monthly_sharpe": 0.4, "profit_factor": 1.2, "monthly_expectancy_positive": True,
    "max_dd_monthly_pct": 10.0, "max_dd_yearly_pct": 20.0, "payoff_ratio": 1.5,
    "spread_cost_multiplier": 3.0, "min_n_trades_effective": 300, "permutation_p_value": 0.05,
}
WEIGHT = 0.5  # spec §2: 等ウェイト（Validation を見る前に固定）


def daily_returns(equity_curve: list[dict], field: str, start: str, end: str) -> pd.Series:
    df = pd.DataFrame(equity_curve)
    ts = pd.to_datetime(df["time"], format="mixed", utc=True).dt.tz_localize(None)
    eq = pd.Series(df[field].astype(float).to_numpy(), index=ts).sort_index()
    eq = eq[(eq.index >= start) & (eq.index <= end)]
    return eq.resample("D").last().ffill().dropna()


def equity_stats(daily_eq: pd.Series) -> dict:
    df = pd.DataFrame({"timestamp": daily_eq.index, "equity": daily_eq.to_numpy()})
    return {
        "n_days": int(len(daily_eq)),
        "total_return_pct": round(float(daily_eq.iloc[-1] / daily_eq.iloc[0] - 1) * 100, 2),
        "monthly_sharpe": round(monthly_sharpe(df), 3),
        "max_dd_pct": round(peak_relative_max_dd_pct(df), 2),
        "max_dd_monthly_pct": round(peak_relative_monthly_max_dd_pct(df), 2),
    }


def evaluate_blend(label: str, blended_eq: pd.Series, trades: list[dict]) -> dict:
    """合成エクイティ＋合成トレード列に、単独戦略と同一の必須KPIゲートを適用する."""
    st = equity_stats(blended_eq)
    pnls = [t["dollar_pnl"] for t in trades]
    n = len(trades)
    pf = profit_factor(pnls) if n else 0.0
    payoff = payoff_ratio(pnls) if n else 0.0

    df = pd.DataFrame({"timestamp": blended_eq.index, "equity": blended_eq.to_numpy()})
    monthly = df.set_index("timestamp")["equity"].resample("ME").last().diff().dropna()
    monthly_positive = bool(monthly.mean() > 0) if len(monthly) else False

    mean_r_gross = float(np.mean([t["r_gross"] for t in trades])) if n else 0.0
    mean_cost = float(np.mean([t["cost_r"] + t["commission_r"] for t in trades])) if n else 0.0
    k5m = mean_r_gross / mean_cost if mean_cost > 0 else None

    per_pair: dict[str, int] = {}
    for t in trades:
        per_pair[t["pair"]] = per_pair.get(t["pair"], 0) + 1
    n_eff = compute_n_trades_effective(per_pair, n, apply_correlation_discount=False)

    perm = perm_day = None
    if n >= 4:
        keys = [f"{pd.Timestamp(t['entry_time']).isocalendar()[0]}-W"
                f"{pd.Timestamp(t['entry_time']).isocalendar()[1]:02d}" for t in trades]
        perm = permutation_test_block(pnls, keys, seed=42)
        # 参考: 日ブロック(プロジェクト従来規約 T-06)。**判定には使わない**が、SYS-FX011 単独の
        # 記録値(Train perm_p=0.035)が日ブロックで算出されているため、比較可能な数字を併記する。
        # spec §3.2 で事前登録したとおり、ゲートはあくまで保守側の週ブロックで判定する。
        perm_day = permutation_test_block(
            pnls, [pd.Timestamp(t["entry_time"]).strftime("%Y-%m-%d") for t in trades], seed=42)

    worst = cur = 0
    for t in sorted(trades, key=lambda x: str(x["exit_time"])):
        cur = cur + 1 if t["dollar_pnl"] < 0 else 0
        worst = max(worst, cur)
    win_rate = (sum(1 for p in pnls if p > 0) / n) if n else 0.0
    k3m = compute_k3m_scale_invariant(n, win_rate, worst) if n else None

    kpi = {
        "monthly_sharpe": st["monthly_sharpe"] >= KPI_THRESHOLDS["monthly_sharpe"],
        "profit_factor": pf >= KPI_THRESHOLDS["profit_factor"],
        "monthly_expectancy_positive": monthly_positive,
        "max_dd_monthly_pct": st["max_dd_monthly_pct"] <= KPI_THRESHOLDS["max_dd_monthly_pct"],
        "max_dd_yearly_pct": st["max_dd_pct"] <= KPI_THRESHOLDS["max_dd_yearly_pct"],
        "payoff_ratio": payoff >= KPI_THRESHOLDS["payoff_ratio"],
        "spread_cost_multiplier": (k5m or 0) >= KPI_THRESHOLDS["spread_cost_multiplier"],
        "min_n_trades_effective": n_eff >= KPI_THRESHOLDS["min_n_trades_effective"],
        "permutation_p_value": (perm is not None) and (perm.p_value < KPI_THRESHOLDS["permutation_p_value"]),
    }
    return {
        "label": label, **st, "n_trades": n, "win_rate": round(win_rate, 4),
        "profit_factor": round(pf, 3), "payoff_ratio": round(payoff, 3),
        "spread_cost_multiplier": round(k5m, 2) if k5m else None,
        "n_trades_effective": round(n_eff, 1),
        "permutation_p_week_block": round(perm.p_value, 4) if perm else None,
        "permutation_p_day_block_reference": round(perm_day.p_value, 4) if perm_day else None,
        "permutation_n_week_clusters": perm.method if perm else None,
        "max_consecutive_losses": worst, "k3m_scale_invariant": k3m,
        "monthly_expectancy_positive": monthly_positive,
        "kpi_pass": kpi,
        "kpi_required_pass_count": f"{sum(kpi.values())}/{len(kpi)}",
        "kpi_required_all_pass": all(kpi.values()),
    }


def main() -> int:
    fx011 = json.loads(FX011_JSON.read_text(encoding="utf-8"))
    gp = json.loads((RESULT_DIR / "grid_params.json").read_text(encoding="utf-8"))["derived"]
    gap = json.loads((RESULT_DIR / "weekend_gap_risk.json").read_text(encoding="utf-8"))["derived"]
    rf = json.loads((RESULT_DIR / "range_filter.json").read_text(encoding="utf-8"))["derived"]

    print("=== EXP-FX000019 / SYS-FX025: SYS-FX011 × SYS-FX024 等ウェイト合成 ===")
    print(f"配分: 等ウェイト {WEIGHT:.0%}:{1-WEIGHT:.0%}（spec §2、Validation を見る前に固定）")
    print(f"戦略Y凍結パラメータ: N={gp['n_levels']} k={gp['grid_step_atr_mult']} "
          f"R={gp['reanchor_bars_h4']} ER<={rf['er_max']}（再導出しない）\n")

    out: dict = {}
    for period, (start, end) in PERIODS.items():
        print(f"--- {period}: {start} 〜 {end} ---")
        simY = simulate(
            PAIRS, start, end,
            n_levels=gp["n_levels"], grid_step_atr_mult=gp["grid_step_atr_mult"],
            reanchor_bars=gp["reanchor_bars_h4"], carry_over=False,
            weekend_carry=True, max_hold_h4_bars=gp["reanchor_bars_h4"] * 2,
            rel_gap_p99=gap["rel_gap_p99"], weekend_gap_budget_pct=gap["weekend_gap_loss_budget_pct"],
            range_filter_er_max=rf["er_max"], range_filter_window=rf["lookback_window"],
            verbose=False,
        )
        eqX = daily_returns(fx011["periods"][period]["equity_curve"], "balance", start, end)
        eqY = daily_returns(simY["equity_curve"], "balance", start, end)
        common = eqX.index.intersection(eqY.index)
        rX, rY = eqX.reindex(common).pct_change(), eqY.reindex(common).pct_change()
        blend_ret = (WEIGHT * rX + (1 - WEIGHT) * rY).dropna()
        blend_eq = (1.0 + blend_ret).cumprod() * 1000.0
        corr = float(pd.concat([rX, rY], axis=1).dropna().corr().iloc[0, 1])

        tX = [{**t, "dollar_pnl": t["dollar_pnl"] * WEIGHT} for t in fx011["periods"][period]["trades"]]
        tY = [{**t, "dollar_pnl": t["dollar_pnl"] * (1 - WEIGHT)} for t in simY["trades"]]
        blended_trades = tX + tY

        sX, sY = equity_stats(eqX.reindex(common)), equity_stats(eqY.reindex(common))
        res = evaluate_blend(f"blend_{period}", blend_eq, blended_trades)
        res["correlation"] = round(corr, 4)
        res["standalone_X_sysfx011"] = sX
        res["standalone_Y_sysfx024_RA"] = sY
        res["n_trades_X"], res["n_trades_Y"] = len(tX), len(tY)

        # spec §4 の3段階判定（結果を見る前に固定済み）
        best_sharpe = max(sX["monthly_sharpe"], sY["monthly_sharpe"])
        best_dd = min(sX["max_dd_pct"], sY["max_dd_pct"])
        cond_a = res["monthly_sharpe"] >= best_sharpe
        cond_b = res["max_dd_pct"] <= best_dd
        res["blend_improves_sharpe"], res["blend_improves_dd"] = cond_a, cond_b
        res["verdict_3stage"] = ("再現した" if (cond_a and cond_b)
                                 else "部分的に再現" if (cond_a or cond_b) else "再現せず")

        print(f"  相関={corr:+.4f}   共通営業日={len(common)}")
        print(f"  単独X(SYS-FX011): {sX['total_return_pct']:+.2f}%  シャープ={sX['monthly_sharpe']}  DD={sX['max_dd_pct']}%")
        print(f"  単独Y(SYS-FX024 R-A): {sY['total_return_pct']:+.2f}%  シャープ={sY['monthly_sharpe']}  DD={sY['max_dd_pct']}%")
        print(f"  **合成**: {res['total_return_pct']:+.2f}%  シャープ={res['monthly_sharpe']}  "
              f"DD={res['max_dd_pct']}%(月間{res['max_dd_monthly_pct']}%)")
        print(f"    → シャープ改善={cond_a}  DD改善={cond_b}  **判定={res['verdict_3stage']}**")
        print(f"  合成KPI: {res['kpi_required_pass_count']}  PF={res['profit_factor']} "
              f"ペイオフ={res['payoff_ratio']} 実効n={res['n_trades_effective']} "
              f"perm_p(週)={res['permutation_p_week_block']} "
              f"[参考 日={res['permutation_p_day_block_reference']}]")
        ng = [k for k, v in res["kpi_pass"].items() if not v]
        print(f"    未達: {ng if ng else 'なし'}\n")
        out[period] = res

    v = out["validation"]
    print("=== spec §4 の選定ルール適用（Validation） ===")
    print(f"  3段階判定: **{v['verdict_3stage']}**")
    if v["verdict_3stage"] == "再現した":
        print(f"  必須9項目すべて達成: {'はい' if v['kpi_required_all_pass'] else 'いいえ'}")
        print("  → " + ("ポートフォリオ採用候補として司令塔に具申" if v["kpi_required_all_pass"]
                       else "未達項目を明記の上、改善ループ（上限2回）へ"))
    else:
        print("  → **合成効果は Train 固有のアーティファクトと判定。REJECT を具申**")
        print("     （spec §4 により、ウェイトを変えての再試行は行わない）")

    path = ROOT / "research" / "method-notes" / "sysfx025_portfolio_blend.json"
    path.write_text(json.dumps(jsonable({
        "generated_at": datetime.now().isoformat(),
        "exp_id": "EXP-FX000019", "sys_id": "SYS-FX025",
        "spec_ref": "research/EXP-FX000019/00-spec.md",
        "weight": WEIGHT, "pairs": PAIRS, "periods": PERIODS,
        "kpi_thresholds": KPI_THRESHOLDS,
        "results": out,
        "validation_verdict": v["verdict_3stage"],
    }), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[出力]: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
