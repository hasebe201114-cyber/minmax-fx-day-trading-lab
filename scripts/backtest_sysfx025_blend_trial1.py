"""EXP-FX000019 amendment-01: 改善ループ第1試行 — 戦略X を SYS-FX018 版へ差し替えた合成.

`00-spec-amendment-01.md` §2〜§3 で結果を見る前に固定した設計をそのまま実行する。

  P0（ベースライン、既出）: 戦略X = SYS-FX011 trailonly（breakeven=1.0）
  P1（第1試行）          : 戦略X = SYS-FX018 版（breakeven_trigger_r=2.0、EXP-FX000012 確定値）

戦略Y（SYS-FX024 R-A）・配分ウェイト 50:50・KPIゲートはいずれも**一切変更しない**。
新規パラメータの導入はゼロ。

出力: research/method-notes/sysfx025_portfolio_blend_trial1.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from backtest_sysfx018_breakeven_sweep_trainonly import run_period as run_fx018  # noqa: E402
from backtest_sysfx024_grid_trainonly import jsonable  # noqa: E402
from backtest_sysfx025_portfolio_blend import (  # noqa: E402
    KPI_THRESHOLDS, PAIRS, PERIODS, RESULT_DIR, WEIGHT,
    daily_returns, equity_stats, evaluate_blend,
)
from grid_portfolio_engine import simulate  # noqa: E402

BREAKEVEN_TRIGGER_R = 2.0  # EXP-FX000012(SYS-FX018) の確定値。本EXPでは探索しない


def main() -> int:
    gp = json.loads((RESULT_DIR / "grid_params.json").read_text(encoding="utf-8"))["derived"]
    gap = json.loads((RESULT_DIR / "weekend_gap_risk.json").read_text(encoding="utf-8"))["derived"]
    rf = json.loads((RESULT_DIR / "range_filter.json").read_text(encoding="utf-8"))["derived"]

    print("=== EXP-FX000019 改善ループ第1試行: 戦略X = SYS-FX018版(breakeven=2.0) ===")
    print(f"配分 {WEIGHT:.0%}:{1-WEIGHT:.0%}（不変）  戦略Y = SYS-FX024 R-A（不変）")
    print(f"breakeven_trigger_r = {BREAKEVEN_TRIGGER_R}（EXP-FX000012 確定値、再探索しない）\n")

    out: dict = {}
    for period, (start, end) in PERIODS.items():
        print(f"--- {period}: {start} 〜 {end} ---", flush=True)
        px = run_fx018(BREAKEVEN_TRIGGER_R, start, end)
        simY = simulate(
            PAIRS, start, end,
            n_levels=gp["n_levels"], grid_step_atr_mult=gp["grid_step_atr_mult"],
            reanchor_bars=gp["reanchor_bars_h4"], carry_over=False,
            weekend_carry=True, max_hold_h4_bars=gp["reanchor_bars_h4"] * 2,
            rel_gap_p99=gap["rel_gap_p99"], weekend_gap_budget_pct=gap["weekend_gap_loss_budget_pct"],
            range_filter_er_max=rf["er_max"], range_filter_window=rf["lookback_window"],
            verbose=False,
        )
        eqX = daily_returns(px["equity_curve"], "balance", start, end)
        eqY = daily_returns(simY["equity_curve"], "balance", start, end)
        common = eqX.index.intersection(eqY.index)
        rX, rY = eqX.reindex(common).pct_change(), eqY.reindex(common).pct_change()
        blend_eq = (1.0 + (WEIGHT * rX + (1 - WEIGHT) * rY).dropna()).cumprod() * 1000.0
        corr = float(pd.concat([rX, rY], axis=1).dropna().corr().iloc[0, 1])

        trades = ([{**t, "dollar_pnl": t["dollar_pnl"] * WEIGHT} for t in px["trades"]]
                  + [{**t, "dollar_pnl": t["dollar_pnl"] * (1 - WEIGHT)} for t in simY["trades"]])

        sX, sY = equity_stats(eqX.reindex(common)), equity_stats(eqY.reindex(common))
        res = evaluate_blend(f"P1_{period}", blend_eq, trades)
        res.update(correlation=round(corr, 4), standalone_X_sysfx018=sX,
                   standalone_Y_sysfx024_RA=sY, n_trades_X=len(px["trades"]),
                   n_trades_Y=len(simY["trades"]),
                   standalone_X_payoff=px.get("payoff_ratio"), standalone_X_pf=px.get("profit_factor"))
        best_sharpe = max(sX["monthly_sharpe"], sY["monthly_sharpe"])
        best_dd = min(sX["max_dd_pct"], sY["max_dd_pct"])
        cond_a, cond_b = res["monthly_sharpe"] >= best_sharpe, res["max_dd_pct"] <= best_dd
        res["blend_improves_sharpe"], res["blend_improves_dd"] = cond_a, cond_b
        res["verdict_3stage"] = ("再現した" if (cond_a and cond_b)
                                 else "部分的に再現" if (cond_a or cond_b) else "再現せず")

        print(f"  相関={corr:+.4f}  共通営業日={len(common)}")
        print(f"  単独X(SYS-FX018版, n={len(px['trades'])}): {sX['total_return_pct']:+.2f}%  "
              f"シャープ={sX['monthly_sharpe']}  DD={sX['max_dd_pct']}%  "
              f"ペイオフ={px.get('payoff_ratio')}")
        print(f"  単独Y(SYS-FX024 R-A, n={len(simY['trades'])}): {sY['total_return_pct']:+.2f}%  "
              f"シャープ={sY['monthly_sharpe']}  DD={sY['max_dd_pct']}%")
        print(f"  **合成P1**: {res['total_return_pct']:+.2f}%  シャープ={res['monthly_sharpe']}  "
              f"DD={res['max_dd_pct']}%(月間{res['max_dd_monthly_pct']}%)")
        print(f"    → シャープ改善={cond_a}  DD改善={cond_b}  **判定={res['verdict_3stage']}**")
        print(f"  合成KPI: {res['kpi_required_pass_count']}  PF={res['profit_factor']}  "
              f"**ペイオフ={res['payoff_ratio']}**  実効n={res['n_trades_effective']}  "
              f"perm_p(週)={res['permutation_p_week_block']} [参考 日={res['permutation_p_day_block_reference']}]")
        ng = [k for k, v in res["kpi_pass"].items() if not v]
        print(f"    未達: {ng if ng else 'なし'}\n", flush=True)
        out[period] = res

    v = out["validation"]
    print("=== amendment-01 §3 の選定ルール適用（Validation） ===")
    print(f"  3段階判定: **{v['verdict_3stage']}**  必須KPI: {v['kpi_required_pass_count']}")
    if v["verdict_3stage"] == "再現した" and v["kpi_required_all_pass"]:
        print("  → **必須9項目すべて達成。ポートフォリオ採用候補として司令塔に具申**")
    elif v["verdict_3stage"] == "再現した":
        print("  → 未達項目を明記の上、改善ループ第2試行（最後の1回）へ")
    else:
        print("  → P0 に劣ると判定。P0 の結果を最終結論として扱う")

    path = ROOT / "research" / "method-notes" / "sysfx025_portfolio_blend_trial1.json"
    path.write_text(json.dumps(jsonable({
        "generated_at": datetime.now().isoformat(),
        "exp_id": "EXP-FX000019", "sys_id": "SYS-FX025",
        "spec_ref": "research/EXP-FX000019/00-spec-amendment-01.md",
        "loop_trial_number": 1, "loop_trial_limit": 2,
        "strategy_x": f"SYS-FX018版 (breakeven_trigger_r={BREAKEVEN_TRIGGER_R})",
        "strategy_y": "SYS-FX024 R-A (凍結)", "weight": WEIGHT,
        "kpi_thresholds": KPI_THRESHOLDS, "results": out,
        "validation_verdict": v["verdict_3stage"],
    }), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[出力]: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
