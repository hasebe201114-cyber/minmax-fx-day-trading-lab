"""EXP-FX000020: 拡張Train（2021-11〜2025-03、41ヶ月）での再評価.

`00-spec.md` §3〜§5 で拡張期間の損益を見る前に固定した設計をそのまま実行する。

対象（§3.1、4件で確定。後から追加しない）:
  SYS-FX011  … trailonly版（breakeven=1.0、H1トレンドフィルターなし）
  SYS-FX012  … 候補①（N_BREAKOUT + H1トレンド判定不能除外、breakeven=1.0）
  SYS-FX018  … SYS-FX012 + breakeven_trigger_r=2.0
  SYS-FX025  … SYS-FX011 × SYS-FX024 R-A の等ウェイト50:50合成

禁止事項（§3.3）: パラメータの再導出は一切行わない。KPI閾値も変更しない。

コスト（§2.2）: base_era_ratio / x1.5 / x2.0 の3水準すべてを実行し併記する。
結論が水準間で反転する場合は最も保守的な x2.0 を判定に用いる。

判定（§5、結果を見る前に固定）:
  1. 決着(有望)   … 実効n>=300 かつ 必須ゲート全達成 → Validation を参照し具申
  2. 決着(不採用) … 実効n>=300 だが 必須ゲートに未達 → REJECT 確定（本EXPの最大の価値）
  3. なお判定不能 … 実効n<300

出力: research/method-notes/extended_train_reevaluation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import extended_data as ext  # noqa: E402
from backtest_sysfx024_grid_trainonly import jsonable  # noqa: E402

COST_LEVELS = ["base_era_ratio", "sensitivity_x1.5", "sensitivity_x2.0"]
MIN_N = 300


def classify(n_eff: float, all_pass: bool) -> str:
    """spec §5 の3分類（結果を見る前に固定）."""
    if n_eff < MIN_N:
        return "3_なお判定不能"
    return "1_決着(有望)" if all_pass else "2_決着(不採用)"


def run_single_strategies(period_name: str, start: str, end: str) -> dict:
    """SYS-FX011 / 012 / 018 を同一期間で評価する."""
    import backtest_sysfx018_breakeven_sweep_trainonly as fx018
    import backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd as v7
    from evaluate_vol_breakout_dow_theory_kpi import evaluate_period

    out = {}
    specs = [
        ("SYS-FX011", lambda: v7.run_period(period_name, start, end)),
        ("SYS-FX012", lambda: fx018.run_period(1.0, start, end)),
        ("SYS-FX018", lambda: fx018.run_period(2.0, start, end)),
    ]
    for name, fn in specs:
        print(f"    [{name}] 実行中...", flush=True)
        p = fn()
        r = evaluate_period(name, p, perm_p_field="perm_p_block",
                            apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)
        r["n_trades"] = p["n_trades"]
        # run_period の実装によって最終残高キーの有無が異なる（SYS-FX012/018 の
        # スイート版は持たない）ため、無ければエクイティカーブ末尾から算出する
        if "final_balance_usd" in p:
            r["final_balance_usd"] = p["final_balance_usd"]
            r["total_return_pct"] = p["total_return_pct"]
        else:
            final = float(p["equity_curve"][-1]["balance"])
            init = float(p["equity_curve"][0]["balance"])
            r["final_balance_usd"] = round(final, 2)
            r["total_return_pct"] = round((final / init - 1) * 100, 2)
        r["classification"] = classify(r["n_trades_effective"], r["kpi_required_all_pass"])
        out[name] = r
        print(f"    [{name}] n={p['n_trades']} 実効n={r['n_trades_effective']} "
              f"最終=${r['final_balance_usd']} シャープ={r['monthly_sharpe']} "
              f"ペイオフ={r['payoff_ratio']} perm_p={r['permutation_p_clustered']} "
              f"必須KPI={r['kpi_required_pass_count']} → {r['classification']}", flush=True)
    return out


def run_blend(period_name: str, start: str, end: str) -> dict:
    """SYS-FX025（SYS-FX011 × SYS-FX024 R-A の等ウェイト合成）を評価する."""
    import backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd as v7
    from backtest_sysfx025_portfolio_blend import (
        PAIRS, RESULT_DIR, WEIGHT, daily_returns, equity_stats, evaluate_blend,
    )
    from grid_portfolio_engine import simulate

    gp = json.loads((RESULT_DIR / "grid_params.json").read_text(encoding="utf-8"))["derived"]
    gap = json.loads((RESULT_DIR / "weekend_gap_risk.json").read_text(encoding="utf-8"))["derived"]
    rf = json.loads((RESULT_DIR / "range_filter.json").read_text(encoding="utf-8"))["derived"]

    print("    [SYS-FX025] 構成要素を実行中...", flush=True)
    px = v7.run_period(period_name, start, end)
    simY = simulate(
        PAIRS, start, end, n_levels=gp["n_levels"], grid_step_atr_mult=gp["grid_step_atr_mult"],
        reanchor_bars=gp["reanchor_bars_h4"], carry_over=False, weekend_carry=True,
        max_hold_h4_bars=gp["reanchor_bars_h4"] * 2, rel_gap_p99=gap["rel_gap_p99"],
        weekend_gap_budget_pct=gap["weekend_gap_loss_budget_pct"],
        range_filter_er_max=rf["er_max"], range_filter_window=rf["lookback_window"], verbose=False,
    )
    eqX = daily_returns(px["equity_curve"], "balance", start, end)
    eqY = daily_returns(simY["equity_curve"], "balance", start, end)
    common = eqX.index.intersection(eqY.index)
    rX, rY = eqX.reindex(common).pct_change(), eqY.reindex(common).pct_change()
    blend_eq = (1.0 + (WEIGHT * rX + (1 - WEIGHT) * rY).dropna()).cumprod() * 1000.0
    trades = ([{**t, "dollar_pnl": t["dollar_pnl"] * WEIGHT} for t in px["trades"]]
              + [{**t, "dollar_pnl": t["dollar_pnl"] * (1 - WEIGHT)} for t in simY["trades"]])
    r = evaluate_blend("SYS-FX025", blend_eq, trades)
    r["correlation"] = round(float(pd.concat([rX, rY], axis=1).dropna().corr().iloc[0, 1]), 4)
    r["standalone_X"] = equity_stats(eqX.reindex(common))
    r["standalone_Y"] = equity_stats(eqY.reindex(common))
    r["n_trades_X"], r["n_trades_Y"] = len(px["trades"]), len(simY["trades"])
    r["classification"] = classify(r["n_trades_effective"], r["kpi_required_all_pass"])
    print(f"    [SYS-FX025] 相関={r['correlation']} 実効n={r['n_trades_effective']} "
          f"シャープ={r['monthly_sharpe']} DD={r['max_dd_pct']}% ペイオフ={r['payoff_ratio']} "
          f"PF={r['profit_factor']} perm_p(週)={r['permutation_p_week_block']} "
          f"必須KPI={r['kpi_required_pass_count']} → {r['classification']}", flush=True)
    return {"SYS-FX025": r}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", default=COST_LEVELS)
    ap.add_argument("--out", default="extended_train_reevaluation.json")
    args = ap.parse_args()

    start, end = ext.EXTENDED_TRAIN
    print("=== EXP-FX000020: 拡張Trainでの再評価 ===")
    print(f"拡張Train: {start} 〜 {end}（41ヶ月、現行17ヶ月の2.4倍）")
    print(f"対象: SYS-FX011 / SYS-FX012 / SYS-FX018 / SYS-FX025（spec §3.1、4件で確定）")
    print("パラメータ再導出なし・KPI閾値変更なし\n")

    cov = ext.coverage_report(["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"])
    junc = ext.junction_continuity(["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"])
    print("--- ゲート1: データ品質（spec §1.1） ---")
    for pair, d in cov.items():
        print(f"  [{pair}] 拡張部={d['n_bars_extension']:,}本/{d['n_days_extension']}日  "
              f"1日あたり中央値={d['median_bars_per_day_extension']:.0f}本"
              f"（現行期間={d['median_bars_per_day_current']:.0f}本、密度{d['density_vs_current_pct']}%）  "
              f"接続部ギャップ={junc.get(pair, {}).get('gap_pips')}pips")
    print()

    results: dict = {"coverage": cov, "junction": junc, "by_cost_level": {}}
    for level in args.levels:
        spreads = ext.patch_pipelines(level)
        print(f"--- コスト水準: {level}  スプレッド={spreads} ---", flush=True)
        r = run_single_strategies("extended_train", start, end)
        r.update(run_blend("extended_train", start, end))
        results["by_cost_level"][level] = {"spread_pips": spreads, "strategies": r}
        print()

    print("=== spec §5 の分類まとめ ===")
    print(f"{'戦略':<12}" + "".join(f"{lv:>26}" for lv in args.levels))
    for name in ["SYS-FX011", "SYS-FX012", "SYS-FX018", "SYS-FX025"]:
        row = f"{name:<12}"
        for lv in args.levels:
            s = results["by_cost_level"][lv]["strategies"].get(name, {})
            row += f"{s.get('classification', '-'):>26}"
        print(row)

    path = ROOT / "research" / "method-notes" / args.out
    path.write_text(json.dumps(jsonable({
        "generated_at": datetime.now().isoformat(),
        "exp_id": "EXP-FX000020", "spec_ref": "research/EXP-FX000020/00-spec.md",
        "extended_train": {"start": start, "end": end},
        "targets": ["SYS-FX011", "SYS-FX012", "SYS-FX018", "SYS-FX025"],
        "excluded_note": "SYS-FX023 は REJECT 確定済みのため spec §3.4 で対象外",
        **results,
    }), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[出力]: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
