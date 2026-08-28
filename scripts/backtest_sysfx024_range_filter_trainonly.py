"""EXP-FX000018 amendment-02: レンジ相場フィルター(ER)の Train 評価 (改善ループ第2試行).

`00-spec-amendment-02.md` §3 で事前登録した4候補を、フェーズゲート2の凍結パラメータ
(N=3・k=1.72×ATR・R=24、再導出しない)・4通貨・Train期間で評価する。

  B-A / W-A: フィルターなし (amendment-01 §8 の既出結果、比較用に再計測)
  R-B / R-A: フィルターあり (改善ループ第2試行、Q10 上限7回のうち2回目)

amendment-02 §4.1 の必須報告項目 (フィルターが「予測力」で効いたのか、単に
「トレード数が減った」だけなのかの切り分け) も併せて出力する。

出力: research/method-notes/sysfx024_range_filter_trainonly_backtest.json
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

import numpy as np  # noqa: E402

from backtest_sysfx024_grid_trainonly import jsonable  # noqa: E402
from evaluate_grid_kpi import KPI_THRESHOLDS, evaluate_grid_period, print_period  # noqa: E402
from grid_portfolio_engine import simulate  # noqa: E402

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
MAX_HOLD_MULTIPLE_OF_R = 2

CANDIDATES = {
    "B-A": {"weekend_carry": False, "filter": False, "label": "週末フラット × G0 × フィルターなし（既出）"},
    "W-A": {"weekend_carry": True, "filter": False, "label": "週末持越可 × G0 × フィルターなし（既出）"},
    "R-B": {"weekend_carry": False, "filter": True, "label": "週末フラット × G0 × **レンジフィルターあり**"},
    "R-A": {"weekend_carry": True, "filter": True, "label": "週末持越可 × G0 × **レンジフィルターあり**"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=TRAIN_START)
    ap.add_argument("--end", default=TRAIN_END)
    ap.add_argument("--out", default="sysfx024_range_filter_trainonly_backtest.json")
    args = ap.parse_args()

    d = ROOT / "research" / "EXP-FX000018" / "10-result"
    gp = json.loads((d / "grid_params.json").read_text(encoding="utf-8"))["derived"]
    gap = json.loads((d / "weekend_gap_risk.json").read_text(encoding="utf-8"))["derived"]
    rf = json.loads((d / "range_filter.json").read_text(encoding="utf-8"))["derived"]
    n_levels, k, r = gp["n_levels"], gp["grid_step_atr_mult"], gp["reanchor_bars_h4"]
    max_hold = r * MAX_HOLD_MULTIPLE_OF_R

    print("=== EXP-FX000018 amendment-02: レンジ相場フィルター Train評価 (改善ループ第2試行) ===")
    print(f"対象通貨: {PAIRS}   期間: {args.start} 〜 {args.end}")
    print(f"凍結パラメータ(再導出しない): N={n_levels}段  k={k}×ATR(H4,14)  R={r}本(H4)")
    print(f"レンジフィルター: ER(W={rf['lookback_window']}本, H4終値・無方向) ≤ {rf['er_max']} "
          f"のアンカーでのみグリッドを張る (閾値=Train 4通貨プールの中央値、損益非依存)")
    print()

    results, sims = {}, {}
    for name, cfg in CANDIDATES.items():
        print(f"--- {name}: {cfg['label']} ---")
        kwargs = dict(n_levels=n_levels, grid_step_atr_mult=k, reanchor_bars=r,
                      carry_over=False, verbose=False)
        if cfg["weekend_carry"]:
            kwargs.update(weekend_carry=True, max_hold_h4_bars=max_hold,
                          rel_gap_p99=gap["rel_gap_p99"],
                          weekend_gap_budget_pct=gap["weekend_gap_loss_budget_pct"])
        if cfg["filter"]:
            kwargs.update(range_filter_er_max=rf["er_max"], range_filter_window=rf["lookback_window"])
        sim = simulate(PAIRS, args.start, args.end, **kwargs)
        res = evaluate_grid_period(name, sim)
        res["label"] = cfg["label"]
        res["range_filter"] = cfg["filter"]
        res["n_generations_created"] = sim["n_generations"]
        res["mean_dollar_pnl"] = round(float(np.mean([t["dollar_pnl"] for t in sim["trades"]])), 4)
        res["mean_r_net"] = round(float(np.mean([t["r_net"] for t in sim["trades"]])), 5)
        stops = [t for t in sim["trades"] if t["outcome"] == "STOP"]
        res["n_stop"] = len(stops)
        res["stop_mean_usd"] = round(float(np.mean([t["dollar_pnl"] for t in stops])), 4) if stops else None
        res["stop_sum_usd"] = round(float(np.sum([t["dollar_pnl"] for t in stops])), 2) if stops else 0.0
        print_period(res)
        print(f"  世代数={res['n_generations_created']}  1トレード平均=${res['mean_dollar_pnl']:+.4f} "
              f"(平均r_net={res['mean_r_net']:+.5f})  STOP={res['n_stop']}件 合計${res['stop_sum_usd']:+.2f}")
        print()
        results[name], sims[name] = res, sim

    def req_count(x: dict) -> int:
        return int(x["kpi_required_pass_count"].split("/")[0])

    best = max(CANDIDATES, key=lambda nm: (req_count(results[nm]), results[nm]["monthly_sharpe"]))
    train_pass = results[best]["kpi_required_all_pass"]

    print("=== サマリ ===")
    print(f"{'候補':<6}{'週末':<9}{'ﾌｨﾙﾀ':<7}{'世代':>6}{'n':>7}{'勝率':>8}{'最終残高':>11}"
          f"{'1件平均':>10}{'Sharpe':>9}{'最大DD':>9}{'PF':>8}{'perm_p':>9}{'必須KPI':>9}")
    for name in CANDIDATES:
        x = results[name]
        wk = "持越可" if CANDIDATES[name]["weekend_carry"] else "フラット"
        fl = "あり" if CANDIDATES[name]["filter"] else "なし"
        print(f"{name:<6}{wk:<9}{fl:<7}{x['n_generations_created']:>6}{x['n_trades']:>7}{x['win_rate']:>8.3f}"
              f"{x['final_balance_usd']:>10.2f}${x['mean_dollar_pnl']:>+10.4f}{x['monthly_sharpe']:>9.3f}"
              f"{x['max_dd_pct']:>8.2f}%{x['profit_factor']:>8.3f}"
              f"{str(x['permutation_p_week_block']):>9}{x['kpi_required_pass_count']:>9}")

    print(f"\n選定ルール(必須KPI達成数 → 同数なら月次シャープ)による採用候補: **{best}**")
    print(f"Train通過(必須11項目すべて達成): {'はい' if train_pass else 'いいえ'}")
    if not train_pass:
        print(f"未達項目: {[k_ for k_, v in results[best]['kpi_pass'].items() if not v]}")

    # amendment-02 §4.1: 「予測力で効いた」のか「件数が減っただけ」なのかの切り分け
    print("\n=== §4.1 切り分け: フィルターは1トレードあたりの質を改善したか ===")
    for base, filt in (("B-A", "R-B"), ("W-A", "R-A")):
        b, f = results[base], results[filt]
        print(f"  {base} → {filt}:  世代 {b['n_generations_created']}→{f['n_generations_created']}  "
              f"n {b['n_trades']}→{f['n_trades']}  "
              f"**1件平均 ${b['mean_dollar_pnl']:+.4f}→${f['mean_dollar_pnl']:+.4f}**  "
              f"平均r_net {b['mean_r_net']:+.5f}→{f['mean_r_net']:+.5f}  "
              f"STOP {b['n_stop']}件→{f['n_stop']}件")

    out = ROOT / "research" / "method-notes" / args.out
    out.write_text(json.dumps(jsonable({
        "generated_at": datetime.now().isoformat(),
        "exp_id": "EXP-FX000018", "sys_id": "SYS-FX024",
        "spec_ref": "research/EXP-FX000018/00-spec-amendment-02.md",
        "loop_trial_number": 2, "loop_trial_limit": 7,
        "period": {"start": args.start, "end": args.end}, "pairs": PAIRS,
        "grid_params_frozen": {"n_levels": n_levels, "grid_step_atr_mult": k, "reanchor_bars_h4": r},
        "range_filter": rf,
        "kpi_thresholds": KPI_THRESHOLDS,
        "candidates": results,
        "selected_candidate": best,
        "train_pass_all_required_kpi": train_pass,
        "equity_curves": {nm: sims[nm]["equity_curve"] for nm in CANDIDATES},
        "trades": {nm: sims[nm]["trades"] for nm in CANDIDATES},
    }), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[出力]: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
