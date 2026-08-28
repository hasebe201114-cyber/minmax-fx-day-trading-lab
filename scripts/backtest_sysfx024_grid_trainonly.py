"""EXP-FX000018 / SYS-FX024: Trainベースライン評価 (G0 MARK方式 vs G1 持ち越し方式).

`00-spec.md` §7 の選定ルール(結果を見る前に固定)に従い、フェーズゲート2で導出した
同一パラメータ(N・k・R、`10-result/grid_params.json`)・同一4通貨・Train期間で
G0/G1 の2候補を評価し、必須10項目の達成数で機械的に選ぶ。

併せて spec §5.2 が要求する反実仮想(証拠金ガード無効時の K7m_unguarded)も算出する。

出力: research/method-notes/sysfx024_grid_trainonly_backtest.json
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

import numpy as np
import pandas as pd

from evaluate_grid_kpi import KPI_THRESHOLDS, evaluate_grid_period, print_period  # noqa: E402
from grid_portfolio_engine import simulate  # noqa: E402

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"

CANDIDATES = {
    "G0_mark": {"carry_over": False, "label": "G0(MARK方式: 再アンカー時に時価強制決済)"},
    "G1_carry": {"carry_over": True, "label": "G1(持ち越し方式: 生まれた世代の利確幅・ストップを保持)"},
}


def jsonable(obj):
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return round(float(obj), 8)
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 8)
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=TRAIN_START)
    ap.add_argument("--end", default=TRAIN_END)
    ap.add_argument("--out", default="sysfx024_grid_trainonly_backtest.json")
    ap.add_argument("--skip-unguarded", action="store_true")
    args = ap.parse_args()

    params_path = ROOT / "research" / "EXP-FX000018" / "10-result" / "grid_params.json"
    with params_path.open(encoding="utf-8") as f:
        gp = json.load(f)["derived"]
    n_levels = gp["n_levels"]
    k = gp["grid_step_atr_mult"]
    r = gp["reanchor_bars_h4"]

    print("=== EXP-FX000018 / SYS-FX024 Trainベースライン評価 ===")
    print(f"対象通貨: {PAIRS}")
    print(f"期間: {args.start} 〜 {args.end}")
    print(f"フェーズゲート2導出値(損益非依存、凍結): N={n_levels}段  k={k}×ATR(H4,14)  R={r}本(H4)")
    print("候補: " + " / ".join(v["label"] for v in CANDIDATES.values()))
    print()

    results, sims = {}, {}
    for name, cfg in CANDIDATES.items():
        print(f"--- {name}: {cfg['label']} ---")
        sim = simulate(PAIRS, args.start, args.end, n_levels=n_levels, grid_step_atr_mult=k,
                       reanchor_bars=r, carry_over=cfg["carry_over"])
        res = evaluate_grid_period(name, sim)
        print_period(res)
        if not args.skip_unguarded:
            sim_ng = simulate(PAIRS, args.start, args.end, n_levels=n_levels, grid_step_atr_mult=k,
                              reanchor_bars=r, carry_over=cfg["carry_over"],
                              margin_guard=False, verbose=False)
            res["k7m_unguarded_sum_pct"] = round(sim_ng["max_margin_sum_pct"], 2)
            res["k7m_unguarded_max_method_pct"] = round(sim_ng["max_margin_max_pct"], 2)
            res["n_trades_unguarded"] = len(sim_ng["trades"])
            print(f"  [反実仮想] ガード無効時 K7m(合算)={res['k7m_unguarded_sum_pct']}%  "
                  f"トレード数={res['n_trades_unguarded']}")
        print()
        results[name] = res
        sims[name] = sim

    # --- spec §7 の選定ルール (結果を見る前に固定済み) ---
    def req_count(r_: dict) -> int:
        return int(r_["kpi_required_pass_count"].split("/")[0])

    best = max(CANDIDATES, key=lambda nm: (req_count(results[nm]), results[nm]["monthly_sharpe"]))
    train_pass = results[best]["kpi_required_all_pass"]

    print("=== サマリ ===")
    print(f"{'候補':<12}{'n':>7}{'勝率':>8}{'Sharpe':>9}{'最大DD':>9}{'PF':>8}{'実効n':>8}{'perm_p':>9}{'K7m':>8}{'必須KPI':>10}")
    for name in CANDIDATES:
        x = results[name]
        print(f"{name:<12}{x['n_trades']:>7}{x['win_rate']:>8.3f}{x['monthly_sharpe']:>9.3f}"
              f"{x['max_dd_pct']:>8.2f}%{x['profit_factor']:>8.3f}{x['n_trades_effective']:>8.0f}"
              f"{str(x['permutation_p_week_block']):>9}{x['k7m_margin_sum_pct']:>7.1f}%"
              f"{x['kpi_required_pass_count']:>10}")
    print(f"\n選定ルール(必須KPI達成数 → 同数なら月次シャープ)による採用候補: **{best}**")
    print(f"Train通過(必須11項目すべて達成): {'はい' if train_pass else 'いいえ'}")
    if not train_pass:
        ng = [k_ for k_, v in results[best]["kpi_pass"].items() if not v]
        print(f"未達項目: {ng}")

    out_path = ROOT / "research" / "method-notes" / args.out
    out_path.write_text(json.dumps(jsonable({
        "generated_at": datetime.now().isoformat(),
        "exp_id": "EXP-FX000018", "sys_id": "SYS-FX024",
        "spec_ref": "research/EXP-FX000018/00-spec.md",
        "period": {"start": args.start, "end": args.end},
        "pairs": PAIRS,
        "grid_params_frozen": {"n_levels": n_levels, "grid_step_atr_mult": k, "reanchor_bars_h4": r,
                               "source": "research/EXP-FX000018/10-result/grid_params.json"},
        "kpi_thresholds": KPI_THRESHOLDS,
        "candidates": {nm: results[nm] for nm in CANDIDATES},
        "selected_candidate": best,
        "train_pass_all_required_kpi": train_pass,
        "equity_curves": {nm: sims[nm]["equity_curve"] for nm in CANDIDATES},
        "margin_curves": {nm: sims[nm]["margin_curve"] for nm in CANDIDATES},
        "trades": {nm: sims[nm]["trades"] for nm in CANDIDATES},
    }), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
