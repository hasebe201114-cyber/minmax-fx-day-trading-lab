"""EXP-FX000018 amendment-01: 週末持ち越し緩和 + リスク対策 の Train 再評価.

司令塔判断(2026-08-28)「1. 緩和しても良い。リスク対策は必要。」を受けた改善ループ
第1試行。`00-spec-amendment-01.md` §4 で事前登録した4候補を、フェーズゲート2の
凍結パラメータ(N=3・k=1.72×ATR・R=24、再導出しない)で評価する。

  B-A / B-B: 週末フラット(従来) × G0 / G1  ← W1(窓開け約定)・W4(ロスカット)を
             適用して再計測したベースライン。改善ループの試行にはカウントしない
  W-A / W-B: 週末持ち越し可 × G0 / G1      ← 改善ループ第1試行(Q10 上限7回の1回目)

リスク対策(amendment-01 §3、すべて結果を見る前に固定):
  W1 窓開け(ギャップ)約定モデル … 寄り値で飛び越えた場合は寄り値で約定
  W2 週末ネットエクスポージャー上限 … 想定窓開け損失 ≤ MTM equity の10%
  W3 最大保有期間 … 48本(H4) = 8営業日 = 再アンカー周期 R の2倍
  W4 証拠金維持率ロスカット … 100%未満で全決済、125%未満で新規停止

出力: research/method-notes/sysfx024_weekend_carry_trainonly_backtest.json
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

from backtest_sysfx024_grid_trainonly import jsonable  # noqa: E402
from evaluate_grid_kpi import KPI_THRESHOLDS, evaluate_grid_period, print_period  # noqa: E402
from grid_portfolio_engine import simulate  # noqa: E402

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"

# amendment-01 §3 W3: 最大保有期間 = 再アンカー周期 R の2倍 (パラメータ探索ではない)
MAX_HOLD_MULTIPLE_OF_R = 2

CANDIDATES = {
    "B-A": {"weekend_carry": False, "carry_over": False,
            "label": "ベースライン: 週末フラット × G0(MARK方式)", "loop_trial": False},
    "B-B": {"weekend_carry": False, "carry_over": True,
            "label": "ベースライン: 週末フラット × G1(持ち越し方式)", "loop_trial": False},
    "W-A": {"weekend_carry": True, "carry_over": False,
            "label": "改善ループ第1試行: 週末持ち越し可 × G0(MARK方式)", "loop_trial": True},
    "W-B": {"weekend_carry": True, "carry_over": True,
            "label": "改善ループ第1試行: 週末持ち越し可 × G1(持ち越し方式)", "loop_trial": True},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=TRAIN_START)
    ap.add_argument("--end", default=TRAIN_END)
    ap.add_argument("--out", default="sysfx024_weekend_carry_trainonly_backtest.json")
    args = ap.parse_args()

    res_dir = ROOT / "research" / "EXP-FX000018" / "10-result"
    gp = json.loads((res_dir / "grid_params.json").read_text(encoding="utf-8"))["derived"]
    gap = json.loads((res_dir / "weekend_gap_risk.json").read_text(encoding="utf-8"))["derived"]
    n_levels, k, r = gp["n_levels"], gp["grid_step_atr_mult"], gp["reanchor_bars_h4"]
    rel_gap_p99 = gap["rel_gap_p99"]
    budget = gap["weekend_gap_loss_budget_pct"]
    max_hold = r * MAX_HOLD_MULTIPLE_OF_R

    print("=== EXP-FX000018 amendment-01: 週末持ち越し緩和 + リスク対策 Train再評価 ===")
    print(f"対象通貨: {PAIRS}   期間: {args.start} 〜 {args.end}")
    print(f"凍結パラメータ(再導出しない): N={n_levels}段  k={k}×ATR(H4,14)  R={r}本(H4)")
    print(f"リスク対策: W1(窓開け寄り値約定) / W2(週末想定窓開け損失≤{budget}% of equity, "
          f"rel_gap_p99={rel_gap_p99}) / W3(最大保有{max_hold}本=R×{MAX_HOLD_MULTIPLE_OF_R}) / "
          f"W4(維持率100%でロスカット・125%で新規停止)")
    print()

    results, sims = {}, {}
    for name, cfg in CANDIDATES.items():
        print(f"--- {name}: {cfg['label']} ---")
        kwargs = dict(n_levels=n_levels, grid_step_atr_mult=k, reanchor_bars=r,
                      carry_over=cfg["carry_over"], verbose=False)
        if cfg["weekend_carry"]:
            kwargs.update(weekend_carry=True, max_hold_h4_bars=max_hold,
                          rel_gap_p99=rel_gap_p99, weekend_gap_budget_pct=budget)
        sim = simulate(PAIRS, args.start, args.end, **kwargs)
        res = evaluate_grid_period(name, sim)
        res["loop_trial"] = cfg["loop_trial"]
        res["label"] = cfg["label"]
        print_period(res)
        print()
        results[name], sims[name] = res, sim

    # --- amendment-01 §5 選定ルール (親spec §7 から変更なし) ---
    def req_count(x: dict) -> int:
        return int(x["kpi_required_pass_count"].split("/")[0])

    best = max(CANDIDATES, key=lambda nm: (req_count(results[nm]), results[nm]["monthly_sharpe"]))
    train_pass = results[best]["kpi_required_all_pass"]

    print("=== サマリ ===")
    print(f"{'候補':<6}{'週末':<8}{'n':>7}{'勝率':>8}{'最終残高':>11}{'Sharpe':>9}{'最大DD':>9}"
          f"{'PF':>8}{'perm_p':>9}{'K7m':>8}{'必須KPI':>9}")
    for name in CANDIDATES:
        x = results[name]
        wk = "持越可" if CANDIDATES[name]["weekend_carry"] else "フラット"
        print(f"{name:<6}{wk:<8}{x['n_trades']:>7}{x['win_rate']:>8.3f}"
              f"{x['final_balance_usd']:>10.2f}${x['monthly_sharpe']:>9.3f}{x['max_dd_pct']:>8.2f}%"
              f"{x['profit_factor']:>8.3f}{str(x['permutation_p_week_block']):>9}"
              f"{x['k7m_margin_sum_pct']:>7.1f}%{x['kpi_required_pass_count']:>9}")

    print(f"\n選定ルール(必須KPI達成数 → 同数なら月次シャープ)による採用候補: **{best}**")
    print(f"Train通過(必須11項目すべて達成): {'はい' if train_pass else 'いいえ'}")
    if not train_pass:
        print(f"未達項目: {[k_ for k_, v in results[best]['kpi_pass'].items() if not v]}")

    out = ROOT / "research" / "method-notes" / args.out
    out.write_text(json.dumps(jsonable({
        "generated_at": datetime.now().isoformat(),
        "exp_id": "EXP-FX000018", "sys_id": "SYS-FX024",
        "spec_ref": "research/EXP-FX000018/00-spec-amendment-01.md",
        "loop_trial_number": 1, "loop_trial_limit": 7,
        "period": {"start": args.start, "end": args.end},
        "pairs": PAIRS,
        "grid_params_frozen": {"n_levels": n_levels, "grid_step_atr_mult": k, "reanchor_bars_h4": r},
        "risk_controls": {"W1": "窓開けは寄り値約定", "W2_budget_pct": budget,
                          "W2_rel_gap_p99": rel_gap_p99, "W3_max_hold_h4_bars": max_hold,
                          "W4_liquidation_pct": 100.0, "W4_alert_pct": 125.0},
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
