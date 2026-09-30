"""OBS000015: 合計レバレッジ上限(25倍)の導入前後で、公式評価の数値を同じ実行内で比較する。

修正前 = money_management.DEFAULT_AGGREGATE_LEVERAGE_CAP を None にした場合(従来の処理と完全一致、
tests/test_money_management.py で確認済み)。修正後 = 既定値25倍。
対象: SYS-FX012(candidate1)・SYS-FX011(v7 trailonly)・SYS-FX018(breakeven 2.0) の Train/Validation、
SYS-FX012 フォワード台帳。Test 期間は計算しない(凍結ホールドアウト)。

出力: research/method-notes/obs000015_aggregate_leverage_recalc.json
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "forward_test"))

import minmax_fx_dt.backtest.money_management as mm  # noqa: E402
import backtest_sysfx018_breakeven_sweep_trainonly as fx018  # noqa: E402
import backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd as v7  # noqa: E402
import backtest_vol_continuation_candidates_trendfilter_4pairs_trainonly as cand  # noqa: E402
import run_forward_test_cycle as fwd  # noqa: E402
from evaluate_vol_breakout_dow_theory_kpi import evaluate_period  # noqa: E402

OUT = ROOT / "research" / "method-notes" / "obs000015_aggregate_leverage_recalc.json"
KPI_KEYS = ["monthly_sharpe", "max_dd_pct", "max_dd_monthly_pct", "profit_factor", "payoff_ratio",
            "n_trades_effective", "permutation_p_clustered", "kpi_required_pass_count"]


def summarize(result: dict, period: str) -> dict:
    kpi = evaluate_period(period, result, perm_p_field="perm_p_block",
                          apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)
    final = result.get("final_balance") or result.get("final_balance_usd")
    if final is None and result.get("equity_curve"):
        final = result["equity_curve"][-1]["balance"]
    return {
        "n_trades": result.get("n_trades"),
        "final_balance": round(float(final), 2) if final is not None else None,
        "total_return_pct": round((float(final) / v7.INITIAL_CAPITAL_USD - 1) * 100, 2) if final is not None else None,
        "win_rate": result.get("win_rate"), "mean_r_net": result.get("mean_r_net"),
        "money_management": result.get("money_management"),
        **{k: kpi.get(k) for k in KPI_KEYS},
    }


def run_all() -> dict:
    out: dict = {}
    for period in ("train", "validation"):
        start, end = v7.PERIODS[period]
        out[f"SYS-FX012/{period}"] = summarize(
            cand.run_period("candidate1_n_breakout_only_trendfilter", cand.detect_candidate1, start, end), period)
        out[f"SYS-FX011/{period}"] = summarize(v7.run_period(period, start, end), period)
        out[f"SYS-FX018/{period}"] = summarize(fx018.run_period(2.0, start, end), period)
    return out


def run_forward(out_path: Path) -> dict:
    cfg = copy.deepcopy(fwd.STRATEGIES["sysfx012"])
    cfg["out_path"] = out_path
    res = fwd.run_forward_cycle(cfg)
    bt = res.get("backtest", res)
    return {"n_trades_closed": bt.get("n_trades_closed"), "final_balance": bt.get("final_balance"),
            "money_management": bt.get("money_management")}


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    mm.DEFAULT_AGGREGATE_LEVERAGE_CAP = None
    before = run_all()
    before["SYS-FX012/forward"] = run_forward(tmp / "before.json")
    mm.DEFAULT_AGGREGATE_LEVERAGE_CAP = mm.GMO_MAX_LEVERAGE
    after = run_all()
    after["SYS-FX012/forward"] = run_forward(tmp / "after.json")
    OUT.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(), "obs": "OBS000015",
        "before": "合計レバレッジ上限なし(従来)", "after": f"合計レバレッジ上限 {mm.GMO_MAX_LEVERAGE}倍(超過分を縮小)",
        "results": {k: {"before": before[k], "after": after[k]} for k in before},
    }, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"{'対象':22s} {'件数':>9s} {'リターン%':>17s} {'最大DD%':>13s} {'Sharpe':>13s} {'PF':>11s} {'KPI':>9s} 合計レバ最大")
    for k in before:
        b, a = before[k], after[k]
        if k.endswith("forward"):
            print(f"{k:22s} {b['n_trades_closed']}→{a['n_trades_closed']}  残高 {b['final_balance']}→{a['final_balance']}  "
                  f"合計レバ最大 {b['money_management']['max_aggregate_leverage']}→{a['money_management']['max_aggregate_leverage']}")
            continue
        print(f"{k:22s} {b['n_trades']:>4}→{a['n_trades']:<4} {b['total_return_pct']:>7}→{a['total_return_pct']:<8} "
              f"{b['max_dd_pct']:>5}→{a['max_dd_pct']:<6} {b['monthly_sharpe']:>5}→{a['monthly_sharpe']:<6} "
              f"{b['profit_factor']:>5}→{a['profit_factor']:<5} {b['kpi_required_pass_count']}→{a['kpi_required_pass_count']} "
              f"{b['money_management']['max_aggregate_leverage']}→{a['money_management']['max_aggregate_leverage']}"
              f" (縮小{a['money_management']['n_aggregate_cap_shrunk']} 見送{a['money_management']['n_aggregate_cap_skipped']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
