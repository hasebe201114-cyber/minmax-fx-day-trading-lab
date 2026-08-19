"""EXP-FX000004: ストップ無しベースラインのTrain/Validation/Test 3期間評価.

背景: `backtest_carry_baseline.py`(Train)・`analyze_carry_kstop_validation_check.py`
(Validation)で、週内ATRストップの追加はいずれもTrain最良値がValidationで
崩れる過学習を示した一方、**ストップ無しのシンプルな設計がValidationで
Sharpe4.721・DD0.74%・PF2.731・ペイオフ1.638と全KPI基準をクリア**すると
判明した。本スクリプトはTest期間を追加実行し、3期間を通した最終的な判定を
行う。ストップは追加せず、Trainで確定した設計(常にロング・週次サイクル・
固定ロット)をそのまま3期間に適用する(パラメータの再チューニングはしない)。

あわせて、提案5(通貨間相関を考慮したpermutation test)のインフラを使い、
名目週次サイクル数ではなく実効サンプル数でmin_n_trades基準を評価する
(spec `00-spec.md`で構造的制約として事前に記録済み)。

出力: research/method-notes/carry_no_stop_tvt.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_carry_baseline as base  # noqa: E402
from minmax_fx_dt.backtest.permutation import permutation_test_clustered  # noqa: E402
from minmax_fx_dt.decision.criteria import compute_n_trades_effective  # noqa: E402

PERIODS = {
    "train":      ("2023-11-01", "2025-03-31"),
    "validation": ("2025-04-01", "2025-11-30"),
    "test":       ("2025-12-01", "2026-08-15"),
}

# spec (00-spec.md) のKPI閾値
KPI_THRESHOLDS = {
    "monthly_sharpe": 0.4, "profit_factor": 1.2, "max_dd_monthly_pct": 10.0,
    "max_dd_yearly_pct": 20.0, "payoff_ratio": 1.5, "min_n_trades_effective": 300,
}


def run_period(period_name: str, start: str, end: str) -> dict:
    base.TRAIN_START, base.TRAIN_END = start, end
    cycles_all = []
    n_by_pair = {}
    for pair in base.PAIRS:
        m5 = base.load_m5(pair)
        d1 = base.to_d1(m5)
        atr_d1 = base.atr_ind(d1["high"], d1["low"], d1["close"], length=14)
        swap_daily = base.load_swap_daily(pair)
        cycles = base.weekly_cycles(pair, m5, d1, atr_d1, swap_daily, k_stop=None)
        cycles_all.extend(cycles)
        n_by_pair[pair] = len(cycles)

    summary = base.summarize(cycles_all, f"no_stop_{period_name}")

    # 実効サンプル数 (提案5のcompute_n_trades_effective()を利用)
    n_eff = compute_n_trades_effective(n_by_pair, summary["n_cycles"])

    # 通貨クラスタを考慮したpermutation test (参考値、決定論的なスワップ成分が
    # 大きいためspec通り主要判定基準ではなく補助指標として扱う)
    pnls = [c["total_pnl_jpy"] for c in cycles_all]
    pairs = [c["pair"] for c in cycles_all]
    perm = permutation_test_clustered(pnls, pairs, seed=42) if len(pnls) >= 4 else None

    kpi_pass = {
        "monthly_sharpe": summary["monthly_sharpe"] >= KPI_THRESHOLDS["monthly_sharpe"],
        "profit_factor": (summary["profit_factor"] or 0) >= KPI_THRESHOLDS["profit_factor"],
        "max_dd_monthly_pct": summary["max_dd_monthly_pct"] <= KPI_THRESHOLDS["max_dd_monthly_pct"],
        "payoff_ratio": (summary["payoff_ratio"] or 0) >= KPI_THRESHOLDS["payoff_ratio"],
        "min_n_trades_effective": n_eff >= KPI_THRESHOLDS["min_n_trades_effective"],
    }

    return {
        "period": period_name, "start": start, "end": end,
        "n_by_pair": n_by_pair,
        "summary": summary,
        "n_trades_effective": round(n_eff, 1),
        "perm_p_clustered": round(perm.p_value, 4) if perm else None,
        "kpi_pass": kpi_pass,
        "kpi_pass_count": f"{sum(kpi_pass.values())}/{len(kpi_pass)}",
    }


def main() -> int:
    print("=== EXP-FX000004: ストップ無しベースライン Train/Validation/Test ===\n")

    results = {}
    for period_name, (start, end) in PERIODS.items():
        print(f"--- {period_name}: {start} 〜 {end} ---")
        r = run_period(period_name, start, end)
        results[period_name] = r
        s = r["summary"]
        print(f"  n={s['n_cycles']}週(実効{r['n_trades_effective']})  "
              f"総リターン={s['total_return_pct']:+.2f}%  Sharpe={s['monthly_sharpe']}  "
              f"最大DD={s['max_dd_pct']}%(月間{s['max_dd_monthly_pct']}%)  "
              f"PF={s['profit_factor']}  ペイオフ={s['payoff_ratio']}  "
              f"perm_p(cluster)={r['perm_p_clustered']}")
        print(f"  KPI: {r['kpi_pass_count']} 達成  {r['kpi_pass']}\n")

    print("=== サマリ ===")
    print(f"{'期間':<12}{'総リターン':>10}{'Sharpe':>8}{'最大DD':>8}{'PF':>7}{'ペイオフ':>9}{'KPI達成':>9}")
    for name in ["train", "validation", "test"]:
        r = results[name]
        s = r["summary"]
        print(f"{name:<12}{s['total_return_pct']:>9.2f}%{s['monthly_sharpe']:>8.3f}"
              f"{s['max_dd_pct']:>7.2f}%{s['profit_factor']:>7.3f}{s['payoff_ratio']:>9.3f}"
              f"{r['kpi_pass_count']:>9}")

    all_pass = all(r["kpi_pass_count"].split("/")[0] == r["kpi_pass_count"].split("/")[1] for r in results.values())
    print(f"\n3期間すべてで全KPI達成: {'はい' if all_pass else 'いいえ'}")

    out_path = ROOT / "research" / "method-notes" / "carry_no_stop_tvt.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "design": "常にロング・週次サイクル・ストップ無し・固定ロット(1lot=1000通貨)",
            "kpi_thresholds": KPI_THRESHOLDS,
            "periods": results,
            "all_periods_all_kpi_pass": all_pass,
            "_note": (
                "実運用コスト(スプレッド/スリッページ/手数料)・複利ポジションサイジング"
                "は含まない簡易版。K6m(バックテスト/フォワード比較)・K7m(両建て証拠金)"
                "は本戦略の設計上判定対象外。permutation_p_valueはspec通り補助指標。"
            ),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
