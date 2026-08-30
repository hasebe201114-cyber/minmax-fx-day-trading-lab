"""月次均一配置の DSR 分布算出スクリプト (v0.3 M-R1 対応).

起源:
    v0.3 spec §6 月次均一配置の取り扱い.
    親 PJ の `distribute_pnls_to_months()` は trade_pnls を期間内の月へ均一に
    分配するが、実際のトレード月は不明。**均一配置は偶然 sharpe を押し上げる方向に
    作用する可能性**がある。

    本スクリプトは月配置を N 回ランダムにシャッフルし、DSR の分布
    (p5/p50/p95) を算出する。**p5 が DSR 閾値を超える場合のみ PASS** とする。

使用方法:
    python scripts/calc_dsr_with_distribution.py
    # または
    python scripts/calc_dsr_with_distribution.py --n-samples 500 --top-k 6
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from minmax_fx_eval.statistics.dsr import deflated_sharpe_ratio


# ============================================================
# 親 PJ の loaders（再利用可能）
# ============================================================

PARENT_PJ = Path("C:/Users/Atsushi Hasebe/.minimax-agent/projects/minmax-fx-day-trading-lab")


def months_in_period(start: str, end: str) -> list[str]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    months = []
    y, m = s.year, s.month
    while (y, m) <= (e.year, e.month):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def distribute_pnls_to_months(
    trade_pnls: list[float], start: str, end: str
) -> dict[str, float]:
    """トレード PnL を均一に月配置（v0.2 と同じロジック）."""
    months = months_in_period(start, end)
    n_months = len(months)
    n_trades = len(trade_pnls)
    if n_trades == 0 or n_months == 0:
        return {m: 0.0 for m in months}
    base = n_trades // n_months
    remainder = n_trades % n_months
    month_pnls: dict[str, float] = {}
    idx = 0
    for i, m in enumerate(months):
        count = base + (1 if i < remainder else 0)
        chunk = trade_pnls[idx : idx + count]
        month_pnls[m] = sum(chunk)
        idx += count
    return month_pnls


def compute_monthly_returns_from_pnls(
    month_pnls: dict[str, float], initial_cash: float
) -> list[float]:
    return [pnl / initial_cash for pnl in month_pnls.values()]


def randomize_monthly_pnls(
    trade_pnls: list[float], n_months: int, rng: np.random.Generator
) -> list[float]:
    """トレード PnL をランダムに N ヶ月に分配.

    親 PJ の `distribute_pnls_to_months()` は deterministic な均等配置だが、
    本関数ではランダムに月配置をシャッフルする。N ヶ月合計は元の合計と一致するが、
    個別の月の PnL は変動する。
    """
    monthly_pnls = np.zeros(n_months)
    # 各トレードをランダムな月に割り当て
    month_indices = rng.integers(0, n_months, size=len(trade_pnls))
    for trade_pnl, month_idx in zip(trade_pnls, month_indices):
        monthly_pnls[month_idx] += trade_pnl
    return monthly_pnls.tolist()


def compute_dsr_distribution(
    trade_pnls: list[float],
    start: str,
    end: str,
    *,
    initial_cash: float,
    n_trials: int,
    n_samples: int = 200,
    seed: int = 42,
    periods_per_year: int = 12,
) -> dict[str, Any]:
    """月次配置を N 回ランダム化した DSR 分布を算出 (v0.3 M-R1).

    Returns:
        {
            "n_samples": int,
            "n_trades": int,
            "n_months": int,
            "n_trials": int,
            "dsr_baseline": float,  # 均一配置での DSR
            "dsr_distribution": {
                "p5": float,
                "p50": float,
                "p95": float,
                "mean": float,
                "std": float,
                "min": float,
                "max": float,
            },
            "dsr_samples": [float, ...],  # 全サンプル
            "passes_with_p5": bool,  # p5 >= threshold
        }
    """
    rng = np.random.default_rng(seed)
    n_months = len(months_in_period(start, end))

    # Baseline: 均一配置
    base_month_pnls = distribute_pnls_to_months(trade_pnls, start, end)
    base_returns = compute_monthly_returns_from_pnls(base_month_pnls, initial_cash)
    if len(base_returns) < 3:
        return {
            "n_samples": 0,
            "n_trades": len(trade_pnls),
            "n_months": n_months,
            "n_trials": n_trials,
            "dsr_baseline": None,
            "dsr_distribution": None,
            "error": "insufficient observations",
        }
    base_result = deflated_sharpe_ratio(
        np.asarray(base_returns, dtype=float),
        n_trials=n_trials,
        periods_per_year=periods_per_year,
    )
    dsr_baseline = base_result.dsr

    # ランダム化 N サンプル
    dsr_samples: list[float] = []
    for _ in range(n_samples):
        rand_pnls = randomize_monthly_pnls(trade_pnls, n_months, rng)
        rand_returns = [p / initial_cash for p in rand_pnls]
        if len(rand_returns) < 3:
            continue
        try:
            result = deflated_sharpe_ratio(
                np.asarray(rand_returns, dtype=float),
                n_trials=n_trials,
                periods_per_year=periods_per_year,
            )
            dsr_samples.append(result.dsr)
        except Exception:  # noqa: BLE001
            continue

    if not dsr_samples:
        return {
            "n_samples": 0,
            "n_trades": len(trade_pnls),
            "n_months": n_months,
            "n_trials": n_trials,
            "dsr_baseline": dsr_baseline,
            "dsr_distribution": None,
            "error": "all samples failed",
        }

    arr = np.asarray(dsr_samples)
    return {
        "n_samples": len(dsr_samples),
        "n_trades": len(trade_pnls),
        "n_months": n_months,
        "n_trials": n_trials,
        "dsr_baseline": float(dsr_baseline),
        "dsr_distribution": {
            "p5": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        },
        "dsr_samples": dsr_samples[:50],  # 最初の 50 個のみ保存（サイズ制限）
        "passes_with_p5": bool(float(np.percentile(arr, 5)) >= 0.95),
    }


# ============================================================
# 戦略 loaders（親 PJ データ使用）
# ============================================================


def load_strategy_data() -> list[dict[str, Any]]:
    """親 PJ の戦略ごとの trade_pnls + 期間 + n_trials を返す."""
    strategies = []

    # SYS-FX007: A1_A2_combined 全 15 セル
    f = PARENT_PJ / "research/EXP-FX000001/10-result/train_val_test/tvt_A1_A2_combined.json"
    if f.exists():
        j = json.loads(f.read_text(encoding="utf-8"))
        all_pnls = []
        for cell in j["results"]:
            for period in ["train", "validation", "test"]:
                p = cell["periods"].get(period, {})
                if "trade_pnls" in p:
                    all_pnls.extend(p["trade_pnls"])
        strategies.append({
            "sys_id": "SYS-FX007",
            "trade_pnls": all_pnls,
            "start": "2023-11-01",
            "end": "2026-08-15",
            "initial_cash": 1_000_000.0,
            "n_trials": 6,  # v0.2: 6 (ablations). v0.3 (M-R2): 通貨選択×閾値選択含む
        })

    # SYS-FX008: USD/JPY TVT 3 期間通し
    f = PARENT_PJ / "research/EXP-FX000002/10-result/train_val_test/tvt_USD_JPY_train.json"
    if f.exists():
        j = json.loads(f.read_text(encoding="utf-8"))
        all_pnls = list(j["trade_pnls"])
        for suffix in ["_validation", "_test"]:
            f2 = f.parent / f"tvt_USD_JPY{suffix}.json"
            if f2.exists():
                j2 = json.loads(f2.read_text(encoding="utf-8"))
                all_pnls.extend(j2["trade_pnls"])
        strategies.append({
            "sys_id": "SYS-FX008",
            "trade_pnls": all_pnls,
            "start": "2023-11-01",
            "end": "2026-08-15",
            "initial_cash": 1_000_000.0,
            "n_trials": 3,
        })

    # SYS-FX009: USD/JPY TVT 3 期間通し
    f = PARENT_PJ / "research/EXP-FX000003/10-result/train_val_test/tvt_USD_JPY_train.json"
    if f.exists():
        j = json.loads(f.read_text(encoding="utf-8"))
        all_pnls = list(j["trade_pnls"])
        for suffix in ["_validation", "_test"]:
            f2 = f.parent / f"tvt_USD_JPY{suffix}.json"
            if f2.exists():
                j2 = json.loads(f2.read_text(encoding="utf-8"))
                all_pnls.extend(j2["trade_pnls"])
        strategies.append({
            "sys_id": "SYS-FX009 v2",
            "trade_pnls": all_pnls,
            "start": "2023-11-01",
            "end": "2026-08-15",
            "initial_cash": 1_000_000.0,
            "n_trials": 1,
        })

    # SYS-FX011 v7: monthly フィールドから直接
    f = PARENT_PJ / "research/method-notes/vol_breakout_v7_trade_ledger.json"
    if f.exists():
        j = json.loads(f.read_text(encoding="utf-8"))
        # monthly dict → 再現（month キー順でトレード数比例の均等配分）
        # 注: v7 は月次データがあるため distribution 検証は省略し、baseline のみ
        strategies.append({
            "sys_id": "SYS-FX011 v7",
            "trade_pnls": None,  # 月次データ使用のため trade_pnls 経路はスキップ
            "monthly_pnl_usd": {m["month"]: m["sum_dollar_pnl"] for m in j["monthly"]},
            "n_trials": 7,
            "_skip_distribution": True,
        })

    # SYS-FX011 T-13: 4pairs trailonly 3 期間通し
    f = PARENT_PJ / "research/method-notes/vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd_backtest.json"
    if f.exists():
        j = json.loads(f.read_text(encoding="utf-8"))
        all_trades = []
        for period_name in ["train", "validation", "test"]:
            all_trades.extend(j["periods"][period_name]["trades"])
        trade_pnls = [t["dollar_pnl"] for t in all_trades]
        strategies.append({
            "sys_id": "SYS-FX011 T-13",
            "trade_pnls": trade_pnls,
            "start": "2023-11-01",
            "end": "2026-08-15",
            "initial_cash": 1000.0,
            "n_trials": 7,
        })

    return strategies


# ============================================================
# Main
# ============================================================


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=200, help="ランダム化サンプル数")
    parser.add_argument("--top-k", type=int, default=4, help="上位 K 戦略の詳細表示")
    args = parser.parse_args()

    print("=" * 80)
    print("DSR Distribution (v0.3 M-R1: randomized monthly placement)")
    print("=" * 80)
    print(f"Samples per strategy: {args.n_samples}")
    print()

    strategies = load_strategy_data()
    results: list[dict] = []

    for s in strategies:
        sys_id = s["sys_id"]
        if s.get("_skip_distribution"):
            # 月次データ直接使用の戦略（v7）は distribution 計算スキップ
            print(f"## {sys_id}")
            print("  SKIPPED: monthly data directly used (no need for distribution)")
            print()
            continue

        if not s.get("trade_pnls"):
            print(f"## {sys_id}")
            print("  SKIPPED: no trade_pnls available")
            print()
            continue

        print(f"## {sys_id}")
        result = compute_dsr_distribution(
            trade_pnls=s["trade_pnls"],
            start=s["start"],
            end=s["end"],
            initial_cash=s["initial_cash"],
            n_trials=s["n_trials"],
            n_samples=args.n_samples,
        )
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            print()
            continue

        dist = result["dsr_distribution"]
        baseline = result["dsr_baseline"]
        print(f"  n_trades: {result['n_trades']}, n_months: {result['n_months']}, n_trials: {result['n_trials']}")
        print(f"  DSR baseline (uniform):    {baseline:.4f}")
        print(f"  DSR distribution (random): p5={dist['p5']:.4f}, p50={dist['p50']:.4f}, p95={dist['p95']:.4f}")
        print(f"                            mean={dist['mean']:.4f} ± {dist['std']:.4f}, min={dist['min']:.4f}, max={dist['max']:.4f}")
        print(f"  PASS with p5 >= 0.95:      {result['passes_with_p5']}")
        print()
        results.append({"sys_id": sys_id, **result})

    # 保存
    output_dir = Path("research/フレームワーク再設計/02-比較")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dsr_distribution_v03.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
