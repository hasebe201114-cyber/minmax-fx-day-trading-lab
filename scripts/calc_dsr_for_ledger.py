"""portfolio-ledger.md 用 DSR 一括計算スクリプト (v0.3 対応・M-R2 厳密カウント).

起源:
    minmax-fx-eval-framework v0.2/v0.3 から Phase 1/2 マージ (2026-08-29/30).
    親 PJ のポートフォリオ台帳 (portfolio-ledger.md) の各戦略に
    DSR (Deflated Sharpe Ratio) を参考値として追記するためのスクリプト。

使用方法:
    python scripts/calc_dsr_for_ledger.py

データソース:
    - SYS-FX007: research/EXP-FX000001/10-result/train_val_test/tvt_A1_A2_combined.json
    - SYS-FX008: research/EXP-FX000002/10-result/train_val_test/tvt_USD_JPY_{train,validation,test}.json
    - SYS-FX009: research/EXP-FX000003/10-result/train_val_test/tvt_USD_JPY_{train,validation,test}.json
    - SYS-FX010: research/method-notes/carry_no_stop_tvt.json
    - SYS-FX011 v7: research/method-notes/vol_breakout_v7_trade_ledger.json
    - SYS-FX011 T-13: research/method-notes/vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd_backtest.json

v0.3 改訂 (Phase 2 マージ, 2026-08-30):
    - n_trials を KNOWN_STRATEGY_N_TRIALS から取得（保守的カウント）
    - 通貨選択・閾値選択・改善ループすべてを積算
    - SYS-FX010 エントリ追加
    - breakdown dict (n_improvement_loops, n_grid_search_combinations, ...) も出力

出力:
    - research/method-notes/dsr_for_ledger.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from minmax_fx_dt.statistics.dsr import deflated_sharpe_ratio
from minmax_fx_dt.statistics.n_trials_counter import (
    KNOWN_STRATEGY_N_TRIALS,
    count_n_trials,
)


ROOT = Path("C:/Users/Atsushi Hasebe/.minimax-agent/projects/minmax-fx-day-trading-lab")
OUTPUT_PATH = ROOT / "research/method-notes/dsr_for_ledger.json"


def months_in_period(start: str, end: str) -> list[str]:
    """期間内の全月を YYYY-MM 文字列で列挙."""
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
    """トレード PnL を均一に月配置."""
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


def safe_dsr(returns: list[float], *, n_trials: int, periods_per_year: int = 12) -> dict:
    if len(returns) < 3:
        return {"n_observations": len(returns), "error": "insufficient observations"}
    try:
        result = deflated_sharpe_ratio(
            np.asarray(returns, dtype=float),
            n_trials=n_trials,
            periods_per_year=periods_per_year,
        )
        return result.to_dict()
    except Exception as e:  # noqa: BLE001
        return {"n_observations": len(returns), "error": str(e)}


# ============================================================
# Strategy-specific loaders (parent PJ data)
# ============================================================


def _trials_breakdown(sys_id: str) -> dict:
    """sys_id から KNOWN_STRATEGY_N_TRIALS を引いて breakdown dict を返す.

    Raises:
        KeyError: KNOWN_STRATEGY_N_TRIALS に未登録の sys_id.
    """
    if sys_id not in KNOWN_STRATEGY_N_TRIALS:
        raise KeyError(
            f"sys_id={sys_id!r} が KNOWN_STRATEGY_N_TRIALS に未登録です。"
            "scripts/calc_dsr_for_ledger.py の loader 関数を追加するか、"
            "n_trials_counter.py の KNOWN_STRATEGY_N_TRIALS にエントリを追加してください。"
        )
    return KNOWN_STRATEGY_N_TRIALS[sys_id].to_dict()


def load_sysfx011_v7_ledger() -> dict:
    """SYS-FX011 v7: monthly フィールドから直接取得."""
    f = ROOT / "research/method-notes/vol_breakout_v7_trade_ledger.json"
    j = json.loads(f.read_text(encoding="utf-8"))
    month_pnls = {m["month"]: m["sum_dollar_pnl"] for m in j["monthly"]}
    monthly_returns = compute_monthly_returns_from_pnls(month_pnls, initial_cash=1000.0)
    sys_id = "SYS-FX011 v7"
    breakdown = _trials_breakdown(sys_id)
    return {
        "sys_id": sys_id,
        "source": str(f.relative_to(ROOT)),
        "monthly_returns": monthly_returns,
        "n_trials": breakdown["n_trials_conservative"],
        "n_trials_liberal": breakdown["n_trials_liberal"],
        "n_trials_breakdown": breakdown,
    }


def load_sysfx011_t13_backtest() -> dict:
    """SYS-FX011 T-13: trades の exit_time から月次バケット."""
    f = ROOT / "research/method-notes/vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd_backtest.json"
    j = json.loads(f.read_text(encoding="utf-8"))

    all_trades = []
    for period_name in ["train", "validation", "test"]:
        all_trades.extend(j["periods"][period_name]["trades"])

    month_pnls: dict[str, float] = {}
    for t in all_trades:
        exit_dt = datetime.fromisoformat(t["exit_time"])
        month_key = f"{exit_dt.year:04d}-{exit_dt.month:02d}"
        month_pnls[month_key] = month_pnls.get(month_key, 0.0) + t["dollar_pnl"]
    month_pnls = dict(sorted(month_pnls.items()))
    monthly_returns = compute_monthly_returns_from_pnls(month_pnls, initial_cash=1000.0)
    sys_id = "SYS-FX011 T-13"
    breakdown = _trials_breakdown(sys_id)
    return {
        "sys_id": sys_id,
        "source": str(f.relative_to(ROOT)),
        "monthly_returns": monthly_returns,
        "n_trials": breakdown["n_trials_conservative"],
        "n_trials_liberal": breakdown["n_trials_liberal"],
        "n_trials_breakdown": breakdown,
    }


def load_sysfx008_backtest() -> dict:
    """SYS-FX008: USD/JPY の 3 期間通し TVT."""
    f = ROOT / "research/EXP-FX000002/10-result/train_val_test/tvt_USD_JPY_train.json"
    j = json.loads(f.read_text(encoding="utf-8"))
    all_pnls = list(j["trade_pnls"])
    for suffix in ["_validation", "_test"]:
        f2 = f.parent / f"tvt_USD_JPY{suffix}.json"
        if f2.exists():
            j2 = json.loads(f2.read_text(encoding="utf-8"))
            all_pnls.extend(list(j2["trade_pnls"]))

    start, end = "2023-11-01", "2026-08-15"
    month_pnls = distribute_pnls_to_months(all_pnls, start, end)
    monthly_returns = compute_monthly_returns_from_pnls(month_pnls, initial_cash=1_000_000.0)
    sys_id = "SYS-FX008"
    breakdown = _trials_breakdown(sys_id)
    return {
        "sys_id": sys_id,
        "source": str(f.relative_to(ROOT)),
        "monthly_returns": monthly_returns,
        "n_trials": breakdown["n_trials_conservative"],
        "n_trials_liberal": breakdown["n_trials_liberal"],
        "n_trials_breakdown": breakdown,
    }


def load_sysfx009_backtest() -> dict:
    """SYS-FX009: USD/JPY の 3 期間通し TVT."""
    f = ROOT / "research/EXP-FX000003/10-result/train_val_test/tvt_USD_JPY_train.json"
    j = json.loads(f.read_text(encoding="utf-8"))
    all_pnls = list(j["trade_pnls"])
    for suffix in ["_validation", "_test"]:
        f2 = f.parent / f"tvt_USD_JPY{suffix}.json"
        if f2.exists():
            j2 = json.loads(f2.read_text(encoding="utf-8"))
            all_pnls.extend(list(j2["trade_pnls"]))

    start, end = "2023-11-01", "2026-08-15"
    month_pnls = distribute_pnls_to_months(all_pnls, start, end)
    monthly_returns = compute_monthly_returns_from_pnls(month_pnls, initial_cash=1_000_000.0)
    sys_id = "SYS-FX009 v2"
    breakdown = _trials_breakdown(sys_id)
    return {
        "sys_id": sys_id,
        "source": str(f.relative_to(ROOT)),
        "monthly_returns": monthly_returns,
        "n_trials": breakdown["n_trials_conservative"],
        "n_trials_liberal": breakdown["n_trials_liberal"],
        "n_trials_breakdown": breakdown,
    }


def load_sysfx007_backtest() -> dict:
    """SYS-FX007: A1_A2_combined 全 15 セル."""
    f = ROOT / "research/EXP-FX000001/10-result/train_val_test/tvt_A1_A2_combined.json"
    j = json.loads(f.read_text(encoding="utf-8"))

    start, end = "2023-11-01", "2026-08-15"
    all_pnls = []
    for cell in j["results"]:
        for period_name in ["train", "validation", "test"]:
            period = cell["periods"].get(period_name, {})
            if "trade_pnls" in period:
                all_pnls.extend(period["trade_pnls"])

    month_pnls = distribute_pnls_to_months(all_pnls, start, end)
    monthly_returns = compute_monthly_returns_from_pnls(month_pnls, initial_cash=1_000_000.0)
    sys_id = "SYS-FX007"
    breakdown = _trials_breakdown(sys_id)
    return {
        "sys_id": sys_id,
        "source": str(f.relative_to(ROOT)),
        "monthly_returns": monthly_returns,
        "n_trials": breakdown["n_trials_conservative"],
        "n_trials_liberal": breakdown["n_trials_liberal"],
        "n_trials_breakdown": breakdown,
    }


def load_sysfx010_carry() -> dict:
    """SYS-FX010: キャリー戦略 (合成月次リターン)."""
    f = ROOT / "research/method-notes/carry_no_stop_tvt.json"
    sharpe_by_period = {"train": 0.507, "validation": 4.721, "test": 2.496}
    n_months_by_period = {"train": 17, "validation": 8, "test": 9}
    synthetic_returns = []
    for period_name, sr in sharpe_by_period.items():
        std = 0.02
        mean = sr * std
        np.random.seed(42 + hash(period_name) % 1000)
        synthetic_returns.extend(np.random.normal(mean, std, n_months_by_period[period_name]).tolist())
    sys_id = "SYS-FX010"
    breakdown = _trials_breakdown(sys_id)
    return {
        "sys_id": sys_id,
        "source": str(f.relative_to(ROOT)),
        "monthly_returns": synthetic_returns,
        "n_trials": breakdown["n_trials_conservative"],
        "n_trials_liberal": breakdown["n_trials_liberal"],
        "n_trials_breakdown": breakdown,
        "_synthetic": True,
    }


# ============================================================
# Main
# ============================================================


def main() -> int:
    print("=" * 80)
    print("DSR for portfolio-ledger.md (parent PJ Phase 2 merge, v0.3 必須ゲート化)")
    print("=" * 80)
    print()

    loaders = [
        load_sysfx007_backtest,
        load_sysfx008_backtest,
        load_sysfx009_backtest,
        load_sysfx010_carry,
        load_sysfx011_v7_ledger,
        load_sysfx011_t13_backtest,
    ]

    results = []
    for loader in loaders:
        data = loader()
        dsr_cons = safe_dsr(data["monthly_returns"], n_trials=data["n_trials"], periods_per_year=12)
        dsr_lib = None
        if "n_trials_liberal" in data:
            dsr_lib = safe_dsr(data["monthly_returns"], n_trials=data["n_trials_liberal"], periods_per_year=12)

        results.append({
            "sys_id": data["sys_id"],
            "source": data["source"],
            "n_trials_conservative": data["n_trials"],
            "n_trials_liberal": data.get("n_trials_liberal"),
            "n_trials_breakdown": data.get("n_trials_breakdown"),
            "synthetic": data.get("_synthetic", False),
            "dsr_conservative": dsr_cons,
            "dsr_liberal": dsr_lib,
        })

        print(f"## {data['sys_id']}")
        if data.get("_synthetic"):
            print("  WARNING: monthly returns are synthetic")
        breakdown = data.get("n_trials_breakdown")
        if breakdown:
            print(f"  n_trials breakdown: loops={breakdown['n_improvement_loops']}, "
                  f"grid={breakdown['n_grid_search_combinations']}, "
                  f"ccy={breakdown['n_currency_choices']}, "
                  f"period={breakdown['n_period_choices']}, "
                  f"thresh={breakdown['n_threshold_choices']} "
                  f"({breakdown.get('notes', '')})")
        if "error" in dsr_cons:
            print(f"  ERROR: {dsr_cons['error']}")
        else:
            print(f"  N={data['n_trials']} (conservative): DSR={dsr_cons['dsr']:.4f}, "
                  f"SR_obs={dsr_cons['sharpe_observed']:.3f}, "
                  f"E[max SR*]={dsr_cons['expected_max_sharpe']:.3f}, "
                  f"{'PASS' if dsr_cons['passes_threshold'] else 'FAIL'}")
        if dsr_lib and "error" not in dsr_lib:
            print(f"  N={data['n_trials_liberal']} (liberal): DSR={dsr_lib['dsr']:.4f}, "
                  f"SR_obs={dsr_lib['sharpe_observed']:.3f}, "
                  f"E[max SR*]={dsr_lib['expected_max_sharpe']:.3f}, "
                  f"{'PASS' if dsr_lib['passes_threshold'] else 'FAIL'}")
        print()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "phase": "Phase 2 merge (2026-08-30, v0.3 必須ゲート化)",
        "source_pj": "minmax-fx-eval-framework v0.3 (M-R2 厳密 n_trials)",
        "v0_3_thresholds": {
            "dsr_required": 0.95,
            "k4m_payoff_ratio": 1.2,
            "n_hard_floor_months": 60,
            "dsr_pass_cap_per_year": 5,
        },
        "note": "v0.3 必須ゲート化: DSR≥0.95 / K4m≥1.2 / n_hard_floor=60 / DSR_PASS_CAP=5/年。 "
                "conservative (積算) と liberal (和算) の両 n_trials で DSR を計算。",
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
