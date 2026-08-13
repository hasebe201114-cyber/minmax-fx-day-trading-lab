"""撤退/採用判定基準 (SYS-FX007 レンジブレイク・プルバック戦略向け).

背景 (親プロジェクト minmax-trading-pilot の criteria.py を流用・拡張):
- 親PJは単一 Sharpe 閾値で判定 (044=0.5, 046=1.0, 048=0.5, 050=0.5)
- 本PJの SYS-FX007 は K1m〜K7m の 7 指標で多面的に評価
  * K1m: 月次 Sharpe ≥ 0.4, PF ≥ 1.2, 期待値 > 0
  * K2m: 月間 DD ≤ 10%, 年間 DD ≤ 20%
  * K3m: 最大連続損失 ≤ 5 トレード
  * K4m: ペイオフレシオ ≥ 1.5
  * K5m: 1トレード期待値 > スプレッド往復 × 3
  * K6m: バックテスト ↔ フォワード乖離率 ≤ 30%
  * K7m: 両建て証拠金消費率 ≤ 30%

ルール (本PJ 2026-08-13 司令塔 GO + spec 評価基準を数値固定):
1. Day 30 まで: 累積 Sharpe < -0.5 → 撤退検討 / 取引数 0 → 戦略ロジック再点検
2. Day 30: RANGE 2 シグナル以上 + Sharpe < 0 → 評価延長
3. Day 60: 累積 Sharpe < 0.0 + RANGE 2 シグナル以上 → Phase 2 延長 or 撤退
4. Day 90: 累積 Sharpe < 0.4 → 本採用見送り
5. Day 90: 取引数 < 3 → 06 ヶ月延長を自動オプション化

親PJ criteria.py からの差分:
- 単一 Sharpe → K1m〜K7m の 7 指標で評価
- STRATEGY_GO_THRESHOLD を 1 戦略に最適化 (SYS-FX007 = 0.4)
- 多通貨 (5 通貨) の KPI 合算判定
- 弱いブレイク排除率 (v2 拡張) を追加チェック
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[3]
REGIME_LATEST = ROOT / "research" / "regime_latest.json"


class Verdict(str, Enum):
    """撤退/採用判定."""

    GO = "GO"
    WATCH = "WATCH"
    REJECT = "REJECT"
    SAMPLE_DEFICIT = "SAMPLE-DEFICIT"
    REGIME_EXTEND = "REGIME-EXTEND"  # レンジ相場で評価延長


# 戦略別 GO 閾値 (本PJは SYS-FX007 のみ)
# 親PJ: {"EXP-OBS000044": 0.5, "EXP-OBS000046": 1.0, ...}
STRATEGY_GO_THRESHOLD: dict[str, float] = {
    "SYS-FX007": 0.4,  # レンジブレイク・プルバック (中期スイング + 両建て)
}

# SYS-FX007 の K1m〜K7m 閾値 (research/EXP-FX000001/00-spec.md から転記)
KPI_THRESHOLDS: dict[str, dict[str, float]] = {
    "SYS-FX007": {
        "sharpe_monthly": 0.4,            # K1m
        "profit_factor_monthly": 1.2,     # K1m
        "max_dd_monthly_pct": 10.0,       # K2m (証拠金 %)
        "max_dd_yearly_pct": 20.0,        # K2m
        "max_consecutive_losses": 5,      # K3m
        "payoff_ratio": 1.5,              # K4m
        "spread_cost_multiple": 3.0,      # K5m
        "backtest_forward_divergence_pct": 30.0,  # K6m
        "max_margin_usage_pct": 30.0,     # K7m
        "weak_breakout_exclusion_pct": 30.0,  # v2 拡張
        "min_n_trades": 60,               # 統計的有意性
        "permutation_p_value": 0.05,      # 統計的有意性
    },
}

# Day 30 短期撤退閾値 (親PJ踏襲)
DAY30_SHARPE_REJECT = -0.5

# Day 60 中期撤退閾値 (親PJ踏襲)
DAY60_SHARPE_REJECT = 0.0

# 取引数 SAMPLE-DEFICIT 閾値 (親PJ踏襲)
SAMPLE_DEFICIT_N_TRADES = 3


@dataclass
class KPIEvaluation:
    """K1m〜K7m 評価結果."""

    metric: str
    observed: float
    threshold: float
    pass_: bool
    note: str = ""


class Stats(TypedDict, total=False):
    """撤退判定に必要な統計量 (SYS-FX007 用に拡張)."""

    strategy_id: str  # 例: "SYS-FX007"
    n_days: int
    n_trades: int
    sharpe: float
    max_dd: float  # 口座 % で記録
    regime: str  # "TREND" / "RANGE" / "RANGE_WARN" / "INSUFFICIENT_DATA"
    # v2 拡張: 多 KPI
    profit_factor: float
    payoff_ratio: float
    max_consecutive_losses: int
    max_margin_usage_pct: float
    weak_breakout_exclusion_pct: float
    backtest_forward_divergence_pct: float
    permutation_p_value: float
    n_trades_per_currency: dict[str, int]  # 通貨別取引数


def load_regime() -> str:
    """regime_latest.json からレジームを読み込み (親PJ踏襲)."""
    if not REGIME_LATEST.exists():
        return "INSUFFICIENT_DATA"
    try:
        data = json.loads(REGIME_LATEST.read_text(encoding="utf-8"))
        return str(data.get("regime", "INSUFFICIENT_DATA"))
    except (json.JSONDecodeError, OSError):
        return "INSUFFICIENT_DATA"


def is_range_regime(regime: str) -> bool:
    """レンジ相場判定 (RANGE 確定, 親PJ踏襲)."""
    return regime == "RANGE"


def evaluate_kpis(stats: Stats) -> list[KPIEvaluation]:
    """K1m〜K7m 評価 (本PJ固有)."""
    strategy_id = stats.get("strategy_id", "")
    thresholds = KPI_THRESHOLDS.get(strategy_id)
    if not thresholds:
        return []

    evals: list[KPIEvaluation] = []

    # K1m: 月次 Sharpe
    sharpe = float(stats.get("sharpe", 0.0))
    evals.append(
        KPIEvaluation(
            "K1m_sharpe",
            sharpe,
            thresholds["sharpe_monthly"],
            sharpe >= thresholds["sharpe_monthly"],
        )
    )

    # K1m: 月次 PF
    pf = float(stats.get("profit_factor", 0.0))
    evals.append(
        KPIEvaluation(
            "K1m_pf",
            pf,
            thresholds["profit_factor_monthly"],
            pf >= thresholds["profit_factor_monthly"],
        )
    )

    # K2m: 月間 DD
    max_dd = float(stats.get("max_dd", 0.0))
    evals.append(
        KPIEvaluation(
            "K2m_dd_monthly",
            abs(max_dd),
            thresholds["max_dd_monthly_pct"],
            abs(max_dd) <= thresholds["max_dd_monthly_pct"],
            "証拠金 % に対する比率",
        )
    )

    # K3m: 最大連続損失
    mcl = int(stats.get("max_consecutive_losses", 999))
    evals.append(
        KPIEvaluation(
            "K3m_max_consecutive_losses",
            float(mcl),
            thresholds["max_consecutive_losses"],
            float(mcl) <= thresholds["max_consecutive_losses"],
        )
    )

    # K4m: ペイオフレシオ
    pr = float(stats.get("payoff_ratio", 0.0))
    evals.append(
        KPIEvaluation(
            "K4m_payoff",
            pr,
            thresholds["payoff_ratio"],
            pr >= thresholds["payoff_ratio"],
        )
    )

    # K7m: 両建て証拠金消費率
    margin = float(stats.get("max_margin_usage_pct", 100.0))
    evals.append(
        KPIEvaluation(
            "K7m_margin_usage",
            margin,
            thresholds["max_margin_usage_pct"],
            margin <= thresholds["max_margin_usage_pct"],
        )
    )

    # v2 拡張: 弱いブレイク排除率
    we = float(stats.get("weak_breakout_exclusion_pct", 0.0))
    evals.append(
        KPIEvaluation(
            "v2_weak_breakout_exclusion",
            we,
            thresholds["weak_breakout_exclusion_pct"],
            we >= thresholds["weak_breakout_exclusion_pct"],
        )
    )

    return evals


def evaluate(stats: Stats) -> tuple[Verdict, str]:
    """撤退/採用判定 (SYS-FX007 レンジブレイク・プルバック戦略).

    K1m〜K7m すべて満たせば GO、Day ベースの特例判定あり。
    """
    strategy_id = stats.get("strategy_id", "")
    n_days = int(stats.get("n_days", 0))
    n_trades = int(stats.get("n_trades", 0))
    sharpe = float(stats.get("sharpe", 0.0))
    regime = stats.get("regime") or load_regime()

    go_threshold = STRATEGY_GO_THRESHOLD.get(strategy_id, 0.4)

    # 取引数 0 → SAMPLE-DEFICIT
    if n_trades == 0:
        return Verdict.SAMPLE_DEFICIT, f"取引数 0 (Day {n_days})"

    # 取引数 < 3 (Day 90 経過後) → 06 ヶ月延長オプション
    if n_days >= 90 and n_trades < SAMPLE_DEFICIT_N_TRADES:
        return (
            Verdict.SAMPLE_DEFICIT,
            f"取引数 {n_trades} < {SAMPLE_DEFICIT_N_TRADES} (Day {n_days} 経過) → 06 ヶ月延長検討",
        )

    # K1m〜K7m 評価
    kpi_evals = evaluate_kpis(stats)
    failed_kpis = [e.metric for e in kpi_evals if not e.pass_]

    # Day 30 まで: 短期撤退閾値
    if n_days < 30:
        if sharpe < DAY30_SHARPE_REJECT:
            return (
                Verdict.REJECT,
                f"短期: Sharpe {sharpe:.2f} < {DAY30_SHARPE_REJECT} (Day {n_days})",
            )
        if sharpe < 0 and is_range_regime(regime):
            return (
                Verdict.REGIME_EXTEND,
                f"レンジ相場例外: Sharpe {sharpe:.2f} < 0 だが regime={regime} で評価延長",
            )
        if sharpe >= go_threshold and not failed_kpis:
            return Verdict.GO, f"早期 GO: Sharpe {sharpe:.2f} >= {go_threshold} (Day {n_days})"
        if failed_kpis:
            return (
                Verdict.WATCH,
                f"Day {n_days}: Sharpe {sharpe:.2f}, KPI 失敗: {','.join(failed_kpis)}",
            )
        return Verdict.WATCH, f"Day {n_days}: Sharpe {sharpe:.2f}"

    # Day 30-60: 中期評価
    if n_days < 60:
        if sharpe >= go_threshold and not failed_kpis:
            return Verdict.GO, f"GO: Sharpe {sharpe:.2f} >= {go_threshold} (Day {n_days})"
        if sharpe < 0 and is_range_regime(regime):
            return (
                Verdict.REGIME_EXTEND,
                f"レンジ相場例外: Sharpe {sharpe:.2f} < 0 だが regime={regime} で評価延長",
            )
        if failed_kpis:
            return (
                Verdict.WATCH,
                f"Day {n_days}: Sharpe {sharpe:.2f}, KPI 失敗: {','.join(failed_kpis)}",
            )
        return Verdict.WATCH, f"Day {n_days}: Sharpe {sharpe:.2f}"

    # Day 60-90: 中期撤退閾値
    if n_days < 90:
        if sharpe >= go_threshold and not failed_kpis:
            return Verdict.GO, f"GO: Sharpe {sharpe:.2f} >= {go_threshold} (Day {n_days})"
        if sharpe < DAY60_SHARPE_REJECT and is_range_regime(regime):
            return (
                Verdict.REJECT,
                f"中期 REJECT: Sharpe {sharpe:.2f} < {DAY60_SHARPE_REJECT} + regime={regime}",
            )
        if sharpe < DAY60_SHARPE_REJECT:
            return (
                Verdict.WATCH,
                f"Day {n_days}: Sharpe {sharpe:.2f} < {DAY60_SHARPE_REJECT} 注視",
            )
        if failed_kpis:
            return (
                Verdict.WATCH,
                f"Day {n_days}: Sharpe {sharpe:.2f}, KPI 失敗: {','.join(failed_kpis)}",
            )
        return Verdict.WATCH, f"Day {n_days}: Sharpe {sharpe:.2f}"

    # Day 90+: 本採用判定 (KPI すべて pass 必須)
    if sharpe >= go_threshold and not failed_kpis:
        return Verdict.GO, f"GO: Sharpe {sharpe:.2f} >= {go_threshold} (Day {n_days})"
    if failed_kpis:
        return (
            Verdict.REJECT,
            f"本採用見送り: KPI 失敗 {','.join(failed_kpis)} (Day {n_days})",
        )
    return (
        Verdict.REJECT,
        f"本採用見送り: Sharpe {sharpe:.2f} < {go_threshold} (Day {n_days})",
    )


def main() -> int:
    """CLI エントリポイント: SYS-FX007 テストケース."""
    test_cases: list[Stats] = [
        {
            "strategy_id": "SYS-FX007",
            "n_days": 25,
            "n_trades": 5,
            "sharpe": -0.6,
            "max_dd": 8.0,
            "regime": "TREND",
            "profit_factor": 1.3,
            "payoff_ratio": 1.8,
            "max_consecutive_losses": 3,
            "max_margin_usage_pct": 15.0,
            "weak_breakout_exclusion_pct": 35.0,
        },
        {
            "strategy_id": "SYS-FX007",
            "n_days": 95,
            "n_trades": 80,
            "sharpe": 0.5,
            "max_dd": 7.0,
            "regime": "TREND",
            "profit_factor": 1.4,
            "payoff_ratio": 2.0,
            "max_consecutive_losses": 4,
            "max_margin_usage_pct": 22.0,
            "weak_breakout_exclusion_pct": 40.0,
        },
        {
            "strategy_id": "SYS-FX007",
            "n_days": 95,
            "n_trades": 2,
            "sharpe": 0.3,
            "max_dd": 5.0,
            "regime": "TREND",
            "profit_factor": 1.1,
            "payoff_ratio": 1.5,
            "max_consecutive_losses": 2,
            "max_margin_usage_pct": 10.0,
            "weak_breakout_exclusion_pct": 30.0,
        },
    ]
    for stats in test_cases:
        verdict, reason = evaluate(stats)
        print(
            f"{stats['strategy_id']} Day {stats['n_days']:>3} "
            f"Sharpe {stats['sharpe']:>5.2f} regime={stats['regime']:<6} "
            f"-> {verdict.value:<18} {reason}"
        )
    print(f"\n[decision_criteria] ts={datetime.now(UTC).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
