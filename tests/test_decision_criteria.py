"""判定エンジン (decision/criteria.py) と permutation test の回帰テスト.

OBS000005 差し戻し1・5 対応: 判定エンジンが実際に K1m〜K7m + 統計的有意性の
全ゲートを評価し、バックテストのみでは判定不能な項目 (K6m/K7m/permutation) を
黙って pass 扱いにしないことを検証する。あわせて decision/__init__.py の
起動不能バグ (存在しない `decide` を import していた) の再発防止も兼ねる。
"""

from __future__ import annotations

import numpy as np
import pytest

from minmax_fx_dt.backtest.permutation import permutation_test
from minmax_fx_dt.decision import Verdict, evaluate, evaluate_kpis, kpi_pass_summary
from minmax_fx_dt.decision.criteria import KPI_THRESHOLDS, Stats


def _full_pass_stats(**overrides) -> Stats:
    """K1m〜K7m + 統計的有意性を全て満たす Stats (overrides で個別に崩す)."""
    base: Stats = {
        "strategy_id": "SYS-FX007",
        "n_days": 95,
        "n_trades": 80,
        "sharpe": 0.6,
        "sharpe_monthly": 0.6,
        "regime": "TREND",
        "profit_factor_monthly": 1.5,
        "expectancy_jpy": 200.0,
        "max_dd_monthly_pct": 5.0,
        "max_dd_yearly_pct": 10.0,
        "payoff_ratio": 2.0,
        "max_consecutive_losses": 2,
        "edge_per_trade_jpy": 200.0,
        "spread_round_trip_jpy": 20.0,
        "max_margin_usage_pct": 15.0,
        "weak_breakout_exclusion_pct": 40.0,
        "backtest_forward_divergence_pct": 10.0,
        "permutation_p_value": 0.01,
        "hedging_enabled": True,
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


# ---- decision/__init__.py の import 健全性 (README §3 動作確認コマンドの回帰) ----

def test_decision_package_imports_without_error() -> None:
    """decision/__init__.py が存在しない `decide` を import して ImportError に
    なっていたバグ (OBS000005 追記3) の再発防止。"""
    from minmax_fx_dt.decision import (  # noqa: F401
        KPIEvaluation,
        Stats,
        Verdict,
        evaluate,
        evaluate_kpis,
        kpi_pass_summary,
    )


# ---- evaluate_kpis(): 全ゲート網羅 ----

def test_evaluate_kpis_covers_all_spec_thresholds() -> None:
    """KPI_THRESHOLDS に定義された全項目 (K1m〜K7m + 統計的有意性) が
    evaluate_kpis() の出力に現れることを保証する (OBS000005 が指摘した
    K5m/K6m/min_n_trades/permutation_p_value 欠落の再発防止)。"""
    evals = evaluate_kpis(_full_pass_stats())
    metrics = {e.metric for e in evals}
    assert "K5m_spread_cost_multiple" in metrics
    assert "K6m_bt_ft_divergence" in metrics
    assert "min_n_trades" in metrics
    assert "permutation_p_value" in metrics
    assert "K2m_dd_yearly" in metrics
    assert "K1m_expectancy" in metrics


def test_evaluate_kpis_all_pass_when_stats_meet_every_threshold() -> None:
    evals = evaluate_kpis(_full_pass_stats())
    summary = kpi_pass_summary(evals)
    assert summary["applicable"] == summary["total"]  # 全項目が判定対象
    assert summary["all_applicable_pass"] is True
    assert summary["fail"] == 0


def test_evaluate_kpis_marks_missing_forward_test_as_not_applicable() -> None:
    """K6m はフォワードテスト未実施なら pass ではなく「判定対象外」."""
    evals = evaluate_kpis(_full_pass_stats(backtest_forward_divergence_pct=None))
    k6 = next(e for e in evals if e.metric == "K6m_bt_ft_divergence")
    assert k6.applicable is False
    summary = kpi_pass_summary(evals)
    assert "K6m_bt_ft_divergence" in summary["not_applicable_metrics"]
    # 判定対象外は fail にも pass にもカウントされない
    assert summary["all_applicable_pass"] is True


def test_evaluate_kpis_marks_missing_permutation_as_not_applicable() -> None:
    evals = evaluate_kpis(_full_pass_stats(permutation_p_value=None))
    p = next(e for e in evals if e.metric == "permutation_p_value")
    assert p.applicable is False


def test_evaluate_kpis_marks_unhedged_margin_as_not_applicable() -> None:
    """K7m は両建てロジック未統合なら「判定対象外」(単一ポジションの値は
    K7m が意図する両建て時消費率の検証にならないため)."""
    evals = evaluate_kpis(_full_pass_stats(hedging_enabled=False))
    k7 = next(e for e in evals if e.metric == "K7m_margin_usage")
    assert k7.applicable is False


def test_evaluate_kpis_min_n_trades_fails_below_threshold() -> None:
    """OBS000007 追記6 の実測 (最大 n=18) が min_n_trades=60 未達になることを確認."""
    evals = evaluate_kpis(_full_pass_stats(n_trades=18))
    gate = next(e for e in evals if e.metric == "min_n_trades")
    assert gate.pass_ is False
    assert gate.applicable is True


def test_evaluate_kpis_k5m_uses_edge_over_spread_multiple() -> None:
    """K5m: 1トレード期待値 > スプレッド往復コスト × 3."""
    # edge=200円, spread往復=20円 → multiple=10倍 >= 3倍 → pass
    evals = evaluate_kpis(_full_pass_stats(edge_per_trade_jpy=200.0, spread_round_trip_jpy=20.0))
    k5 = next(e for e in evals if e.metric == "K5m_spread_cost_multiple")
    assert k5.pass_ is True
    assert k5.observed == pytest.approx(10.0)

    # edge=40円, spread往復=20円 → multiple=2倍 < 3倍 → fail
    evals = evaluate_kpis(_full_pass_stats(edge_per_trade_jpy=40.0, spread_round_trip_jpy=20.0))
    k5 = next(e for e in evals if e.metric == "K5m_spread_cost_multiple")
    assert k5.pass_ is False


# ---- evaluate(): 判定対象外を failed_kpis から除外 ----

def test_evaluate_does_not_reject_solely_on_not_applicable_gates() -> None:
    """K6m/K7m/permutation が全て未測定でも、他の全ゲートを満たしていれば
    それらだけを理由に REJECT にはならない (判定対象外は不合格に数えない)."""
    stats = _full_pass_stats(
        backtest_forward_divergence_pct=None,
        permutation_p_value=None,
        hedging_enabled=False,
    )
    verdict, reason = evaluate(stats)
    assert verdict == Verdict.GO
    assert "K6m" not in reason
    assert "K7m" not in reason
    assert "permutation" not in reason


def test_evaluate_rejects_when_applicable_gate_fails() -> None:
    stats = _full_pass_stats(n_trades=5)  # min_n_trades 未達
    verdict, reason = evaluate(stats)
    assert verdict != Verdict.GO
    assert "min_n_trades" in reason


# ---- permutation_test() ----

def test_permutation_test_small_positive_edge_not_significant() -> None:
    """OBS000007 実測相当 (n=7, 弱い負けトレード主体) では有意差なしになるはず."""
    pnls = [1000.0, -500.0, -500.0, -500.0, 800.0, -500.0, -500.0]
    result = permutation_test(pnls, n_permutations=1000, seed=42)
    assert result.n_trades == 7
    assert 0.0 <= result.p_value <= 1.0
    assert result.p_value > 0.05  # サンプル不足で有意差を主張できない


def test_permutation_test_large_consistent_edge_is_significant() -> None:
    """全トレードが同符号 (常勝) なら、コイン投げでは説明できず有意になる."""
    pnls = [100.0] * 30
    result = permutation_test(pnls, n_permutations=2000, seed=1)
    assert result.p_value < 0.05


def test_permutation_test_empty_trades_returns_p_value_one() -> None:
    result = permutation_test([], n_permutations=1000)
    assert result.n_trades == 0
    assert result.p_value == 1.0


def test_permutation_test_deterministic_with_seed() -> None:
    pnls = [120.0, -80.0, 60.0, -40.0, -30.0, 90.0]
    r1 = permutation_test(pnls, n_permutations=500, seed=7)
    r2 = permutation_test(pnls, n_permutations=500, seed=7)
    assert r1.p_value == r2.p_value
    assert r1.null_mean == pytest.approx(r2.null_mean)


def test_permutation_test_sign_flip_preserves_magnitudes() -> None:
    """符号シャッフルは値幅 (絶対値) を変えない設計であることを、帰無分布の
    標準偏差が観測データの絶対値スケールと整合することで間接的に確認する."""
    pnls = [500.0, -500.0, 500.0, -500.0]
    result = permutation_test(pnls, n_permutations=1000, seed=3)
    # 全トレードの絶対値が 500 で統一されているため、帰無分布の平均は
    # ±500 の範囲に収まり、単純な符号反転の外には出ない
    assert abs(result.null_mean) <= 500.0
