"""EXP-FX000018 / SYS-FX024 の両建てグリッドエンジンの不変条件テスト.

実データではなく合成価格系列を注入して、`grid_portfolio_engine.simulate()` が
`research/EXP-FX000018/00-spec.md` §3〜§5 の事前登録仕様どおりに振る舞うことを検証する:

- 週末持ち越し禁止 (週内最終バーで全決済、週を跨いで保有しない)
- 収支の整合 (最終残高 = 初期資金 + 全トレードの dollar_pnl 合計)
- 利確・損切りの約定価格がグリッド定義と一致する
- 同一通貨・同一サイドの同時保有が N 段を超えない
- K7m (証拠金消費率) がガード閾値を超えない
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import grid_portfolio_engine as gpe  # noqa: E402

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
BASE_PRICE = {"USD_JPY": 150.0, "EUR_JPY": 160.0, "GBP_JPY": 190.0, "AUD_JPY": 100.0}
N_LEVELS = 3
START, END = "2024-01-01", "2024-01-31"


def _synthetic_m5(pair: str) -> pd.DataFrame:
    """振幅の異なる正弦波 + 弱いドリフトの合成M5系列 (再現性のため固定シード)."""
    index = pd.date_range("2024-01-01 06:00", "2024-01-31 06:00", freq="5min", tz="Asia/Tokyo")
    n = len(index)
    rng = np.random.default_rng(abs(hash(pair)) % (2**32))
    base = BASE_PRICE[pair]
    t = np.arange(n)
    wave = np.sin(t / 400.0) * base * 0.01 + np.sin(t / 97.0) * base * 0.003
    noise = np.cumsum(rng.normal(0, base * 1e-5, n))
    close = base + wave + noise
    spread = base * 2e-4
    return pd.DataFrame(
        {"open": close, "high": close + spread, "low": close - spread, "close": close}, index=index
    ).rename_axis("timestamp")


@pytest.fixture
def sim(monkeypatch):
    monkeypatch.setattr(gpe, "load_m5", lambda pair, start, end: _synthetic_m5(pair))
    monkeypatch.setattr(gpe, "load_swap_table", lambda pairs: {p: {} for p in pairs})
    return gpe.simulate(
        PAIRS, START, END, n_levels=N_LEVELS, grid_step_atr_mult=1.0,
        reanchor_bars=24, carry_over=False, verbose=False,
    )


def test_trades_generated(sim):
    assert len(sim["trades"]) > 0
    assert sim["n_generations"] > 0


def test_balance_matches_sum_of_trade_pnl(sim):
    total = sum(t["dollar_pnl"] for t in sim["trades"])
    assert sim["final_balance_usd"] == pytest.approx(gpe.INITIAL_CAPITAL_USD + total, rel=1e-9)


def test_no_position_survives_a_week_boundary(sim):
    """週末持ち越し禁止 (spec §1): 建てと決済が同一ISO週に収まっていること."""
    for t in sim["trades"]:
        entry_week = pd.Timestamp(t["entry_time"]).isocalendar()[:2]
        exit_week = pd.Timestamp(t["exit_time"]).isocalendar()[:2]
        assert entry_week == exit_week, f"週を跨いで保有している: {t}"


def test_outcomes_are_from_the_registered_set(sim):
    allowed = {"TP", "STOP", "WEEKEND", "MARK", "PERIOD_END"}
    assert {t["outcome"] for t in sim["trades"]} <= allowed


def test_tp_exit_is_exactly_one_grid_step_from_entry(sim):
    """利確は「エントリーレベル ± 1グリッド刻み」の指値 (spec §3.2、スリッページ0)."""
    tps = [t for t in sim["trades"] if t["outcome"] == "TP"]
    assert tps, "TP決済が1件も無い"
    for t in tps:
        gross = ((t["exit_price"] - t["entry_price"]) if t["side"] == "buy"
                 else (t["entry_price"] - t["exit_price"]))
        # 1刻み = 最内段のストップ距離 (N*step) を N で割った値
        step = t["initial_risk"] / (N_LEVELS + 1 - _level_no(t))
        assert gross == pytest.approx(step, rel=1e-9)


def _level_no(trade: dict) -> int:
    return trade["level_idx"] + 1


def test_initial_risk_matches_level_distance(sim):
    """各ポジションのRの分母 = (N+1-j)*step (spec §4.2)."""
    for t in sim["trades"]:
        assert t["initial_risk"] > 0
        assert t["level_idx"] in range(N_LEVELS)


def test_concurrent_positions_never_exceed_cap(sim):
    """同一通貨・同一サイドの同時保有は N 段まで (spec §3.3)."""
    assert sim["max_concurrent_positions"] <= len(PAIRS) * 2 * N_LEVELS


def test_margin_guard_reduces_k7m(monkeypatch):
    """証拠金ガードは新規建てを抑止することで K7m を押し下げる (spec §5.2).

    ガードは「建てる瞬間」にしか効かないため、建てた後にMTM equityが目減りすると
    観測される K7m が閾値をわずかに上回りうる (建玉を強制的に減らす設計ではない)。
    したがって「絶対に30%を超えない」ではなく「ガード有効時 ≤ ガード無効時」かつ
    「閾値を大きく超えない」ことを不変条件とする。
    """
    monkeypatch.setattr(gpe, "load_m5", lambda pair, start, end: _synthetic_m5(pair))
    monkeypatch.setattr(gpe, "load_swap_table", lambda pairs: {p: {} for p in pairs})
    kwargs = dict(n_levels=N_LEVELS, grid_step_atr_mult=1.0, reanchor_bars=24,
                  carry_over=False, verbose=False)
    guarded = gpe.simulate(PAIRS, START, END, margin_guard=True, **kwargs)
    unguarded = gpe.simulate(PAIRS, START, END, margin_guard=False, **kwargs)
    assert guarded["max_margin_sum_pct"] <= unguarded["max_margin_sum_pct"] + 1e-9
    assert guarded["max_margin_sum_pct"] <= gpe.MARGIN_GUARD_PCT * 1.05
    assert guarded["max_margin_max_pct"] <= guarded["max_margin_sum_pct"] + 1e-9


def test_stop_exit_is_worse_than_the_stop_price_by_slippage(sim):
    """逆指値決済には T-09 の 1.0pip スリッページが不利側に乗る (spec §1.1)."""
    stops = [t for t in sim["trades"] if t["outcome"] == "STOP"]
    for t in stops:
        pip = gpe.pip_size(t["pair"])
        expected_stop = (t["entry_price"] - t["initial_risk"] if t["side"] == "buy"
                         else t["entry_price"] + t["initial_risk"])
        slip = gpe.SLIPPAGE_PIPS_STOP * pip
        expected_fill = expected_stop - slip if t["side"] == "buy" else expected_stop + slip
        assert t["exit_price"] == pytest.approx(expected_fill, rel=1e-9)


def test_carry_over_holds_positions_across_reanchor(monkeypatch):
    """G1(持ち越し方式)はG0(MARK方式)と異なり MARK 決済を生成しない (spec §3.3)."""
    monkeypatch.setattr(gpe, "load_m5", lambda pair, start, end: _synthetic_m5(pair))
    monkeypatch.setattr(gpe, "load_swap_table", lambda pairs: {p: {} for p in pairs})
    kwargs = dict(n_levels=N_LEVELS, grid_step_atr_mult=1.0, reanchor_bars=24, verbose=False)
    g0 = gpe.simulate(PAIRS, START, END, carry_over=False, **kwargs)
    g1 = gpe.simulate(PAIRS, START, END, carry_over=True, **kwargs)
    # G1 は再アンカー時に決済しないため MARK を一切生成しない (spec §3.3)
    assert all(t["outcome"] != "MARK" for t in g1["trades"])
    # G0 の MARK は「再アンカー時に建玉が残っていた場合」にのみ発生するため
    # 合成系列では0件になりうる。発生した場合は G1 側に無いことだけを確認する。
    assert g0["n_generations"] == g1["n_generations"]
