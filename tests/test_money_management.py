"""OBS000015: 合計レバレッジ上限付き資金管理のテスト。"""
from __future__ import annotations

import copy

import pandas as pd
import pytest

from minmax_fx_dt.backtest.money_management import is_active, settle_trades


def _legacy(all_trades, initial_capital, risk_pct, max_leverage=25.0):
    """修正前の各スクリプトにあった処理の写し(比較用)。"""
    events = []
    for idx, t in enumerate(all_trades):
        events.append((t["entry_time"], 0, idx, "ENTRY"))
        events.append((t["exit_time"], 1, idx, "EXIT"))
    events.sort(key=lambda e: (e[0], e[1]))
    balance, ruined = initial_capital, False
    for _time, _o, idx, kind in events:
        t = all_trades[idx]
        if kind == "ENTRY":
            if ruined:
                t["risk_dollars"], t["skipped_ruin"] = 0.0, True
            else:
                mr = max_leverage / t["leverage_ratio"] if t["leverage_ratio"] > 0 else risk_pct
                t["risk_dollars"], t["skipped_ruin"] = balance * min(risk_pct, mr), False
        else:
            t["dollar_pnl"] = 0.0 if t["skipped_ruin"] else t["r_net"] * t["risk_dollars"]
            balance += t["dollar_pnl"]
            if balance <= 0:
                balance, ruined = 0.0, True
            t["balance_after"] = balance
    return balance


def _trade(entry, exit_, lev, r):
    return {"entry_time": pd.Timestamp(entry), "exit_time": pd.Timestamp(exit_), "leverage_ratio": lev, "r_net": r}


def _sample():
    return [
        _trade("2024-01-01 10:00", "2024-01-01 12:00", 1500.0, 1.2),
        _trade("2024-01-01 10:30", "2024-01-01 11:30", 1500.0, -1.0),
        _trade("2024-01-01 10:45", "2024-01-01 13:00", 1500.0, 2.0),
        _trade("2024-01-02 09:00", "2024-01-02 10:00", 4000.0, -1.1),
        _trade("2024-01-02 09:30", "2024-01-02 11:00", 800.0, 0.5),
    ]


def test_no_aggregate_cap_matches_legacy_exactly():
    a, b = _sample(), _sample()
    legacy_balance = _legacy(a, 1000.0, 0.01)
    balance, curve, info = settle_trades(b, initial_capital=1000.0, risk_pct=0.01, start_label="s",
                                         aggregate_leverage_cap=None)
    assert balance == pytest.approx(legacy_balance, abs=1e-12)
    for x, y in zip(a, b):
        assert y["risk_dollars"] == pytest.approx(x["risk_dollars"], abs=1e-12)
        assert y["dollar_pnl"] == pytest.approx(x["dollar_pnl"], abs=1e-12)
    assert info["n_aggregate_cap_shrunk"] == info["n_aggregate_cap_skipped"] == 0
    assert len(curve) == 1 + len(b)


def test_aggregate_cap_bounds_open_notional():
    trades = _sample()
    _, _, info = settle_trades(trades, initial_capital=1000.0, risk_pct=0.01, start_label="s",
                               aggregate_leverage_cap=25.0)
    assert info["max_aggregate_leverage"] <= 25.0 + 1e-9
    # 1本目 15倍 → 2本目は残り10倍に縮小 → 3本目は枠なしで見送り
    assert trades[0]["aggregate_cap_action"] == "none"
    assert trades[1]["aggregate_cap_action"] == "shrink"
    assert trades[1]["notional_usd"] == pytest.approx(10.0 * 1000.0)
    assert trades[2]["aggregate_cap_action"] == "skip" and not is_active(trades[2])
    assert trades[2]["dollar_pnl"] == 0.0


def test_skip_mode_skips_instead_of_shrinking():
    trades = _sample()
    settle_trades(trades, initial_capital=1000.0, risk_pct=0.01, start_label="s",
                  aggregate_leverage_cap=25.0, cap_mode="skip")
    # 1本目 15倍が保有中のため、2本目・3本目(各15倍)は全量が入らず見送り
    assert trades[1]["aggregate_cap_action"] == "skip"
    assert trades[2]["aggregate_cap_action"] == "skip"
    assert trades[3]["aggregate_cap_action"] == "none"


def test_per_position_cap_still_applies():
    trades = [_trade("2024-01-01", "2024-01-02", 4000.0, 1.0)]
    settle_trades(trades, initial_capital=1000.0, risk_pct=0.01, start_label="s", aggregate_leverage_cap=None)
    assert trades[0]["leverage_capped"] is True
    assert trades[0]["notional_usd"] == pytest.approx(25.0 * 1000.0)


def test_capacity_is_released_after_exit():
    trades = _sample()
    settle_trades(trades, initial_capital=1000.0, risk_pct=0.01, start_label="s", aggregate_leverage_cap=25.0)
    # 翌日の取引は前日のポジションが全て決済済みのため、上限の影響を受けない
    assert trades[3]["aggregate_cap_action"] == "none"


def test_default_is_gmo_cap():
    a, b = copy.deepcopy(_sample()), copy.deepcopy(_sample())
    settle_trades(a, initial_capital=1000.0, risk_pct=0.01, start_label="s")
    settle_trades(b, initial_capital=1000.0, risk_pct=0.01, start_label="s", aggregate_leverage_cap=25.0)
    assert [t["risk_dollars"] for t in a] == [t["risk_dollars"] for t in b]


def test_invalid_cap_mode():
    with pytest.raises(ValueError):
        settle_trades(_sample(), initial_capital=1000.0, risk_pct=0.01, start_label="s", cap_mode="drop")
