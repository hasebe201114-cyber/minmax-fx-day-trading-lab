"""SYS-FX011 (EXP-FX000005) 週末強制クローズ不具合(F1)の回帰テスト.

外部レビュー(`obs/.../85外部レビュー/2026-08-20_EXP-FX000005_External_Review/00_REVIEW_SUMMARY.md`
F1)で、`is_weekend_close_time(ts) = (ts.weekday()==5 and ts.hour>=6)` が
「土曜06:00 JST以降のH1バーの存在」を前提にしているが、市場が閉まっている
ためそのバー自体がデータに存在せず、週末強制クローズが一度も発動していない
ことが判明した(622件中41件=6.6%が週末を跨いで保有)。

修正: 次バーとのISO週番号を比較し「週内の最終バー(次バーが週明けになる
バー)」で強制クローズする方式に変更(T-01)。本テストはこの修正の回帰確認。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_vol_breakout_dow_theory import (  # noqa: E402
    is_weekend_close_time, simulate_scaled_scheme,
)


def test_last_bar_before_week_boundary_is_true():
    """金曜最終バー(次バーが月曜=週またぎ)ではTrue."""
    idx = pd.DatetimeIndex([
        "2024-01-05 21:00", "2024-01-05 22:00", "2024-01-05 23:00",  # 金曜
        "2024-01-08 07:00", "2024-01-08 08:00",  # 月曜(市場休場中のバーは存在しない)
    ])
    assert is_weekend_close_time(idx, 2) is True   # 金曜最終バー
    assert is_weekend_close_time(idx, 0) is False  # 金曜だが最終バーではない
    assert is_weekend_close_time(idx, 3) is False  # 月曜バー


def test_last_bar_of_dataset_is_false():
    """データの末尾(次バーが存在しない)は週またぎ判定できないためFalse."""
    idx = pd.DatetimeIndex(["2024-01-05 21:00", "2024-01-05 22:00"])
    assert is_weekend_close_time(idx, 1) is False


def _build_h1_with_weekend_gap() -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.DatetimeIndex([
        "2024-01-05 20:00", "2024-01-05 21:00", "2024-01-05 22:00",  # 金曜(最終バーは22:00)
        "2024-01-08 07:00", "2024-01-08 08:00", "2024-01-08 09:00",  # 月曜
    ])
    n = len(idx)
    h1 = pd.DataFrame({
        "open": [150.0] * n, "high": [150.2] * n, "low": [149.8] * n, "close": [150.0] * n,
    }, index=idx)
    atr_h1 = pd.Series([0.2] * n, index=idx)
    return h1, atr_h1


def test_position_force_closed_at_last_bar_of_week_not_carried_over():
    """金曜21:40エントリーのポジションが、週明け(月曜)まで持ち越されず
    金曜最終バー(22:00)でWEEKEND_*決済されることを確認する(F1の再現防止)。"""
    h1, atr_h1 = _build_h1_with_weekend_gap()
    entry_ts = pd.Timestamp("2024-01-05 21:40")
    entry = dict(direction="UP", entry_idx=1, entry_price=150.0,
                 stop0=149.0, initial_risk=1.0, entry_ts=entry_ts)

    result = simulate_scaled_scheme(h1, atr_h1, entry, trail_mult=3.0)

    assert result["exit_time"] == pd.Timestamp("2024-01-05 22:00")
    assert result["exit_reason"] in ("WEEKEND_NO_TP", "TP_THEN_WEEKEND")
    assert result["exit_time"] < pd.Timestamp("2024-01-08 07:00")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
