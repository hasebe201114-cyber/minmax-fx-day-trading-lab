"""SYS-FX011 (EXP-FX000005) simulate_scaled_scheme の境界値回帰テスト.

改善ループ第4試行(ブレイク検出閾値N=2.5への緩和感度分析)で、エントリーが
利用可能なH1データの最終バー付近で発生するケースが急増し、以前は
exit_time < entry_time というイベント順序違反を引き起こす潜在バグが顕在化した
(KeyError: 'risk_dollars' として$1,000バックテストのイベント駆動エクイティ
カーブ構築時に発覚)。修正: エントリー後にH1バーが1本も残っていない場合、
即座にr=0でフラット決済しexit_time=entry_ts(エントリー自身のM5時刻)を返す。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_vol_breakout_dow_theory import simulate_scaled_scheme  # noqa: E402


def _build_h1(n: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2024-01-01 00:00", periods=n, freq="h")
    h1 = pd.DataFrame({
        "open": [150.0] * n, "high": [150.2] * n, "low": [149.8] * n, "close": [150.0] * n,
    }, index=idx)
    atr_h1 = pd.Series([0.2] * n, index=idx)
    return h1, atr_h1


def test_entry_on_last_h1_bar_does_not_exit_before_entry():
    """エントリーが利用可能な最終H1バーで発生した場合、exit_time >= entry_tsを保証する."""
    h1, atr_h1 = _build_h1(n=10)
    entry_ts = h1.index[-1] + pd.Timedelta(minutes=30)  # 最終H1バー内のM5エントリー時刻
    entry = dict(direction="UP", entry_idx=len(h1) - 1, entry_price=150.0,
                 stop0=149.5, initial_risk=0.5, entry_ts=entry_ts)

    result = simulate_scaled_scheme(h1, atr_h1, entry, trail_mult=3.0)

    assert result["exit_time"] >= entry_ts
    assert result["r"] == 0.0
    assert result["exit_reason"] == "NO_H1_DATA_AFTER_ENTRY"


def test_entry_with_h1_bars_remaining_still_resolves_normally():
    """エントリー後にH1バーが残っている通常ケースは従来通り動作する(回帰確認)."""
    h1, atr_h1 = _build_h1(n=10)
    entry_ts = h1.index[2] + pd.Timedelta(minutes=15)
    entry = dict(direction="UP", entry_idx=2, entry_price=150.0,
                 stop0=149.5, initial_risk=0.5, entry_ts=entry_ts)

    result = simulate_scaled_scheme(h1, atr_h1, entry, trail_mult=3.0)

    assert result["exit_time"] >= entry_ts
    assert result["exit_reason"] != "NO_H1_DATA_AFTER_ENTRY"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
