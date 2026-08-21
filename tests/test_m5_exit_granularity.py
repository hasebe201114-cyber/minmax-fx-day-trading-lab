"""SYS-FX011 (EXP-FX000005) 決済判定のM5化(§1-2/T-03)の回帰テスト.

外部レビュー §1-2: エントリーはM5時刻で決まるが、決済判定
(`simulate_scaled_scheme`)は`entry_idx+1`(=エントリーしたH1バーの次のバー)
から始まっていたため、エントリー直後〜そのH1バー終了まで(平均約30分)の
SL/TP到達が完全に不可視だった。`m5_exit`パラメータでM5バー単位の決済判定に
切り替えられるようにした修正の回帰確認。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_vol_breakout_dow_theory import simulate_scaled_scheme  # noqa: E402


def _build_bars():
    h1_idx = pd.date_range("2024-01-02 00:00", periods=3, freq="h")
    h1 = pd.DataFrame({"open": [150.0] * 3, "high": [150.0] * 3,
                        "low": [150.0] * 3, "close": [150.0] * 3}, index=h1_idx)
    atr_h1 = pd.Series([0.2] * 3, index=h1_idx)

    m5_idx = pd.date_range("2024-01-02 00:00", periods=36, freq="5min")
    lows = [150.0] * 36
    lows[8] = 148.0  # エントリー(M5 idx=6)自身のH1バー内(00:40)でストップを割り込む
    m5 = pd.DataFrame({"open": [150.0] * 36, "high": [150.0] * 36,
                        "low": lows, "close": [150.0] * 36}, index=m5_idx)
    return h1, atr_h1, m5


def test_h1_mode_misses_sl_within_entry_bar_remainder():
    """旧仕様(H1バー単位)は、エントリーした H1 バー自身の残り時間内のSL到達を見逃す
    (entry_idx+1から判定が始まるため)。この挙動自体は後方互換として維持する。"""
    h1, atr_h1, m5 = _build_bars()
    entry = dict(direction="UP", entry_idx=0, entry_m5_idx=6, entry_price=150.0,
                 stop0=149.0, initial_risk=1.0, entry_ts=m5.index[6])

    result = simulate_scaled_scheme(h1, atr_h1, entry, trail_mult=3.0)

    assert result["exit_reason"] != "SL_INITIAL_NO_TP"


def test_m5_mode_catches_sl_within_entry_bar_remainder():
    """m5_exitを渡すと、同じシナリオでエントリー直後のSL到達を正しく検知する。"""
    h1, atr_h1, m5 = _build_bars()
    entry = dict(direction="UP", entry_idx=0, entry_m5_idx=6, entry_price=150.0,
                 stop0=149.0, initial_risk=1.0, entry_ts=m5.index[6])

    result = simulate_scaled_scheme(h1, atr_h1, entry, trail_mult=3.0, m5_exit=m5)

    assert result["exit_reason"] == "SL_INITIAL_NO_TP"
    assert result["r"] == -1.0
    assert result["exit_time"] == m5.index[8]


def test_m5_mode_normal_case_still_resolves():
    """M5モードでも、SL/TPに触れない通常ケースはMAX_HOLDで正常終了する(回帰確認)。"""
    h1, atr_h1, m5 = _build_bars()
    m5.loc[m5.index[8], "low"] = 150.0  # ブリーチを取り除く
    entry = dict(direction="UP", entry_idx=0, entry_m5_idx=6, entry_price=150.0,
                 stop0=149.0, initial_risk=1.0, entry_ts=m5.index[6])

    result = simulate_scaled_scheme(h1, atr_h1, entry, trail_mult=3.0, m5_exit=m5)

    assert result["exit_reason"] == "MAX_HOLD"
    assert result["exit_time"] >= entry["entry_ts"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
