"""SYS-FX011 (EXP-FX000005) 重複トレード生成バグの回帰テスト.

`find_trades_for_period()`が同一方向の連続H1ブレイクバーごとに独立した追跡
チェーンを開始し、同一M5構造ピボットに収束した際に完全に同一のトレードが
複数回生成される問題(2026-08-21、証拠金維持率ストレステスト(T-12)の実施中に
発見)への回帰テスト。実測で2026-07-29〜31にUSD/JPYで5本のH1ブレイクバーが
連続発生し、同一トレードが5回生成されていた(改善ループ第7試行の時点で
既に存在していたバグ)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_vol_breakout_dow_theory import select_non_overlapping_breakout_events  # noqa: E402


def _h1_index(n: int = 200) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01 00:00", periods=n, freq="h")


def test_consecutive_same_direction_events_are_deduplicated():
    """実測ケースの再現: 同一方向のブレイクバーが短時間で連続発生した場合、
    最初の1件だけが残る(追跡窓72h以内はすべて重複として間引かれる)。"""
    idx = _h1_index()
    positions = [10, 12, 15, 20]  # 全て10時間以内(72h窓に収まる)
    directions = ["UP", "UP", "UP", "UP"]

    selected = select_non_overlapping_breakout_events(idx, positions, directions, max_trend_hours=72)

    assert selected == [10]


def test_events_beyond_tracking_window_are_kept():
    """追跡窓(72h=3H1バー×24)を超えて発生した同一方向イベントは、独立した
    新規チェーンとして許可される(真に別のトレンドイベント)。"""
    idx = _h1_index()
    positions = [10, 10 + 73, 10 + 73 + 73]  # 各73時間後(72h窓の外)
    directions = ["UP", "UP", "UP"]

    selected = select_non_overlapping_breakout_events(idx, positions, directions, max_trend_hours=72)

    assert selected == positions


def test_opposite_direction_within_window_is_kept_as_reversal():
    """方向が異なるイベントは追跡窓内でも新規チェーンとして許可される
    (真の反転の可能性を排除しない)。"""
    idx = _h1_index()
    positions = [10, 15, 20]
    directions = ["UP", "DOWN", "UP"]

    selected = select_non_overlapping_breakout_events(idx, positions, directions, max_trend_hours=72)

    # UP(10)は残る。DOWN(15)は方向が異なるため残る。UP(20)はUP(10)の窓内(72h)
    # かつ同方向のため重複として間引かれる。
    assert selected == [10, 15]


def test_empty_input_returns_empty():
    idx = _h1_index()
    assert select_non_overlapping_breakout_events(idx, [], []) == []


def test_mismatched_lengths_raise_value_error():
    idx = _h1_index()
    with pytest.raises(ValueError):
        select_non_overlapping_breakout_events(idx, [1, 2], ["UP"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
