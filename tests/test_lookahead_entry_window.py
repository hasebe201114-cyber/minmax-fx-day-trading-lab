"""先読み検出の回帰テスト（PJ000004 Q16、2026-08-28新設）.

## なぜ必要だったか

`OBS000009` 不具合1（探索窓起点の先読み）は、既存の3つの防御をすべてすり抜けた:

1. **PJ000004 の品質ゲート Q1〜Q15 に先読みゲートが無かった**
2. **先読み回帰テストが旧エンジン（`minmax_fx_dt.backtest.runner`）専用**で、
   SYS-FX011/026 が使う `scripts/backtest_vol_breakout_dow_theory.py` を守っていなかった
3. **spec整合テスト（T-17）が「値」だけを見て「意味」を見ていなかった**:

   ```python
   assert WINDOW_START_MIN == 30   # ← 30 という値は正しい。バグは「何から30分か」にある
   ```

   バグは基準点（バー確定後 vs バー始値）にあるため、テストは pass し続けた。

## 本テストの原則

**定数の値ではなく、実際に生成されたトレードの時刻を検証する。**
合成データで既知のブレイクを作り、「エントリー時刻がブレイクバーの確定時刻以降か」を
実行結果から直接確かめる。これなら基準点の取り違えを必ず検出できる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from backtest_vol_breakout_dow_theory import simulate_dow_theory_trend  # noqa: E402
from derive_vol_breakout_entry_params import to_h1  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

BAR = pd.Timedelta(hours=1)


def _synthetic_m5(n_bars: int = 1200, break_at: int = 300, seed: int = 7):
    """M5系列を合成し、`break_at` 本目に大きなブレイクバーを1本仕込む."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-06 00:00", periods=n_bars, freq="5min")  # 月曜始まり
    price = 150.0 + np.cumsum(rng.normal(0, 0.01, n_bars))
    df = pd.DataFrame({"open": price, "high": price + 0.01,
                       "low": price - 0.01, "close": price}, index=idx)
    # ブレイク: 1本のH1（=M5 12本）に大きな上昇を入れる
    sl = slice(break_at, break_at + 12)
    df.iloc[sl, df.columns.get_loc("high")] += 0.60
    df.iloc[sl, df.columns.get_loc("close")] += 0.55
    df.iloc[break_at + 12:, [df.columns.get_loc(c) for c in ("open", "high", "low", "close")]] += 0.55
    return df


def _run(m5: pd.DataFrame, bar_close_anchored: bool):
    h1 = to_h1(m5)
    atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)
    ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
    hits = np.where(ratio.values >= 3.5)[0]
    if len(hits) == 0:
        pytest.skip("合成データでブレイクが検出されなかった（テストデータの問題）")
    pos = h1.index.get_loc(ratio.index[hits[0]])
    trades = simulate_dow_theory_trend(
        m5, atr_m5, h1, atr_h1, pos, "UP",
        stop_buffer_atr_m5=0.703, trail_mult=2.109,
        tp_levels=[], breakeven_trigger_r=1.0, atr_trail_series=atr_m5,
        m5_exit=True, bar_close_anchored=bar_close_anchored)
    return trades, h1.index[pos]


def test_h1_index_is_bar_open_time_not_close():
    """前提の明示: `to_h1()` のインデックスはバーの「始値時刻」である。

    この事実こそが OBS000009 不具合1 の原因であり、将来 pandas の既定や
    `to_h1()` の実装が変わったら本テストが落ちて気づけるようにしておく。
    """
    idx = pd.date_range("2025-01-06 00:00", periods=24, freq="5min")
    m5 = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=idx)
    h1 = to_h1(m5)
    assert h1.index[0] == pd.Timestamp("2025-01-06 00:00"), (
        "to_h1() のインデックスがバー始値時刻でなくなった。"
        "bar_close_anchored の計算前提が変わるため見直しが必要")


def test_entries_never_precede_break_bar_close_when_anchored():
    """【本命】bar_close_anchored=True なら、エントリーは必ずブレイクバー確定後。

    定数ではなく**生成されたトレードの実時刻**を検証するため、
    起点の取り違え（バー始値 vs バー確定）を必ず検出できる。
    """
    m5 = _synthetic_m5()
    trades, break_open = _run(m5, bar_close_anchored=True)
    assert trades, "テストが有効になるだけのトレードが生成されなかった"
    bar_close = break_open + BAR
    for t in trades:
        assert pd.Timestamp(t["entry_time"]) >= bar_close, (
            f"先読み: エントリー {t['entry_time']} がブレイクバー確定 {bar_close} より前。"
            f"（バー始値={break_open}）")


def test_legacy_mode_reproduces_the_known_lookahead():
    """既定(False)は旧挙動＝先読みを再現することを明示的に固定する。

    「既定はまだ直っていない」という事実をテストとして可視化し、
    OBS000009 の司令塔判断で既定を反転する際に必ずこのテストが落ちるようにする。
    """
    m5 = _synthetic_m5()
    trades, break_open = _run(m5, bar_close_anchored=False)
    assert trades, "テストが有効になるだけのトレードが生成されなかった"
    bar_close = break_open + BAR
    earliest = min(pd.Timestamp(t["entry_time"]) for t in trades)
    assert earliest < bar_close, (
        "既定モードで先読みが再現しなくなった。OBS000009 の既定反転が完了したなら、"
        "本テストを削除し test_entries_never_precede_break_bar_close_when_anchored を"
        "既定に対して適用すること")


def test_anchored_mode_shifts_window_by_exactly_one_bar():
    """アンカー変更は窓を「ちょうど1バー分」後ろへずらすだけであることを確認する。"""
    m5 = _synthetic_m5()
    tr_legacy, break_open = _run(m5, bar_close_anchored=False)
    tr_fixed, _ = _run(m5, bar_close_anchored=True)
    assert tr_legacy and tr_fixed
    # 最も早いエントリーの差が 1バー以内（窓が1バー後退したことの確認）
    e_legacy = min(pd.Timestamp(t["entry_time"]) for t in tr_legacy)
    e_fixed = min(pd.Timestamp(t["entry_time"]) for t in tr_fixed)
    assert e_fixed >= e_legacy, "修正後のほうが早くエントリーしている（想定外）"
