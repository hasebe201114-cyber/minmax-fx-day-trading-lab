"""EXP-FX000005 改善ループ第7試行: カレンダー固定窓に代わる価格反応型
ショック抑制フィルター.

背景: 機会損失分析(`analyze_missed_entry_opportunities.py`)で、BOJ/FOMC
カレンダーブラックアウトが機会損失87件中81件(93%)の原因であり、これらは
追跡窓内でMFE中央値6.62Rという大きな順行機会を犠牲にしていたと判明した。
一方でカレンダーフィルターは予定外の地政学ショック(2025-06-13イスラエルの
イラン攻撃)を原理的に防げない。司令塔提案「カレンダで固定期間をフィルター
するのではなく、ニュースを検知してエントリー抑制、値動きみて再開判断した
ほうがよいかも」を受け、価格反応型(price-reactive)のショック抑制フィルター
を設計する。

## 改訂履歴

初版(v1)は「いずれかの通貨のH1レンジ/ATR比が7.0(=N_BREAKOUT×2)に達したら
ショックと判定」する単一バー極値トリガーで設計したが、2025-06-13のイスラエル・
イラン攻撃クラスタで検証したところ、当時の最大比率(AUD/JPY 09:00, 4.80)が
7.0に届かず検出できなかったと判明した。連敗クラスタの実測(第3試行の診断分析)
では「複数通貨がほぼ同時にブレイク検知する」こと自体が共通パターンであり、
単一バーの極端さではなく通貨間の同時性が真のショックシグナルだったと考えられる。
これを受けv2として、複数通貨の同時ブレイクを判定基準にするトリガーへ改訂した。

事前登録(v2、結果を見る前に固定):
- 抑制トリガー: 同一H1バー時刻で、対象4通貨中SIMULTANEOUS_PAIRS_REQUIRED
  (=2)通貨以上が既存の検出閾値N_BREAKOUT(=3.5)以上のレンジ/ATR比を記録
  したら「相関ショック」と判定する(新規の恣意的な閾値を導入せず、既存の
  検出閾値をそのまま流用する)。
- 抑制範囲: 相関ショック確定時点で、対象4通貨すべての新規エントリーを
  抑制する。ダウ理論のスイング構造追跡自体は継続する(カレンダーフィルター
  と同じ扱い)。
- 再開条件: 4通貨中の最大レンジ/ATR比がCALM_RATIO(=2.0)未満の状態が
  CALM_BARS_REQUIRED(=3)本のH1バー連続したら抑制解除する(v1から変更なし)。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

N_BREAKOUT_THRESHOLD = 3.5  # 既存の検出閾値を流用(新規の恣意的閾値を導入しない)
SIMULTANEOUS_PAIRS_REQUIRED = 2
CALM_RATIO = 2.0
CALM_BARS_REQUIRED = 3


def build_shock_suppression_series(h1_by_pair: dict[str, pd.DataFrame],
                                    atr_h1_by_pair: dict[str, pd.Series]) -> pd.Series:
    """4通貨のH1データから、各H1タイムスタンプでの抑制状態(bool)を返す。
    タイムスタンプは全通貨のH1インデックスの和集合。トリガーはv2(複数通貨
    同時ブレイク)、再開条件はv1から変更なし(最大比率の沈静化)。"""
    all_ts = sorted(set().union(*[set(h1.index) for h1 in h1_by_pair.values()]))
    max_ratio: dict[pd.Timestamp, float] = {}
    n_breaking: dict[pd.Timestamp, int] = {}
    for ts in all_ts:
        ratios = []
        for pair, h1 in h1_by_pair.items():
            if ts in h1.index:
                atr_h1 = atr_h1_by_pair[pair]
                a = atr_h1.get(ts)
                if pd.notna(a) and a > 0:
                    bar = h1.loc[ts]
                    ratios.append((bar["high"] - bar["low"]) / a)
        max_ratio[ts] = max(ratios) if ratios else 0.0
        n_breaking[ts] = sum(1 for r in ratios if r >= N_BREAKOUT_THRESHOLD)

    suppressed = {}
    is_suppressed = False
    calm_streak = 0
    for ts in all_ts:
        r = max_ratio[ts]
        if n_breaking[ts] >= SIMULTANEOUS_PAIRS_REQUIRED:
            is_suppressed = True
            calm_streak = 0
        elif is_suppressed:
            if r < CALM_RATIO:
                calm_streak += 1
                if calm_streak >= CALM_BARS_REQUIRED:
                    is_suppressed = False
                    calm_streak = 0
            else:
                calm_streak = 0
        suppressed[ts] = is_suppressed

    return pd.Series(suppressed).sort_index()


def make_price_shock_check(h1_by_pair: dict[str, pd.DataFrame], atr_h1_by_pair: dict[str, pd.Series]):
    """simulate_dow_theory_trendのblackout_check互換コールバックを返す。"""
    suppression = build_shock_suppression_series(h1_by_pair, atr_h1_by_pair)
    idx = suppression.index

    def check(ts: pd.Timestamp) -> bool:
        pos = idx.searchsorted(ts, side="right") - 1
        if pos < 0:
            return False
        return bool(suppression.iloc[pos])

    return check
