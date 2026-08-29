"""EXP-FX000005 T-17: 仕様書と実装の突き合わせ回帰テスト.

外部レビューF1(週末クローズ判定)・F3(ATRトレール倍率)はいずれも
「`01-trade-scenario-definition.md`の記述と実装の乖離」が原因だった。
`01-trade-scenario-definition.md`を実装から完全自動生成するのはコストが
大きいため、代わりにドリフト検出用のトリップワイヤとして本テストを新設する。

このテストが書く値は`01-trade-scenario-definition.md`の該当節から転記した
ものであり、**どちらか一方を変更したら必ずもう一方も同じコミットで更新する
こと**(00-spec.md「T-17」節、`00-spec-amendment-NN.md`を参照)。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_vol_breakout_dow_theory import (  # noqa: E402
    MAX_HOLD_BARS, MAX_TREND_HOURS, WINDOW_START_MIN, ZIGZAG_THRESHOLD_ATR_M5,
)
from backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd import (  # noqa: E402
    ATR_TRAIL_MULTIPLIER_M5, BREAKEVEN_TRIGGER_R, COMMISSION_RATE_ROUND_TRIP,
    INITIAL_CAPITAL_USD, MAX_LEVERAGE, N_BREAKOUT, RISK_PCT_PER_TRADE,
    SLIPPAGE_PIPS_MARKET_LEG, SLIPPAGE_PIPS_STOP_TRIGGERED, SPREAD_PIPS,
    STOP_BUFFER_ATR_M5, TP_LEVELS,
)
from price_shock_filter import (  # noqa: E402
    CALM_BARS_REQUIRED, CALM_RATIO, N_BREAKOUT_THRESHOLD,
    SIMULTANEOUS_PAIRS_REQUIRED,
)


def test_detection_layer_matches_spec_section3():
    """§3 検出層(H1)。"""
    assert N_BREAKOUT == 3.5


def test_entry_layer_matches_spec_section4():
    """§4.1 追跡開始・§4.2 スイング検出・§4.7 安全上限。

    ⚠️ 本テストは「定数の値」しか検証していない。2026-08-28、この形式では
    `WINDOW_START_MIN == 30` が正しいまま **「何から30分か」という基準点の
    取り違え**（OBS000009 不具合1、バー確定後のはずがバー始値起点だった）を
    検出できないことが判明した。**意味の検証は
    `tests/test_lookahead_entry_window.py` が担う**（PJ000004 Q16 R3）。

    時刻・期間・基準点に関わるパラメータを追加する場合、本ファイルへの
    定数 assert だけで済ませてはならない。
    """
    assert WINDOW_START_MIN == 30  # ブレイクバー確定後30分(準備期間)。基準点の検証は上記参照
    assert ZIGZAG_THRESHOLD_ATR_M5 == 1.0
    assert MAX_TREND_HOURS == 72  # 追跡上限時間
    assert MAX_HOLD_BARS == 24 * 10  # 240 H1バー相当=2,880 M5バー=10日


def test_price_shock_filter_matches_spec_section4_8():
    """§4.8 価格反応型ショック抑制フィルター。"""
    assert N_BREAKOUT_THRESHOLD == 3.5  # 既存の検出閾値を流用
    assert SIMULTANEOUS_PAIRS_REQUIRED == 2  # 対象通貨中2通貨以上
    assert CALM_RATIO == 2.0  # 再開条件: 最大レンジ/ATR比がこの値未満
    assert CALM_BARS_REQUIRED == 3  # この本数連続で再開


def test_exit_layer_matches_spec_section5():
    """§5.1 ストップロス・§5.2 段階利確・§5.3 トレーリング(T-13後、trailonly版)。"""
    assert STOP_BUFFER_ATR_M5 == 0.703  # 4通貨プールTrain導出値
    assert TP_LEVELS == []  # T-13: 段階利確を全廃
    assert BREAKEVEN_TRIGGER_R == 1.0  # 旧TP1水準(1.0R)を建値移行の唯一のトリガーに転用
    assert ATR_TRAIL_MULTIPLIER_M5 == STOP_BUFFER_ATR_M5 * 1.0  # 1Rの1.0倍


def test_sizing_matches_spec_section6():
    """§6 ポジションサイジング・リスク管理。"""
    assert INITIAL_CAPITAL_USD == 1000.0
    assert RISK_PCT_PER_TRADE == 0.01  # 口座残高の1%(複利)
    assert MAX_LEVERAGE == 25.0  # GMOコイン外国為替FX規制上限


def test_cost_model_matches_spec_section7():
    """§7 コストモデル。"""
    assert SPREAD_PIPS == {
        "USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6, "EUR_USD": 0.3,
    }
    assert SLIPPAGE_PIPS_MARKET_LEG == 0.5  # 週末強制クローズ・MAX_HOLD向け
    assert SLIPPAGE_PIPS_STOP_TRIGGERED == 1.0  # SL・トレーリング向け(T-09)
    assert COMMISSION_RATE_ROUND_TRIP == 0.00004  # 約定金額×0.002%×2
