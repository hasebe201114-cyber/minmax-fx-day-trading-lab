"""T-15当たりづけ: 検出層を「H1ボラティリティ・ブレイク」から「H1トレンド継続
そのもの」に置き換えた場合、Train単独でトレード数・質がどう変わるかを見る
簡易シミュレーション.

事前登録(結果を見る前に固定、探索的診断のためHARKing防止の正式プロトコル
対象外だが同じ規律で進める):
  - 検出トリガー: H1終値が直近`donchian_length`本(現在バー除く)の高値/安値を
    更新したら「トレンド継続」と判定する。`donchian_length=20`は
    `indicators.py`の`donchian()`の既定値をそのまま採用し、本検証用に
    新規調整しない(HARKing回避)。単一バーのボラティリティ極値
    (N_BREAKOUT)は一切使わない。
  - ダウ理論のスイング追跡・エントリー・イグジット・コストモデル・
    重複イベント除去・価格反応型ショック抑制は現行best-candidate
    (T-13 trailonly版)と完全に同一のロジックをそのまま使う。変えるのは
    「追跡を開始する条件」だけ。
  - Train単独(4通貨)のみで評価する(HARKing防止、Validation/Testは
    正式化する場合に初めて参照)。
  - 比較対象: 現行best-candidateのTrain(n=524)を同一の要約関数
    (`summarize()`、r_gross<=-0.05を「真の初期SL相当」とする代理指標)で
    再集計した値。

出力: `research/method-notes/h1_donchian_continuation_trainonly.json`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from backtest_vol_breakout_dow_theory import (  # noqa: E402
    select_non_overlapping_breakout_events, simulate_dow_theory_trend,
)
from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402
from backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd import (  # noqa: E402
    ATR_TRAIL_MULTIPLIER_M5, BREAKEVEN_TRIGGER_R, N_BREAKOUT, PERIODS,
    STOP_BUFFER_ATR_M5, TP_LEVELS_TRAILONLY, load_m5_period, pip_size,
)
from derive_vol_breakout_entry_params import to_h1  # noqa: E402
from price_shock_filter import make_price_shock_check  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402
from minmax_fx_dt.strategy.indicators import donchian  # noqa: E402

DONCHIAN_LENGTH = 20  # indicators.py donchian()の既定値をそのまま採用


def detect_donchian_continuation_events(h1: pd.DataFrame) -> tuple[list[int], list[str]]:
    """先読みなし: バーiの終値を、バーi-1までのDonchian(20)と比較する。"""
    dc = donchian(h1["high"], h1["low"], DONCHIAN_LENGTH, DONCHIAN_LENGTH).shift(1)
    close = h1["close"]
    up = close > dc["DCU"]
    down = close < dc["DCL"]
    positions, directions = [], []
    for i in range(len(h1)):
        if bool(up.iloc[i]):
            positions.append(i)
            directions.append("UP")
        elif bool(down.iloc[i]):
            positions.append(i)
            directions.append("DOWN")
    return positions, directions


def summarize(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0}
    sl_n = sum(1 for t in trades if t["r"] <= -0.05)
    r = np.array([t["r"] for t in trades])
    wins = r[r > 0]
    losses = r[r < 0]
    return {
        "n": n,
        "sl_rate": round(sl_n / n, 4),
        "mean_r_gross": round(float(r.mean()), 4),
        "win_rate": round(float((r > 0).mean()), 4),
        "profit_factor": round(float(wins.sum() / abs(losses.sum())), 3) if len(losses) else None,
        "payoff_ratio": round(float(wins.mean() / abs(losses.mean())), 3) if len(wins) and len(losses) else None,
    }


def run_variant(period_name: str, start: str, end: str, detector: str) -> dict:
    m5_by_pair, h1_by_pair, atr_h1_by_pair = {}, {}, {}
    for pair in SELECTED_PAIRS:
        m5 = load_m5_period(pair, start, end)
        if len(m5) < 1000:
            continue
        m5_by_pair[pair] = m5
        h1_by_pair[pair] = to_h1(m5)
        atr_h1_by_pair[pair] = atr_ind(h1_by_pair[pair]["high"], h1_by_pair[pair]["low"],
                                        h1_by_pair[pair]["close"], length=14)
    shock_check = make_price_shock_check(h1_by_pair, atr_h1_by_pair)

    all_trades: list[dict] = []
    n_events_raw, n_events_dedup = 0, 0
    for pair, m5 in m5_by_pair.items():
        h1 = h1_by_pair[pair]
        atr_h1 = atr_h1_by_pair[pair]
        atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)

        if detector == "donchian":
            positions, directions = detect_donchian_continuation_events(h1)
        else:
            ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
            idxs = np.where(ratio.values >= N_BREAKOUT)[0]
            positions = [h1.index.get_loc(ratio.index[i]) for i in idxs]
            directions = ["UP" if h1.iloc[pos]["close"] > h1.iloc[pos]["open"] else "DOWN" for pos in positions]

        n_events_raw += len(positions)
        dedup_positions = select_non_overlapping_breakout_events(h1.index, positions, directions)
        n_events_dedup += len(dedup_positions)
        dedup_directions = {pos: d for pos, d in zip(positions, directions)}

        for pos in dedup_positions:
            direction = dedup_directions[pos]
            all_trades.extend(simulate_dow_theory_trend(
                m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER_M5,
                blackout_check=shock_check, tp_levels=TP_LEVELS_TRAILONLY, skip_first_entry=False,
                atr_trail_series=atr_m5, m5_exit=True, breakeven_trigger_r=BREAKEVEN_TRIGGER_R))

    stats = summarize(all_trades)
    stats["n_events_raw"] = n_events_raw
    stats["n_events_dedup"] = n_events_dedup
    return stats


def main() -> int:
    start, end = PERIODS["train"]
    baseline = run_variant("train", start, end, detector="n_breakout")
    donchian_variant = run_variant("train", start, end, detector="donchian")

    print(f"[現行best-candidate(N_BREAKOUT={N_BREAKOUT})] {baseline}")
    print(f"[Donchian({DONCHIAN_LENGTH})継続検出] {donchian_variant}")

    result = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "purpose": "T-15当たりづけ: 検出層をH1ボラブレイクからH1トレンド継続(Donchian20ブレイク)に"
                   "置き換えた場合のTrain単独でのトレード数・質の変化を見る簡易シミュレーション",
        "caveat": "正式プロトコル外の探索的診断(HARKing防止プロトコルの対象外だが同じ規律で実施)。"
                  "Train単独のみ、donchian_length=20はindicators.pyの既定値をそのまま採用し調整していない。"
                  "00-spec.md等は変更しない。r_gross(コスト控除前)ベース",
        "detection_layer_current_best_candidate": {"method": f"H1レンジ/ATR比>=N_BREAKOUT({N_BREAKOUT})",
                                                     **baseline},
        "detection_layer_donchian_continuation": {"method": f"H1終値が直近{DONCHIAN_LENGTH}本の高値/安値を更新",
                                                    **donchian_variant},
    }
    out_path = ROOT / "research" / "method-notes" / "h1_donchian_continuation_trainonly.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n書き出し: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
