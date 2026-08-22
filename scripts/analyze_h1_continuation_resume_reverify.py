"""T-15検討: 「H1継続確認による再開」エントリーの優位性を現行best-candidate
(dedup修正+T-01〜T-03+T-13 trailonly)データで再検証する.

2026-08-20時点の`entry_exit_diagnostics.json`(再開エントリーSL率36.1% vs
非再開42.6%)は、重複トレード生成バグ修正(T-12診断中に発見)・週末クローズ
修正(T-01)・ATRトレール再導出(T-02)・決済のM5化(T-03)・出口設計のトレール
専業化(T-13)のいずれよりも前の、改善ループ第3試行時点のデータに基づく。
T-15(検出層をH1のボラティリティ・ブレイクから「H1継続確認」そのものへ
移す設計)の起票要否を判断する前提として、この統計が現行データでも
成立するかを確認する。

正式プロトコル外の探索的診断(HARKing防止の事前登録対象外)。
00-spec.md等は変更しない。
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
    ATR_TRAIL_MULTIPLIER_M5, BREAKEVEN_TRIGGER_R, PERIODS, STOP_BUFFER_ATR_M5,
    TP_LEVELS_TRAILONLY, load_m5_period, pip_size,
)
from derive_vol_breakout_entry_params import N_BREAKOUT, to_h1  # noqa: E402
from price_shock_filter import make_price_shock_check  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402


def find_trades_with_diagnostics(pair: str, m5: pd.DataFrame, shock_check) -> list[dict]:
    h1 = to_h1(m5)
    atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)

    ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
    idxs = np.where(ratio.values >= N_BREAKOUT)[0]

    positions = [h1.index.get_loc(ratio.index[i]) for i in idxs]
    directions = ["UP" if h1.iloc[pos]["close"] > h1.iloc[pos]["open"] else "DOWN" for pos in positions]
    dedup_positions = select_non_overlapping_breakout_events(h1.index, positions, directions)
    dedup_directions = {pos: d for pos, d in zip(positions, directions)}

    trades = []
    for pos in dedup_positions:
        direction = dedup_directions[pos]
        trades.extend(simulate_dow_theory_trend(
            m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER_M5,
            blackout_check=shock_check, tp_levels=TP_LEVELS_TRAILONLY, skip_first_entry=False,
            atr_trail_series=atr_m5, m5_exit=True, breakeven_trigger_r=BREAKEVEN_TRIGGER_R))
    for t in trades:
        t["pair"] = pair
    return trades


def summarize(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0}
    # T-13でtp_levels=[]となったため、n_levels_hitは常に0でexit_reasonは
    # ほぼ全件"SL_INITIAL_NO_TP"になり、ラベルではSL到達を判別できない
    # (C1所見と同じ問題)。r_gross<=-0.05(実質的にほぼ初期ストップ幅そのもの)を
    # 「真の初期SL相当」の代理指標として使う。
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


def main() -> int:
    per_period: dict[str, dict] = {}
    pooled_resumed: list[dict] = []
    pooled_not_resumed: list[dict] = []

    for period_name, (start, end) in PERIODS.items():
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
        for pair, m5 in m5_by_pair.items():
            all_trades.extend(find_trades_with_diagnostics(pair, m5, shock_check))

        resumed = [t for t in all_trades if t["resumed_since_last_entry"]]
        not_resumed = [t for t in all_trades if not t["resumed_since_last_entry"]]
        pooled_resumed.extend(resumed)
        pooled_not_resumed.extend(not_resumed)

        # entry_seq=1(トレンドイベント内の初回エントリー)は定義上resumed=Falseにしか
        # なりえない。「resumedの優位性」がentry_seq効果の言い換えに過ぎないかを見るため、
        # entry_seq>=2に限定した上でresumed/not_resumedを再比較する。
        seq2plus = [t for t in all_trades if t["entry_seq"] >= 2]
        resumed_seq2plus = [t for t in seq2plus if t["resumed_since_last_entry"]]
        not_resumed_seq2plus = [t for t in seq2plus if not t["resumed_since_last_entry"]]

        per_period[period_name] = {
            "n_total": len(all_trades),
            "resumed": summarize(resumed),
            "not_resumed": summarize(not_resumed),
            "entry_seq1_only": summarize([t for t in all_trades if t["entry_seq"] == 1]),
            "entry_seq2plus_resumed": summarize(resumed_seq2plus),
            "entry_seq2plus_not_resumed": summarize(not_resumed_seq2plus),
        }
        print(f"{period_name}: n_total={len(all_trades)}  resumed(n={len(resumed)})={summarize(resumed)}  "
              f"not_resumed(n={len(not_resumed)})={summarize(not_resumed)}")
        print(f"  [entry_seq>=2限定] resumed(n={len(resumed_seq2plus)})={summarize(resumed_seq2plus)}  "
              f"not_resumed(n={len(not_resumed_seq2plus)})={summarize(not_resumed_seq2plus)}")

    pooled_seq2plus = [t for t in (pooled_resumed + pooled_not_resumed) if t["entry_seq"] >= 2]
    pooled_resumed_seq2plus = [t for t in pooled_seq2plus if t["resumed_since_last_entry"]]
    pooled_not_resumed_seq2plus = [t for t in pooled_seq2plus if not t["resumed_since_last_entry"]]

    result = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "purpose": ("T-15検討の前提確認: 2026-08-20時点entry_exit_diagnostics.json(改善ループ第3試行"
                    "データ、resumed SL率36.1% n=371 vs not_resumed 42.6% n=319)が、"
                    "現行best-candidate(dedup+T-01〜T-03+T-13 trailonly)データでも成立するかを再検証。"
                    "あわせて『resumedの優位性はentry_seq(初回か否か)の言い換えに過ぎないか』を、"
                    "entry_seq>=2に限定した比較で切り分ける"),
        "caveat": ("正式プロトコル外の探索的診断。00-spec.md等は変更しない。r_gross(コスト控除前)ベース、"
                   "決済判定はM5・出口はトレール専業(T-13)のため、exit_reasonラベルでのSL判別は不能"
                   "(n_levels_hit常に0)。代わりにr_gross<=-0.05を『真の初期SL相当』の代理指標として使用"),
        "old_reference_2026_08_20": {
            "resumed": {"n": 371, "sl_rate": 0.3612},
            "not_resumed": {"n": 319, "sl_rate": 0.4263},
            "source": "research/method-notes/entry_exit_diagnostics.json",
        },
        "per_period": per_period,
        "pooled": {
            "resumed": summarize(pooled_resumed),
            "not_resumed": summarize(pooled_not_resumed),
            "entry_seq2plus_resumed": summarize(pooled_resumed_seq2plus),
            "entry_seq2plus_not_resumed": summarize(pooled_not_resumed_seq2plus),
        },
    }

    out_path = ROOT / "research" / "method-notes" / "h1_continuation_resume_reverify.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n書き出し: {out_path}")
    print(f"\nプール全体: resumed={summarize(pooled_resumed)}")
    print(f"プール全体: not_resumed={summarize(pooled_not_resumed)}")
    print(f"\n[entry_seq>=2限定・プール] resumed={summarize(pooled_resumed_seq2plus)}")
    print(f"[entry_seq>=2限定・プール] not_resumed={summarize(pooled_not_resumed_seq2plus)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
