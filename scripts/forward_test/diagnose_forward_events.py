"""SYS-FX012 フォワードテスト: cutoff 以降のイベントがトレードに至らない理由の内訳.

ledger (`sysfx012_forward_test_ledger.json`) は集計値
(n_events_raw / n_events_dedup / n_events_trendfiltered / n_trades_*) しか
持たないため、「イベントは出ているのにトレードが 0 件」という状態が、

  (a) H1 ダウ理論トレンド判定不能フィルターで落ちた
  (b) 価格反応型ショック抑制フィルター (複数通貨同時ブレイク) で落ちた
  (c) M5 の押し目エントリーが成立しなかった

のどれによるものかを ledger だけからは切り分けられない。本スクリプトは
イベント 1 件ごとに上記の内訳を出力し、`research/method-notes/
sysfx012_forward_event_diagnostic.json` に保存する。

判定ロジックは `run_forward_test_cycle.py` の凍結設計と同一のものを再利用する
(新しいパラメータ・新しい閾値は一切導入しない。純粋な内訳の可視化)。

Usage: PYTHONPATH=src:scripts python3 scripts/forward_test/diagnose_forward_events.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from analyze_n_breakout_h1_dow_trend_alignment import h1_dow_trend_direction  # noqa: E402
from backtest_vol_breakout_dow_theory import (  # noqa: E402
    select_non_overlapping_breakout_events, simulate_dow_theory_trend,
)
from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402
from backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd import (  # noqa: E402
    ATR_TRAIL_MULTIPLIER_M5, BREAKEVEN_TRIGGER_R, STOP_BUFFER_ATR_M5, TP_LEVELS_TRAILONLY,
)
from backtest_vol_continuation_candidates_trendfilter_4pairs_trainonly import (  # noqa: E402
    detect_candidate1,
)
from derive_vol_breakout_entry_params import to_h1  # noqa: E402
from forward_test.run_forward_test_cycle import CUTOFF, load_m5_forward  # noqa: E402
from price_shock_filter import make_price_shock_check  # noqa: E402

from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

OUT_PATH = ROOT / "research" / "method-notes" / "sysfx012_forward_event_diagnostic.json"


def diagnose() -> dict:
    m5_by_pair, h1_by_pair, atr_h1_by_pair, atr_m5_by_pair = {}, {}, {}, {}
    for pair in SELECTED_PAIRS:
        m5 = load_m5_forward(pair)
        m5_by_pair[pair] = m5
        h1 = to_h1(m5)
        h1_by_pair[pair] = h1
        atr_h1_by_pair[pair] = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        atr_m5_by_pair[pair] = atr_ind(m5["high"], m5["low"], m5["close"], length=14)
    shock_check = make_price_shock_check(h1_by_pair, atr_h1_by_pair)

    events = []
    for pair in SELECTED_PAIRS:
        h1, atr_h1 = h1_by_pair[pair], atr_h1_by_pair[pair]
        up, down = detect_candidate1(h1, atr_h1)
        positions, directions = [], []
        for i in range(len(h1)):
            if h1.index[i] < CUTOFF:
                continue
            if bool(up.iloc[i]):
                positions.append(i)
                directions.append("UP")
            elif bool(down.iloc[i]):
                positions.append(i)
                directions.append("DOWN")
        dedup = select_non_overlapping_breakout_events(h1.index, positions, directions)
        dir_map = dict(zip(positions, directions, strict=True))

        for pos in dedup:
            trend = h1_dow_trend_direction(h1, atr_h1, pos)
            record = {
                "pair": pair,
                "h1_time": str(h1.index[pos]),
                "direction": dir_map[pos],
                "h1_dow_trend": trend,
                "shock_blocked": None,
                "n_trades": 0,
                "dropped_by": None,
            }
            if trend is None:
                record["dropped_by"] = "h1_trend_undetermined"
                events.append(record)
                continue

            record["shock_blocked"] = bool(shock_check(h1.index[pos]))
            trades = simulate_dow_theory_trend(
                m5_by_pair[pair], atr_m5_by_pair[pair], h1, atr_h1, pos, dir_map[pos],
                STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER_M5, blackout_check=shock_check,
                tp_levels=TP_LEVELS_TRAILONLY, skip_first_entry=False,
                atr_trail_series=atr_m5_by_pair[pair], m5_exit=True,
                breakeven_trigger_r=BREAKEVEN_TRIGGER_R)
            record["n_trades"] = len(trades)
            if not trades:
                record["dropped_by"] = (
                    "price_shock_filter" if record["shock_blocked"] else "no_m5_entry"
                )
            events.append(record)

    reasons: dict[str, int] = {}
    for e in events:
        key = e["dropped_by"] or "traded"
        reasons[key] = reasons.get(key, 0) + 1

    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "cutoff": str(CUTOFF),
        "latest_bar_by_pair": {p: str(m5_by_pair[p].index.max()) for p in SELECTED_PAIRS},
        "note": (
            "凍結設計のまま、cutoff以降のイベントがトレードに至らなかった理由を1件ずつ"
            "内訳表示したもの。新規パラメータ・閾値の導入は一切なし"
        ),
        "n_events_dedup": len(events),
        "drop_reason_counts": reasons,
        "events": events,
    }


def main() -> int:
    out = diagnose()
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"=== SYS-FX012 フォワードイベント内訳 (cutoff {out['cutoff']}) ===")
    print(f"最新バー: {out['latest_bar_by_pair']}")
    print(f"イベント(dedup後) {out['n_events_dedup']}件 内訳: {out['drop_reason_counts']}")
    for e in out["events"]:
        print(f"  {e['pair']:<8} {e['h1_time']} dir={e['direction']:<4} "
              f"h1_trend={str(e['h1_dow_trend']):<4} shock={str(e['shock_blocked']):<5} "
              f"trades={e['n_trades']} → {e['dropped_by']}")
    print(f"\n[出力]: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
