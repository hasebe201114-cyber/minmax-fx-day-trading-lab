"""SYS-FX012 週次レポート Step 2/3 集計 (2026-09-07 週).

cutoff 2026-08-15 06:00 JST 〜 latest 2026-09-05 05:55 JST の値動き集計と
検出イベントの詳細、M5 エントリー判定。

出力:
  - logs/forward_test_cycle/weekly_summary_v2_20260907.json
  - logs/forward_test_cycle/weekly_events_v4_20260907.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\Atsushi Hasebe\.minimax-agent\projects\minmax-fx-day-trading-lab")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Project modules
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

from derive_vol_breakout_entry_params import N_BREAKOUT, to_h1  # noqa: E402
from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402
from backtest_vol_continuation_candidates_trendfilter_4pairs_trainonly import (  # noqa: E402
    detect_candidate1,
)
from analyze_n_breakout_h1_dow_trend_alignment import h1_dow_trend_direction  # noqa: E402
from backtest_vol_breakout_dow_theory import (  # noqa: E402
    select_non_overlapping_breakout_events,
)

CUTOFF = pd.Timestamp("2026-08-15 06:00:00")
LATEST = pd.Timestamp("2026-09-05 05:55:00")
REPORT_DATE = "2026-09-07"

DS1_JSON = PROJECT_ROOT / "data" / "curated" / "ds-1.json"
DS1_FORWARD_JSON = PROJECT_ROOT / "data" / "curated" / "ds-1-forward.json"
RAW_FORWARD_DIR = PROJECT_ROOT / "data" / "raw" / "ds-1-forward"
LEDGER_JSON = PROJECT_ROOT / "research" / "method-notes" / "sysfx012_forward_test_ledger.json"
OUT_DIR = PROJECT_ROOT / "logs" / "forward_test_cycle"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_m5(pair: str) -> pd.DataFrame:
    frames = []
    if DS1_JSON.exists():
        with DS1_JSON.open(encoding="utf-8") as f:
            ds1 = json.load(f)
        if pair in ds1.get("pairs", {}):
            df = pd.DataFrame(ds1["pairs"][pair]["data"])
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            frames.append(df.set_index("timestamp"))
    for f in sorted(RAW_FORWARD_DIR.glob(f"ohlcv_{pair}_5min_*.csv")):
        df = pd.read_csv(f, parse_dates=["timestamp"])
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        frames.append(df.set_index("timestamp"))
    if DS1_FORWARD_JSON.exists():
        with DS1_FORWARD_JSON.open(encoding="utf-8") as f:
            dsf = json.load(f)
        if pair in dsf.get("pairs", {}):
            df = pd.DataFrame(dsf["pairs"][pair]["data"])
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            frames.append(df.set_index("timestamp"))
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def to_h1_simple(m5: pd.DataFrame) -> pd.DataFrame:
    return m5.resample("1h", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()


def zigzag_pivots_simple(h1: pd.DataFrame, atr_h1: pd.Series, threshold_atr: float = 2.0) -> pd.Series:
    """簡易 ZigZag ピボット判定. threshold_atr * ATR 以上の swing をトレンド判定の補強に使用."""
    pivots = pd.Series(index=h1.index, dtype=object)
    if len(h1) < 3 or atr_h1.isna().all():
        return pivots
    last_pivot_idx = 0
    last_pivot_type = None  # "H" or "L"
    for i in range(1, len(h1)):
        hi = h1["high"].iloc[i]
        lo = h1["low"].iloc[i]
        if pd.isna(atr_h1.iloc[i]) or atr_h1.iloc[i] == 0:
            continue
        thr = threshold_atr * atr_h1.iloc[i]
        if last_pivot_type is None:
            if hi - h1["low"].iloc[last_pivot_idx] >= thr:
                pivots.iloc[last_pivot_idx] = "L"
                last_pivot_type = "H"
            elif h1["high"].iloc[last_pivot_idx] - lo >= thr:
                pivots.iloc[last_pivot_idx] = "H"
                last_pivot_type = "L"
            continue
        if last_pivot_type == "H":
            if hi > h1["high"].iloc[last_pivot_idx]:
                pivots.iloc[last_pivot_idx] = None
                last_pivot_idx = i
                last_pivot_type = "H"
            elif h1["high"].iloc[last_pivot_idx] - lo >= thr:
                pivots.iloc[last_pivot_idx] = "H"
                last_pivot_idx = i
                last_pivot_type = "L"
        elif last_pivot_type == "L":
            if lo < h1["low"].iloc[last_pivot_idx]:
                pivots.iloc[last_pivot_idx] = None
                last_pivot_idx = i
                last_pivot_type = "L"
            elif hi - h1["low"].iloc[last_pivot_idx] >= thr:
                pivots.iloc[last_pivot_idx] = "L"
                last_pivot_idx = i
                last_pivot_type = "H"
    if last_pivot_type is not None and last_pivot_idx < len(h1):
        pivots.iloc[last_pivot_idx] = last_pivot_type
    return pivots


def main():
    print("=" * 78)
    print(f"SYS-FX012 週次サマリ ({REPORT_DATE})")
    print(f"cutoff={CUTOFF}, latest={LATEST}, period={(LATEST - CUTOFF).total_seconds()/86400:.2f} days")
    print("=" * 78)

    # ----- 1. 値動きサマリ -----
    summary = {"generated_at": datetime.now().isoformat(),
               "cutoff": str(CUTOFF), "latest": str(LATEST), "pairs": {}}
    for pair in SELECTED_PAIRS:
        m5 = load_m5(pair)
        h1 = to_h1_simple(m5)
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)
        post = m5[m5.index >= CUTOFF]
        post_h1 = h1[(h1.index >= CUTOFF) & (h1.index <= LATEST + pd.Timedelta(hours=1))]
        if len(post) == 0 or len(post_h1) == 0:
            print(f"  [WARN] {pair}: no data in window")
            continue
        first_close = post.iloc[0]["close"]
        last_close = post.iloc[-1]["close"]
        change = last_close - first_close
        pct = change / first_close * 100
        pip = 0.01
        pips_change = change / pip
        period_high = post_h1["high"].max()
        period_low = post_h1["low"].min()
        atr_h1_then = atr_h1[atr_h1.index >= CUTOFF].iloc[0]
        atr_h1_now = atr_h1.iloc[-1]
        atr_h1_change_pct = (atr_h1_now - atr_h1_then) / atr_h1_then * 100
        # M5 ローソク分布
        n_m5 = len(post)
        n_up = int((post["close"] > post["open"]).sum())
        n_down = int((post["close"] < post["open"]).sum())
        n_doji = int((post["close"] == post["open"]).sum())
        # M5 ATR
        atr_m5_first = atr_m5[atr_m5.index >= CUTOFF].iloc[0] if len(atr_m5[atr_m5.index >= CUTOFF]) > 0 else None
        atr_m5_last = atr_m5.iloc[-1] if len(atr_m5) > 0 else None
        atr_m5_change_pct = ((atr_m5_last - atr_m5_first) / atr_m5_first * 100) if (atr_m5_first and atr_m5_last) else 0.0
        pair_summary = {
            "first_close": float(first_close),
            "last_close": float(last_close),
            "change": float(change),
            "pct_change": float(pct),
            "pips_change": float(pips_change),
            "period_high": float(period_high),
            "period_low": float(period_low),
            "atr_h1_then": float(atr_h1_then),
            "atr_h1_now": float(atr_h1_now),
            "atr_h1_change_pct": float(atr_h1_change_pct),
            "m5_n_bars": n_m5,
            "m5_up": n_up,
            "m5_down": n_down,
            "m5_doji": n_doji,
            "m5_up_pct": n_up / n_m5 * 100,
            "m5_down_pct": n_down / n_m5 * 100,
            "m5_doji_pct": n_doji / n_m5 * 100,
            "atr_m5_then": float(atr_m5_first) if atr_m5_first else None,
            "atr_m5_now": float(atr_m5_last) if atr_m5_last else None,
            "atr_m5_change_pct": float(atr_m5_change_pct),
        }
        summary["pairs"][pair] = pair_summary
        print(f"  {pair}: {first_close:.4f} → {last_close:.4f} ({change:+.4f}, {pct:+.2f}%, {pips_change:+.1f} pips)")
        print(f"    range: {period_low:.4f} - {period_high:.4f}  ATR(H1) {atr_h1_then:.4f}→{atr_h1_now:.4f} ({atr_h1_change_pct:+.1f}%)")
        print(f"    M5: {n_m5}本 UP={n_up} DOWN={n_down} DOJI={n_doji}  ATR(M5) {atr_m5_first:.5f}→{atr_m5_last:.5f}")

    # ----- 2. 検出イベント (raw / dedup / trend-pass) -----
    print("\n--- 検出イベント (raw / dedup / trend-pass) ---")
    events_data = {"generated_at": datetime.now().isoformat(),
                   "cutoff": str(CUTOFF), "latest": str(LATEST),
                   "n_breakout": N_BREAKOUT, "events": []}
    n_raw_total = n_dedup_total = n_trend_total = 0
    for pair in SELECTED_PAIRS:
        m5 = load_m5(pair)
        # Use project's to_h1 (not the simple one) to match cycle
        h1 = to_h1(m5)
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)
        up_raw, down_raw = detect_candidate1(h1, atr_h1)
        # raw 検出 (cutoff 以降)
        raw_pos, raw_dir = [], []
        for i in range(len(h1)):
            if h1.index[i] < CUTOFF:
                continue
            if bool(up_raw.iloc[i]):
                raw_pos.append(i); raw_dir.append("UP")
            elif bool(down_raw.iloc[i]):
                raw_pos.append(i); raw_dir.append("DOWN")
        # dedup
        dedup_pos = select_non_overlapping_breakout_events(h1.index, raw_pos, raw_dir)
        dedup_directions = {p: d for p, d in zip(raw_pos, raw_dir)}
        n_raw = len(raw_pos); n_dedup = len(dedup_pos)
        n_trend = 0
        # イベント詳細
        for pos in raw_pos:
            ts = h1.index[pos]
            h1_row = h1.iloc[pos]
            range_v = h1_row["high"] - h1_row["low"]
            atr_v = atr_h1.iloc[pos]
            range_atr = range_v / atr_v if atr_v > 0 else 0.0
            direction = dedup_directions[pos]
            trend = h1_dow_trend_direction(h1, atr_h1, pos)
            in_dedup = pos in dedup_pos
            m5_first5 = m5[(m5.index > ts) & (m5.index <= ts + pd.Timedelta(minutes=30))]
            if len(m5_first5) >= 2:
                if direction == "UP":
                    higher_low = all(m5_first5["low"].iloc[i] >= m5_first5["low"].iloc[i-1]
                                     for i in range(1, len(m5_first5)))
                else:
                    higher_low = all(m5_first5["high"].iloc[i] <= m5_first5["high"].iloc[i-1]
                                     for i in range(1, len(m5_first5)))
            else:
                higher_low = None
            ev = {
                "pair": pair, "time": str(ts), "direction": direction,
                "h1_open": float(h1_row["open"]), "h1_high": float(h1_row["high"]),
                "h1_low": float(h1_row["low"]), "h1_close": float(h1_row["close"]),
                "range": float(range_v), "atr_h1": float(atr_v),
                "range_atr": float(range_atr), "h1_trend": trend,
                "in_dedup": bool(in_dedup),
                "m5_higher_low_or_lower_high": higher_low,
            }
            events_data["events"].append(ev)
            if in_dedup and trend is not None:
                n_trend += 1
        n_raw_total += n_raw
        n_dedup_total += n_dedup
        n_trend_total += n_trend
        print(f"  {pair}: raw={n_raw}, dedup={n_dedup}, trend-pass={n_trend}")
    events_data["n_raw_total"] = n_raw_total
    events_data["n_dedup_total"] = n_dedup_total
    events_data["n_trend_total"] = n_trend_total
    print(f"  TOTAL: raw={n_raw_total}, dedup={n_dedup_total}, trend-pass={n_trend_total}")

    # ----- 3. トレード集計 (ledger から) -----
    print("\n--- トレード集計 (ledger) ---")
    with LEDGER_JSON.open(encoding="utf-8") as f:
        ledger = json.load(f)
    bt = ledger["backtest"]
    trades = bt.get("trades", [])
    n_total = bt["n_trades_total"]
    n_closed = bt["n_trades_closed"]
    n_open = bt["n_trades_open"]
    win_rate = bt.get("win_rate")
    mean_r_net = bt.get("mean_r_net")
    pf = bt.get("profit_factor")
    payoff = bt.get("payoff_ratio")
    final_balance = bt["final_balance"]
    perm_p = bt.get("perm_p_block")
    print(f"  n_trades_total={n_total} (closed={n_closed}, open={n_open})")
    print(f"  win_rate={win_rate}  mean_r_net={mean_r_net}  PF={pf}  payoff={payoff}")
    print(f"  perm_p_block={perm_p}  final_balance=${final_balance}")
    # 通貨別集計
    pair_stats = {}
    for pair in SELECTED_PAIRS:
        p_trades = [t for t in trades if t["pair"] == pair]
        p_wins = [t for t in p_trades if t["r_net"] > 0]
        p_loss = [t for t in p_trades if t["r_net"] <= 0]
        if p_trades:
            pair_stats[pair] = {
                "n_trades": len(p_trades),
                "wins": len(p_wins),
                "losses": len(p_loss),
                "win_rate": len(p_wins) / len(p_trades),
                "sum_r_net": sum(t["r_net"] for t in p_trades),
                "mean_r_net": sum(t["r_net"] for t in p_trades) / len(p_trades),
                "sum_dollar_pnl": sum(t.get("dollar_pnl", 0) for t in p_trades),
            }
            ps = pair_stats[pair]
            print(f"  {pair}: n={ps['n_trades']} W={ps['wins']} L={ps['losses']} WR={ps['win_rate']*100:.1f}% sum_r={ps['sum_r_net']:+.2f} $={ps['sum_dollar_pnl']:+.2f}")

    # ----- 4. 出力 -----
    out_summary = OUT_DIR / f"weekly_summary_v2_{REPORT_DATE}.json"
    out_summary.write_text(json.dumps({
        **summary,
        "n_raw_total": n_raw_total, "n_dedup_total": n_dedup_total, "n_trend_total": n_trend_total,
        "n_trades": n_total, "win_rate": win_rate, "mean_r_net": mean_r_net,
        "profit_factor": pf, "payoff_ratio": payoff, "perm_p_block": perm_p,
        "final_balance": final_balance, "pair_stats": pair_stats,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] summary: {out_summary}")

    out_events = OUT_DIR / f"weekly_events_v4_{REPORT_DATE}.json"
    out_events.write_text(json.dumps(events_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] events:  {out_events}")


if __name__ == "__main__":
    main()
