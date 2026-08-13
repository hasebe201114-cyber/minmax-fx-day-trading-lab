"""バックテスト実行 + trade history 詳細分析.

USD/JPY 2024年データで実行し、以下を出力:
- エントリー条件の成立パターン別集計
- エントリー方向別 (BUY/SELL) 集計
- 決済条件 (TP/SL/REVERSE/WEEKEND/END) 別集計
- 勝ち負け・平均損益詳細
- 連続勝ち/負けの最大値
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from minmax_fx_dt.backtest import run_backtest
from minmax_fx_dt.backtest.simulator import SimulatorConfig, Trade
from minmax_fx_dt.strategy.multi_timeframe import MTFConfig


def load_ohlcv_from_ds1(symbol: str) -> pd.DataFrame:
    """DS-1 JSON から指定通貨の OHLCV を読み込み."""
    ds1_path = ROOT / "data" / "curated" / "ds-1.json"
    with ds1_path.open(encoding="utf-8") as f:
        ds1 = json.load(f)
    if symbol not in ds1["pairs"]:
        raise ValueError(f"DS-1 に {symbol} がありません: {list(ds1['pairs'].keys())}")
    records = ds1["pairs"][symbol]["data"]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def aggregate_to_mtf(m5_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """M5 足から H4 / D1 を集約."""
    h4_df = pd.DataFrame({
        "open": m5_df["open"].resample("4h").first(),
        "high": m5_df["high"].resample("4h").max(),
        "low": m5_df["low"].resample("4h").min(),
        "close": m5_df["close"].resample("4h").last(),
    }).dropna()
    d1_df = pd.DataFrame({
        "open": h4_df["open"].resample("D").first(),
        "high": h4_df["high"].resample("D").max(),
        "low": h4_df["low"].resample("D").min(),
        "close": h4_df["close"].resample("D").last(),
    }).dropna()
    m15_df = pd.DataFrame({
        "open": m5_df["open"].resample("15min").first(),
        "high": m5_df["high"].resample("15min").max(),
        "low": m5_df["low"].resample("15min").min(),
        "close": m5_df["close"].resample("15min").last(),
    }).dropna()
    return d1_df, h4_df, m15_df


def analyze_trades(trades: list[Trade]) -> dict:
    """trade 履歴の詳細分析."""
    if not trades:
        return {"n_trades": 0}

    # 決済条件別
    by_exit_reason: dict[str, dict] = {}
    for t in trades:
        key = t.exit_reason
        if key not in by_exit_reason:
            by_exit_reason[key] = {"count": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "total_pips": 0.0}
        d = by_exit_reason[key]
        d["count"] += 1
        d["total_pnl"] += t.pnl
        d["total_pips"] += t.pnl_pips
        if t.pnl > 0:
            d["wins"] += 1
        else:
            d["losses"] += 1

    # エントリー方向別
    by_side: dict[str, dict] = {}
    for t in trades:
        key = t.side
        if key not in by_side:
            by_side[key] = {"count": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "total_pips": 0.0}
        d = by_side[key]
        d["count"] += 1
        d["total_pnl"] += t.pnl
        d["total_pips"] += t.pnl_pips
        if t.pnl > 0:
            d["wins"] += 1
        else:
            d["losses"] += 1

    # エントリー条件の成立パターン別
    by_cond_pattern: dict[str, dict] = {}
    for t in trades:
        if t.entry_conditions is None:
            pattern = "N/A"
        else:
            pattern = "".join(["1" if v else "0" for k, v in sorted(t.entry_conditions.items())])
        if pattern not in by_cond_pattern:
            by_cond_pattern[pattern] = {"count": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}
        d = by_cond_pattern[pattern]
        d["count"] += 1
        d["total_pnl"] += t.pnl
        if t.pnl > 0:
            d["wins"] += 1
        else:
            d["losses"] += 1

    # LT 方向別
    by_lt: dict[str, dict] = {}
    for t in trades:
        key = t.lt_direction or "N/A"
        if key not in by_lt:
            by_lt[key] = {"count": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}
        d = by_lt[key]
        d["count"] += 1
        d["total_pnl"] += t.pnl
        if t.pnl > 0:
            d["wins"] += 1
        else:
            d["losses"] += 1

    # 個別条件の pass 率
    cond_pass_rate: dict[str, dict] = {}
    for t in trades:
        if t.entry_conditions is None:
            continue
        for k, v in t.entry_conditions.items():
            if k not in cond_pass_rate:
                cond_pass_rate[k] = {"pass": 0, "fail": 0, "wins_when_pass": 0, "losses_when_pass": 0}
            if v:
                cond_pass_rate[k]["pass"] += 1
                if t.pnl > 0:
                    cond_pass_rate[k]["wins_when_pass"] += 1
                else:
                    cond_pass_rate[k]["losses_when_pass"] += 1
            else:
                cond_pass_rate[k]["fail"] += 1

    # 連続勝ち / 連続負け
    max_win = 0
    max_loss = 0
    cur_w = 0
    cur_l = 0
    for t in trades:
        if t.pnl > 0:
            cur_w += 1
            cur_l = 0
            max_win = max(max_win, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_loss = max(max_loss, cur_l)

    # 月次別
    monthly: dict[str, dict] = {}
    for t in trades:
        if t.exit_time is None:
            continue
        key = t.exit_time.strftime("%Y-%m")
        if key not in monthly:
            monthly[key] = {"count": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}
        d = monthly[key]
        d["count"] += 1
        d["total_pnl"] += t.pnl
        if t.pnl > 0:
            d["wins"] += 1
        else:
            d["losses"] += 1

    return {
        "n_trades": len(trades),
        "by_exit_reason": by_exit_reason,
        "by_side": by_side,
        "by_cond_pattern": by_cond_pattern,
        "by_lt_direction": by_lt,
        "by_condition_pass_rate": cond_pass_rate,
        "max_consecutive_wins": max_win,
        "max_consecutive_losses": max_loss,
        "by_month": monthly,
        "total_pnl": sum(t.pnl for t in trades),
        "total_pips": sum(t.pnl_pips for t in trades),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="トレード詳細分析")
    parser.add_argument("--pair", default="USD_JPY", help="通貨ペア")
    parser.add_argument("--save-trades", action="store_true", help="trade history を JSON 保存")
    args = parser.parse_args()

    sim_config = SimulatorConfig(
        initial_cash_jpy=1_000_000.0,
        lot_size=1_000,
        spread_pips=0.3,
        slippage_pips=0.5,
        is_jpy_pair=True,
        weekend_close=True,
        max_dd_pause_threshold_pct=50.0,
    )
    mtf_config = MTFConfig(
        lt_sma_short=20,
        lt_sma_long=50,
        lt_adx_threshold=20.0,
        mt_donchian_length=20,
        mt_atr_length=14,
        sr_min_touches=3,
        sr_cluster_threshold_pct=0.5,
        sr_fractal_window=5,
    )

    print(f"=== {args.pair} 2024 バックテスト実行 + 詳細分析 ===\n")
    m5_df = load_ohlcv_from_ds1(args.pair)
    print(f"M5 データ: {len(m5_df)} bars")
    lt_df, mt_df, st_df = aggregate_to_mtf(m5_df)

    t0 = time.time()
    result = run_backtest(
        lt_ohlcv=lt_df,
        mt_ohlcv=mt_df,
        st_ohlcv=st_df,
        pair=args.pair,
        sim_config=sim_config,
        mtf_config=mtf_config,
    )
    elapsed = time.time() - t0
    print(f"実行時間: {elapsed:.2f}秒\n")

    # trade history 詳細分析
    analysis = analyze_trades(result.state.trade_history)

    # 結果表示
    print("=" * 70)
    print("[1. エントリー条件の成立パターン別]")
    print("=" * 70)
    print(f"{'パターン':<10} {'件数':>6} {'勝':>4} {'負':>4} {'勝率':>7} {'総PnL':>12}")
    # パターンを 5 条件 AND の名前にデコード (1=pass, 0=fail)
    cond_names = ["1_LT", "2_MT1", "3_MT2", "4_MT3", "5_ST"]
    for pattern in sorted(analysis["by_cond_pattern"].keys(), reverse=True):
        d = analysis["by_cond_pattern"][pattern]
        winrate = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        marks = " ".join([cond_names[i] + ("V" if pattern[i] == "1" else "X") for i in range(5)])
        print(f"  {pattern} {marks:<30} {d['count']:>6} {d['wins']:>4} {d['losses']:>4} {winrate:>6.1f}% {d['total_pnl']:>+12,.0f}")

    print(f"\n--- 個別条件の pass 率 ---")
    for k, d in analysis["by_condition_pass_rate"].items():
        total = d["pass"] + d["fail"]
        pass_rate = d["pass"] / total * 100 if total > 0 else 0
        win_rate = d["wins_when_pass"] / d["pass"] * 100 if d["pass"] > 0 else 0
        print(f"  {k:<10} pass {d['pass']:>4}/{total:<4} ({pass_rate:>5.1f}%)  pass時の勝率: {win_rate:>5.1f}%")

    print("\n" + "=" * 70)
    print("[2. エントリー方向別]")
    print("=" * 70)
    print(f"{'方向':<8} {'件数':>6} {'勝':>4} {'負':>4} {'勝率':>7} {'総PnL':>12} {'総pips':>10}")
    for side, d in sorted(analysis["by_side"].items()):
        winrate = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        print(f"  {side:<8} {d['count']:>6} {d['wins']:>4} {d['losses']:>4} {winrate:>6.1f}% {d['total_pnl']:>+12,.0f} {d['total_pips']:>+10,.1f}")

    print("\n" + "=" * 70)
    print("[3. 決済条件別]")
    print("=" * 70)
    print(f"{'決済理由':<10} {'件数':>6} {'勝':>4} {'負':>4} {'勝率':>7} {'総PnL':>12} {'平均PnL':>10}")
    for reason, d in sorted(analysis["by_exit_reason"].items(), key=lambda x: -x[1]["count"]):
        winrate = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        avg_pnl = d["total_pnl"] / d["count"] if d["count"] > 0 else 0
        print(f"  {reason:<10} {d['count']:>6} {d['wins']:>4} {d['losses']:>4} {winrate:>6.1f}% {d['total_pnl']:>+12,.0f} {avg_pnl:>+10,.0f}")

    print("\n" + "=" * 70)
    print("[4. 勝ち負け集計]")
    print("=" * 70)
    n_trades = analysis["n_trades"]
    n_wins = sum(d["wins"] for d in analysis["by_exit_reason"].values())
    n_losses = sum(d["losses"] for d in analysis["by_exit_reason"].values())
    winrate = n_wins / n_trades * 100 if n_trades > 0 else 0
    avg_win = sum(t.pnl for t in result.state.trade_history if t.pnl > 0) / max(1, n_wins)
    avg_loss = sum(t.pnl for t in result.state.trade_history if t.pnl < 0) / max(1, n_losses)
    print(f"  総トレード数:       {n_trades:>6}")
    print(f"  勝ち:              {n_wins:>6} ({winrate:.1f}%)")
    print(f"  負け:              {n_losses:>6} ({100 - winrate:.1f}%)")
    print(f"  ペイオフレシオ:     {avg_win / abs(avg_loss):.3f}" if avg_loss != 0 else "  ペイオフレシオ: N/A")
    print(f"  平均利益 (勝ち):    {avg_win:>+10,.0f} JPY")
    print(f"  平均損失 (負け):    {avg_loss:>+10,.0f} JPY")
    print(f"  総損益:             {analysis['total_pnl']:>+10,.0f} JPY")
    print(f"  総pips:            {analysis['total_pips']:>+10,.1f}")
    print(f"  最大連続勝ち:       {analysis['max_consecutive_wins']:>6}")
    print(f"  最大連続負け:       {analysis['max_consecutive_losses']:>6}")

    print("\n" + "=" * 70)
    print("[5. LT 方向別]")
    print("=" * 70)
    print(f"{'LT 方向':<10} {'件数':>6} {'勝':>4} {'負':>4} {'勝率':>7} {'総PnL':>12}")
    for lt, d in sorted(analysis["by_lt_direction"].items()):
        winrate = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        print(f"  {lt:<10} {d['count']:>6} {d['wins']:>4} {d['losses']:>4} {winrate:>6.1f}% {d['total_pnl']:>+12,.0f}")

    print("\n" + "=" * 70)
    print("[6. 月次別]")
    print("=" * 70)
    print(f"{'年月':<10} {'件数':>6} {'勝':>4} {'負':>4} {'総PnL':>12}")
    for month in sorted(analysis["by_month"].keys()):
        d = analysis["by_month"][month]
        print(f"  {month:<10} {d['count']:>6} {d['wins']:>4} {d['losses']:>4} {d['total_pnl']:>+12,.0f}")

    # trade history JSON 保存
    if args.save_trades:
        out_dir = ROOT / "research" / "EXP-FX000001" / "10-result"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"trades-{args.pair}-2024.json"
        trades_data = []
        for t in result.state.trade_history:
            trades_data.append({
                "side": t.side,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "entry_price": t.entry_price,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "exit_price": t.exit_price,
                "size": t.size,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "pnl": t.pnl,
                "pnl_pips": t.pnl_pips,
                "exit_reason": t.exit_reason,
                "entry_conditions": t.entry_conditions,
                "lt_direction": t.lt_direction,
                "sr_level_price": t.sr_level_price,
            })
        out_file.write_text(json.dumps({
            "generated_at": datetime.now().isoformat(),
            "pair": args.pair,
            "n_trades": len(trades_data),
            "trades": trades_data,
        }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\n[trade history 保存]: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
