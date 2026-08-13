"""実データ (DS-1) を用いた SYS-FX007 バックテスト.

5 通貨 (USD/JPY, EUR/JPY, GBP/JPY, AUD/JPY, EUR/USD) の 2024 年 M5 足データで実行.

Usage:
    python scripts/run_backtest_real.py --pair USD_JPY
    python scripts/run_backtest_real.py --all-pairs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from minmax_fx_dt.backtest import run_backtest, to_dict
from minmax_fx_dt.backtest.simulator import SimulatorConfig
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


def aggregate_to_mtf(
    m5_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """M5 足から H4 / D1 を集約して 3 レイヤ MTF データを生成.

    Returns:
        (lt_df, mt_df, st_df) - D1, H4, M15 の OHLCV.
    """
    # H4: M5 集約 (4h = 48 本の M5)
    h4_df = pd.DataFrame({
        "open": m5_df["open"].resample("4h").first(),
        "high": m5_df["high"].resample("4h").max(),
        "low": m5_df["low"].resample("4h").min(),
        "close": m5_df["close"].resample("4h").last(),
    }).dropna()

    # D1: H4 集約 (1d = 6 本の H4)
    d1_df = pd.DataFrame({
        "open": h4_df["open"].resample("D").first(),
        "high": h4_df["high"].resample("D").max(),
        "low": h4_df["low"].resample("D").min(),
        "close": h4_df["close"].resample("D").last(),
    }).dropna()

    # ST (M15): M5 を 3 本集約
    m15_df = pd.DataFrame({
        "open": m5_df["open"].resample("15min").first(),
        "high": m5_df["high"].resample("15min").max(),
        "low": m5_df["low"].resample("15min").min(),
        "close": m5_df["close"].resample("15min").last(),
    }).dropna()

    return d1_df, h4_df, m15_df


def main() -> int:
    parser = argparse.ArgumentParser(description="実データ バックテスト (SYS-FX007)")
    parser.add_argument("--pair", help="通貨ペア (例: USD_JPY)")
    parser.add_argument("--all-pairs", action="store_true", help="5 通貨全て")
    args = parser.parse_args()

    pairs = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
    if args.all_pairs:
        targets = pairs
    elif args.pair:
        targets = [args.pair]
    else:
        targets = pairs  # デフォルトは全部

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

    all_results: list[dict] = []

    for symbol in targets:
        print(f"\n{'=' * 70}")
        print(f"[{symbol}] バックテスト実行中...")
        print(f"{'=' * 70}")

        try:
            m5_df = load_ohlcv_from_ds1(symbol)
            print(f"  M5 データ: {len(m5_df)} bars ({m5_df.index[0].date()} - {m5_df.index[-1].date()})")
            lt_df, mt_df, st_df = aggregate_to_mtf(m5_df)
            print(f"  D1 (LT): {len(lt_df)} bars")
            print(f"  H4 (MT): {len(mt_df)} bars")
            print(f"  M15 (ST): {len(st_df)} bars")
        except Exception as e:
            print(f"  [NG] データ読み込み失敗: {e}")
            continue

        t0 = time.time()
        result = run_backtest(
            lt_ohlcv=lt_df,
            mt_ohlcv=mt_df,
            st_ohlcv=st_df,
            pair=symbol,
            sim_config=sim_config,
            mtf_config=mtf_config,
        )
        elapsed = time.time() - t0
        m = result.metrics
        print(f"  実行時間: {elapsed:.2f}秒")
        print(f"  トレード: {m.n_trades} (勝 {m.n_wins}, 負 {m.n_losses}, 勝率 {m.win_rate:.1f}%)")
        print(f"  最終資金: {result.state.cash:,.0f} JPY (PnL: {result.state.cash - 1_000_000:+,.0f})")
        print(f"  月次シャープ: {m.sharpe_monthly:.3f} | PF: {m.profit_factor_monthly:.3f} | 期待値: {m.expectancy_jpy:+,.0f} JPY")
        print(f"  最大DD: {m.max_dd_jpy:,.0f} JPY ({m.max_dd_monthly_pct:.2f}%) | 連続損失: {m.max_consecutive_losses}")
        print(f"  ペイオフ: {m.payoff_ratio:.3f} | 証拠金使用率: {m.max_margin_usage_pct:.2f}%")
        print(f"  v2 弱いブレイク排除率: {m.weak_breakout_exclusion_pct:.1f}%")

        all_results.append({
            "pair": symbol,
            "metrics": to_dict(m),
            "elapsed_sec": elapsed,
        })

    # サマリ
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print(f"全 {len(all_results)} 通貨のサマリ")
        print(f"{'=' * 70}")
        total_trades = sum(r["metrics"]["n_trades"] for r in all_results)
        total_pnl = 0
        for r in all_results:
            print(f"  {r['pair']:<10} trades={r['metrics']['n_trades']:>4}  sharpe={r['metrics']['sharpe_monthly']:>7.4f}  PF={r['metrics']['profit_factor_monthly']:>5.2f}  DD={r['metrics']['max_dd_monthly_pct']:>5.2f}%")
        print(f"  合計トレード: {total_trades}")

    # 結果 JSON 保存
    output_dir = ROOT / "research" / "EXP-FX000001" / "10-result"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "real-metrics-2024.json"
    output_file.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "data_period": "2024-01-01 to 2024-12-31",
            "n_pairs": len(all_results),
            "total_bars_per_pair": 74232,
            "results": all_results,
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
