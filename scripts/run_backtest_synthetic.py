"""SYS-FX007 バックテスト実行スクリプト (合成データ版).

合成 MTF データを生成し、レンジブレイク・プルバック戦略のバックテストを実行する。
実データでの実行は scripts/run_backtest.py (将来実装) を参照。

Usage:
    python scripts/run_backtest_synthetic.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# src/ を PYTHONPATH に追加
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from minmax_fx_dt.backtest import run_backtest, to_dict
from minmax_fx_dt.backtest.simulator import SimulatorConfig
from minmax_fx_dt.strategy.multi_timeframe import MTFConfig


def generate_synthetic_mtf(
    start_date: str = "2024-01-01",
    n_days: int = 365,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """合成 MTF データを生成 (レンジ → ブレイク → トレンド).

    Returns:
        (lt_df, mt_df, st_df) - D1, H4, M15 の OHLCV.
    """
    np.random.seed(seed)
    start = pd.Timestamp(start_date)

    # D1 (1 年)
    d1_idx = pd.date_range(start=start, periods=n_days, freq="D")
    # ベーストレンド + レンジサイクル
    base = 150.0
    trend = np.linspace(0, 5, n_days)  # 150 → 155 へ緩やかな上昇
    cycle = 3 * np.sin(np.linspace(0, 4 * np.pi, n_days))  # レンジ ±3
    noise = np.random.normal(0, 0.5, n_days)
    d1_close = base + trend + cycle + noise
    d1_high = d1_close + np.abs(np.random.normal(0.4, 0.1, n_days))
    d1_low = d1_close - np.abs(np.random.normal(0.4, 0.1, n_days))
    d1_open = d1_close + np.random.normal(0, 0.2, n_days)
    d1_volume = np.random.randint(1000, 10000, n_days)

    # H4 (1 日 6 本 × n_days)
    n_h4 = n_days * 6
    h4_idx = pd.date_range(start=start, periods=n_h4, freq="4h")
    # D1 を補間して H4 を作成 (スケールを xp 範囲に合わせる)
    h4_close_interp = np.interp(
        np.linspace(0, n_days - 1, n_h4),  # 0〜59 を 360 等分
        np.arange(n_days),                  # 0〜59
        d1_close,
    )
    h4_close = h4_close_interp + np.random.normal(0, 0.2, n_h4)
    h4_high = h4_close + np.abs(np.random.normal(0.25, 0.08, n_h4))
    h4_low = h4_close - np.abs(np.random.normal(0.25, 0.08, n_h4))
    h4_open = h4_close + np.random.normal(0, 0.1, n_h4)
    h4_volume = np.random.randint(100, 1000, n_h4)

    # M15 (1 日 24 本 × n_days)
    n_m15 = n_days * 24
    m15_idx = pd.date_range(start=start, periods=n_m15, freq="15min")
    m15_close_interp = np.interp(
        np.linspace(0, n_h4 - 1, n_m15),  # 0〜359 を 1440 等分
        np.arange(n_h4),                   # 0〜359
        h4_close,
    )
    m15_close = m15_close_interp + np.random.normal(0, 0.1, n_m15)
    m15_high = m15_close + np.abs(np.random.normal(0.12, 0.04, n_m15))
    m15_low = m15_close - np.abs(np.random.normal(0.12, 0.04, n_m15))
    m15_open = m15_close + np.random.normal(0, 0.05, n_m15)
    m15_volume = np.random.randint(10, 100, n_m15)

    lt_df = pd.DataFrame({
        "open": d1_open, "high": d1_high, "low": d1_low, "close": d1_close, "volume": d1_volume,
    }, index=d1_idx)
    mt_df = pd.DataFrame({
        "open": h4_open, "high": h4_high, "low": h4_low, "close": h4_close, "volume": h4_volume,
    }, index=h4_idx)
    st_df = pd.DataFrame({
        "open": m15_open, "high": m15_high, "low": m15_low, "close": m15_close, "volume": m15_volume,
    }, index=m15_idx)

    return lt_df, mt_df, st_df


def main() -> int:
    """メイン: 合成データでバックテスト実行."""
    print("=" * 70)
    print("SYS-FX007 レンジブレイク・プルバック戦略 バックテスト (合成データ)")
    print("=" * 70)

    # 合成データ生成
    print("\n[1] 合成データ生成中...")
    lt_df, mt_df, st_df = generate_synthetic_mtf(start_date="2024-01-01", n_days=365, seed=42)
    print(f"  D1  (LT): {len(lt_df)} bars, {lt_df.index[0].date()} - {lt_df.index[-1].date()}")
    print(f"  H4  (MT): {len(mt_df)} bars")
    print(f"  M15 (ST): {len(st_df)} bars")

    # バックテスト設定
    sim_config = SimulatorConfig(
        initial_cash_jpy=1_000_000.0,
        lot_size=1_000,
        spread_pips=0.3,
        slippage_pips=0.5,
        is_jpy_pair=True,
        weekend_close=True,
        max_hold_days=20,
        max_dd_pause_threshold_pct=50.0,
    )
    mtf_config = MTFConfig(
        lt_sma_short=20,
        lt_sma_long=50,
        lt_adx_threshold=20.0,
        mt_donchian_length=20,
        mt_atr_length=14,
        sr_min_touches=2,
        sr_cluster_threshold_pct=0.5,
    )

    # バックテスト実行
    print("\n[2] バックテスト実行中...")
    result = run_backtest(
        lt_ohlcv=lt_df,
        mt_ohlcv=mt_df,
        st_ohlcv=st_df,
        pair="USDJPY (synthetic)",
        sim_config=sim_config,
        mtf_config=mtf_config,
    )

    # 結果表示
    print("\n" + "=" * 70)
    print("バックテスト結果")
    print("=" * 70)
    m = result.metrics
    print(f"\n[通貨ペア]: {result.pair}")
    print(f"[期間]: {m.period_start.date()} - {m.period_end.date()}")
    print(f"[初期資金]: {sim_config.initial_cash_jpy:,.0f} JPY")
    print(f"[最終資金]: {result.state.cash:,.0f} JPY")
    print(f"[総損益]: {result.state.cash - sim_config.initial_cash_jpy:+,.0f} JPY")
    print()
    print("--- K1m: 月次シャープ / PF / 期待値 ---")
    print(f"  月次シャープレシオ: {m.sharpe_monthly:.4f}  (目標: >= 0.4)")
    print(f"  年次シャープレシオ: {m.sharpe_yearly:.4f}")
    print(f"  月次 Profit Factor:  {m.profit_factor_monthly:.4f}  (目標: >= 1.2)")
    print(f"  期待値 (JPY/トレード): {m.expectancy_jpy:+,.2f}  (目標: > 0)")
    print()
    print("--- K2m: 最大 DD ---")
    print(f"  最大 DD (JPY):     {m.max_dd_jpy:,.0f}")
    print(f"  月間最大 DD (%):   {m.max_dd_monthly_pct:.4f}  (目標: <= 10.0)")
    print(f"  年間最大 DD (%):   {m.max_dd_yearly_pct:.4f}  (目標: <= 20.0)")
    print()
    print("--- K3m / K4m: 連続損失 / ペイオフレシオ ---")
    print(f"  最大連続損失:       {m.max_consecutive_losses}  (目標: <= 5)")
    print(f"  ペイオフレシオ:     {m.payoff_ratio:.4f}  (目標: >= 1.5)")
    print()
    print("--- K7m: 両建て証拠金消費率 ---")
    print(f"  最大証拠金消費率:   {m.max_margin_usage_pct:.4f}  (目標: <= 30.0)")
    print()
    print("--- v2 拡張: 弱いブレイク排除率 ---")
    print(f"  弱いブレイク排除率: {m.weak_breakout_exclusion_pct:.2f}  (目標: >= 30.0)")
    print()
    print("--- 取引統計 ---")
    print(f"  総トレード数:       {m.n_trades}")
    print(f"  勝ち:              {m.n_wins}")
    print(f"  負け:              {m.n_losses}")
    print(f"  勝率:              {m.win_rate:.2f}%")

    # JSON 出力
    output_dir = ROOT / "research" / "EXP-FX000001" / "10-result"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "synthetic-metrics.json"
    output_file.write_text(
        json.dumps(to_dict(m), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {output_file}")

    # GO/PAUSE 判定 (K1m〜K7m すべて満たすか)
    print("\n" + "=" * 70)
    print("GO/PAUSE 判定 (K1m〜K7m すべて満たすか)")
    print("=" * 70)
    checks = [
        ("K1m 月次シャープ >= 0.4", m.sharpe_monthly >= 0.4),
        ("K1m 月次 PF >= 1.2", m.profit_factor_monthly >= 1.2),
        ("K1m 期待値 > 0", m.expectancy_jpy > 0),
        ("K2m 月間 DD <= 10%", m.max_dd_monthly_pct <= 10.0),
        ("K2m 年間 DD <= 20%", m.max_dd_yearly_pct <= 20.0),
        ("K3m 最大連続損失 <= 5", m.max_consecutive_losses <= 5),
        ("K4m ペイオフレシオ >= 1.5", m.payoff_ratio >= 1.5),
        ("K7m 証拠金消費率 <= 30%", m.max_margin_usage_pct <= 30.0),
        ("v2 弱いブレイク排除率 >= 30%", m.weak_breakout_exclusion_pct >= 30.0),
    ]
    for name, ok in checks:
        mark = "OK " if ok else "NG "
        print(f"  [{mark}] {name}")
    all_pass = all(ok for _, ok in checks)
    print(f"\n総合判定: {'GO' if all_pass else 'WATCH/REJECT'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
