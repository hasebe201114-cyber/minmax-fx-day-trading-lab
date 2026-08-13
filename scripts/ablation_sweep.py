"""パラメータ・アブレーション分析.

USD/JPY 2024 で 1 プリセットずつ実行し、ベース基準 (前回結果) と比較.

プリセット定義:
  Base (前回結果):   SL=1.0 ATR, TP=2R, Donchian=20, ADX=20, SMA 20/50
  A1_SL_TP:          SL=1.5 ATR, TP=3R (RR 1:2 維持), Donchian=20, ADX=20, SMA 20/50
  A2_Donchian50:     SL=1.0 ATR, TP=2R, Donchian=50, ADX=20, SMA 20/50
  A1_A2_combined:    SL=1.5 ATR, TP=3R, Donchian=50, ADX=20, SMA 20/50
  A3_ADX30:          A1+A2 + ADX 30 + LT SMA 50/200 (長期トレンドフィルター強化)
  A4_PullbackDepth:  A1+A2 + ST プルバック深さ 0.5 ATR 以上 (浅いノイズプルバック排除)

Usage:
    # 1 プリセットだけ実行 (約 9 分, デフォルト USD_JPY)
    python scripts/ablation_sweep.py --preset A1_SL_TP
    python scripts/ablation_sweep.py --preset A3_ADX30
    python scripts/ablation_sweep.py --preset A4_PullbackDepth

    # 通貨ピボット (B1): 5 通貨 × A1_A2_combined (約 45 分)
    python scripts/ablation_sweep.py --preset A1_A2_combined --pair EUR_JPY
    python scripts/ablation_sweep.py --preset A1_A2_combined --pair GBP_JPY
    python scripts/ablation_sweep.py --preset A1_A2_combined --pair AUD_JPY
    python scripts/ablation_sweep.py --preset A1_A2_combined --pair EUR_USD

    # まとめて A1/A2/A1+A2 を連続実行 (約 27 分)
    python scripts/ablation_sweep.py --all

    # 通貨別レポート (バックテスト実行なし)
    python scripts/ablation_sweep.py --report --pair USD_JPY
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd

from minmax_fx_dt.backtest import run_backtest
from minmax_fx_dt.backtest.metrics import BacktestMetrics
from minmax_fx_dt.backtest.simulator import SimulatorConfig
from minmax_fx_dt.strategy.multi_timeframe import MTFConfig


def load_ohlcv_from_ds1(symbol: str) -> pd.DataFrame:
    """DS-1 JSON から指定通貨の OHLCV を読み込み."""
    ds1_path = ROOT / "data" / "curated" / "ds-1.json"
    with ds1_path.open(encoding="utf-8") as f:
        ds1 = json.load(f)
    if symbol not in ds1["pairs"]:
        raise ValueError(f"DS-1 に {symbol} がありません")
    records = ds1["pairs"][symbol]["data"]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def aggregate_to_mtf(m5_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


# プリセット定義
PRESETS: dict[str, dict] = {
    "Base": {
        "atr_stop_multiplier": 1.0,
        "reward_risk_ratio": 2.0,
        "mt_donchian_length": 20,
        "lt_adx_threshold": 20.0,
        "lt_sma_short": 20,
        "lt_sma_long": 50,
        "pb_min_depth_atr": 0.0,
    },
    "A1_SL_TP": {
        "atr_stop_multiplier": 1.5,
        "reward_risk_ratio": 2.0,  # TP = 1.5 * 2 = 3 ATR
        "mt_donchian_length": 20,
        "lt_adx_threshold": 20.0,
        "lt_sma_short": 20,
        "lt_sma_long": 50,
        "pb_min_depth_atr": 0.0,
    },
    "A2_Donchian50": {
        "atr_stop_multiplier": 1.0,
        "reward_risk_ratio": 2.0,
        "mt_donchian_length": 50,
        "lt_adx_threshold": 20.0,
        "lt_sma_short": 20,
        "lt_sma_long": 50,
        "pb_min_depth_atr": 0.0,
    },
    "A1_A2_combined": {
        "atr_stop_multiplier": 1.5,
        "reward_risk_ratio": 2.0,
        "mt_donchian_length": 50,
        "lt_adx_threshold": 20.0,
        "lt_sma_short": 20,
        "lt_sma_long": 50,
        "pb_min_depth_atr": 0.0,
    },
    # A3: A1+A2 をベースに、ADX 30 + 50/200 SMA で長期トレンドフィルター強化
    "A3_ADX30": {
        "atr_stop_multiplier": 1.5,
        "reward_risk_ratio": 2.0,  # TP = 1.5 * 2 = 3 ATR
        "mt_donchian_length": 50,
        "lt_adx_threshold": 30.0,
        "lt_sma_short": 50,
        "lt_sma_long": 200,
        "pb_min_depth_atr": 0.0,
    },
    # A4: A1+A2 をベースに、ST プルバック深さ 0.5 ATR 以上必須で浅いノイズプルバックを排除
    "A4_PullbackDepth": {
        "atr_stop_multiplier": 1.5,
        "reward_risk_ratio": 2.0,  # TP = 1.5 * 2 = 3 ATR
        "mt_donchian_length": 50,
        "lt_adx_threshold": 20.0,
        "lt_sma_short": 20,
        "lt_sma_long": 50,
        "pb_min_depth_atr": 0.5,  # breakout_level から 0.5 ATR 以上深く戻る
    },
}


def run_preset(preset_name: str, params: dict, symbol: str = "USD_JPY") -> dict:
    """1 プリセット実行."""
    print(f"\n{'=' * 70}")
    print(f"[{preset_name}] バックテスト実行中... パラメータ: {params}")
    print(f"{'=' * 70}")

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
        lt_sma_short=params["lt_sma_short"],
        lt_sma_long=params["lt_sma_long"],
        lt_adx_threshold=params["lt_adx_threshold"],
        mt_donchian_length=params["mt_donchian_length"],
        mt_atr_length=14,
        sr_min_touches=3,
        sr_cluster_threshold_pct=0.5,
        sr_fractal_window=5,
        atr_stop_multiplier=params["atr_stop_multiplier"],
        reward_risk_ratio=params["reward_risk_ratio"],
        pb_min_depth_atr=params.get("pb_min_depth_atr", 0.0),
    )

    m5_df = load_ohlcv_from_ds1(symbol)
    lt_df, mt_df, st_df = aggregate_to_mtf(m5_df)

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
    print(f"  実行時間: {elapsed:.1f}秒")
    print(f"  トレード: {m.n_trades} (勝 {m.n_wins}, 負 {m.n_losses}, 勝率 {m.win_rate:.1f}%)")
    print(f"  PnL: {result.state.cash - 1_000_000:+,.0f} JPY")
    print(f"  月次シャープ: {m.sharpe_monthly:.3f} | PF: {m.profit_factor_monthly:.3f}")
    print(f"  最大DD: {m.max_dd_pct if hasattr(m, 'max_dd_pct') else 0:.2f}% | 連続損失: {m.max_consecutive_losses}")

    return {
        "preset": preset_name,
        "params": params,
        "pair": symbol,
        "elapsed_sec": elapsed,
        "n_trades": m.n_trades,
        "n_wins": m.n_wins,
        "n_losses": m.n_losses,
        "win_rate": m.win_rate,
        "profit_factor": m.profit_factor_monthly,
        "sharpe_monthly": m.sharpe_monthly,
        "expectancy_jpy": m.expectancy_jpy,
        "max_dd_jpy": m.max_dd_jpy,
        "max_dd_pct": m.max_dd_monthly_pct,
        "max_consecutive_losses": m.max_consecutive_losses,
        "payoff_ratio": m.payoff_ratio,
        "max_margin_usage_pct": m.max_margin_usage_pct,
        "weak_breakout_exclusion_pct": m.weak_breakout_exclusion_pct,
        "total_pnl": result.state.cash - 1_000_000,
    }


# ベース基準は前回の分析結果から取得 (再実行不要)
BASE_RESULT = {
    "preset": "Base",
    "params": PRESETS["Base"],
    "pair": "USD_JPY",
    "elapsed_sec": 538.58,
    "n_trades": 130,
    "n_wins": 20,
    "n_losses": 110,
    "win_rate": 15.4,
    "profit_factor": 0.245,
    "sharpe_monthly": -3.876,
    "expectancy_jpy": -270,
    "max_dd_jpy": 36924,
    "max_dd_pct": 3.58,
    "max_consecutive_losses": 17,
    "payoff_ratio": 1.347,
    "max_margin_usage_pct": 0.65,
    "weak_breakout_exclusion_pct": 100.0,
    "total_pnl": -35108,
    "_source": "前回分析 (analyze_trades.py)",
}


def load_results(per_preset_dir: Path, pair: str = "USD_JPY") -> dict:
    """既存結果 (per-preset JSON) を読み込み (通貨別)."""
    if not per_preset_dir.exists():
        return {}
    out = {}
    for p in per_preset_dir.glob(f"ablation-{pair}-2024-*.json"):
        with p.open(encoding="utf-8") as f:
            r = json.load(f)
        out[r["preset"]] = r
    return out


def main() -> int:
    """メイン: --preset で 1 プリセット実行、--all で 3 プリセット連続実行."""
    parser = argparse.ArgumentParser(description="USD/JPY 2024 パラメータ・アブレーション")
    parser.add_argument(
        "--preset",
        choices=["A1_SL_TP", "A2_Donchian50", "A1_A2_combined", "A3_ADX30", "A4_PullbackDepth"],
        help="1 プリセットだけ実行 (約 9 分)",
    )
    parser.add_argument(
        "--pair",
        choices=["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"],
        default="USD_JPY",
        help="対象通貨ペア (デフォルト: USD_JPY)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="A1_SL_TP → A2_Donchian50 → A1_A2_combined を連続実行 (約 27 分)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="既存結果から比較表を生成 (バックテスト実行なし)",
    )
    args = parser.parse_args()

    output_dir = ROOT / "research" / "EXP-FX000001" / "10-result"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_preset_dir = output_dir / "ablation"
    per_preset_dir.mkdir(parents=True, exist_ok=True)

    # レポートモード: 既存結果だけ表示
    if args.report:
        existing = load_results(per_preset_dir, args.pair)
        # USD_JPY のみ Base を含む
        base = BASE_RESULT if args.pair == "USD_JPY" else None
        all_results = ([base] if base else []) + [existing[k] for k in ["A1_SL_TP", "A2_Donchian50", "A1_A2_combined", "A3_ADX30", "A4_PullbackDepth"] if k in existing]
        print_comparison(all_results, args.pair)
        return 0

    # 実行対象プリセット
    if args.preset:
        target_presets = [args.preset]
    elif args.all:
        target_presets = ["A1_SL_TP", "A2_Donchian50", "A1_A2_combined"]
    else:
        print("ERROR: --preset <name> か --all か --report を指定してください")
        return 1

    print("=" * 70)
    print(f"パラメータ・アブレーション分析 ({args.pair} 2024)")
    print("=" * 70)
    print(f"実行対象: {target_presets}")
    print(f"推定時間: {len(target_presets) * 9} 分 (1 プリセット約 9 分)")

    overall_start = time.time()
    for preset_name in target_presets:
        params = PRESETS[preset_name]
        result = run_preset(preset_name, params, symbol=args.pair)
        per_file = per_preset_dir / f"ablation-{args.pair}-2024-{preset_name}.json"
        per_file.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"  [保存]: {per_file}")

    overall_elapsed = time.time() - overall_start
    print(f"\n[実行時間]: {overall_elapsed / 60:.1f} 分")

    # 統合ファイル + 比較表を更新
    existing = load_results(per_preset_dir, args.pair)
    base = BASE_RESULT if args.pair == "USD_JPY" else None
    all_results = ([base] if base else []) + [existing[k] for k in ["A1_SL_TP", "A2_Donchian50", "A1_A2_combined", "A3_ADX30"] if k in existing]
    combined = output_dir / f"ablation-sweep-{args.pair}-2024.json"
    combined.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "data_period": "2024-01-01 to 2024-12-31",
            "pair": args.pair,
            "presets": PRESETS,
            "results": all_results,
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"[統合ファイル]: {combined}")
    print_comparison(all_results, args.pair)
    return 0


def print_comparison(results: list, pair: str = "USD_JPY") -> None:
    """比較表 + Base 改善度を出力 (通貨別)."""
    if not results:
        print("(結果なし)")
        return
    print(f"\n{'=' * 78}")
    print(f"アブレーション結果比較 ({pair} 2024)")
    print(f"{'=' * 78}")
    print(
        f"{'プリセット':<20} {'n_trades':>8} {'勝率':>6} {'PF':>6} {'シャープ':>8} "
        f"{'PnL(JPY)':>12} {'連続負':>7} {'最大DD%':>7}"
    )
    print("-" * 78)
    for r in results:
        print(
            f"{r['preset']:<20} {r['n_trades']:>8} {r['win_rate']:>5.1f}% {r['profit_factor']:>5.2f} "
            f"{r['sharpe_monthly']:>8.3f} {r['total_pnl']:>+12,.0f} {r['max_consecutive_losses']:>7} "
            f"{r['max_dd_pct']:>6.2f}%"
        )

    base = next((r for r in results if r["preset"] == "Base"), None)
    if base:
        print(f"\n{'=' * 78}")
        print("Base からの改善度")
        print(f"{'=' * 78}")
        for r in results:
            if r["preset"] == "Base":
                continue
            d_pnl = r["total_pnl"] - base["total_pnl"]
            d_pf = r["profit_factor"] - base["profit_factor"]
            d_wr = r["win_rate"] - base["win_rate"]
            d_sharpe = r["sharpe_monthly"] - base["sharpe_monthly"]
            d_n = r["n_trades"] - base["n_trades"]
            d_dd = r["max_dd_pct"] - base["max_dd_pct"]
            print(
                f"{r['preset']:<20} PnL {d_pnl:+10,.0f} | PF {d_pf:+5.2f} | 勝率 {d_wr:+5.1f}% | "
                f"シャープ {d_sharpe:+6.3f} | DD {d_dd:+5.2f}% | トレード {d_n:+5}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
