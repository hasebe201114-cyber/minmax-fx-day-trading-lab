"""train/val/test 分離プロトコルでの SYS-FX007 v2 バックテスト.

差し戻し 3 対応: spec で固定された K1m〜K7m 評価基準で 3 期間の成績を評価.

期間 (2026-08-15 spec v2.2 改訂、OBS000007 追記5 参照):
  GMO 公開 klines API の実データ保持期間が 2023-10-27 頃以降しかないと判明したため、
  当初の 2020-2025 (6年) から取得可能な範囲 (2023-11 〜 現在、約2年9か月) に短縮再設計。
  spec 元の 50%/25%/25% 比率を踏襲。
  - Train:      2023-11-01 〜 2025-03-31 (約 17 か月)
  - Validation: 2025-04-01 〜 2025-11-30 (約 8 か月)
  - Test:       2025-12-01 〜 2026-08-15 (約 8.5 か月)

  旧期間定義 (実行不可、参考): Train 2020-01〜2022-12 / Validation 2023-01〜2024-06 / Test 2024-07〜2025-12

Usage:
  python scripts/run_train_val_test.py --pair USD_JPY --preset A1_A2_combined
  python scripts/run_train_val_test.py --all-pairs --preset A1_A2_combined
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

import pickle

from minmax_fx_dt.backtest import run_backtest, to_dict
from minmax_fx_dt.backtest.simulator import SimulatorConfig
from minmax_fx_dt.strategy.multi_timeframe import MTFConfig

# MTF キャッシュ (precompute_mtf.py で生成)
MTF_CACHE_DIR = ROOT / "data" / "curated" / "mtf_cache"


def _try_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


_USE_PARQUET = _try_parquet()


def load_mtf_cache(symbol: str) -> dict[str, pd.DataFrame] | None:
    """MTF キャッシュ (M5/M15/H4/D1) を読み込み. なければ None."""
    if _USE_PARQUET:
        d1 = MTF_CACHE_DIR / f"{symbol}_D1.parquet"
        if not d1.exists():
            return None
        return {
            "M5":  pd.read_parquet(MTF_CACHE_DIR / f"{symbol}_M5.parquet"),
            "M15": pd.read_parquet(MTF_CACHE_DIR / f"{symbol}_M15.parquet"),
            "H4":  pd.read_parquet(MTF_CACHE_DIR / f"{symbol}_H4.parquet"),
            "D1":  pd.read_parquet(MTF_CACHE_DIR / f"{symbol}_D1.parquet"),
        }
    pkl = MTF_CACHE_DIR / f"{symbol}.pkl"
    if not pkl.exists():
        return None
    with pkl.open("rb") as f:
        return pickle.load(f)


# spec で固定された期間 (00-spec.md §採用 / 確認プロトコル、v2.2 2026-08-15 改訂)
# GMO 公開 API のデータ保持期間制約 (2023-10-27 頃以降のみ) により、
# 取得可能な範囲に短縮再設計 (OBS000007 追記5)。
PERIODS = {
    "train":      ("2023-11-01", "2025-03-31"),
    "validation": ("2025-04-01", "2025-11-30"),
    "test":       ("2025-12-01", "2026-08-15"),
}

# プリセット定義 (ablation_sweep.py と同期)
PRESETS = {
    "Base": {
        "atr_stop_multiplier": 1.0,
        "reward_risk_ratio": 2.0,
        "mt_donchian_length": 20,
        "lt_adx_threshold": 20.0,
        "lt_sma_short": 20,
        "lt_sma_long": 50,
    },
    "A1_SL_TP": {
        "atr_stop_multiplier": 1.5,
        "reward_risk_ratio": 3.0,
        "mt_donchian_length": 20,
        "lt_adx_threshold": 20.0,
        "lt_sma_short": 20,
        "lt_sma_long": 50,
    },
    "A2_Donchian50": {
        "atr_stop_multiplier": 1.0,
        "reward_risk_ratio": 2.0,
        "mt_donchian_length": 50,
        "lt_adx_threshold": 20.0,
        "lt_sma_short": 20,
        "lt_sma_long": 50,
    },
    "A1_A2_combined": {
        "atr_stop_multiplier": 1.5,
        "reward_risk_ratio": 3.0,
        "mt_donchian_length": 50,
        "lt_adx_threshold": 20.0,
        "lt_sma_short": 20,
        "lt_sma_long": 50,
    },
    "A3_ADX30": {
        "atr_stop_multiplier": 1.5,
        "reward_risk_ratio": 3.0,
        "mt_donchian_length": 50,
        "lt_adx_threshold": 30.0,
        "lt_sma_short": 50,
        "lt_sma_long": 200,
    },
}

# 通貨別スプレッド (spec §コスト前提)
SPREAD_PIPS = {
    "USD_JPY": 0.3,
    "EUR_JPY": 0.5,
    "GBP_JPY": 0.7,
    "AUD_JPY": 0.6,
    "EUR_USD": 0.3,
}


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
    """M5 → H4 / D1 / M15 集約."""
    h4_df = pd.DataFrame({
        "open": m5_df["open"].resample("4h").first(),
        "high": m5_df["high"].resample("4h").max(),
        "low": m5_df["low"].resample("4h").min(),
        "close": m5_df["close"].resample("4h").last(),
    }).dropna()
    d1_df = pd.DataFrame({
        "open": h4_df["open"].resample("D").first(),
        "high": h4_df["high"].resample("D").max(),
        "low": h1_df["low"].resample("D").min() if False else h4_df["low"].resample("D").min(),
        "close": h4_df["close"].resample("D").last(),
    }).dropna()
    m15_df = pd.DataFrame({
        "open": m5_df["open"].resample("15min").first(),
        "high": m5_df["high"].resample("15min").max(),
        "low": m5_df["low"].resample("15min").min(),
        "close": m5_df["close"].resample("15min").last(),
    }).dropna()
    return d1_df, h4_df, m15_df


def filter_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """期間でフィルタリング."""
    return df[(df.index >= start) & (df.index <= end)]


def evaluate_kpis(metrics: dict) -> dict:
    """spec で固定された K1m〜K7m 評価基準.

    Returns:
        dict: 各 KPI の合否
    """
    sharpe = metrics.get("sharpe_monthly", float("nan"))
    pf = metrics.get("profit_factor_monthly", float("nan"))
    expectancy = metrics.get("expectancy_jpy", float("nan"))
    max_dd_monthly = metrics.get("max_dd_monthly_pct", float("nan"))
    max_dd_yearly = metrics.get("max_dd_yearly_pct", float("nan"))
    payoff = metrics.get("payoff_ratio", float("nan"))
    max_consec = metrics.get("max_consecutive_losses", 999)
    margin = metrics.get("max_margin_usage_pct", float("nan"))
    weak_breakout = metrics.get("weak_breakout_exclusion_pct", float("nan"))

    return {
        "K1m_sharpe_ge_0.4":     sharpe >= 0.4,
        "K1m_pf_ge_1.2":         pf >= 1.2,
        "K1m_expectancy_pos":    expectancy > 0,
        "K2m_dd_monthly_le_10":  max_dd_monthly <= 10.0,
        "K2m_dd_yearly_le_20":   max_dd_yearly <= 20.0,
        "K3m_max_consec_le_5":   max_consec <= 5,
        "K4m_payoff_ge_1.5":     payoff >= 1.5,
        "K7m_margin_le_30":      margin <= 30.0,
        "weak_breakout_ge_30":   weak_breakout >= 30.0,
    }


def run_one(symbol: str, preset_name: str, periods: dict = None) -> dict:
    """1 通貨 × 1 プリセット × 指定期間で実行.

    Args:
        symbol: 通貨ペア
        preset_name: プリセット名
        periods: {period_name: (start, end)} の辞書. None なら全 3 期間.
    """
    if periods is None:
        periods = PERIODS
    preset = PRESETS[preset_name]
    print(f"\n{'=' * 70}")
    print(f"[{symbol}] preset={preset_name}")
    print(f"{'=' * 70}")

    # MTF キャッシュ (precompute_mtf.py) があれば使用、なければライブ集約
    t_load0 = time.time()
    cache = load_mtf_cache(symbol)
    if cache is not None:
        m5_full = cache["M5"]
        print(f"  [CACHE HIT] M5: {len(m5_full)} bars ({m5_full.index[0].date()} - {m5_full.index[-1].date()})  load={time.time()-t_load0:.1f}秒")
        use_cache = True
    else:
        m5_full = load_ohlcv_from_ds1(symbol)
        print(f"  [CACHE MISS] M5 ライブ読み込み: {len(m5_full)} bars ({m5_full.index[0].date()} - {m5_full.index[-1].date()})  load={time.time()-t_load0:.1f}秒")
        use_cache = False

    sim_config = SimulatorConfig(
        initial_cash_jpy=1_000_000.0,
        lot_size=1_000,
        spread_pips=SPREAD_PIPS.get(symbol, 0.5),
        slippage_pips=0.5,
        is_jpy_pair="JPY" in symbol,
        weekend_close=True,
        max_dd_pause_threshold_pct=50.0,
    )
    mtf_config = MTFConfig(
        lt_sma_short=preset["lt_sma_short"],
        lt_sma_long=preset["lt_sma_long"],
        lt_adx_threshold=preset["lt_adx_threshold"],
        mt_donchian_length=preset["mt_donchian_length"],
        mt_atr_length=14,
        sr_min_touches=3,
        sr_cluster_threshold_pct=0.5,
        sr_fractal_window=5,
    )

    period_results = {}
    for period_name, (start, end) in periods.items():
        if use_cache:
            # キャッシュ済み: 期間フィルタのみ (高速)
            lt_df = filter_period(cache["D1"], start, end)
            mt_df = filter_period(cache["H4"], start, end)
            st_df = filter_period(cache["M15"], start, end)
        else:
            # ライブ: M5 読み込み + MTF 集約
            m5_period = filter_period(m5_full, start, end)
            if len(m5_period) < 1000:
                print(f"  [{period_name}] データ不足 ({len(m5_period)} bars), スキップ")
                continue
            try:
                lt_df, mt_df, st_df = aggregate_to_mtf(m5_period)
            except Exception as e:
                print(f"  [{period_name}] MTF 集約失敗: {e}")
                continue
        if len(mt_df) < 50:
            print(f"  [{period_name}] MT データ不足 ({len(mt_df)} bars), スキップ")
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
        m_dict = to_dict(result.metrics)
        kpis = evaluate_kpis(m_dict)
        n_pass = sum(1 for v in kpis.values() if v)

        print(f"  [{period_name}] {start} - {end} ({elapsed:.1f}秒)")
        print(f"    trades={m_dict['n_trades']:>4}  sharpe={m_dict['sharpe_monthly']:>7.3f}  PF={m_dict['profit_factor_monthly']:>5.2f}  "
              f"DD(m)={m_dict['max_dd_monthly_pct']:>5.2f}%  consec={m_dict['max_consecutive_losses']:>2}  "
              f"KPI pass={n_pass}/9")

        period_results[period_name] = {
            "start": start,
            "end": end,
            "metrics": m_dict,
            "kpi_pass": kpis,
            "kpi_pass_count": n_pass,
            "elapsed_sec": elapsed,
        }

    return {
        "pair": symbol,
        "preset": preset_name,
        "preset_params": preset,
        "periods": period_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="train/val/test 分離 SYS-FX007 バックテスト")
    parser.add_argument("--pair", help="通貨ペア (例: USD_JPY)")
    parser.add_argument("--all-pairs", action="store_true", help="5 通貨全て")
    parser.add_argument("--period", choices=list(PERIODS.keys()),
                        help="単一期間指定 (train/validation/test). 1 セル単独実行用")
    parser.add_argument("--preset", default="A1_A2_combined", choices=list(PRESETS.keys()),
                        help="プリセット名 (デフォルト A1_A2_combined)")
    args = parser.parse_args()

    pairs = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
    if args.all_pairs:
        targets = pairs
    elif args.pair:
        targets = [args.pair]
    else:
        targets = pairs

    # --period 指定時はその期間のみ実行
    if args.period:
        selected_periods = {args.period: PERIODS[args.period]}
    else:
        selected_periods = PERIODS

    print(f"=== train/val/test 分離バックテスト ===")
    print(f"プリセット: {args.preset}")
    print(f"期間: {[(k, v) for k, v in selected_periods.items()]}")
    print(f"対象通貨: {targets}")
    print()

    all_results = []
    total_t0 = time.time()
    for symbol in targets:
        try:
            r = run_one(symbol, args.preset, selected_periods)
            all_results.append(r)
        except Exception as e:
            print(f"[NG] {symbol}: {e}")

    total_elapsed = time.time() - total_t0

    # サマリ
    print(f"\n{'=' * 70}")
    print(f"全 {len(all_results)} 通貨のサマリ (preset={args.preset}, period={list(selected_periods.keys())})")
    print(f"{'=' * 70}")
    print(f"{'Pair':<10} {'Period':<12} {'trades':>6} {'sharpe':>7} {'PF':>6} {'DD(m)%':>7} {'consec':>6} {'pass':>5}")
    for r in all_results:
        for period_name, pr in r["periods"].items():
            m = pr["metrics"]
            print(f"  {r['pair']:<10} {period_name:<12} {m['n_trades']:>6} {m['sharpe_monthly']:>7.3f} "
                  f"{m['profit_factor_monthly']:>6.2f} {m['max_dd_monthly_pct']:>7.2f} "
                  f"{m['max_consecutive_losses']:>6} {pr['kpi_pass_count']:>5}")

    # JSON 出力
    out_dir = ROOT / "research" / "EXP-FX000001" / "10-result" / "train_val_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --period 指定時: 1 セルごとにユニークファイル
    if args.period:
        for r in all_results:
            for period_name, pr in r["periods"].items():
                cell_file = out_dir / f"tvt_{args.preset}_{r['pair']}_{period_name}.json"
                cell_file.write_text(
                    json.dumps({
                        "generated_at": datetime.now().isoformat(),
                        "preset": args.preset,
                        "preset_params": PRESETS[args.preset],
                        "pair": r["pair"],
                        "period": period_name,
                        "period_range": pr["start"] + " - " + pr["end"],
                        "metrics": pr["metrics"],
                        "kpi_pass": pr["kpi_pass"],
                        "kpi_pass_count": pr["kpi_pass_count"],
                        "elapsed_sec": pr["elapsed_sec"],
                    }, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                print(f"  [CELL] {cell_file.name} ({pr['elapsed_sec']:.0f}秒)")
    else:
        # 全期間: 単一ファイル
        out_file = out_dir / f"tvt_{args.preset}.json"
        out_file.write_text(
            json.dumps({
                "generated_at": datetime.now().isoformat(),
                "preset": args.preset,
                "preset_params": PRESETS[args.preset],
                "periods_definition": PERIODS,
                "n_pairs": len(all_results),
                "total_elapsed_sec": total_elapsed,
                "results": all_results,
            }, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"\n[出力]: {out_file}")

    print(f"総時間: {total_elapsed:.1f}秒")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
