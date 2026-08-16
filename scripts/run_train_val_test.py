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
from minmax_fx_dt.backtest.permutation import DEFAULT_N_PERMUTATIONS, permutation_test
from minmax_fx_dt.backtest.simulator import SimulatorConfig
from minmax_fx_dt.decision.criteria import Stats, evaluate_kpis, kpi_pass_summary
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
    # OBS000005/00-spec.md v2.3「改善ループの停止条件」に基づく残り3試行枠。
    # 司令塔判断(2026-08-15、統計的有意性なしへの対応方針②選択)を受け、
    # A1_A2_combined を土台にエントリー条件を1軸ずつ緩和した候補。
    # Train期間のみで評価し、validation/testには絞り込み後に1度だけ触れる
    # (spec v2.3 の逐次探索プロトコル)。
    "B1_LooseSR": {
        # MT-2 (S/Rライン) の最小接触回数を3→2に緩和。ATRベースの許容誤差
        # (OBS000007で修正したバグ)自体は変更せず、S/Rラインと認定される
        # 価格帯を増やす方向で頻度を上げる。
        "atr_stop_multiplier": 1.5,
        "reward_risk_ratio": 3.0,
        "mt_donchian_length": 50,
        "lt_adx_threshold": 20.0,
        "lt_sma_short": 20,
        "lt_sma_long": 50,
        "sr_min_touches": 2,
    },
    "B2_ShortDonchian": {
        # MT-1 (レンジブレイク) のDonchian期間を50→20に短縮。ブレイク判定の
        # 参照レンジを短くし、ブレイクそのものの検出頻度を上げる。
        "atr_stop_multiplier": 1.5,
        "reward_risk_ratio": 3.0,
        "mt_donchian_length": 20,
        "lt_adx_threshold": 20.0,
        "lt_sma_short": 20,
        "lt_sma_long": 50,
    },
    "B3_LowerADX": {
        # LT (長期方向)のADX閾値を20→15に緩和。トレンド強度フィルターを
        # 通過しやすくし、LT条件でのシグナル排除を減らす。
        "atr_stop_multiplier": 1.5,
        "reward_risk_ratio": 3.0,
        "mt_donchian_length": 50,
        "lt_adx_threshold": 15.0,
        "lt_sma_short": 20,
        "lt_sma_long": 50,
    },
    # OBS000006 Phase 1 (追記4): トレンド強度指標の代替比較。
    # lt_adx_threshold は通貨ペア別に TREND_STRENGTH_THRESHOLDS から上書きする
    # (_trend_strength_key 参照)。他の条件は A1_A2_combined と同一に固定し、
    # 指標選択と閾値のみを変数として分離する。
    "C1_ADXPercentile70": {
        "atr_stop_multiplier": 1.5,
        "reward_risk_ratio": 3.0,
        "mt_donchian_length": 50,
        "lt_adx_threshold": 20.0,  # _trend_strength_key で上書きされる
        "lt_sma_short": 20,
        "lt_sma_long": 50,
        "lt_trend_strength_method": "adx",
        "_trend_strength_key": "C1_ADXPercentile70",
    },
    "C2_WilderADXPercentile70": {
        "atr_stop_multiplier": 1.5,
        "reward_risk_ratio": 3.0,
        "mt_donchian_length": 50,
        "lt_adx_threshold": 20.0,
        "lt_sma_short": 20,
        "lt_sma_long": 50,
        "lt_trend_strength_method": "adx_wilder",
        "_trend_strength_key": "C2_WilderADXPercentile70",
    },
    "C3_MASpreadATRPercentile70": {
        "atr_stop_multiplier": 1.5,
        "reward_risk_ratio": 3.0,
        "mt_donchian_length": 50,
        "lt_adx_threshold": 1.5,
        "lt_sma_short": 20,
        "lt_sma_long": 50,
        "lt_trend_strength_method": "ma_spread_atr",
        "_trend_strength_key": "C3_MASpreadATRPercentile70",
    },
}


def load_trend_strength_thresholds() -> dict[str, dict[str, float]]:
    """OBS000006 Phase 2 で事前登録した通貨ペア別トレンド強度閾値を読み込み."""
    path = ROOT / "research" / "EXP-FX000001" / "10-result" / "trend_strength_thresholds.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["pairs"]


TREND_STRENGTH_THRESHOLDS = load_trend_strength_thresholds()

# 通貨別スプレッド (spec §コスト前提)
SPREAD_PIPS = {
    "USD_JPY": 0.3,
    "EUR_JPY": 0.5,
    "GBP_JPY": 0.7,
    "AUD_JPY": 0.6,
    "EUR_USD": 0.3,
}


def load_swap_rates() -> dict[str, dict[str, float]]:
    """DS-7 (data/curated/ds-7.json) からスワップレートを読み込み.

    2026-08-16 ACTIVE.md フェーズゲート対応: これまで SimulatorConfig の
    swap_long/short_jpy_per_lot_per_day がデフォルト 0.0 のまま一度も接続
    されておらず、全 TVT がスワップ無視で実行されていた (OBS000005 追記4)。
    ds-7.json は 2024 年単年の概算値であり実運用前の GMO 公式データでの
    再計算が必要 (metadata._note 参照) だが、「0円固定」よりは実態に近い
    近似として接続する。
    """
    ds7_path = ROOT / "data" / "curated" / "ds-7.json"
    with ds7_path.open(encoding="utf-8") as f:
        ds7 = json.load(f)
    return {
        pair: {
            "long": v["swap_long_jpy_per_lot_per_day"],
            "short": v["swap_short_jpy_per_lot_per_day"],
        }
        for pair, v in ds7["pairs"].items()
    }


SWAP_RATES = load_swap_rates()


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


# OBS000005 差し戻し1 (独立監査追記2 で明記): 判定は自前再実装ではなく
# src/minmax_fx_dt/decision/criteria.py の evaluate_kpis() を経由する。
# K5m・K6m・min_n_trades・permutation_p_value を含む spec 記載の全ゲートを
# コード化された単一の判定基準で評価するため (旧実装は 9 ゲートのみで
# この 4 項目が丸ごと欠落していた)。


def build_stats(
    m_dict: dict,
    *,
    perm_p_value: float | None,
) -> Stats:
    """BacktestMetrics.to_dict() の出力から decision.criteria.Stats を構築."""
    stats: Stats = {
        "strategy_id": "SYS-FX007",
        "n_trades": m_dict["n_trades"],
        "sharpe_monthly": m_dict["sharpe_monthly"],
        "profit_factor_monthly": m_dict["profit_factor_monthly"],
        "expectancy_jpy": m_dict["expectancy_jpy"],
        "max_dd_monthly_pct": m_dict["max_dd_monthly_pct"],
        "max_dd_yearly_pct": m_dict["max_dd_yearly_pct"],
        "payoff_ratio": m_dict["payoff_ratio"],
        "max_consecutive_losses": m_dict["max_consecutive_losses"],
        "edge_per_trade_jpy": m_dict["edge_per_trade_jpy"],
        "spread_round_trip_jpy": m_dict["spread_round_trip_jpy"],
        "max_margin_usage_pct": m_dict["max_margin_usage_pct"],
        "weak_breakout_exclusion_pct": m_dict["weak_breakout_exclusion_pct"],
        # K6m: このスクリプトは train/val/test の各期間を独立にバックテストする
        # だけで、フォワードテスト(実運用)との比較は行っていないため判定対象外。
        "backtest_forward_divergence_pct": None,
        "permutation_p_value": perm_p_value,
        # 両建てロジックは runner.py に未統合 (PJ000003 既知の制約)。
        # margin usage は単一ポジションの値のため K7m は判定対象外として扱う。
        "hedging_enabled": False,
    }
    return stats


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

    swap = SWAP_RATES.get(symbol, {"long": 0.0, "short": 0.0})
    sim_config = SimulatorConfig(
        initial_cash_jpy=1_000_000.0,
        lot_size=1_000,
        spread_pips=SPREAD_PIPS.get(symbol, 0.5),
        slippage_pips=0.5,
        is_jpy_pair="JPY" in symbol,
        weekend_close=True,
        max_dd_pause_threshold_pct=50.0,
        swap_long_jpy_per_lot_per_day=swap["long"],
        swap_short_jpy_per_lot_per_day=swap["short"],
    )
    trend_strength_key = preset.get("_trend_strength_key")
    if trend_strength_key:
        lt_adx_threshold = TREND_STRENGTH_THRESHOLDS[symbol][trend_strength_key]
    else:
        lt_adx_threshold = preset["lt_adx_threshold"]

    mtf_config = MTFConfig(
        lt_sma_short=preset["lt_sma_short"],
        lt_sma_long=preset["lt_sma_long"],
        lt_adx_threshold=lt_adx_threshold,
        lt_trend_strength_method=preset.get("lt_trend_strength_method", "adx"),
        mt_donchian_length=preset["mt_donchian_length"],
        mt_atr_length=14,
        # OBS000005/00-spec.md v2.3 差し戻し2 対応: エントリー条件緩和候補
        # (B1_LooseSR 等) が sr_min_touches 等を上書きできるよう preset から
        # 読み取る。未指定の既存プリセットは従来通りの既定値のまま。
        sr_min_touches=preset.get("sr_min_touches", 3),
        sr_cluster_threshold_pct=preset.get("sr_cluster_threshold_pct", 0.5),
        sr_fractal_window=preset.get("sr_fractal_window", 5),
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

        trade_pnls = [t.pnl for t in result.state.trade_history]
        perm_result = permutation_test(trade_pnls, n_permutations=DEFAULT_N_PERMUTATIONS)

        stats = build_stats(m_dict, perm_p_value=perm_result.p_value if trade_pnls else None)
        kpi_evals = evaluate_kpis(stats)
        summary = kpi_pass_summary(kpi_evals)

        print(f"  [{period_name}] {start} - {end} ({elapsed:.1f}秒)")
        print(f"    trades={m_dict['n_trades']:>4}  sharpe={m_dict['sharpe_monthly']:>7.3f}  PF={m_dict['profit_factor_monthly']:>5.2f}  "
              f"DD(m)={m_dict['max_dd_monthly_pct']:>5.2f}%  consec={m_dict['max_consecutive_losses']:>2}  "
              f"perm_p={perm_result.p_value:>5.3f}  "
              f"KPI pass={summary['pass']}/{summary['applicable']} (対象外{summary['not_applicable']})")

        period_results[period_name] = {
            "start": start,
            "end": end,
            "metrics": m_dict,
            "kpi_evals": [
                {
                    "metric": e.metric,
                    "observed": e.observed,
                    "threshold": e.threshold,
                    "pass": e.pass_,
                    "applicable": e.applicable,
                    "note": e.note,
                }
                for e in kpi_evals
            ],
            "kpi_summary": summary,
            "permutation_test": perm_result.to_dict(),
            # OBS000005 追記2 差し戻し (成果物の再現可能性): permutation test や
            # 事後の再集計を、フルバックテストを再実行せずに行えるようトレード毎の
            # 損益を保存しておく。
            "trade_pnls": trade_pnls,
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
    print(f"{'Pair':<10} {'Period':<12} {'trades':>6} {'sharpe':>7} {'PF':>6} {'DD(m)%':>7} {'consec':>6} {'perm_p':>7} {'pass':>7}")
    for r in all_results:
        for period_name, pr in r["periods"].items():
            m = pr["metrics"]
            s = pr["kpi_summary"]
            perm_p = pr["permutation_test"]["p_value"]
            print(f"  {r['pair']:<10} {period_name:<12} {m['n_trades']:>6} {m['sharpe_monthly']:>7.3f} "
                  f"{m['profit_factor_monthly']:>6.2f} {m['max_dd_monthly_pct']:>7.2f} "
                  f"{m['max_consecutive_losses']:>6} {perm_p:>7.3f} {s['pass']:>3}/{s['applicable']:<3}")

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
                        "kpi_evals": pr["kpi_evals"],
                        "kpi_summary": pr["kpi_summary"],
                        "permutation_test": pr["permutation_test"],
                        "trade_pnls": pr["trade_pnls"],
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
