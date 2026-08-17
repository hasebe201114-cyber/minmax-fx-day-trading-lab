"""train/val/test 分離プロトコルでの SYS-FX008 トレンドフォロー・MAクロス バックテスト.

EXP-FX000002/00-spec.md v1.1 で確定した基準・期間分割で評価する。
SYS-FX007 (run_train_val_test.py) と同じ Train/Validation/Test 期間分割・
同じ decision.criteria 判定エンジン・同じ permutation test を使う。

期間 (SYS-FX007 と同一、GMOデータ保持期間制約による):
  - Train:      2023-11-01 〜 2025-03-31 (約 17 か月)
  - Validation: 2025-04-01 〜 2025-11-30 (約 8 か月)
  - Test:       2025-12-01 〜 2026-08-15 (約 8.5 か月)

Usage:
  python scripts/run_train_val_test_fx008.py --pair USD_JPY --period train
  python scripts/run_train_val_test_fx008.py --all-pairs --period train
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

import pandas as pd

from minmax_fx_dt.backtest.metrics import to_dict
from minmax_fx_dt.backtest.permutation import DEFAULT_N_PERMUTATIONS, permutation_test
from minmax_fx_dt.backtest.simulator import SimulatorConfig
from minmax_fx_dt.backtest.trend_follow_runner import run_trend_follow_backtest
from minmax_fx_dt.decision.criteria import Stats, evaluate_kpis, kpi_pass_summary
from minmax_fx_dt.strategy.trend_following import TrendFollowConfig

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]

PERIODS = {
    "train":      ("2023-11-01", "2025-03-31"),
    "validation": ("2025-04-01", "2025-11-30"),
    "test":       ("2025-12-01", "2026-08-15"),
}

# 通貨別スプレッド (SYS-FX007 run_train_val_test.py §コスト前提と同一値)
SPREAD_PIPS = {
    "USD_JPY": 0.3,
    "EUR_JPY": 0.5,
    "GBP_JPY": 0.7,
    "AUD_JPY": 0.6,
    "EUR_USD": 0.3,
}


def load_swap_rates() -> dict[str, dict[str, float]]:
    ds7_path = ROOT / "data" / "curated" / "ds-7.json"
    with ds7_path.open(encoding="utf-8") as f:
        ds7 = json.load(f)
    return {
        pair: {"long": v["swap_long_jpy_per_lot_per_day"], "short": v["swap_short_jpy_per_lot_per_day"]}
        for pair, v in ds7["pairs"].items()
    }


def load_trend_follow_params() -> dict[str, dict]:
    path = ROOT / "research" / "EXP-FX000002" / "10-result" / "trend_follow_params.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)["pairs"]


SWAP_RATES = load_swap_rates()
TREND_FOLLOW_PARAMS = load_trend_follow_params()


def is_jpy_pair(pair: str) -> bool:
    return "JPY" in pair


def load_ohlcv(symbol: str) -> pd.DataFrame:
    ds1_path = ROOT / "data" / "curated" / "ds-1.json"
    with ds1_path.open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][symbol]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def to_d1(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "open": m5["open"].resample("D").first(),
        "high": m5["high"].resample("D").max(),
        "low": m5["low"].resample("D").min(),
        "close": m5["close"].resample("D").last(),
    }).dropna()


def to_h4(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "open": m5["open"].resample("4h").first(),
        "high": m5["high"].resample("4h").max(),
        "low": m5["low"].resample("4h").min(),
        "close": m5["close"].resample("4h").last(),
    }).dropna()


def build_stats(m_dict: dict, *, perm_p_value: float | None) -> Stats:
    """BacktestMetrics.to_dict() の出力から decision.criteria.Stats を構築 (SYS-FX008 版)."""
    stats: Stats = {
        "strategy_id": "SYS-FX008",
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
        # K6m: train/val/testの各期間を独立にバックテストするだけで、
        # フォワードテスト(実運用)との比較は行っていないため判定対象外。
        "backtest_forward_divergence_pct": None,
        "permutation_p_value": perm_p_value,
        # SYS-FX008は片側ポジションのみ (00-spec.md §両建て: 対象外)。K7m は判定対象外。
        "hedging_enabled": False,
    }
    return stats


def run_one(symbol: str, periods: dict = None) -> dict:
    if periods is None:
        periods = PERIODS

    print(f"\n{'=' * 70}")
    print(f"[{symbol}] SYS-FX008 トレンドフォロー")
    print(f"{'=' * 70}")

    t_load0 = time.time()
    m5_full = load_ohlcv(symbol)
    print(f"  M5: {len(m5_full)} bars ({m5_full.index[0].date()} - {m5_full.index[-1].date()})  load={time.time()-t_load0:.1f}秒")

    swap = SWAP_RATES.get(symbol, {"long": 0.0, "short": 0.0})
    sim_config = SimulatorConfig(
        initial_cash_jpy=1_000_000.0,
        lot_size=1_000,
        spread_pips=SPREAD_PIPS.get(symbol, 0.5),
        slippage_pips=0.5,
        is_jpy_pair=is_jpy_pair(symbol),
        weekend_close=True,
        max_dd_pause_threshold_pct=50.0,
        swap_long_jpy_per_lot_per_day=swap["long"],
        swap_short_jpy_per_lot_per_day=swap["short"],
    )

    p = TREND_FOLLOW_PARAMS[symbol]
    tf_config = TrendFollowConfig(
        lt_sma_short=p["ma_short"],
        lt_sma_long=p["ma_long"],
        lt_trend_strength_threshold=p["adx_threshold"],
    )

    period_results = {}
    for period_name, (start, end) in periods.items():
        m5_period = m5_full[(m5_full.index >= start) & (m5_full.index <= end)]
        if len(m5_period) < 1000:
            print(f"  [{period_name}] データ不足 ({len(m5_period)} bars), スキップ")
            continue

        lt_df = to_d1(m5_period)
        mt_df = to_h4(m5_period)
        if len(lt_df) < tf_config.lt_sma_long or len(mt_df) < tf_config.mt_confirm_sma_length:
            print(f"  [{period_name}] LT/MTデータ不足 (D1={len(lt_df)}, H4={len(mt_df)}), スキップ")
            continue

        t0 = time.time()
        result = run_trend_follow_backtest(
            lt_ohlcv=lt_df, mt_ohlcv=mt_df, st_ohlcv=m5_period,
            pair=symbol, sim_config=sim_config, tf_config=tf_config,
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
                {"metric": e.metric, "observed": e.observed, "threshold": e.threshold,
                 "pass": e.pass_, "applicable": e.applicable, "note": e.note}
                for e in kpi_evals
            ],
            "kpi_summary": summary,
            "permutation_test": perm_result.to_dict(),
            "trade_pnls": trade_pnls,
            "elapsed_sec": elapsed,
        }

    return {
        "pair": symbol,
        "tf_params": {"lt_sma_short": tf_config.lt_sma_short, "lt_sma_long": tf_config.lt_sma_long,
                      "lt_trend_strength_threshold": tf_config.lt_trend_strength_threshold},
        "periods": period_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="train/val/test 分離 SYS-FX008 バックテスト")
    parser.add_argument("--pair", help="通貨ペア (例: USD_JPY)")
    parser.add_argument("--all-pairs", action="store_true", help="5 通貨全て")
    parser.add_argument("--period", choices=list(PERIODS.keys()),
                        help="単一期間指定 (train/validation/test). 1 セル単独実行用")
    args = parser.parse_args()

    targets = PAIRS if (args.all_pairs or not args.pair) else [args.pair]
    selected_periods = {args.period: PERIODS[args.period]} if args.period else PERIODS

    print("=== train/val/test 分離バックテスト (SYS-FX008) ===")
    print(f"期間: {[(k, v) for k, v in selected_periods.items()]}")
    print(f"対象通貨: {targets}")
    print()

    all_results = []
    total_t0 = time.time()
    for symbol in targets:
        try:
            r = run_one(symbol, selected_periods)
            all_results.append(r)
        except Exception as e:
            print(f"[NG] {symbol}: {e}")
            raise

    total_elapsed = time.time() - total_t0

    print(f"\n{'=' * 70}")
    print(f"全 {len(all_results)} 通貨のサマリ (period={list(selected_periods.keys())})")
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

    out_dir = ROOT / "research" / "EXP-FX000002" / "10-result" / "train_val_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in all_results:
        for period_name, pr in r["periods"].items():
            cell_file = out_dir / f"tvt_{r['pair']}_{period_name}.json"
            cell_file.write_text(
                json.dumps({
                    "generated_at": datetime.now().isoformat(),
                    "pair": r["pair"],
                    "period": period_name,
                    "tf_params": r["tf_params"],
                    **pr,
                }, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

    print(f"\n総時間: {total_elapsed:.1f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
