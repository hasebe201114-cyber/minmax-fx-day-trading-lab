"""analyze_fx009_lt_asof_lookahead_impact.py の高速版(フォールバック).

production engineでのTrain感度分析(run_train_kpi)が5通貨×2バリアントで
非常に重い(M5行単位のPythonループ)ため、代表通貨(USD_JPY)のみに絞った版。
パラメータ導出部分(derivation)は全5通貨で実施(軽量、こちらは元データが正しい)。

出力: research/method-notes/fx009_lt_asof_lookahead_impact.json (同一出力先)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_fx009_lt_asof_lookahead_impact as base  # noqa: E402

# production engine感度分析を USD_JPY のみに縮小(高速化)
base.PAIRS_FOR_TRAIN_KPI = ["USD_JPY"]

_original_run_train_kpi = base.run_train_kpi


def run_train_kpi_fast(atr_trail_multiplier: float) -> dict:
    import json

    import pandas as pd

    from minmax_fx_dt.backtest.double_pattern_runner import run_double_pattern_backtest
    from minmax_fx_dt.backtest.metrics import to_dict
    from minmax_fx_dt.backtest.permutation import DEFAULT_N_PERMUTATIONS, permutation_test
    from minmax_fx_dt.backtest.simulator import SimulatorConfig
    from minmax_fx_dt.strategy.double_pattern_strategy import DoublePatternStrategyConfig
    from minmax_fx_dt.strategy.pattern_detection import DoublePatternConfig

    params_path = ROOT / "research" / "EXP-FX000003" / "10-result" / "double_pattern_params.json"
    with params_path.open(encoding="utf-8") as f:
        dp_params = json.load(f)
    ds7_path = ROOT / "data" / "curated" / "ds-7.json"
    with ds7_path.open(encoding="utf-8") as f:
        ds7 = json.load(f)
    swap_rates = {
        pair: {"long": v["swap_long_jpy_per_lot_per_day"], "short": v["swap_short_jpy_per_lot_per_day"]}
        for pair, v in ds7["pairs"].items()
    }
    spread_pips = {"USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6, "EUR_USD": 0.3}

    per_pair = {}
    all_pnls = []
    for pair in base.PAIRS_FOR_TRAIN_KPI:
        ds1_path = ROOT / "data" / "curated" / "ds-1.json"
        with ds1_path.open(encoding="utf-8") as f:
            ds1 = json.load(f)
        df = pd.DataFrame(ds1["pairs"][pair]["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        m5_period = df[(df.index >= base.TRAIN_START) & (df.index <= base.TRAIN_END)]

        lt_df = base.to_d1(m5_period)
        mt_df = base.to_h4(m5_period)

        dp_config = DoublePatternStrategyConfig(
            lt_sma_short=dp_params["lt_sma_short"], lt_sma_long=dp_params["lt_sma_long"],
            pattern=DoublePatternConfig(
                zigzag_threshold_atr=dp_params["zigzag_threshold_atr"],
                pattern_tolerance_atr=dp_params["pattern_tolerance_atr"],
                stop_buffer_atr=dp_params["stop_buffer_atr"],
                max_bars_since_second_pivot=dp_params["max_bars_since_second_pivot"],
            ),
            atr_length=14, atr_trail_multiplier=atr_trail_multiplier,
        )
        swap = swap_rates.get(pair, {"long": 0.0, "short": 0.0})
        sim_config = SimulatorConfig(
            initial_cash_jpy=1_000_000.0, lot_size=1_000,
            spread_pips=spread_pips.get(pair, 0.5), slippage_pips=0.5,
            is_jpy_pair="JPY" in pair, weekend_close=True, max_dd_pause_threshold_pct=50.0,
            swap_long_jpy_per_lot_per_day=swap["long"], swap_short_jpy_per_lot_per_day=swap["short"],
        )
        result = run_double_pattern_backtest(
            lt_ohlcv=lt_df, mt_ohlcv=mt_df, st_ohlcv=m5_period,
            pair=pair, sim_config=sim_config, dp_config=dp_config,
        )
        pnls = [t.pnl for t in result.state.trade_history]
        all_pnls.extend(pnls)
        per_pair[pair] = {"n_trades": len(pnls), "metrics": to_dict(result.metrics)}

    perm_result = permutation_test(all_pnls, n_permutations=DEFAULT_N_PERMUTATIONS) if all_pnls else None
    return {
        "atr_trail_multiplier": atr_trail_multiplier,
        "n_trades_total_5pairs": len(all_pnls),
        "pairs_evaluated": base.PAIRS_FOR_TRAIN_KPI,
        "perm_p_value_pooled": round(perm_result.p_value, 4) if perm_result else None,
        "per_pair": {
            p: {
                "n_trades": v["n_trades"], "sharpe_monthly": v["metrics"]["sharpe_monthly"],
                "profit_factor_monthly": v["metrics"]["profit_factor_monthly"],
                "max_dd_monthly_pct": v["metrics"]["max_dd_monthly_pct"],
                "payoff_ratio": v["metrics"]["payoff_ratio"],
            }
            for p, v in per_pair.items()
        },
    }


base.run_train_kpi = run_train_kpi_fast

if __name__ == "__main__":
    raise SystemExit(base.main())
