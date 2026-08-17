"""【診断専用・採用検討ではない】週末強制クローズを無効化した場合のSYS-FX009 v2再評価.

司令塔からの指摘: 「週末強制クローズ」と「H4スケールのダブルトップ/ボトム検出」は
時間軸が根本的にミスマッチしている可能性がある(H4パターンは形成に数週間かかる
一方、週末クローズにより実際の保有は最大4日に制限されている、
research/EXP-FX000003/00-spec.md §追記3参照)。この仮説を切り分けるため、
週末強制クローズのみを無効化し、他は一切変更せず(パラメータ再導出なし、
陳腐化シグナル回転売買バグ修正は適用済み)再評価する。

**重要: これは診断目的のみ。CLAUDE.md「週末持ち越し: 不可」は本PJの絶対制約であり、
ここでの結果がどうであれ、週末持ち越しをSYS-FX009またはその亜種の採用条件に
含めることはない。** 目的は「時間軸ミスマッチが本当に主因か」を切り分けることのみ。

出力: research/EXP-FX000003/10-result/trade_breakdown_no_weekend_close_DIAGNOSTIC.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd

from minmax_fx_dt.backtest.double_pattern_runner import run_double_pattern_backtest
from minmax_fx_dt.backtest.simulator import SimulatorConfig
from minmax_fx_dt.strategy.double_pattern_strategy import DoublePatternStrategyConfig
from minmax_fx_dt.strategy.pattern_detection import DoublePatternConfig

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
PERIODS = {
    "train":      ("2023-11-01", "2025-03-31"),
    "validation": ("2025-04-01", "2025-11-30"),
    "test":       ("2025-12-01", "2026-08-15"),
}
SPREAD_PIPS = {"USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6, "EUR_USD": 0.3}


def load_double_pattern_params() -> dict:
    path = ROOT / "research" / "EXP-FX000003" / "10-result" / "double_pattern_params.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_swap_rates() -> dict:
    with (ROOT / "data" / "curated" / "ds-7.json").open(encoding="utf-8") as f:
        ds7 = json.load(f)
    return {p: {"long": v["swap_long_jpy_per_lot_per_day"], "short": v["swap_short_jpy_per_lot_per_day"]}
            for p, v in ds7["pairs"].items()}


def load_ohlcv(symbol: str) -> pd.DataFrame:
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][symbol]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp").sort_index()


def to_d1(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: m5[c].resample("D").agg(a) for c, a in
                          [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]}).dropna()


def to_h4(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: m5[c].resample("4h").agg(a) for c, a in
                          [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]}).dropna()


def main() -> int:
    dp_params = load_double_pattern_params()
    swap_rates = load_swap_rates()
    dp_config = DoublePatternStrategyConfig(
        lt_sma_short=dp_params["lt_sma_short"],
        lt_sma_long=dp_params["lt_sma_long"],
        pattern=DoublePatternConfig(
            zigzag_threshold_atr=dp_params["zigzag_threshold_atr"],
            pattern_tolerance_atr=dp_params["pattern_tolerance_atr"],
            stop_buffer_atr=dp_params["stop_buffer_atr"],
            max_bars_since_second_pivot=dp_params["max_bars_since_second_pivot"],
        ),
        atr_length=14,
        atr_trail_multiplier=dp_params["atr_trail_multiplier"],
    )

    all_trades: list[dict] = []
    for pair in PAIRS:
        is_jpy = "JPY" in pair
        swap = swap_rates.get(pair, {"long": 0.0, "short": 0.0})
        sim_config = SimulatorConfig(
            initial_cash_jpy=1_000_000.0, lot_size=1_000,
            spread_pips=SPREAD_PIPS.get(pair, 0.5), slippage_pips=0.5,
            is_jpy_pair=is_jpy,
            weekend_close=False,  # ★診断: 週末強制クローズを無効化 (他は一切変更しない)
            max_dd_pause_threshold_pct=50.0,
            swap_long_jpy_per_lot_per_day=swap["long"], swap_short_jpy_per_lot_per_day=swap["short"],
        )
        m5_full = load_ohlcv(pair)
        for period_name, (start, end) in PERIODS.items():
            t0 = time.time()
            m5 = m5_full[(m5_full.index >= start) & (m5_full.index <= end)]
            lt_df, mt_df = to_d1(m5), to_h4(m5)
            result = run_double_pattern_backtest(
                lt_ohlcv=lt_df, mt_ohlcv=mt_df, st_ohlcv=m5,
                pair=pair, sim_config=sim_config, dp_config=dp_config,
            )
            for t in result.state.trade_history:
                all_trades.append({
                    "pair": pair, "period": period_name,
                    "side": t.side, "entry_time": str(t.entry_time), "exit_time": str(t.exit_time),
                    "entry_price": t.entry_price, "exit_price": t.exit_price,
                    "pnl": t.pnl, "pnl_pips": t.pnl_pips, "swap_pnl": t.swap_pnl,
                    "hold_days": t.hold_days, "exit_reason": t.exit_reason,
                    "initial_risk": t.initial_risk, "target_reached": t.target_reached,
                    "lt_direction": t.lt_direction,
                })
            print(f"[{pair}/{period_name}] trades={len(result.state.trade_history)} ({time.time()-t0:.1f}s)")

    out_path = ROOT / "research" / "EXP-FX000003" / "10-result" / "trade_breakdown_no_weekend_close_DIAGNOSTIC.json"
    out_path.write_text(json.dumps(all_trades, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[出力]: {out_path} (n={len(all_trades)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
