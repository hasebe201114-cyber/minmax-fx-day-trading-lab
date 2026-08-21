"""EXP-FX000005 外部レビューT-06+T-07対応: permutation検定のブロック順列化と
実効n二重計上の解消.

外部レビュー(F2)・C査読(`research/EXP-FX000005/20-c-review.md`)が共通して指摘:
`permutation_test_clustered()`は通貨ペア単位(高々4種類)でしか符号を独立に引かない
ため、検定の実効標本サイズが「通貨ペア数」まで縮小し、4通貨構成では全勝ケースでも
p値が0.3158までしか下がらない(nをいくら増やしても不変)。

T-06: `permutation_test_block()`(`src/minmax_fx_dt/backtest/permutation.py`に新設)
をこのスクリプトから使用する。クラスタキーはエントリー日(JST暦日)。同日に複数通貨が
エントリーした場合はそのクラスタ内で符号が連動するため、依存構造は実際の共起パターン
として自然に捉えられ、通貨ペア相関行列によるガウス・コピュラの仮定を必要としない。
比較のため旧`permutation_test_clustered()`の値も`perm_p_clustered`として併記する。

T-07: 相関補正を「検定」(T-06のブロック順列)と「実効トレード数」の両方に入れると
罰則が二重に効く(min_n_trades=300に対し4通貨では名目1,027件を要求する一方、検定側
もp値の下限に張り付く)。T-06で検定側が依存構造を直接捕捉するようになったため、
`compute_n_trades_effective(..., apply_correlation_discount=False)`でn側の相関割引を
撤廃する(KPI評価スクリプト側で対応、`evaluate_vol_breakout_dow_theory_4pairs_v7_t06fix_kpi.py`)。

T-01(週末強制クローズ)・T-02(ATRトレール倍率M5再導出)・T-03(決済判定のM5化)・
T-09(コストモデルのスリッページ復活)はこのスクリプトでも適用済み(共有モジュールの
修正が既定で効くため)。検出層・エントリー層・TP水準(TP3=4R)・価格反応型ショック
抑制フィルターは第7試行から完全に同一。Test期間は`00-spec.md`「Test凍結宣言」節に
より参照するが、修正の妥当性確認のみを目的とし、結論・意思決定には用いない
(Train+Validationのみで判断する)。

出力: research/method-notes/vol_breakout_dow_theory_4pairs_v7_t06fix_1000usd_backtest.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from backtest_vol_breakout_dow_theory import simulate_dow_theory_trend  # noqa: E402
from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402
from derive_vol_breakout_entry_params import N_BREAKOUT, to_h1  # noqa: E402
from price_shock_filter import make_price_shock_check  # noqa: E402
from minmax_fx_dt.backtest.permutation import permutation_test_block, permutation_test_clustered  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_train.json").open(encoding="utf-8") as f:
    TRAIN_RESULT = json.load(f)
TRAIN_PARAMS = TRAIN_RESULT["params"]
STOP_BUFFER_ATR_M5 = TRAIN_PARAMS["stop_buffer_atr_m5"]
ATR_TRAIL_MULTIPLIER = TRAIN_PARAMS["atr_trail_multiplier"]  # 旧値(H1設計転用、比較用に保持するのみ)
# T-02事前登録: トレール幅=1R×1.0(0.5〜1.5Rレンジの中央値)をATR(M5)基準で計算
ATR_TRAIL_MULTIPLIER_M5 = STOP_BUFFER_ATR_M5 * 1.0

PERIODS = {
    "train":      ("2023-11-01", "2025-03-31"),
    "validation": ("2025-04-01", "2025-11-30"),
    "test":       ("2025-12-01", "2026-08-15"),
}

TP_LEVELS_V5 = [(1.0, 0.40), (2.0, 0.35), (4.0, 0.25)]  # 改善ループ第5試行: TP3を3R→4Rへ
TP_LEVELS = TP_LEVELS_V5  # コストモデルのTP_CUM_FRACTION計算にも同じ水準を使用

SPREAD_PIPS = {"USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6, "EUR_USD": 0.3}
SLIPPAGE_PIPS_MARKET_LEG = 0.5  # 週末強制クローズ・MAX_HOLD向け(高ボラ非依存の一般成行)
SLIPPAGE_PIPS_STOP_TRIGGERED = 1.0  # T-09: SL・トレーリング向け(高ボラ局面限定の逆指値、レビュー提案レンジ0.5〜1.0pipの上限)
COMMISSION_RATE_ROUND_TRIP = 0.00004

_cum = 0.0
TP_CUM_FRACTION = [0.0]
for _r_level, _frac in TP_LEVELS:
    _cum += _frac
    TP_CUM_FRACTION.append(_cum)

RISK_PCT_PER_TRADE = 0.01
MAX_LEVERAGE = 25.0
INITIAL_CAPITAL_USD = 1000.0


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def load_m5_period(pair: str, start: str, end: str) -> pd.DataFrame:
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= start) & (df.index <= end)]


def find_trades_for_period(pair: str, m5: pd.DataFrame, shock_check) -> tuple[list[dict], pd.DataFrame, pd.Series]:
    h1 = to_h1(m5)
    atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)

    ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
    idxs = np.where(ratio.values >= N_BREAKOUT)[0]

    trades = []
    for i in idxs:
        pos = h1.index.get_loc(ratio.index[i])
        bar = h1.iloc[pos]
        direction = "UP" if bar["close"] > bar["open"] else "DOWN"
        trades.extend(simulate_dow_theory_trend(m5, atr_m5, h1, atr_h1, pos, direction,
                                                  STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER_M5,
                                                  blackout_check=shock_check,
                                                  tp_levels=TP_LEVELS_V5, skip_first_entry=False,
                                                  atr_trail_series=atr_m5, m5_exit=True))
    return trades, h1, atr_h1


def run_period(period_name: str, start: str, end: str) -> dict:
    print(f"\n=== {period_name}: {start} 〜 {end} ===")
    m5_by_pair, h1_by_pair, atr_h1_by_pair = {}, {}, {}
    for pair in SELECTED_PAIRS:
        m5 = load_m5_period(pair, start, end)
        if len(m5) < 1000:
            print(f"  [{pair}] データ不足 ({len(m5)}bars)、スキップ")
            continue
        m5_by_pair[pair] = m5
        h1_by_pair[pair] = to_h1(m5)
        atr_h1_by_pair[pair] = atr_ind(h1_by_pair[pair]["high"], h1_by_pair[pair]["low"], h1_by_pair[pair]["close"], length=14)
    shock_check = make_price_shock_check(h1_by_pair, atr_h1_by_pair)

    all_trades: list[dict] = []
    for pair, m5 in m5_by_pair.items():
        trades, h1, atr_h1 = find_trades_for_period(pair, m5, shock_check)
        spread = SPREAD_PIPS.get(pair, 0.5)
        pip = pip_size(pair)
        for sim in trades:
            fraction_via_tp = TP_CUM_FRACTION[sim["n_levels_hit"]]
            fraction_remaining = 1.0 - fraction_via_tp
            remaining_is_market = sim["exit_reason"] in ("WEEKEND_NO_TP", "TP_THEN_WEEKEND", "MAX_HOLD")
            remaining_is_stop_triggered = sim["exit_reason"] in ("SL_INITIAL_NO_TP", "TP_THEN_SL_TRAIL")
            entry_pips = spread + SLIPPAGE_PIPS_MARKET_LEG
            if remaining_is_market:
                exit_slippage = fraction_remaining * SLIPPAGE_PIPS_MARKET_LEG
            elif remaining_is_stop_triggered:
                exit_slippage = fraction_remaining * SLIPPAGE_PIPS_STOP_TRIGGERED
            else:
                exit_slippage = 0.0
            exit_pips = spread + exit_slippage
            cost_price = (entry_pips + exit_pips) * pip
            cost_r = cost_price / sim["initial_risk"]
            leverage_ratio = sim["entry_price"] / sim["initial_risk"]
            commission_r = COMMISSION_RATE_ROUND_TRIP * leverage_ratio
            r_net = sim["r"] - cost_r - commission_r
            all_trades.append({
                "pair": pair, "direction": sim["direction"],
                "entry_time": pd.Timestamp(sim["entry_time"]), "exit_time": sim["exit_time"],
                "entry_price": sim["entry_price"], "initial_risk": sim["initial_risk"],
                "exit_reason": sim["exit_reason"], "n_levels_hit": sim["n_levels_hit"],
                "fraction_via_tp": fraction_via_tp,
                "r_gross": sim["r"], "cost_r": cost_r, "commission_r": commission_r,
                "r_net": r_net, "leverage_ratio": leverage_ratio,
            })
        print(f"  [{pair}] トレード={len(trades)}件")

    all_trades.sort(key=lambda t: t["entry_time"])
    events = []
    for idx, t in enumerate(all_trades):
        events.append((t["entry_time"], 0, idx, "ENTRY"))
        events.append((t["exit_time"], 1, idx, "EXIT"))
    events.sort(key=lambda e: (e[0], e[1]))

    balance = INITIAL_CAPITAL_USD
    ruined = False
    equity_curve = [{"time": str(pd.Timestamp(start)), "balance": balance}]
    for time_, _order, idx, kind in events:
        t = all_trades[idx]
        if kind == "ENTRY":
            if ruined:
                t["risk_dollars"] = 0.0
                t["skipped_ruin"] = True
            else:
                max_risk_pct = MAX_LEVERAGE / t["leverage_ratio"] if t["leverage_ratio"] > 0 else RISK_PCT_PER_TRADE
                effective_risk_pct = min(RISK_PCT_PER_TRADE, max_risk_pct)
                t["risk_dollars"] = balance * effective_risk_pct
                t["effective_risk_pct"] = effective_risk_pct
                t["leverage_capped"] = effective_risk_pct < RISK_PCT_PER_TRADE
                t["skipped_ruin"] = False
        else:
            if t.get("skipped_ruin"):
                t["dollar_pnl"] = 0.0
            else:
                t["dollar_pnl"] = t["r_net"] * t["risk_dollars"]
                balance += t["dollar_pnl"]
                if balance <= 0:
                    balance = 0.0
                    ruined = True
            t["balance_after"] = balance
            equity_curve.append({"time": str(time_), "balance": balance})

    n = len(all_trades)
    n_effective_trades = sum(1 for t in all_trades if not t.get("skipped_ruin"))
    r_values = [t["r_net"] for t in all_trades if not t.get("skipped_ruin")]
    pairs_for_perm = [t["pair"] for t in all_trades if not t.get("skipped_ruin")]
    # T-06: クラスタキー=エントリー日(JST暦日)。同日にエントリーした複数通貨の
    # トレードは同じ符号を引く(=真の依存構造を通貨ペアの固定属性ではなく
    # 実際の共起パターンで捉える)。
    day_clusters_for_perm = [t["entry_time"].strftime("%Y-%m-%d") for t in all_trades if not t.get("skipped_ruin")]

    n_wins = sum(1 for r in r_values if r > 0)
    win_rate = n_wins / len(r_values) if r_values else None
    mean_r_net = float(np.mean(r_values)) if r_values else None
    final_balance = balance
    total_return_pct = (final_balance / INITIAL_CAPITAL_USD - 1.0) * 100.0

    balances = [pt["balance"] for pt in equity_curve]
    running_max = np.maximum.accumulate(balances) if balances else np.array([INITIAL_CAPITAL_USD])
    drawdowns = [(b - m) / m * 100.0 if m > 0 else 0.0 for b, m in zip(balances, running_max)]
    max_dd_pct = min(drawdowns) if drawdowns else 0.0

    perm_result = None
    perm_result_block = None
    if len(r_values) >= 4:
        perm_result = permutation_test_clustered(r_values, pairs_for_perm, seed=42)
        perm_result_block = permutation_test_block(r_values, day_clusters_for_perm, seed=42)

    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r < 0]
    profit_factor_val = (sum(wins) / abs(sum(losses))) if losses else None
    payoff_val = (float(np.mean(wins)) / abs(float(np.mean(losses)))) if wins and losses else None

    print(f"  トレード数={n} (破産ガードでスキップ={n - n_effective_trades})  最終残高=${final_balance:.2f}  "
          f"総リターン={total_return_pct:.1f}%  最大DD(初期資金比)={max_dd_pct:.1f}%")
    print(f"  勝率={win_rate:.3f}  平均r_net={mean_r_net:.4f}  PF={profit_factor_val}  ペイオフ={payoff_val}"
          f"{f'  perm_p(pair-cluster,旧)={perm_result.p_value:.4f}' if perm_result else ''}"
          f"{f'  perm_p(day-block,T-06新)={perm_result_block.p_value:.4f}' if perm_result_block else ''}")

    return {
        "period": period_name, "start": start, "end": end,
        "n_trades": n, "n_effective_trades": n_effective_trades,
        "final_balance_usd": round(final_balance, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "mean_r_net": round(mean_r_net, 4) if mean_r_net is not None else None,
        "profit_factor": round(profit_factor_val, 3) if profit_factor_val else None,
        "payoff_ratio": round(payoff_val, 3) if payoff_val else None,
        "perm_p_clustered": round(perm_result.p_value, 4) if perm_result else None,
        "perm_p_block": round(perm_result_block.p_value, 4) if perm_result_block else None,
        "perm_p_block_method": perm_result_block.method if perm_result_block else None,
        "leverage_ratio_stats": {
            "median": round(float(np.median([t["leverage_ratio"] for t in all_trades])), 1) if all_trades else None,
            "max": round(float(np.max([t["leverage_ratio"] for t in all_trades])), 1) if all_trades else None,
        },
        "n_leverage_capped": sum(1 for t in all_trades if t.get("leverage_capped")),
        "trades": [
            {k: (str(v) if isinstance(v, pd.Timestamp) else (round(v, 6) if isinstance(v, float) else v))
             for k, v in t.items()}
            for t in all_trades
        ],
        "equity_curve": equity_curve,
    }


def main() -> int:
    print("=== SYS-FX011 外部レビュー対応T-06+T-07(検定のブロック順列化・実効n是正) ===")
    print(f"対象通貨: {SELECTED_PAIRS}")
    print(f"事前登録(4通貨Train導出値をそのまま使用): N={N_BREAKOUT}, "
          f"zigzag_threshold_atr_m5={TRAIN_PARAMS['zigzag_threshold_atr_m5']}, "
          f"stop_buffer_atr_m5={STOP_BUFFER_ATR_M5}, "
          f"atr_trail_multiplier(旧H1流用)={ATR_TRAIL_MULTIPLIER}, atr_trail_multiplier_m5(新)={ATR_TRAIL_MULTIPLIER_M5}, "
          f"tp_levels={TP_LEVELS_V5}, filter=price_shock_reactive(trigger=7.0,calm=2.0,calm_bars=3), "
          f"risk_pct_per_trade={RISK_PCT_PER_TRADE}, initial_capital=${INITIAL_CAPITAL_USD}")

    period_results = {}
    for period_name, (start, end) in PERIODS.items():
        period_results[period_name] = run_period(period_name, start, end)

    print(f"\n=== サマリ ===")
    print(f"{'期間':<12}{'取引数':>8}{'最終残高':>12}{'総リターン':>10}{'最大DD':>8}{'勝率':>8}{'PF':>8}{'perm_p':>8}")
    for name, r in period_results.items():
        print(f"{name:<12}{r['n_trades']:>8}{'$'+str(r['final_balance_usd']):>12}"
              f"{str(r['total_return_pct'])+'%':>10}{str(r['max_drawdown_pct'])+'%':>8}"
              f"{r['win_rate']:>8}{r['profit_factor']:>8}{r['perm_p_clustered']:>8}")

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_v7_t06fix_1000usd_backtest.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "selected_pairs": SELECTED_PAIRS,
            "design": {
                "n_breakout": N_BREAKOUT, "zigzag_threshold_atr_m5": TRAIN_PARAMS["zigzag_threshold_atr_m5"],
                "stop_buffer_atr_m5": STOP_BUFFER_ATR_M5,
                "atr_trail_multiplier_h1_legacy": ATR_TRAIL_MULTIPLIER,
                "atr_trail_multiplier_m5": ATR_TRAIL_MULTIPLIER_M5,
                "exit_granularity": "m5",
                "risk_pct_per_trade": RISK_PCT_PER_TRADE, "max_leverage": MAX_LEVERAGE,
                "initial_capital_usd": INITIAL_CAPITAL_USD,
                "tp_levels": TP_LEVELS, "spread_pips": SPREAD_PIPS,
                "slippage_pips_per_market_leg": SLIPPAGE_PIPS_MARKET_LEG,
                "slippage_pips_stop_triggered": SLIPPAGE_PIPS_STOP_TRIGGERED,
                "commission_rate_round_trip": COMMISSION_RATE_ROUND_TRIP,
                "permutation_method": "block_sign_flip(cluster=entry_date_jst)",
            },
            "periods": period_results,
            "_note": (
                "T-01+T-02+T-03+T-09に、外部レビューT-06(permutation検定のブロック順列化、"
                "クラスタ=エントリー日)+T-07(実効n二重計上の解消、KPI評価スクリプト側で対応)"
                "を追加適用。perm_p_block(新)とperm_p_clustered(旧、比較用)を両方記録する。"
                "Testも参照するが、00-spec.md「Test凍結宣言」によりTrain+Validationのみで意思決定する。"
            ),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
