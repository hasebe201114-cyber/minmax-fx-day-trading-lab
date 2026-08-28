"""コスト構造改善(方向性a)の一次診断: ATRトレール幅を広げるとK5mに余地があるか.

## 位置づけ

**正式プロトコル外の探索的診断**（OBS000011の`explore_grid_strategy_sanity_check.py`と
同じ位置づけ）。まだSYS番号・EXP番号は採番していない、着手前の一次診断。Train期間
（現行GMOデータ、2023-11-01〜2025-03-31）のみを使用し、Validation/Testには一切触れない。

## 仮説（結果を見る前に記録）

K5m = mean(r_gross) / mean(cost_r + commission_r) は、`initial_risk`（=1R、
stop_buffer_atr_m5 × ATR(M5)）のスケーリングでは動かない（r_gross・cost_rの両方が
同じ1/initial_riskで割られるため、比率としては打ち消し合う）。実質的に
K5m ≈ mean(価格変動幅) / mean(固定pipsコスト) であり、これを改善する唯一の筋の良い
レバーは「勝ちトレードが実際にどれだけ長くトレンドを追えるか」＝ATRトレール幅の設計である。

SYS-FX011の現行トレール幅（atr_trail_multiplier_m5 = stop_buffer_atr_m5 × 1.0）は
損切り幅と同じ1R相当と非常にタイトで、拡張Trainのペイオフレシオが1.07（ほぼ1:1）と
「トレンドフォロー型の出口設計」にしては低い。これはトレールが早く効きすぎて、
勝ちトレードでもトレンドを十分に追えていない可能性を示唆する。

トレール幅を1R→1.5R/2.0R/3.0Rへ広げた場合、勝ちトレードの平均値幅（r_gross）が
伸びてK5m・ペイオフレシオが改善するか、それとも早すぎる利益確定を失うだけで
Sharpe/PFが悪化するかを、Train単独でスイープして確認する。

## 出力
research/method-notes/trail_multiplier_headroom_diagnostic.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd as v7  # noqa: E402
from minmax_fx_dt.backtest.metrics import payoff_ratio, profit_factor  # noqa: E402


def simple_monthly_sharpe_r(monthly_r_sums: list[float]) -> float | None:
    """月次R合計の平均/標準偏差(簡易版、診断専用)。equity曲線を必要とする
    正式版`minmax_fx_dt.backtest.metrics.monthly_sharpe`とは入力形式が異なる
    ため、本診断ではRベースの簡易指標として代用する(正式KPI判定には使わない)。"""
    arr = np.array(monthly_r_sums, dtype=float)
    if len(arr) < 2 or arr.std(ddof=1) == 0:
        return None
    return float(arr.mean() / arr.std(ddof=1))

TRAIN_START, TRAIN_END = v7.PERIODS["train"]
STOP_BUFFER_ATR_M5 = v7.STOP_BUFFER_ATR_M5

# 事前登録する候補（結果を見る前に固定）: 現行1.0倍を含め4水準
TRAIL_CANDIDATES = [1.0, 1.5, 2.0, 3.0]


def run_candidate(trail_mult_factor: float) -> dict:
    """trail_mult_factor × stop_buffer_atr_m5 をトレール幅としてTrainを再評価する.

    `find_trades_for_period`はモジュールレベル定数ATR_TRAIL_MULTIPLIER_M5を参照するため、
    呼び出し前に一時的に上書きする(診断専用、正式パイプラインのファイルは変更しない)。
    """
    original = v7.ATR_TRAIL_MULTIPLIER_M5
    v7.ATR_TRAIL_MULTIPLIER_M5 = STOP_BUFFER_ATR_M5 * trail_mult_factor
    try:
        m5_by_pair, h1_by_pair, atr_h1_by_pair = {}, {}, {}
        for pair in v7.SELECTED_PAIRS:
            m5 = v7.load_m5_period(pair, TRAIN_START, TRAIN_END)
            if len(m5) < 1000:
                continue
            m5_by_pair[pair] = m5
            h1_by_pair[pair] = v7.to_h1(m5)
            atr_h1_by_pair[pair] = v7.atr_ind(h1_by_pair[pair]["high"], h1_by_pair[pair]["low"],
                                               h1_by_pair[pair]["close"], length=14)
        shock_check = v7.make_price_shock_check(h1_by_pair, atr_h1_by_pair)

        all_trades: list[dict] = []
        for pair, m5 in m5_by_pair.items():
            trades, _, _ = v7.find_trades_for_period(pair, m5, shock_check)
            spread = v7.SPREAD_PIPS.get(pair, 0.5)
            pip = v7.pip_size(pair)
            for sim in trades:
                entry_pips = spread + v7.SLIPPAGE_PIPS_MARKET_LEG
                remaining_is_market = sim["exit_reason"] in ("WEEKEND_NO_TP", "TP_THEN_WEEKEND", "MAX_HOLD")
                remaining_is_stop_triggered = sim["exit_reason"] in ("SL_INITIAL_NO_TP", "TP_THEN_SL_TRAIL")
                if remaining_is_market:
                    exit_slippage = v7.SLIPPAGE_PIPS_MARKET_LEG
                elif remaining_is_stop_triggered:
                    exit_slippage = v7.SLIPPAGE_PIPS_STOP_TRIGGERED
                else:
                    exit_slippage = 0.0
                exit_pips = spread + exit_slippage
                cost_price = (entry_pips + exit_pips) * pip
                cost_r = cost_price / sim["initial_risk"]
                leverage_ratio = sim["entry_price"] / sim["initial_risk"]
                commission_r = v7.COMMISSION_RATE_ROUND_TRIP * leverage_ratio
                r_net = sim["r"] - cost_r - commission_r
                all_trades.append({
                    "pair": pair, "r_gross": sim["r"], "cost_r": cost_r,
                    "commission_r": commission_r, "r_net": r_net,
                    "entry_time": pd.Timestamp(sim["entry_time"]),
                })
    finally:
        v7.ATR_TRAIL_MULTIPLIER_M5 = original

    if not all_trades:
        return {"trail_mult_factor": trail_mult_factor, "n_trades": 0}

    r_gross = np.array([t["r_gross"] for t in all_trades])
    cost = np.array([t["cost_r"] + t["commission_r"] for t in all_trades])
    r_net = np.array([t["r_net"] for t in all_trades])
    wins = r_net > 0
    df = pd.DataFrame(all_trades).sort_values("entry_time")
    monthly_r = df.set_index("entry_time")["r_net"].resample("ME").sum()

    return {
        "trail_mult_factor": trail_mult_factor,
        "trail_width_price_note": "stop_buffer_atr_m5 × factor",
        "n_trades": len(all_trades),
        "win_rate": round(float(wins.mean()), 4),
        "mean_r_gross": round(float(r_gross.mean()), 4),
        "mean_cost_r": round(float(cost.mean()), 4),
        "spread_cost_multiplier_k5m": round(float(r_gross.mean() / cost.mean()), 3),
        "payoff_ratio": round(float(payoff_ratio(r_net.tolist())), 4) if wins.any() and (~wins).any() else None,
        "profit_factor_r": round(float(profit_factor(r_net.tolist())), 4),
        "mean_r_net": round(float(r_net.mean()), 4),
        "monthly_sharpe_r": round(simple_monthly_sharpe_r(monthly_r.tolist()), 4)
        if simple_monthly_sharpe_r(monthly_r.tolist()) is not None else None,
        "total_r_net": round(float(r_net.sum()), 2),
    }


def main() -> None:
    print(f"=== コスト構造改善 一次診断: ATRトレール幅スイープ (Train {TRAIN_START}〜{TRAIN_END}のみ) ===")
    print(f"stop_buffer_atr_m5={STOP_BUFFER_ATR_M5}, 候補={TRAIL_CANDIDATES}\n")

    results = []
    for factor in TRAIL_CANDIDATES:
        r = run_candidate(factor)
        results.append(r)
        print(f"  factor={factor}x  n={r.get('n_trades')}  win_rate={r.get('win_rate')}  "
              f"K5m={r.get('spread_cost_multiplier_k5m')}  payoff={r.get('payoff_ratio')}  "
              f"PF(R)={r.get('profit_factor_r')}  mean_r_net={r.get('mean_r_net')}  "
              f"monthly_sharpe(R)={r.get('monthly_sharpe_r')}")

    out = {
        "purpose": "コスト構造改善(方向性a)の一次診断。正式プロトコル外、Train単独、"
                   "K4m/K5mへの感度をトレール幅のみで確認する",
        "period": {"start": TRAIN_START, "end": TRAIN_END},
        "stop_buffer_atr_m5": STOP_BUFFER_ATR_M5,
        "candidates": TRAIL_CANDIDATES,
        "results": results,
    }
    out_path = ROOT / "research" / "method-notes" / "trail_multiplier_headroom_diagnostic.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n出力: {out_path}")


if __name__ == "__main__":
    main()
