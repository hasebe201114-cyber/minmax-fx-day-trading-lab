"""EXP-FX000005 改善ループ第7試行(価格反応型ショック抑制フィルター、採用中の
最良候補)の取引明細・傾向分析. 年率換算リターンを参考値として追加。

出力: research/method-notes/vol_breakout_v7_trade_ledger.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_v7_1000usd_backtest.json").open(
    encoding="utf-8"
) as f:
    BACKTEST = json.load(f)

INITIAL_CAPITAL = BACKTEST["design"]["initial_capital_usd"]


def load_all_trades() -> pd.DataFrame:
    rows = []
    for period_name, p in BACKTEST["periods"].items():
        for t in p["trades"]:
            row = dict(t)
            row["period"] = period_name
            rows.append(row)
    df = pd.DataFrame(rows)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["hold_hours"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600.0
    df["is_win"] = df["dollar_pnl"] > 0
    df["month"] = df["entry_time"].dt.to_period("M").astype(str)
    return df


def group_stats(df: pd.DataFrame, by: str) -> list[dict]:
    out = []
    for key, g in df.groupby(by):
        out.append({
            by: key,
            "n_trades": len(g),
            "win_rate": round(float(g["is_win"].mean()), 3),
            "mean_r_net": round(float(g["r_net"].mean()), 4),
            "mean_r_gross": round(float(g["r_gross"].mean()), 4),
            "sum_dollar_pnl": round(float(g["dollar_pnl"].sum()), 2),
            "mean_dollar_pnl": round(float(g["dollar_pnl"].mean()), 2),
        })
    return sorted(out, key=lambda r: -r["n_trades"])


def max_streak(df: pd.DataFrame, win: bool) -> int:
    df_sorted = df.sort_values("entry_time")
    worst = cur = 0
    for is_win in df_sorted["is_win"]:
        if is_win == win:
            cur += 1
            worst = max(worst, cur)
        else:
            cur = 0
    return worst


def annualized_return(final_balance: float, start: str, end: str) -> float:
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    years = days / 365.25
    total_return = final_balance / INITIAL_CAPITAL
    return (total_return ** (1.0 / years) - 1.0) * 100.0, years


def main() -> int:
    print("=== EXP-FX000005 改善ループ第7試行 取引明細・傾向分析(年率参考値付き) ===\n")
    df = load_all_trades()
    print(f"全トレード数: {len(df)}件 (Train/Validation/Test)")

    overall = {
        "n_trades": len(df),
        "win_rate": round(float(df["is_win"].mean()), 3),
        "mean_r_net": round(float(df["r_net"].mean()), 4),
        "mean_r_gross": round(float(df["r_gross"].mean()), 4),
        "sum_dollar_pnl": round(float(df["dollar_pnl"].sum()), 2),
        "profit_factor": round(float(df.loc[df["dollar_pnl"] > 0, "dollar_pnl"].sum() /
                                      abs(df.loc[df["dollar_pnl"] < 0, "dollar_pnl"].sum())), 3),
        "max_win_streak": max_streak(df, True),
        "max_loss_streak": max_streak(df, False),
        "hold_hours_median": round(float(df["hold_hours"].median()), 2),
        "hold_hours_p75": round(float(df["hold_hours"].quantile(0.75)), 2),
        "hold_hours_max": round(float(df["hold_hours"].max()), 2),
    }
    print(f"勝率={overall['win_rate']}  平均r_net={overall['mean_r_net']}  PF={overall['profit_factor']}  "
          f"最大連勝={overall['max_win_streak']}  最大連敗={overall['max_loss_streak']}")

    by_pair = group_stats(df, "pair")
    by_direction = group_stats(df, "direction")
    by_exit_reason = group_stats(df, "exit_reason")
    by_n_levels = group_stats(df, "n_levels_hit")

    print("\n--- 通貨別 ---")
    for r in by_pair:
        print(f"  {r['pair']}: n={r['n_trades']} win_rate={r['win_rate']} mean_r_net={r['mean_r_net']} sum_pnl=${r['sum_dollar_pnl']}")
    print("\n--- 方向別 ---")
    for r in by_direction:
        print(f"  {r['direction']}: n={r['n_trades']} win_rate={r['win_rate']} mean_r_net={r['mean_r_net']} sum_pnl=${r['sum_dollar_pnl']}")
    print("\n--- 決済理由別 ---")
    for r in by_exit_reason:
        print(f"  {r['exit_reason']}: n={r['n_trades']} win_rate={r['win_rate']} mean_r_net={r['mean_r_net']} sum_pnl=${r['sum_dollar_pnl']}")

    # 期間別: 総リターン・年率換算リターン
    period_returns = {}
    print("\n--- 期間別リターン(年率は参考値) ---")
    for period_name, p in BACKTEST["periods"].items():
        ann, years = annualized_return(p["final_balance_usd"], p["start"], p["end"])
        period_returns[period_name] = {
            "start": p["start"], "end": p["end"], "years": round(years, 3),
            "n_trades": p["n_trades"],
            "final_balance_usd": p["final_balance_usd"],
            "total_return_pct": p["total_return_pct"],
            "annualized_return_pct": round(ann, 2),
            "max_drawdown_pct": p["max_drawdown_pct"],
        }
        print(f"  {period_name}({p['start']}〜{p['end']}, {years:.2f}年): "
              f"総リターン={p['total_return_pct']}%  年率換算={ann:.1f}%  最大DD={p['max_drawdown_pct']}%")

    # 月次分布
    monthly = df.groupby("month").agg(
        n_trades=("dollar_pnl", "count"),
        sum_dollar_pnl=("dollar_pnl", "sum"),
        win_rate=("is_win", "mean"),
    ).reset_index()
    monthly_records = [
        {"month": row["month"], "n_trades": int(row["n_trades"]),
         "sum_dollar_pnl": round(float(row["sum_dollar_pnl"]), 2), "win_rate": round(float(row["win_rate"]), 3)}
        for _, row in monthly.iterrows()
    ]

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_v7_trade_ledger.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "source": "vol_breakout_dow_theory_4pairs_v7_1000usd_backtest.json(改善ループ第7試行、採用中の最良候補)",
            "overall": overall,
            "by_pair": by_pair,
            "by_direction": by_direction,
            "by_exit_reason": by_exit_reason,
            "by_n_levels_hit": by_n_levels,
            "period_returns": period_returns,
            "monthly": monthly_records,
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
