"""EXP-FX000005 改善ループ第3試行(4通貨版+BOJ/FOMCカレンダー)の取引明細・傾向分析.

司令塔依頼「第三試行をベースに取引明細と傾向分析をまとめて下さい」への対応。
`vol_breakout_dow_theory_4pairs_calendar_1000usd_backtest.json`(採用中の最良候補、
Train5/10・Validation2/10・Test6/10)のトレードデータを集計し、通貨別・方向別・
決済理由別・TP到達段階別・保有時間・月次分布・連続損益の傾向を出力する。

出力: research/method-notes/vol_breakout_third_trial_trade_ledger.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_calendar_1000usd_backtest.json").open(
    encoding="utf-8"
) as f:
    BACKTEST = json.load(f)


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


def find_streak_trades(df: pd.DataFrame, win: bool, length: int) -> list[dict]:
    """指定の長さの連勝/連敗が最初に現れる区間のトレードを返す(時系列順)."""
    df_sorted = df.sort_values("entry_time").reset_index(drop=True)
    cur_start = None
    cur_len = 0
    for i, is_win in enumerate(df_sorted["is_win"]):
        if is_win == win:
            if cur_start is None:
                cur_start = i
            cur_len += 1
            if cur_len == length:
                return df_sorted.iloc[cur_start:i + 1].to_dict("records")
        else:
            cur_start = None
            cur_len = 0
    return []


def main() -> int:
    print("=== EXP-FX000005 改善ループ第3試行 取引明細・傾向分析 ===\n")
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
    print(f"保有時間: 中央値={overall['hold_hours_median']}h  p75={overall['hold_hours_p75']}h  "
          f"最大={overall['hold_hours_max']}h\n")

    by_pair = group_stats(df, "pair")
    by_direction = group_stats(df, "direction")
    by_exit_reason = group_stats(df, "exit_reason")
    by_n_levels = group_stats(df, "n_levels_hit")
    by_period = group_stats(df, "period")

    print("--- 通貨別 ---")
    for r in by_pair:
        print(f"  {r['pair']}: n={r['n_trades']} win_rate={r['win_rate']} mean_r_net={r['mean_r_net']} "
              f"sum_pnl=${r['sum_dollar_pnl']}")
    print("\n--- 方向別 ---")
    for r in by_direction:
        print(f"  {r['direction']}: n={r['n_trades']} win_rate={r['win_rate']} mean_r_net={r['mean_r_net']} "
              f"sum_pnl=${r['sum_dollar_pnl']}")
    print("\n--- 決済理由別 ---")
    for r in by_exit_reason:
        print(f"  {r['exit_reason']}: n={r['n_trades']} win_rate={r['win_rate']} mean_r_net={r['mean_r_net']} "
              f"sum_pnl=${r['sum_dollar_pnl']}")
    print("\n--- TP到達段階別(n_levels_hit) ---")
    for r in by_n_levels:
        print(f"  {r['n_levels_hit']}段階到達: n={r['n_trades']} win_rate={r['win_rate']} mean_r_net={r['mean_r_net']}")

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

    # 保有時間 x 決済理由
    hold_by_reason = df.groupby("exit_reason")["hold_hours"].agg(["median", "max", "count"]).reset_index()
    hold_by_reason_records = [
        {"exit_reason": row["exit_reason"], "median_hours": round(float(row["median"]), 2),
         "max_hours": round(float(row["max"]), 2), "n": int(row["count"])}
        for _, row in hold_by_reason.iterrows()
    ]

    # 顕著なトレード抽出: 最初に現れる最大連敗クラスタ、最良の連続TP到達トレード
    worst_loss_streak_trades = find_streak_trades(df, False, overall["max_loss_streak"])
    best_win_streak_trades = find_streak_trades(df, True, min(overall["max_win_streak"], 8))

    # 最大単一利益トレード(TP_FULL, n_levels_hit=3)
    top_winners = df.sort_values("dollar_pnl", ascending=False).head(5)[
        ["period", "pair", "direction", "entry_time", "exit_time", "entry_price", "initial_risk",
         "exit_reason", "n_levels_hit", "r_net", "dollar_pnl", "hold_hours"]
    ].copy()
    top_winners["entry_time"] = top_winners["entry_time"].astype(str)
    top_winners["exit_time"] = top_winners["exit_time"].astype(str)

    def _clean(records):
        out = []
        for r in records:
            r = dict(r)
            r["entry_time"] = str(r["entry_time"])
            r["exit_time"] = str(r["exit_time"])
            out.append(r)
        return out

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_third_trial_trade_ledger.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "source": "vol_breakout_dow_theory_4pairs_calendar_1000usd_backtest.json(改善ループ第3試行、採用中の最良候補)",
            "overall": overall,
            "by_pair": by_pair,
            "by_direction": by_direction,
            "by_exit_reason": by_exit_reason,
            "by_n_levels_hit": by_n_levels,
            "by_period": by_period,
            "monthly": monthly_records,
            "hold_hours_by_exit_reason": hold_by_reason_records,
            "notable_trades": {
                "worst_loss_streak": _clean(worst_loss_streak_trades),
                "best_win_streak": _clean(best_win_streak_trades),
                "top_5_winners": top_winners.to_dict("records"),
            },
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
