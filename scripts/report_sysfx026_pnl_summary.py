"""SYS-FX026 の3期間（Train/Validation/Test）収支レポート生成.

司令塔依頼（2026-08-28）「バックテスト結果、収支が分かるようにまとめて下さい」。

## 本スクリプトの位置づけ（重要）

**新しい検証ではない。** `00-spec-amendment-01/02` で既に確定済みの結果を、
損益（ドル建て）の観点から読めるように集計し直すだけの**レポート生成**である。

パラメータは凍結値をそのまま使うため、実行するたびに同じ数値が出る決定論的な処理であり、
Test を「もう一度引く」ことにはあたらない（amendment-02 §5 の一度限り原則は
「結果を見た後にパラメータを変えて引き直すこと」の禁止であり、同一構成の再集計は該当しない）。

## 出力
- research/method-notes/sysfx026_pnl_summary.json  … 機械可読な全数値
- 標準出力に人間可読のサマリ表
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd as v7  # noqa: E402
from backtest_sysfx026_sizing_trial1 import RISK_PCT_TRIAL1, TRAIL_MULT_FACTOR  # noqa: E402

INITIAL_CAPITAL = 1000.0


def run_frozen(period_name: str) -> dict:
    """凍結構成（trail 3.0x・risk 0.65%）で1期間を実行する."""
    orig_trail, orig_risk = v7.ATR_TRAIL_MULTIPLIER_M5, v7.RISK_PCT_PER_TRADE
    v7.ATR_TRAIL_MULTIPLIER_M5 = v7.STOP_BUFFER_ATR_M5 * TRAIL_MULT_FACTOR
    v7.RISK_PCT_PER_TRADE = RISK_PCT_TRIAL1
    try:
        start, end = v7.PERIODS[period_name]
        return v7.run_period(period_name, start, end)
    finally:
        v7.ATR_TRAIL_MULTIPLIER_M5, v7.RISK_PCT_PER_TRADE = orig_trail, orig_risk


def summarize(period_name: str, p: dict) -> dict:
    trades = p["trades"]
    start, end = v7.PERIODS[period_name]
    months = (pd.Timestamp(end) - pd.Timestamp(start)).days / 30.44

    pnl = np.array([float(t["dollar_pnl"]) for t in trades], dtype=float)
    r_net = np.array([float(t["r_net"]) for t in trades], dtype=float)
    r_gross = np.array([float(t["r_gross"]) for t in trades], dtype=float)
    cost_r = np.array([float(t["cost_r"]) + float(t["commission_r"]) for t in trades], dtype=float)
    risk_d = np.array([float(t["risk_dollars"]) for t in trades], dtype=float)

    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_profit, gross_loss = float(wins.sum()), float(abs(losses.sum()))
    # コストを金額へ換算（R単位のコスト × そのトレードのリスク額）
    cost_dollars = float((cost_r * risk_d).sum())

    # 月次収支
    monthly = defaultdict(float)
    for t in trades:
        monthly[pd.Timestamp(t["exit_time"]).strftime("%Y-%m")] += float(t["dollar_pnl"])
    monthly = dict(sorted(monthly.items()))
    monthly_vals = np.array(list(monthly.values()), dtype=float)

    # 通貨別収支
    by_pair = {}
    for pair in sorted({t["pair"] for t in trades}):
        tp = [t for t in trades if t["pair"] == pair]
        ppnl = np.array([float(x["dollar_pnl"]) for x in tp], dtype=float)
        by_pair[pair] = {
            "n": len(tp),
            "net_pnl_usd": round(float(ppnl.sum()), 2),
            "win_rate": round(float((ppnl > 0).mean()), 4),
            "mean_r_net": round(float(np.mean([float(x["r_net"]) for x in tp])), 4),
        }

    # 決済理由別
    by_exit = {}
    for reason in sorted({t["exit_reason"] for t in trades}):
        tr = [t for t in trades if t["exit_reason"] == reason]
        rpnl = np.array([float(x["dollar_pnl"]) for x in tr], dtype=float)
        by_exit[reason] = {"n": len(tr), "net_pnl_usd": round(float(rpnl.sum()), 2),
                           "mean_pnl_usd": round(float(rpnl.mean()), 2)}

    # 損益集中度（頑健性チェック）: 上位数件のトレードに利益が偏っていないか。
    # 少数の大当たりに依存する戦略は、その数件を引けなかっただけで成績が崩れる。
    sorted_pnl = np.sort(pnl)[::-1]
    concentration = {}
    for k in (1, 3, 5):
        if len(sorted_pnl) >= k:
            topk = float(sorted_pnl[:k].sum())
            concentration[f"top{k}"] = {
                "sum_usd": round(topk, 2),
                "pct_of_gross_profit": round(topk / gross_profit * 100, 1) if gross_profit else None,
                "pct_of_net_pnl": round(topk / float(pnl.sum()) * 100, 1) if pnl.sum() else None,
            }
    # 上位5件を除いたときに純損益が正のままか
    if len(sorted_pnl) > 5:
        ex_top5 = float(sorted_pnl[5:].sum())
        concentration["net_pnl_excl_top5_usd"] = round(ex_top5, 2)
        concentration["still_profitable_excl_top5"] = bool(ex_top5 > 0)

    # エクイティカーブからピーク比DD
    eq = [float(e["balance"]) for e in p["equity_curve"]]
    peak, max_dd = INITIAL_CAPITAL, 0.0
    for b in eq:
        peak = max(peak, b)
        max_dd = max(max_dd, (peak - b) / peak * 100)

    return {
        "period": period_name,
        "start": start, "end": end, "months": round(months, 1),
        "n_trades": len(trades),
        "initial_capital_usd": INITIAL_CAPITAL,
        "final_balance_usd": p["final_balance_usd"],
        "net_pnl_usd": round(float(pnl.sum()), 2),
        "total_return_pct": p["total_return_pct"],
        "monthly_return_pct_avg": round(float(p["total_return_pct"]) / months, 2),
        "max_dd_peak_relative_pct": round(max_dd, 2),
        "gross_profit_usd": round(gross_profit, 2),
        "gross_loss_usd": round(gross_loss, 2),
        "total_cost_usd": round(cost_dollars, 2),
        "cost_pct_of_gross_profit": round(cost_dollars / gross_profit * 100, 1) if gross_profit else None,
        "win_rate": p["win_rate"],
        "n_wins": int((pnl > 0).sum()), "n_losses": int((pnl < 0).sum()),
        "avg_win_usd": round(float(wins.mean()), 2) if wins.size else None,
        "avg_loss_usd": round(float(losses.mean()), 2) if losses.size else None,
        "largest_win_usd": round(float(pnl.max()), 2),
        "largest_loss_usd": round(float(pnl.min()), 2),
        "profit_factor": p["profit_factor"],
        "payoff_ratio": p["payoff_ratio"],
        "mean_r_net": p["mean_r_net"],
        "mean_r_gross": round(float(r_gross.mean()), 4),
        "mean_cost_r": round(float(cost_r.mean()), 4),
        "k5m_spread_cost_multiplier": round(float(r_gross.mean() / cost_r.mean()), 3),
        "monthly_pnl_usd": {k: round(v, 2) for k, v in monthly.items()},
        "n_months_positive": int((monthly_vals > 0).sum()),
        "n_months_negative": int((monthly_vals < 0).sum()),
        "worst_month_usd": round(float(monthly_vals.min()), 2) if monthly_vals.size else None,
        "best_month_usd": round(float(monthly_vals.max()), 2) if monthly_vals.size else None,
        "pnl_concentration": concentration,
        "by_pair": by_pair,
        "by_exit_reason": by_exit,
        "perm_p_block": p["perm_p_block"],
        "leverage_ratio_stats": p["leverage_ratio_stats"],
        "n_leverage_capped": p["n_leverage_capped"],
    }


def main() -> None:
    print("=" * 92)
    print("SYS-FX026 収支レポート（Train / Validation / Test、凍結構成）")
    print(f"  trail = stop_buffer × {TRAIL_MULT_FACTOR} = {v7.STOP_BUFFER_ATR_M5 * TRAIL_MULT_FACTOR:.4f}")
    print(f"  risk_pct = {RISK_PCT_TRIAL1 * 100:.2f}% / 初期資金 ${INITIAL_CAPITAL:,.0f} / 複利")
    print("=" * 92)

    results = {}
    for name in ("train", "validation", "test"):
        p = run_frozen(name)
        results[name] = summarize(name, p)

    print("\n" + "=" * 92)
    print("【収支サマリ】")
    print("=" * 92)
    hdr = f"{'':<22}{'Train':>18}{'Validation':>18}{'Test':>18}"
    print(hdr)
    rows = [
        ("期間", "start_end"), ("期間(月)", "months"), ("トレード数", "n_trades"),
        ("初期資金", "initial_capital_usd"), ("最終残高", "final_balance_usd"),
        ("純損益", "net_pnl_usd"), ("総リターン%", "total_return_pct"),
        ("月平均リターン%", "monthly_return_pct_avg"), ("最大DD%(ピーク比)", "max_dd_peak_relative_pct"),
        ("総利益", "gross_profit_usd"), ("総損失", "gross_loss_usd"), ("総コスト", "total_cost_usd"),
        ("コスト/総利益%", "cost_pct_of_gross_profit"),
        ("勝率", "win_rate"), ("平均利益", "avg_win_usd"), ("平均損失", "avg_loss_usd"),
        ("最大利益", "largest_win_usd"), ("最大損失", "largest_loss_usd"),
        ("PF", "profit_factor"), ("ペイオフ", "payoff_ratio"),
        ("平均r_net", "mean_r_net"), ("K5m", "k5m_spread_cost_multiplier"),
        ("prm p(日BL)", "perm_p_block"),
        ("月次+/-", "months_pos_neg"), ("最良月", "best_month_usd"), ("最悪月", "worst_month_usd"),
    ]
    for label, key in rows:
        cells = []
        for name in ("train", "validation", "test"):
            r = results[name]
            if key == "start_end":
                v = f"{r['start'][2:]}〜{r['end'][2:]}"
            elif key == "months_pos_neg":
                v = f"{r['n_months_positive']}+ / {r['n_months_negative']}-"
            else:
                v = r.get(key)
            cells.append(f"{str(v):>18}")
        print(f"{label:<22}" + "".join(cells))

    print("\n" + "=" * 92)
    print("【通貨別 純損益 (USD)】")
    print("=" * 92)
    all_pairs = sorted({p for r in results.values() for p in r["by_pair"]})
    print(f"{'':<12}{'Train':>26}{'Validation':>26}{'Test':>26}")
    for pair in all_pairs:
        cells = []
        for name in ("train", "validation", "test"):
            d = results[name]["by_pair"].get(pair)
            if d:
                text = "${:,.0f} (n={})".format(d["net_pnl_usd"], d["n"])
            else:
                text = "-"
            cells.append("{:>26}".format(text))
        print(f"{pair:<12}" + "".join(cells))

    print("\n" + "=" * 92)
    print("【月次収支 (USD)】")
    print("=" * 92)
    for name in ("train", "validation", "test"):
        r = results[name]
        print(f"\n-- {name} ({r['start']}〜{r['end']}) --")
        for i, (m, v) in enumerate(r["monthly_pnl_usd"].items()):
            end = "\n" if (i + 1) % 6 == 0 else "  "
            print(f"{m}: {v:>10,.2f}", end=end)
        print()

    print("\n" + "=" * 92)
    print("【損益集中度（少数の大当たりに依存していないか）】")
    print("=" * 92)
    for name in ("train", "validation", "test"):
        c = results[name]["pnl_concentration"]
        print(f"\n-- {name} --")
        for k in ("top1", "top3", "top5"):
            if k in c:
                print(f"  {k:<6} 合計=${c[k]['sum_usd']:>9,.2f}  総利益の{c[k]['pct_of_gross_profit']:>5}%  "
                      f"純損益の{c[k]['pct_of_net_pnl']:>6}%")
        if "net_pnl_excl_top5_usd" in c:
            ok = "OK(なお黒字)" if c["still_profitable_excl_top5"] else "NG(赤字転落)"
            print(f"  上位5件を除いた純損益 = ${c['net_pnl_excl_top5_usd']:>9,.2f}   {ok}")

    print("\n" + "=" * 92)
    print("【決済理由別】")
    print("=" * 92)
    for name in ("train", "validation", "test"):
        r = results[name]
        print(f"\n-- {name} --")
        for reason, d in sorted(r["by_exit_reason"].items(), key=lambda x: x[1]["net_pnl_usd"]):
            print(f"  {reason:<24} n={d['n']:>4}  純損益=${d['net_pnl_usd']:>10,.2f}  平均=${d['mean_pnl_usd']:>8,.2f}")

    out = {
        "purpose": "SYS-FX026 の3期間収支レポート（新しい検証ではなく既確定結果の再集計）",
        "params": {"trail_mult_factor": TRAIL_MULT_FACTOR,
                   "atr_trail_multiplier_m5": v7.STOP_BUFFER_ATR_M5 * TRAIL_MULT_FACTOR,
                   "risk_pct_per_trade": RISK_PCT_TRIAL1,
                   "initial_capital_usd": INITIAL_CAPITAL,
                   "sizing": "複利（その時点の残高 × risk_pct をリスクに晒す）"},
        "periods": results,
    }
    out_path = ROOT / "research" / "method-notes" / "sysfx026_pnl_summary.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n出力: {out_path}")


if __name__ == "__main__":
    main()
