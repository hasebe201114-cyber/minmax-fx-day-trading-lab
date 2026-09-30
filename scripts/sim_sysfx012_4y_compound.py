"""SYS-FX012 現行ルールの約4年通し複利シミュレーション（司令塔依頼 2026-09-30）。

期間: 2021-11-01〜2025-11-30（拡張Train＋Train＋Validation を1本の連続期間として実行）。
Test 期間（2025-12-01〜2026-08-15）は凍結ホールドアウトのため含めない（司令塔判断）。

ルール: SYS-FX012（N_BREAKOUT＋H1トレンド判定不能除外、+1R建値移動→ATR(M5)×0.703トレール、TPなし、
先読み修正済み）。パラメータの変更は一切しない。データ・コストは 41か月再評価（EXP-FX000020）と同じ
（2021-11〜2023-10 は Dukascopy M5、以降は DS-1。スプレッドは base_era_ratio、ストレスとして x2.0）。

資金管理は2通り（初期資金 $1,000、複利）:
  A 研究用の公式条件  … 1%リスク、1ポジション・合計レバレッジとも25倍（超過分は縮小）
  B 本番移行案(OBS000014) … 0.5%リスク、1ポジション・合計とも10倍、当日の確定損益が−3R以下なら当日の新規停止

出力: research/method-notes/sysfx012_4y_compound_simulation.json
"""
from __future__ import annotations

import copy
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import extended_data as ext  # noqa: E402
from minmax_fx_dt.backtest.money_management import settle_trades  # noqa: E402

START, END = "2021-11-01", "2025-11-30"
INITIAL = 1000.0
SEGMENTS = [("拡張期間", "2021-11-01", "2023-10-31"), ("Train", "2023-11-01", "2025-03-31"),
            ("Validation", "2025-04-01", "2025-11-30")]
SCENARIOS = {
    "A_公式(1%・25倍)": dict(risk_pct=0.01, per_cap=25.0, agg_cap=25.0, daily_stop_r=None),
    "B_本番案(0.5%・10倍・日次-3R停止)": dict(risk_pct=0.005, per_cap=10.0, agg_cap=10.0, daily_stop_r=-3.0),
}
OUT = ROOT / "research" / "method-notes" / "sysfx012_4y_compound_simulation.json"
CACHE_DIR = Path(__import__("tempfile").gettempdir())


def settle(trades: list[dict], risk_pct: float, per_cap: float, agg_cap: float, daily_stop_r: float | None):
    """money_management.settle_trades と同じ処理に、日次の新規停止を加えたもの。

    daily_stop_r=None のとき settle_trades と完全一致する（main で検証）。
    日付は取引時刻（JST、タイムゾーン表記なし）の暦日。確定損益は当日に決済した取引の r_net 合計。
    """
    events = []
    for i, t in enumerate(trades):
        events.append((t["entry_time"], 0, i, "ENTRY"))
        events.append((t["exit_time"], 1, i, "EXIT"))
    events.sort(key=lambda e: (e[0], e[1]))
    bal, open_notional, max_agg = INITIAL, 0.0, 0.0
    realized_r_by_day: dict = defaultdict(float)
    curve = [(pd.Timestamp(START), INITIAL)]
    for ts, _o, i, kind in events:
        t = trades[i]
        if kind == "ENTRY":
            t.update(skipped_cap=False, skipped_daily=False, skipped_ruin=False, notional_usd=0.0,
                     aggregate_cap_action="none")
            if bal <= 0:
                t.update(risk_dollars=0.0, skipped_ruin=True)
                continue
            if daily_stop_r is not None and realized_r_by_day[ts.date()] <= daily_stop_r:
                t.update(risk_dollars=0.0, skipped_daily=True)
                continue
            lev = t["leverage_ratio"]
            eff = min(risk_pct, per_cap / lev) if lev > 0 else risk_pct
            rd = bal * eff
            room = agg_cap * bal - open_notional
            if rd * lev > room + 1e-9:
                if room > 1e-9:
                    rd = room / lev
                    t["aggregate_cap_action"] = "shrink"
                else:
                    rd = 0.0
                    t.update(aggregate_cap_action="skip", skipped_cap=True)
            t.update(risk_dollars=rd, effective_risk_pct=rd / bal, notional_usd=rd * lev,
                     leverage_capped=eff < risk_pct)
            open_notional += t["notional_usd"]
            max_agg = max(max_agg, open_notional / bal)
        else:
            open_notional -= t.get("notional_usd", 0.0)
            if t["skipped_cap"] or t["skipped_daily"] or t["skipped_ruin"]:
                t["dollar_pnl"] = 0.0
            else:
                t["dollar_pnl"] = t["r_net"] * t["risk_dollars"]
                bal = max(0.0, bal + t["dollar_pnl"])
                realized_r_by_day[ts.date()] += t["r_net"]
            t["balance_after"] = bal
            curve.append((ts, bal))
    return bal, curve, max_agg


def active(t: dict) -> bool:
    return not (t.get("skipped_cap") or t.get("skipped_daily") or t.get("skipped_ruin"))


def max_drawdown(values: np.ndarray) -> tuple[float, int, int]:
    peak, mdd, pk_i, lo_i, best = values[0], 0.0, 0, 0, (0, 0)
    for i, v in enumerate(values):
        if v > peak:
            peak, pk_i = v, i
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > mdd:
            mdd, best = dd, (pk_i, i)
    return mdd, best[0], best[1]


def metrics(trades: list[dict], curve: list, final: float, max_agg: float) -> dict:
    act = [t for t in trades if active(t)]
    r = np.array([t["r_net"] for t in act])
    wins, losses = r[r > 0], r[r < 0]
    s = pd.Series([b for _, b in curve], index=pd.DatetimeIndex([c for c, _ in curve]))
    mdd, pi, li = max_drawdown(s.values)
    years = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
    month_end = s.resample("ME").last().ffill()
    monthly = month_end.pct_change()
    monthly.iloc[0] = month_end.iloc[0] / INITIAL - 1
    yearly = {}
    prev = INITIAL
    for y, v in s.resample("YE").last().ffill().items():
        yearly[str(y.year)] = {"end_balance": round(v, 2), "return_pct": round((v / prev - 1) * 100, 1)}
        prev = v
    seg = {}
    for name, a, b in SEGMENTS:
        before = s[s.index < pd.Timestamp(a)]
        start_bal = before.iloc[-1] if len(before) else INITIAL
        within = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b) + pd.Timedelta(days=1))]
        end_bal = within.iloc[-1] if len(within) else start_bal
        seg_act = [t for t in act if pd.Timestamp(a) <= t["entry_time"] <= pd.Timestamp(b) + pd.Timedelta(days=1)]
        seg[name] = {"period": f"{a}〜{b}", "start_balance": round(start_bal, 2), "end_balance": round(end_bal, 2),
                     "return_pct": round((end_bal / start_bal - 1) * 100, 1), "n_trades": len(seg_act),
                     "sum_r": round(sum(t["r_net"] for t in seg_act), 2)}
    # 撤退ライン（OBS000014）の判定（R基準・建てた取引のみ、決済順）
    act_sorted = sorted(act, key=lambda t: t["exit_time"])
    cum_r = np.cumsum([t["r_net"] for t in act_sorted])
    peak_r, max_dd_r, first_20r = 0.0, 0.0, None
    for t, c in zip(act_sorted, cum_r):
        peak_r = max(peak_r, c)
        if peak_r - c > max_dd_r:
            max_dd_r = peak_r - c
        if first_20r is None and peak_r - c >= 20:
            first_20r = str(t["exit_time"])
    r_month = pd.Series([t["r_net"] for t in act_sorted],
                        index=pd.DatetimeIndex([t["exit_time"] for t in act_sorted])).resample("ME").sum()
    roll6 = r_month.rolling(6).sum().dropna()
    neg6 = [str(i.date())[:7] for i, v in roll6.items() if v < 0]
    streak = best_streak = 0
    for x in r:
        streak = streak + 1 if x < 0 else 0
        best_streak = max(best_streak, streak)
    pair = {}
    for p in sorted({t["pair"] for t in act}):
        rp = np.array([t["r_net"] for t in act if t["pair"] == p])
        pair[p] = {"n": len(rp), "win_rate": round(float((rp > 0).mean()), 3), "sum_r": round(float(rp.sum()), 2),
                   "sum_usd": round(sum(t["dollar_pnl"] for t in act if t["pair"] == p), 2)}
    return {
        "final_balance": round(final, 2), "total_return_pct": round((final / INITIAL - 1) * 100, 1),
        "cagr_pct": round(((final / INITIAL) ** (1 / years) - 1) * 100, 1),
        "max_dd_pct": round(mdd * 100, 2), "max_dd_peak": str(s.index[pi]), "max_dd_trough": str(s.index[li]),
        "n_trades_total": len(trades), "n_trades_active": len(act),
        "n_skipped_cap": sum(1 for t in trades if t.get("skipped_cap")),
        "n_skipped_daily_stop": sum(1 for t in trades if t.get("skipped_daily")),
        "n_shrunk": sum(1 for t in trades if t.get("aggregate_cap_action") == "shrink"),
        "trades_per_month": round(len(act) / (years * 12), 1),
        "win_rate": round(float((r > 0).mean()), 4), "mean_r": round(float(r.mean()), 4),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 3),
        "payoff_ratio": round(float(wins.mean() / -losses.mean()), 3),
        "sum_r": round(float(r.sum()), 2), "max_win_r": round(float(r.max()), 2), "max_loss_r": round(float(r.min()), 2),
        "longest_losing_streak": best_streak,
        "monthly_sharpe_annualized": round(float(monthly.mean() / monthly.std() * np.sqrt(12)), 3),
        "months": len(monthly), "positive_months": int((monthly > 0).sum()),
        "worst_month_pct": round(float(monthly.min() * 100), 2), "worst_month": str(monthly.idxmin().date())[:7],
        "best_month_pct": round(float(monthly.max() * 100), 2),
        "max_aggregate_leverage": round(max_agg, 2),
        "yearly": yearly, "segments": seg, "by_pair": pair,
        "withdrawal_lines": {
            "max_dd_r": round(max_dd_r, 2), "dd_20r_first_hit": first_20r,
            "rolling_6m_negative_windows_end_month": neg6,
            "trades_worse_than_minus_3r": int((r < -3).sum()),
        },
        "monthly_returns_pct": {str(k.date())[:7]: round(v * 100, 2) for k, v in monthly.items()},
    }


def main() -> int:
    import backtest_sysfx018_breakeven_sweep_trainonly as fx018

    out: dict = {"generated_at": datetime.now().isoformat(), "period": {"start": START, "end": END},
                 "note": "Test期間(2025-12-01〜2026-08-15)は凍結ホールドアウトのため含めない。パラメータ変更なし。",
                 "scenarios": {k: v for k, v in SCENARIOS.items()}, "results": {}}
    for level in ("base_era_ratio", "sensitivity_x2.0"):
        spreads = ext.patch_pipelines(level)
        print(f"=== コスト {level} {spreads} ===", flush=True)
        cache = CACHE_DIR / f"sysfx012_4y_raw_{level}.pkl"
        if cache.exists():
            raw, official_final = pd.read_pickle(cache)
        else:
            p = fx018.run_period(1.0, START, END)
            raw = [{k: t[k] for k in ("pair", "direction", "entry_time", "exit_time", "exit_reason", "r_gross",
                                      "r_net", "leverage_ratio")} for t in p["trades"]]
            official_final = float(p["equity_curve"][-1]["balance"])
            pd.to_pickle((raw, official_final), cache)
        def to_jst_naive(x):  # 取引時刻は JST(+09:00)。資産曲線の起点と揃えるためタイムゾーン表記を外す
            ts = pd.Timestamp(x)
            return ts.tz_convert("Asia/Tokyo").tz_localize(None) if ts.tzinfo else ts
        for t in raw:
            t["entry_time"], t["exit_time"] = to_jst_naive(t["entry_time"]), to_jst_naive(t["exit_time"])
        # 検証: 日次停止なしの自作処理が公式の settle_trades と一致すること
        a, b = copy.deepcopy(raw), copy.deepcopy(raw)
        bal_official, _, _ = settle_trades(a, initial_capital=INITIAL, risk_pct=0.01, start_label=START)
        bal_mine, _, _ = settle(b, 0.01, 25.0, 25.0, None)
        assert abs(bal_official - bal_mine) < 1e-6, (bal_official, bal_mine)
        assert abs(bal_official / official_final - 1) < 1e-6  # 出力側でr_net等が小数6桁に丸められているため相対誤差で比較
        res = {"spread_pips": spreads, "n_raw_trades": len(raw)}
        for name, sc in SCENARIOS.items():
            tr = copy.deepcopy(raw)
            final, curve, max_agg = settle(tr, sc["risk_pct"], sc["per_cap"], sc["agg_cap"], sc["daily_stop_r"])
            m = metrics(tr, curve, final, max_agg)
            res[name] = m
            print(f"  {name}: ${m['final_balance']:,} ({m['total_return_pct']}%, CAGR {m['cagr_pct']}%) "
                  f"DD {m['max_dd_pct']}% n={m['n_trades_active']} 勝率{m['win_rate']} PF{m['profit_factor']} "
                  f"Sharpe{m['monthly_sharpe_annualized']} 撤退DD_R={m['withdrawal_lines']['max_dd_r']}", flush=True)
        out["results"][level] = res
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[出力] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
