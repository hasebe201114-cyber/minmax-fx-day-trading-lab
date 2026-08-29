"""OBS000009 不具合1（探索窓起点の先読み）が SYS-FX026 に与える影響の検算.

## 背景

C品質チーム査読（2026-08-28）が「SYS-FX026 の3期間すべてに OBS000009 不具合1 の
先読みが載っている」と指摘した。本スクリプトはその指摘を独立に検算する。

## 先読みの機序（コードとpandas仕様から確定）

- `derive_vol_breakout_entry_params.to_h1()` は `resample("1h")` を label 既定 (=left) で使う
  → **H1インデックスはバーの「始値時刻」**
- `backtest_vol_breakout_dow_theory.py`:
    break_time = h1.index[break_idx]          # = バー始値時刻
    start_time = break_time + 30分            # = バー確定の30分「前」
- ブレイク判定自体は (high-low)/ATR というバー**確定後**にしか分からない値を使う
- したがって「まだ確定していないH1バーの高値・安値・終値」を使ってエントリーしうる

`00-spec.md` は「ブレイクバー**確定後**30分」と明記しており、実装は仕様と食い違う。

## 本スクリプトがすること

各トレードに生成元のブレイクバー始値時刻を紐づけ、
`entry_time - break_time < 60分` なら「バー確定前エントリー（先読み該当）」と判定する。
該当トレードを除外した場合の主判定4条件（amendment-02 §4）を再計算する。

**注意**: これは「Testを引き直す」ことではない。既に消費済みのTest結果に対する
バグ診断であり、パラメータは一切変更していない。除外は正しい修正の近似であって、
本来は窓起点を +1h して全期間を再実行する必要がある（ポジション占有のゲーティングが
変わるため別のトレード集合になる）。

出力: research/method-notes/sysfx026_lookahead_impact.json
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

import backtest_vol_breakout_dow_theory as bd  # noqa: E402
import backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd as v7  # noqa: E402
from backtest_sysfx026_sizing_trial1 import RISK_PCT_TRIAL1, TRAIL_MULT_FACTOR  # noqa: E402
from minmax_fx_dt.backtest.permutation import permutation_test_block  # noqa: E402

BAR_MINUTES = 60  # H1


def run_instrumented(period_name: str) -> list[dict]:
    """各トレードに break_time を紐づけて1期間を実行する.

    `run_period()` はトレード辞書を固定キーで再構築するため、注入したキーは落ちる。
    そこで (entry_time, entry_price, direction) をキーにした対応表を別途作り、後で突き合わせる。
    """
    orig_sim = bd.simulate_dow_theory_trend
    orig_trail, orig_risk = v7.ATR_TRAIL_MULTIPLIER_M5, v7.RISK_PCT_PER_TRADE
    v7.ATR_TRAIL_MULTIPLIER_M5 = v7.STOP_BUFFER_ATR_M5 * TRAIL_MULT_FACTOR
    v7.RISK_PCT_PER_TRADE = RISK_PCT_TRIAL1
    break_map: dict[tuple, pd.Timestamp] = {}

    def wrapped(m5, atr_m5, h1, atr_h1, break_idx, direction, *a, **kw):
        out = orig_sim(m5, atr_m5, h1, atr_h1, break_idx, direction, *a, **kw)
        bt = h1.index[break_idx]
        for t in out:
            break_map[(str(t["entry_time"]), round(float(t["entry_price"]), 6), t["direction"])] = bt
        return out

    # v7 は from-import しているので v7 側の名前も差し替える
    bd.simulate_dow_theory_trend = wrapped
    v7.simulate_dow_theory_trend = wrapped
    try:
        start, end = v7.PERIODS[period_name]
        p = v7.run_period(period_name, start, end)
    finally:
        bd.simulate_dow_theory_trend = orig_sim
        v7.simulate_dow_theory_trend = orig_sim
        v7.ATR_TRAIL_MULTIPLIER_M5, v7.RISK_PCT_PER_TRADE = orig_trail, orig_risk

    trades = p["trades"]
    missing = 0
    for t in trades:
        key = (str(t["entry_time"]), round(float(t["entry_price"]), 6), t["direction"])
        bt = break_map.get(key)
        if bt is None:
            missing += 1
        t["_break_time"] = bt
    if missing:
        raise RuntimeError(f"{period_name}: break_time を紐づけられないトレードが {missing} 件")
    return trades


def four_conditions(trades: list[dict]) -> dict:
    """amendment-02 §4 の主判定4条件を計算する."""
    r_net = np.array([float(t["r_net"]) for t in trades], dtype=float)
    r_gross = np.array([float(t["r_gross"]) for t in trades], dtype=float)
    cost = np.array([float(t["cost_r"]) + float(t["commission_r"]) for t in trades], dtype=float)
    wins, losses = r_net[r_net > 0], r_net[r_net < 0]
    payoff = float(wins.mean() / abs(losses.mean())) if wins.size and losses.size else None
    k5m = float(r_gross.mean() / cost.mean())
    perm = permutation_test_block(
        [float(t["r_net"]) for t in trades],
        [pd.Timestamp(t["entry_time"]).normalize() for t in trades],
        n_permutations=2000, seed=42)
    p_val = float(perm.p_value)
    unmet = []
    if not r_net.mean() > 0:
        unmet.append("平均r_net>0")
    if not (payoff is not None and payoff >= 1.5):
        unmet.append("K4mペイオフ>=1.5")
    if not k5m >= 3.0:
        unmet.append("K5m>=3.0")
    if not p_val < 0.05:
        unmet.append("permutation_p<0.05")
    cls = ("C_再現しなかった" if (r_net.mean() <= 0 or len(unmet) >= 2)
           else "B_部分的に再現" if len(unmet) == 1 else "A_再現した")
    return {"n": len(trades), "mean_r_net": round(float(r_net.mean()), 4),
            "payoff_r": round(payoff, 3) if payoff else None,
            "k5m": round(k5m, 3), "perm_p_block": round(p_val, 4),
            "unmet": unmet, "classification": cls}


def main() -> None:
    print("=" * 88)
    print("OBS000009 不具合1（探索窓起点の先読み）が SYS-FX026 に与える影響の検算")
    print("=" * 88)
    print(f"機序: to_h1()のresample('1h')はlabel既定=left → H1indexはバー始値時刻。")
    print(f"      start_time = 始値時刻 + {bd.WINDOW_START_MIN}分 = バー確定の{BAR_MINUTES - bd.WINDOW_START_MIN}分『前』")
    print(f"      判定: entry_time - break_time < {BAR_MINUTES}分 なら『バー確定前エントリー』\n")

    out = {}
    for period in ("train", "validation", "test"):
        trades = run_instrumented(period)
        for t in trades:
            off = (pd.Timestamp(t["entry_time"]) - pd.Timestamp(t["_break_time"])).total_seconds() / 60
            t["_offset_min"] = off
            t["_lookahead"] = off < BAR_MINUTES

        la = [t for t in trades if t["_lookahead"]]
        clean = [t for t in trades if not t["_lookahead"]]
        pnl_all = sum(float(t["dollar_pnl"]) for t in trades)
        pnl_la = sum(float(t["dollar_pnl"]) for t in la)
        offs = [t["_offset_min"] for t in la]

        print(f"--- {period} ---")
        print(f"  全トレード {len(trades)}件 / うち確定前エントリー {len(la)}件 "
              f"({len(la)/len(trades)*100:.1f}%)  オフセット中央値={np.median(offs):.0f}分" if la else
              f"  全トレード {len(trades)}件 / 確定前エントリー 0件")
        print(f"  純損益: 全体 ${pnl_all:,.2f} / うち該当分 ${pnl_la:,.2f} "
              f"({pnl_la/pnl_all*100:.1f}%)")
        full = four_conditions(trades)
        excl = four_conditions(clean) if clean else None
        print(f"  [公表値  ] n={full['n']:>4} r_net={full['mean_r_net']:>7} ペイオフ={full['payoff_r']:>6} "
              f"K5m={full['k5m']:>6} perm_p={full['perm_p_block']:>6} → {full['classification']}")
        if excl:
            print(f"  [先読み除外] n={excl['n']:>4} r_net={excl['mean_r_net']:>7} ペイオフ={excl['payoff_r']:>6} "
                  f"K5m={excl['k5m']:>6} perm_p={excl['perm_p_block']:>6} → {excl['classification']}")
            if full["classification"] != excl["classification"]:
                print(f"  ★★ 判定が {full['classification']} → {excl['classification']} へ反転 ★★")
        print()

        out[period] = {
            "n_trades": len(trades), "n_lookahead": len(la),
            "pct_lookahead": round(len(la) / len(trades) * 100, 1),
            "offset_min_median": round(float(np.median(offs)), 1) if la else None,
            "net_pnl_usd": round(pnl_all, 2),
            "net_pnl_lookahead_usd": round(pnl_la, 2),
            "pct_pnl_from_lookahead": round(pnl_la / pnl_all * 100, 1) if pnl_all else None,
            "four_conditions_reported": full,
            "four_conditions_excl_lookahead": excl,
            "verdict_flips": bool(excl and full["classification"] != excl["classification"]),
        }

    out_meta = {
        "purpose": "OBS000009 不具合1（探索窓起点の先読み）の SYS-FX026 への影響検算（C査読指摘の独立確認）",
        "mechanism": "to_h1()のresample('1h')はlabel既定=left(バー始値時刻)。start_time=始値+30分は"
                     "バー確定の30分前にあたり、確定後にしか分からない高値/安値/終値を使ってエントリーしうる",
        "spec_says": "00-spec.md は「ブレイクバー確定後30分」と明記しており実装と食い違う",
        "caveat": "『該当トレードを除外』は正しい修正の近似。本来は窓起点を+1hして全期間を再実行すべきで、"
                  "その場合ポジション占有のゲーティングが変わり別のトレード集合になる",
        "not_a_test_redraw": "これはTestを引き直す行為ではなく、消費済みTest結果に対するバグ診断である",
        "periods": out,
    }
    path = ROOT / "research" / "method-notes" / "sysfx026_lookahead_impact.json"
    path.write_text(json.dumps(out_meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"出力: {path}")


if __name__ == "__main__":
    main()
