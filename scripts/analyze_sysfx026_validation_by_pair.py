"""EXP-FX000021 第1試行の頑健性チェック: SYS-FX026 Validation の通貨別内訳.

## 背景（なぜこのチェックが必要か）

EXP-FX000005 の C 品質チーム査読（`research/EXP-FX000005/20-c-review.md`）は、
SYS-FX011 について **「GBP/JPY の薄いエッジ」と「EUR/JPY(Validation n=6)の極小
サンプル依存」** を独立集計で新規発見し、差し戻し理由の一つに挙げていた。

SYS-FX026 の Validation も n=91（USD/JPY 38・EUR/JPY 6・GBP/JPY 17・AUD/JPY 30）と
小さく、**同じ弱点を引き継いでいないか**を、結果を良く見せる方向ではなく
「1通貨に依存していないか」を暴く方向で確認する。

## 判定の考え方（結果を見る前に記録）

- 4通貨のうち **1通貨を除いただけで平均 r_net の符号が反転する / K5m が 3.0 を割る**
  なら、その結果は単一通貨依存であり信頼できない
- 逆に **どの1通貨を抜いても K4m・K5m が基準を維持する**なら、通貨横断的な効果と言える

出力: research/method-notes/sysfx026_validation_by_pair.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd as v7  # noqa: E402
from backtest_sysfx026_sizing_trial1 import RISK_PCT_TRIAL1, TRAIL_MULT_FACTOR  # noqa: E402


def stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0}
    r_net = np.array([t["r_net"] for t in trades], dtype=float)
    r_gross = np.array([t["r_gross"] for t in trades], dtype=float)
    cost = np.array([t["cost_r"] + t["commission_r"] for t in trades], dtype=float)
    wins = r_net[r_net > 0]
    losses = r_net[r_net < 0]
    return {
        "n": len(trades),
        "win_rate": round(float((r_net > 0).mean()), 4),
        "mean_r_net": round(float(r_net.mean()), 4),
        "profit_factor_r": round(float(wins.sum() / abs(losses.sum())), 3) if losses.size else None,
        "payoff_ratio_r": round(float(wins.mean() / abs(losses.mean())), 3)
        if wins.size and losses.size else None,
        "spread_cost_multiplier_k5m": round(float(r_gross.mean() / cost.mean()), 3),
    }


def main() -> None:
    print("=== SYS-FX026 Validation 通貨別内訳・1通貨除外感度 ===")
    orig_trail, orig_risk = v7.ATR_TRAIL_MULTIPLIER_M5, v7.RISK_PCT_PER_TRADE
    v7.ATR_TRAIL_MULTIPLIER_M5 = v7.STOP_BUFFER_ATR_M5 * TRAIL_MULT_FACTOR
    v7.RISK_PCT_PER_TRADE = RISK_PCT_TRIAL1
    try:
        start, end = v7.PERIODS["validation"]
        p = v7.run_period("validation", start, end)
    finally:
        v7.ATR_TRAIL_MULTIPLIER_M5, v7.RISK_PCT_PER_TRADE = orig_trail, orig_risk

    trades = p["trades"]
    pairs = sorted({t["pair"] for t in trades})

    print(f"\n--- 通貨別（Validation 全 n={len(trades)}） ---")
    by_pair = {}
    for pair in pairs:
        s = stats([t for t in trades if t["pair"] == pair])
        by_pair[pair] = s
        print(f"  {pair:<9} n={s['n']:>3}  勝率={s['win_rate']}  平均r_net={s['mean_r_net']:>8}  "
              f"PF(R)={s['profit_factor_r']}  ペイオフ(R)={s['payoff_ratio_r']}  K5m={s['spread_cost_multiplier_k5m']}")

    print(f"\n--- 1通貨除外感度（その通貨を抜いた残り3通貨での値） ---")
    leave_one_out = {}
    for pair in pairs:
        s = stats([t for t in trades if t["pair"] != pair])
        leave_one_out[pair] = s
        k5m_ok = "OK" if s["spread_cost_multiplier_k5m"] >= 3.0 else "NG"
        payoff_ok = "OK" if (s["payoff_ratio_r"] or 0) >= 1.5 else "NG"
        sign_ok = "OK" if s["mean_r_net"] > 0 else "NG(符号反転)"
        print(f"  {pair}を除外: n={s['n']:>3}  平均r_net={s['mean_r_net']:>8}({sign_ok})  "
              f"ペイオフ(R)={s['payoff_ratio_r']}({payoff_ok})  K5m={s['spread_cost_multiplier_k5m']}({k5m_ok})")

    overall = stats(trades)
    out = {
        "purpose": "SYS-FX026 Validation の単一通貨依存を暴くための独立集計（C査読の過去指摘に対応）",
        "period": {"start": v7.PERIODS["validation"][0], "end": v7.PERIODS["validation"][1]},
        "params": {"trail_mult_factor": TRAIL_MULT_FACTOR, "risk_pct_per_trade": RISK_PCT_TRIAL1},
        "overall": overall,
        "by_pair": by_pair,
        "leave_one_out": leave_one_out,
    }
    out_path = ROOT / "research" / "method-notes" / "sysfx026_validation_by_pair.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n出力: {out_path}")


if __name__ == "__main__":
    main()
