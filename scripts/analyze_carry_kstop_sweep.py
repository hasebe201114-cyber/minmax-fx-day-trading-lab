"""EXP-FX000004 フェーズゲート2続き: k_stop(週内ATRストップ幅)の感度スイープ.

背景: `backtest_carry_baseline.py`でk_stop=2.386(価格変動リターンp10から導出)
を検証した結果、Train全体でわずかにマイナス(-4,313円)だった。内訳を分解すると、
2024年7月・9月の持続的トレンド局面ではストップが大きく機能した(+19,487円・
+3,682円)一方、2024年8月5日の単発急落→即反発週では大きく裏目に出た
(-17,771円)。**ただし最大の押し下げ要因はそれ以外の「平常時」(-9,711円)**で、
通常のノイズレベルの逆行でストップが頻繁に発動し、金曜には回復していたはずの
ケースを損切り確定させていたと推測される。

本スクリプトは、ストップ幅を広げることで平常時の誤発動を減らせるかを確認する
ため、k_stopの候補値を体系的にスイープする(単一の「最良値」を後付けで選ぶ
のではなく、スイープ全体の傾向を確認するのが目的)。

事前登録 (結果を見る前に固定):
    - 候補: k_stop ∈ {なし(ストップ無し), 1.5, 2.0, 2.386(前回検証値), 3.0,
      4.0, 5.0, 6.0} (ATR(D1,14)の倍率)
    - 評価指標: 総リターン・月次シャープ・最大DD(月間/年間)・PF・ペイオフ比・
      ストップ発動件数を候補ごとに算出し、単調な傾向があるかを確認する
    - 対象期間: Train(2023-11-01〜2025-03-31)のみ

出力: research/method-notes/carry_kstop_sweep.json
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

import backtest_carry_baseline as base  # noqa: E402

K_STOP_CANDIDATES = [None, 1.5, 2.0, 2.386, 3.0, 4.0, 5.0, 6.0]


def main() -> int:
    print("=== EXP-FX000004: k_stop感度スイープ (Train期間) ===\n")

    all_data = {}
    for pair in base.PAIRS:
        m5 = base.load_m5(pair)
        d1 = base.to_d1(m5)
        atr_d1 = base.atr_ind(d1["high"], d1["low"], d1["close"], length=14)
        swap_daily = base.load_swap_daily(pair)
        all_data[pair] = (m5, d1, atr_d1, swap_daily)

    results = {}
    print(f"{'k_stop':<10}{'総リターン':>10}{'月次Sharpe':>12}{'最大DD':>9}{'月間DD':>9}{'PF':>7}{'ペイオフ':>9}{'発動数':>7}")
    for k_stop in K_STOP_CANDIDATES:
        cycles_all = []
        for pair in base.PAIRS:
            m5, d1, atr_d1, swap_daily = all_data[pair]
            cycles = base.weekly_cycles(pair, m5, d1, atr_d1, swap_daily, k_stop=k_stop)
            cycles_all.extend(cycles)
        label = "no_stop" if k_stop is None else f"k={k_stop}"
        summary = base.summarize(cycles_all, label)
        results[label] = summary
        print(f"{label:<10}{summary['total_return_pct']:>9.2f}%{summary['monthly_sharpe']:>12.3f}"
              f"{summary['max_dd_pct']:>8.2f}%{summary['max_dd_monthly_pct']:>8.2f}%"
              f"{summary['profit_factor']:>7.3f}{summary['payoff_ratio']:>9.3f}{summary['n_stopped']:>7d}")

    print("\n=== 傾向の確認 ===")
    returns_by_k = [(k, results["no_stop" if k is None else f"k={k}"]["total_return_pct"]) for k in K_STOP_CANDIDATES]
    best_k, best_ret = max(returns_by_k, key=lambda kv: kv[1])
    print(f"総リターン最良: k_stop={best_k}  ({best_ret:+.2f}%)")
    no_stop_ret = results["no_stop"]["total_return_pct"]
    print(f"ストップ無し: {no_stop_ret:+.2f}%")
    monotonic_toward_no_stop = all(
        returns_by_k[i][1] <= returns_by_k[i + 1][1] for i in range(len(returns_by_k) - 1)
        if returns_by_k[i][0] is not None
    )
    print(f"ストップ幅を広げるほど単調にストップ無しへ近づく傾向: {'あり' if monotonic_toward_no_stop else 'なし(非単調)'}")

    out_path = ROOT / "research" / "method-notes" / "carry_kstop_sweep.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [base.TRAIN_START, base.TRAIN_END],
            "k_stop_candidates": [k if k is not None else "no_stop" for k in K_STOP_CANDIDATES],
            "results": results,
            "_note": (
                "週内ATRストップ幅(k_stop)を{None,1.5,2.0,2.386,3.0,4.0,5.0,6.0}で"
                "スイープし、単一の最良値を後付けで選ぶのではなく傾向を確認する。"
                "carry_baseline_train.jsonのk_stop=2.386(価格変動p10由来)は、この"
                "スイープの1候補として含まれる。"
            ),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
