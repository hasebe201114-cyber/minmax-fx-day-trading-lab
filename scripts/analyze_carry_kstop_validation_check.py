"""EXP-FX000004: k_stopスイープでTrain最良だったk=1.5をValidationで即座に確認する.

背景: `analyze_carry_kstop_sweep.py`で8候補をスイープした結果、Trainでは
k_stop=1.5が総リターン+7.40%(ストップ無しの+4.39%を大きく上回る)・
Sharpe1.407・最大DD4.10%と最良だった。しかし本PJは直前のSYS-FX009タイト
ストップ検証(k=0.5)で「Trainで最良だった候補がValidation/Testでは崩れる」
という過学習を経験済み(Train+14.3%→Validation-31.7%→Test-38.4%)。同じ
過ちを繰り返さないため、Train最良候補を正式採用する前に、まずValidation
期間で即座に確認する。

事前登録: k_stop=1.5(Trainスイープの最良値)とストップ無しをValidation期間
(2025-04-01〜2025-11-30)で比較する。追加のチューニングは行わない
(Validationで崩れた場合、Trainの最良値選択自体が過学習だったと判断する)。

出力: research/method-notes/carry_kstop_validation_check.json
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

VALIDATION_START, VALIDATION_END = "2025-04-01", "2025-11-30"
K_STOP_CHECK = [None, 1.5, 2.386]  # ストップ無し・Train最良・前回検証値の3つのみ


def main() -> int:
    print("=== EXP-FX000004: Train最良k_stop(1.5)のValidation確認 ===\n")

    # baseモジュールのTRAIN_START/ENDを一時的にValidation期間へ差し替える
    base.TRAIN_START, base.TRAIN_END = VALIDATION_START, VALIDATION_END
    print(f"対象期間: {VALIDATION_START} 〜 {VALIDATION_END} (Validation)\n")

    all_data = {}
    for pair in base.PAIRS:
        m5 = base.load_m5(pair)
        d1 = base.to_d1(m5)
        atr_d1 = base.atr_ind(d1["high"], d1["low"], d1["close"], length=14)
        swap_daily = base.load_swap_daily(pair)
        all_data[pair] = (m5, d1, atr_d1, swap_daily)
        print(f"  [{pair}] {len(m5)}本のM5データ")

    print(f"\n{'k_stop':<10}{'総リターン':>10}{'月次Sharpe':>12}{'最大DD':>9}{'月間DD':>9}{'PF':>7}{'ペイオフ':>9}{'発動数':>7}")
    results = {}
    for k_stop in K_STOP_CHECK:
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

    print("\n=== 判定 ===")
    no_stop_ret = results["no_stop"]["total_return_pct"]
    k15_ret = results["k=1.5"]["total_return_pct"]
    print(f"ストップ無し: {no_stop_ret:+.2f}%  k=1.5: {k15_ret:+.2f}%")
    if k15_ret > no_stop_ret:
        verdict = "Validationでも改善を維持 → k_stop=1.5の効果は再現性がある可能性"
    else:
        verdict = "Validationで改善が消失/逆転 → Trainでの最良値選択は過学習だった可能性が高い"
    print(verdict)

    out_path = ROOT / "research" / "method-notes" / "carry_kstop_validation_check.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "validation_period": [VALIDATION_START, VALIDATION_END],
            "results": results,
            "verdict": verdict,
            "_note": "Trainスイープ(carry_kstop_sweep.json)で最良だったk_stop=1.5を、追加チューニング無しでValidation期間に適用した確認結果。",
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
