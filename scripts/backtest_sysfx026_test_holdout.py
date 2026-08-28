"""EXP-FX000021: SYS-FX026 の凍結ホールドアウト（Test）一度限りの参照.

事前登録: `research/EXP-FX000021/00-spec-amendment-02.md`
（**Test の損益を一切見ずに確定し、コミット b161333 で先に記録済み**）

## 一度限りの原則（amendment-02 §5）

**本スクリプトの結果を見た後に、パラメータ・通貨構成・出口設計・サイジングを変更して
再び Test を引くことを、いかなる理由でも行わない。** EXP-FX000005 T-05 の凍結宣言は
本参照をもって SYS-FX011 系列について消費される。

## パラメータ（amendment-02 §2、すべて Train 由来の凍結値）

trail = stop_buffer_atr_m5 × 3.0、risk_pct = 0.65%。他は SYS-FX011 T-13 構成のまま。
**Validation の結果を用いて決めたパラメータは1つもない。**

## 判定ルール（amendment-02 §4、Test を見る前に固定）

実効n は約8.5ヶ月窓では 300 未達と事前予想されるため、主判定は
サイジングで動かせない指標に絞った4条件で行う（実効n は実測値を必ず併記）:

  ①平均r_net > 0  ②K4m ペイオフ ≥1.5  ③K5m コスト倍率 ≥3.0  ④permutation p(日ブロック) <0.05

  A: 4条件すべて達成 → 再現した     → フォワード投入へ
  B: 1つ未達         → 部分的に再現 → フォワード投入するが採用GOは具申しない
  C: 2つ以上未達、または平均r_netが負 → 再現せず → 採用を具申しない

出力: research/method-notes/sysfx026_test_holdout.json
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
from analyze_sysfx026_validation_by_pair import stats  # noqa: E402
from backtest_sysfx026_sizing_trial1 import RISK_PCT_TRIAL1, TRAIL_MULT_FACTOR  # noqa: E402
from evaluate_vol_breakout_dow_theory_kpi import evaluate_period  # noqa: E402


def classify(mean_r_net: float, payoff_r: float | None, k5m: float, perm_p: float) -> tuple[str, list[str]]:
    """amendment-02 §4 の判定ルール（Test を見る前に固定）."""
    unmet = []
    if not mean_r_net > 0:
        unmet.append("平均r_net>0")
    if not (payoff_r is not None and payoff_r >= 1.5):
        unmet.append("K4mペイオフ>=1.5")
    if not k5m >= 3.0:
        unmet.append("K5m>=3.0")
    if not perm_p < 0.05:
        unmet.append("permutation_p<0.05")

    if mean_r_net <= 0 or len(unmet) >= 2:
        return "C_再現しなかった", unmet
    if len(unmet) == 1:
        return "B_部分的に再現", unmet
    return "A_再現した", unmet


def main() -> None:
    print("=" * 78)
    print("EXP-FX000021: SYS-FX026 凍結ホールドアウト（Test）一度限りの参照")
    print("事前登録: amendment-02（コミット b161333、Test の損益を見る前に確定済み）")
    print("=" * 78)

    orig_trail, orig_risk = v7.ATR_TRAIL_MULTIPLIER_M5, v7.RISK_PCT_PER_TRADE
    v7.ATR_TRAIL_MULTIPLIER_M5 = v7.STOP_BUFFER_ATR_M5 * TRAIL_MULT_FACTOR
    v7.RISK_PCT_PER_TRADE = RISK_PCT_TRIAL1
    print(f"\nパラメータ（全てTrain由来の凍結値）: "
          f"trail={v7.ATR_TRAIL_MULTIPLIER_M5:.4f} (={TRAIL_MULT_FACTOR}x), "
          f"risk_pct={RISK_PCT_TRIAL1*100:.2f}%")
    try:
        start, end = v7.PERIODS["test"]
        p = v7.run_period("test", start, end)
    finally:
        v7.ATR_TRAIL_MULTIPLIER_M5, v7.RISK_PCT_PER_TRADE = orig_trail, orig_risk

    r = evaluate_period("test", p, perm_p_field="perm_p_block",
                        apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)
    trades = p["trades"]
    overall = stats(trades)

    print("\n" + "-" * 78)
    print("--- Test 正式KPI（必須9ゲート、閾値は一切変更していない） ---")
    for k in ("n_trades", "n_trades_effective", "monthly_sharpe", "profit_factor",
              "payoff_ratio", "spread_cost_multiplier", "max_dd_monthly_pct", "max_dd_pct",
              "permutation_p_clustered", "monthly_expectancy_positive",
              "final_balance_usd", "total_return_pct",
              "kpi_required_pass_count", "kpi_required_all_pass"):
        print(f"  {k}: {r.get(k)}")
    print(f"  kpi_pass: {r.get('kpi_pass')}")

    # amendment-02 §4 の主判定
    verdict, unmet = classify(overall["mean_r_net"], overall["payoff_ratio_r"],
                              overall["spread_cost_multiplier_k5m"],
                              float(r["permutation_p_clustered"]))
    print("\n--- 主判定4条件（amendment-02 §4、Test を見る前に固定） ---")
    print(f"  ①平均r_net           = {overall['mean_r_net']}  ({'OK' if overall['mean_r_net']>0 else 'NG'})")
    print(f"  ②K4mペイオフ(R)      = {overall['payoff_ratio_r']}  "
          f"({'OK' if (overall['payoff_ratio_r'] or 0)>=1.5 else 'NG'})")
    print(f"  ③K5mコスト倍率       = {overall['spread_cost_multiplier_k5m']}  "
          f"({'OK' if overall['spread_cost_multiplier_k5m']>=3.0 else 'NG'})")
    print(f"  ④permutation p(日BL) = {r['permutation_p_clustered']}  "
          f"({'OK' if float(r['permutation_p_clustered'])<0.05 else 'NG'})")
    print(f"\n  >>> 判定: {verdict}" + (f"  未達={unmet}" if unmet else ""))

    # 通貨別・1通貨除外感度（C査読が行われないため、唯一の通貨集中チェック）
    pairs = sorted({t["pair"] for t in trades})
    print("\n--- 通貨別内訳（C査読が無いため唯一の通貨集中チェック） ---")
    by_pair = {}
    for pair in pairs:
        s = stats([t for t in trades if t["pair"] == pair])
        by_pair[pair] = s
        print(f"  {pair:<9} n={s['n']:>3}  勝率={s['win_rate']}  平均r_net={s['mean_r_net']:>8}  "
              f"PF(R)={s['profit_factor_r']}  ペイオフ(R)={s['payoff_ratio_r']}  K5m={s['spread_cost_multiplier_k5m']}")

    print("\n--- 1通貨除外感度 ---")
    leave_one_out = {}
    for pair in pairs:
        s = stats([t for t in trades if t["pair"] != pair])
        leave_one_out[pair] = s
        flags = (f"{'OK' if s['mean_r_net']>0 else 'NG(符号反転)'}/"
                 f"{'OK' if (s['payoff_ratio_r'] or 0)>=1.5 else 'NG'}/"
                 f"{'OK' if s['spread_cost_multiplier_k5m']>=3.0 else 'NG'}")
        print(f"  {pair}を除外: n={s['n']:>3}  平均r_net={s['mean_r_net']:>8}  "
              f"ペイオフ(R)={s['payoff_ratio_r']}  K5m={s['spread_cost_multiplier_k5m']}  [{flags}]")

    out = {
        "purpose": "SYS-FX026 の凍結ホールドアウト（Test）一度限りの参照。"
                   "amendment-02 で Test を見る前に事前登録済み（コミット b161333）",
        "one_shot_declaration": "本結果を見た後にパラメータを変更して再度Testを引くことは禁止"
                                "（amendment-02 §5）。T-05の凍結はこの参照で消費された",
        "period": {"start": v7.PERIODS["test"][0], "end": v7.PERIODS["test"][1]},
        "params_all_train_derived": {
            "trail_mult_factor": TRAIL_MULT_FACTOR,
            "atr_trail_multiplier_m5": v7.STOP_BUFFER_ATR_M5 * TRAIL_MULT_FACTOR,
            "risk_pct_per_trade": RISK_PCT_TRIAL1,
        },
        "kpi": r,
        "primary_verdict": {"classification": verdict, "unmet_conditions": unmet},
        "overall_r_stats": overall,
        "by_pair": by_pair,
        "leave_one_out": leave_one_out,
        "caveat": "Test はトレール幅・サイジング変更に対しては清潔だが、基盤設計に対しては"
                  "清潔でない（SYS-FX011の改善ループでT-05凍結前に14回以上参照済み）。"
                  "良い結果は割り引いて読み、悪い結果は割り引かずに読む（amendment-02 §6）",
    }
    out_path = ROOT / "research" / "method-notes" / "sysfx026_test_holdout.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n出力: {out_path}")


if __name__ == "__main__":
    main()
