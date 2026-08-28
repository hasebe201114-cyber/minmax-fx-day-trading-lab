"""EXP-FX000021 改善ループ第1試行: SYS-FX026 のサイジング調整（risk_pct 1.00%→0.65%）.

事前登録: `research/EXP-FX000021/00-spec-amendment-01.md`（結果を見る前に確定）

## 本試行の性質（amendment-01 §2、結果を見る前に宣言済み）

**これは「戦略の質の改善」ではなく「単位の変更」である。** 複利サイジングでは
r_net・勝率・PF・ペイオフ(K4m)・K5m・実効n・permutation p値はすべて R 単位の量で
サイジングと独立なため **完全に不変** であり、変わるのは DD・総リターン・最終残高だけ
（月次シャープはほぼ不変）。したがって Train で DD が下がること自体は検証ではなく算術。

**正の期待値を持つ戦略はサイジングを絞れば K2m を必ず通せるため、K2m は本枠組みでは
戦略の質を選別するゲットとして機能していない。** 仮に Train 9/9 になっても
「全ゲート突破」と読んではならない（amendment-01 §2）。

## risk_pct の導出（amendment-01 §3、スイープしない）

    risk_pct = 1.00% × (TARGET_DD 8.0% / OBSERVED_TRAIN_DD 12.08%) = 0.6623%
             → 0.05% 刻みで切り捨て（保守側） → 0.65%

## 選定ルール（amendment-01 §4）

Train で必須9ゲート全達成 → Validation を参照（パラメータ再導出なし）。
Test は EXP-FX000005 T-05 で凍結宣言済みのため一切参照しない。

出力: research/method-notes/sysfx026_sizing_trial1.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd as v7  # noqa: E402
from evaluate_vol_breakout_dow_theory_kpi import evaluate_period  # noqa: E402

# 親spec で事前登録済み（変更しない）
TRAIL_MULT_FACTOR = 3.0

# amendment-01 §3 の導出（結果を見る前に確定、スイープしない）
RISK_PCT_BASE = 0.01
TARGET_DD_PCT = 8.0
OBSERVED_TRAIN_DD_PCT = 12.08
_raw = RISK_PCT_BASE * (TARGET_DD_PCT / OBSERVED_TRAIN_DD_PCT)
RISK_PCT_TRIAL1 = int(_raw * 100 / 0.05) * 0.05 / 100  # 0.05%刻みで切り捨て → 0.0065


def run(period_name: str) -> dict:
    """指定期間を、トレール幅3.0倍・risk_pct=0.65% で評価する."""
    orig_trail = v7.ATR_TRAIL_MULTIPLIER_M5
    orig_risk = v7.RISK_PCT_PER_TRADE
    v7.ATR_TRAIL_MULTIPLIER_M5 = v7.STOP_BUFFER_ATR_M5 * TRAIL_MULT_FACTOR
    v7.RISK_PCT_PER_TRADE = RISK_PCT_TRIAL1
    try:
        start, end = v7.PERIODS[period_name]
        p = v7.run_period(period_name, start, end)
    finally:
        v7.ATR_TRAIL_MULTIPLIER_M5 = orig_trail
        v7.RISK_PCT_PER_TRADE = orig_risk

    r = evaluate_period(period_name, p, perm_p_field="perm_p_block",
                        apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)
    r["n_trades"] = p["n_trades"]
    r["final_balance_usd"] = p["final_balance_usd"]
    r["total_return_pct"] = p["total_return_pct"]
    r["risk_pct_per_trade"] = RISK_PCT_TRIAL1
    r["atr_trail_multiplier_m5"] = v7.STOP_BUFFER_ATR_M5 * TRAIL_MULT_FACTOR
    return r


def show(label: str, r: dict) -> None:
    print(f"\n--- {label} ---")
    for k in ("n_trades", "n_trades_effective", "monthly_sharpe", "profit_factor",
              "payoff_ratio", "spread_cost_multiplier", "max_dd_monthly_pct", "max_dd_pct",
              "permutation_p_clustered", "monthly_expectancy_positive",
              "final_balance_usd", "total_return_pct",
              "kpi_required_pass_count", "kpi_required_all_pass"):
        print(f"  {k}: {r.get(k)}")
    print(f"  kpi_pass: {r.get('kpi_pass')}")


def main() -> None:
    print("=== EXP-FX000021 改善ループ第1試行: サイジング調整 ===")
    print(f"事前登録: trail={TRAIL_MULT_FACTOR}x（親specから不変）, "
          f"risk_pct={RISK_PCT_TRIAL1:.4f} (={RISK_PCT_TRIAL1*100:.2f}%, "
          f"1.00% × {TARGET_DD_PCT}/{OBSERVED_TRAIN_DD_PCT} を0.05%刻みで切り捨て)")
    print("注意: r_net/PF/ペイオフ/K5m/実効n/permutation はサイジングと独立で不変。"
          "変わるのはDD・リターンのみ（amendment-01 §2）\n")

    train = run("train")
    show("Train", train)

    out: dict = {
        "purpose": "EXP-FX000021 改善ループ第1試行（サイジング調整）。"
                   "amendment-01 §2 のとおり、質の改善ではなく単位の変更である",
        "pre_registration": {
            "trail_mult_factor": TRAIL_MULT_FACTOR,
            "risk_pct_per_trade": RISK_PCT_TRIAL1,
            "derivation": f"1.00% × ({TARGET_DD_PCT}/{OBSERVED_TRAIN_DD_PCT}) → 0.05%刻み切り捨て",
            "caveat": "K2mはサイジングで満たせる非拘束的なゲートであり、"
                      "Train 9/9 を『全ゲート突破』と読んではならない（amendment-01 §2）",
        },
        "train": train,
    }

    # amendment-01 §4: Train が必須9ゲート全達成の場合のみ Validation を参照する
    if train.get("kpi_required_all_pass"):
        print("\n>>> Train 必須9ゲート全達成。事前登録した選定ルールに従い Validation を参照する")
        validation = run("validation")
        show("Validation", validation)
        out["validation"] = validation
    else:
        print("\n>>> Train に未達が残るため、事前登録した選定ルールにより Validation は参照しない")
        out["validation"] = None

    out_path = ROOT / "research" / "method-notes" / "sysfx026_sizing_trial1.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n出力: {out_path}")


if __name__ == "__main__":
    main()
