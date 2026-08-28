"""EXP-FX000021: SYS-FX026（SYS-FX011の出口再設計、ATRトレール幅3.0倍）Trainベースライン.

## 位置づけ

`obs/.../01開発アイデア/OBS000013-コスト構造改善-ATRトレール幅拡大.md` の一次診断
（R指標ベース、簡略Sharpe使用）で確認した「ATRトレール幅を広げるとK5m/K4mが
改善する」という傾向を、**正式$パイプライン**（複利1%リスクサイジング・ピーク比DD・
permutation test・K3mスケール不変判定）で確認する。

## 事前登録（結果を見る前に固定、OBS000013 §4 の宣言どおり）

- **候補は 3.0 倍のみ**。一次診断のスイープ{1.0,1.5,2.0,3.0}で最も明確にK5m(≥3.0)・
  K4m(≥1.5)の両方をクリアした水準。中間値(1.5x/2.0x)を正式パイプラインで併せて
  試し「良い方を選ぶ」ことはしない(HARKing防止)
- **他のパラメータは一切再導出しない**（N_BREAKOUT・stop_buffer_atr_m5・
  breakeven_trigger_r・フィルター等はSYS-FX011 Train導出値をそのまま使用。
  atr_trail_multiplier_m5のみ stop_buffer_atr_m5×3.0 に変更）
- **KPI閾値はSYS-FX011公式spec(`00-spec.md`)と同一**（K1m〜K7m + min_n_trades(300)
  + permutation p<0.05）。SYS-FX011の公式評価基準(base Train、拡張前)と直接比較できる
- 評価期間はSYS-FX011の公式Train期間（2023-11-01〜2025-03-31、GMOデータ）のみ。
  Train通過した場合のみValidationを参照する

出力: research/method-notes/sysfx026_wide_trail_trainonly.json
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

TRAIL_MULT_FACTOR = 3.0  # 事前登録: これ以外の水準は正式パイプラインで試さない


def main() -> None:
    print("=== SYS-FX026: SYS-FX011の出口再設計(ATRトレール幅×3.0)、正式$パイプラインTrain評価 ===")
    print(f"stop_buffer_atr_m5={v7.STOP_BUFFER_ATR_M5}, "
          f"atr_trail_multiplier_m5(現行)={v7.ATR_TRAIL_MULTIPLIER_M5}, "
          f"atr_trail_multiplier_m5(SYS-FX026)={v7.STOP_BUFFER_ATR_M5 * TRAIL_MULT_FACTOR}")

    original = v7.ATR_TRAIL_MULTIPLIER_M5
    v7.ATR_TRAIL_MULTIPLIER_M5 = v7.STOP_BUFFER_ATR_M5 * TRAIL_MULT_FACTOR
    try:
        start, end = v7.PERIODS["train"]
        p = v7.run_period("train", start, end)
    finally:
        v7.ATR_TRAIL_MULTIPLIER_M5 = original

    r = evaluate_period("SYS-FX026", p, perm_p_field="perm_p_block",
                         apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)
    r["n_trades"] = p["n_trades"]
    r["final_balance_usd"] = p["final_balance_usd"]
    r["total_return_pct"] = p["total_return_pct"]
    r["trail_mult_factor"] = TRAIL_MULT_FACTOR
    r["atr_trail_multiplier_m5"] = v7.STOP_BUFFER_ATR_M5 * TRAIL_MULT_FACTOR

    print("\n=== 正式KPI判定結果 ===")
    for k, v in r.items():
        if k not in ("kpi_pass",):
            print(f"  {k}: {v}")
    print(f"  kpi_pass: {r.get('kpi_pass')}")

    out = {
        "purpose": "SYS-FX026(SYS-FX011のATRトレール幅3.0倍再設計)の正式$パイプラインTrain評価",
        "pre_registration": {
            "trail_mult_factor": TRAIL_MULT_FACTOR,
            "note": "OBS000013で事前登録した単一候補。他水準はこのEXPでは試さない",
        },
        "result": r,
    }
    out_path = ROOT / "research" / "method-notes" / "sysfx026_wide_trail_trainonly.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n出力: {out_path}")


if __name__ == "__main__":
    main()
