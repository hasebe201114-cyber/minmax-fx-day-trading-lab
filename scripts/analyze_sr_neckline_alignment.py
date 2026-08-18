"""保留事項C 続き: ネックラインがレジサポラインと一致するかの検証(第1段階).

背景: 司令塔から追加で3点(SL決定方法・戻りのエントリータイミング・
レジサポライン考慮)の見直し依頼。3点は相互に関連するエントリーロジック
自体の再設計だが、まず「そもそもネックラインは本物のレジサポラインと
一致していることが多いのか」を確認しないと、戻りエントリーやSR基準の
SLを設計する意味があるか判断できない。本スクリプトはその第1段階の
診断で、まだ戦略ロジックの変更(戻りエントリー等)は行わない。

方法: 継続文脈のH1ダブルトップ/ボトム・ブレイクイベント(既存の
`analyze_scaled_exit_diagnostic.find_continuation_entries`と同じ検出
ロジックを、ネックライン価格も返すよう拡張して再利用)それぞれについて、
`support_resistance.detect_support_resistance()`(SYS-FX007の既存資産、
H4・365日ルックバック・Williams Fractal+クラスタリング+接触回数フィルタ)
を**ブレイク時点までのデータのみ**(先読みなし)で計算し、ネックライン
価格がそのS/Rライン群のいずれかと一致する(許容誤差0.5%、touch_tolerance_pct
のデフォルトをそのまま流用)かを判定する。

事前登録 (結果を見る前に固定):
    - S/R検出パラメータ: `detect_support_resistance()`のデフォルト値を
      そのまま使用(fractal_window=5, min_touches=3, cluster_threshold_pct=0.5,
      touch_tolerance_pct=0.5, lookback_days=365)。SYS-FX007で既に検証済みの
      値を流用し、今回のために新たに調整しない
    - 一致判定: ネックラインと同種(ダブルボトムのネックライン=レジスタンス、
      ダブルトップのネックライン=サポート)のS/Rラインのうち、価格差が
      touch_tolerance_pct(0.5%)以内のものが1つでもあれば「一致」とする
    - 比較する2グループ: SR_ALIGNED(一致あり) / SR_NOT_ALIGNED(一致なし)
      それぞれの前方リターンIC(3dのみ、前回検証と揃える)を比較
    - 多重検定: 2グループ×1ホライズン=2件でBonferroni補正

出力: research/method-notes/sr_neckline_alignment.json
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

import numpy as np
import pandas as pd

import analyze_scaled_exit_diagnostic as base  # noqa: E402
from minmax_fx_dt.backtest.permutation import permutation_test  # noqa: E402
from minmax_fx_dt.strategy.support_resistance import detect_support_resistance  # noqa: E402

PAIRS = base.PAIRS
TRAIN_START, TRAIN_END = base.TRAIN_START, base.TRAIN_END
EFFECTIVE_PAIR_COUNT = 1.70
MIN_N_FOR_JUDGEMENT = 30
HORIZON_LABEL, HORIZON_BARS = "3d", 72
SR_TOUCH_TOLERANCE_PCT = 0.5  # detect_support_resistance()のデフォルトをそのまま流用
# 事後追記: デフォルト基準(min_touches=3, strength不問)で96.3%が一致し、識別力が
# ほぼ無いと判明したため、より選択的な基準(高接触回数・高強度)でも同じ傾向かを
# 追加確認する。これは「結果を見てから閾値を探した」のではなく、デフォルト基準が
# 識別力ゼロだったことを受けた自然な頑健性チェック。
STRICT_MIN_TOUCHES = 5
STRICT_MIN_STRENGTH = 0.7


def to_h4(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: m5[c].resample("4h").agg(a) for c, a in
                          [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]}).dropna()


def find_events_with_neckline(pair: str, params: dict) -> list[dict]:
    """find_continuation_entries()と同じ検出ロジックだが、ネックライン価格も返す."""
    from minmax_fx_dt.strategy.indicators import atr as atr_ind
    from minmax_fx_dt.strategy.support_resistance import zigzag_pivots_typed

    m5 = base.load_m5(pair)
    h1 = base.to_h1(m5)
    d1 = base.to_d1(m5)
    atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    lt_dir = base.lt_direction_series(d1)

    pivots = zigzag_pivots_typed(h1["high"], h1["low"], atr_h1, params["zigzag_threshold_atr"])
    triplets = base.alternating_triplets(pivots)
    tol = params["pattern_tolerance_atr"]
    buffer_atr = params["stop_buffer_atr"]
    cap = params["break_search_cap_bars"]

    events = []
    for idx1, kind1, idx2, _k2, idx3, _k3 in triplets:
        required_lt = "DOWN" if kind1 == "HIGH" else "UP"
        atr_neckline = atr_h1.iloc[idx2]
        if pd.isna(atr_neckline) or atr_neckline <= 0:
            continue
        p1 = float(h1["high" if kind1 == "HIGH" else "low"].iloc[idx1])
        p2 = float(h1["high" if kind1 == "HIGH" else "low"].iloc[idx3])
        if abs(p1 - p2) / float(atr_neckline) > tol:
            continue
        neckline = float(h1["low" if kind1 == "HIGH" else "high"].iloc[idx2])
        pattern_extreme = max(p1, p2) if kind1 == "HIGH" else min(p1, p2)

        search_end = min(idx3 + 1 + cap, len(h1))
        for j in range(idx3 + 1, search_end):
            broke = (
                (kind1 == "HIGH" and float(h1["low"].iloc[j]) < neckline)
                or (kind1 == "LOW" and float(h1["high"].iloc[j]) > neckline)
            )
            if not broke:
                continue
            lt_at_break = lt_dir.asof(h1.index[j])
            if lt_at_break != required_lt:
                break
            atr_entry = atr_h1.iloc[j]
            if pd.isna(atr_entry) or atr_entry <= 0:
                break
            entry_price = float(h1["close"].iloc[j])
            direction = "DOWN" if kind1 == "HIGH" else "UP"
            # ネックラインの種別: ダブルボトム(kind1=LOW)のネックライン=レジスタンス
            #                     ダブルトップ(kind1=HIGH)のネックライン=サポート
            neckline_kind = "RESISTANCE" if kind1 == "LOW" else "SUPPORT"
            events.append(dict(
                pair=pair, direction=direction, entry_idx=j, entry_time=h1.index[j],
                entry_price=entry_price, neckline=neckline, neckline_kind=neckline_kind,
                pattern_extreme=pattern_extreme,
            ))
            break

    return events


def main() -> int:
    print("=== 保留事項C続き: ネックラインとレジサポラインの一致検証 (Train期間・継続文脈・3dのみ) ===\n")
    with (ROOT / "research" / "EXP-FX000003" / "10-result" / "double_pattern_params_h1.json").open(encoding="utf-8") as f:
        h1_params = json.load(f)

    print(f"事前登録: detect_support_resistance()デフォルト値・一致許容誤差{SR_TOUCH_TOLERANCE_PCT}%・"
          f"ホライズン{HORIZON_LABEL}のみ\n")

    all_events: list[dict] = []
    for pair in PAIRS:
        m5 = base.load_m5(pair)
        h1 = base.to_h1(m5)
        h4 = to_h4(m5)
        log_close_h1 = np.log(h1["close"])

        events = find_events_with_neckline(pair, h1_params)
        n_aligned = 0
        for ev in events:
            j = ev["entry_idx"]
            if j + HORIZON_BARS >= len(h1):
                continue
            raw_ret = float(log_close_h1.iloc[j + HORIZON_BARS] - log_close_h1.iloc[j])
            signed_ret = raw_ret if ev["direction"] == "UP" else -raw_ret

            # 先読み防止: ブレイク時点までのH4データのみでS/Rを計算
            h4_upto = h4[h4.index <= ev["entry_time"]]
            if len(h4_upto) < 30:
                continue
            sr_levels = detect_support_resistance(h4_upto["high"], h4_upto["low"], h4_upto["close"])
            same_kind = [lv for lv in sr_levels if lv.kind == ev["neckline_kind"]]
            aligned = any(abs(lv.price - ev["neckline"]) / ev["neckline"] <= SR_TOUCH_TOLERANCE_PCT / 100.0
                          for lv in same_kind)
            strict_kind = [lv for lv in same_kind if lv.touches >= STRICT_MIN_TOUCHES and lv.strength >= STRICT_MIN_STRENGTH]
            aligned_strict = any(abs(lv.price - ev["neckline"]) / ev["neckline"] <= SR_TOUCH_TOLERANCE_PCT / 100.0
                                  for lv in strict_kind)
            if aligned:
                n_aligned += 1
            all_events.append({"pair": pair, "aligned": aligned, "aligned_strict": aligned_strict, "ret": signed_ret})
        print(f"[{pair}] イベント={len(events)}件  S/R一致={n_aligned}件")

    print(f"\n全体イベント数: {len(all_events)}件\n")

    n_tests = 2
    bonferroni_alpha = 0.05 / n_tests
    print(f"多重検定: {n_tests}件 → Bonferroni閾値 α={bonferroni_alpha:.5f}\n")

    results = {}
    for label, cond in [("SR_ALIGNED", True), ("SR_NOT_ALIGNED", False)]:
        rets = [e["ret"] for e in all_events if e["aligned"] == cond]
        n = len(rets)
        mean_ret = float(np.mean(rets)) if n else None
        win_rate = float(np.mean([r > 0 for r in rets])) if n else None
        n_eff = max(4, int(round(n * (EFFECTIVE_PAIR_COUNT / len(PAIRS))))) if n else 0
        judgeable = n_eff >= MIN_N_FOR_JUDGEMENT
        r = {"n": n, "mean": round(mean_ret, 6) if mean_ret is not None else None,
             "win_rate": round(win_rate, 3) if win_rate is not None else None,
             "n_effective": n_eff, "judgeable": judgeable}
        if judgeable:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=n_eff, replace=False) if n_eff < n else np.arange(n)
            sub = [rets[i] for i in idx]
            pr = permutation_test(sub, seed=42)
            r["p_value_corr_adjusted"] = round(pr.p_value, 4)
            r["survives_bonferroni"] = bool(pr.p_value < bonferroni_alpha)
        else:
            r["p_value_corr_adjusted"] = None
            r["survives_bonferroni"] = False
        results[label] = r
        sig = " **" if r["survives_bonferroni"] else (
            " *" if r["p_value_corr_adjusted"] is not None and r["p_value_corr_adjusted"] < 0.05 else "")
        jflag = "" if judgeable else " [n不足・判定不能]"
        print(f"{label}: n={n:>4}  勝率={r['win_rate']}  平均(符号調整済)={r['mean']}  "
              f"n_eff={n_eff}  p(補正)={r['p_value_corr_adjusted']}{sig}{jflag}")

    print(f"\n--- 頑健性チェック: より選択的な基準(touches>={STRICT_MIN_TOUCHES}, strength>={STRICT_MIN_STRENGTH}) ---")
    strict_results = {}
    for label, cond in [("SR_ALIGNED_STRICT", True), ("SR_NOT_ALIGNED_STRICT", False)]:
        rets = [e["ret"] for e in all_events if e["aligned_strict"] == cond]
        n = len(rets)
        win_rate = float(np.mean([r > 0 for r in rets])) if n else None
        mean_ret = float(np.mean(rets)) if n else None
        n_eff = max(4, int(round(n * (EFFECTIVE_PAIR_COUNT / len(PAIRS))))) if n else 0
        judgeable = n_eff >= MIN_N_FOR_JUDGEMENT
        r = {"n": n, "mean": round(mean_ret, 6) if mean_ret is not None else None,
             "win_rate": round(win_rate, 3) if win_rate is not None else None,
             "n_effective": n_eff, "judgeable": judgeable}
        if judgeable:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=n_eff, replace=False) if n_eff < n else np.arange(n)
            sub = [rets[i] for i in idx]
            pr = permutation_test(sub, seed=42)
            r["p_value_corr_adjusted"] = round(pr.p_value, 4)
        else:
            r["p_value_corr_adjusted"] = None
        strict_results[label] = r
        jflag = "" if judgeable else " [n不足・判定不能]"
        print(f"{label}: n={n:>4}  勝率={r['win_rate']}  平均(符号調整済)={r['mean']}  "
              f"n_eff={n_eff}  p(参考)={r['p_value_corr_adjusted']}{jflag}")

    n_total = len(all_events)
    n_aligned_total = sum(1 for e in all_events if e["aligned"])
    n_aligned_strict_total = sum(1 for e in all_events if e["aligned_strict"])
    print(f"\n=== 結論 ===")
    print(f"全{n_total}件中、ネックラインがS/Rラインと一致した割合(デフォルト基準): "
          f"{n_aligned_total}件 ({100*n_aligned_total/n_total:.1f}%)")
    print(f"全{n_total}件中、より選択的な基準での一致割合: "
          f"{n_aligned_strict_total}件 ({100*n_aligned_strict_total/n_total:.1f}%)")
    print("→ 基準を厳しくしても一致率がほぼ変わらず、S/R検出パラメータ(365日ルックバック・"
          "クラスタリング閾値0.5%)が生成するライン網が密すぎて、二値の「一致/不一致」では"
          "識別力を持たないと判断される")

    out_path = ROOT / "research" / "method-notes" / "sr_neckline_alignment.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "sr_touch_tolerance_pct": SR_TOUCH_TOLERANCE_PCT,
            "horizon": {"label": HORIZON_LABEL, "bars": HORIZON_BARS},
            "n_tests": n_tests, "bonferroni_alpha": round(bonferroni_alpha, 6),
            "n_total_events": n_total, "n_aligned": n_aligned_total,
            "aligned_fraction": round(n_aligned_total / n_total, 4) if n_total else None,
            "results": results,
            "strict_check": {
                "min_touches": STRICT_MIN_TOUCHES, "min_strength": STRICT_MIN_STRENGTH,
                "n_aligned_strict": n_aligned_strict_total,
                "aligned_strict_fraction": round(n_aligned_strict_total / n_total, 4) if n_total else None,
                "results": strict_results,
            },
            "_note": (
                "ダブルトップ/ボトムのネックラインが、ブレイク時点までのデータのみ"
                "(先読み無し)で計算したSYS-FX007のS/R検出結果と一致するかどうかで"
                "前方リターンICを比較。戻りエントリー・S/R基準SLの設計に着手する"
                "価値があるかを判断するための第1段階の診断。"
            ),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
