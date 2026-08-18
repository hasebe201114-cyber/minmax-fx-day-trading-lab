"""SLの見直し: MAE/MFE分析(提案4)で判明したストップ幅の余裕を踏まえ、
ストップをタイト化した場合の効果を検証する.

背景 (`mae_mfe_decomposition.json`): H1継続文脈イベント(n=673、新方式
40/35/25%段階利確)のWONトレード333件について、イグジット確定バーまでの
MAE(最大逆行幅)中央値は-0.29Rで、現行の-1.0Rストップには大きな余裕が
あると判明した。仮にストップを-0.5Rに縮小しても72.4%、-0.7Rでも87.1%の
勝ちトレードが生き残っていたはず、という事後集計(サバイバル率)は既に
`analyze_mae_mfe_decomposition.py`で算出済み。本スクリプトはこれを実際の
バックテストとして再現し、TP価格水準(1R/2R/3R、元のリスク単位で固定)は
変えずにストップだけをタイト化した場合の平均R・勝率・統計的有意性を測定する。

設計 (事前登録・結果を見る前に固定):
    - ストップ候補 k ∈ {0.7, 0.5, 0.3} (元のリスク単位 risk_original に対する
      倍率)。これらの値は`mae_mfe_decomposition.json`のサバイバル率集計で
      既に使用した閾値(-0.3R/-0.5R/-0.7R)をそのまま流用し、本スクリプト用に
      新たに選び直さない
    - 新ストップ: entry_price - k*risk_original (UP) / entry_price + k*risk_original (DOWN)
    - TP価格水準: entry_price ± {1,2,3}*risk_original (元のリスク単位のまま、
      変更しない)。ストップをタイト化しても目標価格は動かさない、
      すなわちリスクだけを圧縮してペイオフ構造を改善できるかを見る設計
    - R値の単位: 常に risk_original (元のストップ距離) で正規化する。
      ストップをタイト化しても分母は変えない (タイト化そのものの効果を
      「同じモノサシ」で測るため)
    - 建値ストップ移動・ATRトレーリングのロジックは既存(`simulate_scaled_scheme`)
      と同一。TP1到達後に建値へストップを移動する仕様も不変
    - 比較基準: k=1.0 (現行、既知の結果 mean_R=+0.0052, perm_p=0.801, win=49.5%)
    - 多重検定: 新規候補3件(k=0.7/0.5/0.3)でBonferroni補正 (k=1.0は新規検定
      ではなく既知の参照値のため対象外)

出力: research/method-notes/tighter_stop_diagnostic.json
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

PAIRS = base.PAIRS
TRAIN_START, TRAIN_END = base.TRAIN_START, base.TRAIN_END
TP_LEVELS = base.TP_LEVELS
MAX_HOLD_BARS = base.MAX_HOLD_BARS
EFFECTIVE_PAIR_COUNT = 1.70
MIN_N_FOR_JUDGEMENT = 30

STOP_K_CANDIDATES = [0.7, 0.5, 0.3]  # mae_mfe_decomposition.jsonで既に使用した閾値をそのまま流用
N_TESTS = len(STOP_K_CANDIDATES)
BONFERRONI_ALPHA = 0.05 / N_TESTS


def simulate_scaled_scheme_tighter_stop(h1: pd.DataFrame, atr_h1: pd.Series, entry: dict,
                                          trail_mult: float, stop_k: float) -> dict:
    """新方式(40/35/25%段階利確)だが、初期ストップだけを stop_k*risk_original に
    タイト化する。TP価格水準は risk_original ベースのまま変更しない。R値は常に
    risk_original で正規化する(stop_kによらず同じモノサシ)。
    """
    direction = entry["direction"]
    entry_price = entry["entry_price"]
    risk = entry["initial_risk"]  # モノサシは常に元のリスク単位
    stop = entry_price - stop_k * risk if direction == "UP" else entry_price + stop_k * risk
    levels = [(r, frac, entry_price + r * risk if direction == "UP" else entry_price - r * risk, False)
              for r, frac in TP_LEVELS]
    remaining_fraction = 1.0
    realized_r = 0.0
    be_moved = False
    n = len(h1)
    start = entry["entry_idx"] + 1
    end = min(n, start + MAX_HOLD_BARS)
    for i in range(start, end):
        ts = h1.index[i]
        o, h, low, c = float(h1["open"].iloc[i]), float(h1["high"].iloc[i]), float(h1["low"].iloc[i]), float(h1["close"].iloc[i])
        n_levels_hit = sum(1 for lv in levels if lv[3])
        if base.is_weekend_close_time(ts):
            exit_r = (c - entry_price) / risk if direction == "UP" else (entry_price - c) / risk
            reason = "WEEKEND_NO_TP" if n_levels_hit == 0 else "TP_THEN_WEEKEND"
            return {"r": realized_r + remaining_fraction * exit_r, "exit_reason": reason, "n_levels_hit": n_levels_hit}
        stop_hit = (low <= stop) if direction == "UP" else (h >= stop)
        if stop_hit:
            exit_r = (stop - entry_price) / risk if direction == "UP" else (entry_price - stop) / risk
            reason = "SL_INITIAL_NO_TP" if n_levels_hit == 0 else "TP_THEN_SL_TRAIL"
            return {"r": realized_r + remaining_fraction * exit_r, "exit_reason": reason, "n_levels_hit": n_levels_hit}
        for idx_lv, (r_level, frac, price_level, hit) in enumerate(levels):
            if hit or remaining_fraction <= 0:
                continue
            reached = (h >= price_level) if direction == "UP" else (low <= price_level)
            if reached:
                realized_r += frac * r_level
                remaining_fraction -= frac
                levels[idx_lv] = (r_level, frac, price_level, True)
                if not be_moved:
                    stop = max(stop, entry_price) if direction == "UP" else min(stop, entry_price)
                    be_moved = True
        if be_moved and remaining_fraction > 0:
            atr_i = atr_h1.asof(ts)
            if pd.notna(atr_i) and atr_i > 0:
                if direction == "UP":
                    new_stop = o - trail_mult * float(atr_i)
                    stop = max(stop, new_stop)
                else:
                    new_stop = o + trail_mult * float(atr_i)
                    stop = min(stop, new_stop)
        if remaining_fraction <= 1e-9:
            return {"r": realized_r, "exit_reason": "TP_FULL", "n_levels_hit": 3}
    c = float(h1["close"].iloc[end - 1])
    exit_r = (c - entry_price) / risk if direction == "UP" else (entry_price - c) / risk
    n_levels_hit = sum(1 for lv in levels if lv[3])
    return {"r": realized_r + remaining_fraction * exit_r, "exit_reason": "MAX_HOLD", "n_levels_hit": n_levels_hit}


def summarize(results: list[dict], pairs: list[str]) -> dict:
    rs = [r["r"] for r in results]
    n = len(rs)
    mean_r = float(np.mean(rs))
    win_rate = float(np.mean([r > 0 for r in rs]))
    n_eff = max(4, int(round(n * (EFFECTIVE_PAIR_COUNT / len(PAIRS)))))
    judgeable = n_eff >= MIN_N_FOR_JUDGEMENT
    exit_reason_counts: dict[str, int] = {}
    for r in results:
        exit_reason_counts[r["exit_reason"]] = exit_reason_counts.get(r["exit_reason"], 0) + 1
    out = {"n": n, "mean_r": round(mean_r, 4), "win_rate": round(win_rate, 3),
           "n_effective": n_eff, "judgeable": judgeable,
           "exit_reason_counts": dict(sorted(exit_reason_counts.items(), key=lambda kv: -kv[1]))}
    if judgeable:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=n_eff, replace=False) if n_eff < n else np.arange(n)
        sub = [rs[i] for i in idx]
        pr = permutation_test(sub, seed=42)
        out["perm_p"] = round(pr.p_value, 4)
    else:
        out["perm_p"] = None
    return out


def main() -> int:
    print("=== SLの見直し: ストップのタイト化診断 (H1継続文脈、新方式イグジット) ===\n")
    with (ROOT / "research" / "EXP-FX000003" / "10-result" / "double_pattern_params_h1.json").open(encoding="utf-8") as f:
        params = json.load(f)
    trail_mult = params["atr_trail_multiplier"]
    print(f"事前登録した候補: k={STOP_K_CANDIDATES} (mae_mfe_decomposition.jsonのサバイバル率"
          f"集計で使用済みの閾値をそのまま流用)\n")

    entries_by_pair: dict[str, list[dict]] = {}
    for pair in PAIRS:
        m5 = base.load_m5(pair)
        h1 = base.to_h1(m5)
        entries = base.find_continuation_entries(pair, params)
        entries_by_pair[pair] = entries
        print(f"[{pair}] 継続文脈エントリー={len(entries)}件")
    n_total = sum(len(v) for v in entries_by_pair.values())
    print(f"\n全体エントリー数: {n_total}件\n")

    baseline_results: list[dict] = []
    candidate_results: dict[float, list[dict]] = {k: [] for k in STOP_K_CANDIDATES}

    for pair in PAIRS:
        m5 = base.load_m5(pair)
        h1 = base.to_h1(m5)
        atr_h1 = base.atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        for e in entries_by_pair[pair]:
            baseline_results.append(simulate_scaled_scheme_tighter_stop(h1, atr_h1, e, trail_mult, 1.0))
            for k in STOP_K_CANDIDATES:
                candidate_results[k].append(simulate_scaled_scheme_tighter_stop(h1, atr_h1, e, trail_mult, k))

    baseline_summary = summarize(baseline_results, PAIRS)
    print(f"--- k=1.0 (現行、再現確認用) ---")
    print(f"  n={baseline_summary['n']}  平均R={baseline_summary['mean_r']}  勝率={baseline_summary['win_rate']}  "
          f"perm_p={baseline_summary['perm_p']}")
    print(f"  (既知の参照値: mean_R=+0.0052, win=0.495, perm_p=0.801 と比較して再現確認)\n")

    print(f"多重検定: {N_TESTS}件 (k={STOP_K_CANDIDATES}) → Bonferroni閾値 α={BONFERRONI_ALPHA:.5f}\n")

    candidate_summaries = {}
    for k in STOP_K_CANDIDATES:
        summary = summarize(candidate_results[k], PAIRS)
        summary["survives_bonferroni"] = bool(summary["perm_p"] is not None and summary["perm_p"] < BONFERRONI_ALPHA)
        candidate_summaries[k] = summary
        sig = " **" if summary["survives_bonferroni"] else (
            " *" if summary["perm_p"] is not None and summary["perm_p"] < 0.05 else "")
        print(f"--- k={k} ---")
        print(f"  n={summary['n']}  平均R={summary['mean_r']}  勝率={summary['win_rate']}  "
              f"perm_p={summary['perm_p']}{sig}")
        print(f"  決済内訳: {summary['exit_reason_counts']}\n")

    print("=== 結論 ===")
    best_k = max(candidate_summaries, key=lambda k: candidate_summaries[k]["mean_r"])
    print(f"平均Rが最良だった候補: k={best_k} (平均R={candidate_summaries[best_k]['mean_r']}, "
          f"baseline={baseline_summary['mean_r']})")
    any_significant = any(s["survives_bonferroni"] for s in candidate_summaries.values())
    print(f"Bonferroni補正後に有意だった候補: {'あり' if any_significant else 'なし'}")

    out_path = ROOT / "research" / "method-notes" / "tighter_stop_diagnostic.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "timeframe": "H1",
            "context": "continuation_only, scaled_exit_scheme(40/35/25%)",
            "stop_k_candidates": STOP_K_CANDIDATES,
            "n_tests": N_TESTS, "bonferroni_alpha": round(BONFERRONI_ALPHA, 6),
            "n_total_events": n_total,
            "baseline_k1_0": baseline_summary,
            "candidates": {str(k): v for k, v in candidate_summaries.items()},
            "_note": (
                "TP価格水準(1R/2R/3R、risk_original基準)は固定したまま、初期ストップ"
                "だけをrisk_originalに対するk倍へタイト化した場合の効果を測定。"
                "R値は常にrisk_originalで正規化(kによらず同じモノサシ)。方向性エッジ"
                "自体は既に非有意と確定済みのため、この結果はペイオフ構造改善のみを"
                "対象とした診断であり、有意化しても即採用可を意味しない(K1m〜K7m等"
                "他のKPIも別途要確認)。"
            ),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
