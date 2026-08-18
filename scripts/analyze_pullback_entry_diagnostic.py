"""保留事項C 続き: 戻りエントリー(ブレイク&リテスト)のタイミング検証.

背景: 司令塔から3点(SL決定方法・戻りのエントリータイミング・レジサポ
ライン考慮)の見直し依頼のうち、レジサポライン軸は`analyze_sr_neckline_alignment.py`
の結果(96.3%が一致・識別力なし)を受けて一旦停止(選択肢c)。本スクリプトは
残る2点のうち「戻りのエントリータイミング」を検証する。SLの決定方法は
変数を1つに絞るため、既存の計算式(pattern_extreme ± stop_buffer_atr*ATR)
をそのまま維持する(参照するATRバーが変わるのみ)。

現行本番: ネックライン ブレイク確定バー(継続文脈ゲート通過)の終値で
即エントリー。

本スクリプトの新方式: ブレイク確定後、価格がネックライン付近まで
戻る(リテスト)のを一定期間待ち、リテスト確認バーの終値でエントリー
する「ブレイク&リテスト」方式。戻りを待つ間にpattern_extremeを
逆行突破した場合はセットアップ無効(トレードなし)として扱う。

事前登録 (結果を見る前に固定):
    - MAX_PULLBACK_WAIT_BARS = 24 (H1、1日相当)
    - PULLBACK_TOLERANCE_ATR = 0.3 (リテスト判定: ネックラインからこの
      ATR倍数以内に安値/高値が接近すること)
    - リテスト確認条件 (UP方向の例): 待機バーkで
      low_k <= neckline + tol*ATR_k かつ close_k >= neckline
      (ネックラインへ接近しつつ、終値はネックラインを維持=リテストが
      機能している証拠。DOWN方向は符号を反転)
    - 無効化条件: 待機中にclose_kがpattern_extremeを逆行突破したら
      そのセットアップは以後トレードなし(タイムアウトと区別して集計)
    - SL式は変更しない。stop0 = pattern_extreme ± stop_buffer_atr*ATR_k
      (ATRの参照バーがブレイク確定バーjからリテスト確認バーkに変わる
      のみで、式自体は本番と同一)
    - 比較対象のイグジット方式: 新方式(40/35/25%段階利確、既に採用
      方向として確定済み)のみを使用。旧方式との比較はここでは行わない
    - 比較方法: 有効なリテストが発生したイベントに限定したペア比較
      (即時エントリー vs 戻りエントリー、同一イベント・同一イグジット
      方式)。加えて戻りエントリー単体の前方成績もpermutation testで検定
    - 多重検定: 新規に立てる仮説は1件(戻りエントリーの平均Rがゼロと
      有意に異なるか)。Bonferroni補正の対象はこの1件のみ

出力: research/method-notes/pullback_entry_diagnostic.json
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
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402
from minmax_fx_dt.strategy.support_resistance import zigzag_pivots_typed  # noqa: E402

PAIRS = base.PAIRS
TRAIN_START, TRAIN_END = base.TRAIN_START, base.TRAIN_END
EFFECTIVE_PAIR_COUNT = 1.70
MIN_N_FOR_JUDGEMENT = 30

MAX_PULLBACK_WAIT_BARS = 24
PULLBACK_TOLERANCE_ATR = 0.3
N_TESTS = 1
BONFERRONI_ALPHA = 0.05 / N_TESTS


def find_events(pair: str, params: dict) -> list[dict]:
    """ブレイク確定イベントを検出し、即時エントリー用情報とパターン情報の両方を返す."""
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
            atr_j = atr_h1.iloc[j]
            if pd.isna(atr_j) or atr_j <= 0:
                break
            direction = "DOWN" if kind1 == "HIGH" else "UP"
            immediate_entry_price = float(h1["close"].iloc[j])
            buffer_j = buffer_atr * float(atr_j)
            immediate_stop0 = pattern_extreme + buffer_j if direction == "DOWN" else pattern_extreme - buffer_j
            immediate_risk = abs(immediate_entry_price - immediate_stop0)
            if immediate_risk <= 0:
                break
            events.append(dict(
                pair=pair, direction=direction, break_idx=j, neckline=neckline,
                pattern_extreme=pattern_extreme,
                immediate=dict(pair=pair, direction=direction, entry_idx=j,
                                entry_time=str(h1.index[j]), entry_price=immediate_entry_price,
                                stop0=immediate_stop0, initial_risk=immediate_risk),
            ))
            break

    return events


def resolve_pullback_entry(h1: pd.DataFrame, atr_h1: pd.Series, ev: dict, buffer_atr: float) -> tuple[dict | None, str]:
    """戻りエントリーを解決する. 戻り値: (entryまたはNone, ステータス).

    ステータス: TRIGGERED / INVALIDATED / TIMEOUT
    """
    direction = ev["direction"]
    neckline = ev["neckline"]
    pattern_extreme = ev["pattern_extreme"]
    j = ev["break_idx"]
    end = min(len(h1), j + 1 + MAX_PULLBACK_WAIT_BARS)
    for k in range(j + 1, end):
        close_k = float(h1["close"].iloc[k])
        invalidated = (close_k < pattern_extreme) if direction == "UP" else (close_k > pattern_extreme)
        if invalidated:
            return None, "INVALIDATED"
        atr_k = atr_h1.iloc[k]
        if pd.isna(atr_k) or atr_k <= 0:
            continue
        low_k, high_k = float(h1["low"].iloc[k]), float(h1["high"].iloc[k])
        tol_price = PULLBACK_TOLERANCE_ATR * float(atr_k)
        if direction == "UP":
            retested = low_k <= neckline + tol_price and close_k >= neckline
        else:
            retested = high_k >= neckline - tol_price and close_k <= neckline
        if not retested:
            continue
        buffer_k = buffer_atr * float(atr_k)
        stop0 = pattern_extreme + buffer_k if direction == "DOWN" else pattern_extreme - buffer_k
        initial_risk = abs(close_k - stop0)
        if initial_risk <= 0:
            continue
        entry = dict(pair=ev["pair"], direction=direction, entry_idx=k, entry_time=str(h1.index[k]),
                      entry_price=close_k, stop0=stop0, initial_risk=initial_risk)
        return entry, "TRIGGERED"
    return None, "TIMEOUT"


def main() -> int:
    print("=== 保留事項C続き: 戻りエントリー(ブレイク&リテスト)診断 (Train期間・継続文脈・新方式イグジットのみ) ===\n")
    with (ROOT / "research" / "EXP-FX000003" / "10-result" / "double_pattern_params_h1.json").open(encoding="utf-8") as f:
        params = json.load(f)
    trail_mult = params["atr_trail_multiplier"]
    buffer_atr = params["stop_buffer_atr"]
    print(f"事前登録: MAX_PULLBACK_WAIT_BARS={MAX_PULLBACK_WAIT_BARS}, "
          f"PULLBACK_TOLERANCE_ATR={PULLBACK_TOLERANCE_ATR}, SL式は不変 "
          f"(stop_buffer_atr={buffer_atr}をそのまま流用)\n")

    status_counts: dict[str, int] = {"TRIGGERED": 0, "INVALIDATED": 0, "TIMEOUT": 0}
    matched_immediate: list[dict] = []
    matched_pullback: list[dict] = []
    n_events_by_pair: dict[str, int] = {}

    for pair in PAIRS:
        m5 = base.load_m5(pair)
        h1 = base.to_h1(m5)
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        events = find_events(pair, params)
        n_events_by_pair[pair] = len(events)
        n_trig = 0
        for ev in events:
            pullback_entry, status = resolve_pullback_entry(h1, atr_h1, ev, buffer_atr)
            status_counts[status] += 1
            if status != "TRIGGERED":
                continue
            n_trig += 1
            matched_immediate.append(base.simulate_scaled_scheme(h1, atr_h1, ev["immediate"], trail_mult))
            matched_pullback.append(base.simulate_scaled_scheme(h1, atr_h1, pullback_entry, trail_mult))
        print(f"[{pair}] ブレイクイベント={len(events)}件  戻り成立(TRIGGERED)={n_trig}件")

    n_total = sum(n_events_by_pair.values())
    print(f"\n全体ブレイクイベント数: {n_total}件")
    print(f"内訳: TRIGGERED(戻り成立)={status_counts['TRIGGERED']}件 "
          f"({100*status_counts['TRIGGERED']/n_total:.1f}%)  "
          f"INVALIDATED(逆行で無効化)={status_counts['INVALIDATED']}件 "
          f"({100*status_counts['INVALIDATED']/n_total:.1f}%)  "
          f"TIMEOUT(期限内に戻らず)={status_counts['TIMEOUT']}件 "
          f"({100*status_counts['TIMEOUT']/n_total:.1f}%)\n")

    print(f"多重検定: {N_TESTS}件 → Bonferroni閾値 α={BONFERRONI_ALPHA:.5f}\n")

    def summarize_matched(results: list[dict], label: str) -> dict:
        rs = [r["r"] for r in results]
        n = len(rs)
        mean_r = float(np.mean(rs)) if n else None
        win_rate = float(np.mean([r > 0 for r in rs])) if n else None
        n_eff = max(4, int(round(n * (EFFECTIVE_PAIR_COUNT / len(PAIRS))))) if n else 0
        judgeable = n_eff >= MIN_N_FOR_JUDGEMENT
        out = {"n": n, "mean_r": round(mean_r, 4) if mean_r is not None else None,
               "win_rate": round(win_rate, 3) if win_rate is not None else None,
               "n_effective": n_eff, "judgeable": judgeable}
        if judgeable:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=n_eff, replace=False) if n_eff < n else np.arange(n)
            sub = [rs[i] for i in idx]
            pr = permutation_test(sub, seed=42)
            out["perm_p"] = round(pr.p_value, 4)
            out["survives_bonferroni"] = bool(pr.p_value < BONFERRONI_ALPHA) if label == "戻りエントリー" else None
        else:
            out["perm_p"] = None
            out["survives_bonferroni"] = None
        print(f"{label}: n={n}  平均R={out['mean_r']}  勝率={out['win_rate']}  "
              f"n_eff={n_eff}  perm_p={out['perm_p']}"
              f"{'' if judgeable else '  [n不足・判定不能]'}")
        return out

    immediate_summary = summarize_matched(matched_immediate, "即時エントリー(マッチ後、同一母集団)")
    pullback_summary = summarize_matched(matched_pullback, "戻りエントリー")

    diff_summary = {}
    if len(matched_immediate) == len(matched_pullback) and matched_immediate:
        imm_rs = [r["r"] for r in matched_immediate]
        pb_rs = [r["r"] for r in matched_pullback]
        diffs = [p - i for p, i in zip(pb_rs, imm_rs)]
        mean_diff = float(np.mean(diffs))
        pullback_better_rate = float(np.mean([d > 0 for d in diffs]))
        diff_summary = {"n": len(diffs), "mean_diff_r": round(mean_diff, 4),
                         "pullback_better_fraction": round(pullback_better_rate, 3)}
        print(f"\n--- ペア差分 (戻りエントリー - 即時エントリー、同一イベント・新方式イグジット) ---")
        print(f"  平均差分R={diff_summary['mean_diff_r']}  戻りエントリーが上回った割合={diff_summary['pullback_better_fraction']}")

    print(f"\n=== 結論 ===")
    trigger_rate = status_counts["TRIGGERED"] / n_total if n_total else None
    print(f"戻り成立率: {100*trigger_rate:.1f}% (この方式を採用すると母数が{n_total}件→"
          f"{status_counts['TRIGGERED']}件に減る)")
    if pullback_summary["judgeable"]:
        verdict = "有意" if pullback_summary["survives_bonferroni"] else "非有意"
        print(f"戻りエントリー単体の平均Rはゼロと{verdict} (p={pullback_summary['perm_p']}, α={BONFERRONI_ALPHA:.5f})")
    else:
        print("戻りエントリー単体はn不足のため判定不能")
    if diff_summary:
        direction_word = "改善" if diff_summary["mean_diff_r"] > 0 else "悪化"
        print(f"即時エントリーとの比較: 平均{direction_word} {abs(diff_summary['mean_diff_r'])}R "
              f"(同一イベント・新方式イグジットでの対応比較)")

    out_path = ROOT / "research" / "method-notes" / "pullback_entry_diagnostic.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "max_pullback_wait_bars": MAX_PULLBACK_WAIT_BARS,
            "pullback_tolerance_atr": PULLBACK_TOLERANCE_ATR,
            "n_tests": N_TESTS, "bonferroni_alpha": round(BONFERRONI_ALPHA, 6),
            "n_total_break_events": n_total,
            "status_counts": status_counts,
            "trigger_rate": round(trigger_rate, 4) if trigger_rate is not None else None,
            "n_events_by_pair": n_events_by_pair,
            "immediate_entry_matched": immediate_summary,
            "pullback_entry": pullback_summary,
            "paired_diff": diff_summary,
            "_note": (
                "戻りのエントリータイミング(ブレイク&リテスト)を、即時エントリーの"
                "マッチド・ペア比較で検証。SL式は不変(参照ATRバーのみ変更)。"
                "イグジットは新方式(40/35/25%段階利確)のみで比較。戻りが成立した"
                "イベントに限定した診断であり、正式なバックテストKPI評価ではない。"
            ),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
