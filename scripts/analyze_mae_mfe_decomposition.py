"""提案4: MAE/MFE分析によるエントリー/イグジットの切り分け.

背景 (pure-stirring-turtle.md 提案4): SYS-FX009系の実測で「1:1ターゲット未達
トレードの勝率はごく低い(ほぼ全損)」という観測があったが、これが (a) エントリー
自体の方向性が外れている(値動きが最初から逆行し続ける)のか、(b) エントリーの
方向性自体はおおむね合っているがストップ/イグジット設計が早すぎる/機能不足で
含み益を確定できていないのか、切り分けができていなかった。

本スクリプトは、H1継続文脈イベント(`analyze_scaled_exit_diagnostic.py`で確立
した検出ロジック・パラメータをそのまま再利用、n=673、反転文脈は除外)について、
本番の新方式イグジット(40/35/25%段階利確、既存の`simulate_scaled_scheme`と
同一ロジック)で確定した実際の結果と並行して、ストップに拘束されない生の価格
経路からMFE(最大順行幅)・MAE(最大逆行幅)をRマルチプル単位で追跡する。

事前登録 (結果を見る前に固定):
    - トレード母集団: H1継続文脈イベント(n=673、`find_continuation_entries`と
      同一ロジック)。イグジット方式は新方式(40/35/25%段階利確)のみを使用
      (旧方式との比較はここでは行わない)
    - 生の価格経路の追跡窓: 実イグジットと同じ制約(週末強制クローズ・
      MAX_HOLD_BARS=240)を適用する。無制約(週末を無視した仮想延長)は行わない
    - 分類: 新方式の最終Rマルチプルが正ならWON、負ならLOST、0ならFLAT
      (構造上、TP1(1R)に一度でも到達すると建値ストップへ移動するため、
      n_levels_hit>=1のトレードは必ずR>=0.4となりWONに分類される。したがって
      LOSTは構造的にn_levels_hit=0のトレードのみ)
    - LOST トレードについて、実イグジット確定バーの「直前まで」のMFE
      (=イグジットバー自体は含めない、素直に反転して負けた場合と、イグジット
      バー内でTPとSLが同時に条件を満たした「同一バー競合」の場合を区別するため)
      と、イグジットバー「時点」のMFE(同一バー競合の検出用)を分けて集計する
    - WON トレードについて、実イグジット確定バーまでのMAE(最大逆行幅)の
      分布を集計する

出力: research/method-notes/mae_mfe_decomposition.json
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

PAIRS = base.PAIRS
TRAIN_START, TRAIN_END = base.TRAIN_START, base.TRAIN_END
TP_LEVELS = base.TP_LEVELS
MAX_HOLD_BARS = base.MAX_HOLD_BARS


def simulate_with_mae_mfe_path(h1: pd.DataFrame, atr_h1: pd.Series, entry: dict, trail_mult: float) -> dict:
    """新方式(40/35/25%段階利確)の実イグジットを計算しつつ、並行してストップに
    拘束されない生のMFE/MAE経路(Rマルチプル単位)を同じ観測窓(週末クローズ・
    MAX_HOLD_BARS)で追跡する。

    Returns:
        r, exit_reason, n_levels_hit: `simulate_scaled_scheme`と同一
        mfe_before_exit: イグジット確定バーの直前まで(バー未満)の最大順行幅
        mfe_at_exit_bar: イグジット確定バー単体でのバー高値/安値による順行幅
            (同一バー内でTPとSLが両方条件を満たした「同一バー競合」の検出用)
        mae_before_exit: イグジット確定バーの直前までの最大逆行幅 (負値)
    """
    direction = entry["direction"]
    entry_price = entry["entry_price"]
    risk = entry["initial_risk"]
    stop = entry["stop0"]
    levels = [(r, frac, entry_price + r * risk if direction == "UP" else entry_price - r * risk, False)
              for r, frac in TP_LEVELS]
    remaining_fraction = 1.0
    realized_r = 0.0
    be_moved = False
    n = len(h1)
    start = entry["entry_idx"] + 1
    end = min(n, start + MAX_HOLD_BARS)

    mfe_before = 0.0
    mae_before = 0.0

    def bar_mfe(h: float, low: float) -> float:
        return (h - entry_price) / risk if direction == "UP" else (entry_price - low) / risk

    def bar_mae(h: float, low: float) -> float:
        return (low - entry_price) / risk if direction == "UP" else (entry_price - h) / risk

    for i in range(start, end):
        ts = h1.index[i]
        o, h, low, c = float(h1["open"].iloc[i]), float(h1["high"].iloc[i]), float(h1["low"].iloc[i]), float(h1["close"].iloc[i])
        n_levels_hit = sum(1 for lv in levels if lv[3])
        this_bar_mfe = bar_mfe(h, low)
        this_bar_mae = bar_mae(h, low)

        if base.is_weekend_close_time(ts):
            exit_r = (c - entry_price) / risk if direction == "UP" else (entry_price - c) / risk
            reason = "WEEKEND_NO_TP" if n_levels_hit == 0 else "TP_THEN_WEEKEND"
            return {"r": realized_r + remaining_fraction * exit_r, "exit_reason": reason,
                    "n_levels_hit": n_levels_hit, "mfe_before_exit": round(mfe_before, 4),
                    "mfe_at_exit_bar": round(this_bar_mfe, 4), "mae_before_exit": round(mae_before, 4)}
        stop_hit = (low <= stop) if direction == "UP" else (h >= stop)
        if stop_hit:
            exit_r = (stop - entry_price) / risk if direction == "UP" else (entry_price - stop) / risk
            reason = "SL_INITIAL_NO_TP" if n_levels_hit == 0 else "TP_THEN_SL_TRAIL"
            return {"r": realized_r + remaining_fraction * exit_r, "exit_reason": reason,
                    "n_levels_hit": n_levels_hit, "mfe_before_exit": round(mfe_before, 4),
                    "mfe_at_exit_bar": round(this_bar_mfe, 4), "mae_before_exit": round(mae_before, 4)}

        mfe_before = max(mfe_before, this_bar_mfe)
        mae_before = min(mae_before, this_bar_mae)

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
            return {"r": realized_r, "exit_reason": "TP_FULL", "n_levels_hit": 3,
                    "mfe_before_exit": round(mfe_before, 4), "mfe_at_exit_bar": round(this_bar_mfe, 4),
                    "mae_before_exit": round(mae_before, 4)}

    c = float(h1["close"].iloc[end - 1])
    exit_r = (c - entry_price) / risk if direction == "UP" else (entry_price - c) / risk
    n_levels_hit = sum(1 for lv in levels if lv[3])
    return {"r": realized_r + remaining_fraction * exit_r, "exit_reason": "MAX_HOLD",
            "n_levels_hit": n_levels_hit, "mfe_before_exit": round(mfe_before, 4),
            "mfe_at_exit_bar": 0.0, "mae_before_exit": round(mae_before, 4)}


def percentiles(values: list[float], qs: list[float]) -> dict:
    if not values:
        return {f"p{int(q*100)}": None for q in qs}
    arr = np.array(values)
    return {f"p{int(q*100)}": round(float(np.percentile(arr, q * 100)), 4) for q in qs}


def main() -> int:
    print("=== 提案4: MAE/MFE分析によるエントリー/イグジットの切り分け (H1継続文脈、新方式イグジット) ===\n")
    with (ROOT / "research" / "EXP-FX000003" / "10-result" / "double_pattern_params_h1.json").open(encoding="utf-8") as f:
        params = json.load(f)
    trail_mult = params["atr_trail_multiplier"]

    all_results: list[dict] = []
    for pair in PAIRS:
        m5 = base.load_m5(pair)
        h1 = base.to_h1(m5)
        atr_h1 = base.atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        entries = base.find_continuation_entries(pair, params)
        for e in entries:
            res = simulate_with_mae_mfe_path(h1, atr_h1, e, trail_mult)
            res["pair"] = pair
            all_results.append(res)
        print(f"[{pair}] 継続文脈エントリー={len(entries)}件")

    n_total = len(all_results)
    print(f"\n全体エントリー数: {n_total}件\n")

    won = [r for r in all_results if r["r"] > 1e-9]
    lost = [r for r in all_results if r["r"] < -1e-9]
    flat = [r for r in all_results if abs(r["r"]) <= 1e-9]
    print(f"WON(最終R>0)={len(won)}件  LOST(最終R<0)={len(lost)}件  FLAT(最終R=0)={len(flat)}件\n")

    # LOST トレードの検証: 構造上 n_levels_hit==0 のはず (念のため確認)
    lost_with_tp = [r for r in lost if r["n_levels_hit"] > 0]
    if lost_with_tp:
        print(f"[想定外] TP到達済みなのにLOSTになったトレード: {len(lost_with_tp)}件 (要調査)\n")

    print("--- LOSTトレードの分解: エントリー方向性 vs イグジット設計 ---")
    mfe_before_lost = [r["mfe_before_exit"] for r in lost]
    mfe_at_exit_lost = [r["mfe_at_exit_bar"] for r in lost]
    n_never_favorable = sum(1 for v in mfe_before_lost if v <= 0.1)
    n_moderately_favorable = sum(1 for v in mfe_before_lost if 0.1 < v < 1.0)
    n_same_bar_conflict = sum(1 for v in mfe_at_exit_lost if v >= 1.0)
    print(f"  イグジットバー直前までのMFE分布(n={len(lost)}): "
          f"{percentiles(mfe_before_lost, [0.25, 0.5, 0.75, 0.9])}")
    print(f"  一度も含み益0.1R超えず反転(=エントリー方向性が外れた可能性): {n_never_favorable}件 "
          f"({100*n_never_favorable/len(lost):.1f}%)")
    print(f"  含み益0.1R〜1.0Rまで到達後に反転(=方向は合っていたがイグジット未確定): "
          f"{n_moderately_favorable}件 ({100*n_moderately_favorable/len(lost):.1f}%)")
    print(f"  イグジット確定バー自体でTP1水準(1.0R)以上に到達(=同一バー内TP/SL競合、"
          f"保守的ルールでSL優先処理された可能性): {n_same_bar_conflict}件 "
          f"({100*n_same_bar_conflict/len(lost):.1f}%)")

    print("\n--- WONトレードの分解: ストップ幅の妥当性 ---")
    mae_won = [r["mae_before_exit"] for r in won]
    mae_pctl = percentiles(mae_won, [0.1, 0.25, 0.5, 0.75])
    print(f"  イグジット確定バー直前までのMAE分布(n={len(won)}): {mae_pctl}")
    for thresh in [-0.3, -0.5, -0.7, -1.0]:
        n_would_survive = sum(1 for v in mae_won if v >= thresh)
        print(f"  仮にストップを{thresh}Rに縮小した場合でも生き残ったはずの勝ちトレード: "
              f"{n_would_survive}/{len(won)}件 ({100*n_would_survive/len(won):.1f}%)")

    print(f"\n=== 結論(暫定) ===")
    if len(lost) > 0:
        print(f"LOSTトレード{len(lost)}件のうち、{100*n_never_favorable/len(lost):.1f}%は一度も"
              f"含み益が乗らずに反転しており、エントリー方向性そのものが外れているケースが主体。")
        print(f"一方、{100*n_moderately_favorable/len(lost):.1f}%は途中まで含み益が乗ってから反転しており、"
              f"これらはイグジット設計(より早い部分利確・タイトなトレーリング)で救済できる可能性がある候補。")
    print(f"WONトレードのMAE中央値は{mae_pctl.get('p50')}Rで、現行の-1.0Rストップには"
          f"{'余裕がある' if mae_pctl.get('p50') is not None and mae_pctl['p50'] > -0.5 else '余裕は少ない'}。")

    out_path = ROOT / "research" / "method-notes" / "mae_mfe_decomposition.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "timeframe": "H1",
            "context": "continuation_only, scaled_exit_scheme(40/35/25%)のみ",
            "n_total": n_total,
            "n_won": len(won), "n_lost": len(lost), "n_flat": len(flat),
            "lost_breakdown": {
                "mfe_before_exit_percentiles": percentiles(mfe_before_lost, [0.25, 0.5, 0.75, 0.9]),
                "n_never_favorable_mfe_le_0_1R": n_never_favorable,
                "n_moderately_favorable_0_1_to_1_0R": n_moderately_favorable,
                "n_same_bar_tp_sl_conflict": n_same_bar_conflict,
            },
            "won_breakdown": {
                "mae_before_exit_percentiles": mae_pctl,
                "hypothetical_survival_rate_by_stop_width": {
                    str(t): sum(1 for v in mae_won if v >= t) / len(won) if won else None
                    for t in [-0.3, -0.5, -0.7, -1.0]
                },
            },
            "_note": (
                "H1継続文脈イベント(n=673、新方式イグジットのみ)のMAE/MFE分解。"
                "LOSTトレードはn_levels_hit==0の構造(TP1到達で建値ストップへ移動する"
                "ため、TP到達後は必ずR>=0.4でWONになる)。方向性エッジ自体は本セッション"
                "の一連のIC分析で無い(統計的に有意でない)と既に確認済みのため、本分析は"
                "「エッジがある前提での設計診断」であり、それ自体が採用可否を左右する"
                "ものではない。"
            ),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
