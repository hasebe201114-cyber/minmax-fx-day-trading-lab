"""SYS-FX009派生 診断: H1タイムフレーム + 段階利確(40/35/25%)の効果測定.

背景: 司令塔から3点の指摘を受けた。
    1. H4での代表例(継続文脈)がレンジの中の値動きにも見え、綺麗なダブルボトム
       と言い切れない
    2. H4→H1に変更してサンプル数を増やし、綺麗なダブルボトムを検出する
    3. 1:1到達で全量利確(旧: ブレイクイーブン+ATRトレーリング)ではなく、
       40%を利確、その後残りを伸ばして35%・25%の利確ポイントを追加設定する
       段階利確方式に変更する

本スクリプトは3.の効果を、H1タイムフレーム・修正済みZigZag・新規導出した
H1パラメータ(derive_double_pattern_params_h1.py)を使い、実際の本番エントリー
条件(継続文脈のみ、pattern_detection.evaluate_double_pattern_signalと同じ
LT一致ゲート)で検出したパターンイベントに対して、旧方式(1R到達で全量+BE
+ATRトレーリング)と新方式(1R=40%/2R=35%/3R=25%、1R到達後は残玉をBE+ATR
トレーリング)をバーパスシミュレーション(H1のOHLCのみ、スプレッド/スリッページ
は含めない簡易版、値幅はすべてRマルチプル単位)で比較する。

事前登録 (結果を見る前に固定):
    - TP1=1R(40%) / TP2=2R(35%) / TP3=3R(25%) — 司令塔指示の按分比率をそのまま採用
    - 1R到達後、残玉のストップは建値へ移動し、以降はATRトレーリング(既存の
      atr_trail_multiplier=2.44を流用、H1パラメータは別途導出したがトレーリング
      幅の再導出はここでは行わない)
    - 同一バー内でストップとTPが両方条件を満たす場合はストップを先に処理する
      (保守的な順序、楽観的バイアス回避)
    - 週末強制クローズ(土曜06:00 JST)は既存ルールをそのまま踏襲
    - 効果測定は継続文脈のみ(本番のevaluate_double_pattern_signalと同じゲート)

出力: research/method-notes/scaled_exit_diagnostic.json
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

from minmax_fx_dt.backtest.permutation import permutation_test
from minmax_fx_dt.strategy.indicators import atr as atr_ind
from minmax_fx_dt.strategy.support_resistance import zigzag_pivots_typed

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
EFFECTIVE_PAIR_COUNT = 1.70  # market_character.json 実測値
MIN_N_FOR_JUDGEMENT = 30

# 事前登録: 段階利確の按分比率・R水準 (結果を見る前に固定、司令塔指示をそのまま反映)
TP_LEVELS = [(1.0, 0.40), (2.0, 0.35), (3.0, 0.25)]
LT_SMA_SHORT, LT_SMA_LONG = 10, 20
MAX_HOLD_BARS = 24 * 10  # 10日相当 (H1、暴走ループ防止の安全上限)


def load_m5(pair: str) -> pd.DataFrame:
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]


def to_h1(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: m5[c].resample("1h").agg(a) for c, a in
                          [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]}).dropna()


def to_d1(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: m5[c].resample("D").agg(a) for c, a in
                          [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]}).dropna()


def lt_direction_series(d1: pd.DataFrame) -> pd.Series:
    sma_short = d1["close"].rolling(LT_SMA_SHORT, min_periods=LT_SMA_SHORT).mean()
    sma_long = d1["close"].rolling(LT_SMA_LONG, min_periods=LT_SMA_LONG).mean()
    direction = pd.Series("NONE", index=d1.index, dtype="object")
    direction = direction.mask(sma_short > sma_long, "UP")
    direction = direction.mask(sma_short < sma_long, "DOWN")
    return direction


def alternating_triplets(pivots: list[tuple[int, str]]) -> list[tuple[int, str, int, str, int, str]]:
    triplets = []
    for i in range(2, len(pivots)):
        idx1, kind1 = pivots[i - 2]
        idx2, kind2 = pivots[i - 1]
        idx3, kind3 = pivots[i]
        if kind1 == kind3 and kind1 != kind2:
            triplets.append((idx1, kind1, idx2, kind2, idx3, kind3))
    return triplets


def is_weekend_close_time(ts: pd.Timestamp) -> bool:
    """土曜06:00 JST以降(既存simulator.pyのルールと同一趣旨)."""
    return ts.weekday() == 5 and ts.hour >= 6


def find_continuation_entries(pair: str, params: dict) -> list[dict]:
    """本番と同じ継続文脈ゲートでH1エントリーイベントを検出する."""
    m5 = load_m5(pair)
    h1 = to_h1(m5)
    d1 = to_d1(m5)
    atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    lt_dir = lt_direction_series(d1)

    pivots = zigzag_pivots_typed(h1["high"], h1["low"], atr_h1, params["zigzag_threshold_atr"])
    triplets = alternating_triplets(pivots)
    tol = params["pattern_tolerance_atr"]
    buffer_atr = params["stop_buffer_atr"]
    cap = params["break_search_cap_bars"]

    entries = []
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
            buffer = buffer_atr * float(atr_entry)
            stop0 = pattern_extreme + buffer if direction == "DOWN" else pattern_extreme - buffer
            initial_risk = abs(entry_price - stop0)
            if initial_risk <= 0:
                break
            entries.append(dict(
                pair=pair, direction=direction, entry_idx=j, entry_time=str(h1.index[j]),
                entry_price=entry_price, stop0=stop0, initial_risk=initial_risk,
            ))
            break

    return entries


def count_reversal_signals(pair: str, params: dict) -> int:
    """参考値: 継続文脈ゲートで除外される反転文脈(LT不一致)のブレイクが何件あるかを数える.

    本番のevaluate_double_pattern_signal・find_continuation_entries()と同じく
    シミュレーション対象には一切含めない。除外されていることの確認用。
    """
    m5 = load_m5(pair)
    h1 = to_h1(m5)
    d1 = to_d1(m5)
    atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    lt_dir = lt_direction_series(d1)

    pivots = zigzag_pivots_typed(h1["high"], h1["low"], atr_h1, params["zigzag_threshold_atr"])
    triplets = alternating_triplets(pivots)
    tol = params["pattern_tolerance_atr"]
    cap = params["break_search_cap_bars"]

    n_reversal = 0
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
                n_reversal += 1
            break
    return n_reversal


def simulate_old_scheme(h1: pd.DataFrame, atr_h1: pd.Series, entry: dict, trail_mult: float) -> dict:
    """旧方式: 1R到達で全量、以降BE+ATRトレーリング。

    Returns: {"r": Rマルチプル, "exit_reason": 決済種類, "target_reached": bool}
    決済種類: SL_INITIAL(1R未到達のままSLで損切り) / WEEKEND_BEFORE_TARGET(1R未到達のまま週末クローズ)
             / TARGET_THEN_SL_TRAIL(1R到達後BE/トレーリングSLで決済) / TARGET_THEN_WEEKEND(1R到達後週末クローズ)
             / MAX_HOLD(保有上限、安全策)
    """
    direction = entry["direction"]
    entry_price = entry["entry_price"]
    risk = entry["initial_risk"]
    stop = entry["stop0"]
    target = entry_price + risk if direction == "UP" else entry_price - risk
    target_reached = False
    n = len(h1)
    start = entry["entry_idx"] + 1
    end = min(n, start + MAX_HOLD_BARS)
    for i in range(start, end):
        ts = h1.index[i]
        o, h, low, c = float(h1["open"].iloc[i]), float(h1["high"].iloc[i]), float(h1["low"].iloc[i]), float(h1["close"].iloc[i])
        if is_weekend_close_time(ts):
            r = (c - entry_price) / risk if direction == "UP" else (entry_price - c) / risk
            return {"r": r, "exit_reason": "TARGET_THEN_WEEKEND" if target_reached else "WEEKEND_BEFORE_TARGET",
                    "target_reached": target_reached}
        stop_hit = (low <= stop) if direction == "UP" else (h >= stop)
        if stop_hit:
            r = (stop - entry_price) / risk if direction == "UP" else (entry_price - stop) / risk
            return {"r": r, "exit_reason": "TARGET_THEN_SL_TRAIL" if target_reached else "SL_INITIAL",
                    "target_reached": target_reached}
        if not target_reached:
            reached_now = (h >= target) if direction == "UP" else (low <= target)
            if reached_now:
                target_reached = True
                stop = max(stop, entry_price) if direction == "UP" else min(stop, entry_price)
        if target_reached:
            atr_i = atr_h1.asof(ts)
            if pd.notna(atr_i) and atr_i > 0:
                if direction == "UP":
                    new_stop = o - trail_mult * float(atr_i)
                    stop = max(stop, new_stop)
                else:
                    new_stop = o + trail_mult * float(atr_i)
                    stop = min(stop, new_stop)
    c = float(h1["close"].iloc[end - 1])
    r = (c - entry_price) / risk if direction == "UP" else (entry_price - c) / risk
    return {"r": r, "exit_reason": "MAX_HOLD", "target_reached": target_reached}


def simulate_scaled_scheme(h1: pd.DataFrame, atr_h1: pd.Series, entry: dict, trail_mult: float) -> dict:
    """新方式: 1R=40%/2R=35%/3R=25%の段階利確、1R到達後はBE+ATRトレーリング。

    Returns: {"r": Rマルチプル, "exit_reason": 決済種類, "n_levels_hit": 0-3}
    決済種類: SL_INITIAL_NO_TP / WEEKEND_NO_TP / TP_THEN_SL_TRAIL / TP_THEN_WEEKEND
             / TP_FULL(TP1-3すべて到達し残玉ゼロ) / MAX_HOLD
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
    for i in range(start, end):
        ts = h1.index[i]
        o, h, low, c = float(h1["open"].iloc[i]), float(h1["high"].iloc[i]), float(h1["low"].iloc[i]), float(h1["close"].iloc[i])
        n_levels_hit = sum(1 for lv in levels if lv[3])
        if is_weekend_close_time(ts):
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


def summarize(results: list[dict], label: str) -> dict:
    rs = [r["r"] for r in results]
    n = len(rs)
    if n == 0:
        return {"n": 0, "mean_r": None, "win_rate": None, "judgeable": False}
    mean_r = float(np.mean(rs))
    n_wins = sum(1 for r in rs if r > 0)
    n_losses = sum(1 for r in rs if r < 0)
    n_flat = n - n_wins - n_losses
    win_rate = float(np.mean([r > 0 for r in rs]))
    n_eff = max(4, int(round(n * (EFFECTIVE_PAIR_COUNT / len(PAIRS)))))
    judgeable = n_eff >= MIN_N_FOR_JUDGEMENT
    exit_reason_counts: dict[str, int] = {}
    for r in results:
        exit_reason_counts[r["exit_reason"]] = exit_reason_counts.get(r["exit_reason"], 0) + 1
    result = {
        "n": n, "mean_r": round(mean_r, 4), "win_rate": round(win_rate, 3),
        "n_wins": n_wins, "n_losses": n_losses, "n_flat": n_flat,
        "n_effective": n_eff, "judgeable": judgeable,
        "exit_reason_counts": dict(sorted(exit_reason_counts.items(), key=lambda kv: -kv[1])),
    }
    if judgeable:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=n_eff, replace=False) if n_eff < n else np.arange(n)
        sub = [rs[i] for i in idx]
        pr = permutation_test(sub, seed=42)
        result["perm_p"] = round(pr.p_value, 4)
    return result


def print_breakdown(summary: dict, label: str) -> None:
    print(f"--- {label} ---")
    print(f"  取引数 n={summary['n']}  平均R={summary['mean_r']}  勝率={summary['win_rate']}")
    print(f"  勝ち={summary['n_wins']}件  負け={summary['n_losses']}件  引き分け(R=0)={summary['n_flat']}件")
    print(f"  n_eff={summary.get('n_effective')}  perm_p={summary.get('perm_p')}"
          f"{'' if summary['judgeable'] else '  [n不足・判定不能]'}")
    print("  決済種類の内訳:")
    for reason, cnt in summary["exit_reason_counts"].items():
        pct = 100.0 * cnt / summary["n"]
        print(f"    {reason:<22} {cnt:>4}件 ({pct:5.1f}%)")
    print()


def main() -> int:
    print("=== SYS-FX009派生診断: H1 + 段階利確(40/35/25%)の効果測定 (Train期間・継続文脈のみ、反転文脈は除外) ===\n")
    with (ROOT / "research" / "EXP-FX000003" / "10-result" / "double_pattern_params_h1.json").open(encoding="utf-8") as f:
        params = json.load(f)
    trail_mult = params["atr_trail_multiplier"]
    print(f"H1パラメータ(既に導出済みのものを再利用): zigzag_threshold_atr={params['zigzag_threshold_atr']}, "
          f"pattern_tolerance_atr={params['pattern_tolerance_atr']}, stop_buffer_atr={params['stop_buffer_atr']}, "
          f"break_search_cap_bars={params['break_search_cap_bars']}, atr_trail_multiplier={trail_mult}")
    print(f"事前登録した段階利確レベル: {TP_LEVELS} (Rマルチプル, 按分比率)\n")

    old_results: list[dict] = []
    scaled_results: list[dict] = []
    n_entries_by_pair: dict[str, int] = {}
    n_reversal_by_pair: dict[str, int] = {}

    for pair in PAIRS:
        m5 = load_m5(pair)
        h1 = to_h1(m5)
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        entries = find_continuation_entries(pair, params)
        n_entries_by_pair[pair] = len(entries)
        n_reversal_by_pair[pair] = count_reversal_signals(pair, params)
        for e in entries:
            old_results.append(simulate_old_scheme(h1, atr_h1, e, trail_mult))
            scaled_results.append(simulate_scaled_scheme(h1, atr_h1, e, trail_mult))
        print(f"[{pair}] 継続文脈エントリー(シミュレーション対象)={len(entries)}件  "
              f"反転文脈ブレイク(除外・参考値)={n_reversal_by_pair[pair]}件")

    print(f"\n全体エントリー数(継続文脈のみ): {len(old_results)}件")
    print(f"参考: 反転文脈のため除外されたブレイク総数: {sum(n_reversal_by_pair.values())}件\n")

    old_summary = summarize(old_results, "旧方式")
    scaled_summary = summarize(scaled_results, "新方式")

    print_breakdown(old_summary, "旧方式: 1R到達で全量利確、以降BE+ATRトレーリング")
    print_breakdown(scaled_summary, "新方式: 1R=40%/2R=35%/3R=25%の段階利確、1R到達後はBE+ATRトレーリング")

    if old_summary["n"] == scaled_summary["n"] and old_summary["n"] > 0:
        old_rs = [r["r"] for r in old_results]
        scaled_rs = [r["r"] for r in scaled_results]
        diffs = [s - o for s, o in zip(scaled_rs, old_rs)]
        mean_diff = float(np.mean(diffs))
        win_rate_diff = float(np.mean([d > 0 for d in diffs]))
        print(f"--- ペア差分 (新方式 - 旧方式、同一エントリーの対応比較) ---")
        print(f"  平均差分R={round(mean_diff,4)}  新方式が上回った割合={round(win_rate_diff,3)}")

    out_path = ROOT / "research" / "method-notes" / "scaled_exit_diagnostic.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "timeframe": "H1",
            "context": "continuation_only (本番のevaluate_double_pattern_signalと同じLT一致ゲート、反転文脈は除外)",
            "tp_levels": TP_LEVELS,
            "h1_params_used": params,
            "n_entries_by_pair": n_entries_by_pair,
            "n_reversal_excluded_by_pair": n_reversal_by_pair,
            "old_scheme": old_summary,
            "scaled_scheme": scaled_summary,
            "old_scheme_results": [{"r": round(r["r"], 4), "exit_reason": r["exit_reason"]} for r in old_results],
            "scaled_scheme_results": [
                {"r": round(r["r"], 4), "exit_reason": r["exit_reason"], "n_levels_hit": r["n_levels_hit"]}
                for r in scaled_results
            ],
            "_note": (
                "スプレッド/スリッページ/スワップ等のコストは含めない簡易バーパス"
                "シミュレーション(H1 OHLCのみ)。方向性の比較診断であり、正式な"
                "バックテストKPI評価ではない。継続文脈のみ(本番と同じLT一致ゲート)、"
                "反転文脈のブレイクはシミュレーション対象から除外している。"
            ),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
