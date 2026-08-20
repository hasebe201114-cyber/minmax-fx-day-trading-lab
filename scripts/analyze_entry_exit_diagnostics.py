"""EXP-FX000005 エントリー/イグジット診断分析(司令塔4点依頼への対応).

司令塔依頼:
1. 初動のH1ボラブレーク後、戻り後に伸びるケースと反転するケースを見極めて
   からエントリーできないか(方向の事前判別)
2. SL到達の内訳: ブレーク検知後初回エントリーか、トレンド発生後のエントリーか
3. TP_FULLの率が高いので、TP値をもう少し高められないか
4. 高ボラの前提のため、建値へのトレーリングを早められないか

対象は改善ループ第3試行(4通貨版+BOJ/FOMCブラックアウト窓、採用中の最良候補)の
Train/Validation/Test全期間・全トレード。`simulate_dow_theory_trend()`に
2026-08-20追加した診断用フィールド(entry_seq/resumed_since_last_entry/
bars_since_tracking_start/break_price/break_time/entry_h1_idx)と
breakeven_trigger_rパラメータを利用する。既存の採用candidateの数値は
一切変更しない(すべて追加的な診断)。

出力: research/method-notes/entry_exit_diagnostics.json
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

from backtest_vol_breakout_dow_theory import (  # noqa: E402
    ATR_TRAIL_MULTIPLIER, MAX_HOLD_BARS, simulate_dow_theory_trend,
)
from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402
from derive_vol_breakout_entry_params import N_BREAKOUT, to_h1  # noqa: E402
from economic_calendar import is_blackout  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_train.json").open(encoding="utf-8") as f:
    TRAIN_RESULT = json.load(f)
STOP_BUFFER_ATR_M5 = TRAIN_RESULT["params"]["stop_buffer_atr_m5"]

PERIODS = {
    "train":      ("2023-11-01", "2025-03-31"),
    "validation": ("2025-04-01", "2025-11-30"),
    "test":       ("2025-12-01", "2026-08-15"),
}

RNG_SEED = 42
N_PERM = 5000


def load_m5_period(pair: str, start: str, end: str) -> pd.DataFrame:
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= start) & (df.index <= end)]


def mfe_r(h1: pd.DataFrame, entry_h1_idx: int, entry_price: float, initial_risk: float,
          direction: str, end_idx_exclusive: int) -> float | None:
    """entry_h1_idx+1からend_idx_exclusive-1までの生の価格経路から、Rマルチプル換算の
    最大順行幅(MFE)を計算する(実際のストップ/トレーリングとは無関係の生値)。"""
    start = entry_h1_idx + 1
    end = min(len(h1), end_idx_exclusive)
    if start >= end:
        return None
    seg = h1.iloc[start:end]
    if direction == "UP":
        best = float(seg["high"].max())
        return (best - entry_price) / initial_risk
    else:
        best = float(seg["low"].min())
        return (entry_price - best) / initial_risk


def perm_test_mean_diff(a: np.ndarray, b: np.ndarray, seed: int = RNG_SEED, n_perm: int = N_PERM) -> tuple[float, float]:
    """2群の平均差の単純シャッフル検定(scipy不使用)。observed diff(a-b)とp値を返す。"""
    rng = np.random.default_rng(seed)
    observed = float(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = float(np.mean(pooled[:n_a]) - np.mean(pooled[n_a:]))
        if abs(diff) >= abs(observed):
            count += 1
    return observed, count / n_perm


def collect_trades() -> pd.DataFrame:
    rows = []
    for period_name, (start, end) in PERIODS.items():
        for pair in SELECTED_PAIRS:
            m5 = load_m5_period(pair, start, end)
            if len(m5) < 1000:
                continue
            h1 = to_h1(m5)
            atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
            atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)
            ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
            idxs = np.where(ratio.values >= N_BREAKOUT)[0]

            for i in idxs:
                pos = h1.index.get_loc(ratio.index[i])
                bar = h1.iloc[pos]
                direction = "UP" if bar["close"] > bar["open"] else "DOWN"
                trades = simulate_dow_theory_trend(
                    m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER,
                    blackout_check=is_blackout,
                )
                for t in trades:
                    entry_h1_idx = t["entry_h1_idx"]
                    entry_price = t["entry_price"]
                    initial_risk = t["initial_risk"]
                    exit_time_idx = h1.index.searchsorted(pd.Timestamp(t["exit_time"]), side="right")
                    atr_at_entry = float(atr_h1.iloc[entry_h1_idx]) if pd.notna(atr_h1.iloc[entry_h1_idx]) else None
                    atr_at_break = float(atr_h1.iloc[pos]) if pd.notna(atr_h1.iloc[pos]) else None
                    atr_h1_ratio = (atr_at_entry / atr_at_break) if (atr_at_entry and atr_at_break) else None
                    if direction == "UP":
                        price_move_from_breakout_r = (entry_price - t["break_price"]) / initial_risk
                    else:
                        price_move_from_breakout_r = (t["break_price"] - entry_price) / initial_risk

                    rows.append({
                        "period": period_name, "pair": pair, "direction": direction,
                        "entry_time": t["entry_time"], "exit_time": str(t["exit_time"]),
                        "exit_reason": t["exit_reason"], "n_levels_hit": t["n_levels_hit"],
                        "r": t["r"], "entry_seq": t["entry_seq"],
                        "resumed_since_last_entry": t["resumed_since_last_entry"],
                        "bars_since_tracking_start": t["bars_since_tracking_start"],
                        "atr_h1_ratio": atr_h1_ratio,
                        "price_move_from_breakout_r": price_move_from_breakout_r,
                        "mfe_r_within_trade": mfe_r(h1, entry_h1_idx, entry_price, initial_risk, direction, exit_time_idx),
                        "mfe_r_extended": mfe_r(h1, entry_h1_idx, entry_price, initial_risk, direction,
                                                 entry_h1_idx + 1 + MAX_HOLD_BARS),
                    })
    return pd.DataFrame(rows)


def point1_direction_discrimination(df: pd.DataFrame) -> dict:
    continued = df[df["exit_reason"].isin(["TP_FULL", "TP_THEN_SL_TRAIL"])]
    reversed_ = df[df["exit_reason"] == "SL_INITIAL_NO_TP"]
    features = ["entry_seq", "bars_since_tracking_start", "atr_h1_ratio", "price_move_from_breakout_r"]
    out = {"n_continued": len(continued), "n_reversed": len(reversed_), "features": {}}
    for feat in features:
        a = continued[feat].dropna().to_numpy()
        b = reversed_[feat].dropna().to_numpy()
        if len(a) < 5 or len(b) < 5:
            continue
        diff, p = perm_test_mean_diff(a, b)
        out["features"][feat] = {
            "mean_continued": round(float(np.mean(a)), 4), "mean_reversed": round(float(np.mean(b)), 4),
            "median_continued": round(float(np.median(a)), 4), "median_reversed": round(float(np.median(b)), 4),
            "diff": round(diff, 4), "perm_p": round(p, 4),
        }
    resumed_rate_continued = float(continued["resumed_since_last_entry"].mean())
    resumed_rate_reversed = float(reversed_["resumed_since_last_entry"].mean())
    out["resumed_since_last_entry_rate"] = {
        "continued": round(resumed_rate_continued, 4), "reversed": round(resumed_rate_reversed, 4),
    }
    return out


def point2_sl_breakdown_by_entry_seq(df: pd.DataFrame) -> dict:
    def bucket(seq):
        return "1(初回)" if seq == 1 else ("2" if seq == 2 else ("3" if seq == 3 else "4+"))
    df = df.copy()
    df["seq_bucket"] = df["entry_seq"].apply(bucket)
    out = {}
    for b, g in df.groupby("seq_bucket"):
        n = len(g)
        n_sl = int((g["exit_reason"] == "SL_INITIAL_NO_TP").sum())
        out[b] = {"n_trades": n, "n_sl_initial": n_sl, "sl_rate": round(n_sl / n, 4) if n else None,
                   "mean_r": round(float(g["r"].mean()), 4)}
    # resumed後の初回エントリーのSL率
    resumed = df[df["resumed_since_last_entry"]]
    non_resumed = df[~df["resumed_since_last_entry"]]
    out["_resumed_vs_not"] = {
        "resumed": {"n": len(resumed), "sl_rate": round(float((resumed["exit_reason"] == "SL_INITIAL_NO_TP").mean()), 4) if len(resumed) else None},
        "not_resumed": {"n": len(non_resumed), "sl_rate": round(float((non_resumed["exit_reason"] == "SL_INITIAL_NO_TP").mean()), 4) if len(non_resumed) else None},
    }
    return out


def point3_tp_full_extension_room(df: pd.DataFrame) -> dict:
    tp_full = df[df["exit_reason"] == "TP_FULL"]
    mfe = tp_full["mfe_r_extended"].dropna()
    if len(mfe) == 0:
        return {"n": 0}
    pct = {f"p{p}": round(float(np.percentile(mfe, p)), 3) for p in [10, 25, 50, 75, 90]}
    thresholds = [3.0, 4.0, 5.0, 6.0, 8.0]
    reach_rate = {f">={t}R": round(float((mfe >= t).mean()), 4) for t in thresholds}
    return {"n": len(mfe), "percentiles": pct, "reach_rate": reach_rate}


def point4_early_breakeven_analysis(df: pd.DataFrame) -> dict:
    sl_trades = df[df["exit_reason"] == "SL_INITIAL_NO_TP"]
    mfe = sl_trades["mfe_r_within_trade"].dropna()
    result: dict = {"n_sl_trades": len(sl_trades)}
    if len(mfe) > 0:
        pct = {f"p{p}": round(float(np.percentile(mfe, p)), 3) for p in [10, 25, 50, 75, 90]}
        thresholds = [0.2, 0.3, 0.5, 0.7, 0.9]
        reach_rate = {f">={t}R": round(float((mfe >= t).mean()), 4) for t in thresholds}
        result["mfe_before_reversal"] = {"percentiles": pct, "reach_rate": reach_rate}
    return result


def counterfactual_breakeven_sweep() -> dict:
    """建値移動トリガーR値を変えたTrain単独の感度確認(HARKing防止、Trainのみ選定)。"""
    from minmax_fx_dt.backtest.permutation import permutation_test_clustered
    from minmax_fx_dt.decision.criteria import compute_n_trades_effective

    start, end = PERIODS["train"]
    results = {}
    for be_r in [None, 0.3, 0.5, 0.7, 1.0]:
        all_r = []
        trades_per_currency: dict[str, int] = {}
        for pair in SELECTED_PAIRS:
            m5 = load_m5_period(pair, start, end)
            h1 = to_h1(m5)
            atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
            atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)
            ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
            idxs = np.where(ratio.values >= N_BREAKOUT)[0]
            pair_rs = []
            for i in idxs:
                pos = h1.index.get_loc(ratio.index[i])
                bar = h1.iloc[pos]
                direction = "UP" if bar["close"] > bar["open"] else "DOWN"
                trades = simulate_dow_theory_trend(
                    m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER,
                    blackout_check=is_blackout, breakeven_trigger_r=be_r,
                )
                pair_rs.extend(t["r"] for t in trades)
            all_r.extend(pair_rs)
            trades_per_currency[pair] = len(pair_rs)
        n = len(all_r)
        wins = [r for r in all_r if r > 0]
        losses = [r for r in all_r if r < 0]
        pf = (sum(wins) / abs(sum(losses))) if losses else None
        n_eff = compute_n_trades_effective(trades_per_currency, n)
        pairs_flat = sum([[p] * c for p, c in trades_per_currency.items()], [])
        perm = permutation_test_clustered(all_r, pairs_flat, n_permutations=20000, seed=42) if n >= 4 else None
        label = "current(TP1=1.0R相当)" if be_r is None else f"{be_r}R"
        results[label] = {
            "breakeven_trigger_r": be_r, "n_trades": n,
            "mean_r": round(float(np.mean(all_r)), 4) if n else None,
            "win_rate": round(float(np.mean([r > 0 for r in all_r])), 4) if n else None,
            "profit_factor": round(pf, 3) if pf else None,
            "n_trades_effective": round(n_eff, 1),
            "permutation_p_clustered": perm.p_value if perm else None,
        }
    return results


def counterfactual_tp_level_sweep() -> dict:
    """TP3水準を引き上げた場合のTrain単独感度確認(HARKing防止、Trainのみ選定)。
    配分比率(40/35/25%)とTP1/TP2(1R/2R)は不変、TP3のみ3R/4R/5R/6Rで比較する。"""
    from minmax_fx_dt.backtest.permutation import permutation_test_clustered
    from minmax_fx_dt.decision.criteria import compute_n_trades_effective

    start, end = PERIODS["train"]
    variants = {
        "current(TP3=3R)": [(1.0, 0.40), (2.0, 0.35), (3.0, 0.25)],
        "TP3=4R": [(1.0, 0.40), (2.0, 0.35), (4.0, 0.25)],
        "TP3=5R": [(1.0, 0.40), (2.0, 0.35), (5.0, 0.25)],
        "TP3=6R": [(1.0, 0.40), (2.0, 0.35), (6.0, 0.25)],
    }
    results = {}
    for label, tp_levels in variants.items():
        all_r = []
        trades_per_currency: dict[str, int] = {}
        for pair in SELECTED_PAIRS:
            m5 = load_m5_period(pair, start, end)
            h1 = to_h1(m5)
            atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
            atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)
            ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
            idxs = np.where(ratio.values >= N_BREAKOUT)[0]
            pair_rs = []
            for i in idxs:
                pos = h1.index.get_loc(ratio.index[i])
                bar = h1.iloc[pos]
                direction = "UP" if bar["close"] > bar["open"] else "DOWN"
                trades = simulate_dow_theory_trend(
                    m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER,
                    blackout_check=is_blackout, tp_levels=tp_levels,
                )
                pair_rs.extend(t["r"] for t in trades)
            all_r.extend(pair_rs)
            trades_per_currency[pair] = len(pair_rs)
        n = len(all_r)
        wins = [r for r in all_r if r > 0]
        losses = [r for r in all_r if r < 0]
        pf = (sum(wins) / abs(sum(losses))) if losses else None
        n_eff = compute_n_trades_effective(trades_per_currency, n)
        pairs_flat = sum([[p] * c for p, c in trades_per_currency.items()], [])
        perm = permutation_test_clustered(all_r, pairs_flat, n_permutations=20000, seed=42) if n >= 4 else None
        results[label] = {
            "tp_levels": tp_levels, "n_trades": n,
            "mean_r": round(float(np.mean(all_r)), 4) if n else None,
            "win_rate": round(float(np.mean([r > 0 for r in all_r])), 4) if n else None,
            "profit_factor": round(pf, 3) if pf else None,
            "n_trades_effective": round(n_eff, 1),
            "permutation_p_clustered": perm.p_value if perm else None,
        }
    return results


def main() -> int:
    print("=== EXP-FX000005 エントリー/イグジット診断分析 ===\n")
    print("トレードデータ収集中(Train/Validation/Test、4通貨、カレンダーフィルター適用)...")
    df = collect_trades()
    print(f"収集トレード数: {len(df)}件\n")

    print("--- 論点1: 継続 vs 反転の事前判別特徴量 ---")
    p1 = point1_direction_discrimination(df)
    for feat, v in p1["features"].items():
        print(f"  {feat}: 継続群平均={v['mean_continued']} 反転群平均={v['mean_reversed']} "
              f"diff={v['diff']} perm_p={v['perm_p']}")
    print(f"  resumed_since_last_entry率: 継続群={p1['resumed_since_last_entry_rate']['continued']} "
          f"反転群={p1['resumed_since_last_entry_rate']['reversed']}\n")

    print("--- 論点2: SL到達の内訳(エントリー順序別) ---")
    p2 = point2_sl_breakdown_by_entry_seq(df)
    for k, v in p2.items():
        if k.startswith("_"):
            continue
        print(f"  entry_seq={k}: n={v['n_trades']} SL率={v['sl_rate']} 平均R={v['mean_r']}")
    print(f"  再開後 vs 非再開: {p2['_resumed_vs_not']}\n")

    print("--- 論点3: TP_FULLトレードの3R超過分の余地 ---")
    p3 = point3_tp_full_extension_room(df)
    print(f"  n={p3.get('n')}  percentiles={p3.get('percentiles')}  reach_rate={p3.get('reach_rate')}\n")

    print("--- 論点4: SL_INITIAL_NO_TPトレードの反転前MFE ---")
    p4 = point4_early_breakeven_analysis(df)
    print(f"  n_sl_trades={p4['n_sl_trades']}  {p4.get('mfe_before_reversal')}\n")

    print("--- 論点4補足: 建値移動トリガーR値のTrain単独感度確認 ---")
    cf = counterfactual_breakeven_sweep()
    for label, v in cf.items():
        print(f"  {label}: n={v['n_trades']} mean_r={v['mean_r']} win_rate={v['win_rate']} "
              f"PF={v['profit_factor']} n_eff={v['n_trades_effective']} perm_p={v['permutation_p_clustered']}")

    print("\n--- 論点3補足: TP3水準引き上げのTrain単独感度確認 ---")
    cf_tp = counterfactual_tp_level_sweep()
    for label, v in cf_tp.items():
        print(f"  {label}: n={v['n_trades']} mean_r={v['mean_r']} win_rate={v['win_rate']} "
              f"PF={v['profit_factor']} n_eff={v['n_trades_effective']} perm_p={v['permutation_p_clustered']}")

    out_path = ROOT / "research" / "method-notes" / "entry_exit_diagnostics.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "source": "改善ループ第3試行(4通貨版+BOJ/FOMCブラックアウト窓)の全期間トレード",
            "n_trades_total": len(df),
            "point1_direction_discrimination": p1,
            "point2_sl_breakdown_by_entry_seq": p2,
            "point3_tp_full_extension_room": p3,
            "point4_early_breakeven_analysis": p4,
            "point4_counterfactual_breakeven_sweep_train_only": cf,
            "point3_counterfactual_tp_level_sweep_train_only": cf_tp,
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
