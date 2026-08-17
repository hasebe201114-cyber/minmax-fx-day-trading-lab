"""シグナルファースト研究基盤 提案2: 条件付きIC分析.

背景: `analyze_signal_ic.py` のベースライン測定で、無条件では相関補正+
多重検定補正を突破する特徴量が0件だった。これは market_character.json の
VR≈1 (無条件・線形の予測可能性なし) と整合する結果だが、**条件付き**の
構造までは否定していない。

本スクリプトは以下3軸でH4バーを層別化し、各層内でIC (analyze_signal_ic.py
と同一のロジック) を再測定する。軸の定義・分位の区切り方は結果を見る前に
以下の通り固定する (HARKing防止):

1. セッション: DS-4 (data/curated/ds-4.json) のセッション定義をそのまま使用
   TOKYO=[9,18) / LONDON=[17,2) / NEW_YORK=[22,7) JST (時間帯は重複しうる)
2. ボラティリティregime: ATR(14,H4) のTrain期間内での3分位 (低/中/高)
3. 曜日: 月〜金 (為替市場が開いている曜日のみ)

各軸×特徴量×ホライズンで、通貨間相関によるサンプル数割引(実効1.70通貨)
+ 層別化を考慮した多重検定補正 (n_tests = 層数 × 特徴量数 × ホライズン数)
を適用する。実効n<30の層は判定不能として明示する。

Usage:
    python scripts/analyze_conditional_ic.py

出力: research/method-notes/conditional_ic.json
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

import analyze_signal_ic as base  # noqa: E402  (共通ロジックの再利用)
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

PAIRS = base.PAIRS
TRAIN_START, TRAIN_END = base.TRAIN_START, base.TRAIN_END
HORIZONS_H4 = base.HORIZONS_H4
EFFECTIVE_PAIR_COUNT = base.EFFECTIVE_PAIR_COUNT
RANDOM_SEED = base.RANDOM_SEED

SESSIONS_JST = {"TOKYO": (9, 18), "LONDON": (17, 2), "NEW_YORK": (22, 7)}
VOL_REGIME_LABELS = ["low", "mid", "high"]  # ATR(14,H4) Train内3分位
MIN_EFFECTIVE_N = 30  # これ未満の層は判定不能として明示


def _in_session(hour: int, start: int, end: int) -> bool:
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end  # 日をまたぐセッション (LONDON, NEW_YORK)


def label_session(index: pd.DatetimeIndex) -> pd.Series:
    hours = index.hour
    labels = pd.Series("NONE", index=index, dtype="object")
    for name, (start, end) in SESSIONS_JST.items():
        mask = np.array([_in_session(h, start, end) for h in hours])
        # 複数セッションに該当する場合は先勝ち (TOKYO > LONDON > NEW_YORK の優先順)
        labels = labels.mask((labels == "NONE") & mask, name)
    return labels


def label_vol_regime(atr_series: pd.Series) -> pd.Series:
    valid = atr_series.dropna()
    q1, q2 = valid.quantile([1 / 3, 2 / 3])
    labels = pd.Series("NONE", index=atr_series.index, dtype="object")
    labels = labels.mask(atr_series <= q1, "low")
    labels = labels.mask((atr_series > q1) & (atr_series <= q2), "mid")
    labels = labels.mask(atr_series > q2, "high")
    return labels


def build_conditions(h4: pd.DataFrame) -> pd.DataFrame:
    atr14 = atr_ind(h4["high"], h4["low"], h4["close"], length=14)
    return pd.DataFrame({
        "session": label_session(h4.index),
        "vol_regime": label_vol_regime(atr14),
        "weekday": h4.index.day_name(),
    })


def evaluate_axis(
    axis_name: str,
    axis_labels: pd.Series,
    all_feats: pd.DataFrame,
    all_fwd: dict[str, pd.Series],
    feature_names: list[str],
) -> dict:
    result: dict = {}
    eff_ratio = EFFECTIVE_PAIR_COUNT / len(PAIRS)
    levels = sorted(lv for lv in axis_labels.unique() if lv != "NONE")
    n_tests = len(levels) * len(feature_names) * len(HORIZONS_H4)
    bonferroni_alpha = 0.05 / n_tests if n_tests > 0 else float("nan")

    for level in levels:
        result[level] = {}
        mask = axis_labels == level
        for h_label, h_bars in HORIZONS_H4.items():
            fwd = all_fwd[h_label]
            sub_feats = all_feats.loc[mask.reindex(all_feats.index, fill_value=False)]
            sub_fwd = fwd.loc[sub_feats.index]
            level_result = {}
            for name in feature_names:
                r = base.compute_ic(sub_feats[name], sub_fwd, h_bars)
                if r["ic_independent"] is not None and r["n_independent"] > 0:
                    n_eff = max(4, int(round(r["n_independent"] * eff_ratio)))
                    if n_eff < MIN_EFFECTIVE_N:
                        r["n_effective"] = n_eff
                        r["p_value_corr_adjusted"] = None
                        r["survives_bonferroni"] = False
                        r["judgeable"] = False
                    else:
                        p_eff = base._fisher_p_value(r["ic_independent"], n_eff)
                        r["n_effective"] = n_eff
                        r["p_value_corr_adjusted"] = round(p_eff, 4) if np.isfinite(p_eff) else None
                        r["survives_bonferroni"] = bool(
                            r["p_value_corr_adjusted"] is not None
                            and r["p_value_corr_adjusted"] < bonferroni_alpha
                        )
                        r["judgeable"] = True
                else:
                    r["n_effective"] = 0
                    r["p_value_corr_adjusted"] = None
                    r["survives_bonferroni"] = False
                    r["judgeable"] = False
                level_result[name] = r
            result[level][h_label] = level_result

    return {
        "levels": levels,
        "n_tests": n_tests,
        "bonferroni_alpha": round(bonferroni_alpha, 6) if np.isfinite(bonferroni_alpha) else None,
        "by_level": result,
    }


def main() -> int:
    print("=== 条件付きIC分析 (Train期間のみ、H4足) ===\n")
    print(f"層別化軸: セッション({list(SESSIONS_JST)}) / ボラregime({VOL_REGIME_LABELS}) / 曜日")
    print(f"最小実効サンプル数: {MIN_EFFECTIVE_N} 未満の層は判定不能とする\n")

    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)

    rng = np.random.default_rng(RANDOM_SEED)
    feature_names: list[str] = []

    axis_data: dict[str, list[pd.DataFrame]] = {"session": [], "vol_regime": [], "weekday": []}
    fwd_data: dict[str, list[pd.Series]] = {h: [] for h in HORIZONS_H4}
    feat_data: list[pd.DataFrame] = []

    for pair in PAIRS:
        h4 = base.load_h4(pair, ds1)
        feats = base.build_features(h4, rng)
        feature_names = list(feats.columns)
        conds = build_conditions(h4)
        log_close = np.log(h4["close"])

        # 5通貨を縦積みすると元のdatetimeインデックスが重複するため、
        # 各ペアごとに連番へリセットしてから積む (位置ベースの整合性を保証)
        feats = feats.reset_index(drop=True)
        conds = conds.reset_index(drop=True)
        feat_data.append(feats)
        for axis in axis_data:
            axis_data[axis].append(conds[axis])
        for h_label, h_bars in HORIZONS_H4.items():
            fwd = (log_close.shift(-h_bars) - log_close).reset_index(drop=True)
            fwd_data[h_label].append(fwd)

    all_feats = pd.concat(feat_data, axis=0, ignore_index=True)
    all_fwd = {h: pd.concat(fwd_data[h], axis=0, ignore_index=True) for h in HORIZONS_H4}

    axis_results: dict[str, dict] = {}
    for axis in axis_data:
        axis_labels = pd.concat(axis_data[axis], axis=0, ignore_index=True)
        print(f"--- 軸: {axis} ---")
        r = evaluate_axis(axis, axis_labels, all_feats, all_fwd, feature_names)
        axis_results[axis] = r
        print(f"  水準: {r['levels']}  検定数: {r['n_tests']}  Bonferroni閾値: {r['bonferroni_alpha']}")
        for level in r["levels"]:
            for h_label in HORIZONS_H4:
                level_res = r["by_level"][level][h_label]
                for name, res in level_res.items():
                    if name.startswith("_"):
                        continue
                    if res["survives_bonferroni"]:
                        print(f"  [有意] {level}/{h_label}/{name}: IC(独立)={res['ic_independent']} "
                              f"p(補正)={res['p_value_corr_adjusted']} n_eff={res['n_effective']}")
        print()

    # サニティ: ランダム対照特徴量がどの層でも有意判定されないことを確認
    ctrl_survivors = [
        (axis, level, h)
        for axis, r in axis_results.items()
        for level in r["levels"]
        for h in HORIZONS_H4
        if r["by_level"][level][h]["_random_control"]["survives_bonferroni"]
    ]
    print(f"[サニティ確認] ランダム対照が有意判定された層: {len(ctrl_survivors)}件 {ctrl_survivors}")
    print("  → 0件でなければ補正ロジックの不備を疑う\n")

    all_survivors = [
        (axis, level, h, name)
        for axis, r in axis_results.items()
        for level in r["levels"]
        for h in HORIZONS_H4
        for name, res in r["by_level"][level][h].items()
        if (not name.startswith("_")) and res["survives_bonferroni"]
    ]
    print("=== 結論 ===")
    print(f"層別化して初めて有意になった特徴量: {len(all_survivors)}件")
    for axis, level, h, name in all_survivors:
        res = axis_results[axis]["by_level"][level][h][name]
        print(f"  {axis}={level} / {h} / {name}: IC(独立)={res['ic_independent']} "
              f"p(補正)={res['p_value_corr_adjusted']} n_eff={res['n_effective']}")
    if not all_survivors:
        print("  なし。条件付き層別化でも統計的に擁護できる方向性エッジは見つからなかった。")

    out_path = ROOT / "research" / "method-notes" / "conditional_ic.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "timeframe": "H4",
            "horizons_h4_bars": HORIZONS_H4,
            "sessions_jst": {k: list(v) for k, v in SESSIONS_JST.items()},
            "vol_regime_labels": VOL_REGIME_LABELS,
            "effective_pair_count": EFFECTIVE_PAIR_COUNT,
            "min_effective_n": MIN_EFFECTIVE_N,
            "random_seed": RANDOM_SEED,
            "_note": (
                "無条件IC (signal_ic_baseline.json) では有意な特徴量が0件だった。"
                "本ファイルはセッション/ボラregime/曜日で層別化した場合の再測定結果。"
                "judgeable=false の層は実効n<30のため判定不能 (有意でも非有意でもない)。"
            ),
            "axes": axis_results,
            "survivors": [
                {"axis": axis, "level": level, "horizon": h, "feature": name}
                for axis, level, h, name in all_survivors
            ],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
