"""シグナルファースト研究基盤 提案6: M15/H1時間軸でのIC分析.

背景: H4/D1のIC分析(analyze_signal_ic.py・analyze_conditional_ic.py)・
反転再解釈(analyze_pattern_reversal_ic.py)のいずれでも、統計的に擁護
できる方向性エッジは見つからなかった。あわせて market_character.json の
分散比(VR)測定で、H4/D1はVR≈1のほぼランダムウォークと判明している。

司令塔選択「1. より短い時間軸(M15/H1)で週内完結を前提に再検証」を受け、
同じIC測定の枠組み(analyze_signal_ic.pyのcompute_ic/_spearman/
_fisher_p_valueをそのまま再利用)を、M15・H1の2時間軸に適用する。

ホライズンは市場性質分析(analyze_market_character.py)で使ったVR_Q_M15・
VR_Q_H1とバー数を揃え、いずれも週末クローズ(最大保有5営業日)に収まる
範囲に事前登録して固定する:
    M15: 1h(4本) / 4h(16本) / 1d(96本) / 3d(288本)
    H1 : 4h(4本) / 1d(24本) / 3d(72本)

特徴量の窓幅もタイムフレームに応じて再スケールする(H4版の「短期/長期
移動平均」という設計思想は維持しつつ、H4での10/20/50本(40h/80h/200h)
に相当する時間感覚をM15/H1のバー数に置き換える)。

Usage:
    python scripts/analyze_signal_ic_intraday.py

出力: research/method-notes/signal_ic_intraday.json
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

import analyze_signal_ic as base  # noqa: E402 (compute_ic/_spearman/_fisher_p_value を再利用)
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402
from minmax_fx_dt.strategy.indicators import bbands, donchian, rsi  # noqa: E402
from minmax_fx_dt.strategy.indicators import sma as sma_ind  # noqa: E402

PAIRS = base.PAIRS
TRAIN_START, TRAIN_END = base.TRAIN_START, base.TRAIN_END
EFFECTIVE_PAIR_COUNT = base.EFFECTIVE_PAIR_COUNT
RANDOM_SEED = base.RANDOM_SEED

# 事前登録: タイムフレームごとの設定 (結果を見る前に固定)
TIMEFRAME_CONFIGS = {
    "m15": {
        "resample_rule": "15min",
        "horizons_bars": {"1h": 4, "4h": 16, "1d": 96, "3d": 288},
        "mom_windows": {"1h": 4, "4h": 16, "1d": 96},
        "sma_short": 20,   # 5h
        "sma_long": 80,    # 20h (約1日弱)
        "sma_extra": 288,  # 3日
        "rsi_length": 14,
        "bb_length": 20,
        "donchian_length": 20,
    },
    "h1": {
        "resample_rule": "1h",
        "horizons_bars": {"4h": 4, "1d": 24, "3d": 72},
        "mom_windows": {"4h": 4, "1d": 24, "3d": 72},
        "sma_short": 10,   # 10h
        "sma_long": 24,    # 1日
        "sma_extra": 72,   # 3日
        "rsi_length": 14,
        "bb_length": 20,
        "donchian_length": 20,
    },
}


def load_ohlc(pair: str, ds1: dict, resample_rule: str) -> pd.DataFrame:
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]
    return pd.DataFrame({
        "open": df["open"].resample(resample_rule).first(),
        "high": df["high"].resample(resample_rule).max(),
        "low": df["low"].resample(resample_rule).min(),
        "close": df["close"].resample(resample_rule).last(),
    }).dropna()


def build_features_intraday(ohlc: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """H4版(analyze_signal_ic.build_features)と同じ設計思想を、タイムフレームに応じた
    バー数へ再スケールして適用する。"""
    close, high, low = ohlc["close"], ohlc["high"], ohlc["low"]
    log_close = np.log(close)
    atr_n = atr_ind(high, low, close, length=cfg["rsi_length"]).replace(0, np.nan)

    feats: dict[str, pd.Series] = {}
    for label, n in cfg["mom_windows"].items():
        feats[f"mom_{label}"] = log_close.diff(n)

    sma_s = sma_ind(close, cfg["sma_short"])
    sma_l = sma_ind(close, cfg["sma_long"])
    sma_x = sma_ind(close, cfg["sma_extra"])
    feats["sma_cross_short_long_atr"] = (sma_s - sma_l) / atr_n
    feats["px_vs_sma_long_atr"] = (close - sma_l) / atr_n
    feats["px_vs_sma_extra_atr"] = (close - sma_x) / atr_n

    feats["rsi_centered"] = rsi(close, cfg["rsi_length"]) - 50.0
    bb = bbands(close, length=cfg["bb_length"], std=2.0)
    feats["bb_pctb_centered"] = bb[f"BBP_{cfg['bb_length']}_2.0"] - 0.5
    roll_mean = close.rolling(cfg["bb_length"], min_periods=cfg["bb_length"]).mean()
    roll_std = close.rolling(cfg["bb_length"], min_periods=cfg["bb_length"]).std(ddof=0).replace(0, np.nan)
    feats["zscore"] = (close - roll_mean) / roll_std

    dc = donchian(high, low, lower_length=cfg["donchian_length"], upper_length=cfg["donchian_length"])
    dc_width = (dc["DCU"] - dc["DCL"]).replace(0, np.nan)
    feats["donchian_pos_centered"] = (close - dc["DCL"]) / dc_width - 0.5

    feats["_random_control"] = pd.Series(rng.standard_normal(len(close)), index=close.index)
    return pd.DataFrame(feats)


def analyze_timeframe(tf_name: str, cfg: dict, ds1: dict, rng: np.random.Generator) -> dict:
    print(f"=== {tf_name.upper()} ===")
    feature_names: list[str] = []
    pooled_rows: dict[str, list[pd.DataFrame]] = {h: [] for h in cfg["horizons_bars"]}

    for pair in PAIRS:
        ohlc = load_ohlc(pair, ds1, cfg["resample_rule"])
        feats = build_features_intraday(ohlc, cfg, rng)
        feature_names = list(feats.columns)
        log_close = np.log(ohlc["close"])
        for h_label, h_bars in cfg["horizons_bars"].items():
            fwd = log_close.shift(-h_bars) - log_close
            merged = feats.copy()
            merged["_fwd"] = fwd
            pooled_rows[h_label].append(merged.dropna())

    eff_ratio = EFFECTIVE_PAIR_COUNT / len(PAIRS)
    n_tests = len(feature_names) * len(cfg["horizons_bars"])
    bonferroni_alpha = 0.05 / n_tests

    pooled: dict[str, dict] = {}
    for h_label, h_bars in cfg["horizons_bars"].items():
        allrows = pd.concat(pooled_rows[h_label], axis=0)
        pooled[h_label] = {}
        for name in feature_names:
            r = base.compute_ic(allrows[name], allrows["_fwd"], h_bars)
            if r["ic_independent"] is not None and r["n_independent"] > 0:
                n_eff = max(4, int(round(r["n_independent"] * eff_ratio)))
                p_eff = base._fisher_p_value(r["ic_independent"], n_eff)
                r["n_effective"] = n_eff
                r["p_value_corr_adjusted"] = round(p_eff, 4) if np.isfinite(p_eff) else None
                r["survives_bonferroni"] = bool(
                    r["p_value_corr_adjusted"] is not None and r["p_value_corr_adjusted"] < bonferroni_alpha
                )
            else:
                r["n_effective"] = 0
                r["p_value_corr_adjusted"] = None
                r["survives_bonferroni"] = False
            pooled[h_label][name] = r

    print(f"多重検定: {n_tests}件 → Bonferroni閾値 α={bonferroni_alpha:.5f}")
    for h_label in cfg["horizons_bars"]:
        print(f"--- 前方リターン {h_label} ---")
        rows = sorted(pooled[h_label].items(), key=lambda kv: abs(kv[1]["ic"] or 0.0), reverse=True)
        for name, r in rows:
            ic = f"{r['ic']:.4f}" if r["ic"] is not None else "n/a"
            pv2 = f"{r['p_value_corr_adjusted']:.4f}" if r["p_value_corr_adjusted"] is not None else "n/a"
            mark = " **" if r["survives_bonferroni"] else (
                " *" if (r["p_value_corr_adjusted"] is not None and r["p_value_corr_adjusted"] < 0.05) else "")
            print(f"  {name:<28}{ic:>9}  p(補正)={pv2:>8}  n_eff={r['n_effective']:>6}{mark}")
    print()

    ctrl_hits = [h for h in cfg["horizons_bars"] if pooled[h]["_random_control"]["survives_bonferroni"]]
    survivors = [(h, n) for h in cfg["horizons_bars"] for n in feature_names
                 if not n.startswith("_") and pooled[h][n]["survives_bonferroni"]]
    print(f"[サニティ] ランダム対照が有意判定された回数: {len(ctrl_hits)}")
    print(f"[結論] {tf_name.upper()}: 相関補正+多重検定補正を突破した特徴量: {len(survivors)}件 {survivors}\n")

    return {
        "n_tests": n_tests, "bonferroni_alpha": round(bonferroni_alpha, 6),
        "pooled": pooled, "survivors": [{"horizon": h, "feature": n} for h, n in survivors],
        "random_control_false_positives": len(ctrl_hits),
    }


def main() -> int:
    print("=== 提案6: M15/H1時間軸でのIC分析 (Train期間のみ) ===\n")
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)

    rng = np.random.default_rng(RANDOM_SEED)
    results = {tf: analyze_timeframe(tf, cfg, ds1, rng) for tf, cfg in TIMEFRAME_CONFIGS.items()}

    all_survivors = [(tf, s["horizon"], s["feature"]) for tf, r in results.items() for s in r["survivors"]]
    print("=== 全体結論 ===")
    print(f"M15・H1あわせて有意な特徴量: {len(all_survivors)}件")
    for tf, h, n in all_survivors:
        print(f"  {tf}/{h}/{n}")
    if not all_survivors:
        print("  なし。")

    out_path = ROOT / "research" / "method-notes" / "signal_ic_intraday.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "effective_pair_count": EFFECTIVE_PAIR_COUNT,
            "random_seed": RANDOM_SEED,
            "timeframe_configs": TIMEFRAME_CONFIGS,
            "_note": (
                "H4/D1(signal_ic_baseline.json)では有意な特徴量が0件だった。"
                "本ファイルは週内完結を前提としたM15/H1タイムフレームでの再測定結果。"
            ),
            "results": results,
            "all_survivors": [{"timeframe": tf, "horizon": h, "feature": n} for tf, h, n in all_survivors],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
