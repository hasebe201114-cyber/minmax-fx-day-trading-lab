"""シグナルファースト研究基盤: 特徴量→前方リターンの予測力(IC)を直接測る.

背景 (計画書 提案1):
    SYS-FX007/008/009 はいずれも「戦略を実装 → 20分バックテスト → KPIを見る」
    というループで検証してきた。この方式ではエントリー判定・イグジット・コスト・
    サイジングが全て混ざり、**予測力の有無を単独で切り分けられない**。実際
    SYS-FX009 では「エントリーの予測力」「週末制約」「ストップ設計」が分離できず、
    切り分けに専用の診断実験が別途必要になった。

    本スクリプトは戦略を一切介さず、各特徴量が将来リターンの方向をどれだけ
    説明するかを Spearman 順位相関 (Information Coefficient, IC) で直接測る。
    ベクトル化により数秒で完了する。

IC の読み方:
    IC > 0 : 特徴量が高いほど将来リターンも高い (モメンタム的)
    IC < 0 : 特徴量が高いほど将来リターンは低い (平均回帰的)
    IC ≈ 0 : 予測力なし
    実務的な目安として |IC| が 0.02〜0.05 あれば弱いながら有用とされる。

方法論上の注意 (いずれも本実装で対処済み):
    1. 先読み防止: 特徴量は t 時点の確定終値までの情報のみ、リターンは t→t+h
    2. 重複窓によるt値膨張: ホライズン h の観測は h-1 本ぶん重複するため、
       有意性検定は h 本ごとの非重複サブサンプルで行う
    3. 通貨間相関: 5通貨は実効1.7通貨相当 (market_character.json)。
       プールICの有意性は名目 n ではなく実効独立数で割り引いて解釈すること
    4. HARKing 防止: **Train期間のみ使用** (Validation/Test は温存)

Usage:
    python scripts/analyze_signal_ic.py

出力: research/method-notes/signal_ic_baseline.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import math

import numpy as np
import pandas as pd

from minmax_fx_dt.strategy.indicators import atr as atr_ind
from minmax_fx_dt.strategy.indicators import bbands, donchian, rsi
from minmax_fx_dt.strategy.indicators import sma as sma_ind

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"

# 前方リターンのホライズン (H4バー本数)。結果を見る前に固定。
HORIZONS_H4 = {"4h": 1, "1d": 6, "3d": 18, "1w": 42}

RANDOM_SEED = 42  # ランダム対照特徴量の再現性確保

# 5通貨プールの実効独立通貨数 (scripts/analyze_market_character.py の実測値)。
# JPYクロス4通貨の平均相関0.808により、名目5通貨は実質1.70通貨相当しかない。
EFFECTIVE_PAIR_COUNT = 1.70


def load_h4(pair: str, ds1: dict) -> pd.DataFrame:
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]
    return pd.DataFrame({
        "open": df["open"].resample("4h").first(),
        "high": df["high"].resample("4h").max(),
        "low": df["low"].resample("4h").min(),
        "close": df["close"].resample("4h").last(),
    }).dropna()


def build_features(h4: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """t時点の確定終値までの情報のみから特徴量を構築する (先読みなし).

    符号の意味を揃えてある: いずれも「値が大きい = 강気/買われすぎ方向」。
    したがって IC>0 ならモメンタム、IC<0 なら平均回帰が効いていると読める。
    """
    close, high, low = h4["close"], h4["high"], h4["low"]
    log_close = np.log(close)
    atr14 = atr_ind(high, low, close, length=14).replace(0, np.nan)

    feats: dict[str, pd.Series] = {}

    # --- モメンタム系 (過去リターン。IC>0ならモメンタム継続、IC<0なら反転) ---
    for label, n in (("1d", 6), ("3d", 18), ("1w", 42)):
        feats[f"mom_{label}"] = log_close.diff(n)

    # --- 移動平均系 (SYS-FX008の中核シグナルをそのまま特徴量化) ---
    sma10, sma20, sma50 = sma_ind(close, 10), sma_ind(close, 20), sma_ind(close, 50)
    feats["sma_cross_10_20_atr"] = (sma10 - sma20) / atr14
    feats["px_vs_sma20_atr"] = (close - sma20) / atr14
    feats["px_vs_sma50_atr"] = (close - sma50) / atr14

    # --- オシレーター系 (平均回帰を捉えることを意図した指標) ---
    feats["rsi14_centered"] = rsi(close, 14) - 50.0
    bb = bbands(close, length=20, std=2.0)
    feats["bb_pctb_centered"] = bb["BBP_20_2.0"] - 0.5
    roll_mean20 = close.rolling(20, min_periods=20).mean()
    roll_std20 = close.rolling(20, min_periods=20).std(ddof=0).replace(0, np.nan)
    feats["zscore_20"] = (close - roll_mean20) / roll_std20

    # --- チャネル内位置 (SYS-FX007のDonchianブレイク条件を連続量として表現) ---
    dc = donchian(high, low, lower_length=20, upper_length=20)
    dc_width = (dc["DCU"] - dc["DCL"]).replace(0, np.nan)
    feats["donchian_pos_centered"] = (close - dc["DCL"]) / dc_width - 0.5

    # --- サニティ対照: 純粋なノイズ。IC≈0 にならなければ実装バグを疑う ---
    feats["_random_control"] = pd.Series(
        rng.standard_normal(len(close)), index=close.index
    )

    return pd.DataFrame(feats)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman順位相関 = 順位に変換した上でのPearson相関 (scipy非依存の自前実装).

    pandas の corr(method="spearman") は内部で scipy を呼ぶため、
    本PJの依存構成 (scipy未導入) では使えない。同値は平均順位で扱う
    (pandas の rank() 既定動作 = scipy と同じ扱い)。
    """
    ra = np.array(a.rank(), dtype=float)
    rb = np.array(b.rank(), dtype=float)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def _fisher_p_value(r: float, n: int) -> float:
    """相関係数の両側p値をFisher z変換 + 正規近似で算出 (scipy非依存).

    z = arctanh(r) * sqrt(n-3) は帰無仮説 r=0 のもとで標準正規に従う。
    n>=30 を前提とするため正規近似で十分 (呼び出し側で n>=30 を保証)。
    """
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1.0:
        return float("nan")
    z = math.atanh(r) * math.sqrt(n - 3)
    # 両側p値 = 2 * (1 - Φ(|z|))
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def compute_ic(feature: pd.Series, fwd_ret: pd.Series, horizon_bars: int) -> dict:
    """Spearman IC と、重複窓を除いた非重複サブサンプルでの有意性を返す."""
    df = pd.concat([feature.rename("f"), fwd_ret.rename("r")], axis=1).dropna()
    if len(df) < 30:
        return {"ic": None, "n_overlapping": len(df), "n_independent": 0,
                "ic_independent": None, "p_value": None}

    ic_all = _spearman(df["f"], df["r"])

    # 重複窓によるt値膨張を避けるため、horizon本ごとに間引いた非重複サンプルで検定
    sub = df.iloc[::horizon_bars]
    if len(sub) < 30:
        return {"ic": round(ic_all, 4), "n_overlapping": len(df),
                "n_independent": len(sub), "ic_independent": None, "p_value": None}
    ic_sub = _spearman(sub["f"], sub["r"])
    p_sub = _fisher_p_value(ic_sub, len(sub))
    return {
        "ic": round(ic_all, 4),
        "n_overlapping": len(df),
        "n_independent": len(sub),
        "ic_independent": round(ic_sub, 4),
        "p_value": round(p_sub, 4) if np.isfinite(p_sub) else None,
    }


def main() -> int:
    print("=== シグナルファーストIC分析 (Train期間のみ、H4足) ===\n")
    print("IC>0=モメンタム的 / IC<0=平均回帰的 / IC≈0=予測力なし")
    print("有意性は重複窓を除いた非重複サブサンプルで評価 (t値膨張の回避)\n")

    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)

    rng = np.random.default_rng(RANDOM_SEED)
    per_pair: dict[str, dict] = {}
    # プール用: 通貨をまたいで特徴量とリターンを縦に積む
    pooled_rows: dict[str, list[pd.DataFrame]] = {h: [] for h in HORIZONS_H4}

    feature_names: list[str] = []
    for pair in PAIRS:
        h4 = load_h4(pair, ds1)
        feats = build_features(h4, rng)
        feature_names = list(feats.columns)
        log_close = np.log(h4["close"])

        per_pair[pair] = {"n_h4_bars": len(h4), "horizons": {}}
        for h_label, h_bars in HORIZONS_H4.items():
            # 前方リターン: t の終値 → t+h の終値 (先読みなし)
            fwd = log_close.shift(-h_bars) - log_close
            res = {name: compute_ic(feats[name], fwd, h_bars) for name in feats.columns}
            per_pair[pair]["horizons"][h_label] = res
            merged = feats.copy()
            merged["_fwd"] = fwd
            pooled_rows[h_label].append(merged.dropna())

    # --- プール集計 ---
    # 通貨間相関の割引: market_character.json の実効独立通貨数を使い、
    # プールの名目サンプル数を実効サンプル数へ縮小してから有意性を再評価する。
    # (JPYクロス4通貨は平均相関0.808でほぼ同一銘柄。名目nをそのまま使うと
    #  p値が楽観側に大きく歪む)
    eff_ratio = EFFECTIVE_PAIR_COUNT / len(PAIRS)
    n_tests = len(feature_names) * len(HORIZONS_H4)
    bonferroni_alpha = 0.05 / n_tests

    pooled: dict[str, dict] = {}
    for h_label, h_bars in HORIZONS_H4.items():
        allrows = pd.concat(pooled_rows[h_label], axis=0)
        pooled[h_label] = {}
        for name in feature_names:
            r = compute_ic(allrows[name], allrows["_fwd"], h_bars)
            # 相関割引後の実効サンプル数で p 値を引き直す
            if r["ic_independent"] is not None and r["n_independent"] > 0:
                n_eff = max(4, int(round(r["n_independent"] * eff_ratio)))
                p_eff = _fisher_p_value(r["ic_independent"], n_eff)
                r["n_effective"] = n_eff
                r["p_value_corr_adjusted"] = round(p_eff, 4) if np.isfinite(p_eff) else None
                r["survives_bonferroni"] = bool(
                    r["p_value_corr_adjusted"] is not None
                    and r["p_value_corr_adjusted"] < bonferroni_alpha
                )
            else:
                r["n_effective"] = 0
                r["p_value_corr_adjusted"] = None
                r["survives_bonferroni"] = False
            pooled[h_label][name] = r

    # --- 表示 ---
    print(f"多重検定: {n_tests}件 (特徴量{len(feature_names)} × ホライズン{len(HORIZONS_H4)})"
          f" → Bonferroni閾値 α={bonferroni_alpha:.5f}")
    print(f"通貨間相関の割引: 名目5通貨 → 実効{EFFECTIVE_PAIR_COUNT}通貨 (n を {eff_ratio:.2f} 倍に縮小)\n")

    for h_label in HORIZONS_H4:
        print(f"--- 前方リターン {h_label} (プール5通貨) ---")
        print(f"{'feature':<26}{'IC':>9}{'IC(独立)':>10}{'p(素)':>9}{'p(相関補正)':>13}{'n(実効)':>9}")
        rows = sorted(pooled[h_label].items(),
                      key=lambda kv: abs(kv[1]["ic"] or 0.0), reverse=True)
        for name, r in rows:
            ic = f"{r['ic']:.4f}" if r["ic"] is not None else "n/a"
            ici = f"{r['ic_independent']:.4f}" if r["ic_independent"] is not None else "n/a"
            pv = f"{r['p_value']:.4f}" if r["p_value"] is not None else "n/a"
            pv2 = f"{r['p_value_corr_adjusted']:.4f}" if r["p_value_corr_adjusted"] is not None else "n/a"
            if r["survives_bonferroni"]:
                mark = " **"   # 相関補正 + 多重検定補正の両方を突破
            elif r["p_value_corr_adjusted"] is not None and r["p_value_corr_adjusted"] < 0.05:
                mark = " *"    # 相関補正のみ突破 (多重検定未補正)
            else:
                mark = ""
            print(f"{name:<26}{ic:>9}{ici:>10}{pv:>9}{pv2:>13}{r['n_effective']:>9}{mark}")
        print()

    ctrl = pooled["1d"]["_random_control"]
    print(f"[サニティ確認] ランダム対照 (1d): IC={ctrl['ic']} / 独立IC={ctrl['ic_independent']}")
    print("  → これが0近傍でなければ実装に先読み等のバグがある可能性が高い\n")

    # --- 機械的な結論 ---
    real_feats = [f for f in feature_names if not f.startswith("_")]
    survivors = [(h, n) for h in HORIZONS_H4 for n in real_feats
                 if pooled[h][n]["survives_bonferroni"]]
    corr_only = [(h, n) for h in HORIZONS_H4 for n in real_feats
                 if (pooled[h][n]["p_value_corr_adjusted"] is not None
                     and pooled[h][n]["p_value_corr_adjusted"] < 0.05
                     and not pooled[h][n]["survives_bonferroni"])]
    print("=== 結論 ===")
    print(f"相関補正+多重検定補正の両方を突破した特徴量: {len(survivors)}件 {survivors}")
    print(f"相関補正のみ突破 (多重検定では脱落): {len(corr_only)}件 {corr_only}")

    # 4h ホライズンでの符号の一貫性 (個別には非有意でも、方向が揃うかは情報になる)
    signs_4h = [pooled["4h"][n]["ic"] for n in real_feats
                if pooled["4h"][n]["ic"] is not None]
    n_neg = sum(1 for s in signs_4h if s < 0)
    print(f"4hホライズンでIC<0 (平均回帰的) の特徴量: {n_neg}/{len(signs_4h)}件")
    print("  ※ 特徴量同士が高相関のため、これは独立した{}件の確認ではない点に注意"
          .format(len(signs_4h)))
    print()

    out_dir = ROOT / "research" / "method-notes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "signal_ic_baseline.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "timeframe": "H4",
            "horizons_h4_bars": HORIZONS_H4,
            "random_seed": RANDOM_SEED,
            "_note": (
                "戦略を介さない特徴量単体の予測力。Train期間のみ使用 (Val/Testは温存)。"
                "p_value は素の値、p_value_corr_adjusted は通貨間相関 (実効1.70通貨) で "
                "サンプル数を割り引いた値。survives_bonferroni は相関補正後に多重検定 "
                "補正も突破したかを示す。判断には survives_bonferroni を使うこと。"
            ),
            "effective_pair_count": EFFECTIVE_PAIR_COUNT,
            "n_tests": n_tests,
            "bonferroni_alpha": round(bonferroni_alpha, 6),
            "pooled": pooled,
            "per_pair": per_pair,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
