"""別取引所データ(Dukascopy)によるIC分析の頑健性チェック.

背景: `analyze_signal_ic.py`(2026-08-17)は、GMOコインの公開klines APIが
データ保持できる範囲(2023-10-27以降)のTrain期間(2023-11-01〜2025-03-31)
で、標準的なテクニカル指標11種の方向性予測力(IC)を測定し、相関補正+多重
検定補正を突破した特徴量は0件という結論に達した。

司令塔から「別取引所データで評価できる可能性はあるか」という提起を受け、
GMOの壁より前の期間(2018-11-01〜2023-10-26、約5年、Trainと非重複)を
認証不要の公開ヒストリカルフィード(Dukascopy)から取得し、**同一の特徴量
定義・同一のIC測定方法**で、この結論がデータソース・期間を変えても
再現するかを確認する。

**重要な位置づけ**: これは正式なKPI評価ではない。Dukascopyのスプレッドは
GMOコインの実際の提示スプレッドと異なり、本PJのコストモデルはGMO専用に
較正されているため、このデータを使ったバックテストの収益性評価は成立
しない。ここで測るのはコスト抜きの「価格系列そのものの方向性予測力」の
頑健性のみであり、採用判断には使わない探索的な補足調査。

方法論はanalyze_signal_ic.pyと完全に共有する(build_features/compute_ic/
_spearman/_fisher_p_valueをそのまま再利用、重複実装しない)。実効独立
通貨数のみ、この新しい期間のデータから改めて計算し直す(2023-11〜2025-03
の1.70をそのまま流用しない)。

Usage:
    python scripts/analyze_signal_ic_dukascopy.py

出力: research/method-notes/signal_ic_dukascopy_robustness.json
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

import analyze_signal_ic as base  # noqa: E402 (build_features/compute_ic/_spearman/_fisher_p_value を再利用)

PAIRS = base.PAIRS  # ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
HORIZONS_H4 = base.HORIZONS_H4
RANDOM_SEED = base.RANDOM_SEED

DUKA_DIR = ROOT / "data" / "raw" / "dukascopy"
DUKA_START, DUKA_END = "2018-11-01", "2023-10-26"  # GMOの壁(2023-10-27)より前、Trainと非重複
FILE_TAG = "2018-11_2023-10"  # fetch_dukascopy_h1.py --start/--end に対応


def load_h4_dukascopy(pair: str) -> pd.DataFrame:
    """DukascopyのH1 CSVをH4へ集約 (analyze_signal_ic.load_h4と同一の集約方式)."""
    path = DUKA_DIR / f"ohlcv_{pair}_h1_{FILE_TAG}.csv"
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[(df.index >= DUKA_START) & (df.index <= DUKA_END)]
    return pd.DataFrame({
        "open": df["open"].resample("4h").first(),
        "high": df["high"].resample("4h").max(),
        "low": df["low"].resample("4h").min(),
        "close": df["close"].resample("4h").last(),
    }).dropna()


def effective_pair_count_this_period(h4_by_pair: dict[str, pd.DataFrame]) -> float:
    """analyze_market_character.effective_independent_countと同一の定義で
    この期間のD1リターン通貨間相関から実効独立通貨数を再計算する."""
    d1_returns = {}
    for pair, h4 in h4_by_pair.items():
        d1_close = h4["close"].resample("D").last().dropna()
        d1_returns[pair] = np.log(d1_close).diff()
    corr = pd.DataFrame(d1_returns).dropna().corr()
    k = corr.shape[0]
    off_diag = corr.values[np.triu_indices(k, 1)]
    rho_bar = float(off_diag.mean())
    n_eff = k / (1.0 + (k - 1) * rho_bar) if (1.0 + (k - 1) * rho_bar) != 0 else float("nan")
    return rho_bar, n_eff, corr


def main() -> int:
    print("=== Dukascopy頑健性チェック: 別取引所データでのIC分析 ===")
    print(f"期間: {DUKA_START} 〜 {DUKA_END} (GMOの壁2023-10-27より前、Trainと非重複)")
    print("※ KPI評価ではない。コスト抜きの方向性予測力のみの頑健性チェック\n")

    missing = [p for p in PAIRS if not (DUKA_DIR / f"ohlcv_{p}_h1_{FILE_TAG}.csv").exists()]
    if missing:
        print(f"[ERROR] 未取得のペア: {missing}")
        print("先に scripts/data/fetch_dukascopy_h1.py を実行してください")
        return 1

    rng = np.random.default_rng(RANDOM_SEED)
    h4_by_pair = {pair: load_h4_dukascopy(pair) for pair in PAIRS}

    rho_bar, eff_pair_count, corr = effective_pair_count_this_period(h4_by_pair)
    eff_ratio = eff_pair_count / len(PAIRS)
    print(f"この期間のD1通貨間相関: 平均{rho_bar:.3f} → 実効独立通貨数={eff_pair_count:.2f} (名目{len(PAIRS)})")
    print("(参考: GMO期間2023-11〜2025-03の実効独立通貨数は1.70)\n")

    per_pair: dict[str, dict] = {}
    pooled_rows: dict[str, list[pd.DataFrame]] = {h: [] for h in HORIZONS_H4}
    feature_names: list[str] = []

    for pair in PAIRS:
        h4 = h4_by_pair[pair]
        feats = base.build_features(h4, rng)
        feature_names = list(feats.columns)
        log_close = np.log(h4["close"])

        per_pair[pair] = {"n_h4_bars": len(h4), "horizons": {}}
        for h_label, h_bars in HORIZONS_H4.items():
            fwd = log_close.shift(-h_bars) - log_close
            res = {name: base.compute_ic(feats[name], fwd, h_bars) for name in feats.columns}
            per_pair[pair]["horizons"][h_label] = res
            merged = feats.copy()
            merged["_fwd"] = fwd
            pooled_rows[h_label].append(merged.dropna())

    n_tests = len(feature_names) * len(HORIZONS_H4)
    bonferroni_alpha = 0.05 / n_tests

    pooled: dict[str, dict] = {}
    for h_label, h_bars in HORIZONS_H4.items():
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
                    r["p_value_corr_adjusted"] is not None
                    and r["p_value_corr_adjusted"] < bonferroni_alpha
                )
            else:
                r["n_effective"] = 0
                r["p_value_corr_adjusted"] = None
                r["survives_bonferroni"] = False
            pooled[h_label][name] = r

    print(f"多重検定: {n_tests}件 → Bonferroni閾値 α={bonferroni_alpha:.5f}\n")
    for h_label in HORIZONS_H4:
        print(f"--- 前方リターン {h_label} (プール5通貨、Dukascopy) ---")
        print(f"{'feature':<26}{'IC':>9}{'IC(独立)':>10}{'p(素)':>9}{'p(相関補正)':>13}{'n(実効)':>9}")
        rows = sorted(pooled[h_label].items(),
                      key=lambda kv: abs(kv[1]["ic"] or 0.0), reverse=True)
        for name, r in rows:
            ic = f"{r['ic']:.4f}" if r["ic"] is not None else "n/a"
            ici = f"{r['ic_independent']:.4f}" if r["ic_independent"] is not None else "n/a"
            pv = f"{r['p_value']:.4f}" if r["p_value"] is not None else "n/a"
            pv2 = f"{r['p_value_corr_adjusted']:.4f}" if r["p_value_corr_adjusted"] is not None else "n/a"
            mark = " **" if r["survives_bonferroni"] else (
                " *" if r["p_value_corr_adjusted"] is not None and r["p_value_corr_adjusted"] < 0.05 else "")
            print(f"{name:<26}{ic:>9}{ici:>10}{pv:>9}{pv2:>13}{r['n_effective']:>9}{mark}")
        print()

    ctrl = pooled["1d"]["_random_control"]
    print(f"[サニティ確認] ランダム対照 (1d): IC={ctrl['ic']} / 独立IC={ctrl['ic_independent']}\n")

    real_feats = [f for f in feature_names if not f.startswith("_")]
    survivors = [(h, n) for h in HORIZONS_H4 for n in real_feats if pooled[h][n]["survives_bonferroni"]]
    corr_only = [(h, n) for h in HORIZONS_H4 for n in real_feats
                 if (pooled[h][n]["p_value_corr_adjusted"] is not None
                     and pooled[h][n]["p_value_corr_adjusted"] < 0.05
                     and not pooled[h][n]["survives_bonferroni"])]
    print("=== 結論 ===")
    print(f"相関補正+多重検定補正の両方を突破した特徴量: {len(survivors)}件 {survivors}")
    print(f"相関補正のみ突破 (多重検定では脱落): {len(corr_only)}件 {corr_only}")

    gmo_result = json.loads((ROOT / "research" / "method-notes" / "signal_ic_baseline.json").read_text())
    gmo_survivors = sum(1 for h in gmo_result["pooled"] for n, r in gmo_result["pooled"][h].items()
                         if r.get("survives_bonferroni"))
    print(f"\n[比較] GMO期間(2023-11〜2025-03)の結果: 突破{gmo_survivors}件")
    print(f"       Dukascopy期間({DUKA_START}〜{DUKA_END})の結果: 突破{len(survivors)}件")
    same_conclusion = (gmo_survivors == 0) == (len(survivors) == 0)
    print(f"       結論の一致: {'YES (両方0件)' if same_conclusion else 'NO (結論が異なる、要精査)'}")
    print()

    out_dir = ROOT / "research" / "method-notes"
    out_path = out_dir / "signal_ic_dukascopy_robustness.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "purpose": "頑健性チェック専用(KPI評価ではない)。GMO壁より前の期間・別データソースでのIC再現性確認",
            "data_source": "Dukascopy公開ヒストリカルフィード(認証不要、BID H1候補足)",
            "period": [DUKA_START, DUKA_END],
            "timeframe": "H4 (H1から集約)",
            "horizons_h4_bars": HORIZONS_H4,
            "random_seed": RANDOM_SEED,
            "effective_pair_count": round(eff_pair_count, 3),
            "mean_offdiag_corr_d1": round(rho_bar, 4),
            "n_tests": n_tests,
            "bonferroni_alpha": round(bonferroni_alpha, 6),
            "comparison_gmo_baseline": {
                "gmo_period": gmo_result["train_period"],
                "gmo_survivors_count": gmo_survivors,
                "dukascopy_survivors_count": len(survivors),
                "same_conclusion": same_conclusion,
            },
            "pooled": pooled,
            "per_pair": per_pair,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
