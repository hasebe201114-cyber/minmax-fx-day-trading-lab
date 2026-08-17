"""市場そのものの性質を測る基礎分析 (戦略に依存しない).

背景: SYS-FX007/008/009 の3戦略が連続REJECTとなった。3戦略はいずれも
「トレンドが持続する」という前提に賭けた設計 (SYS-FX007: ブレイク継続、
SYS-FX008: MAクロス追随、SYS-FX009: トレンド継続パターン) だったが、
**その前提自体をデータで検証したことが一度もなかった**。

本スクリプトは戦略を一切介さず、DS-1 の価格系列そのものから以下を測る:

1. 自己相関 AC(k): リターンに線形の持続性/反転があるか
2. 分散比 VR(q) (Lo-MacKinlay): VR>1 = モメンタム、VR<1 = 平均回帰、VR≈1 = ランダムウォーク
   VR(q) = Var(q期間リターン) / (q * Var(1期間リターン))
3. 通貨間相関と実効独立通貨数: プール評価の統計的前提が成立しているか
   N_eff ≈ k / (1 + (k-1) * rho_bar)

**Train期間のみを使用する** (Validation/Test は将来の検証のために温存、
本PJのHARKing防止プロトコルに準拠)。

Usage:
    python scripts/analyze_market_character.py

出力: research/method-notes/market_character.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"

# 検証するラグ/集約期間 (結果を見る前に固定)
AC_LAGS = [1, 2, 3, 5]
VR_Q_M15 = [4, 16, 96, 288]  # M15足で 1h / 4h / 1日 / 3日 相当 (提案6: 週内完結の時間軸)
VR_Q_H1 = [4, 24, 72]        # H1足で 4h / 1日 / 3日 相当 (提案6)
VR_Q_H4 = [2, 4, 6, 12]   # H4足で 8h / 16h / 1日 / 2日 相当
VR_Q_D1 = [2, 3, 5, 10]   # D1足で 2日 / 3日 / 1週 / 2週 相当


def load_m5(pair: str, ds1: dict) -> pd.DataFrame:
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]


def variance_ratio(log_returns: pd.Series, q: int) -> float:
    """Lo-MacKinlay 分散比. VR>1=モメンタム, VR<1=平均回帰, VR≈1=ランダムウォーク."""
    r = log_returns.dropna().values
    n = len(r)
    if n <= q:
        return float("nan")
    mu = r.mean()
    var_1 = ((r - mu) ** 2).sum() / (n - 1)
    if var_1 <= 0:
        return float("nan")
    r_q = pd.Series(r).rolling(q).sum().dropna().values
    var_q = ((r_q - q * mu) ** 2).sum() / (n - q + 1)
    return float(var_q / (q * var_1))


def analyze_timeframe(close: pd.Series, ac_lags: list[int], vr_qs: list[int]) -> dict:
    log_ret = np.log(close).diff()
    n = int(log_ret.dropna().shape[0])
    # 自己相関の標準誤差 (帰無仮説 AC=0 のもとで約 1/sqrt(n))
    se = 1.0 / np.sqrt(n) if n > 0 else float("nan")
    return {
        "n_bars": n,
        "ac_standard_error": round(se, 4),
        "autocorrelation": {f"lag_{k}": round(float(log_ret.autocorr(k)), 4) for k in ac_lags},
        "variance_ratio": {f"q_{q}": round(variance_ratio(log_ret, q), 4) for q in vr_qs},
    }


def effective_independent_count(corr: pd.DataFrame) -> tuple[float, float]:
    """平均非対角相関と、そこから推定される実効独立銘柄数を返す."""
    k = corr.shape[0]
    off_diag = corr.values[np.triu_indices(k, 1)]
    rho_bar = float(off_diag.mean())
    n_eff = k / (1.0 + (k - 1) * rho_bar) if (1.0 + (k - 1) * rho_bar) != 0 else float("nan")
    return rho_bar, n_eff


def main() -> int:
    print("=== 市場そのものの性質を測る基礎分析 (Train期間のみ) ===\n")
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)

    results: dict = {"m15": {}, "h1": {}, "h4": {}, "d1": {}}
    d1_returns: dict[str, pd.Series] = {}

    for tf_name, resample_rule, vr_qs in (
        ("m15", "15min", VR_Q_M15),  # 提案6: 週内完結を前提とした短い時間軸
        ("h1", "1h", VR_Q_H1),       # 提案6
        ("h4", "4h", VR_Q_H4),
        ("d1", "D", VR_Q_D1),
    ):
        print(f"--- {tf_name.upper()}足 ---")
        header = f"{'pair':<10}" + "".join(f"{'AC('+str(k)+')':>9}" for k in AC_LAGS) \
                 + "".join(f"{'VR('+str(q)+')':>9}" for q in vr_qs)
        print(header)
        for pair in PAIRS:
            close = load_m5(pair, ds1)["close"].resample(resample_rule).last().dropna()
            if tf_name == "d1":
                d1_returns[pair] = np.log(close).diff()
            stats = analyze_timeframe(close, AC_LAGS, vr_qs)
            results[tf_name][pair] = stats
            row = f"{pair:<10}"
            row += "".join(f"{stats['autocorrelation']['lag_'+str(k)]:>9.4f}" for k in AC_LAGS)
            row += "".join(f"{stats['variance_ratio']['q_'+str(q)]:>9.4f}" for q in vr_qs)
            print(row)
        print()

    # --- 通貨間相関と実効独立通貨数 ---
    print("--- D1リターンの通貨間相関 ---")
    corr_all = pd.DataFrame(d1_returns).dropna().corr()
    print(corr_all.round(3).to_string())
    rho_all, n_eff_all = effective_independent_count(corr_all)

    jpy_pairs = [p for p in PAIRS if p.endswith("JPY")]
    corr_jpy = corr_all.loc[jpy_pairs, jpy_pairs]
    rho_jpy, n_eff_jpy = effective_independent_count(corr_jpy)

    print()
    print(f"5通貨全体: 平均相関={rho_all:.3f} → 実効独立通貨数={n_eff_all:.2f} (名目5)")
    print(f"JPYクロス{len(jpy_pairs)}通貨: 平均相関={rho_jpy:.3f} → 実効独立通貨数={n_eff_jpy:.2f} (名目{len(jpy_pairs)})")
    print()
    print("含意: プール評価での permutation test は各トレードの独立性を仮定しているため、")
    print("      相関した通貨のトレードを別サンプルとして数えるとp値が楽観側に歪む。")

    results["cross_pair_correlation"] = {
        "matrix": corr_all.round(4).to_dict(),
        "all_pairs": {"mean_offdiag_corr": round(rho_all, 4),
                       "nominal_count": len(PAIRS),
                       "effective_independent_count": round(n_eff_all, 3)},
        "jpy_crosses_only": {"pairs": jpy_pairs,
                              "mean_offdiag_corr": round(rho_jpy, 4),
                              "nominal_count": len(jpy_pairs),
                              "effective_independent_count": round(n_eff_jpy, 3)},
    }

    out_dir = ROOT / "research" / "method-notes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "market_character.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "_note": "戦略に依存しない市場そのものの性質。Train期間のみ使用 (Val/Testは温存)。",
            "interpretation_guide": {
                "variance_ratio": "VR>1=モメンタム, VR<1=平均回帰, VR≈1=ランダムウォーク",
                "autocorrelation": "|AC| が標準誤差(1/sqrt(n))を明確に超えなければ線形の予測可能性なし",
                "effective_independent_count": "N_eff = k/(1+(k-1)*rho_bar)。プール評価の実質的な独立サンプル数の目安",
            },
            **results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
