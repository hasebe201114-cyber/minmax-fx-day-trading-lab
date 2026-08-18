"""保留事項C対応: LT判定へのトレンド強度フィルター追加 — 「一つ上の足」検証.

背景: 司令塔から2点の指摘。
    1. チャート例示1〜3が表示期間だけ見るとレンジに見える。上昇/下落相場の
       判定の適正化を検討したい（現状のLT判定はD1のSMA(10,20)クロスのみで、
       トレンド強度フィルターを持たない）
    2. ADXでフィルターするなら、判定対象と同じ時間軸ではなく一つ上の時間軸で
       見るべきでは。過去の検証(SYS-FX007・SYS-FX008)はいずれもADX(14, D1)で
       D1自身を確認しており、一つ上(W1)を使ったことは一度も無いと判明済み

本スクリプトは、継続文脈(LT一致)のダブルトップ/ボトム・ブレイクイベント
(H1、`analyze_scaled_exit_diagnostic.find_continuation_entries`を再利用)に
対して、次の2種類のトレンド強度フィルターを追加した場合に前方リターンの
IC(方向性の予測力)が改善するかを比較検証する:

    - D1_ADX_TRENDING: ADX(14, D1, Wilder) が同ペアTrain分布のp70を上回る
      （＝過去に実際に使われていた「同一時間軸」のフィルター）
    - W1_ADX_TRENDING: ADX(14, W1, Wilder) が同ペアTrain分布のp70を上回る
      （＝司令塔提案の「一つ上の時間軸」のフィルター、過去に一度も未検証）

事前登録 (結果を見る前に固定):
    - ADX実装: Wilder標準実装(`indicators.adx_wilder`)。OBS000006でSYS-FX007の
      比較検証により非Wilder版より较正が妥当と判断された経緯を踏襲
    - 閾値: 各ペアTrain期間のADX分布のp70（OBS000006・SYS-FX008で確立した
      既存の百分位ルールをそのまま踏襲、今回のために新たに調整しない）
    - 前方リターン: H1、`signal_ic_intraday.py`のH1設定と同一ホライズン
      (4h=4本 / 1d=24本 / 3d=72本)、パターンの想定方向に符号を揃える
    - 比較する5グループ: continuation_all(参考、他スクリプトで既に0件と判明済み)・
      D1_ADX_TRENDING・D1_ADX_NOT_TRENDING・W1_ADX_TRENDING・W1_ADX_NOT_TRENDING
    - 多重検定: 5グループ×3ホライズン=15件でBonferroni補正
    - 通貨間相関による実効サンプル数補正(EFFECTIVE_PAIR_COUNT=1.70)を適用

出力: research/method-notes/trend_strength_timeframe_ic.json
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
from minmax_fx_dt.strategy.indicators import adx_wilder  # noqa: E402

PAIRS = base.PAIRS
TRAIN_START, TRAIN_END = base.TRAIN_START, base.TRAIN_END
EFFECTIVE_PAIR_COUNT = 1.70
MIN_N_FOR_JUDGEMENT = 30
ADX_LENGTH = 14
ADX_PERCENTILE = 70  # OBS000006・SYS-FX008で確立済みの既存ルールを踏襲
HORIZONS_H1 = {"4h": 4, "1d": 24, "3d": 72}  # signal_ic_intraday.py H1設定と同一


def to_w1(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: m5[c].resample("W").agg(a) for c, a in
                          [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]}).dropna()


def derive_adx_threshold(pair: str, timeframe_bars: pd.DataFrame) -> float:
    adx_df = adx_wilder(timeframe_bars["high"], timeframe_bars["low"], timeframe_bars["close"], length=ADX_LENGTH)
    values = adx_df[f"ADX_{ADX_LENGTH}"].dropna()
    return float(np.percentile(values, ADX_PERCENTILE))


def main() -> int:
    print("=== 保留事項C: LT判定へのトレンド強度フィルター「一つ上の足」検証 (Train期間・継続文脈のみ) ===\n")
    with (ROOT / "research" / "EXP-FX000003" / "10-result" / "double_pattern_params_h1.json").open(encoding="utf-8") as f:
        h1_params = json.load(f)

    print(f"事前登録: ADX({ADX_LENGTH}, Wilder)・p{ADX_PERCENTILE}閾値・H1前方リターン{HORIZONS_H1}\n")

    all_events: list[dict] = []
    d1_thresholds: dict[str, float] = {}
    w1_thresholds: dict[str, float] = {}

    for pair in PAIRS:
        m5 = base.load_m5(pair)
        h1 = base.to_h1(m5)
        d1 = base.to_d1(m5)
        w1 = to_w1(m5)

        d1_thresholds[pair] = derive_adx_threshold(pair, d1)
        w1_thresholds[pair] = derive_adx_threshold(pair, w1)
        d1_adx = adx_wilder(d1["high"], d1["low"], d1["close"], length=ADX_LENGTH)[f"ADX_{ADX_LENGTH}"]
        w1_adx = adx_wilder(w1["high"], w1["low"], w1["close"], length=ADX_LENGTH)[f"ADX_{ADX_LENGTH}"]

        entries = base.find_continuation_entries(pair, h1_params)
        log_close_h1 = np.log(h1["close"])

        for e in entries:
            j = e["entry_idx"]
            entry_ts = h1.index[j]
            d1_val = d1_adx.asof(entry_ts)
            w1_val = w1_adx.asof(entry_ts)
            fwd = {}
            for h_label, h_bars in HORIZONS_H1.items():
                if j + h_bars < len(h1):
                    raw_ret = float(log_close_h1.iloc[j + h_bars] - log_close_h1.iloc[j])
                    signed_ret = raw_ret if e["direction"] == "UP" else -raw_ret
                    fwd[h_label] = signed_ret
                else:
                    fwd[h_label] = None
            all_events.append({
                "pair": pair, "direction": e["direction"], "entry_time": e["entry_time"],
                "d1_adx": float(d1_val) if pd.notna(d1_val) else None,
                "w1_adx": float(w1_val) if pd.notna(w1_val) else None,
                "fwd_returns_signed": fwd,
            })
        print(f"[{pair}] 継続文脈エントリー={len(entries)}件  "
              f"D1_ADX閾値(p{ADX_PERCENTILE})={round(d1_thresholds[pair],2)}  "
              f"W1_ADX閾値(p{ADX_PERCENTILE})={round(w1_thresholds[pair],2)}")

    print(f"\n全体エントリー数: {len(all_events)}件\n")

    def d1_trending(e: dict) -> bool:
        return e["d1_adx"] is not None and e["d1_adx"] > d1_thresholds[e["pair"]]

    def w1_trending(e: dict) -> bool:
        return e["w1_adx"] is not None and e["w1_adx"] > w1_thresholds[e["pair"]]

    groups = {
        "continuation_all(参考)": lambda e: True,
        "D1_ADX_TRENDING(同一時間軸、過去実績あり)": d1_trending,
        "D1_ADX_NOT_TRENDING": lambda e: not d1_trending(e),
        "W1_ADX_TRENDING(一つ上の時間軸、司令塔提案・過去未検証)": w1_trending,
        "W1_ADX_NOT_TRENDING": lambda e: not w1_trending(e),
    }
    n_tests = len(groups) * len(HORIZONS_H1)
    bonferroni_alpha = 0.05 / n_tests
    print(f"多重検定: {n_tests}件 (グループ{len(groups)} × ホライズン{len(HORIZONS_H1)}) → Bonferroni閾値 α={bonferroni_alpha:.5f}\n")

    from minmax_fx_dt.backtest.permutation import permutation_test

    results: dict = {}
    for g_name, g_filter in groups.items():
        results[g_name] = {}
        print(f"--- {g_name} ---")
        for h_label in HORIZONS_H1:
            rets = [e["fwd_returns_signed"][h_label] for e in all_events
                    if g_filter(e) and e["fwd_returns_signed"][h_label] is not None]
            n = len(rets)
            if n == 0:
                results[g_name][h_label] = {"n": 0, "mean": None, "win_rate": None, "judgeable": False}
                print(f"  {h_label}: n=0")
                continue
            mean_ret = float(np.mean(rets))
            win_rate = float(np.mean([r > 0 for r in rets]))
            n_eff = max(4, int(round(n * (EFFECTIVE_PAIR_COUNT / len(PAIRS)))))
            judgeable = n_eff >= MIN_N_FOR_JUDGEMENT
            r = {"n": n, "mean": round(mean_ret, 6), "win_rate": round(win_rate, 3),
                 "n_effective": n_eff, "judgeable": judgeable}
            if judgeable:
                rng = np.random.default_rng(42)
                idx = rng.choice(n, size=n_eff, replace=False) if n_eff < n else np.arange(n)
                sub = [rets[i] for i in idx]
                pr = permutation_test(sub, seed=42)
                r["p_value_corr_adjusted"] = round(pr.p_value, 4)
                r["survives_bonferroni"] = bool(pr.p_value < bonferroni_alpha)
            else:
                r["p_value_corr_adjusted"] = None
                r["survives_bonferroni"] = False
            results[g_name][h_label] = r
            sig = " **" if r["survives_bonferroni"] else (
                " *" if r["p_value_corr_adjusted"] is not None and r["p_value_corr_adjusted"] < 0.05 else "")
            jflag = "" if judgeable else " [n不足・判定不能]"
            print(f"  {h_label}: n={n:>4}  勝率={r['win_rate']}  平均(符号調整済)={r['mean']}  "
                  f"n_eff={n_eff}  p(補正)={r.get('p_value_corr_adjusted')}{sig}{jflag}")
        print()

    survivors = [(g, h) for g in groups for h in HORIZONS_H1 if results[g][h]["survives_bonferroni"]]
    print(f"=== 結論 ===")
    print(f"相関補正+多重検定補正を突破した組み合わせ: {len(survivors)}件 {survivors}")

    out_path = ROOT / "research" / "method-notes" / "trend_strength_timeframe_ic.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "adx_length": ADX_LENGTH, "adx_percentile": ADX_PERCENTILE,
            "horizons_h1_bars": HORIZONS_H1,
            "effective_pair_count": EFFECTIVE_PAIR_COUNT,
            "d1_adx_thresholds": d1_thresholds,
            "w1_adx_thresholds": w1_thresholds,
            "n_total_events": len(all_events),
            "n_tests": n_tests, "bonferroni_alpha": round(bonferroni_alpha, 6),
            "results": results,
            "survivors": [{"group": g, "horizon": h} for g, h in survivors],
            "_note": (
                "継続文脈のダブルトップ/ボトム(H1)に、D1(同一時間軸・過去実績あり)と"
                "W1(一つ上の時間軸・司令塔提案・過去未検証)のADX(14,Wilder)強度フィルター"
                "をそれぞれ追加した場合の前方リターンIC比較。"
            ),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
