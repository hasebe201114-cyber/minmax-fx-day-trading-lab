"""SYS-FX009再検討: ダブルトップ/ボトムの「モメンタムダイバージェンス」IC探索的検証.

背景: SYS-FX009 v2はダブルトップ/ボトムの「形が成立したこと自体」を方向性
シグナルとして扱いREJECT確定した(パターンの形そのものに前方予測力なし、
`pattern_reversal_ic.json`で相関補正+多重検定補正を全滅で確認済み)。

司令塔の新しい仮説: 板情報(注文の厚み)が参照できないFXリテールでは、
「反発エネルギーの蓄積」を代理する手段として、2つの山(谷)を跨いだ
モメンタムのダイバージェンス(価格は同水準/更新するがRSIは弱含む)が
使えないか、という提案。これは`analyze_signal_ic.py`(その時点のRSI水準)
とも`pattern_reversal_ic.py`(パターンの形の有無)とも異なる、"2つのピボット
間の比較量"という未検証の特徴量であるため、SYS-FX009のREJECTを覆すか
どうかに関わらず新規の情報を持つ。

古典的ダイバージェンス定義:
    ダブルトップ(下落想定): 2本目の山でRSIが1本目の山より低い
        → divergence_score = RSI(P1) - RSI(P2)。正 = ダイバージェンス
          (弱気=下落を示唆)。
    ダブルボトム(上昇想定): 2本目の谷でRSIが1本目の谷より高い
        → divergence_score = RSI(P2) - RSI(P1)。正 = ダイバージェンス
          (強気=上昇を示唆)。
いずれも「値が大きい(正)ほどパターンの想定方向に効きそう」という向きに
符号を揃えてある。前方リターンも`pattern_reversal_ic.py`と同じく
パターンの想定方向に符号を揃え済み(正=想定通り)なので、
divergence_scoreと前方リターンのSpearman IC>0であれば
「ダイバージェンスが強いほど、パターンの想定方向により効く」ことを意味する。

方法:
    - パターン検出(ZigZag交互3点組→許容誤差内→ネックライン割れ)は
      `pattern_reversal_ic.py`のロジックをそのまま踏襲(重複実装だが
      スクリプト単体の自己完結という本PJの既存流儀に合わせる)。検出
      パラメータもEXP-FX000003の既存導出値をそのまま流用し、ここでの
      検証のために再調整しない
    - RSIは(14, H4)。2本目の山/谷確定バー(idx3)時点の値と1本目(idx1)
      時点の値を比較
    - LTフィルターは適用しない(all_patterns/double_top/double_bottomの
      3グループのみ)。continuation/reversal文脈での層別化は本検証の
      スコープ外(サンプルをこれ以上細分化すると検出力が失われるため)
    - 有意性判定は`permutation_test()`ではなくSpearman IC + Fisher z
      p値(`analyze_signal_ic.py`と同一方式)。理由: ここで測りたいのは
      「連続量としてのダイバージェンスの強さ」と前方リターンの相関で
      あり、2群比較ではないため
    - 多重検定補正: 3グループ×4ホライズン=12件でBonferroni
    - 通貨間相関による実効サンプル数補正: `market_character.json`の
      実測値EFFECTIVE_PAIR_COUNT=1.70を流用

これは探索的な軽量チェックであり、正式な事前登録トライアル(EXP起票)
ではない。ここで有意な相関が出た場合のみ、正式スクリーニングに進む
価値があると判断する。

出力: research/method-notes/pattern_divergence_ic.json
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from minmax_fx_dt.strategy.indicators import atr as atr_ind
from minmax_fx_dt.strategy.indicators import rsi as rsi_ind
from minmax_fx_dt.strategy.support_resistance import zigzag_pivots_typed

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"

# 事前登録: research/EXP-FX000003/10-result/double_pattern_params.json から転記 (再導出しない)
ZIGZAG_THRESHOLD_ATR = 2.0
BREAK_SEARCH_CAP_BARS = 60  # pattern_reversal_ic.py と同一
RSI_LENGTH = 14

HORIZONS_H4 = {"4h": 1, "1d": 6, "3d": 18, "1w": 42}  # analyze_signal_ic.py と同一
EFFECTIVE_PAIR_COUNT = 1.70  # market_character.json の実測値
MIN_N_FOR_JUDGEMENT = 30


def load_m5(pair: str) -> pd.DataFrame:
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]


def to_h4(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: m5[c].resample("4h").agg(a) for c, a in
                          [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]}).dropna()


def alternating_triplets(pivots: list[tuple[int, str]]) -> list[tuple[int, str, int, str, int, str]]:
    triplets = []
    for i in range(2, len(pivots)):
        idx1, kind1 = pivots[i - 2]
        idx2, kind2 = pivots[i - 1]
        idx3, kind3 = pivots[i]
        if kind1 == kind3 and kind1 != kind2:
            triplets.append((idx1, kind1, idx2, kind2, idx3, kind3))
    return triplets


def find_divergence_events(pair: str, pattern_tolerance_atr: float) -> list[dict]:
    """ダブルトップ/ボトムのネックライン割れイベントに、2ピボット間のRSIダイバージェンスを付与する."""
    m5 = load_m5(pair)
    h4 = to_h4(m5)
    atr_h4 = atr_ind(h4["high"], h4["low"], h4["close"], length=14)
    rsi_h4 = rsi_ind(h4["close"], length=RSI_LENGTH)
    log_close_h4 = np.log(h4["close"])

    pivots = zigzag_pivots_typed(h4["high"], h4["low"], atr_h4, ZIGZAG_THRESHOLD_ATR)
    triplets = alternating_triplets(pivots)

    events: list[dict] = []
    for idx1, kind1, idx2, _kind2, idx3, _kind3 in triplets:
        atr_neckline = atr_h4.iloc[idx2]
        if pd.isna(atr_neckline) or atr_neckline <= 0:
            continue
        p1 = float(h4["high" if kind1 == "HIGH" else "low"].iloc[idx1])
        p2 = float(h4["high" if kind1 == "HIGH" else "low"].iloc[idx3])
        if abs(p1 - p2) / float(atr_neckline) > pattern_tolerance_atr:
            continue

        rsi_p1 = rsi_h4.iloc[idx1]
        rsi_p2 = rsi_h4.iloc[idx3]
        if pd.isna(rsi_p1) or pd.isna(rsi_p2):
            continue
        pattern_type = "double_top" if kind1 == "HIGH" else "double_bottom"
        # 正 = ダイバージェンス(2本目のピボットで勢いが弱まっている) = パターンの想定方向を支持
        divergence_score = float(rsi_p1 - rsi_p2) if pattern_type == "double_top" else float(rsi_p2 - rsi_p1)

        neckline = float(h4["low" if kind1 == "HIGH" else "high"].iloc[idx2])
        search_end = min(idx3 + 1 + BREAK_SEARCH_CAP_BARS, len(h4))
        for j in range(idx3 + 1, search_end):
            broke = (
                (kind1 == "HIGH" and float(h4["low"].iloc[j]) < neckline)
                or (kind1 == "LOW" and float(h4["high"].iloc[j]) > neckline)
            )
            if not broke:
                continue
            entry_log_close = log_close_h4.iloc[j]
            fwd_returns = {}
            for h_label, h_bars in HORIZONS_H4.items():
                if j + h_bars < len(h4):
                    raw_ret = float(log_close_h4.iloc[j + h_bars] - entry_log_close)
                    signed_ret = -raw_ret if pattern_type == "double_top" else raw_ret
                    fwd_returns[h_label] = signed_ret
                else:
                    fwd_returns[h_label] = None
            events.append({
                "pair": pair, "pattern_type": pattern_type, "break_idx": j,
                "break_time": str(h4.index[j]), "divergence_score": divergence_score,
                "fwd_returns_pips_signed": fwd_returns,
            })
            break  # 最初の割れのみ記録 (陳腐化後の重複カウント防止、pattern_reversal_ic.pyと同一方針)

    return events


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy(dtype=float).copy()
    rb = pd.Series(b).rank().to_numpy(dtype=float).copy()
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def _fisher_p_value(r: float, n: int) -> float:
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1.0:
        return float("nan")
    z = math.atanh(r) * math.sqrt(n - 3)
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def summarize_group(events: list[dict], horizon: str, group_filter) -> dict:
    rows = [
        (e["divergence_score"], e["fwd_returns_pips_signed"][horizon])
        for e in events
        if group_filter(e) and e["fwd_returns_pips_signed"][horizon] is not None
    ]
    n = len(rows)
    if n == 0:
        return {"n": 0, "ic": None, "n_effective": 0, "p_value_corr_adjusted": None, "judgeable": False}

    div_scores = np.array([r[0] for r in rows])
    fwd_rets = np.array([r[1] for r in rows])
    n_eff = max(4, int(round(n * (EFFECTIVE_PAIR_COUNT / len(PAIRS)))))
    if n_eff < MIN_N_FOR_JUDGEMENT:
        ic_raw = _spearman(div_scores, fwd_rets)
        return {"n": n, "ic": round(ic_raw, 4), "n_effective": n_eff,
                "p_value_corr_adjusted": None, "judgeable": False}

    ic_raw = _spearman(div_scores, fwd_rets)
    p_raw_n = _fisher_p_value(ic_raw, n)
    p_corr = _fisher_p_value(ic_raw, n_eff)
    return {
        "n": n, "ic": round(ic_raw, 4), "p_raw_n": round(p_raw_n, 4) if np.isfinite(p_raw_n) else None,
        "n_effective": n_eff,
        "p_value_corr_adjusted": round(p_corr, 4) if np.isfinite(p_corr) else None,
        "judgeable": True,
    }


def main() -> int:
    print("=== SYS-FX009再検討: ダブルトップ/ボトムのRSIダイバージェンスIC探索 ===\n")
    with (ROOT / "research" / "EXP-FX000003" / "10-result" / "double_pattern_params.json").open(encoding="utf-8") as f:
        dp_params = json.load(f)
    pattern_tolerance_atr = dp_params["pattern_tolerance_atr"]
    print(f"パターン検出パラメータ (既存導出値を流用): "
          f"zigzag_threshold_atr={ZIGZAG_THRESHOLD_ATR}, pattern_tolerance_atr={pattern_tolerance_atr}, "
          f"rsi_length={RSI_LENGTH}\n")

    all_events: list[dict] = []
    for pair in PAIRS:
        events = find_divergence_events(pair, pattern_tolerance_atr)
        all_events.extend(events)
        n_top = sum(1 for e in events if e["pattern_type"] == "double_top")
        n_bot = sum(1 for e in events if e["pattern_type"] == "double_bottom")
        print(f"[{pair}] ダブルトップ割れ={n_top}件  ダブルボトム割れ={n_bot}件")

    print(f"\n全体イベント数: {len(all_events)}件\n")

    groups = {
        "all_patterns": lambda e: True,
        "double_top_all": lambda e: e["pattern_type"] == "double_top",
        "double_bottom_all": lambda e: e["pattern_type"] == "double_bottom",
    }

    n_tests = len(groups) * len(HORIZONS_H4)
    bonferroni_alpha = 0.05 / n_tests

    results: dict = {}
    for g_name, g_filter in groups.items():
        results[g_name] = {}
        print(f"--- {g_name} ---")
        for h_label in HORIZONS_H4:
            r = summarize_group(all_events, h_label, g_filter)
            r["survives_bonferroni"] = bool(
                r["judgeable"] and r["p_value_corr_adjusted"] is not None
                and r["p_value_corr_adjusted"] < bonferroni_alpha
            )
            results[g_name][h_label] = r
            if r["n"] > 0:
                if r["survives_bonferroni"]:
                    sig = " **"
                elif r["judgeable"] and r["p_value_corr_adjusted"] is not None and r["p_value_corr_adjusted"] < 0.05:
                    sig = " *(素の閾値のみ、多重検定未補正では脱落)"
                else:
                    sig = ""
                jflag = "" if r["judgeable"] else " [n不足・判定不能]"
                print(f"  {h_label}: n={r['n']:>4}  IC={r['ic']}  n_eff={r['n_effective']}  "
                      f"p(補正)={r['p_value_corr_adjusted']}{sig}{jflag}")
        print()

    print(f"多重検定: {n_tests}件 (グループ{len(groups)} × ホライズン{len(HORIZONS_H4)})"
          f" → Bonferroni閾値 α={bonferroni_alpha:.5f}\n")

    survivors = [(g, h) for g in groups for h in HORIZONS_H4 if results[g][h]["survives_bonferroni"]]
    naive_hits = [(g, h) for g in groups for h in HORIZONS_H4
                  if results[g][h]["judgeable"] and results[g][h]["p_value_corr_adjusted"] is not None
                  and results[g][h]["p_value_corr_adjusted"] < 0.05 and not results[g][h]["survives_bonferroni"]]
    print("=== 結論 ===")
    print(f"相関補正+多重検定補正の両方を突破した組み合わせ: {len(survivors)}件")
    for g, h in survivors:
        r = results[g][h]
        print(f"  {g} / {h}: n={r['n']} IC={r['ic']} p={r['p_value_corr_adjusted']}")
    if not survivors:
        print("  なし。")
    print(f"相関補正のみ突破・多重検定では脱落: {len(naive_hits)}件 {naive_hits}")

    out_path = ROOT / "research" / "method-notes" / "pattern_divergence_ic.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "zigzag_threshold_atr": ZIGZAG_THRESHOLD_ATR,
            "pattern_tolerance_atr": pattern_tolerance_atr,
            "rsi_length": RSI_LENGTH,
            "horizons_h4_bars": HORIZONS_H4,
            "effective_pair_count": EFFECTIVE_PAIR_COUNT,
            "_note": (
                "divergence_score = 2本目のピボットで勢いが弱まっている度合い"
                "(ダブルトップ=RSI(P1)-RSI(P2)、ダブルボトム=RSI(P2)-RSI(P1))。"
                "正 = 古典的ダイバージェンス(パターンの想定方向を支持)。"
                "前方リターンはパターンの想定方向に符号を揃え済み(正=想定通り)。"
                "IC(divergence_score, fwd_return)>0なら「ダイバージェンスが強いほど"
                "パターンが機能する」ことを意味する。"
            ),
            "n_total_events": len(all_events),
            "n_tests": n_tests,
            "bonferroni_alpha": round(bonferroni_alpha, 6),
            "groups": results,
            "survivors": [{"group": g, "horizon": h} for g, h in survivors],
            "naive_only_hits": [{"group": g, "horizon": h} for g, h in naive_hits],
            "events": all_events,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
