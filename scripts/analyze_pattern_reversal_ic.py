"""シグナルファースト研究基盤 提案3: ダブルトップ/ボトムを「反転」として検証.

背景: SYS-FX009 v2 は「上位足(LT)トレンドの継続中に出る戻り高値/安値失敗」
という**継続**の文脈でのみダブルトップ/ボトムを有効とした (ダブルトップは
LT=DOWNの時のみ、ダブルボトムはLT=UPの時のみ)。3連続REJECTの根本原因分析
(market_character.json)で「H4/D1はVR≈1のランダムウォークで、トレンド持続
という前提自体がデータに支持されていない」と判明したことを受け、司令塔の
指摘通り**逆方向(反転/逆張り)の文脈**でこのパターンを検証する。

古典的なチャートパターン論の含意: ダブルトップ=上値を2回試して失敗=天井、
ダブルボトム=下値を2回試して失敗=大底。つまり本来は**反転**シグナルであり、
SYS-FX009 v2が採用した「継続」の文脈づけの方が、伝統的解釈からむしろ外れて
いた可能性がある。

方法: `derive_double_pattern_params.py` で確立した「ZigZag交互3点組 → 許容
誤差内 → ネックライン割れ」の検出ロジックをそのまま再利用し(重複実装しない)、
ネックライン割れ後の前方リターンを事前登録済みホライズン(analyze_signal_ic.py
と同一の 4h/1d/3d/1w)で測定する。LTフィルターなし(全パターン)・継続文脈
(LTがパターンの想定方向と一致)・反転文脈(LTがパターンの想定方向と逆)の
3通りに分けて比較する。

有意性判定は `backtest.permutation.permutation_test()` (本PJの実トレード
評価と全く同じ関数) をそのまま使い、通貨間相関による実効サンプル数補正
(EFFECTIVE_PAIR_COUNT=1.70) も適用する。

事前登録 (結果を見る前に固定):
    - パターン検出パラメータ: research/EXP-FX000003 で既に導出済みの値を
      そのまま流用 (zigzag_threshold_atr=2.0, pattern_tolerance_atr,
      max_bars_since_second_pivot)。ここでの検証のために新たに調整しない
    - 前方リターンは「パターンの想定方向」に符号を揃える
      (ダブルトップ=下落を予測するので符号反転、ダブルボトム=上昇予測なのでそのまま)
      → 値が正 = パターンの予測通り、負 = 逆行
    - ホライズン: 4h(1本) / 1d(6本) / 3d(18本) / 1w(42本)

出力: research/method-notes/pattern_reversal_ic.json
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

from minmax_fx_dt.backtest.permutation import permutation_test
from minmax_fx_dt.strategy.indicators import atr as atr_ind
from minmax_fx_dt.strategy.support_resistance import zigzag_pivots_typed

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"

# 事前登録: research/EXP-FX000003/10-result/double_pattern_params.json から転記 (再導出しない)
ZIGZAG_THRESHOLD_ATR = 2.0
LT_SMA_SHORT, LT_SMA_LONG = 10, 20
BREAK_SEARCH_CAP_BARS = 60  # derive_double_pattern_params.py と同一

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


def to_d1(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: m5[c].resample("D").agg(a) for c, a in
                          [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]}).dropna()


def lt_direction_series(d1: pd.DataFrame) -> pd.Series:
    """SMAクロスのみによるLT方向 (SYS-FX008/009と同一ロジック)."""
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


def find_pattern_events(pair: str, pattern_tolerance_atr: float) -> list[dict]:
    """ダブルトップ/ボトムのネックライン割れイベントを、LTフィルター無しで全件検出する."""
    m5 = load_m5(pair)
    h4 = to_h4(m5)
    d1 = to_d1(m5)
    atr_h4 = atr_ind(h4["high"], h4["low"], h4["close"], length=14)
    lt_dir = lt_direction_series(d1)
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
        neckline = float(h4["low" if kind1 == "HIGH" else "high"].iloc[idx2])

        search_end = min(idx3 + 1 + BREAK_SEARCH_CAP_BARS, len(h4))
        for j in range(idx3 + 1, search_end):
            broke = (
                (kind1 == "HIGH" and float(h4["low"].iloc[j]) < neckline)
                or (kind1 == "LOW" and float(h4["high"].iloc[j]) > neckline)
            )
            if not broke:
                continue
            lt_at_break = lt_dir.asof(h4.index[j])
            pattern_type = "double_top" if kind1 == "HIGH" else "double_bottom"
            expected_direction = "DOWN" if kind1 == "HIGH" else "UP"
            context = (
                "continuation" if lt_at_break == expected_direction
                else "reversal" if lt_at_break != "NONE" and pd.notna(lt_at_break)
                else "lt_unknown"
            )
            entry_log_close = log_close_h4.iloc[j]
            fwd_returns = {}
            for h_label, h_bars in HORIZONS_H4.items():
                if j + h_bars < len(h4):
                    raw_ret = float(log_close_h4.iloc[j + h_bars] - entry_log_close)
                    # パターンの想定方向に符号を揃える (ダブルトップは下落予測なので反転)
                    signed_ret = -raw_ret if pattern_type == "double_top" else raw_ret
                    fwd_returns[h_label] = signed_ret
                else:
                    fwd_returns[h_label] = None
            events.append({
                "pair": pair, "pattern_type": pattern_type, "break_idx": j,
                "break_time": str(h4.index[j]), "lt_direction": lt_at_break, "context": context,
                "fwd_returns_pips_signed": fwd_returns,
            })
            break  # このパターンについては最初の割れのみ記録 (陳腐化後の重複カウント防止)

    return events


def summarize_group(events: list[dict], horizon: str, group_filter) -> dict:
    rets = [e["fwd_returns_pips_signed"][horizon] for e in events
            if group_filter(e) and e["fwd_returns_pips_signed"][horizon] is not None]
    n = len(rets)
    if n == 0:
        return {"n": 0, "mean": None, "win_rate": None, "perm_p": None,
                "n_effective": 0, "p_value_corr_adjusted": None, "judgeable": False}
    mean_ret = float(np.mean(rets))
    win_rate = float(np.mean([r > 0 for r in rets]))
    n_eff = max(4, int(round(n * (EFFECTIVE_PAIR_COUNT / len(PAIRS)))))
    if n_eff < MIN_N_FOR_JUDGEMENT:
        return {"n": n, "mean": round(mean_ret, 6), "win_rate": round(win_rate, 3),
                "perm_p": None, "n_effective": n_eff, "p_value_corr_adjusted": None,
                "judgeable": False}
    # 実効サンプル数ぶんだけランダムに間引いてpermutation_testにかける (相関補正)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=n_eff, replace=False) if n_eff < n else np.arange(n)
    sub = [rets[i] for i in idx]
    result = permutation_test(sub, seed=42)
    return {
        "n": n, "mean": round(mean_ret, 6), "win_rate": round(win_rate, 3),
        "perm_p_raw_n": round(permutation_test(rets, seed=42).p_value, 4),
        "n_effective": n_eff,
        "p_value_corr_adjusted": round(result.p_value, 4),
        "judgeable": True,
    }


def main() -> int:
    print("=== 提案3: ダブルトップ/ボトムの反転(逆張り)仮説を検証 ===\n")
    with (ROOT / "research" / "EXP-FX000003" / "10-result" / "double_pattern_params.json").open(encoding="utf-8") as f:
        dp_params = json.load(f)
    pattern_tolerance_atr = dp_params["pattern_tolerance_atr"]
    print(f"パターン検出パラメータ (既存導出値を流用): "
          f"zigzag_threshold_atr={ZIGZAG_THRESHOLD_ATR}, pattern_tolerance_atr={pattern_tolerance_atr}\n")

    all_events: list[dict] = []
    for pair in PAIRS:
        events = find_pattern_events(pair, pattern_tolerance_atr)
        all_events.extend(events)
        n_top = sum(1 for e in events if e["pattern_type"] == "double_top")
        n_bot = sum(1 for e in events if e["pattern_type"] == "double_bottom")
        print(f"[{pair}] ダブルトップ割れ={n_top}件  ダブルボトム割れ={n_bot}件")

    print(f"\n全体イベント数: {len(all_events)}件\n")

    groups = {
        "all_patterns": lambda e: True,
        "double_top_all": lambda e: e["pattern_type"] == "double_top",
        "double_bottom_all": lambda e: e["pattern_type"] == "double_bottom",
        "continuation_context": lambda e: e["context"] == "continuation",
        "reversal_context": lambda e: e["context"] == "reversal",
        "double_top_continuation(SYS-FX009想定)": lambda e: e["pattern_type"] == "double_top" and e["context"] == "continuation",
        "double_top_reversal(新仮説)": lambda e: e["pattern_type"] == "double_top" and e["context"] == "reversal",
        "double_bottom_continuation(SYS-FX009想定)": lambda e: e["pattern_type"] == "double_bottom" and e["context"] == "continuation",
        "double_bottom_reversal(新仮説)": lambda e: e["pattern_type"] == "double_bottom" and e["context"] == "reversal",
    }

    # 多重検定補正: グループ×ホライズンの全組み合わせ数で Bonferroni (analyze_signal_ic.py /
    # analyze_conditional_ic.py と同じ方針)。all_patterns 等の重複集計グループも含めて
    # 保守的にカウントする。
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
                print(f"  {h_label}: n={r['n']:>4}  勝率={r['win_rate']}  平均(符号調整済)={r['mean']}  "
                      f"n_eff={r['n_effective']}  p(補正)={r['p_value_corr_adjusted']}{sig}{jflag}")
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
        print(f"  {g} / {h}: n={r['n']} 平均={r['mean']} 勝率={r['win_rate']} p={r['p_value_corr_adjusted']}")
    if not survivors:
        print("  なし。")
    print(f"相関補正のみ突破・多重検定では脱落: {len(naive_hits)}件 {naive_hits}")

    out_path = ROOT / "research" / "method-notes" / "pattern_reversal_ic.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "zigzag_threshold_atr": ZIGZAG_THRESHOLD_ATR,
            "pattern_tolerance_atr": pattern_tolerance_atr,
            "lt_sma_short": LT_SMA_SHORT,
            "lt_sma_long": LT_SMA_LONG,
            "horizons_h4_bars": HORIZONS_H4,
            "effective_pair_count": EFFECTIVE_PAIR_COUNT,
            "_note": (
                "前方リターンはパターンの想定方向(ダブルトップ=下落, ダブルボトム=上昇)に"
                "符号を揃えてある。値が正=パターンの予測通り、負=逆行。"
                "continuation=LTがパターンの想定方向と一致 (SYS-FX009 v2が採用した文脈)、"
                "reversal=LTが逆 (今回検証する新仮説の文脈)。"
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
