"""保留事項C対応 続き: W1_ADXトレンド強度フィルターの閾値感度検証.

背景: `analyze_trend_strength_timeframe.py` で、継続文脈のH1エントリーに
W1_ADX(14, Wilder)のp70フィルターを追加すると、D1_ADX(同一時間軸)より
一貫して良好なIC(3d: 勝率52.4%・平均+0.00235・p=0.123)を示した。
Bonferroni補正後は非有意だったが、p70という1点だけの検証だったため、
司令塔選択「b.(閾値/実装を変えて再検証)」を受け、閾値の感度を確認する。

事前登録 (結果を見る前に固定):
    - 候補パーセンタイル: p50 / p60 / p70(既検証・参考として再掲) / p80
      （p70で最良だった3dホライズンを中心に、緩める方向(p50/p60)と
      厳しくする方向(p80)の両方を確認する。p90は候補数が実効n<30に
      落ちる可能性が高いため今回は対象外とし、必要なら別途検討する）
    - ADX実装: Wilder標準実装のみ(前回と同一、実装自体は変更しない)
    - 対象ホライズン: 前回最良だった3dのみに絞る(4h/1dは前回時点で
      3dより一貫して劣っていたため、多重検定の対象を広げすぎない)
    - 多重検定: 4パーセンタイル(50/60/70/80)×1ホライズン(3d)=4件で
      Bonferroni補正 (前回の15件とは別の検定ファミリーとして扱う。
      「p70が良かったから追加検証する」という後知恵バイアスを避けるため、
      p70も含めた4点すべてを結果を見る前に対象として登録する)

出力: research/method-notes/w1_adx_percentile_sweep.json
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
import analyze_trend_strength_timeframe as trend  # noqa: E402
from minmax_fx_dt.backtest.permutation import permutation_test  # noqa: E402
from minmax_fx_dt.strategy.indicators import adx_wilder  # noqa: E402

PAIRS = base.PAIRS
TRAIN_START, TRAIN_END = base.TRAIN_START, base.TRAIN_END
EFFECTIVE_PAIR_COUNT = 1.70
MIN_N_FOR_JUDGEMENT = 30
ADX_LENGTH = 14
PERCENTILE_CANDIDATES = [50, 60, 70, 80]  # 事前登録: 結果を見る前に固定
HORIZON_LABEL, HORIZON_BARS = "3d", 72  # 前回最良だったホライズンのみ


def main() -> int:
    print("=== 保留事項C続き: W1_ADXフィルターの閾値感度検証 (Train期間・継続文脈・3dのみ) ===\n")
    with (ROOT / "research" / "EXP-FX000003" / "10-result" / "double_pattern_params_h1.json").open(encoding="utf-8") as f:
        h1_params = json.load(f)

    print(f"事前登録: パーセンタイル候補{PERCENTILE_CANDIDATES}・ホライズン{HORIZON_LABEL}({HORIZON_BARS}本)のみ\n")

    # 通貨ごとにW1 ADX系列とイベントを1回だけ計算し、パーセンタイルだけ振って使い回す
    per_pair_data = {}
    for pair in PAIRS:
        m5 = base.load_m5(pair)
        h1 = base.to_h1(m5)
        w1 = trend.to_w1(m5)
        w1_adx = adx_wilder(w1["high"], w1["low"], w1["close"], length=ADX_LENGTH)[f"ADX_{ADX_LENGTH}"]
        w1_adx_values = w1_adx.dropna()

        entries = base.find_continuation_entries(pair, h1_params)
        log_close_h1 = np.log(h1["close"])
        events = []
        for e in entries:
            j = e["entry_idx"]
            entry_ts = h1.index[j]
            w1_val = w1_adx.asof(entry_ts)
            if j + HORIZON_BARS >= len(h1):
                continue
            raw_ret = float(log_close_h1.iloc[j + HORIZON_BARS] - log_close_h1.iloc[j])
            signed_ret = raw_ret if e["direction"] == "UP" else -raw_ret
            events.append({"w1_adx": float(w1_val) if pd.notna(w1_val) else None, "ret": signed_ret})

        per_pair_data[pair] = {"w1_adx_values": w1_adx_values, "events": events}
        print(f"[{pair}] エントリー(3d先まで測定可)={len(events)}件")

    n_tests = len(PERCENTILE_CANDIDATES)
    bonferroni_alpha = 0.05 / n_tests
    print(f"\n多重検定: {n_tests}件 (パーセンタイル候補数) → Bonferroni閾値 α={bonferroni_alpha:.5f}\n")

    results = {}
    for pct in PERCENTILE_CANDIDATES:
        rets = []
        thresholds = {}
        for pair in PAIRS:
            thr = float(np.percentile(per_pair_data[pair]["w1_adx_values"], pct))
            thresholds[pair] = round(thr, 2)
            for ev in per_pair_data[pair]["events"]:
                if ev["w1_adx"] is not None and ev["w1_adx"] > thr:
                    rets.append(ev["ret"])

        n = len(rets)
        mean_ret = float(np.mean(rets)) if n else None
        win_rate = float(np.mean([r > 0 for r in rets])) if n else None
        n_eff = max(4, int(round(n * (EFFECTIVE_PAIR_COUNT / len(PAIRS))))) if n else 0
        judgeable = n_eff >= MIN_N_FOR_JUDGEMENT
        r = {"percentile": pct, "thresholds": thresholds, "n": n, "mean": round(mean_ret, 6) if mean_ret is not None else None,
             "win_rate": round(win_rate, 3) if win_rate is not None else None, "n_effective": n_eff, "judgeable": judgeable}
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
        results[f"p{pct}"] = r

        sig = " **" if r["survives_bonferroni"] else (
            " *" if r["p_value_corr_adjusted"] is not None and r["p_value_corr_adjusted"] < 0.05 else "")
        jflag = "" if judgeable else " [n不足・判定不能]"
        print(f"p{pct}: n={n:>4}  勝率={r['win_rate']}  平均(符号調整済)={r['mean']}  "
              f"n_eff={n_eff}  p(補正)={r['p_value_corr_adjusted']}{sig}{jflag}")

    survivors = [k for k, r in results.items() if r["survives_bonferroni"]]
    best = min((r for r in results.values() if r["p_value_corr_adjusted"] is not None),
               key=lambda r: r["p_value_corr_adjusted"], default=None)
    print(f"\n=== 結論 ===")
    print(f"Bonferroni補正を突破したパーセンタイル: {len(survivors)}件 {survivors}")
    if best:
        print(f"最良のp値: p{best['percentile']} (p={best['p_value_corr_adjusted']}, n={best['n']}, 勝率={best['win_rate']})")

    out_path = ROOT / "research" / "method-notes" / "w1_adx_percentile_sweep.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "percentile_candidates": PERCENTILE_CANDIDATES,
            "horizon": {"label": HORIZON_LABEL, "bars": HORIZON_BARS},
            "n_tests": n_tests, "bonferroni_alpha": round(bonferroni_alpha, 6),
            "results": results,
            "survivors": survivors,
            "_note": "W1_ADX(14,Wilder)フィルターの閾値パーセンタイルを50/60/70/80で振った感度検証。3dホライズンのみ対象。",
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
