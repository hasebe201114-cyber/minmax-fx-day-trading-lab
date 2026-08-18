"""提案5 成功基準の検証: 過去3戦略の代表セルを新方式(クラスタブロック)で
再評価し、旧方式(独立符号シャッフル)との比較で結論(いずれもREJECT)が
変わらないことを確認する.

対象:
    - SYS-FX007 (EXP-FX000001): D1_DataDrivenDonchianRR, train
      (旧報告: pooled n=125, perm_p=0.80)
    - SYS-FX008 (EXP-FX000002): E1_trail(最終選定プリセット), train
      (旧報告: pooled n=163, perm_p=0.34。3試行中もっともperm_pがα=0.05に
      近く、クラスタ補正の影響を確認する上で最も厳しいケース)
    - SYS-FX009 (EXP-FX000003): baseline, train/validation/test
      (旧報告: perm_p 0.271/0.774/0.474、回転売買バグ修正後の最終数値)

各セルについて、5通貨の trade_pnls を通貨ラベル付きでプールし、
permutation_test()(独立符号シャッフル、旧方式)と
permutation_test_clustered()(通貨クラスタ単位、新方式)の両方でp値を計算し、
新方式のp値が旧方式以上(=より保守的)であること、かつ有意水準0.05を跨いで
結論が変わらないことを確認する。

出力: research/method-notes/cluster_correction_verification.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from minmax_fx_dt.backtest.permutation import permutation_test, permutation_test_clustered  # noqa: E402

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]


def load_pooled(result_dir: Path, filename_fn) -> tuple[list[float], list[str]]:
    pnls: list[float] = []
    pairs: list[str] = []
    for pair in PAIRS:
        path = result_dir / filename_fn(pair)
        if not path.exists():
            print(f"  [警告] 見つからない: {path.name}")
            continue
        with path.open(encoding="utf-8") as f:
            d = json.load(f)
        trade_pnls = d.get("trade_pnls", [])
        pnls.extend(trade_pnls)
        pairs.extend([pair] * len(trade_pnls))
    return pnls, pairs


def evaluate_cell(label: str, pnls: list[float], pairs: list[str], threshold: float = 0.05) -> dict:
    naive = permutation_test(pnls, seed=42)
    clustered = permutation_test_clustered(pnls, pairs, seed=42)
    conclusion_unchanged = (naive.p_value < threshold) == (clustered.p_value < threshold)
    result = {
        "label": label,
        "n_trades": len(pnls),
        "naive_p_value": round(naive.p_value, 4),
        "clustered_p_value": round(clustered.p_value, 4),
        "clustered_more_conservative": clustered.p_value >= naive.p_value,
        "naive_significant": naive.p_value < threshold,
        "clustered_significant": clustered.p_value < threshold,
        "conclusion_unchanged": conclusion_unchanged,
    }
    flag = "OK" if conclusion_unchanged else "**結論変化**"
    print(f"[{label}] n={len(pnls)}  旧p={naive.p_value:.4f}  新p={clustered.p_value:.4f}  "
          f"(保守化: {'○' if result['clustered_more_conservative'] else '×'})  {flag}")
    return result


def main() -> int:
    print("=== 提案5 検証: 過去3戦略の代表セルをクラスタ補正版で再評価 ===\n")
    results = []

    print("--- SYS-FX007: D1_DataDrivenDonchianRR, train ---")
    d1_dir = ROOT / "research" / "EXP-FX000001" / "10-result" / "train_val_test"
    pnls, pairs = load_pooled(d1_dir, lambda p: f"tvt_D1_DataDrivenDonchianRR_{p}_train.json")
    results.append(evaluate_cell("SYS-FX007_D1_train", pnls, pairs))

    print("\n--- SYS-FX008: E1_trail(最終選定), train ---")
    fx008_dir = ROOT / "research" / "EXP-FX000002" / "10-result" / "train_val_test"
    pnls, pairs = load_pooled(fx008_dir, lambda p: f"tvt_{p}_train_E1_trail.json")
    results.append(evaluate_cell("SYS-FX008_E1_trail_train", pnls, pairs))

    print("\n--- SYS-FX009: baseline, train/validation/test ---")
    fx009_dir = ROOT / "research" / "EXP-FX000003" / "10-result" / "train_val_test"
    for period in ["train", "validation", "test"]:
        pnls, pairs = load_pooled(fx009_dir, lambda p, period=period: f"tvt_{p}_{period}.json")
        results.append(evaluate_cell(f"SYS-FX009_baseline_{period}", pnls, pairs))

    n_changed = sum(1 for r in results if not r["conclusion_unchanged"])
    n_more_conservative = sum(1 for r in results if r["clustered_more_conservative"])
    print(f"\n=== 結論 ===")
    print(f"検証セル数: {len(results)}件")
    print(f"クラスタ補正で結論(有意水準0.05を跨ぐか)が変わったセル: {n_changed}件")
    print(f"クラスタ補正がより保守的(p値が増加)だったセル: {n_more_conservative}/{len(results)}件")
    if n_changed == 0:
        print("→ 提案5の成功基準3(過去のREJECT判定が変わらないこと)を満たす")
    else:
        print("→ 要調査: 結論が変化したセルがある")

    out_path = ROOT / "research" / "method-notes" / "cluster_correction_verification.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "significance_threshold": 0.05,
            "n_cells": len(results),
            "n_conclusion_changed": n_changed,
            "n_more_conservative": n_more_conservative,
            "results": results,
            "_note": (
                "permutation_test()(独立符号シャッフル、旧方式)と"
                "permutation_test_clustered()(通貨クラスタ単位、新方式)を"
                "過去3戦略の代表セルで比較。新方式が旧方式よりp値が大きく"
                "(保守的)、かつ有意水準0.05を跨ぐ結論変化が無いことを確認する"
                "提案5の成功基準3の検証。"
            ),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
