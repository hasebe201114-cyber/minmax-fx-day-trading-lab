"""15 セル (5 通貨 × 3 期間) の TVT 結果を統合する集計スクリプト.

完了済みの tvt_{preset}_{pair}_{period}.json を全部読み込み、
5 通貨 × 3 期間のマトリクスと Train→Val→Test 推移 (過学習所見) を出力する。

Usage:
  python scripts/aggregate_tvt_results.py
  python scripts/aggregate_tvt_results.py --preset A1_A2_combined
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "research" / "EXP-FX000001" / "10-result" / "train_val_test"

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
PERIODS = ["train", "validation", "test"]
PERIOD_LABELS = {
    "train": "Train (2020-2022)",
    "validation": "Val (2023-2024H1)",
    "test": "Test (2024H2-2025)",
}
PERIOD_KEYS_KPI = {
    "K1m_sharpe_ge_0.4": "Sharpe≥0.4",
    "K1m_pf_ge_1.2":     "PF≥1.2",
    "K1m_expectancy_pos": "期待値>0",
    "K2m_dd_monthly_le_10": "月DD≤10%",
    "K2m_dd_yearly_le_20":  "年DD≤20%",
    "K3m_max_consec_le_5":  "連敗≤5",
    "K4m_payoff_ge_1.5":    "PF比≥1.5",
    "K7m_margin_le_30":     "証拠金≤30%",
    "weak_breakout_ge_30":  "弱ブレイク≥30%",
}


def collect(preset: str) -> dict:
    """tvt_{preset}_{pair}_{period}.json を全部読み込んで {(pair, period): data} で返す."""
    cells = {}
    for pair in PAIRS:
        for period in PERIODS:
            f = RESULT_DIR / f"tvt_{preset}_{pair}_{period}.json"
            if f.exists():
                cells[(pair, period)] = json.loads(f.read_text(encoding="utf-8"))
    return cells


def fmt_metric(value, fmt: str = ".2f", na: str = "—") -> str:
    if value is None:
        return na
    try:
        return format(value, fmt)
    except (TypeError, ValueError):
        return na


def print_matrix(cells: dict, preset: str) -> None:
    """5 通貨 × 3 期間 のメトリクス (Sharpe, PF, DD, KPI pass) マトリクス."""
    print(f"\n{'=' * 100}")
    print(f"5 通貨 × 3 期間 マトリクス (preset={preset})")
    print(f"{'=' * 100}")

    # Sharpe
    print(f"\n[Sharpe (月次)]")
    print(f"  {'Pair':<10} {'Train':>10} {'Val':>10} {'Test':>10} {'推移所見':<30}")
    for pair in PAIRS:
        line = f"  {pair:<10}"
        sharpes = {}
        for period in PERIODS:
            v = cells.get((pair, period))
            if v is None:
                line += f" {'—':>10}"
            else:
                s = v["metrics"]["sharpe_monthly"]
                sharpes[period] = s
                line += f" {s:>10.3f}"
        line += f"  {interpret_sharpe_trend(sharpes):<30}"
        print(line)

    # PF
    print(f"\n[Profit Factor (月次)]")
    print(f"  {'Pair':<10} {'Train':>10} {'Val':>10} {'Test':>10}")
    for pair in PAIRS:
        line = f"  {pair:<10}"
        for period in PERIODS:
            v = cells.get((pair, period))
            if v is None:
                line += f" {'—':>10}"
            else:
                line += f" {v['metrics']['profit_factor_monthly']:>10.3f}"
        print(line)

    # Max DD monthly
    print(f"\n[Max DD (月次 %)]")
    print(f"  {'Pair':<10} {'Train':>10} {'Val':>10} {'Test':>10}")
    for pair in PAIRS:
        line = f"  {pair:<10}"
        for period in PERIODS:
            v = cells.get((pair, period))
            if v is None:
                line += f" {'—':>10}"
            else:
                line += f" {v['metrics']['max_dd_monthly_pct']:>10.2f}"
        print(line)

    # KPI pass count
    print(f"\n[KPI Pass Count / 9]")
    print(f"  {'Pair':<10} {'Train':>10} {'Val':>10} {'Test':>10}")
    for pair in PAIRS:
        line = f"  {pair:<10}"
        for period in PERIODS:
            v = cells.get((pair, period))
            if v is None:
                line += f" {'—':>10}"
            else:
                line += f" {v['kpi_pass_count']:>10}"
        print(line)

    # Trade count
    print(f"\n[Trade count]")
    print(f"  {'Pair':<10} {'Train':>10} {'Val':>10} {'Test':>10}")
    for pair in PAIRS:
        line = f"  {pair:<10}"
        for period in PERIODS:
            v = cells.get((pair, period))
            if v is None:
                line += f" {'—':>10}"
            else:
                line += f" {v['metrics']['n_trades']:>10}"
        print(line)


def interpret_sharpe_trend(sharpes: dict) -> str:
    """Train→Val→Test の Sharpe 推移から過学習傾向を判定."""
    if len(sharpes) < 2:
        return "-"
    train = sharpes.get("train")
    val = sharpes.get("validation")
    test = sharpes.get("test")
    has_train = train is not None
    has_val = val is not None
    has_test = test is not None

    # Val/Test 両方ある場合
    if has_val and has_test:
        if has_train and train > 0.5 and val < 0 and test < 0:
            return "[NG] 過学習疑い (Train 良→Val/Test 負)"
        if has_train and train > 0.5 and val > 0 and test > 0:
            return "[OK] 安定 (全期間 Sharpe>0)"
        if has_train and val < train * 0.5 and test < train * 0.5:
            return "[WARN] Train 劣化 (半減以下)"
        if not has_train:
            return "[WARN] Train 欠 (Val/Test のみ)"
        return "[WARN] 混合"

    # Val のみ
    if has_val:
        if has_train and train > 0.5 and val < 0:
            return "[NG] Train→Val 崩壊 (要 Test 確認)"
        if has_train and train > 0.5 and val >= 0:
            return "[WARN] Val OK (要 Test 確認)"
        if has_train:
            return "[WARN] Train 弱"
        return "[WARN] Train 欠 (Val のみ)"

    return "-"


def print_kpi_detail(cells: dict) -> None:
    """各セルで通過 / 失敗した KPI を一覧."""
    print(f"\n{'=' * 100}")
    print(f"KPI 通過/失敗 詳細")
    print(f"{'=' * 100}")
    for pair in PAIRS:
        for period in PERIODS:
            v = cells.get((pair, period))
            if v is None:
                continue
            passed = [PERIOD_KEYS_KPI[k] for k, val in v["kpi_pass"].items() if val]
            failed = [PERIOD_KEYS_KPI[k] for k, val in v["kpi_pass"].items() if not val]
            n_pass = v["kpi_pass_count"]
            print(f"\n  [{pair} / {period}] pass={n_pass}/9")
            print(f"    [OK] {', '.join(passed) if passed else '(なし)'}")
            if failed:
                print(f"    [NG] {', '.join(failed)}")


def print_overfit_summary(cells: dict) -> None:
    """過学習判定サマリ (Hasebe-san 向け所見)."""
    print(f"\n{'=' * 100}")
    print(f"過学習判定サマリ (Train→Val→Test シャープ推移)")
    print(f"{'=' * 100}")
    for pair in PAIRS:
        sharpes = {p: cells[(pair, p)]["metrics"]["sharpe_monthly"]
                   for p in PERIODS if (pair, p) in cells}
        if not sharpes:
            continue
        line = f"  {pair:<10} "
        for p in PERIODS:
            v = sharpes.get(p)
            if v is None:
                line += f"{p.upper():>5}={'—':>7}  "
            else:
                line += f"{p.upper():>5}={v:>7.3f}  "
        line += f"  {interpret_sharpe_trend(sharpes)}"
        print(line)


def write_aggregated(cells: dict, preset: str) -> Path:
    """統合 JSON を書き出し."""
    out = RESULT_DIR / f"tvt_{preset}.json"
    matrix = {}
    for pair in PAIRS:
        matrix[pair] = {}
        for period in PERIODS:
            v = cells.get((pair, period))
            if v is not None:
                matrix[pair][period] = {
                    "metrics": v["metrics"],
                    "kpi_pass": v["kpi_pass"],
                    "kpi_pass_count": v["kpi_pass_count"],
                    "elapsed_sec": v["elapsed_sec"],
                }
    payload = {
        "generated_at": datetime.now().isoformat(),
        "preset": preset,
        "n_cells_total": 15,
        "n_cells_completed": len(cells),
        "periods_definition": {
            "train":      "2020-01-01 - 2022-12-31",
            "validation": "2023-01-01 - 2024-06-30",
            "test":       "2024-07-01 - 2025-12-31",
        },
        "matrix": matrix,
        "overfit_summary": {
            pair: interpret_sharpe_trend({p: matrix[pair][p]["metrics"]["sharpe_monthly"]
                                         for p in PERIODS if p in matrix[pair]})
            for pair in PAIRS if any(p in matrix[pair] for p in PERIODS)
        },
        "missing_cells": [
            f"{pair}/{period}"
            for pair in PAIRS for period in PERIODS
            if (pair, period) not in cells
        ],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="TVT 結果集計")
    p.add_argument("--preset", default="A1_A2_combined")
    args = p.parse_args()

    cells = collect(args.preset)
    n_total = len(PAIRS) * len(PERIODS)
    print(f"=== TVT 結果集計 ===")
    print(f"プリセット: {args.preset}")
    print(f"読み込み元: {RESULT_DIR}")
    print(f"完了: {len(cells)}/{n_total} セル")

    if not cells:
        print("[NG] 完了済みセルがありません。")
        return 1

    # 欠損セルの警告
    missing = []
    for pair in PAIRS:
        for period in PERIODS:
            if (pair, period) not in cells:
                missing.append(f"{pair}/{period}")
    if missing:
        print(f"[WARN] 欠損: {', '.join(missing)}")

    print_matrix(cells, args.preset)
    print_kpi_detail(cells)
    print_overfit_summary(cells)

    out = write_aggregated(cells, args.preset)
    print(f"\n[出力]: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
