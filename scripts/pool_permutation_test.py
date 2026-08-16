"""指定プリセットの train_val_test 結果 (5通貨) から trade_pnls をプールし、
permutation test を実行する (spec の選定基準 n_trades>=66 かつ perm_p<0.05 判定用)。

Usage:
    python scripts/pool_permutation_test.py --preset D1_DataDrivenDonchianRR --period train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from minmax_fx_dt.backtest.permutation import permutation_test

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True)
    parser.add_argument("--period", default="train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result_dir = ROOT / "research" / "EXP-FX000001" / "10-result" / "train_val_test"
    all_pnls: list[float] = []
    for pair in PAIRS:
        path = result_dir / f"tvt_{args.preset}_{pair}_{args.period}.json"
        with path.open(encoding="utf-8") as f:
            d = json.load(f)
        all_pnls.extend(d["trade_pnls"])

    print(f"プリセット: {args.preset}  期間: {args.period}")
    print(f"pooled n_trades = {len(all_pnls)}")
    result = permutation_test(all_pnls, seed=args.seed)
    print(f"observed mean pnl = {result.observed_statistic:.2f}")
    print(f"perm_p (one-sided) = {result.p_value:.4f}")
    print(f"perm_p (two-sided) = {result.p_value_two_sided:.4f}")

    selection_pass = len(all_pnls) >= 66 and result.p_value < 0.05
    print(f"\n選定基準 (n>=66 かつ perm_p<0.05): {'PASS' if selection_pass else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
