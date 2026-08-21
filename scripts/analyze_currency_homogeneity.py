"""EXP-FX000005 通貨別均質性の検証(C査読の新規指摘、差し戻し対応).

C査読(`research/EXP-FX000005/20-c-review.md`)が独立集計で新規発見した懸念:
Validation期間でEUR/JPYはn=6件しかなく勝率100%・mean_r=0.748と際立って良いが、
サンプルサイズが極端に小さいことに起因する可能性がある。GBP/JPYはValidation
期間でn=23件・勝率56.5%・mean_r=0.0625とほぼ横ばい。4通貨に一様なエッジが
あるという前提に疑義がある、という指摘への対応。

方法: 現行最良候補(T-01〜T-09+重複トレード生成バグ修正+T-13、trailonly版)の
$1,000バックテスト結果を対象に、
(1) 通貨別の内訳(n・勝率・mean_r_net・sum_r_net)を期間ごとに再集計し、
(2) leave-one-pair-out分析(各通貨を1つずつ除外して残り3通貨でプールした場合の
    mean_r_net・sum_r_netがどう変化するか)で、特定通貨への依存度を定量化する。

注意: leave-one-pair-outは既存トレード明細の事後的な再集計であり、対象通貨を
除いた状態で戦略(価格反応型ショック抑制フィルター等)を再実行したものではない
(フィルターの同時ブレイク判定は対象通貨集合に依存するため、真の意味での
「3通貨構成での再バックテスト」とは異なる近似値である点に留意)。

2026-08-21改訂(T-16再査読の指摘): 初回実行時はT-09+T-06/T-07時点(重複バグ修正前・
T-13前、Train619/Validation143/Test204件)のデータを参照しており、重複バグ修正と
T-13適用後の最終候補データで一度も再実行されていなかった。独立C査読者が最終
データで自ら再集計したところ、報告値より深刻な集中依存と符号反転(GBP/JPYの
Validation・EUR/JPYのTestがともに正→負)が判明したため、参照データをtrailonly版
に差し替えて再実行する。

出力: research/method-notes/currency_homogeneity.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]


def main() -> int:
    print("=== EXP-FX000005 通貨別均質性の検証(C査読差し戻し対応) ===\n")

    with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd_backtest.json").open(
        encoding="utf-8"
    ) as f:
        backtest = json.load(f)

    result: dict = {"generated_at": datetime.now().isoformat(), "periods": {}}

    for period_name in ["train", "validation", "test"]:
        trades = backtest["periods"][period_name]["trades"]
        print(f"--- {period_name} (n={len(trades)}) ---")

        by_pair: dict[str, list[float]] = {p: [] for p in PAIRS}
        for t in trades:
            by_pair.setdefault(t["pair"], []).append(t["r_net"])

        per_pair_stats = {}
        for pair in PAIRS:
            rs = np.array(by_pair.get(pair, []))
            if len(rs) == 0:
                per_pair_stats[pair] = {"n": 0, "win_rate": None, "mean_r_net": None, "sum_r_net": None}
                continue
            per_pair_stats[pair] = {
                "n": int(len(rs)),
                "win_rate": round(float((rs > 0).mean()), 4),
                "mean_r_net": round(float(rs.mean()), 4),
                "sum_r_net": round(float(rs.sum()), 2),
            }
            print(f"  {pair}: n={per_pair_stats[pair]['n']} winrate={per_pair_stats[pair]['win_rate']} "
                  f"mean_r={per_pair_stats[pair]['mean_r_net']} sum_r={per_pair_stats[pair]['sum_r_net']}")

        all_r = np.array([t["r_net"] for t in trades])
        all_stats = {"n": int(len(all_r)), "mean_r_net": round(float(all_r.mean()), 4),
                     "sum_r_net": round(float(all_r.sum()), 2)}
        print(f"  ALL: n={all_stats['n']} mean_r={all_stats['mean_r_net']} sum_r={all_stats['sum_r_net']}")

        leave_one_out = {}
        for excl in PAIRS:
            rest = np.array([t["r_net"] for t in trades if t["pair"] != excl])
            if len(rest) == 0:
                continue
            rest_stats = {
                "n": int(len(rest)),
                "mean_r_net": round(float(rest.mean()), 4),
                "sum_r_net": round(float(rest.sum()), 2),
                "sum_r_net_pct_of_all": round(float(rest.sum()) / all_stats["sum_r_net"] * 100.0, 1)
                if all_stats["sum_r_net"] else None,
            }
            leave_one_out[excl] = rest_stats
            print(f"  excl {excl}: n={rest_stats['n']} mean_r={rest_stats['mean_r_net']} "
                  f"sum_r={rest_stats['sum_r_net']} ({rest_stats['sum_r_net_pct_of_all']}% of ALL)")

        result["periods"][period_name] = {
            "per_pair": per_pair_stats,
            "all": all_stats,
            "leave_one_pair_out": leave_one_out,
        }
        print()

    out_path = ROOT / "research" / "method-notes" / "currency_homogeneity.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
