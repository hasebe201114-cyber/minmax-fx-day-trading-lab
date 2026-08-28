"""EXP-FX000018 amendment-01 §3 W2: 週明け窓開け(ギャップ)リスクのデータ駆動導出.

司令塔判断「週末持ち越しを緩和しても良い。リスク対策は必要」を受け、週末を跨いで
持ち越すポジション量の上限を決めるための統計量を Train 期間から導出する。

**損益・勝率等の成績指標は一切参照しない**(フェーズゲート2 と同じ方針)。測るのは
「週内最終M5バーの終値 → 週明け最初のM5バーの始値」の相対窓開けの分布だけである。

導出する値:
  rel_gap_p99[pair] = 通貨別の相対窓開け |open_mon - close_fri| / close_fri の99パーセンタイル

用途(amendment-01 §3 W2、結果を見る前に固定済み):
  est_gap_loss = Σ_pairs |net_notional_usd[pair]| * rel_gap_p99[pair]
  est_gap_loss <= 0.10 * equity_MTM  を満たすまで含み損の大きいポジションから決済する

出力: research/EXP-FX000018/10-result/weekend_gap_risk.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from grid_portfolio_engine import load_m5, pip_size  # noqa: E402

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
GAP_PERCENTILE = 99  # amendment-01 §3 W2 で事前登録
WEEKEND_GAP_LOSS_BUDGET_PCT = 10.0  # 同上 (K2m 月間最大DD上限と同水準)


def main() -> int:
    print("=== EXP-FX000018 amendment-01: 週明け窓開けリスクの導出 (Train期間のみ) ===")
    print(f"対象: {PAIRS}  期間: {TRAIN_START} 〜 {TRAIN_END}")
    print("※ 損益・勝率等の成績指標は一切参照しない\n")

    per_pair: dict[str, dict] = {}
    for pair in PAIRS:
        m5 = load_m5(pair, TRAIN_START, TRAIN_END)
        iso = m5.index.isocalendar()
        week_key = iso["year"].to_numpy() * 100 + iso["week"].to_numpy()
        # 週内最終バー = 次バーとISO週番号が変わるバー (engine の is_week_last と同一定義)
        last_idx = np.flatnonzero(week_key[1:] != week_key[:-1])
        closes = m5["close"].to_numpy(dtype=float)
        opens = m5["open"].to_numpy(dtype=float)
        fri_close = closes[last_idx]
        mon_open = opens[last_idx + 1]
        rel_gap = np.abs(mon_open - fri_close) / fri_close
        signed_gap = (mon_open - fri_close) / fri_close
        pip = pip_size(pair)
        gap_pips = np.abs(mon_open - fri_close) / pip
        p99 = float(np.percentile(rel_gap, GAP_PERCENTILE))
        per_pair[pair] = {
            "n_weekends": int(len(rel_gap)),
            "rel_gap_p50": round(float(np.percentile(rel_gap, 50)), 6),
            "rel_gap_p95": round(float(np.percentile(rel_gap, 95)), 6),
            f"rel_gap_p{GAP_PERCENTILE}": round(p99, 6),
            "rel_gap_max": round(float(rel_gap.max()), 6),
            "gap_pips_p50": round(float(np.percentile(gap_pips, 50)), 2),
            f"gap_pips_p{GAP_PERCENTILE}": round(float(np.percentile(gap_pips, GAP_PERCENTILE)), 2),
            "gap_pips_max": round(float(gap_pips.max()), 2),
            "signed_gap_mean": round(float(signed_gap.mean()), 6),
            "share_up_gaps": round(float((signed_gap > 0).mean()), 4),
        }
        d = per_pair[pair]
        print(f"  [{pair}] 週数={d['n_weekends']}  窓開け中央値={d['gap_pips_p50']}pips  "
              f"p{GAP_PERCENTILE}={d[f'gap_pips_p{GAP_PERCENTILE}']}pips  最大={d['gap_pips_max']}pips  "
              f"(rel p{GAP_PERCENTILE}={p99:.5f})  上窓の割合={d['share_up_gaps']}")

    rel_gap_p99 = {p: per_pair[p][f"rel_gap_p{GAP_PERCENTILE}"] for p in PAIRS}
    print(f"\n  → 週末ネットエクスポージャー上限の判定式 (amendment-01 §3 W2):")
    print(f"     est_gap_loss = Σ |net_notional_usd[pair]| × rel_gap_p{GAP_PERCENTILE}[pair]")
    print(f"     est_gap_loss ≤ {WEEKEND_GAP_LOSS_BUDGET_PCT}% × equity_MTM を満たすまで含み損の大きい順に決済")
    # 参考: 4通貨に均等配分した場合、equity の何倍まで持ち越せるか
    mean_p99 = float(np.mean(list(rel_gap_p99.values())))
    print(f"     参考: 4通貨均等配分なら合計ネット想定元本の上限 ≈ equity × "
          f"{WEEKEND_GAP_LOSS_BUDGET_PCT / 100 / mean_p99:.1f}倍 "
          f"(証拠金換算 {WEEKEND_GAP_LOSS_BUDGET_PCT / 100 / mean_p99 * 4:.1f}%)")

    result = {
        "generated_at": datetime.now().isoformat(),
        "exp_id": "EXP-FX000018", "sys_id": "SYS-FX024",
        "spec_ref": "research/EXP-FX000018/00-spec-amendment-01.md §3 W2",
        "period": {"train_start": TRAIN_START, "train_end": TRAIN_END},
        "pairs": PAIRS,
        "method": {
            "definition": "週内最終M5バーの終値 → 週明け最初のM5バーの始値の相対変化率 "
                          "|open_mon - close_fri| / close_fri",
            "percentile": GAP_PERCENTILE,
            "aggregation": "通貨別に算出。ポートフォリオ合計は分散効果を認めず単純合算(保守側)。"
                           "同一通貨の両建て分は窓開けに対し物理的に相殺するためネット想定元本で評価",
            "budget_pct": WEEKEND_GAP_LOSS_BUDGET_PCT,
            "pnl_independence": "損益・勝率等の成績指標は本導出で一切参照していない",
        },
        "derived": {"rel_gap_p99": rel_gap_p99, "weekend_gap_loss_budget_pct": WEEKEND_GAP_LOSS_BUDGET_PCT},
        "per_pair": per_pair,
    }
    out = ROOT / "research" / "EXP-FX000018" / "10-result" / "weekend_gap_risk.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[出力]: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
