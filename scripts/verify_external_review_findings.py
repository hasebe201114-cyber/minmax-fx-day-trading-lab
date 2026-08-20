"""外部レビュー(2026-08-20, EXP-FX000005)の指摘を独立に再現・検算するスクリプト.

出典: `obs/minmax_fx_day_trading_lab/85外部レビュー/2026-08-20_EXP-FX000005_External_Review/00_REVIEW_SUMMARY.md`

本スクリプトは **DS-1 (data/curated/ds-1.json) を必要としない**。
コミット済みの結果 JSON (`research/method-notes/*.json`) とプロジェクトの関数だけで、
レビューが主張した以下 5 点を再現する:

    F1: 週末強制クローズが一度も発動していない
    F2: permutation_p < 0.05 は 4 通貨構成では原理的に達成不能 (p 値の下限 ≈ 0.3158)
    F3: ATR トレーリングが事実上一度も作動していない
    K3m: 最大連続損失 ≤ 5 はスケール不変でない (i.i.d. でも約 6 割しか通らない)
    n_eff: min_n_trades=300 に必要な名目トレード数 (4 通貨で 1,027 件)

使い方:
    python scripts/verify_external_review_findings.py
    python scripts/verify_external_review_findings.py --json   # 機械可読出力

判定はすべて「レビューの主張が再現できたか (REPRODUCED / NOT_REPRODUCED)」として出力する。
数値そのものを結論とはせず、C 査読チームが独立に確認するための材料として使うこと。
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minmax_fx_dt.backtest.permutation import (  # noqa: E402
    effective_pair_count,
    permutation_test_clustered,
)

# レビュー時点で「現時点の最良候補」とされていた改善ループ第6試行の結果
BACKTEST_JSON = ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_v6_1000usd_backtest.json"

PAIRS_4 = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
PERIODS = ["train", "validation", "test"]

# KPI 閾値 (decision/criteria.py と同値、参照用に再掲)
MIN_N_TRADES_EFFECTIVE = 300
MAX_CONSECUTIVE_LOSSES = 5
PERMUTATION_ALPHA = 0.05


def load_trades() -> dict[str, list[dict]]:
    if not BACKTEST_JSON.exists():
        raise SystemExit(f"結果 JSON が見つかりません: {BACKTEST_JSON}")
    data = json.loads(BACKTEST_JSON.read_text(encoding="utf-8"))
    return {p: data["periods"][p]["trades"] for p in PERIODS}


# --------------------------------------------------------------------------
# F1: 週末強制クローズが一度も発動していない
# --------------------------------------------------------------------------
def check_f1_weekend_close(trades_by_period: dict[str, list[dict]]) -> dict:
    """土曜 06:00 JST を跨いで保有されたトレードを数え、WEEKEND_* 決済の有無と突き合わせる.

    PJ 共通ルール (週末持ち越し禁止) と `01-trade-scenario-definition.md` §2/§4.7 が
    実装で保証されているなら、跨ぎ件数は 0 か、少なくとも WEEKEND_* 決済が存在するはず。
    """
    per_period = {}
    for period, trades in trades_by_period.items():
        spanning = 0
        for t in trades:
            entry = dt.datetime.fromisoformat(t["entry_time"])
            exit_ = dt.datetime.fromisoformat(t["exit_time"])
            cursor = entry
            while cursor < exit_:
                if cursor.weekday() == 5 and cursor.hour >= 6:  # 土曜 06:00 JST 以降
                    spanning += 1
                    break
                cursor += dt.timedelta(hours=1)
        reasons = collections.Counter(t["exit_reason"] for t in trades)
        weekend_exits = reasons.get("WEEKEND_NO_TP", 0) + reasons.get("TP_THEN_WEEKEND", 0)
        per_period[period] = {
            "n_trades": len(trades),
            "n_spanning_weekend": spanning,
            "pct_spanning_weekend": round(100 * spanning / len(trades), 1) if trades else 0.0,
            "n_weekend_exits": weekend_exits,
            "exit_reasons": dict(reasons),
        }

    total_spanning = sum(v["n_spanning_weekend"] for v in per_period.values())
    total_weekend_exits = sum(v["n_weekend_exits"] for v in per_period.values())
    total_trades = sum(v["n_trades"] for v in per_period.values())
    reproduced = total_spanning > 0 and total_weekend_exits == 0

    return {
        "finding": "F1",
        "claim": "週末強制クローズが一度も発動しておらず、週末跨ぎのトレードが存在する",
        "per_period": per_period,
        "total_trades": total_trades,
        "total_spanning_weekend": total_spanning,
        "total_weekend_exits": total_weekend_exits,
        "verdict": "REPRODUCED" if reproduced else "NOT_REPRODUCED",
    }


# --------------------------------------------------------------------------
# F2: permutation 検定の p 値には下限があり、n に依存しない
# --------------------------------------------------------------------------
def check_f2_permutation_floor(n_permutations: int = 20000, seed: int = 1) -> dict:
    """「全トレード勝ち」= 理論上ありえない最強のエッジを入れても p が 0.05 を切らないことを示す.

    permutation_test_clustered() は通貨ペアごとに符号を 1 個引いて全トレードへ一括適用するため、
    検定の実効標本サイズはトレード数ではなく通貨ペア数になる。
    """
    cases = []
    for n_total in (100, 300, 1000, 4000, 20000):
        per_pair = n_total // len(PAIRS_4)
        pnls: list[float] = []
        pairs: list[str] = []
        for pair in PAIRS_4:
            pnls.extend([1.0] * per_pair)  # 全勝 (mean_R = +1.0)
            pairs.extend([pair] * per_pair)
        res = permutation_test_clustered(pnls, pairs, n_permutations=n_permutations, seed=seed)
        cases.append({
            "n_pairs": len(PAIRS_4),
            "n_trades": len(pnls),
            "observed_mean": round(res.observed_statistic, 4),
            "p_value": round(res.p_value, 4),
            "significant_at_0.05": res.p_value < PERMUTATION_ALPHA,
        })

    # 参考: 単一通貨なら独立符号 flip になり、正常に有意になる
    single = permutation_test_clustered(
        [1.0] * 100, ["USD_JPY"] * 100, n_permutations=n_permutations, seed=seed
    )

    p_floor = min(c["p_value"] for c in cases)
    reproduced = all(not c["significant_at_0.05"] for c in cases)

    return {
        "finding": "F2",
        "claim": "4通貨プールでは全勝ケースでも permutation_p < 0.05 に到達しない (p の下限が存在し n に依存しない)",
        "cases_all_wins_4pairs": cases,
        "p_value_floor_observed": p_floor,
        "reference_single_pair_all_wins_n100_p": round(single.p_value, 6),
        "verdict": "REPRODUCED" if reproduced else "NOT_REPRODUCED",
    }


# --------------------------------------------------------------------------
# F3: ATR トレーリングが作動していない
# --------------------------------------------------------------------------
def check_f3_trailing_never_binds(trades_by_period: dict[str, list[dict]]) -> dict:
    """TP_THEN_SL_TRAIL の r_gross が「建値ストップの値ちょうど」に集中しているかを確認する.

    TP_LEVELS = [(1.0, 0.40), (2.0, 0.35), (4.0, 0.25)] のとき、建値ストップで決済されると
    r_gross は TP1 のみ到達で +0.40、TP1+TP2 到達で +1.10 に厳密に一致する。
    ATR トレーリングが実際に噛んでいれば、これらの値から外れた r が広く分布するはず。
    """
    all_trades = [t for trades in trades_by_period.values() for t in trades]
    trail = [t for t in all_trades if t["exit_reason"] == "TP_THEN_SL_TRAIL"]
    if not trail:
        return {"finding": "F3", "verdict": "NOT_REPRODUCED", "reason": "TP_THEN_SL_TRAIL が 0 件"}

    r_values = [t["r_gross"] for t in trail]
    breakeven_tp1 = sum(1 for r in r_values if abs(r - 0.40) < 1e-3)
    breakeven_tp2 = sum(1 for r in r_values if abs(r - 1.10) < 1e-3)
    at_breakeven_pct = 100 * (breakeven_tp1 + breakeven_tp2) / len(trail)

    by_reason = collections.defaultdict(list)
    for t in all_trades:
        by_reason[t["exit_reason"]].append(t["r_gross"])
    reason_stats = {
        k: {
            "n": len(v),
            "median_r": round(statistics.median(v), 3),
            "mean_r": round(statistics.mean(v), 3),
            "min_r": round(min(v), 3),
            "max_r": round(max(v), 3),
        }
        for k, v in sorted(by_reason.items())
    }

    return {
        "finding": "F3",
        "claim": "ATR トレーリングは建値ストップより有利な位置に来ず、実質的に作動していない",
        "n_tp_then_sl_trail": len(trail),
        "pct_exactly_at_breakeven": round(at_breakeven_pct, 1),
        "n_at_breakeven_after_tp1": breakeven_tp1,
        "n_at_breakeven_after_tp2": breakeven_tp2,
        "max_r_gross_in_trail_exits": round(max(r_values), 3),
        "exit_reason_r_stats": reason_stats,
        "verdict": "REPRODUCED" if at_breakeven_pct >= 80.0 else "NOT_REPRODUCED",
    }


# --------------------------------------------------------------------------
# K3m: 最大連続損失 ≤ 5 はスケール不変でない
# --------------------------------------------------------------------------
def _max_loss_run(losses: np.ndarray) -> int:
    best = cur = 0
    for is_loss in losses:
        cur = cur + 1 if is_loss else 0
        best = max(best, cur)
    return best


def check_k3m_scale_dependence(trades_by_period: dict[str, list[dict]], reps: int = 3000,
                               seed: int = 0) -> dict:
    """観測された n と勝率を持つ i.i.d. 系列で最大連敗をシミュレートし、K3m の情報量を測る.

    「エッジが本物でも K3m ≤ 5 は約 6 割でしか通らない」= この基準はほぼコイン投げ、という主張。
    """
    rng = np.random.default_rng(seed)
    per_period = {}
    for period, trades in trades_by_period.items():
        n = len(trades)
        wins = sum(1 for t in trades if t["r_net"] > 0)
        win_rate = wins / n if n else 0.0
        observed = _max_loss_run(np.array([t["r_net"] <= 0 for t in trades]))
        runs = np.array([_max_loss_run(rng.random(n) >= win_rate) for _ in range(reps)])
        per_period[period] = {
            "n_trades": n,
            "win_rate": round(win_rate, 4),
            "observed_max_consecutive_losses": int(observed),
            "iid_null_mean": round(float(runs.mean()), 2),
            "iid_null_median": int(np.median(runs)),
            "prob_pass_threshold_under_iid": round(float((runs <= MAX_CONSECUTIVE_LOSSES).mean()), 3),
            "observed_percentile_in_null": round(float((runs < observed).mean()), 3),
        }

    probs = [v["prob_pass_threshold_under_iid"] for v in per_period.values()]
    reproduced = all(0.3 <= p <= 0.8 for p in probs)

    return {
        "finding": "K3m",
        "claim": f"最大連続損失 ≤ {MAX_CONSECUTIVE_LOSSES} は n 依存で、i.i.d. でも通過率が 5〜7 割程度にとどまる",
        "threshold": MAX_CONSECUTIVE_LOSSES,
        "per_period": per_period,
        "note": "observed_percentile_in_null が 0.5 前後なら、観測値はランダムと区別がつかない",
        "verdict": "REPRODUCED" if reproduced else "NOT_REPRODUCED",
    }


# --------------------------------------------------------------------------
# n_eff: min_n_trades=300 に必要な名目トレード数
# --------------------------------------------------------------------------
def check_n_eff_requirement(trades_by_period: dict[str, list[dict]]) -> dict:
    """実効トレード数の縮小係数と、min_n_trades を満たすのに必要な名目件数を算出する."""
    eff_pairs = effective_pair_count(PAIRS_4)
    shrink = eff_pairs / len(PAIRS_4)
    required_nominal = MIN_N_TRADES_EFFECTIVE / shrink

    per_period = {}
    for period, trades in trades_by_period.items():
        n = len(trades)
        per_period[period] = {
            "n_trades_nominal": n,
            "n_trades_effective": round(n * shrink, 1),
            "shortfall_nominal_trades": max(0, round(required_nominal - n)),
        }

    return {
        "finding": "n_eff",
        "claim": "min_n_trades=300 は 4 通貨構成では名目 1,000 件超を要求し、戦略コンセプト(週1回の厳選)と両立しない",
        "effective_pair_count": round(eff_pairs, 4),
        "shrink_factor": round(shrink, 4),
        "required_nominal_trades_for_300_effective": round(required_nominal),
        "per_period": per_period,
        "verdict": "REPRODUCED" if required_nominal > 1000 else "NOT_REPRODUCED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="機械可読な JSON で出力する")
    parser.add_argument("--out", type=Path, default=None, help="結果 JSON の保存先")
    args = parser.parse_args()

    trades_by_period = load_trades()
    results = [
        check_f1_weekend_close(trades_by_period),
        check_f2_permutation_floor(),
        check_f3_trailing_never_binds(trades_by_period),
        check_k3m_scale_dependence(trades_by_period),
        check_n_eff_requirement(trades_by_period),
    ]
    payload = {
        "generated_at": dt.datetime.now().isoformat(),
        "source_backtest": str(BACKTEST_JSON.relative_to(ROOT)),
        "review_document": "obs/minmax_fx_day_trading_lab/85外部レビュー/2026-08-20_EXP-FX000005_External_Review/00_REVIEW_SUMMARY.md",
        "results": results,
    }

    if args.out:
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"保存: {args.out}")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("=" * 78)
    print("外部レビュー(2026-08-20 / EXP-FX000005)の指摘 再現チェック")
    print(f"対象: {BACKTEST_JSON.relative_to(ROOT)}")
    print("=" * 78)

    f1 = results[0]
    print(f"\n[F1] {f1['claim']}  → {f1['verdict']}")
    for period, v in f1["per_period"].items():
        print(f"  {period:11s} n={v['n_trades']:4d}  週末跨ぎ={v['n_spanning_weekend']:3d} "
              f"({v['pct_spanning_weekend']:4.1f}%)  WEEKEND_*決済={v['n_weekend_exits']}")
    print(f"  合計: {f1['total_spanning_weekend']}/{f1['total_trades']} 件が週末跨ぎ、"
          f"WEEKEND_* 決済は {f1['total_weekend_exits']} 件")

    f2 = results[1]
    print(f"\n[F2] {f2['claim']}  → {f2['verdict']}")
    print("  4通貨・全トレード勝ち(理論上最強のエッジ)を入れた場合の p 値:")
    for c in f2["cases_all_wins_4pairs"]:
        print(f"    n={c['n_trades']:6d}  observed_mean={c['observed_mean']:+.4f}  p={c['p_value']:.4f}")
    print(f"  観測された p 値の下限 = {f2['p_value_floor_observed']:.4f}  (閾値 {PERMUTATION_ALPHA})")
    print(f"  参考: 単一通貨・全勝 n=100 なら p={f2['reference_single_pair_all_wins_n100_p']:.6f}")

    f3 = results[2]
    print(f"\n[F3] {f3['claim']}  → {f3['verdict']}")
    print(f"  TP_THEN_SL_TRAIL {f3['n_tp_then_sl_trail']} 件のうち "
          f"{f3['pct_exactly_at_breakeven']}% が建値ストップの値ちょうど "
          f"(+0.40: {f3['n_at_breakeven_after_tp1']} 件 / +1.10: {f3['n_at_breakeven_after_tp2']} 件)")
    print(f"  トレール決済の r_gross 最大値 = {f3['max_r_gross_in_trail_exits']}")
    print("  決済理由別 r_gross:")
    for reason, s in f3["exit_reason_r_stats"].items():
        print(f"    {reason:24s} n={s['n']:4d} 中央値={s['median_r']:+.3f} "
              f"平均={s['mean_r']:+.3f} min={s['min_r']:+.2f} max={s['max_r']:+.2f}")

    k3 = results[3]
    print(f"\n[K3m] {k3['claim']}  → {k3['verdict']}")
    for period, v in k3["per_period"].items():
        print(f"  {period:11s} n={v['n_trades']:4d} 勝率={v['win_rate']:.1%}  "
              f"観測={v['observed_max_consecutive_losses']}  i.i.d.期待値={v['iid_null_mean']:.2f}  "
              f"「≤{MAX_CONSECUTIVE_LOSSES}」通過率={v['prob_pass_threshold_under_iid']:.1%}  "
              f"観測のパーセンタイル={v['observed_percentile_in_null']:.2f}")

    ne = results[4]
    print(f"\n[n_eff] {ne['claim']}  → {ne['verdict']}")
    print(f"  実効通貨ペア数={ne['effective_pair_count']}  縮小係数={ne['shrink_factor']}  "
          f"→ n_eff={MIN_N_TRADES_EFFECTIVE} に必要な名目件数="
          f"{ne['required_nominal_trades_for_300_effective']}")
    for period, v in ne["per_period"].items():
        print(f"  {period:11s} 名目={v['n_trades_nominal']:4d}  実効={v['n_trades_effective']:6.1f}  "
              f"不足={v['shortfall_nominal_trades']} 件")

    print("\n" + "=" * 78)
    verdicts = collections.Counter(r["verdict"] for r in results)
    print(f"再現結果: {dict(verdicts)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
