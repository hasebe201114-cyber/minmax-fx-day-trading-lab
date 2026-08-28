"""EXP-FX000018 診断: 探索診断(pipベース)と正式評価(ダラーベース)の乖離要因の分解.

`00-prescreen.md`の探索診断は pipベース単純合計で PF=1.341・合計+23,734 pips という
良好な数字だったが、正式パイプライン(`backtest_sysfx024_grid_trainonly.py`)では
Trainで明確にマイナスになった。両者は **(a) 評価パイプライン** と **(b) グリッド
パラメータ** の2点が同時に違うため、どちらが効いているのかを切り分ける。

- (a) 評価パイプライン: ダラーベース複利サイジング / MTMエクイティ / 手数料 /
  スワップ(DS-7) / M5足での約定順序解決 / 週末強制決済 / 証拠金ガード
- (b) パラメータ: 探索診断の仮置き値 (N=5, k=1.0, R=30) vs フェーズゲート2の
  データ駆動導出値 (N=3, k=1.72, R=24)

**本スクリプトは診断であり、`00-spec.md` §7 の判定を変更しない。** 凍結パラメータ
での判定結果が Train の公式な結論であり、ここでの数字は改善ループの方向を決める
ための材料としてのみ使う(結果を見てから spec の選定ルールを書き換えない)。

参考として、正式エンジンの結果を「均一ロットのpip合計」に読み替えた値も出力し、
探索診断スクリプトの数字と同じ土俵で比較できるようにする(実装の健全性チェック)。

出力: research/method-notes/sysfx024_param_decomposition.json
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

from evaluate_grid_kpi import evaluate_grid_period  # noqa: E402
from grid_portfolio_engine import pip_size, simulate  # noqa: E402

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"

PARAM_SETS = {
    "P_derived(凍結: N=3,k=1.72,R=24)": {"n_levels": 3, "grid_step_atr_mult": 1.72, "reanchor_bars": 24},
    "P_sanity(探索診断の仮置き: N=5,k=1.0,R=30)": {"n_levels": 5, "grid_step_atr_mult": 1.0, "reanchor_bars": 30},
}


def pip_summary(sim: dict) -> dict:
    """均一ロット前提の net pips 合計 (探索診断スクリプトと同じ土俵に読み替える)."""
    net_pips = []
    for t in sim["trades"]:
        pip = pip_size(t["pair"])
        gross = (t["exit_price"] - t["entry_price"]) if t["side"] == "buy" else (t["entry_price"] - t["exit_price"])
        cost = (t["cost_r"] + t["commission_r"]) * t["initial_risk"]
        net_pips.append((gross - cost) / pip)
    wins = [p for p in net_pips if p > 0]
    losses = [p for p in net_pips if p < 0]
    return {
        "n_trades": len(net_pips),
        "sum_net_pips": round(float(np.sum(net_pips)), 1) if net_pips else 0.0,
        "mean_net_pips": round(float(np.mean(net_pips)), 3) if net_pips else 0.0,
        "win_rate": round(len(wins) / len(net_pips), 4) if net_pips else 0.0,
        "profit_factor_pips": round(sum(wins) / abs(sum(losses)), 3) if losses else None,
    }


def main() -> int:
    print("=== EXP-FX000018 診断: pipベース探索診断 vs ダラーベース正式評価 の乖離要因分解 ===")
    print("※ 本診断は spec §7 の判定を変更しない(凍結パラメータの結果が公式な Train 結論)\n")

    out: dict = {}
    for pname, params in PARAM_SETS.items():
        for cand, carry in (("G0_mark", False), ("G1_carry", True)):
            key = f"{pname} / {cand}"
            sim = simulate(PAIRS, TRAIN_START, TRAIN_END, carry_over=carry, verbose=False, **params)
            res = evaluate_grid_period(cand, sim)
            pips = pip_summary(sim)
            out[key] = {
                "params": params, "carry_over": carry,
                "dollar_based": {
                    "n_trades": res["n_trades"], "win_rate": res["win_rate"],
                    "final_balance_usd": res["final_balance_usd"],
                    "total_return_pct": res["total_return_pct"],
                    "profit_factor": res["profit_factor"],
                    "monthly_sharpe": res["monthly_sharpe"],
                    "max_dd_pct_mtm": res["max_dd_pct"],
                    "perm_p_week": res["permutation_p_week_block"],
                    "kpi_required": res["kpi_required_pass_count"],
                    "outcome_breakdown": res["outcome_breakdown"],
                    "swap_total_usd": res["swap_total_usd"],
                },
                "uniform_lot_pip_view": pips,
            }
            print(f"--- {key} ---")
            print(f"  [ダラーベース] n={res['n_trades']}  勝率={res['win_rate']}  "
                  f"最終=${res['final_balance_usd']}({res['total_return_pct']:+.1f}%)  PF={res['profit_factor']}  "
                  f"必須KPI={res['kpi_required_pass_count']}  perm_p(週)={res['permutation_p_week_block']}")
            print(f"  [均一ロットpip換算] n={pips['n_trades']}  合計={pips['sum_net_pips']:+,.0f}pips  "
                  f"平均={pips['mean_net_pips']:+.2f}pips  勝率={pips['win_rate']}  PF={pips['profit_factor_pips']}")
            print(f"  outcome: " + "  ".join(
                f"{k}:{v['n']}({v['share']:.0%})/${v['mean_usd']:+.2f}" for k, v in res["outcome_breakdown"].items()))
            print()

    ref = {
        "prescreen_sanity_check": {
            "source": "research/method-notes/grid_strategy_sanity_check.json "
                      "(scripts/explore_grid_strategy_sanity_check.py、正式プロトコル外)",
            "n_trades": 2592, "win_rate": 0.746, "profit_factor": 1.341, "sum_net_pips": 23734,
            "_diff_from_formal_engine": [
                "H4足の高安のみで判定(同一バー内の順序未解決)。正式エンジンはM5足で順序解決",
                "手数料(0.00004)未計上。正式エンジンは計上",
                "スワップ(DS-7)未計上。正式エンジンは水曜3倍込みで正味計上",
                "週末強制決済なし(本PJ共通ルール違反)。正式エンジンは週内最終H4バーで全決済",
                "証拠金ガードなし。正式エンジンは合算方式30%でガード",
                "ロット均一・複利なし。正式エンジンは残高1%/グリッド面の複利サイジング",
            ],
        }
    }
    out["_reference"] = ref
    out["_generated_at"] = datetime.now().isoformat()

    path = ROOT / "research" / "method-notes" / "sysfx024_param_decomposition.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[出力]: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
