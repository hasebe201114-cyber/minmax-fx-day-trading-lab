"""EXP-FX000005: ダウ理論連続押し目買い版(確定コストモデル)の正式KPI判定.

`backtest_vol_breakout_dow_theory_1000usd.py`の出力(1通貨1ポジション制約+
H1継続確認再開ロジック、エントリー=成行/SL・トレーリング・TP=指値/週末・
MAX_HOLD=成行という確定済みコストモデル)を対象に、`00-spec.md`のK1m〜K7m閾値
+実効サンプル数+permutation有意性の全ゲートで正式判定する。

## 修正 (2026-08-20、司令塔指摘): DDをピーク比で評価するよう変更

司令塔「複利により増えた資産の中でリスク比率からトレードするのでDDの額が
増えるのは問題なく、元本に対してDD比率を評価するのはおかしくないですか」
との指摘を受けて調査した結果、その通りと判明した。本サイジング方式は
「その時点の残高の1%」というリスクベースであり、実際のリスク管理上意味を
持つのは「直近ピークからどれだけ削られたか」(ピーク比DD)である。本PJ標準の
`backtest.metrics.max_drawdown()`/`monthly_max_dd_pct()`は**初期資金比**で
計算するため、複利で大きく資産が増えた後の変動を実態より大きく見せてしまう
(Trainの実測: 金額として最大のDDはピーク$3,836→$2,929の23.7%だが、初期資金
$1,000比では90.7%という、ほぼ全損に近い誤った印象になっていた)。この問題は
SYS-FX007〜010がリターン数%〜十数%程度に留まっていたため表面化しなかった
だけで、SYS-FX011のような大幅な複利成長を伴う戦略で初めて露呈した。

本スクリプトはピーク比DD(`peak_relative_max_dd_pct`/
`peak_relative_monthly_max_dd_pct`)を正式判定に使用し、初期資金比の数値は
参考として並記する。

2026-08-21(T-10対応): 上記2関数は元は本スクリプトにローカル実装されていたが、
「固定ロット=初期資金比DD、複利サイジング=ピーク比DD」という全SYS共通の
恒久ルール(`PJ000004-基本データ層と検証プロセス定義.md`「Q9」)として
`minmax_fx_dt.backtest.metrics`へ昇格した。本スクリプトはそちらをimportする。

出力: research/method-notes/vol_breakout_dow_theory_kpi_evaluation.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd

from minmax_fx_dt.backtest.metrics import (
    max_drawdown, monthly_max_dd_pct, monthly_sharpe, payoff_ratio,
    peak_relative_max_dd_pct, peak_relative_monthly_max_dd_pct, profit_factor,
)
from minmax_fx_dt.decision.criteria import compute_k3m_scale_invariant, compute_n_trades_effective

# 00-spec.md のKPI閾値
KPI_THRESHOLDS = {
    "monthly_sharpe": 0.4,
    "profit_factor": 1.2,
    "monthly_expectancy_positive": True,
    "max_dd_monthly_pct": 10.0,
    "max_dd_yearly_pct": 20.0,
    "max_consecutive_losses": 5,
    "payoff_ratio": 1.5,
    "spread_cost_multiplier": 3.0,
    "min_n_trades_effective": 300,
    "permutation_p_value": 0.05,
}
INITIAL_CAPITAL_USD = 1000.0

# KPI必須/参考ゲート区分 (2026-08-21 T-11対応、外部レビュー指摘)。
# 「5/10」「7/10」という単純な達成数の羅列は、統計的検出力の乏しい指標
# (K3m)を、Sharpe・PF・permutation p値のような比較的頑健な指標と同列に
# 1票として扱っており、採否判断の透明性を損なっていた。
#
# K3m(最大連続損失)を「参考」に分類する根拠: 外部レビュー(2026-08-20)・
# T-16再査読が指摘した通り、i.i.d.帰無仮説下でも観測される最大連敗が
# 通過率の変動要因になりやすく(検出力が乏しい)、T-08でスケール不変な
# パーセンタイル判定へ再定義した後もなお「ランダムと区別がつかない」
# ケースが多い(観測パーセンタイルが帰無分布の中央付近に来ることが多い)。
# 最大連敗自体はリスク管理上の実務的な意味を持つため参考指標としては
# 有用だが、単独でGO/REJECTを左右する必須ゲートとしては扱わない。
# 他の9項目(K1m・K2m・K4m・K5m・実効n・permutation)は、いずれかが単独で
# 不合格になれば戦略の構造的な欠陥を示す可能性が高いため「必須」とする。
KPI_GATE_TIER: dict[str, str] = {
    "monthly_sharpe": "必須",
    "profit_factor": "必須",
    "monthly_expectancy_positive": "必須",
    "max_dd_monthly_pct": "必須",
    "max_dd_yearly_pct": "必須",
    "max_consecutive_losses": "参考",
    "payoff_ratio": "必須",
    "spread_cost_multiplier": "必須",
    "min_n_trades_effective": "必須",
    "permutation_p_value": "必須",
}


def max_consecutive_losses(trades: list[dict]) -> int:
    trades_sorted = sorted(trades, key=lambda t: t["entry_time"])
    worst = cur = 0
    for t in trades_sorted:
        if t["dollar_pnl"] < 0:
            cur += 1
            worst = max(worst, cur)
        else:
            cur = 0
    return worst


def evaluate_period(
    period_name: str,
    p: dict,
    *,
    perm_p_field: str = "perm_p_clustered",
    apply_n_correlation_discount: bool = True,
    apply_k3m_scale_invariant: bool = False,
) -> dict:
    """perm_p_field: 参照するpermutation p値のフィールド名。既定は旧
    `permutation_test_clustered()`由来の"perm_p_clustered"(後方互換)。外部レビュー
    T-06対応でブロック順列(`permutation_test_block()`)を使う場合は
    "perm_p_block"を指定する。

    apply_n_correlation_discount: 実効トレード数(n_eff)算出時に通貨間相関による
    割引を適用するか。既定True(後方互換)。T-07対応: ブロック順列(T-06)を使う
    場合、依存構造は検定側で直接捕捉されるため、Falseを指定してnの相関割引と
    検定側の相関補正の二重計上を避けること。

    apply_k3m_scale_invariant: K3m(最大連続損失)を絶対件数閾値(旧来、既定False
    で後方互換維持)ではなく、`decision.criteria.compute_k3m_scale_invariant()`
    によるi.i.d.帰無分布パーセンタイル判定に切り替えるか(2026-08-21 T-08対応、
    外部レビュー指摘・T-16再査読で優先順位トラッキングから漏れていたことが判明)。
    過去の`_t03fix`等の中間結果スクリプトとの数値再現性を保つため既定はFalseとし、
    最新のtrailonly版評価でのみTrueを指定する。"""
    eq_curve = pd.DataFrame(p["equity_curve"])
    eq_curve["timestamp"] = pd.to_datetime(eq_curve["time"], format="mixed", utc=True).dt.tz_localize(None)
    eq_curve["equity"] = eq_curve["balance"]

    m_sharpe = monthly_sharpe(eq_curve)
    _dd_usd, dd_pct_of_initial = max_drawdown(eq_curve)
    dd_monthly_pct_of_initial = monthly_max_dd_pct(eq_curve, INITIAL_CAPITAL_USD)
    dd_pct = peak_relative_max_dd_pct(eq_curve)
    dd_monthly_pct = peak_relative_monthly_max_dd_pct(eq_curve)

    dollar_pnls = [t["dollar_pnl"] for t in p["trades"]]
    pf = profit_factor(dollar_pnls)
    payoff = payoff_ratio(dollar_pnls)
    max_losses = max_consecutive_losses(p["trades"])

    n_trades_total = len(p["trades"])
    win_rate = (sum(1 for pnl in dollar_pnls if pnl > 0) / n_trades_total) if n_trades_total else 0.0
    if apply_k3m_scale_invariant and n_trades_total > 0:
        k3m_result = compute_k3m_scale_invariant(n_trades_total, win_rate, max_losses)
        max_consecutive_losses_pass = k3m_result["pass_"]
    else:
        k3m_result = None
        max_consecutive_losses_pass = max_losses <= KPI_THRESHOLDS["max_consecutive_losses"]

    # 月次期待値: 月ごとのドルP&L平均が正か
    eq_curve_i = eq_curve.set_index("timestamp")
    monthly_pnl = eq_curve_i["equity"].resample("ME").last().diff().dropna()
    monthly_expectancy_positive = bool(monthly_pnl.mean() > 0) if len(monthly_pnl) > 0 else False

    # K5m: スプレッドコスト倍率 (平均粗エッジ ÷ 平均コスト)
    mean_r_gross = sum(t["r_gross"] for t in p["trades"]) / len(p["trades"])
    mean_cost = sum(t["cost_r"] + t["commission_r"] for t in p["trades"]) / len(p["trades"])
    spread_cost_multiplier = mean_r_gross / mean_cost if mean_cost > 0 else None

    trades_per_currency: dict[str, int] = {}
    for t in p["trades"]:
        trades_per_currency[t["pair"]] = trades_per_currency.get(t["pair"], 0) + 1
    n_eff = compute_n_trades_effective(
        trades_per_currency, len(p["trades"]), apply_correlation_discount=apply_n_correlation_discount
    )

    perm_p = p[perm_p_field]

    kpi_pass = {
        "monthly_sharpe": m_sharpe >= KPI_THRESHOLDS["monthly_sharpe"],
        "profit_factor": pf >= KPI_THRESHOLDS["profit_factor"],
        "monthly_expectancy_positive": monthly_expectancy_positive,
        "max_dd_monthly_pct": dd_monthly_pct <= KPI_THRESHOLDS["max_dd_monthly_pct"],
        "max_dd_yearly_pct": abs(dd_pct) <= KPI_THRESHOLDS["max_dd_yearly_pct"],
        "max_consecutive_losses": max_consecutive_losses_pass,
        "payoff_ratio": payoff >= KPI_THRESHOLDS["payoff_ratio"],
        "spread_cost_multiplier": (spread_cost_multiplier or 0) >= KPI_THRESHOLDS["spread_cost_multiplier"],
        "min_n_trades_effective": n_eff >= KPI_THRESHOLDS["min_n_trades_effective"],
        "permutation_p_value": (perm_p is not None) and (perm_p < KPI_THRESHOLDS["permutation_p_value"]),
    }

    required_pass = {k: v for k, v in kpi_pass.items() if KPI_GATE_TIER[k] == "必須"}
    reference_pass = {k: v for k, v in kpi_pass.items() if KPI_GATE_TIER[k] == "参考"}

    return {
        "period": period_name,
        "monthly_sharpe": round(m_sharpe, 3),
        "max_dd_pct": round(dd_pct, 2),
        "max_dd_monthly_pct": round(dd_monthly_pct, 2),
        "max_dd_pct_of_initial_capital_reference": round(dd_pct_of_initial, 2),
        "max_dd_monthly_pct_of_initial_capital_reference": round(dd_monthly_pct_of_initial, 2),
        "profit_factor": round(pf, 3),
        "payoff_ratio": round(payoff, 3),
        "max_consecutive_losses": max_losses,
        "k3m_scale_invariant": k3m_result,
        "monthly_expectancy_positive": monthly_expectancy_positive,
        "spread_cost_multiplier": round(spread_cost_multiplier, 2) if spread_cost_multiplier else None,
        "n_trades_effective": round(n_eff, 1),
        "permutation_p_clustered": perm_p,
        "kpi_pass": kpi_pass,
        "kpi_pass_count": f"{sum(kpi_pass.values())}/{len(kpi_pass)}",
        # T-11対応(2026-08-21): 必須ゲート(9項目、全てPASS必須)と参考指標
        # (K3mのみ、GO/REJECTを単独では左右しない)を分離した集計。
        "kpi_required_pass_count": f"{sum(required_pass.values())}/{len(required_pass)}",
        "kpi_required_all_pass": all(required_pass.values()),
        "kpi_reference_pass_count": f"{sum(reference_pass.values())}/{len(reference_pass)}" if reference_pass else None,
    }


def main() -> int:
    print("=== EXP-FX000005: SYS-FX011 ダウ理論連続押し目買い版(確定コストモデル) 正式KPI判定 ===\n")

    with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_1000usd_backtest.json").open(encoding="utf-8") as f:
        backtest = json.load(f)

    results = {}
    for period_name in ["train", "validation", "test"]:
        r = evaluate_period(period_name, backtest["periods"][period_name])
        results[period_name] = r
        print(f"--- {period_name} ---")
        print(f"  月次シャープ={r['monthly_sharpe']}  最大DD(ピーク比)={r['max_dd_pct']}%(月間{r['max_dd_monthly_pct']}%)  "
              f"[参考:初期資金比={r['max_dd_pct_of_initial_capital_reference']}%]  "
              f"PF={r['profit_factor']}  ペイオフ={r['payoff_ratio']}  最大連敗={r['max_consecutive_losses']}")
        print(f"  スプレッドコスト倍率={r['spread_cost_multiplier']}  実効n={r['n_trades_effective']}  "
              f"perm_p={r['permutation_p_clustered']}")
        print(f"  KPI: {r['kpi_pass_count']} 達成  {r['kpi_pass']}\n")

    print("=== サマリ ===")
    print(f"{'期間':<12}{'Sharpe':>8}{'最大DD':>8}{'PF':>7}{'ペイオフ':>9}{'実効n':>8}{'perm_p':>8}{'KPI達成':>9}")
    for name in ["train", "validation", "test"]:
        r = results[name]
        print(f"{name:<12}{r['monthly_sharpe']:>8}{r['max_dd_pct']:>7.2f}%{r['profit_factor']:>7}"
              f"{r['payoff_ratio']:>9}{r['n_trades_effective']:>8}{str(r['permutation_p_clustered']):>8}"
              f"{r['kpi_pass_count']:>9}")

    all_pass = all(r["kpi_pass_count"].split("/")[0] == r["kpi_pass_count"].split("/")[1] for r in results.values())
    print(f"\n3期間すべてで全KPI達成: {'はい' if all_pass else 'いいえ'}")

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_kpi_evaluation.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "kpi_thresholds": KPI_THRESHOLDS,
            "periods": results,
            "all_periods_all_kpi_pass": all_pass,
            "_note": (
                "確定コストモデル(エントリー=成行/SL・トレーリング・TP=指値/週末・MAX_HOLD=成行)、"
                "$1,000初期資金・1%リスクサイジングでのTVT結果を対象にK1m〜K7m相当+実効n+"
                "permutation有意性の全ゲートで正式判定。K6m(フォワード比較)・K7m(両建て証拠金)"
                "は本戦略の現行設計では判定対象外。"
            ),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
