"""EXP-FX000005 T-12: 証拠金維持率ベースのロスカット判定(ストレステスト近似).

外部レビュー・C査読が共通して指摘した未評価事項: 実質レバレッジがp90超で
25倍キャップ超過に達するケースがあり、DDが基準内でも複数通貨のポジションが
同時に保有されている局面で、GMOコイン外国為替FXの証拠金維持率ルール
(有効証拠金÷必要証拠金×100。125%未満でロスカットアラート、100%未満で
ロスカット=全ポジション強制決済。出典: GMOコインサポート
https://support.coin.z.com/hc/ja/articles/17884183390105 、
2026-08-21 WebSearchで確認)に基づく強制決済が、個々のポジションの
ストップロス(SL)到達より先に発動する経路が評価されていなかった。

## 再実施(2026-08-21、重複トレード生成バグ修正後)

初回実施時、本スクリプトの結果(最大同時保有ポジション数13件、1通貨1ポジション
制約下では本来最大4件のはず)が、`find_trades_for_period()`の重複トレード生成
バグ(同一方向の連続H1ブレイクバーが独立した追跡チェーンを開始し、同一トレード
が複数回生成される)を発見する直接の端緒となった。バグ修正
(`select_non_overlapping_breakout_events()`新設)後のデータで再実施する。

## 制約と近似(結果を見る前に方法論を固定)

継続時間軸(M5/H1バーごと)の含み損益を全ポジション分マーク・トゥ・マーケット
する完全なシミュレーションは、既存のイベント駆動型($1,000バックテストは
ENTRY/EXIT時点でのみ残高を更新する設計)を大幅に拡張する必要があり、本対応
の範囲を超える。そこで、より保守的(悲観的)な**ワーストケース・ストレス
近似**を採用する:

各ENTRYイベント時点(=新規ポジション追加直後)で、その時点までに未決済の
全ポジション(最大4、1通貨1ポジション制約下)について、
  - 必要証拠金合計 = Σ(risk_dollars_i × leverage_ratio_i / MAX_LEVERAGE)
    (`risk_dollars×leverage_ratio`はエントリー時のポジション想定元本に相当
    し、それをGMOコインの最大レバレッジ25倍で割ったものが必要証拠金)
  - ワーストケース含み損 = Σ(risk_dollars_i)
    (=全建玉が同時に初期ストップ(-1R)に到達した場合の損失。実際にはSLで
    自動決済されるため通常は発生しないが、複数通貨が相関して急変動する
    局面(価格反応型フィルターが検知する「相関ショック」がまさにこれ)
    では、複数ポジションが同時にSL到達直前まで含み損を抱える瞬間が
    現実にありうるため、上限としては妥当な仮定)
  - ワーストケース有効証拠金 = 直近残高 - ワーストケース含み損
  - ワーストケース証拠金維持率 = ワーストケース有効証拠金 ÷ 必要証拠金合計 × 100

この値が125%を下回ればアラート相当、100%を下回ればロスカット相当と判定する。
実際の含み損益は連続的に変動するため、この近似は「最悪の場合どこまで悪化
しうるか」の上限を示すものであり、実際にこの水準に達したことを意味しない
(保守的すぎる可能性がある)点に注意。

## 再々実施(2026-08-21、T-16再査読の指摘によりtrailonly版データへ差し替え)

上記「再実施」時点ではdedupfix版(T-13適用前、Train540/Validation138/Test173件)を
参照していたが、T-13(出口設計のトレール専業化)適用後の最終候補(trailonly版、
Train524/Validation133/Test170件)では一度も再実行されていなかった。独立C査読が
この不整合を指摘したため、参照データをtrailonly版に差し替えて再実行する。
最大同時保有ポジション数(=1通貨1ポジション制約の理論上限4)は変わらないと
推定されるが、ロスカット相当水準の発生割合(約25〜35%)自体は未検証だった
数値であり、確定させる。

出力: research/method-notes/margin_call_stress.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAX_LEVERAGE = 25.0
INITIAL_CAPITAL_USD = 1000.0
ALERT_THRESHOLD_PCT = 125.0
LOSSCUT_THRESHOLD_PCT = 100.0


def main() -> int:
    print("=== EXP-FX000005 T-12: 証拠金維持率ベースのロスカット判定(ストレステスト近似) ===\n")

    with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd_backtest.json").open(
        encoding="utf-8"
    ) as f:
        backtest = json.load(f)

    result: dict = {"generated_at": datetime.now().isoformat(), "periods": {}}

    for period_name in ["train", "validation", "test"]:
        trades = backtest["periods"][period_name]["trades"]
        trades = [t for t in trades if not t.get("skipped_ruin")]
        trades_sorted = sorted(trades, key=lambda t: t["entry_time"])

        events = []
        for idx, t in enumerate(trades_sorted):
            events.append((pd.Timestamp(t["entry_time"]), 0, idx, "ENTRY"))
            events.append((pd.Timestamp(t["exit_time"]), 1, idx, "EXIT"))
        events.sort(key=lambda e: (e[0], e[1]))

        balance = INITIAL_CAPITAL_USD
        open_positions: dict[int, dict] = {}
        min_ratio = float("inf")
        min_ratio_time = None
        max_concurrent = 0
        max_required_margin_pct_of_balance = 0.0
        n_alert_breaches = 0
        n_losscut_breaches = 0

        for time_, _order, idx, kind in events:
            t = trades_sorted[idx]
            if kind == "ENTRY":
                open_positions[idx] = t
                max_concurrent = max(max_concurrent, len(open_positions))

                required_margin_total = sum(
                    p["risk_dollars"] * p["leverage_ratio"] / MAX_LEVERAGE for p in open_positions.values()
                )
                worst_case_loss = sum(p["risk_dollars"] for p in open_positions.values())
                worst_case_equity = balance - worst_case_loss

                if required_margin_total > 0:
                    ratio = worst_case_equity / required_margin_total * 100.0
                    if ratio < min_ratio:
                        min_ratio = ratio
                        min_ratio_time = str(time_)
                    if ratio < ALERT_THRESHOLD_PCT:
                        n_alert_breaches += 1
                    if ratio < LOSSCUT_THRESHOLD_PCT:
                        n_losscut_breaches += 1
                    req_pct_of_balance = required_margin_total / balance * 100.0
                    max_required_margin_pct_of_balance = max(max_required_margin_pct_of_balance, req_pct_of_balance)
            else:
                open_positions.pop(idx, None)
                balance = t["balance_after"]

        n_trades = len(trades_sorted)
        print(f"--- {period_name} (n={n_trades}) ---")
        print(f"  最大同時保有ポジション数: {max_concurrent}")
        print(f"  最大必要証拠金(対残高比): {max_required_margin_pct_of_balance:.2f}%")
        print(f"  ワーストケース証拠金維持率の最小値: {min_ratio:.1f}%  (発生時刻: {min_ratio_time})")
        print(f"  125%(アラート相当)を下回った回数: {n_alert_breaches}")
        print(f"  100%(ロスカット相当)を下回った回数: {n_losscut_breaches}\n")

        result["periods"][period_name] = {
            "n_trades": n_trades,
            "max_concurrent_positions": max_concurrent,
            "max_required_margin_pct_of_balance": round(max_required_margin_pct_of_balance, 2),
            "worst_case_maintenance_ratio_min_pct": round(min_ratio, 1) if min_ratio != float("inf") else None,
            "worst_case_maintenance_ratio_min_time": min_ratio_time,
            "n_alert_threshold_breaches_125pct": n_alert_breaches,
            "n_losscut_threshold_breaches_100pct": n_losscut_breaches,
        }

    out_path = ROOT / "research" / "method-notes" / "margin_call_stress.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
