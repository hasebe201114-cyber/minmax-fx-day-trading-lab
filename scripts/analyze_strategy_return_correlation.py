"""既存戦略同士のリターン相関診断（提案C、OBS000012 §4）— 追加データ不要.

これまで全戦略を「**単独で**全KPIゲートを通すか」で評価してきたが、SYS-FX024 の
検証で「グリッドはトレンド局面で負け、SYS-FX011 系はトレンド局面で勝つ」という
構造的に逆向きの損益プロファイルが見えた。両者のリターン相関が明確に負なら、
合成でDDが縮小しシャープが改善しうる。

本スクリプトは既存のエクイティカーブ同士を**同一の日次グリッドに揃えて**相関を測り、
等ウェイト合成した場合の月次シャープ・最大DD（ピーク比）を算出する。

**正直な留保（結果を見る前に記録）**: SYS-FX024 R-A の期待値は依然として負
(PF 0.978)。期待値が負の戦略を足して合成シャープが上がるのは、相関が強い負でかつ
期待値がほぼゼロの場合に限られる。成功確率は高くないが、検証コストが極めて低い。

出力: research/method-notes/strategy_return_correlation.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from minmax_fx_dt.backtest.metrics import monthly_sharpe, peak_relative_max_dd_pct

TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
MN = ROOT / "research" / "method-notes"

SOURCES = {
    "SYS-FX011 (trailonly版, Train)": {
        "file": MN / "vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd_backtest.json",
        "path": ["periods", "train", "equity_curve"], "field": "balance",
    },
    "SYS-FX024 B-A (グリッド/週末フラット)": {
        "file": MN / "sysfx024_range_filter_trainonly_backtest.json",
        "path": ["equity_curves", "B-A"], "field": "balance",
    },
    "SYS-FX024 R-A (グリッド/週末持越+ﾚﾝｼﾞﾌｨﾙﾀ)": {
        "file": MN / "sysfx024_range_filter_trainonly_backtest.json",
        "path": ["equity_curves", "R-A"], "field": "balance",
    },
}


def load_daily_returns(cfg: dict) -> pd.Series:
    obj = json.loads(cfg["file"].read_text(encoding="utf-8"))
    for key in cfg["path"]:
        obj = obj[key]
    df = pd.DataFrame(obj)
    ts = pd.to_datetime(df["time"], format="mixed", utc=True).dt.tz_localize(None)
    eq = pd.Series(df[cfg["field"]].astype(float).to_numpy(), index=ts).sort_index()
    eq = eq[(eq.index >= TRAIN_START) & (eq.index <= TRAIN_END)]
    # 日次終値へ揃える（イベント時刻がバラバラなので前方補完）
    daily = eq.resample("D").last().ffill().dropna()
    return daily


def curve_stats(daily_eq: pd.Series, label: str) -> dict:
    df = pd.DataFrame({"timestamp": daily_eq.index, "equity": daily_eq.to_numpy()})
    return {
        "label": label,
        "n_days": int(len(daily_eq)),
        "total_return_pct": round(float(daily_eq.iloc[-1] / daily_eq.iloc[0] - 1) * 100, 2),
        "monthly_sharpe": round(monthly_sharpe(df), 3),
        "max_dd_peak_relative_pct": round(peak_relative_max_dd_pct(df), 2),
    }


def main() -> int:
    print("=== 既存戦略のリターン相関診断（提案C、追加データ不要）— Train期間 ===")
    print(f"期間: {TRAIN_START} 〜 {TRAIN_END}\n")

    curves: dict[str, pd.Series] = {}
    for label, cfg in SOURCES.items():
        if not cfg["file"].exists():
            print(f"  [skip] {label}: {cfg['file'].name} が見つかりません")
            continue
        curves[label] = load_daily_returns(cfg)
        st = curve_stats(curves[label], label)
        print(f"  {label}")
        print(f"    日数={st['n_days']}  総リターン={st['total_return_pct']:+.2f}%  "
              f"月次シャープ={st['monthly_sharpe']}  最大DD(ピーク比)={st['max_dd_peak_relative_pct']}%")

    common = None
    for s in curves.values():
        common = s.index if common is None else common.intersection(s.index)
    rets = pd.DataFrame({k: v.reindex(common).pct_change() for k, v in curves.items()}).dropna()
    print(f"\n  共通営業日数={len(rets)}")

    print("\n--- 日次リターン相関 ---")
    corr = rets.corr()
    labels = list(rets.columns)
    for a, b in combinations(labels, 2):
        print(f"  {a}\n   × {b}\n     相関 = {corr.loc[a, b]:+.4f}")

    print("\n--- 等ウェイト合成（各戦略に資金を等分、日次リバランス想定） ---")
    combos = {}
    for a, b in combinations(labels, 2):
        blend = (rets[a] + rets[b]) / 2.0
        eq = (1.0 + blend).cumprod() * 1000.0
        df = pd.DataFrame({"timestamp": eq.index, "equity": eq.to_numpy()})
        combos[f"{a} + {b}"] = {
            "correlation": round(float(corr.loc[a, b]), 4),
            "total_return_pct": round(float(eq.iloc[-1] / 1000.0 - 1) * 100, 2),
            "monthly_sharpe": round(monthly_sharpe(df), 3),
            "max_dd_peak_relative_pct": round(peak_relative_max_dd_pct(df), 2),
            "component_sharpes": [curve_stats(curves[a], a)["monthly_sharpe"],
                                  curve_stats(curves[b], b)["monthly_sharpe"]],
        }
        c = combos[f"{a} + {b}"]
        best_component = max(c["component_sharpes"])
        verdict = "改善" if c["monthly_sharpe"] > best_component else "改善せず"
        print(f"  {a}\n   + {b}")
        print(f"     相関={c['correlation']:+.4f}  合成リターン={c['total_return_pct']:+.2f}%  "
              f"合成シャープ={c['monthly_sharpe']}  合成DD={c['max_dd_peak_relative_pct']}%")
        print(f"     → 単独最良シャープ {best_component} に対し **{verdict}**")

    out = {
        "generated_at": datetime.now().isoformat(),
        "status": "探索診断（正式プロトコル外・spec編集なし）",
        "question": "既存戦略のリターン相関は、合成によってシャープ改善が見込める水準か",
        "period": {"train_start": TRAIN_START, "train_end": TRAIN_END, "common_days": int(len(rets))},
        "standalone": {k: curve_stats(v, k) for k, v in curves.items()},
        "correlation_matrix": {a: {b: round(float(corr.loc[a, b]), 4) for b in labels} for a in labels},
        "equal_weight_blends": combos,
        "caveats": [
            "Train期間のみ。合成の効果が Validation で再現するかは未確認",
            "日次リバランス・取引コストなしの理想化。実際には合成のためのリバランスコストがかかる",
            "SYS-FX024 の期待値は負(PF<1)であり、期待値が負の戦略を足して合成シャープが上がるのは"
            "相関が強い負でかつ期待値がほぼゼロの場合に限られる",
        ],
    }
    (MN / "strategy_return_correlation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[出力]: {MN / 'strategy_return_correlation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
