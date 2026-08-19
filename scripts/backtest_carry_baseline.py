"""EXP-FX000004 フェーズゲート2: スワップキャリー戦略のベースライン(ストップ無し)
バックテストと、週内ストップ幅(k_stop)のデータ駆動導出.

`00-spec.md`で確定した仕様(常にロング・週次サイクル・JPYクロス4通貨)を、まず
価格リスクオーバーレイ無しで実行し、Train期間の週次リターン分布(特に2024年8月
の日銀利上げショック)を実測する。次に、価格変動分のみの週次リターン分布から
パーセンタイルベースでk_stopを導出し、ストップ有りバージョンと比較する。

事前登録 (結果を見る前に固定):
    - ポジションサイジング: 固定ロット(1lot=1,000通貨)、初期資金1,000,000円
      (SYS-FX007/008/009のSimulatorConfigと同一の慣行を踏襲)
    - 「トレード」の単位: 1週間の保有サイクル
    - k_stop候補: 週次価格変動リターン(スワップ除く)の下位10%点(p10)。
      OBS000006のパーセンタイル導出慣行(p70等)に合わせ、片側の裾から
      閾値を取る考え方を踏襲。ATR(D1,14)の週初値に対する倍率として表現する
    - ストップ判定: 週の各日について、その日の安値(ロングの場合)が
      entry_price - k_stop*ATR を下回ったら、その日の終値で手仕舞う
      (週内であればいつでも、金曜を待たない)
    - 対象期間: Train(2023-11-01〜2025-03-31)のみ。Validation/Testは
      k_stop確定後に別途実施

出力: research/method-notes/carry_baseline_train.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from minmax_fx_dt.backtest.metrics import (
    max_drawdown,
    monthly_max_dd_pct,
    monthly_sharpe,
    payoff_ratio,
    profit_factor,
)
from minmax_fx_dt.strategy.indicators import atr as atr_ind

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
LOT_SIZE = 1000
INITIAL_CASH_JPY = 1_000_000.0
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"


def load_m5(pair: str) -> pd.DataFrame:
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]


def load_swap_daily(pair: str) -> pd.Series:
    with (ROOT / "data" / "curated" / "ds-7.json").open(encoding="utf-8") as f:
        ds7 = json.load(f)
    series = ds7["pairs"][pair]["daily_series"]
    df = pd.DataFrame(series)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df["swap_long_jpy"]


def to_d1(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "open": m5["open"].resample("D").first(), "high": m5["high"].resample("D").max(),
        "low": m5["low"].resample("D").min(), "close": m5["close"].resample("D").last(),
    }).dropna()


def is_weekend_close_time(ts: pd.Timestamp) -> bool:
    return ts.weekday() == 5 and ts.hour >= 6


def weekly_cycles(pair: str, m5: pd.DataFrame, d1: pd.DataFrame, atr_d1: pd.Series,
                   swap_daily: pd.Series, k_stop: float | None) -> list[dict]:
    """1週間=1サイクルとして、ストップ有り/無しの結果を計算する.

    k_stop=Noneならストップ無し(金曜まで保有)。数値ならATR(D1,14)*k_stop
    だけ不利に動いた時点でその日のうちに手仕舞う。
    """
    weeks = m5.groupby(m5.index.to_period("W-SAT"))
    cycles = []
    for _period, week_df in weeks:
        tradeable = week_df[~week_df.index.map(is_weekend_close_time)]
        if len(tradeable) < 2:
            continue
        entry_time = tradeable.index[0]
        entry_price = float(tradeable["open"].iloc[0])
        atr_at_entry = atr_d1.asof(entry_time)
        if pd.isna(atr_at_entry) or atr_at_entry <= 0:
            continue

        stop_price = entry_price - k_stop * float(atr_at_entry) if k_stop is not None else None
        exit_price = float(tradeable["close"].iloc[-1])
        exit_time = tradeable.index[-1]
        stopped = False
        if stop_price is not None:
            hit = tradeable[tradeable["low"] <= stop_price]
            if len(hit) > 0:
                exit_time = hit.index[0]
                exit_price = float(hit["close"].iloc[0])
                stopped = True

        price_pnl = LOT_SIZE * (exit_price - entry_price)
        week_dates = tradeable[tradeable.index <= exit_time].index.tz_localize(None).normalize().unique()
        swap_pnl = float(swap_daily[swap_daily.index.isin(week_dates)].sum())
        cycles.append({
            "pair": pair, "entry_time": str(entry_time), "exit_time": str(exit_time),
            "entry_price": entry_price, "exit_price": exit_price, "stopped": stopped,
            "price_pnl_jpy": round(price_pnl, 1), "swap_pnl_jpy": round(swap_pnl, 1),
            "total_pnl_jpy": round(price_pnl + swap_pnl, 1),
        })
    return cycles


def build_equity_curve(cycles: list[dict]) -> pd.DataFrame:
    cycles_sorted = sorted(cycles, key=lambda c: c["exit_time"])
    equity = INITIAL_CASH_JPY
    # 起点は実際の最初のサイクルのentry_timeを使う (任意の期間境界(TRAIN_START)を
    # UTC変換すると前日にずれ、月次DD計算(metrics.monthly_max_dd_pct)の最初の月バケツが
    # 空になりNaNを返す既知の罠があるため)
    first_entry = pd.Timestamp(min(c["entry_time"] for c in cycles_sorted))
    rows = [{"timestamp": first_entry, "equity": equity}]
    for c in cycles_sorted:
        equity += c["total_pnl_jpy"]
        rows.append({"timestamp": pd.Timestamp(c["exit_time"]), "equity": equity})
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def summarize(cycles: list[dict], label: str) -> dict:
    pnls = [c["total_pnl_jpy"] for c in cycles]
    eq = build_equity_curve(cycles)
    dd_jpy, dd_pct = max_drawdown(eq)
    n_wins = sum(1 for p in pnls if p > 0)
    summary = {
        "label": label, "n_cycles": len(cycles),
        "final_equity_jpy": round(float(eq["equity"].iloc[-1]), 0),
        "total_return_pct": round((float(eq["equity"].iloc[-1]) / INITIAL_CASH_JPY - 1) * 100, 2),
        "win_rate": round(n_wins / len(pnls), 3) if pnls else None,
        "monthly_sharpe": round(monthly_sharpe(eq), 3),
        "max_dd_pct": round(dd_pct, 2),
        "max_dd_monthly_pct": round(monthly_max_dd_pct(eq, INITIAL_CASH_JPY), 2),
        "profit_factor": round(profit_factor(pnls), 3) if pnls else None,
        "payoff_ratio": round(payoff_ratio(pnls), 3) if pnls else None,
        "n_stopped": sum(1 for c in cycles if c.get("stopped")),
    }
    return summary


def main() -> int:
    print("=== EXP-FX000004 フェーズゲート2: キャリー戦略ベースライン (Train期間) ===\n")

    all_data = {}
    for pair in PAIRS:
        m5 = load_m5(pair)
        d1 = to_d1(m5)
        atr_d1 = atr_ind(d1["high"], d1["low"], d1["close"], length=14)
        swap_daily = load_swap_daily(pair)
        all_data[pair] = (m5, d1, atr_d1, swap_daily)

    print("--- ステップ1: ストップ無しベースライン ---")
    baseline_cycles_all = []
    price_only_returns = []  # k_stop導出用 (スワップ除く価格変動分)
    for pair in PAIRS:
        m5, d1, atr_d1, swap_daily = all_data[pair]
        cycles = weekly_cycles(pair, m5, d1, atr_d1, swap_daily, k_stop=None)
        baseline_cycles_all.extend(cycles)
        for c in cycles:
            price_only_returns.append(c["price_pnl_jpy"])
        print(f"  [{pair}] {len(cycles)}週  合計PnL={sum(c['total_pnl_jpy'] for c in cycles):,.0f}円")

    baseline_summary = summarize(baseline_cycles_all, "baseline_no_stop")
    print(f"\n  プール(4通貨): n={baseline_summary['n_cycles']}週  "
          f"最終資産={baseline_summary['final_equity_jpy']:,.0f}円 "
          f"({baseline_summary['total_return_pct']:+.1f}%)")
    print(f"  月次シャープ={baseline_summary['monthly_sharpe']}  最大DD={baseline_summary['max_dd_pct']}%  "
          f"月間最大DD={baseline_summary['max_dd_monthly_pct']}%  PF={baseline_summary['profit_factor']}  "
          f"ペイオフ={baseline_summary['payoff_ratio']}")

    # 2024年8月ショックの実際の影響を確認
    aug2024 = [c for c in baseline_cycles_all if "2024-08" in c["entry_time"]]
    print(f"\n  2024年8月の週次サイクル({len(aug2024)}件):")
    for c in aug2024:
        print(f"    {c['pair']}  entry={c['entry_time'][:10]}  価格PnL={c['price_pnl_jpy']:>10,.0f}円  "
              f"スワップPnL={c['swap_pnl_jpy']:>7,.0f}円  合計={c['total_pnl_jpy']:>10,.0f}円")

    print("\n--- ステップ2: k_stopの導出 (価格変動分のp10) ---")
    k_stop_price_pnls = np.array(price_only_returns)
    p10_price_pnl = float(np.percentile(k_stop_price_pnls, 10))
    # p10に相当する損失をATR単位に変換 (プール平均ではなく、各週のATRで正規化した比率のp10を使う方が
    # 正確だが、事前登録した「価格変動リターンの下位10%点」をそのままATR比に変換する簡易版として、
    # プール全体でのATR比のp10をk_stopとする
    atr_ratios = []
    for pair in PAIRS:
        m5, d1, atr_d1, swap_daily = all_data[pair]
        cycles = [c for c in baseline_cycles_all if c["pair"] == pair]
        for c in cycles:
            entry_time = pd.Timestamp(c["entry_time"])
            atr_at_entry = atr_d1.asof(entry_time)
            if pd.isna(atr_at_entry) or atr_at_entry <= 0:
                continue
            price_move = (c["exit_price"] - c["entry_price"])
            atr_ratios.append(price_move / float(atr_at_entry))
    atr_ratios = np.array(atr_ratios)
    k_stop = float(-np.percentile(atr_ratios, 10))  # 下位10%点(負の値)の絶対値をストップ幅とする
    print(f"  価格変動リターンp10={p10_price_pnl:,.0f}円  ATR比のp10={-k_stop:.3f}  → k_stop={k_stop:.3f}")

    print(f"\n--- ステップ3: k_stop={k_stop:.3f} でのストップ有りバックテスト ---")
    stopped_cycles_all = []
    for pair in PAIRS:
        m5, d1, atr_d1, swap_daily = all_data[pair]
        cycles = weekly_cycles(pair, m5, d1, atr_d1, swap_daily, k_stop=k_stop)
        stopped_cycles_all.extend(cycles)
        print(f"  [{pair}] {len(cycles)}週  ストップ発動={sum(1 for c in cycles if c['stopped'])}件  "
              f"合計PnL={sum(c['total_pnl_jpy'] for c in cycles):,.0f}円")

    stopped_summary = summarize(stopped_cycles_all, f"k_stop={k_stop:.3f}")
    print(f"\n  プール(4通貨): n={stopped_summary['n_cycles']}週  "
          f"最終資産={stopped_summary['final_equity_jpy']:,.0f}円 "
          f"({stopped_summary['total_return_pct']:+.1f}%)")
    print(f"  月次シャープ={stopped_summary['monthly_sharpe']}  最大DD={stopped_summary['max_dd_pct']}%  "
          f"月間最大DD={stopped_summary['max_dd_monthly_pct']}%  PF={stopped_summary['profit_factor']}  "
          f"ペイオフ={stopped_summary['payoff_ratio']}  ストップ発動={stopped_summary['n_stopped']}件")

    aug2024_stopped = [c for c in stopped_cycles_all if "2024-08" in c["entry_time"]]
    print(f"\n  2024年8月の週次サイクル(ストップ有り、{len(aug2024_stopped)}件):")
    for c in aug2024_stopped:
        print(f"    {c['pair']}  entry={c['entry_time'][:10]}  stopped={c['stopped']}  合計={c['total_pnl_jpy']:>10,.0f}円")

    out_path = ROOT / "research" / "method-notes" / "carry_baseline_train.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "lot_size": LOT_SIZE, "initial_cash_jpy": INITIAL_CASH_JPY,
            "k_stop_derived": round(k_stop, 4),
            "baseline_no_stop": baseline_summary,
            "with_stop": stopped_summary,
            "baseline_cycles": baseline_cycles_all,
            "with_stop_cycles": stopped_cycles_all,
            "_note": (
                "常にロング・週次サイクル(月曜建て・金曜手仕舞い)のスワップキャリー"
                "戦略のTrain期間ベースライン。k_stopは価格変動リターン(スワップ除く)"
                "のATR比p10から導出。固定ロットサイジング(1lot=1000通貨)、実運用コスト"
                "(スプレッド/スリッページ/手数料)は含まない簡易版。"
            ),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
