"""EXP-FX000004 着手前の簡易診断: 週末持ち越し禁止ルールがキャリー戦略の
収益にどれだけ影響するかを、無制約(真のバイ&ホールド)との比較で定量化する.

prescreen(`research/EXP-FX000004/00-prescreen.md`)で指摘した通り、本PJの
既存スコープ規定「週末持ち越し不可」は古典的なキャリー戦略の前提と衝突する。
フェーズゲート・spec確定などの本格的な検証に入る前に、まずこの制約単体が
戦略の成立可能性をどれだけ損なうかを確認する。

対象: JPYクロス4通貨(USD/JPY, EUR/JPY, GBP/JPY, AUD/JPY)。実際の政策金利
推移(DS-7 v0.2)では、日銀の政策金利が検証期間全体を通じて他4中央銀行より
低いため、方向判定(ロング/ショート)は不要で「常にロング」の単純なバイ&
ホールドとして扱う(この前提が崩れる場合は別途フィルターが必要になる)。

事前登録 (結果を見る前に固定):
    - 無制約シナリオ: 期間開始時に1lotロングで建て、期間終了まで保有し続ける
      (価格変動によるP&L + 保有全暦日分のスワップ受取)
    - 週次サイクル制約シナリオ: 毎週月曜の最初の取引時刻にロングで建て、
      金曜の最終取引時刻(土曜06:00 JST close前)で手仕舞う。スワップは
      平日保有分のみ受取。週をまたぐ再エントリーのスリッページ/スプレッド
      コストはこの診断では含めない(制約の「構造的な」影響のみを見るため)
    - 対象期間: Train(2023-11-01〜2025-03-31)相当。まずTrainのみで確認する

出力: research/method-notes/carry_weekend_constraint_diagnostic.json
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

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
LOT_SIZE = 1000
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
    return df["swap_long_jpy"][(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]


def is_weekend_close_time(ts: pd.Timestamp) -> bool:
    return ts.weekday() == 5 and ts.hour >= 6


def main() -> int:
    print("=== EXP-FX000004 簡易診断: 週末持ち越し禁止ルールの収益影響 (Train期間) ===\n")
    results = {}
    for pair in PAIRS:
        m5 = load_m5(pair)
        swap_daily = load_swap_daily(pair)

        start_price = float(m5["open"].iloc[0])
        end_price = float(m5["close"].iloc[-1])

        # --- 無制約シナリオ ---
        unconstrained_price_pnl = LOT_SIZE * (end_price - start_price)
        unconstrained_swap_pnl = float(swap_daily.sum())
        unconstrained_total = unconstrained_price_pnl + unconstrained_swap_pnl

        # --- 週次サイクル制約シナリオ ---
        weeks = m5.groupby(m5.index.to_period("W-SAT"))
        constrained_price_pnl = 0.0
        constrained_swap_pnl = 0.0
        n_weeks = 0
        for _period, week_df in weeks:
            tradeable = week_df[~week_df.index.map(is_weekend_close_time)]
            if len(tradeable) < 2:
                continue
            entry_price = float(tradeable["open"].iloc[0])
            exit_price = float(tradeable["close"].iloc[-1])
            constrained_price_pnl += LOT_SIZE * (exit_price - entry_price)
            # swap_daily.indexはtz-naive(日付文字列由来)なので、tz-aware側を
            # 揃えてから比較する(揃えないとisin()が常にFalseになる既知の罠)
            week_dates = tradeable.index.tz_localize(None).normalize().unique()
            week_swap = swap_daily[swap_daily.index.isin(week_dates)].sum()
            constrained_swap_pnl += float(week_swap)
            n_weeks += 1
        constrained_total = constrained_price_pnl + constrained_swap_pnl

        results[pair] = {
            "unconstrained": {
                "price_pnl_jpy": round(unconstrained_price_pnl, 0),
                "swap_pnl_jpy": round(unconstrained_swap_pnl, 0),
                "total_jpy": round(unconstrained_total, 0),
            },
            "weekly_cycle_constrained": {
                "n_weeks": n_weeks,
                "price_pnl_jpy": round(constrained_price_pnl, 0),
                "swap_pnl_jpy": round(constrained_swap_pnl, 0),
                "total_jpy": round(constrained_total, 0),
            },
            "swap_capture_ratio": round(constrained_swap_pnl / unconstrained_swap_pnl, 4) if unconstrained_swap_pnl else None,
            "total_capture_ratio": round(constrained_total / unconstrained_total, 4) if unconstrained_total else None,
        }
        r = results[pair]
        print(f"[{pair}]")
        print(f"  無制約:       価格PnL={r['unconstrained']['price_pnl_jpy']:>10,.0f}円  "
              f"スワップPnL={r['unconstrained']['swap_pnl_jpy']:>10,.0f}円  合計={r['unconstrained']['total_jpy']:>10,.0f}円")
        print(f"  週次制約({n_weeks}週): 価格PnL={r['weekly_cycle_constrained']['price_pnl_jpy']:>10,.0f}円  "
              f"スワップPnL={r['weekly_cycle_constrained']['swap_pnl_jpy']:>10,.0f}円  合計={r['weekly_cycle_constrained']['total_jpy']:>10,.0f}円")
        print(f"  スワップ捕捉率={r['swap_capture_ratio']}  総合捕捉率={r['total_capture_ratio']}\n")

    out_path = ROOT / "research" / "method-notes" / "carry_weekend_constraint_diagnostic.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "lot_size": LOT_SIZE,
            "results": results,
            "_note": (
                "無制約(真のバイ&ホールド)と週次サイクル制約(週末持ち越し禁止ルール"
                "遵守)を比較し、本PJのスコープ制約がキャリー戦略の収益(特にスワップ"
                "捕捉率)にどれだけ影響するかを定量化。再エントリーのスリッページ/"
                "スプレッドコストは含めない(制約自体の構造的影響のみを見るため)。"
            ),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
