"""SYS-FX010(EXP-FX000004)REJECT再検証: backtest_carry_baseline.pyの
`atr_d1.asof(entry_time)` に存在する先読み経路の影響を定量化する.

## 発見した先読み経路(コード上で特定)

`scripts/backtest_carry_baseline.py` の `weekly_cycles()` は、週初(月曜)の
M5始値でエントリーする際、そのエントリー時点のATR(D1,14)を
`atr_d1.asof(entry_time)` (99行目・211行目)で取得している。

`atr_d1`はD1(resample('D')、既定label='left')でインデックスされており、
`atr_d1.index[k]`はその日の**開始**時刻(00:00)である。しかしATR(D1,14)の
値自体は当日の終値・高安を含むTrue Range計算(標準的なATR実装)のため、
`atr_d1.iloc[k]`は当日23:55時点でなければ確定しない。`entry_time`は
月曜の最初のM5バー(週初、00:00台)であり、`atr_d1.asof(entry_time)`は
「月曜の(まだ形成中の)日足ATR」を返してしまう。これはH1確定時刻先読み
バグ(h1_confirm_time_lookahead_impact.json)と同型で、対象がD1バーになった
もの。最大先読み幅はD1バー1本分=24時間。

## 影響範囲の特定(偽陽性チェック、最重要)

このバグは2箇所で使われる:
  (a) `weekly_cycles(..., k_stop=k_stop)` (ストップ有りバリアント、
      backtest_carry_baseline.py ステップ3、Trainのみで試行): stop_price
      算出に `atr_at_entry` の値を直接使うため、**影響あり**
      (ストップ発動タイミング・PnLが変わりうる)
  (b) `weekly_cycles(..., k_stop=None)` (ストップ無しバリアント=
      backtest_carry_no_stop_tvt.pyが呼ぶ**公式Train/Validation/Test評価**):
      `stop_price = ... if k_stop is not None else None` により
      k_stop=Noneの場合はstop_price自体が常にNoneとなり、atr_at_entryの
      **値**はPnL・イグジットタイミングに一切使われない。使われるのは
      `if pd.isna(atr_at_entry) or atr_at_entry <= 0: continue` という
      「その週をサンプルに含めるか」の判定のみ。ATRがNaNになるのは系列
      先頭のウォームアップ期間(14営業日)のみであり、通常運用中の週で
      NaN/0になることは実質的にない。→ **公式のREJECT判定(no-stopバリアント)
      への影響は事実上なし**と予想されるが、本スクリプトで実測して確認する

SYS-FX010のREJECT確定(2026-08-19)は、この`no_stop`バリアントの
Train/Validation/Test評価(`carry_no_stop_tvt.json`)、および価格変動と
スワップの収益構造分解(価格変動由来78〜86%、スワップ由来14〜23%、
相関ほぼゼロ)を根拠にしている。stopバリアント(ステップ3)はTrainで
過学習と判明しValidationで崩れたため不採用となっており、そもそも
最終判定には使われていない。

## 検証方法

`atr_d1`のインデックスラベルを+1日(D1バー長)シフトしたコピーを作り、
`weekly_cycles()`(既存の共有本番コード、無変更)にそのまま渡す。
(a) ストップ有り(Train、k_stop再導出込み)と(b) ストップ無し
(Train/Validation/Test、公式判定に使われたバリアント)の両方で
シフト前後のサンプル数・KPIを比較する。

出力: research/method-notes/fx010_carry_atr_asof_lookahead_impact.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import backtest_carry_baseline as base  # noqa: E402
from minmax_fx_dt.decision.criteria import compute_n_trades_effective  # noqa: E402
from minmax_fx_dt.backtest.permutation import permutation_test_clustered  # noqa: E402

CONFIRM_SHIFT = pd.Timedelta(days=1)  # D1バー長(既定resample('D'))と一致させる

PERIODS = {
    "train":      ("2023-11-01", "2025-03-31"),
    "validation": ("2025-04-01", "2025-11-30"),
    "test":       ("2025-12-01", "2026-08-15"),
}
KPI_THRESHOLDS = {
    "monthly_sharpe": 0.4, "profit_factor": 1.2, "max_dd_monthly_pct": 10.0,
    "max_dd_yearly_pct": 20.0, "payoff_ratio": 1.5, "min_n_trades_effective": 300,
}


def shifted_atr(atr_d1: pd.Series) -> pd.Series:
    atr_c = atr_d1.copy()
    atr_c.index = atr_d1.index + CONFIRM_SHIFT
    return atr_c


def run_no_stop_period(period_name: str, start: str, end: str, use_confirm_time: bool) -> dict:
    base.TRAIN_START, base.TRAIN_END = start, end
    cycles_all = []
    n_by_pair = {}
    for pair in base.PAIRS:
        m5 = base.load_m5(pair)
        d1 = base.to_d1(m5)
        atr_d1 = base.atr_ind(d1["high"], d1["low"], d1["close"], length=14)
        if use_confirm_time:
            atr_d1 = shifted_atr(atr_d1)
        swap_daily = base.load_swap_daily(pair)
        cycles = base.weekly_cycles(pair, m5, d1, atr_d1, swap_daily, k_stop=None)
        cycles_all.extend(cycles)
        n_by_pair[pair] = len(cycles)

    summary = base.summarize(cycles_all, f"no_stop_{period_name}")
    pnls = [c["total_pnl_jpy"] for c in cycles_all]
    pairs_list = [c["pair"] for c in cycles_all]
    perm = permutation_test_clustered(pnls, pairs_list, seed=42) if len(pnls) >= 4 else None
    n_eff = compute_n_trades_effective(n_by_pair, summary["n_cycles"]) if pairs_list else 0
    return {
        "period": period_name, "n_cycles_total": len(cycles_all), "n_by_pair": n_by_pair,
        "n_trades_effective": n_eff,
        "perm_p_value": round(perm.p_value, 4) if perm else None,
        "summary": summary,
    }


def run_with_stop_train(use_confirm_time: bool) -> dict:
    base.TRAIN_START, base.TRAIN_END = PERIODS["train"]
    all_data = {}
    for pair in base.PAIRS:
        m5 = base.load_m5(pair)
        d1 = base.to_d1(m5)
        atr_d1 = base.atr_ind(d1["high"], d1["low"], d1["close"], length=14)
        if use_confirm_time:
            atr_d1 = shifted_atr(atr_d1)
        swap_daily = base.load_swap_daily(pair)
        all_data[pair] = (m5, d1, atr_d1, swap_daily)

    baseline_cycles_all = []
    price_only_returns = []
    for pair in base.PAIRS:
        m5, d1, atr_d1, swap_daily = all_data[pair]
        cycles = base.weekly_cycles(pair, m5, d1, atr_d1, swap_daily, k_stop=None)
        baseline_cycles_all.extend(cycles)
        price_only_returns.extend(c["price_pnl_jpy"] for c in cycles)

    atr_ratios = []
    for pair in base.PAIRS:
        m5, d1, atr_d1, swap_daily = all_data[pair]
        cycles = [c for c in baseline_cycles_all if c["pair"] == pair]
        for c in cycles:
            entry_time = pd.Timestamp(c["entry_time"])
            atr_at_entry = atr_d1.asof(entry_time)
            if pd.isna(atr_at_entry) or atr_at_entry <= 0:
                continue
            price_move = c["exit_price"] - c["entry_price"]
            atr_ratios.append(price_move / float(atr_at_entry))
    atr_ratios = np.array(atr_ratios)
    k_stop = float(-np.percentile(atr_ratios, 10)) if len(atr_ratios) else None

    stopped_cycles_all = []
    for pair in base.PAIRS:
        m5, d1, atr_d1, swap_daily = all_data[pair]
        cycles = base.weekly_cycles(pair, m5, d1, atr_d1, swap_daily, k_stop=k_stop)
        stopped_cycles_all.extend(cycles)

    stopped_summary = base.summarize(stopped_cycles_all, f"k_stop={k_stop:.3f}" if k_stop else "k_stop=None")
    return {
        "n_cycles_no_stop": len(baseline_cycles_all),
        "k_stop_derived": round(k_stop, 4) if k_stop else None,
        "n_cycles_with_stop": len(stopped_cycles_all),
        "n_stopped": sum(1 for c in stopped_cycles_all if c.get("stopped")),
        "with_stop_summary": stopped_summary,
    }


def main() -> int:
    print("=== SYS-FX010: backtest_carry_baseline.py atr_d1.asof() 先読み影響の定量化 ===\n")

    print("[1/2] 公式バリアント(ストップ無し、REJECT判定に使用): Train/Validation/Test x バグ再現/修正版")
    no_stop_results = {}
    for period_name, (start, end) in PERIODS.items():
        bug = run_no_stop_period(period_name, start, end, use_confirm_time=False)
        fixed = run_no_stop_period(period_name, start, end, use_confirm_time=True)
        print(f"  [{period_name}] バグ版 n_cycles={bug['n_cycles_total']} sharpe={bug['summary']['monthly_sharpe']} "
              f"PF={bug['summary']['profit_factor']}  |  修正版 n_cycles={fixed['n_cycles_total']} "
              f"sharpe={fixed['summary']['monthly_sharpe']} PF={fixed['summary']['profit_factor']}")
        no_stop_results[period_name] = {"reproduced_bug": bug, "confirm_time_fixed": fixed}

    official_path = ROOT / "research" / "method-notes" / "carry_no_stop_tvt.json"
    official = None
    if official_path.exists():
        with official_path.open(encoding="utf-8") as f:
            official = json.load(f)

    print("\n[2/2] 参考バリアント(ストップ有り、Trainのみ試行・過学習のため不採用): バグ再現/修正版")
    with_stop_bug = run_with_stop_train(use_confirm_time=False)
    with_stop_fixed = run_with_stop_train(use_confirm_time=True)
    print(f"  バグ版 k_stop={with_stop_bug['k_stop_derived']}  n_stopped={with_stop_bug['n_stopped']}/"
          f"{with_stop_bug['n_cycles_with_stop']}  sharpe={with_stop_bug['with_stop_summary']['monthly_sharpe']}")
    print(f"  修正版 k_stop={with_stop_fixed['k_stop_derived']}  n_stopped={with_stop_fixed['n_stopped']}/"
          f"{with_stop_fixed['n_cycles_with_stop']}  sharpe={with_stop_fixed['with_stop_summary']['monthly_sharpe']}")

    no_stop_unaffected = all(
        no_stop_results[p]["reproduced_bug"]["n_cycles_total"] == no_stop_results[p]["confirm_time_fixed"]["n_cycles_total"]
        and no_stop_results[p]["reproduced_bug"]["summary"]["total_return_pct"]
        == no_stop_results[p]["confirm_time_fixed"]["summary"]["total_return_pct"]
        for p in PERIODS
    )

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "system": "SYS-FX010 (EXP-FX000004)",
        "purpose": (
            "過去REJECT戦略について、H1確定時刻先読みバグと同種の問題が判定に"
            "影響していないかをbacktest_carry_baseline.pyについて確認する"
        ),
        "lookahead_path_found": True,
        "lookahead_path_location": "scripts/backtest_carry_baseline.py:99,211 "
                                    "atr_at_entry = atr_d1.asof(entry_time)",
        "lookahead_path_description": (
            "atr_d1(D1バーの左ラベル=開始時刻でインデックス)を、週初のM5エントリー"
            "時刻(月曜00:00台)に対して.asof()で参照すると、まだ形成中の月曜の"
            "日足ATR(23:55まで確定しない)を返してしまう。最大先読み幅=D1バー1本=24時間。"
        ),
        "false_positive_check": {
            "no_stop_variant(公式REJECT判定に使用)": (
                "k_stop=Noneのためstop_price=None固定となり、atr_at_entryの値自体は"
                "PnL/イグジットタイミングに一切使われない。使われるのはNaN判定による"
                "サンプル除外のみ(ATRウォームアップ期間14営業日以外はNaNにならない)。"
                "実測結果: " + ("シフト前後で完全に同一(影響なしを実証)" if no_stop_unaffected
                              else "シフト前後で差異あり(下記no_stop_variant_resultsを参照)")
            ),
            "with_stop_variant(Trainのみの試行、過学習のため不採用・最終判定には未使用)": (
                "stop_price算出にatr_at_entryの値を直接使うため影響あり。"
                "ただしこのバリアントはTrainで過学習と判明しValidationで崩れたため、"
                "そもそもSYS-FX010のREJECT根拠(no_stopバリアントの3期間評価 + "
                "価格変動/スワップの収益構造分解)には採用されていない"
            ),
        },
        "reproduction_check": {
            "official_carry_no_stop_tvt_json_exists": official is not None,
            "official_summary_by_period": (
                {p: official["periods"][p]["summary"] for p in PERIODS if official and p in official.get("periods", {})}
                if official else None
            ),
        },
        "no_stop_variant_results(official_reject_basis)": no_stop_results,
        "with_stop_variant_results(unused_reference)": {
            "reproduced_bug": with_stop_bug,
            "confirm_time_fixed": with_stop_fixed,
        },
        "conclusion": (
            "公式のREJECT判定根拠(no_stopバリアント、Train/Validation/Test)は"
            + ("先読みバグの影響を受けていないことを実測で確認した(シフト前後で結果が完全一致)。"
               if no_stop_unaffected else
               "先読みバグの影響を一部受けている可能性がある(詳細は上記no_stop_variant_resultsを参照)。")
            + " ストップ有りバリアント(不採用)はatr_at_entryの値を直接使うため影響を受けるが、"
              "そもそも最終判定には使われていないため、SYS-FX010のREJECT判定自体は覆らない。"
              "REJECT根拠の本質(価格変動とスワップの収益構造分解、方向性予測力の欠如)は"
              "本バグの影響を受けない別系統の分析であり、この点でもREJECT判定は堅牢。"
        ),
    }
    out_path = ROOT / "research" / "method-notes" / "fx010_carry_atr_asof_lookahead_impact.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
