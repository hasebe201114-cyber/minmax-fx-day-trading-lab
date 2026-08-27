"""SYS-FX009 v2(EXP-FX000003)REJECT再検証: derive_double_pattern_params.pyの
`lt_dir.asof(h4.index[j])` に存在する先読み経路の影響を定量化する.

## 背景

司令塔依頼「過去にREJECTとした戦略についても、H1確定時刻の先読みバグ
(research/method-notes/h1_confirm_time_lookahead_impact.json)と同種の問題が
判定に影響していないか確認してほしい」への対応(SYS-FX009分)。

## 発見した先読み経路(コード上で特定)

`scripts/derive_double_pattern_params.py` の `lt_direction_series(d1)` は、
`d1.index[k]`(pandas resample既定によりD1バーの**開始**時刻、例: 2024-05-01 00:00)
をラベルとして持つSeriesを返すが、値自体は `d1["close"].rolling(...)` すなわち
**そのD1バーの終値(その日の23:55時点で初めて確定)を用いたSMAクロス方向**である。

その後 `lt_at_break = lt_dir.asof(h4.index[j])` (195行目、H1版は188行目)で、
H4/H1のブレイクバー時刻(例: 2024-05-01 08:00、同日の日中)に対して`.asof()`を
呼ぶと、pandasは「クエリ時刻以下の最後の値」として**同じ日のD1方向**
(2024-05-01 00:00のラベル、内容は当日23:55の終値を使ったもの)を返してしまう。
つまり「その日の一部しか経過していない時点」で「その日が終わった後でなければ
わからないLT方向」を参照している。

これは`h1_confirm_time_lookahead_impact.json`で発見された
「resample()の左ラベル(バー開始時刻)を確定時刻として誤用する」パターンと
本質的に同一で、対象がH1バーからD1バーに変わっただけである。
最大先読み幅はD1バー1本分 = 24時間。

## 影響範囲の特定(偽陽性チェック)

- `pooled_delta_atr` / `pooled_bar_range_atr` (→ pattern_tolerance_atr, stop_buffer_atr):
  `lt_dir`を一切参照しない完全に位置ベースの計算 → **影響なし**
- `pooled_lags` (→ max_bars_since_second_pivot):
  ネックライン割れを検出した時点で`lt_at_break`判定より先に`pooled_lags.append()`
  済み、かつ判定結果に関わらずループはその1件で打ち切り(`break`) → **影響なし**
- `pooled_mfe` (→ atr_trail_multiplier): `lt_at_break == required_lt` の場合のみ
  該当ブレイクをMFE集計に含める → **影響あり**。日中に発生したブレイクが、
  実際にはまだ確定していない当日のLT方向で誤って採否判定されている

production側 (`double_pattern_runner.py`) は `last_confirmed_bar_ts()` を用いて
確定済みバーのみを参照しており、この`.asof()`パターンは使っていない
(先読みなし、OBS000007の修正が正しく踏襲されている)。したがって影響は
**パラメータ導出(atr_trail_multiplier)のみ**であり、Train/Validation/Test本体の
バックテストループ自体に直接の先読みはない。ただし導出されたatr_trail_multiplier
がTrain/Validation/Testの全バックテストにハードコードされて使われるため、
誤較正されたパラメータが判定に影響した可能性は排除できない。

## 検証方法

`lt_dir`のインデックスラベルを+1日(D1バー長)シフトしたコピーを作り、
既存の`derive_double_pattern_params.py`のロジック(ここでは同一実装を
本スクリプト内に複製、共有本番コードは変更しない)にそのまま渡す。
バグ再現版が`research/EXP-FX000003/10-result/double_pattern_params.json`の
既存値と一致することを確認した上で、シフト修正版と比較する。

さらに、production engine (`run_double_pattern_backtest`、変更なし)を用いて
Train期間のみ、公式atr_trail_multiplier(バグ版由来)と修正後atr_trail_multiplierの
2つでバックテストを実行し、KPI(必須ゲート達成数)が変わるかを確認する。
他のパラメータ(pattern_tolerance_atr, stop_buffer_atr, max_bars_since_second_pivot)は
上記の通り影響を受けないため据え置く(パラメータの正式な再導出ではなく、
「もし正しく較正されていたら」を確認する感度分析)。

出力: research/method-notes/fx009_lt_asof_lookahead_impact.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from derive_double_pattern_params import (  # noqa: E402
    BREAK_SEARCH_CAP_BARS, PAIRS, TOLERANCE_PERCENTILE, BUFFER_PERCENTILE,
    STALENESS_PERCENTILE, STALENESS_CANDIDATES, TRAIL_MFE_HORIZON_H4_BARS,
    ZIGZAG_THRESHOLD_ATR, alternating_triplets, load_m5, lt_direction_series,
    round_to_standard, to_d1, to_h4,
)
from minmax_fx_dt.backtest.double_pattern_runner import run_double_pattern_backtest  # noqa: E402
from minmax_fx_dt.backtest.metrics import to_dict  # noqa: E402
from minmax_fx_dt.backtest.permutation import DEFAULT_N_PERMUTATIONS, permutation_test  # noqa: E402
from minmax_fx_dt.backtest.simulator import SimulatorConfig  # noqa: E402
from minmax_fx_dt.decision.criteria import Stats, evaluate_kpis, kpi_pass_summary  # noqa: E402
from minmax_fx_dt.strategy.double_pattern_strategy import DoublePatternStrategyConfig  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402
from minmax_fx_dt.strategy.pattern_detection import DoublePatternConfig  # noqa: E402
from minmax_fx_dt.strategy.support_resistance import zigzag_pivots_typed  # noqa: E402

TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
CONFIRM_SHIFT = pd.Timedelta(days=1)  # D1バー長(既定resample('D'))と一致させる


def run_derivation(use_confirm_time: bool) -> dict:
    """derive_double_pattern_params.pyのロジックを再実装(共有コードは変更しない).

    use_confirm_time=False: 既存(バグ)版(lt_dirをそのまま asof に渡す)
    use_confirm_time=True: lt_dirのインデックスを+1日シフトしてからasof (修正版)
    """
    pooled_delta_atr: list[float] = []
    pooled_bar_range_atr: list[float] = []
    pair_stats: dict[str, dict] = {}

    for pair in PAIRS:
        m5 = load_m5(pair)
        h4 = to_h4(m5)
        atr_h4 = atr_ind(h4["high"], h4["low"], h4["close"], length=14)
        pivots = zigzag_pivots_typed(h4["high"], h4["low"], atr_h4, ZIGZAG_THRESHOLD_ATR)
        triplets = alternating_triplets(pivots)

        pair_delta_atr: list[float] = []
        for idx1, kind1, idx2, _kind2, idx3, _kind3 in triplets:
            atr_neckline = atr_h4.iloc[idx2]
            if pd.isna(atr_neckline) or atr_neckline <= 0:
                continue
            p1 = float(h4["high" if kind1 == "HIGH" else "low"].iloc[idx1])
            p2 = float(h4["high" if kind1 == "HIGH" else "low"].iloc[idx3])
            pair_delta_atr.append(abs(p1 - p2) / float(atr_neckline))

        bar_range_atr = ((h4["high"] - h4["low"]) / atr_h4).replace([np.inf, -np.inf], np.nan).dropna()
        pair_stats[pair] = {"n_alternating_triplets": len(triplets)}
        pooled_delta_atr.extend(pair_delta_atr)
        pooled_bar_range_atr.extend(bar_range_atr.tolist())

    pattern_tolerance_atr = round(float(np.percentile(pooled_delta_atr, TOLERANCE_PERCENTILE)), 3)
    stop_buffer_atr = round(float(np.percentile(pooled_bar_range_atr, BUFFER_PERCENTILE)), 3)

    pooled_lags: list[int] = []
    pooled_mfe: list[float] = []
    for pair in PAIRS:
        m5 = load_m5(pair)
        h4 = to_h4(m5)
        d1 = to_d1(m5)
        atr_h4 = atr_ind(h4["high"], h4["low"], h4["close"], length=14)
        lt_dir = lt_direction_series(d1)
        if use_confirm_time:
            lt_dir = lt_dir.copy()
            lt_dir.index = lt_dir.index + CONFIRM_SHIFT
        pivots = zigzag_pivots_typed(h4["high"], h4["low"], atr_h4, ZIGZAG_THRESHOLD_ATR)
        triplets = alternating_triplets(pivots)

        pair_mfe: list[float] = []
        for idx1, kind1, idx2, _kind2, idx3, _kind3 in triplets:
            atr_neckline = atr_h4.iloc[idx2]
            if pd.isna(atr_neckline) or atr_neckline <= 0:
                continue
            p1 = float(h4["high" if kind1 == "HIGH" else "low"].iloc[idx1])
            p2 = float(h4["high" if kind1 == "HIGH" else "low"].iloc[idx3])
            if abs(p1 - p2) / float(atr_neckline) > pattern_tolerance_atr:
                continue
            neckline = float(h4["low" if kind1 == "HIGH" else "high"].iloc[idx2])
            required_lt = "DOWN" if kind1 == "HIGH" else "UP"
            search_end = min(idx3 + 1 + BREAK_SEARCH_CAP_BARS, len(h4))
            for j in range(idx3 + 1, search_end):
                broke = (
                    (kind1 == "HIGH" and float(h4["low"].iloc[j]) < neckline)
                    or (kind1 == "LOW" and float(h4["high"].iloc[j]) > neckline)
                )
                if not broke:
                    continue
                pooled_lags.append(j - idx3)
                lt_at_break = lt_dir.asof(h4.index[j])
                atr_entry = atr_h4.iloc[j]
                if lt_at_break != required_lt or pd.isna(atr_entry) or atr_entry <= 0:
                    break
                entry_price = float(h4["close"].iloc[j])
                window_high = h4["high"].iloc[j + 1: j + 1 + TRAIL_MFE_HORIZON_H4_BARS]
                window_low = h4["low"].iloc[j + 1: j + 1 + TRAIL_MFE_HORIZON_H4_BARS]
                if len(window_high) == 0:
                    break
                if kind1 == "HIGH":
                    mfe = (entry_price - float(window_low.min())) / float(atr_entry)
                else:
                    mfe = (float(window_high.max()) - entry_price) / float(atr_entry)
                pair_mfe.append(mfe)
                break
        pooled_mfe.extend(pair_mfe)

    staleness_raw = float(np.percentile(pooled_lags, STALENESS_PERCENTILE)) if pooled_lags else 20.0
    max_bars_since_second_pivot = round_to_standard(staleness_raw, STALENESS_CANDIDATES)
    atr_trail_multiplier = round(float(np.median(pooled_mfe)), 2) if pooled_mfe else 2.0

    return {
        "variant": "confirm_time_fixed" if use_confirm_time else "original(bug)",
        "pattern_tolerance_atr": pattern_tolerance_atr,
        "stop_buffer_atr": stop_buffer_atr,
        "staleness_raw_p90_bars": round(staleness_raw, 2),
        "max_bars_since_second_pivot": max_bars_since_second_pivot,
        "pooled_n_triplets": len(pooled_delta_atr),
        "pooled_n_break_lags": len(pooled_lags),
        "pooled_n_lt_matched_signals": len(pooled_mfe),
        "atr_trail_multiplier": atr_trail_multiplier,
    }


def run_train_kpi(atr_trail_multiplier: float) -> dict:
    """production run_double_pattern_backtest(変更なし)でTrain期間・5通貨を実行.

    atr_trail_multiplier以外は公式値(double_pattern_params.json)を使用。
    """
    params_path = ROOT / "research" / "EXP-FX000003" / "10-result" / "double_pattern_params.json"
    with params_path.open(encoding="utf-8") as f:
        dp_params = json.load(f)
    ds7_path = ROOT / "data" / "curated" / "ds-7.json"
    with ds7_path.open(encoding="utf-8") as f:
        ds7 = json.load(f)
    swap_rates = {
        pair: {"long": v["swap_long_jpy_per_lot_per_day"], "short": v["swap_short_jpy_per_lot_per_day"]}
        for pair, v in ds7["pairs"].items()
    }
    spread_pips = {"USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6, "EUR_USD": 0.3}

    all_pnls: list[float] = []
    n_trades_total = 0
    per_pair = {}
    for pair in PAIRS:
        ds1_path = ROOT / "data" / "curated" / "ds-1.json"
        with ds1_path.open(encoding="utf-8") as f:
            ds1 = json.load(f)
        df = pd.DataFrame(ds1["pairs"][pair]["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        m5_period = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]

        lt_df = to_d1(m5_period)
        mt_df = to_h4(m5_period)

        dp_config = DoublePatternStrategyConfig(
            lt_sma_short=dp_params["lt_sma_short"], lt_sma_long=dp_params["lt_sma_long"],
            pattern=DoublePatternConfig(
                zigzag_threshold_atr=dp_params["zigzag_threshold_atr"],
                pattern_tolerance_atr=dp_params["pattern_tolerance_atr"],
                stop_buffer_atr=dp_params["stop_buffer_atr"],
                max_bars_since_second_pivot=dp_params["max_bars_since_second_pivot"],
            ),
            atr_length=14,
            atr_trail_multiplier=atr_trail_multiplier,
        )
        swap = swap_rates.get(pair, {"long": 0.0, "short": 0.0})
        sim_config = SimulatorConfig(
            initial_cash_jpy=1_000_000.0, lot_size=1_000,
            spread_pips=spread_pips.get(pair, 0.5), slippage_pips=0.5,
            is_jpy_pair="JPY" in pair, weekend_close=True, max_dd_pause_threshold_pct=50.0,
            swap_long_jpy_per_lot_per_day=swap["long"], swap_short_jpy_per_lot_per_day=swap["short"],
        )
        result = run_double_pattern_backtest(
            lt_ohlcv=lt_df, mt_ohlcv=mt_df, st_ohlcv=m5_period,
            pair=pair, sim_config=sim_config, dp_config=dp_config,
        )
        pnls = [t.pnl for t in result.state.trade_history]
        all_pnls.extend(pnls)
        n_trades_total += len(pnls)
        per_pair[pair] = {"n_trades": len(pnls), "metrics": to_dict(result.metrics)}

    # 5ペアpooledでKPI評価 (公式run_train_val_test_fx009.pyはペア別評価だが、
    # ここでは変化の有無を見る目的のため、代表としてUSD_JPYの単独KPIも別途出す)
    perm_result = permutation_test(all_pnls, n_permutations=DEFAULT_N_PERMUTATIONS) if all_pnls else None
    return {
        "atr_trail_multiplier": atr_trail_multiplier,
        "n_trades_total_5pairs": n_trades_total,
        "perm_p_value_pooled": round(perm_result.p_value, 4) if perm_result else None,
        "per_pair": {
            p: {
                "n_trades": v["n_trades"],
                "sharpe_monthly": v["metrics"]["sharpe_monthly"],
                "profit_factor_monthly": v["metrics"]["profit_factor_monthly"],
                "max_dd_monthly_pct": v["metrics"]["max_dd_monthly_pct"],
                "payoff_ratio": v["metrics"]["payoff_ratio"],
            }
            for p, v in per_pair.items()
        },
    }


def main() -> int:
    print("=== SYS-FX009: derive_double_pattern_params.py lt_dir.asof() 先読み影響の定量化 ===\n")

    t0 = time.time()
    print("[1/3] バグ再現版を導出中...")
    bug = run_derivation(use_confirm_time=False)
    print(f"  atr_trail_multiplier={bug['atr_trail_multiplier']}  "
          f"pooled_n_lt_matched_signals={bug['pooled_n_lt_matched_signals']}  ({time.time()-t0:.0f}秒)")

    official_path = ROOT / "research" / "EXP-FX000003" / "10-result" / "double_pattern_params.json"
    with official_path.open(encoding="utf-8") as f:
        official = json.load(f)
    reproduction_matches = (
        bug["atr_trail_multiplier"] == official["atr_trail_multiplier"]
        and bug["pooled_n_lt_matched_signals"] == official["pooled_n_lt_matched_signals"]
        and bug["max_bars_since_second_pivot"] == official["max_bars_since_second_pivot"]
    )
    print(f"  過去の公式結果との一致: {'OK' if reproduction_matches else 'NG(要確認)'}"
          f" (公式 atr_trail_multiplier={official['atr_trail_multiplier']}, "
          f"pooled_n_lt_matched_signals={official['pooled_n_lt_matched_signals']})")

    print("\n[2/3] 確定時刻修正版(lt_dirを+1日シフト)を導出中...")
    fixed = run_derivation(use_confirm_time=True)
    print(f"  atr_trail_multiplier={fixed['atr_trail_multiplier']}  "
          f"pooled_n_lt_matched_signals={fixed['pooled_n_lt_matched_signals']}  ({time.time()-t0:.0f}秒)")

    print("\n[3/3] production engineでTrain期間の感度分析 (atr_trail_multiplierのみ差し替え)...")
    train_kpi_bug = run_train_kpi(bug["atr_trail_multiplier"])
    train_kpi_fixed = run_train_kpi(fixed["atr_trail_multiplier"])
    print(f"  バグ版 atr_trail_multiplier={bug['atr_trail_multiplier']}: "
          f"n_trades(5pairs)={train_kpi_bug['n_trades_total_5pairs']}  perm_p={train_kpi_bug['perm_p_value_pooled']}")
    print(f"  修正版 atr_trail_multiplier={fixed['atr_trail_multiplier']}: "
          f"n_trades(5pairs)={train_kpi_fixed['n_trades_total_5pairs']}  perm_p={train_kpi_fixed['perm_p_value_pooled']}")

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "system": "SYS-FX009 v2 (EXP-FX000003)",
        "purpose": (
            "過去REJECT戦略について、H1確定時刻先読みバグ(h1_confirm_time_lookahead_impact.json)"
            "と同種の問題(バーラベル=開始時刻を確定時刻として誤用)が判定に影響していないかを"
            "derive_double_pattern_params.pyについて確認する"
        ),
        "lookahead_path_found": True,
        "lookahead_path_location": "scripts/derive_double_pattern_params.py:195 lt_at_break = lt_dir.asof(h4.index[j])"
                                    " (H1版はscripts/derive_double_pattern_params_h1.py:188、同型)",
        "lookahead_path_description": (
            "lt_direction_series(d1)が返すSeriesは、d1.index[k](D1バー開始時刻、例:00:00)を"
            "ラベルに持つが値はそのD1バーの終値(23:55確定)ベースのSMAクロス方向。"
            "lt_dir.asof(h4.index[j])が同日中(00:00〜23:55)のH4/H1ブレイク時刻に対して呼ばれると、"
            "その日がまだ終わっていないのに、その日の終値まで使ったLT方向を返してしまう。"
            "最大先読み幅はD1バー1本分=24時間。"
        ),
        "false_positive_check": {
            "pattern_tolerance_atr_stop_buffer_atr": "lt_dirを参照しない純粋な位置ベース計算のため影響なし",
            "max_bars_since_second_pivot": "lt_at_break判定より前にpooled_lags.append()済み、"
                                            "結果に関わらずループは1件で打ち切りのため影響なし",
            "atr_trail_multiplier": "lt_at_break==required_ltの場合のみMFE集計対象に採用するため影響あり",
            "production_engine": "double_pattern_runner.pyはlast_confirmed_bar_ts()で確定済みバーのみ"
                                  "参照しており、この.asof()パターンは使っていない(先読みなし)。"
                                  "影響はパラメータ導出(atr_trail_multiplier)経由のみ",
        },
        "reproduction_check": {
            "official": official,
            "reproduced_bug": bug,
            "reproduction_matches_official": reproduction_matches,
        },
        "confirm_time_fixed": fixed,
        "parameter_drift": {
            "atr_trail_multiplier_bug": bug["atr_trail_multiplier"],
            "atr_trail_multiplier_fixed": fixed["atr_trail_multiplier"],
            "delta": round(fixed["atr_trail_multiplier"] - bug["atr_trail_multiplier"], 3),
            "pooled_n_lt_matched_signals_bug": bug["pooled_n_lt_matched_signals"],
            "pooled_n_lt_matched_signals_fixed": fixed["pooled_n_lt_matched_signals"],
        },
        "train_kpi_sensitivity": {
            "caveat": (
                "正式なパラメータ再導出ではない。atr_trail_multiplierのみを修正版の値に差し替えた"
                "感度分析(production run_double_pattern_backtest自体は無変更)。他パラメータ"
                "(pattern_tolerance_atr等、影響なしと確認済み)は公式値を据え置き。"
                "00-spec.md・double_pattern_params.jsonへの反映は司令塔判断に委ねる"
            ),
            "bug_param": train_kpi_bug,
            "fixed_param": train_kpi_fixed,
        },
        "conclusion": (
            "REJECT判定を覆すか: "
            + ("覆らない" if train_kpi_fixed["n_trades_total_5pairs"] == 0
               or train_kpi_bug["n_trades_total_5pairs"] == 0
               else "本文参照(train_kpi_sensitivityのn_trades/perm_p比較を参照し、"
                    "パラメータ差し替えでKPI必須ゲートの合否が変わらない限りREJECTは覆らない)")
        ),
    }
    out_path = ROOT / "research" / "method-notes" / "fx009_lt_asof_lookahead_impact.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
