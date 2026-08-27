"""ニュース窓ボラティリティ戦略の探索的検証: 「避ける」から「刈る」への転換
(Train期間のみ・探索的診断).

## 背景・動機

司令塔との対話「エッジとは幻。エッジが無いことを前提にした戦略が大事」を受け、
未検証の戦略カテゴリを棚卸しした結果、**ニュース窓(BOJ/FOMC会合)を新規エントリー
禁止(ブラックアウト)にしか使っておらず、値幅そのものを収益源にする設計を一度も
試していない**と判明した。

根拠となる既存データ: `research/method-notes/missed_entry_opportunities.json`
(2026-08-20生成)を再集計すると、ブラックアウトで見送られたイベント(n=81)は
継続方向(ブレイクバー自身の方向)基準で **MFE中央値6.62R・MAE中央値5.26R・
ネット方向中央値+0.81R(プラス率51.9%)**。順行にも逆行にも巨大に動く一方、
方向はほぼコインフリップ。「方向は読めないが値幅は確実に出る」という、
まさに"エッジ不要"設計が刺さる形をしている。

## 検証したい命題

SYS-FX011の凍結済み最良候補(候補①、T-13トレール専業版)のエントリー・出口・
コストモデルを**一切変更せず**、唯一の違いとして「BOJ/FOMCブラックアウト窓
(既存`economic_calendar.py`と同一定義)を新規エントリー禁止からむしろ対象に
含める」場合、窓内イベントは窓外イベントと比べて成績がどう違うか。

- 既存の`blackout_check`は「Trueを返す時刻は新規エントリーを見送る」実装。
  本スクリプトは`blackout_check=None`(フィルターなし)で全イベントを追跡し、
  各イベントのブレイク時刻が窓内/窓外どちらかで事後的に層別する
  (フィルターとして機能させるのではなく、比較のための層別)。
- 検出層(N_BREAKOUT=3.5)・エントリー層(M5ダウ理論連続追跡+H1継続確認再開)・
  出口(トレール専業、breakeven_trigger_r=1.0)・コストモデルは全て
  `backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd.py`の
  確定値をそのままimportして使う(新規パラメータの導入・調整は一切しない)。

## 位置づけ(HARKing防止のため明記)

- 本スクリプトは**正式な検証プロトコル外の探索的診断**であり、`00-spec.md`等の
  事前登録文書は一切編集しない。
- 使用期間は**Train (2023-11-01〜2025-03-31)のみ**。Validation/Testは正式検証の
  ために温存する。
- BOJ/FOMC会合日程は既存`economic_calendar.py`のリストをそのまま使う(先読みなし、
  会合日程は事前に公表されている)。

出力: research/method-notes/calendar_window_volatility_harvest.json
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
import pandas as pd

from backtest_vol_breakout_dow_theory import (  # noqa: E402
    select_non_overlapping_breakout_events, simulate_dow_theory_trend,
)
from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402
from backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd import (  # noqa: E402
    ATR_TRAIL_MULTIPLIER_M5, BREAKEVEN_TRIGGER_R, COMMISSION_RATE_ROUND_TRIP,
    SLIPPAGE_PIPS_MARKET_LEG, SLIPPAGE_PIPS_STOP_TRIGGERED, SPREAD_PIPS,
    STOP_BUFFER_ATR_M5, TP_CUM_FRACTION, TP_LEVELS_TRAILONLY, pip_size,
)
from derive_vol_breakout_entry_params import N_BREAKOUT, to_h1  # noqa: E402
from economic_calendar import BOJ_MEETINGS, FOMC_MEETINGS, build_blackout_windows  # noqa: E402
from minmax_fx_dt.backtest.permutation import permutation_test_clustered  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"


def load_m5_period(pair: str, start: str, end: str) -> pd.DataFrame:
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= start) & (df.index <= end)]


def in_any_window(t: pd.Timestamp, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> bool:
    t = t.tz_localize("Asia/Tokyo") if t.tzinfo is None else t
    return any(w0 <= t <= w1 for w0, w1 in windows)


def which_source(t: pd.Timestamp, boj_windows, fomc_windows) -> str:
    t = t.tz_localize("Asia/Tokyo") if t.tzinfo is None else t
    in_boj = any(w0 <= t <= w1 for w0, w1 in boj_windows)
    in_fomc = any(w0 <= t <= w1 for w0, w1 in fomc_windows)
    if in_boj and in_fomc:
        return "both"
    if in_boj:
        return "boj_only"
    if in_fomc:
        return "fomc_only"
    return "outside_window"


def cost_for_trade(sim: dict, pair: str) -> dict:
    """v7トレール専業版と同一のコストモデル(T-09確定版)を適用."""
    spread = SPREAD_PIPS.get(pair, 0.5)
    pip = pip_size(pair)
    fraction_via_tp = TP_CUM_FRACTION[sim["n_levels_hit"]]
    fraction_remaining = 1.0 - fraction_via_tp
    remaining_is_market = sim["exit_reason"] in ("WEEKEND_NO_TP", "TP_THEN_WEEKEND", "MAX_HOLD")
    remaining_is_stop_triggered = sim["exit_reason"] in ("SL_INITIAL_NO_TP", "TP_THEN_SL_TRAIL")
    entry_pips = spread + SLIPPAGE_PIPS_MARKET_LEG
    if remaining_is_market:
        exit_slippage = fraction_remaining * SLIPPAGE_PIPS_MARKET_LEG
    elif remaining_is_stop_triggered:
        exit_slippage = fraction_remaining * SLIPPAGE_PIPS_STOP_TRIGGERED
    else:
        exit_slippage = 0.0
    exit_pips = spread + exit_slippage
    cost_price = (entry_pips + exit_pips) * pip
    cost_r = cost_price / sim["initial_risk"]
    leverage_ratio = sim["entry_price"] / sim["initial_risk"]
    commission_r = COMMISSION_RATE_ROUND_TRIP * leverage_ratio
    r_net = sim["r"] - cost_r - commission_r
    return {"cost_r": cost_r, "commission_r": commission_r, "r_net": r_net}


def main() -> int:
    print("=== ニュース窓ボラティリティ戦略: BOJ/FOMC窓 内 vs 外 の層別比較 (Train期間のみ・探索的) ===\n")

    windows = build_blackout_windows()  # 既存定義そのまま(BOJ+FOMC、前後24hバッファ)
    boj_windows = build_blackout_windows(meetings=BOJ_MEETINGS)
    fomc_windows = build_blackout_windows(meetings=FOMC_MEETINGS)
    print(f"BOJ会合 {len(BOJ_MEETINGS)}回 + FOMC会合 {len(FOMC_MEETINGS)}回 "
          f"= ブラックアウト窓 {len(windows)}個(各[開催前24h, 開催後24h+24hバッファ])\n")

    trades_by_group: dict[str, list[dict]] = {"inside_window": [], "outside_window": []}
    trades_by_source: dict[str, list[dict]] = {"boj_only": [], "fomc_only": [], "both": [], "outside_window": []}
    trades_by_pair_group: dict[str, list[dict]] = {f"{p}_{g}": [] for p in SELECTED_PAIRS for g in ("inside", "outside")}
    n_events_by_group = {"inside_window": 0, "outside_window": 0}

    for pair in SELECTED_PAIRS:
        m5 = load_m5_period(pair, TRAIN_START, TRAIN_END)
        h1 = to_h1(m5)
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)
        ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
        idxs = np.where(ratio.values >= N_BREAKOUT)[0]
        positions = [h1.index.get_loc(ratio.index[i]) for i in idxs]
        directions = ["UP" if h1.iloc[pos]["close"] > h1.iloc[pos]["open"] else "DOWN" for pos in positions]
        dedup_positions = select_non_overlapping_breakout_events(h1.index, positions, directions)
        dedup_directions = {pos: d for pos, d in zip(positions, directions)}

        for pos in dedup_positions:
            break_time = h1.index[pos]
            group = "inside_window" if in_any_window(break_time, windows) else "outside_window"
            n_events_by_group[group] += 1
            direction = dedup_directions[pos]
            sims = simulate_dow_theory_trend(
                m5, atr_m5, h1, atr_h1, pos, direction,
                STOP_BUFFER_ATR_M5, ATR_TRAIL_MULTIPLIER_M5,
                blackout_check=None,  # 本スクリプトはフィルターではなく層別比較のため常にNone
                tp_levels=TP_LEVELS_TRAILONLY, skip_first_entry=False,
                atr_trail_series=atr_m5, m5_exit=True,
                breakeven_trigger_r=BREAKEVEN_TRIGGER_R,
            )
            source = which_source(break_time, boj_windows, fomc_windows)
            for sim in sims:
                c = cost_for_trade(sim, pair)
                rec = {
                    "pair": pair, "direction": direction,
                    "entry_time": str(sim["entry_time"]), "exit_reason": sim["exit_reason"],
                    "r_gross": sim["r"], **c,
                }
                trades_by_group[group].append(rec)
                trades_by_source[source].append(rec)
                trades_by_pair_group[f"{pair}_{'inside' if group == 'inside_window' else 'outside'}"].append(rec)

    def summarize(trades: list[dict]) -> dict:
        n = len(trades)
        if n == 0:
            return {"n": 0}
        r_net = [t["r_net"] for t in trades]
        r_gross = [t["r_gross"] for t in trades]
        wins = [r for r in r_net if r > 0]
        losses = [r for r in r_net if r <= 0]
        pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None
        payoff = (float(np.mean(wins)) / abs(float(np.mean(losses)))) if wins and losses else None
        pairs_list = [t["pair"] for t in trades]
        perm = permutation_test_clustered(r_net, pairs_list, seed=42) if n >= 4 else None
        return {
            "n": n,
            "win_rate": round(len(wins) / n, 4),
            "mean_r_gross": round(float(np.mean(r_gross)), 4),
            "mean_r_net": round(float(np.mean(r_net)), 4),
            "median_r_net": round(float(np.median(r_net)), 4),
            "profit_factor": round(pf, 3) if pf else None,
            "payoff_ratio": round(payoff, 3) if payoff else None,
            "perm_p_clustered": round(perm.p_value, 4) if perm else None,
        }

    out = {
        "generated_at": datetime.now().isoformat(),
        "status": "探索的診断(正式プロトコル外・spec編集なし)",
        "question": "SYS-FX011凍結ロジックをBOJ/FOMC窓の内側に適用すると、窓外と比べて成績はどう違うか"
                    "(避ける対象から刈る対象への転換は有望か)",
        "period": {"train_start": TRAIN_START, "train_end": TRAIN_END},
        "pairs": SELECTED_PAIRS,
        "n_events_by_group": n_events_by_group,
        "definitions": {
            "windows": "既存economic_calendar.build_blackout_windows()と同一定義"
                      "(BOJ 22回+FOMC 22回、各[開催初日-24h, 開催最終日+24h+24h])",
            "entry_exit_logic": "SYS-FX011凍結最良候補(T-13トレール専業版)と完全同一。"
                                "STOP_BUFFER_ATR_M5/ATR_TRAIL_MULTIPLIER_M5/breakeven_trigger_r=1.0/"
                                "tp_levels=[]/m5_exit=Trueをそのままimport、新規パラメータなし",
            "cost_model": "T-09確定版と完全同一(往復スプレッド+SL/トレールスリッページ1.0pip+手数料)",
            "difference_from_production": "blackout_check=Noneとし、窓内イベントもエントリー対象に含めた上で、"
                                          "事後的に窓内/窓外で層別しただけ(フィルターとしては機能させていない)",
        },
        "results": {g: summarize(trades_by_group[g]) for g in trades_by_group},
        "results_by_source": {g: summarize(trades_by_source[g]) for g in trades_by_source},
        "results_by_pair": {g: summarize(trades_by_pair_group[g]) for g in trades_by_pair_group},
        "caveats": [
            "1通貨1ポジション制約・重複排除は既存ロジックのまま(SYS-FX011全体と同一)。",
            "窓の定義(前後24hバッファ)はブラックアウト用に設計されたもので、"
            "ニュース窓戦略専用に最適化されたものではない(既存資産の転用)。",
            "Train期間のみ。Validation/Testは正式検証のために温存している。",
            "perm_p_clusteredは通貨クラスタでのpermutation testだが、窓内サンプル数が"
            "少ない場合は検出力が低い点に留意。",
        ],
    }

    out_path = ROOT / "research" / "method-notes" / "calendar_window_volatility_harvest.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"イベント数: 窓内={n_events_by_group['inside_window']} / 窓外={n_events_by_group['outside_window']}\n")
    for g in trades_by_group:
        s = out["results"][g]
        print(f"[{g}] n={s.get('n')} win_rate={s.get('win_rate')} "
              f"r_gross={s.get('mean_r_gross')} r_net={s.get('mean_r_net')} "
              f"PF={s.get('profit_factor')} payoff={s.get('payoff_ratio')} "
              f"perm_p={s.get('perm_p_clustered')}")
    print("\n--- BOJのみ / FOMCのみ / 両方重複 / 窓外 ---")
    for g in trades_by_source:
        s = out["results_by_source"][g]
        print(f"[{g}] n={s.get('n')} win_rate={s.get('win_rate')} "
              f"r_net={s.get('mean_r_net')} PF={s.get('profit_factor')} payoff={s.get('payoff_ratio')}")
    print("\n--- 通貨別 窓内 vs 窓外 ---")
    for pair in SELECTED_PAIRS:
        si = out["results_by_pair"][f"{pair}_inside"]
        so = out["results_by_pair"][f"{pair}_outside"]
        print(f"[{pair}] 窓内: n={si.get('n')} r_net={si.get('mean_r_net')} PF={si.get('profit_factor')} | "
              f"窓外: n={so.get('n')} r_net={so.get('mean_r_net')} PF={so.get('profit_factor')}")
    print(f"\n出力: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
