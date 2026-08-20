"""EXP-FX000005 改善ループ第1試行: M5ダウ理論に基づく連続押し目買い版.

背景: `backtest_vol_breakout_train.py`(単発戻り確認エントリー)のTrainベースラインが
5通貨すべてマイナス(mean_R=-0.242)だったことを受け、司令塔からの設計変更案
「検知後は5分足のダウ理論に則って押し目買いを繰り返す。一度トレンドが終わったら終了」
を実装する。事前登録の詳細は`00-spec.md`§改善ループ第1試行を参照。

検出層(H1、N=3.5)は変更しない。エントリー層のみ、単発の戻り確認から、M5の
ZigZag(閾値1.0×ATR(M5)、暫定固定値)によるダウ理論スイング追跡へ変更する:
  - UP方向: 新しい安値ピボットが直前確定安値より高ければ(Higher Low)押し目買い
    エントリー、低ければ(Lower Low)トレンド終了(型崩れ)
  - 追跡起点はブレイクバー自身の安値/高値
  - 安全上限: ブレイク確定後72時間(H1継続確認による再開後も延長しない)
SL/TP構造(stop_buffer_atr_m5×ATR(M5)、SYS-FX009段階利確方式、atr_trail_multiplier
=3.23をH1バーで判定)は`backtest_vol_breakout_train.py`と同一方法論を踏襲。

## 修正1 (2026-08-20、司令塔指摘): 1通貨1ポジション制約

初版は1トレンドイベント中に確定したHigher Low(UP)/Lower High(DOWN)ピボットを
すべて独立トレードとして同時多発的にオープンする設計だった。司令塔から
「基本1通貨1ポジションに絞る想定。ポジションクローズ後、トレンド継続なら
押し目でポジションを立てるイメージ」との訂正を受け、`simulate_dow_theory_trend()`
に書き換えた: ダウ理論のスイング構造追跡(Higher Low/Lower Low判定)自体は
ポジション保有状況に関わらず継続するが、**新規エントリーは直前ポジションの
決済(exit_time)後にのみ許可する**。決済前に確定したHigher Low/Lower Highは
機会として見送り(キューイングしない、決済後に新たに形成される押し目を待つ)。
簡略化(正直に明記): 1トレンドイベント内の1通貨1ポジションは保証するが、
同一通貨で**別のトレンドイベント**が後続して開始した場合の重複は保証しない
(実務上はポジション保有期間が短いため稀だが、完全な排他制御ではない)。

## 修正2 (2026-08-20、司令塔追加指示): M5型崩れ後のH1継続確認による再開

司令塔「5分足でトレンドが終わったあとに、1時間足のトレンド継続を確認できた
場合は、その後、5分足の押し目買いを継続してください」を受け追加。M5でLower
Low(UP)/Higher High(DOWN)が確定しても、即座にトレンド終了とはせず「H1継続
確認待ち」状態に入る:
  - H1継続確認の定義(司令塔確認済み、選択肢「H1が型崩れ前の高値を更新」):
    型崩れ以降のH1バーの高値(UP方向)/安値(DOWN方向)が、当該トレンド開始
    (ブレイクバー)からM5型崩れ時点までのH1高値/安値を更新したら継続確認
  - 確認できたら、M5ダウ理論追跡を再開する。再開時の基準安値/高値
    (last_confirmed_extreme)は、型崩れ確認待ち中にM5で記録した最安値/最高値
    (押し目の実際の底/天井)にリセットする
  - 確認前に安全上限(ブレイク確定後72時間、再開しても延長しない)に達するか
    週末クローズになった場合はそのままトレンド終了として扱う
  - 1通貨1ポジション制約(修正1)は再開後も継続して適用する

出力: research/method-notes/vol_breakout_dow_theory_train.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from derive_vol_breakout_entry_params import N_BREAKOUT, load_m5, to_h1, PAIRS as ALL_PAIRS
from minmax_fx_dt.backtest.permutation import permutation_test_clustered
from minmax_fx_dt.decision.criteria import compute_n_trades_effective
from minmax_fx_dt.strategy.indicators import atr as atr_ind

TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"

# 事前登録 (00-spec.md §改善ループ第1試行、結果を見る前に固定)
ZIGZAG_THRESHOLD_ATR_M5 = 1.0
MAX_TREND_HOURS = 72
WINDOW_START_MIN = 30
BUFFER_PERCENTILE = 25
TP_LEVELS = [(1.0, 0.40), (2.0, 0.35), (3.0, 0.25)]
MAX_HOLD_BARS = 24 * 10  # H1、10日相当(暴走ループ防止の安全上限)

with (ROOT / "research" / "EXP-FX000005" / "10-result" / "vol_breakout_entry_params.json").open(encoding="utf-8") as f:
    H1_PARAMS = json.load(f)
ATR_TRAIL_MULTIPLIER = H1_PARAMS["atr_trail_multiplier"]


def is_weekend_close_time(ts: pd.Timestamp) -> bool:
    return ts.weekday() == 5 and ts.hour >= 6


def simulate_scaled_scheme(h1: pd.DataFrame, atr_h1: pd.Series, entry: dict, trail_mult: float) -> dict:
    """段階利確(40/35/25%)、1R到達後BE+ATRトレーリング。exit_timeを含めて返す
    (ポジション保有中判定=1通貨1ポジション制約のゲーティングに必要なため、
    `analyze_scaled_exit_diagnostic.simulate_scaled_scheme`をexit_time付きで
    ローカルに複製したもの)。"""
    direction = entry["direction"]
    entry_price = entry["entry_price"]
    risk = entry["initial_risk"]
    stop = entry["stop0"]
    levels = [(r, frac, entry_price + r * risk if direction == "UP" else entry_price - r * risk, False)
              for r, frac in TP_LEVELS]
    remaining_fraction = 1.0
    realized_r = 0.0
    be_moved = False
    n = len(h1)
    start = entry["entry_idx"] + 1
    end = min(n, start + MAX_HOLD_BARS)
    for i in range(start, end):
        ts = h1.index[i]
        o, h, low, c = float(h1["open"].iloc[i]), float(h1["high"].iloc[i]), float(h1["low"].iloc[i]), float(h1["close"].iloc[i])
        n_levels_hit = sum(1 for lv in levels if lv[3])
        if is_weekend_close_time(ts):
            exit_r = (c - entry_price) / risk if direction == "UP" else (entry_price - c) / risk
            reason = "WEEKEND_NO_TP" if n_levels_hit == 0 else "TP_THEN_WEEKEND"
            return {"r": realized_r + remaining_fraction * exit_r, "exit_reason": reason,
                    "n_levels_hit": n_levels_hit, "exit_time": ts}
        stop_hit = (low <= stop) if direction == "UP" else (h >= stop)
        if stop_hit:
            exit_r = (stop - entry_price) / risk if direction == "UP" else (entry_price - stop) / risk
            reason = "SL_INITIAL_NO_TP" if n_levels_hit == 0 else "TP_THEN_SL_TRAIL"
            return {"r": realized_r + remaining_fraction * exit_r, "exit_reason": reason,
                    "n_levels_hit": n_levels_hit, "exit_time": ts}
        for idx_lv, (r_level, frac, price_level, hit) in enumerate(levels):
            if hit or remaining_fraction <= 0:
                continue
            reached = (h >= price_level) if direction == "UP" else (low <= price_level)
            if reached:
                realized_r += frac * r_level
                remaining_fraction -= frac
                levels[idx_lv] = (r_level, frac, price_level, True)
                if not be_moved:
                    stop = max(stop, entry_price) if direction == "UP" else min(stop, entry_price)
                    be_moved = True
        if be_moved and remaining_fraction > 0:
            atr_i = atr_h1.asof(ts)
            if pd.notna(atr_i) and atr_i > 0:
                if direction == "UP":
                    stop = max(stop, o - trail_mult * float(atr_i))
                else:
                    stop = min(stop, o + trail_mult * float(atr_i))
        if remaining_fraction <= 1e-9:
            return {"r": realized_r, "exit_reason": "TP_FULL", "n_levels_hit": 3, "exit_time": ts}
    ts_last = h1.index[end - 1]
    c = float(h1["close"].iloc[end - 1])
    exit_r = (c - entry_price) / risk if direction == "UP" else (entry_price - c) / risk
    n_levels_hit = sum(1 for lv in levels if lv[3])
    return {"r": realized_r + remaining_fraction * exit_r, "exit_reason": "MAX_HOLD",
            "n_levels_hit": n_levels_hit, "exit_time": ts_last}


def simulate_dow_theory_trend(m5: pd.DataFrame, atr_m5: pd.Series, h1: pd.DataFrame, atr_h1: pd.Series,
                               break_idx: int, direction: str, stop_buffer_atr_m5: float,
                               trail_mult: float, blackout_check=None,
                               zigzag_threshold_atr_m5: float | None = None) -> list[dict]:
    """1トレンドイベントをM5ダウ理論で追跡し、1通貨1ポジション制約下で連続的に
    押し目買い/戻り売りをシミュレートする。M5で型崩れしてもH1が型崩れ前の高値
    (UP)/安値(DOWN)を更新したら追跡を再開する。

    blackout_check: Optional[Callable[[pd.Timestamp], bool]]。Trueを返す時刻は
    新規エントリーを見送る(経済指標カレンダーのブラックアウト窓フィルター用、
    2026-08-20追加)。ダウ理論のスイング構造追跡自体は影響を受けない(型崩れ判定は
    ブラックアウト中も継続する)。

    zigzag_threshold_atr_m5: 省略時はモジュール定数ZIGZAG_THRESHOLD_ATR_M5(=1.0、
    事前登録の暫定値)を使用。エントリー機会の感度分析(改善ループ第4試行)用に
    呼び出し側から上書きできるよう追加(2026-08-20)。"""
    zz_thresh = zigzag_threshold_atr_m5 if zigzag_threshold_atr_m5 is not None else ZIGZAG_THRESHOLD_ATR_M5
    break_bar = h1.iloc[break_idx]
    break_time = h1.index[break_idx]
    start_time = break_time + pd.Timedelta(minutes=WINDOW_START_MIN)
    end_time = break_time + pd.Timedelta(hours=MAX_TREND_HOURS)

    start_pos = m5.index.searchsorted(start_time, side="right")
    end_pos = m5.index.searchsorted(end_time, side="right")
    if start_pos >= len(m5) or start_pos >= end_pos:
        return []

    trades: list[dict] = []
    last_confirmed_extreme = float(break_bar["low"]) if direction == "UP" else float(break_bar["high"])
    state = "SEARCHING_HIGH" if direction == "UP" else "SEARCHING_LOW"
    running_extreme = float(m5["high"].iloc[start_pos]) if direction == "UP" else float(m5["low"].iloc[start_pos])
    running_extreme_idx = start_pos
    position_open_until: pd.Timestamp | None = None

    # H1継続確認まわりの状態
    h1_extreme_since_break = float(break_bar["high"]) if direction == "UP" else float(break_bar["low"])
    last_h1_pos_checked = break_idx
    awaiting_h1_confirm = False
    h1_confirm_threshold: float | None = None
    pause_extreme: float | None = None  # 一時停止中に記録したM5最安値(UP)/最高値(DOWN)

    for i in range(start_pos, end_pos):
        ts = m5.index[i]
        if is_weekend_close_time(ts):
            break
        h_i, l_i, c_i = float(m5["high"].iloc[i]), float(m5["low"].iloc[i]), float(m5["close"].iloc[i])

        # H1バーの確定を検知し、継続確認 & h1_extreme_since_breakを更新
        cur_h1_pos = int(h1.index.searchsorted(ts, side="right") - 1)
        if cur_h1_pos > last_h1_pos_checked:
            for hp in range(last_h1_pos_checked + 1, cur_h1_pos + 1):
                h1_high_hp = float(h1["high"].iloc[hp])
                h1_low_hp = float(h1["low"].iloc[hp])
                if direction == "UP":
                    if awaiting_h1_confirm and h1_high_hp > h1_confirm_threshold:
                        awaiting_h1_confirm = False
                        state = "SEARCHING_HIGH"
                        last_confirmed_extreme = pause_extreme if pause_extreme is not None else last_confirmed_extreme
                        running_extreme, running_extreme_idx = h_i, i
                    h1_extreme_since_break = max(h1_extreme_since_break, h1_high_hp)
                else:
                    if awaiting_h1_confirm and h1_low_hp < h1_confirm_threshold:
                        awaiting_h1_confirm = False
                        state = "SEARCHING_LOW"
                        last_confirmed_extreme = pause_extreme if pause_extreme is not None else last_confirmed_extreme
                        running_extreme, running_extreme_idx = l_i, i
                    h1_extreme_since_break = min(h1_extreme_since_break, h1_low_hp)
            last_h1_pos_checked = cur_h1_pos

        if awaiting_h1_confirm:
            if direction == "UP":
                pause_extreme = l_i if pause_extreme is None else min(pause_extreme, l_i)
            else:
                pause_extreme = h_i if pause_extreme is None else max(pause_extreme, h_i)
            continue

        atr_i = atr_m5.iloc[i]
        if pd.isna(atr_i) or atr_i <= 0:
            continue
        thresh = zz_thresh * float(atr_i)

        if direction == "UP":
            if state == "SEARCHING_HIGH":
                if h_i > running_extreme:
                    running_extreme, running_extreme_idx = h_i, i
                if running_extreme - l_i >= thresh:
                    state = "SEARCHING_LOW"
                    running_extreme, running_extreme_idx = l_i, i
            else:  # SEARCHING_LOW
                if l_i < running_extreme:
                    running_extreme, running_extreme_idx = l_i, i
                if h_i - running_extreme >= thresh:
                    pivot_low = running_extreme
                    if pivot_low > last_confirmed_extreme:
                        if (position_open_until is None or ts > position_open_until) and not (blackout_check and blackout_check(ts)):
                            buffer = stop_buffer_atr_m5 * float(atr_i)
                            stop0 = pivot_low - buffer
                            entry_price = c_i
                            initial_risk = abs(entry_price - stop0)
                            entry_h1_idx = int(h1.index.searchsorted(ts, side="right") - 1)
                            if initial_risk > 0 and 0 <= entry_h1_idx < len(h1):
                                entry = dict(direction=direction, entry_idx=entry_h1_idx, entry_price=entry_price,
                                             stop0=stop0, initial_risk=initial_risk)
                                res = simulate_scaled_scheme(h1, atr_h1, entry, trail_mult)
                                res["entry_time"] = str(ts)
                                res["direction"] = direction
                                res["entry_price"] = entry_price
                                res["initial_risk"] = initial_risk
                                trades.append(res)
                                position_open_until = res["exit_time"]
                        last_confirmed_extreme = pivot_low
                        state = "SEARCHING_HIGH"
                        running_extreme, running_extreme_idx = h_i, i
                    else:
                        awaiting_h1_confirm = True
                        h1_confirm_threshold = h1_extreme_since_break
                        pause_extreme = min(pivot_low, l_i)
        else:  # DOWN
            if state == "SEARCHING_LOW":
                if l_i < running_extreme:
                    running_extreme, running_extreme_idx = l_i, i
                if h_i - running_extreme >= thresh:
                    state = "SEARCHING_HIGH"
                    running_extreme, running_extreme_idx = h_i, i
            else:  # SEARCHING_HIGH
                if h_i > running_extreme:
                    running_extreme, running_extreme_idx = h_i, i
                if running_extreme - l_i >= thresh:
                    pivot_high = running_extreme
                    if pivot_high < last_confirmed_extreme:
                        if (position_open_until is None or ts > position_open_until) and not (blackout_check and blackout_check(ts)):
                            buffer = stop_buffer_atr_m5 * float(atr_i)
                            stop0 = pivot_high + buffer
                            entry_price = c_i
                            initial_risk = abs(entry_price - stop0)
                            entry_h1_idx = int(h1.index.searchsorted(ts, side="right") - 1)
                            if initial_risk > 0 and 0 <= entry_h1_idx < len(h1):
                                entry = dict(direction=direction, entry_idx=entry_h1_idx, entry_price=entry_price,
                                             stop0=stop0, initial_risk=initial_risk)
                                res = simulate_scaled_scheme(h1, atr_h1, entry, trail_mult)
                                res["entry_time"] = str(ts)
                                res["direction"] = direction
                                res["entry_price"] = entry_price
                                res["initial_risk"] = initial_risk
                                trades.append(res)
                                position_open_until = res["exit_time"]
                        last_confirmed_extreme = pivot_high
                        state = "SEARCHING_LOW"
                        running_extreme, running_extreme_idx = l_i, i
                    else:
                        awaiting_h1_confirm = True
                        h1_confirm_threshold = h1_extreme_since_break
                        pause_extreme = max(pivot_high, h_i)

    return trades


def main() -> int:
    print("=== EXP-FX000005 改善ループ第1試行: M5ダウ理論連続押し目買い Trainベースライン ===\n")
    print(f"事前登録: zigzag_threshold_atr_m5={ZIGZAG_THRESHOLD_ATR_M5}, "
          f"max_trend_hours={MAX_TREND_HOURS}, atr_trail_multiplier(H1流用)={ATR_TRAIL_MULTIPLIER}\n"
          f"1通貨1ポジション制約 + M5型崩れ後のH1継続確認による再開ロジックを適用\n")

    pooled_bar_range_atr_m5: list[float] = []
    h1_cache: dict[str, pd.DataFrame] = {}
    atr_h1_cache: dict[str, pd.Series] = {}
    m5_cache: dict[str, pd.DataFrame] = {}
    atr_m5_cache: dict[str, pd.Series] = {}
    break_events: dict[str, list[tuple[int, str]]] = {}

    for pair in ALL_PAIRS:
        m5 = load_m5(pair)
        h1 = to_h1(m5)
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)
        h1_cache[pair], atr_h1_cache[pair] = h1, atr_h1
        m5_cache[pair], atr_m5_cache[pair] = m5, atr_m5

        bar_range_atr_m5 = ((m5["high"] - m5["low"]) / atr_m5).replace([np.inf, -np.inf], np.nan).dropna()
        pooled_bar_range_atr_m5.extend(bar_range_atr_m5.tolist())

        ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
        idxs = np.where(ratio.values >= N_BREAKOUT)[0]
        events = []
        for i in idxs:
            pos = h1.index.get_loc(ratio.index[i])
            bar = h1.iloc[pos]
            direction = "UP" if bar["close"] > bar["open"] else "DOWN"
            events.append((pos, direction))
        break_events[pair] = events

    stop_buffer_atr_m5 = round(float(np.percentile(pooled_bar_range_atr_m5, BUFFER_PERCENTILE)), 3)
    print(f"stop_buffer_atr_m5: pooled M5バーレンジ/ATR比 n={len(pooled_bar_range_atr_m5)}件の"
          f"p{BUFFER_PERCENTILE} = {stop_buffer_atr_m5}\n")

    all_results: list[dict] = []
    trades_per_currency: dict[str, int] = {}
    n_trend_events = 0

    for pair in ALL_PAIRS:
        h1, atr_h1 = h1_cache[pair], atr_h1_cache[pair]
        m5, atr_m5 = m5_cache[pair], atr_m5_cache[pair]
        pair_results = []
        for break_idx, direction in break_events[pair]:
            n_trend_events += 1
            trades = simulate_dow_theory_trend(m5, atr_m5, h1, atr_h1, break_idx, direction,
                                                stop_buffer_atr_m5, ATR_TRAIL_MULTIPLIER)
            for t in trades:
                t["pair"] = pair
            pair_results.extend(trades)
        all_results.extend(pair_results)
        trades_per_currency[pair] = len(pair_results)
        mean_r = float(np.mean([r["r"] for r in pair_results])) if pair_results else None
        print(f"[{pair}] トレンドイベント={len(break_events[pair])}件  押し目買いトレード={len(pair_results)}件" +
              (f"  mean_R={mean_r:.4f}" if pair_results else ""))

    n_total = len(all_results)
    print(f"\n全体: トレンドイベント={n_trend_events}件  トレード={n_total}件  "
          f"平均トレード/イベント={n_total / n_trend_events:.2f}" if n_trend_events else "")

    if n_total == 0:
        print("\n有効トレードなし")
        rs, exit_reason_counts, n_eff, perm_p = [], {}, 0.0, None
        profit_factor_val = payoff_val = None
    else:
        rs = [r["r"] for r in all_results]
        exit_reason_counts = {}
        for r in all_results:
            exit_reason_counts[r["exit_reason"]] = exit_reason_counts.get(r["exit_reason"], 0) + 1
        print(f"\nプール(5通貨) n={n_total}  mean_R={np.mean(rs):.4f}  win_rate={np.mean([r>0 for r in rs]):.3f}")
        print(f"イグジット内訳: {exit_reason_counts}")

        n_eff = compute_n_trades_effective(trades_per_currency, n_total)
        pairs_flat = [r["pair"] for r in all_results]
        perm = permutation_test_clustered(rs, pairs_flat, n_permutations=20000, seed=42) if n_total >= 4 else None
        perm_p = perm.p_value if perm else None
        print(f"実効トレード数: {n_eff:.1f}  min_n_trades(>=300)={'PASS' if n_eff >= 300 else 'FAIL'}  "
              f"permutation_p(clustered)={perm_p}")

        wins = [r for r in rs if r > 0]
        losses = [r for r in rs if r < 0]
        profit_factor_val = (sum(wins) / abs(sum(losses))) if losses else None
        payoff_val = (float(np.mean(wins)) / abs(float(np.mean(losses)))) if wins and losses else None
        print(f"PF(Rベース)={profit_factor_val:.3f}" if profit_factor_val else "PF: 算出不可")
        print(f"ペイオフ比(Rベース)={payoff_val:.3f}" if payoff_val else "ペイオフ比: 算出不可")

    out_path = ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_train.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "params": {"n_breakout": N_BREAKOUT, "zigzag_threshold_atr_m5": ZIGZAG_THRESHOLD_ATR_M5,
                       "max_trend_hours": MAX_TREND_HOURS, "stop_buffer_atr_m5": stop_buffer_atr_m5,
                       "atr_trail_multiplier": ATR_TRAIL_MULTIPLIER},
            "n_trend_events": n_trend_events,
            "n_trades_total": n_total,
            "trades_per_currency": trades_per_currency,
            "mean_r": round(float(np.mean(rs)), 4) if rs else None,
            "win_rate": round(float(np.mean([r > 0 for r in rs])), 3) if rs else None,
            "profit_factor_r": round(profit_factor_val, 3) if profit_factor_val else None,
            "payoff_ratio_r": round(payoff_val, 3) if payoff_val else None,
            "exit_reason_counts": exit_reason_counts,
            "n_trades_effective": n_eff,
            "min_n_trades_pass": n_eff >= 300 if rs else False,
            "permutation_p_clustered": perm_p,
            "_note": (
                "改善ループ第1試行(00-spec.md参照)。M5ダウ理論スイングによる連続押し目買い版。"
                "1通貨1ポジション制約 + M5型崩れ後のH1継続確認による再開ロジックを適用(2026-08-20修正)。"
                "検出層(H1,N=3.5)・SL/TP構造(段階利確)は単発版と同一、エントリー層のみ変更。"
            ),
            "trades": all_results,
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
