"""司令塔提案「戻りは短期足の反発シグナルで判定・SLは底値で浅めに張る・TPは50%戻し」の
定量検証(Train期間のみ・探索的診断).

## 位置づけ(HARKing防止のため明記)

- 本スクリプトは**正式な検証プロトコル外の探索的診断**であり、`00-spec.md`等の
  事前登録文書は一切編集しない(前例: `post_breakout_trend_visual_check.json`)。
- 使用期間は**Train (2023-11-01〜2025-03-31)のみ**。
- 新規パラメータは全て本スクリプト内で明示・固定する(結果を見てからの後付け調整はしない)。

## 検証したい命題

司令塔提案の3要素をそれぞれ具体的・実行可能なルールに分解する:

1. 「戻りは短期足の反発シグナルで判定」
   → ブレイク確定後、M5の**ランニング安値/高値**(DOWN/UPそれぞれ)を追跡し、
     その安値/高値を付けたバーの高値/安値を、後続バーの終値が反対方向に
     ブレイクした時点を「反発シグナル」とする(先読みなし、シグナルバー確定後の
     次バー始値でエントリー)。
2. 「SLは底値で浅めに張る、深追いしない」
   → シグナル確定時点のランニング安値/高値から、ごくわずかなバッファ
     (0.1×ATR(M5))だけ離した位置に置く「浅めSL」と、比較用の
     「深めSL」(0.5×ATR(M5))の2パターンを両方試す。
3. 「TPは50%」
   → ブレイクバーのレンジ(高値-安値)の50%戻し水準
     (既存の`RETRACE_SPLIT=0.5`・spec の retrace_ratio 系と同じ定義)。

反発シグナルの探索窓は2パターン試す: 既存の戻り確認窓(確定後3時間、
`vol_breakout_retrace_window.json`と同じ)、および長め(24時間)。

出力: research/method-notes/counter_trend_reversal_entry.json
"""

from __future__ import annotations

import glob
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from minmax_fx_dt.strategy.indicators import atr as atr_ind

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
N_BREAKOUT = 3.5
ATR_LEN = 14
PREP_MINUTES = 30  # 確定後30分は反発シグナル探索の対象外(既存M窓の準備期間定義を踏襲)
SIGNAL_SEARCH_HOURS = {"short_3h": 3, "long_24h": 24}
TP_FRAC = 0.5  # ブレイクバーレンジの50%戻し
SL_VARIANTS = {"shallow_0.1atr": 0.1, "deep_0.5atr": 0.5}
MAX_HOLD_HOURS = 72  # 既存安全上限(MAX_TREND_HOURS)を踏襲
SPREAD_PIPS = {"USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6, "EUR_USD": 0.3}
SLIPPAGE_PIPS = 0.5  # 往路・復路それぞれに適用(既存コストモデルの一般成行/逆指値の中間的な単純化)


def load_m5(pair: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "ds-1" / f"ohlcv_{pair}_5min_*.csv")))
    frames = [pd.read_csv(f, parse_dates=["timestamp"]) for f in files]
    df = pd.concat(frames).drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    return df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]


def resample(m5: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]
    return pd.DataFrame({c: m5[c].resample(rule).agg(a) for c, a in agg}).dropna()


def find_reversal_signal(m5: pd.DataFrame, atr_m5: pd.Series, search_start: pd.Timestamp,
                          search_end: pd.Timestamp, direction: int) -> dict | None:
    """search_start以降のM5バーで、causalな「反発シグナル」を探す(先読みなし)。

    direction: -1 = DOWNブレイク(逆張りはLONG方向を探す)、+1 = UPブレイク(逆張りはSHORT方向)

    ロジック: ランニング安値/高値(direction=-1なら安値、+1なら高値)を更新し続け、
    「更新した瞬間のバーの高値/安値」を反対方向へ後続バーの終値がブレイクしたら
    シグナル確定。シグナルバーの次バー始値でエントリーする(同バー終値では
    約定できないという実務上の制約を反映)。
    """
    win = m5[(m5.index >= search_start) & (m5.index < search_end)]
    if len(win) < 2:
        return None
    highs, lows, closes = win["high"].to_numpy(), win["low"].to_numpy(), win["close"].to_numpy()
    idx = win.index

    if direction < 0:  # DOWN breakout -> ランニング安値を追跡、終値が「安値バーの高値」を上抜けたらシグナル
        running_extreme = lows[0]
        extreme_bar_high = highs[0]
        for i in range(1, len(win)):
            if lows[i] < running_extreme:
                running_extreme = lows[i]
                extreme_bar_high = highs[i]
                continue
            if closes[i] > extreme_bar_high:
                if i + 1 >= len(win):
                    return None  # シグナルは出たがエントリー用の次バーがサーチ窓外
                entry_time = idx[i + 1]
                entry_price = float(win["open"].iloc[i + 1])
                a = float(atr_m5.reindex([idx[i]]).iloc[0]) if idx[i] in atr_m5.index else np.nan
                return {
                    "signal_time": str(idx[i]), "entry_time": str(entry_time),
                    "entry_price": entry_price, "running_extreme": float(running_extreme),
                    "atr_m5_at_signal": a,
                }
        return None
    else:  # UP breakout -> ランニング高値を追跡、終値が「高値バーの安値」を下抜けたらシグナル
        running_extreme = highs[0]
        extreme_bar_low = lows[0]
        for i in range(1, len(win)):
            if highs[i] > running_extreme:
                running_extreme = highs[i]
                extreme_bar_low = lows[i]
                continue
            if closes[i] < extreme_bar_low:
                if i + 1 >= len(win):
                    return None
                entry_time = idx[i + 1]
                entry_price = float(win["open"].iloc[i + 1])
                a = float(atr_m5.reindex([idx[i]]).iloc[0]) if idx[i] in atr_m5.index else np.nan
                return {
                    "signal_time": str(idx[i]), "entry_time": str(entry_time),
                    "entry_price": entry_price, "running_extreme": float(running_extreme),
                    "atr_m5_at_signal": a,
                }
        return None


def simulate_trade(m5: pd.DataFrame, entry_time: pd.Timestamp, entry_price: float,
                    sl_price: float, tp_price: float, direction_counter: int,
                    max_hold_hours: int) -> dict:
    """direction_counter: +1=LONGで逆張り(DOWNブレイク後)、-1=SHORTで逆張り(UPブレイク後)。"""
    end_time = entry_time + pd.Timedelta(hours=max_hold_hours)
    path = m5[(m5.index >= entry_time) & (m5.index < end_time)]
    if len(path) == 0:
        return {"outcome": "NO_DATA"}
    for ts, row in path.iterrows():
        hi, lo = float(row["high"]), float(row["low"])
        if direction_counter > 0:
            hit_tp, hit_sl = hi >= tp_price, lo <= sl_price
        else:
            hit_tp, hit_sl = lo <= tp_price, hi >= sl_price
        if hit_tp and hit_sl:
            return {"outcome": "AMBIGUOUS", "exit_time": str(ts)}
        if hit_tp:
            return {"outcome": "TP", "exit_time": str(ts), "exit_price": tp_price}
        if hit_sl:
            return {"outcome": "SL", "exit_time": str(ts), "exit_price": sl_price}
    last_close = float(path["close"].iloc[-1])
    return {"outcome": "TIMEOUT", "exit_time": str(path.index[-1]), "exit_price": last_close}


def main() -> int:
    print("=== 逆張り(反発シグナル+浅めSL+50%TP)の定量検証 (Train期間のみ・探索的) ===\n")

    results: dict = {sw: {slv: [] for slv in SL_VARIANTS} for sw in SIGNAL_SEARCH_HOURS}
    n_breakout_events_total = 0
    n_signal_found = {sw: 0 for sw in SIGNAL_SEARCH_HOURS}

    for pair in PAIRS:
        m5 = load_m5(pair)
        h1 = resample(m5, "1h")
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=ATR_LEN)
        atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=ATR_LEN)
        ratio = (h1["high"] - h1["low"]) / atr_h1
        body_dir = np.sign((h1["close"] - h1["open"]).to_numpy())
        valid = np.isfinite(ratio.to_numpy()) & (body_dir != 0)
        is_break = valid & (ratio.to_numpy() >= N_BREAKOUT)
        positions = np.where(is_break)[0]
        n_breakout_events_total += len(positions)
        pip = 0.01 if "JPY" in pair else 0.0001
        rt_cost_price = (2 * SPREAD_PIPS[pair] + 2 * SLIPPAGE_PIPS) * pip

        for pos in positions:
            bar = h1.iloc[pos]
            rng = float(bar["high"] - bar["low"])
            if rng <= 0:
                continue
            d = int(body_dir[pos])  # +1=UPブレイク, -1=DOWNブレイク
            confirmed_time = h1.index[pos] + pd.Timedelta(hours=1)  # バー確定時刻(先読みなし)
            search_start = confirmed_time + pd.Timedelta(minutes=PREP_MINUTES)

            if d < 0:  # DOWNブレイク -> 逆張りLONG、TPはbreak_low+50%range
                tp_price = float(bar["low"] + TP_FRAC * rng)
                counter_dir = +1
            else:  # UPブレイク -> 逆張りSHORT、TPはbreak_high-50%range
                tp_price = float(bar["high"] - TP_FRAC * rng)
                counter_dir = -1

            for sw_name, sw_hours in SIGNAL_SEARCH_HOURS.items():
                search_end = confirmed_time + pd.Timedelta(hours=sw_hours)
                sig = find_reversal_signal(m5, atr_m5, search_start, search_end, d)
                if sig is None:
                    continue
                n_signal_found[sw_name] += 1
                a = sig["atr_m5_at_signal"]
                if not np.isfinite(a) or a <= 0:
                    continue
                entry_time = pd.Timestamp(sig["entry_time"])
                entry_price = sig["entry_price"]
                for slv_name, slv_mult in SL_VARIANTS.items():
                    if counter_dir > 0:
                        sl_price = sig["running_extreme"] - slv_mult * a
                    else:
                        sl_price = sig["running_extreme"] + slv_mult * a
                    sim = simulate_trade(m5, entry_time, entry_price, sl_price, tp_price,
                                          counter_dir, MAX_HOLD_HOURS)
                    risk = abs(entry_price - sl_price)
                    reward = abs(tp_price - entry_price)
                    if risk <= 0:
                        continue
                    rr = reward / risk
                    r_gross = None
                    if sim["outcome"] == "TP":
                        r_gross = rr
                    elif sim["outcome"] == "SL":
                        r_gross = -1.0
                    elif sim["outcome"] == "TIMEOUT":
                        r_gross = counter_dir * (sim["exit_price"] - entry_price) / risk
                    cost_r = rt_cost_price / risk
                    r_net = (r_gross - cost_r) if r_gross is not None else None
                    results[sw_name][slv_name].append({
                        "pair": pair, "direction_counter": counter_dir,
                        "entry_time": str(entry_time), "outcome": sim["outcome"],
                        "risk_price": risk, "reward_price": reward, "rr": rr,
                        "r_gross": r_gross, "cost_r": cost_r, "r_net": r_net,
                    })

    def summarize(trades: list[dict]) -> dict:
        n = len(trades)
        if n == 0:
            return {"n": 0}
        by_outcome = {}
        for o in ("TP", "SL", "TIMEOUT", "AMBIGUOUS"):
            by_outcome[o] = sum(1 for t in trades if t["outcome"] == o)
        decided = by_outcome["TP"] + by_outcome["SL"]
        win_rate = by_outcome["TP"] / decided if decided else None
        rr_list = [t["rr"] for t in trades]
        r_gross_list = [t["r_gross"] for t in trades if t["r_gross"] is not None]
        cost_r_list = [t["cost_r"] for t in trades]
        r_net_list = [t["r_net"] for t in trades if t["r_net"] is not None]
        wins = [t["r_net"] for t in trades if t["r_net"] is not None and t["r_net"] > 0]
        losses = [t["r_net"] for t in trades if t["r_net"] is not None and t["r_net"] <= 0]
        pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None
        gross_expectancy = win_rate * float(np.median(rr_list)) - (1 - win_rate) if win_rate is not None else None
        return {
            "n": n,
            "by_outcome": by_outcome,
            "win_rate_decided_TP_vs_SL": round(win_rate, 4) if win_rate is not None else None,
            "median_rr_reward_over_risk": round(float(np.median(rr_list)), 3),
            "mean_r_gross_before_cost": round(float(np.mean(r_gross_list)), 4) if r_gross_list else None,
            "mean_cost_r": round(float(np.mean(cost_r_list)), 4),
            "median_cost_r": round(float(np.median(cost_r_list)), 4),
            "mean_r_net": round(float(np.mean(r_net_list)), 4) if r_net_list else None,
            "median_r_net": round(float(np.median(r_net_list)), 4) if r_net_list else None,
            "profit_factor_r_net": round(pf, 3) if pf else None,
            "positive_r_net_rate": round(sum(1 for r in r_net_list if r > 0) / len(r_net_list), 4) if r_net_list else None,
        }

    out = {
        "generated_at": datetime.now().isoformat(),
        "status": "探索的診断(正式プロトコル外・spec編集なし)",
        "question": "戻りを短期反発シグナルで判定し、浅めSL+50%TPで逆張りすれば勝率80%は実現できるか",
        "period": {"train_start": TRAIN_START, "train_end": TRAIN_END},
        "pairs": PAIRS,
        "n_breakout_events_total": int(n_breakout_events_total),
        "n_signal_found_by_search_window": n_signal_found,
        "definitions": {
            "reversal_signal": "ブレイク確定後30分の準備期間を空け、M5のランニング安値/高値"
                               "(DOWN/UPそれぞれ)を追跡。その安値/高値を付けたバーの高値/安値を、"
                               "後続バーの終値が反対方向にブレイクした時点をシグナル確定とし、"
                               "次バー始値でエントリー(先読みなし)",
            "sl_shallow": "シグナル確定時点のランニング安値/高値から0.1×ATR(M5)だけ離れた位置",
            "sl_deep": "比較用: 同じ起点から0.5×ATR(M5)だけ離れた位置",
            "tp": "ブレイクバー(H1)のレンジの50%戻し水準",
            "cost_model": f"往復スプレッド×2+スリッページ{SLIPPAGE_PIPS}pip×2(簡易モデル、"
                          "既存T-09のSL/トレール向けスリッページより単純化)",
            "max_hold_hours": MAX_HOLD_HOURS,
        },
        "results": {
            sw: {slv: summarize(results[sw][slv]) for slv in SL_VARIANTS}
            for sw in SIGNAL_SEARCH_HOURS
        },
        "caveats": [
            "反発シグナルが探索窓内に出たイベントのみを対象にしている(出なかったイベントは除外、"
            "その除外率はn_signal_found_by_search_window / n_breakout_events_totalで確認できる)。",
            "1トレンドイベントにつき1トレードのみ(ピラミッディングなし、既存の1イベント1トラッキングとは別モデル)。",
            "週末クローズ・両建て証拠金制約は考慮していない。",
            "Train期間のみ。Validation/Testは正式検証のために温存している。",
        ],
    }

    out_path = ROOT / "research" / "method-notes" / "counter_trend_reversal_entry.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"H1ブレイクイベント総数: {n_breakout_events_total}")
    for sw in SIGNAL_SEARCH_HOURS:
        print(f"\n--- 探索窓={sw} (シグナル成立 {n_signal_found[sw]}/{n_breakout_events_total} "
              f"= {n_signal_found[sw]/max(1,n_breakout_events_total):.1%}) ---")
        for slv in SL_VARIANTS:
            s = out["results"][sw][slv]
            print(f"  [{slv}] n={s['n']} 内訳={s.get('by_outcome')} "
                  f"win_rate(TP/(TP+SL))={s.get('win_rate_decided_TP_vs_SL')} "
                  f"median_RR={s.get('median_rr_reward_over_risk')} "
                  f"r_gross(コスト抜き)={s.get('mean_r_gross_before_cost')} "
                  f"mean_cost_r={s.get('mean_cost_r')} "
                  f"r_net(コスト込み)={s.get('mean_r_net')} PF={s.get('profit_factor_r_net')} "
                  f"positive_rate={s.get('positive_r_net_rate')}")
    print(f"\n出力: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
