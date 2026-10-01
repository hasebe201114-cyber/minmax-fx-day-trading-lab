"""OBS000020 C品質チーム検証(2026-10-01): 打ち切りテスト(実装に依存しない先読み検出).

公式の取引(Train/Validation)それぞれについて、「エントリー判定時刻(エントリー5分足の確定時)までに
存在したデータだけ」(全4通貨の5分足を entry_ts 以前で打ち切り、H1・ATR・急変フィルターを打ち切り後の
データから再計算)で、公式関数 simulate_dow_theory_trend を同じブレイクイベントについて再実行し、
同じ時刻・同じ価格のエントリーが再現するかを調べる。

因果的なシミュレーションなら、エントリー判定はエントリー以降のデータに依存しないので 100% 再現する。
再現しない取引 = エントリー時点ではまだ存在しない未来のデータに依存して発生した取引。
(注: これは必要条件の検査。エントリーより前の時間帯で先読みした結果、経路がずれた取引は検出できない。
 経路の影響は review_c_causal_reimpl.py の実時間版で測る。)

モード:
  all       : 全部を打ち切り後データで再計算(公式コードの総合判定)
  confirm   : 継続確認(H1)だけ打ち切り後データ、急変フィルターは全データ(公式と同じ)
  shock     : 急変フィルターだけ打ち切り後データ、継続確認は全データ
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_sysfx018_breakeven_sweep_trainonly as fx018  # noqa: E402
from analyze_n_breakout_h1_dow_trend_alignment import h1_dow_trend_direction  # noqa: E402
from backtest_vol_breakout_dow_theory import select_non_overlapping_breakout_events, simulate_dow_theory_trend  # noqa: E402
import backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd as v7  # noqa: E402
from derive_vol_breakout_entry_params import N_BREAKOUT, to_h1  # noqa: E402
from price_shock_filter import make_price_shock_check  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

KW = dict(blackout_check=None, tp_levels=v7.TP_LEVELS_TRAILONLY, skip_first_entry=False, m5_exit=True,
          breakeven_trigger_r=1.0, bar_close_anchored=True)


def prep(m5):
    h1 = to_h1(m5)
    return h1, atr_ind(h1["high"], h1["low"], h1["close"], length=14), atr_ind(m5["high"], m5["low"], m5["close"], length=14)


def official_events(pair, m5, h1, atr_h1, atr_m5, shock):
    ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
    idxs = np.where(ratio.values >= N_BREAKOUT)[0]
    positions = [h1.index.get_loc(ratio.index[i]) for i in idxs]
    directions = ["UP" if h1.iloc[p]["close"] > h1.iloc[p]["open"] else "DOWN" for p in positions]
    dedup = select_non_overlapping_breakout_events(h1.index, positions, directions)
    dmap = dict(zip(positions, directions))
    out = []
    for pos in dedup:
        if h1_dow_trend_direction(h1, atr_h1, pos) is None:
            continue
        for t in simulate_dow_theory_trend(m5, atr_m5, h1, atr_h1, pos, dmap[pos], v7.STOP_BUFFER_ATR_M5,
                                           v7.ATR_TRAIL_MULTIPLIER_M5, atr_trail_series=atr_m5,
                                           **(KW | {"blackout_check": shock})):
            out.append(dict(pair=pair, break_time=h1.index[pos], direction=dmap[pos], entry_time=pd.Timestamp(t["entry_time"]),
                            entry_price=t["entry_price"], resumed=t["resumed_since_last_entry"], entry_seq=t["entry_seq"]))
    return out


def main() -> int:
    report = {}
    for per in ("train", "validation"):
        s, e = v7.PERIODS[per]
        m5s = {p: v7.load_m5_period(p, s, e) for p in fx018.SELECTED_PAIRS}
        m5s = {p: m for p, m in m5s.items() if len(m) >= 1000}
        full = {p: prep(m) for p, m in m5s.items()}
        shock_full = make_price_shock_check({p: f[0] for p, f in full.items()}, {p: f[1] for p, f in full.items()})
        trades = []
        for p, m5 in m5s.items():
            h1, ah, am = full[p]
            trades += official_events(p, m5, h1, ah, am, shock_full)
        print(f"[{per}] 公式取引(会計前の生成数) {len(trades)}件", flush=True)
        rows = []
        for k, t in enumerate(trades):
            cut = t["entry_time"]
            tr_m5 = {p: m[m.index <= cut] for p, m in m5s.items()}
            tr = {p: prep(m) for p, m in tr_m5.items()}
            shock_tr = make_price_shock_check({p: f[0] for p, f in tr.items()}, {p: f[1] for p, f in tr.items()})
            p = t["pair"]
            res = {}
            for mode in ("all", "confirm", "shock"):
                if mode in ("all", "confirm"):
                    m5, (h1, ah, am) = tr_m5[p], tr[p]
                else:
                    m5, (h1, ah, am) = m5s[p], full[p]
                sh = shock_tr if mode in ("all", "shock") else shock_full
                pos = h1.index.get_loc(t["break_time"])
                sims = simulate_dow_theory_trend(m5, am, h1, ah, pos, t["direction"], v7.STOP_BUFFER_ATR_M5,
                                                 v7.ATR_TRAIL_MULTIPLIER_M5, atr_trail_series=am, **(KW | {"blackout_check": sh}))
                res[mode] = any(pd.Timestamp(x["entry_time"]) == cut and abs(x["entry_price"] - t["entry_price"]) < 1e-9
                                for x in sims)
            rows.append(dict(pair=p, entry_time=str(cut), break_time=str(t["break_time"]), direction=t["direction"],
                             resumed=bool(t["resumed"]), entry_seq=t["entry_seq"], **{f"reproduced_{m}": v for m, v in res.items()}))
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(trades)}", flush=True)
        df = pd.DataFrame(rows)
        summ = {"n": len(df)}
        for m in ("all", "confirm", "shock"):
            col = f"reproduced_{m}"
            summ[f"not_reproduced_{m}"] = int((~df[col]).sum())
            summ[f"not_reproduced_{m}_resumed"] = int((~df[col] & df["resumed"]).sum())
        summ["n_resumed"] = int(df["resumed"].sum())
        report[per] = {"summary": summ, "rows": rows}
        print(f"[{per}] {json.dumps(summ, ensure_ascii=False)}", flush=True)
    out = ROOT / "scripts" / "obs000020" / "review_c_truncation_test_result.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[出力] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
