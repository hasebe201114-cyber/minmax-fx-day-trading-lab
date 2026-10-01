"""OBS000020 C品質チーム検証(2026-10-01): simulate_dow_theory_trend の独立再実装.

オーケストレーターの測定(confirm_bars の差し替え・index の +55分ずらし)とは別に、
エントリー層の追跡ロジックを独立に書き直し、「各5分足 i の判定時刻(= ts+5分、足の確定時)に
何が見えているか」をモードごとに明示して数値を出す。既存の公式スクリプトは変更しない。

継続確認(confirm)モード:
  official      : 公式と同じ(ts を含む H1 足の high/low を、その時間の最初の5分足で丸ごと参照=先読み)。
                  本再実装が公式(308件/+99.22R、83件/+46.26R)を完全再現することの確認用。
  running       : 実時間版。各5分足の確定時点の「形成途中の H1 足のその時点までの高値/安値」で判定。
                  仕様(「型崩れ以降の H1 高値が、トレンド開始から M5 型崩れ時点までの H1 高値を更新」)の
                  忠実な実時間解釈。
  closed        : 確定済み H1 足のみ(オーケストレーター版と同じ。閾値は確定足のみで作る)。
  closed_strict : 確定済み H1 足のみ。ただし閾値には「型崩れ時点までの形成途中の H1 高値」も含める
                  (closed では、型崩れ前に付けた同じ時間内の高値で「再開」してしまう余地があるため)。

急変フィルター(shock)モード:
  official : 公式(形成途中の H1 足の値幅を、その時間の最初から丸ごと参照=先読み)
  closed   : 確定済み H1 足のみ(H1 足 HH は HH:55 開始の5分足=HH+1:00 確定時から有効)
  running  : 確定済み H1 足の状態 + 形成途中の H1 足のその時点までの値幅で発動(解除は確定足のみ)

方向 DOWN は価格を符号反転して UP 形式で追跡する(決済シミュレーションは元の価格で公式関数を使う)。

使い方: python scripts/obs000020/review_c_causal_reimpl.py  (Train/Validation のみ。Test は計算しない)
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
import price_shock_filter as psf  # noqa: E402
from backtest_vol_breakout_dow_theory import (  # noqa: E402
    MAX_TREND_HOURS, WINDOW_START_MIN, ZIGZAG_THRESHOLD_ATR_M5, is_weekend_close_time, simulate_scaled_scheme,
)
import backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd as v7  # noqa: E402

ORIG_SIM = fx018.simulate_dow_theory_trend
ORIG_SHOCK = fx018.make_price_shock_check
ORIG_LOAD = fx018.load_m5_period
H1 = pd.Timedelta(hours=1)
M5 = pd.Timedelta(minutes=5)

_M5_CACHE: dict[str, pd.DataFrame] = {}
TRADE_LOG: list[dict] = []
EVENT_LOG: list[dict] = []  # 時刻追跡用(型崩れ→待機開始 / 再開)


def _hour_to_date(m5: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    hour = m5.index.floor("h")
    htd_high = m5["high"].groupby(hour).cummax().to_numpy(dtype=float)
    htd_low = m5["low"].groupby(hour).cummin().to_numpy(dtype=float)
    return htd_high, htd_low


def make_sim(confirm_mode: str):
    assert confirm_mode in ("official", "running", "closed", "closed_strict")

    def sim(m5, atr_m5, h1, atr_h1, break_idx, direction, stop_buffer_atr_m5, trail_mult, blackout_check=None,
            breakeven_trigger_r=None, tp_levels=None, skip_first_entry=False, atr_trail_series=None,
            m5_exit=False, bar_close_anchored=False, **kw):
        assert not kw, kw
        assert bar_close_anchored and m5_exit and not skip_first_entry
        sgn = 1.0 if direction == "UP" else -1.0
        # UP 形式の配列(DOWN は符号反転: high' = -low, low' = -high)
        if direction == "UP":
            H, L = m5["high"].to_numpy(float), m5["low"].to_numpy(float)
            H1H, H1L = h1["high"].to_numpy(float), h1["low"].to_numpy(float)
        else:
            H, L = -m5["low"].to_numpy(float), -m5["high"].to_numpy(float)
            H1H, H1L = -h1["low"].to_numpy(float), -h1["high"].to_numpy(float)
        C = m5["close"].to_numpy(float)
        htd_h, htd_l = _hour_to_date(m5)
        HTD = htd_h if direction == "UP" else -htd_l
        ATR5 = atr_m5.to_numpy(float)
        idx = m5.index
        h1_start = h1.index
        h1_visible_from = h1_start + pd.Timedelta(minutes=55)  # 5分足の開始時刻がこれ以上なら H1 足は確定済み

        break_time = h1_start[break_idx]
        bar_duration = pd.Timedelta(np.median(np.diff(h1_start.values)))
        anchor = break_time + bar_duration
        start_pos = idx.searchsorted(anchor + pd.Timedelta(minutes=WINDOW_START_MIN), side="right")
        end_pos = idx.searchsorted(anchor + pd.Timedelta(hours=MAX_TREND_HOURS), side="right")
        if start_pos >= len(m5) or start_pos >= end_pos:
            return []

        trades = []
        last_conf = H1L[break_idx]
        state = "SH"  # SEARCHING_HIGH(UP形式)
        run_ext = H[start_pos]
        pos_open_until = None
        ext = H1H[break_idx]
        if confirm_mode == "official":
            last_checked = int(h1_start.searchsorted(break_time, side="right") - 1)
        elif confirm_mode in ("closed", "closed_strict"):
            last_checked = int(h1_visible_from.searchsorted(break_time, side="right") - 1)
        else:  # running: ブレイク足確定〜追跡開始直前までの5分足高値を閾値に含める
            pre = (idx >= anchor) & (np.arange(len(idx)) < start_pos)
            if pre.any():
                ext = max(ext, float(H[pre].max()))
            last_checked = None
        awaiting = False
        thr = None
        pause = None
        entry_seq = 0
        resumed = False
        resume_time = None

        for i in range(start_pos, end_pos):
            ts = idx[i]
            if is_weekend_close_time(idx, i):
                break
            h_i, l_i, c_i = H[i], L[i], C[i]

            # ---- 継続確認 ----
            if confirm_mode == "running":
                if awaiting and HTD[i] > thr:
                    awaiting = False
                    state = "SH"
                    last_conf = pause if pause is not None else last_conf
                    run_ext = h_i
                    resumed = True
                    resume_time = ts
                    EVENT_LOG.append(dict(kind="resume", break_time=str(break_time), ts=str(ts), basis=sgn * HTD[i]))
                ext = max(ext, HTD[i])
            else:
                if confirm_mode == "official":
                    cur = int(h1_start.searchsorted(ts, side="right") - 1)
                else:
                    cur = int(h1_visible_from.searchsorted(ts, side="right") - 1)
                if cur > last_checked:
                    for hp in range(last_checked + 1, cur + 1):
                        if awaiting and H1H[hp] > thr:
                            awaiting = False
                            state = "SH"
                            last_conf = pause if pause is not None else last_conf
                            run_ext = h_i
                            resumed = True
                            resume_time = ts
                            EVENT_LOG.append(dict(kind="resume", break_time=str(break_time), ts=str(ts), h1_bar=str(h1_start[hp]),
                                                  basis=sgn * H1H[hp]))
                        ext = max(ext, H1H[hp])
                    last_checked = cur

            if awaiting:
                pause = l_i if pause is None else min(pause, l_i)
                continue

            atr_i = ATR5[i]
            if not np.isfinite(atr_i) or atr_i <= 0:
                continue
            th = ZIGZAG_THRESHOLD_ATR_M5 * atr_i
            if state == "SH":
                if h_i > run_ext:
                    run_ext = h_i
                if run_ext - l_i >= th:
                    state = "SL"
                    run_ext = l_i
            else:
                if l_i < run_ext:
                    run_ext = l_i
                if h_i - run_ext >= th:
                    pivot = run_ext
                    if pivot > last_conf:
                        if (pos_open_until is None or ts > pos_open_until) and not (blackout_check and blackout_check(ts)):
                            stop0 = sgn * (pivot - stop_buffer_atr_m5 * atr_i)
                            entry_price = c_i
                            risk = abs(entry_price - stop0)
                            eh = int(h1_start.searchsorted(ts, side="right") - 1)
                            if risk > 0 and 0 <= eh < len(h1):
                                entry_seq += 1
                                entry = dict(direction=direction, entry_idx=eh, entry_m5_idx=i, entry_price=entry_price,
                                             stop0=stop0, initial_risk=risk, entry_ts=ts)
                                res = simulate_scaled_scheme(h1, atr_h1, entry, trail_mult,
                                                             breakeven_trigger_r=breakeven_trigger_r, tp_levels=tp_levels,
                                                             atr_trail_series=atr_trail_series, m5_exit=m5)
                                res.update(entry_time=str(ts), direction=direction, entry_price=entry_price,
                                           initial_risk=risk, entry_seq=entry_seq, resumed_since_last_entry=resumed,
                                           resume_time=str(resume_time) if resumed else None,
                                           break_time=str(break_time))
                                trades.append(res)
                                TRADE_LOG.append(dict(res))
                                pos_open_until = res["exit_time"]
                                resumed = False
                        last_conf = pivot
                        state = "SH"
                        run_ext = h_i
                    else:
                        awaiting = True
                        pause = min(pivot, l_i)
                        thr = ext
                        if confirm_mode == "closed_strict":
                            thr = max(thr, HTD[i])
                        EVENT_LOG.append(dict(kind="await", break_time=str(break_time), ts=str(ts), thr=sgn * thr,
                                              pivot=sgn * pivot))
        return trades

    return sim


def _load_recording(pair, start, end):
    m5 = ORIG_LOAD(pair, start, end)
    _M5_CACHE[pair] = m5
    return m5


def make_shock_builder(shock_mode: str):
    assert shock_mode in ("official", "closed", "running")
    if shock_mode == "official":
        return ORIG_SHOCK

    def builder(h1_by_pair, atr_by_pair):
        s = psf.build_shock_suppression_series(h1_by_pair, atr_by_pair)
        vis_idx = s.index + pd.Timedelta(minutes=55)
        vis_val = s.to_numpy(bool)
        if shock_mode == "closed":
            def check(ts):
                p = vis_idx.searchsorted(ts, side="right") - 1
                return False if p < 0 else bool(vis_val[p])
            return check
        # running: 形成途中の H1 足のその時点までの値幅/ATR(形成途中の TR を含む14本平均)で発動
        partial = {}
        for pair, h1 in h1_by_pair.items():
            m5 = _M5_CACHE[pair]
            hh, ll = _hour_to_date(m5)
            hour = m5.index.floor("h")
            prev_close = h1["close"].shift(1)
            tr_full = pd.concat([h1["high"] - h1["low"], (h1["high"] - prev_close).abs(),
                                 (h1["low"] - prev_close).abs()], axis=1).max(axis=1)
            s13 = tr_full.rolling(13, min_periods=13).sum().shift(1)  # 直前13本の確定 TR 合計
            pc = prev_close.reindex(hour).to_numpy(float)
            s13v = s13.reindex(hour).to_numpy(float)
            tr_p = np.nanmax(np.vstack([hh - ll, np.abs(hh - pc), np.abs(ll - pc)]), axis=0)
            atr_p = (s13v + tr_p) / 14.0
            ratio = np.where(atr_p > 0, (hh - ll) / atr_p, np.nan)
            partial[pair] = (m5.index, hour, ratio)

        def check(ts):
            p = vis_idx.searchsorted(ts, side="right") - 1
            if p >= 0 and vis_val[p]:
                return True
            cur_hour = ts.floor("h")
            n = 0
            for pair, (mi, hour, ratio) in partial.items():
                q = mi.searchsorted(ts, side="right") - 1
                if q >= 0 and hour[q] == cur_hour and np.isfinite(ratio[q]) and ratio[q] >= psf.N_BREAKOUT_THRESHOLD:
                    n += 1
            return n >= psf.SIMULTANEOUS_PAIRS_REQUIRED
        return check

    return builder


def run(confirm_mode: str, shock_mode: str, seed_note: bool = False) -> dict:
    fx018.load_m5_period = _load_recording
    fx018.simulate_dow_theory_trend = make_sim(confirm_mode)
    fx018.make_price_shock_check = make_shock_builder(shock_mode)
    out = {}
    try:
        for per in ("train", "validation"):
            s, e = v7.PERIODS[per]
            TRADE_LOG.clear()
            p = fx018.run_period(1.0, s, e)
            r = np.array([t["r_net"] for t in p["trades"]])
            log = {(t["entry_time"], round(t["entry_price"], 6)): t for t in TRADE_LOG}
            resumed = [log.get((str(pd.Timestamp(t["entry_time"])), round(t["entry_price"], 6)), {}).get(
                "resumed_since_last_entry") for t in p["trades"]]
            res_mask = np.array([bool(x) for x in resumed])
            out[per] = {
                "n": int(len(r)), "sum_r": round(float(r.sum()), 2), "mean_r": round(float(r.mean()), 4),
                "win_rate": p["win_rate"], "pf": p["profit_factor"], "perm_p_block": p["perm_p_block"],
                "final_usd": round(float(p["equity_curve"][-1]["balance"]), 2),
                "n_resumed": int(res_mask.sum()), "sum_r_resumed": round(float(r[res_mask].sum()), 2),
                "sum_r_not_resumed": round(float(r[~res_mask].sum()), 2),
                "trades": [{k: t[k] for k in ("pair", "direction", "entry_time", "exit_time", "entry_price", "r_net",
                                              "exit_reason")} | {"resumed": bool(m)}
                           for t, m in zip(p["trades"], res_mask)],
            }
    finally:
        fx018.load_m5_period = ORIG_LOAD
        fx018.simulate_dow_theory_trend = ORIG_SIM
        fx018.make_price_shock_check = ORIG_SHOCK
    return out


def main() -> int:
    combos = [("official", "official"), ("running", "official"), ("closed", "official"),
              ("closed_strict", "official"),
              ("official", "closed"), ("official", "running"),
              ("running", "closed"), ("running", "running"),
              ("closed", "closed"), ("closed", "running"),
              ("closed_strict", "closed"), ("closed_strict", "running")]
    if len(sys.argv) > 1:
        combos = [tuple(a.split(":")) for a in sys.argv[1:]]
    results = {}
    for cm, sm in combos:
        key = f"confirm={cm}|shock={sm}"
        results[key] = run(cm, sm)
        tr, va = results[key]["train"], results[key]["validation"]
        print(f"{key:40s} Train n={tr['n']:3d} R={tr['sum_r']:+8.2f} p={tr['perm_p_block']}  "
              f"Val n={va['n']:3d} R={va['sum_r']:+8.2f} p={va['perm_p_block']}  "
              f"(resumed Train {tr['n_resumed']}件 {tr['sum_r_resumed']:+.1f}R / Val {va['n_resumed']}件 {va['sum_r_resumed']:+.1f}R)",
              flush=True)
    out = ROOT / "scripts" / "obs000020" / "review_c_causal_reimpl_result.json"
    tag = "_".join(f"{a}-{b}" for a, b in combos) if len(sys.argv) > 1 else "all"
    if tag != "all":
        out = out.with_name(f"review_c_causal_reimpl_result_{tag}.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[出力] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
