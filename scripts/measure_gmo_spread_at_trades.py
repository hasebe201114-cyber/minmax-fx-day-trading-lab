"""SYS-FX012 の各取引のエントリー・決済時刻に、GMOコイン外国為替FXで実際についていたスプレッドを測る。

GMO 公開 API（認証不要）`/public/v1/klines` の BID と ASK の5分足を取得し、
エントリー／決済時刻 t に始まる5分足の始値の差（ASK始値 − BID始値）を、その時刻のスプレッドとする。
実発注・認証情報は一切使わない。

対象: Train（2023-11〜2025-03）・Validation（2025-04〜2025-11）の公式出力と、フォワード台帳（2026-08-15〜）。
Test 期間（2025-12-01〜2026-08-15）の取引は対象にない（凍結ホールドアウトは参照しない）。

コストの再計算: minmax の公式の式（スプレッド＋スリッページ、手数料）で、スプレッドだけを実測値に置き換える。
  公式: cost_price = (spread + 0.5) + (spread + exit_slippage)  [pips]
  実測: cost_price = (spread_entry + 0.5) + (spread_exit + exit_slippage)
  （エントリーは買いならASK・売りならBIDで約定、決済は反対側。片道ずつの実測スプレッドを使う）

出力: research/method-notes/gmo_spread_at_trades.json と同名 .csv（取引ごと）
取得した5分足は一時ディレクトリにキャッシュする（リポジトリには入れない）。
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
MN = ROOT / "research" / "method-notes"
CACHE = Path(tempfile.gettempdir()) / "gmo_klines_cache"
URL = "https://forex-api.coin.z.com/public/v1/klines"
MODEL_SPREAD = {"USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6}
SLIP_MARKET, SLIP_STOP = 0.5, 1.0
MARKET_EXITS = ("WEEKEND_NO_TP", "TP_THEN_WEEKEND", "MAX_HOLD")
STOP_EXITS = ("SL_INITIAL_NO_TP", "TP_THEN_SL_TRAIL")
TEST_START, TEST_END = pd.Timestamp("2025-12-01"), pd.Timestamp("2026-08-15 06:00")


def fetch_day(pair: str, side: str, date: str) -> list[dict]:
    path = CACHE / pair / side / f"{date}.json"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(4):
        try:
            r = requests.get(URL, params={"symbol": pair, "priceType": side, "interval": "5min", "date": date},
                             timeout=20)
            body = r.json()
            if body.get("status") == 0:
                data = body.get("data", [])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
                time.sleep(0.25)
                return data
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 ** attempt)
    return []


def open_price_at(pair: str, side: str, t: pd.Timestamp) -> float | None:
    """JST の時刻 t に始まる5分足の始値。GMO の日付は営業日（6時または7時開始）のため前後日を読む。"""
    target_ms = int((t - timedelta(hours=9)).tz_localize("UTC").timestamp() * 1000)
    for d in (t.date(), (t + timedelta(days=1)).date(), (t - timedelta(days=1)).date()):
        for bar in fetch_day(pair, side, d.strftime("%Y%m%d")):
            if int(bar["openTime"]) == target_ms:
                return float(bar["open"])
    return None


def load_trades() -> pd.DataFrame:
    frames = []
    tr = json.loads((MN / "vol_continuation_candidates_trendfilter_4pairs_trainonly_backtest.json").read_text())
    frames.append(pd.DataFrame(tr["backtest"]["candidate1_n_breakout_only"]["trades"]).assign(period="train"))
    va = json.loads((MN / "vol_breakout_trendfilter_candidate1_validation_backtest.json").read_text())
    frames.append(pd.DataFrame(va["backtest"]["trades"]).assign(period="validation"))
    fw = json.loads((MN / "sysfx012_forward_test_ledger.json").read_text())
    fwd = pd.DataFrame(fw.get("backtest", fw)["trades"])
    frames.append(fwd[fwd["exit_time"].notna()].assign(period="forward"))
    df = pd.concat(frames, ignore_index=True)
    for c in ("entry_time", "exit_time"):
        df[c] = pd.to_datetime(df[c].astype(str).str.slice(0, 19))
    assert not ((df["entry_time"] >= TEST_START) & (df["entry_time"] < TEST_END)).any(), "Test 期間の取引が含まれている"
    return df


def main() -> int:
    df = load_trades()
    rows = []
    for t in df.itertuples(index=False):
        pip = 0.01
        # 5分足の境界に揃っていない時刻（週末決済 05:55 等）もそのまま探す
        ent = {s: open_price_at(t.pair, s, t.entry_time) for s in ("ASK", "BID")}
        ext = {s: open_price_at(t.pair, s, t.exit_time) for s in ("ASK", "BID")}
        sp_in = (ent["ASK"] - ent["BID"]) / pip if None not in ent.values() else np.nan
        sp_out = (ext["ASK"] - ext["BID"]) / pip if None not in ext.values() else np.nan
        rows.append({"period": t.period, "pair": t.pair, "direction": t.direction, "entry_time": t.entry_time,
                     "exit_time": t.exit_time, "exit_reason": t.exit_reason, "initial_risk": t.initial_risk,
                     "fraction_via_tp": t.fraction_via_tp, "r_gross": t.r_gross, "cost_r": t.cost_r,
                     "commission_r": t.commission_r, "r_net": t.r_net,
                     "spread_model": MODEL_SPREAD[t.pair], "spread_entry": sp_in, "spread_exit": sp_out})
    m = pd.DataFrame(rows)
    remaining = 1.0 - m["fraction_via_tp"]
    exit_slip = np.where(m["exit_reason"].isin(MARKET_EXITS), remaining * SLIP_MARKET,
                         np.where(m["exit_reason"].isin(STOP_EXITS), remaining * SLIP_STOP, 0.0))
    ok = m["spread_entry"].notna() & m["spread_exit"].notna()
    cost_live = ((m["spread_entry"] + SLIP_MARKET) + (m["spread_exit"] + exit_slip)) * 0.01 / m["initial_risk"]
    m["cost_r_live"] = np.where(ok, cost_live, np.nan)
    m["r_net_live"] = m["r_gross"] - m["cost_r_live"] - m["commission_r"]
    m["cost_ratio_live_vs_model"] = (m["spread_entry"] + m["spread_exit"]) / (2 * m["spread_model"])

    def summary(g: pd.DataFrame) -> dict:
        g2 = g[g["spread_entry"].notna() & g["spread_exit"].notna()]
        wins, losses = g2["r_net_live"][g2["r_net_live"] > 0], g2["r_net_live"][g2["r_net_live"] < 0]
        return {
            "n": int(len(g)), "n_measured": int(len(g2)),
            "spread_entry_median": round(float(g2["spread_entry"].median()), 3),
            "spread_entry_mean": round(float(g2["spread_entry"].mean()), 3),
            "spread_entry_p90": round(float(g2["spread_entry"].quantile(0.9)), 3),
            "spread_exit_median": round(float(g2["spread_exit"].median()), 3),
            "spread_exit_mean": round(float(g2["spread_exit"].mean()), 3),
            "cost_ratio_live_vs_model_mean": round(float(g2["cost_ratio_live_vs_model"].mean()), 3),
            "mean_cost_r_model": round(float(g2["cost_r"].mean()), 4),
            "mean_cost_r_live": round(float(g2["cost_r_live"].mean()), 4),
            "mean_r_net_model": round(float(g2["r_net"].mean()), 4),
            "mean_r_net_live": round(float(g2["r_net_live"].mean()), 4),
            "sum_r_net_model": round(float(g2["r_net"].sum()), 2),
            "sum_r_net_live": round(float(g2["r_net_live"].sum()), 2),
            "win_rate_live": round(float((g2["r_net_live"] > 0).mean()), 4),
            "pf_live": round(float(wins.sum() / -losses.sum()), 3) if len(losses) else None,
        }

    out = {"source": "GMOコイン外国為替FX 公開API /v1/klines（BID・ASK 5分足、始値の差）",
           "model_spread_pips": MODEL_SPREAD, "note": "Test期間(2025-12-01〜2026-08-15)の取引は含まない",
           "by_period": {p: summary(g) for p, g in m.groupby("period")},
           "by_period_pair": {f"{p}/{q}": summary(g) for (p, q), g in m.groupby(["period", "pair"])},
           "by_pair_all": {q: summary(g) for q, g in m.groupby("pair")},
           "all": summary(m)}
    (MN / "gmo_spread_at_trades.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    m.to_csv(MN / "gmo_spread_at_trades.csv", index=False)
    print(json.dumps({k: out[k] for k in ("by_period", "by_pair_all", "all")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
