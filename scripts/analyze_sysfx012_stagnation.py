"""SYS-FX012 停滞期（2023年）の特性分析と、指標で捉えられるかの探索（司令塔依頼 2026-09-30）。

対象: sim_sysfx012_4y_compound.py と同じ取引（2021-11〜2025-11、コスト base_era_ratio）。Test 期間は含めない。
これは探索的な記述分析であり、ルール変更の根拠にはしない（採用するなら事前登録した別検証が必要）。

各取引のエントリー時点で既に分かっている値だけで指標を作る:
  vol_ratio   … H1 ATR(14) ÷ 過去100日の H1 ATR 中央値（ボラティリティの水準）
  er_24/er_120… H1 終値の効率比（24本・120本。1に近いほど一方向、0に近いほど往復）
  trend_align … 過去20日の値動きの向きと取引方向が一致(+1)/逆(-1)
  activity_30d… 直前30日の全通貨の取引数
  past_r_20   … 直前に決済済みの20件の平均 r_net（成績そのもの）
  be_rate_20  … 直前20件のうち建値移動(+1R)まで到達した割合
予測力は、取引ごとの r_net との順位相関と、指標系列を時間方向に循環シフトした並べ替え検定（自己相関を保つ）で見る。

出力: research/method-notes/sysfx012_stagnation_analysis.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import extended_data as ext  # noqa: E402

START, END = "2021-11-01", "2025-11-30"
CACHE = Path(tempfile.gettempdir()) / "sysfx012_4y_raw_base_era_ratio.pkl"
OUT = ROOT / "research" / "method-notes" / "sysfx012_stagnation_analysis.json"
PERIODS = {
    "停滞(最大DD) 2023-04-28〜07-24": ("2023-04-28", "2023-07-24"),
    "2023年通年": ("2023-01-01", "2023-12-31"),
}
FEATURES = ["vol_ratio", "er_24", "er_120", "trend_align", "activity_30d", "past_r_20", "be_rate_20"]
N_PERM = 2000


def jst(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    return ts.tz_convert("Asia/Tokyo").tz_localize(None) if ts.tzinfo else ts


def load_trades() -> pd.DataFrame:
    if not CACHE.exists():
        import backtest_sysfx018_breakeven_sweep_trainonly as fx018
        ext.patch_pipelines("base_era_ratio")
        p = fx018.run_period(1.0, START, END)
        raw = [{k: t[k] for k in ("pair", "direction", "entry_time", "exit_time", "exit_reason", "r_gross",
                                  "r_net", "leverage_ratio")} for t in p["trades"]]
        pd.to_pickle((raw, float(p["equity_curve"][-1]["balance"])), CACHE)
    raw, _ = pd.read_pickle(CACHE)
    df = pd.DataFrame(raw)
    df["entry_time"] = df["entry_time"].map(jst)
    df["exit_time"] = df["exit_time"].map(jst)
    return df.sort_values("entry_time").reset_index(drop=True)


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    return (close - close.shift(n)).abs() / close.diff().abs().rolling(n).sum()


def market_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, g in df.groupby("pair"):
        m5 = ext.load_m5_extended(pair, "2021-06-01", END)
        m5.index = m5.index.tz_convert("Asia/Tokyo").tz_localize(None)
        h1 = m5.resample("1h", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        tr = pd.concat([h1["high"] - h1["low"], (h1["high"] - h1["close"].shift()).abs(),
                        (h1["low"] - h1["close"].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        feat = pd.DataFrame({
            "vol_ratio": atr / atr.rolling(2400, min_periods=500).median(),
            "er_24": efficiency_ratio(h1["close"], 24),
            "er_120": efficiency_ratio(h1["close"], 120),
            "ret_480": h1["close"] / h1["close"].shift(480) - 1,
        })
        feat.index = feat.index + pd.Timedelta(hours=1)  # 確定時刻に付け替え（エントリー前に確定した足だけ使う）
        for i, t in g.iterrows():
            f = feat[feat.index <= t["entry_time"]].iloc[-1]
            d = 1 if t["direction"] == "UP" else -1
            rows.append({"idx": i, "vol_ratio": f["vol_ratio"], "er_24": f["er_24"], "er_120": f["er_120"],
                         "trend_align": float(np.sign(f["ret_480"]) * d)})
    return pd.DataFrame(rows).set_index("idx").sort_index()


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    exits = df.sort_values("exit_time")
    for i, t in df.iterrows():
        prior = exits[exits["exit_time"] < t["entry_time"]].tail(20)
        recent = df[(df["entry_time"] < t["entry_time"]) & (df["entry_time"] >= t["entry_time"] - pd.Timedelta(days=30))]
        out.append({"idx": i, "past_r_20": prior["r_net"].mean() if len(prior) == 20 else np.nan,
                    "be_rate_20": (prior["r_gross"] > -0.9).mean() if len(prior) == 20 else np.nan,
                    "activity_30d": float(len(recent))})
    return pd.DataFrame(out).set_index("idx")


def structure(sub: pd.DataFrame, months: float) -> dict:
    r, g = sub["r_net"], sub["r_gross"]
    w, l = r[r > 0], r[r < 0]
    return {
        "n": int(len(sub)), "trades_per_month": round(len(sub) / months, 1),
        "win_rate": round(float((r > 0).mean()), 3), "mean_r": round(float(r.mean()), 3), "sum_r": round(float(r.sum()), 1),
        "payoff": round(float(w.mean() / -l.mean()), 3) if len(w) and len(l) else None,
        "initial_sl_share": round(float((g <= -0.99).mean()), 3),
        "reached_be_share": round(float((g > -0.9).mean()), 3),
        "mean_winner_r": round(float(w.mean()), 3) if len(w) else None,
        "big_win_share_gt3r": round(float((r > 3).mean()), 3),
        "sum_r_by_pair": {p: round(float(v), 1) for p, v in sub.groupby("pair")["r_net"].sum().items()},
        "sum_r_by_direction": {p: round(float(v), 1) for p, v in sub.groupby("direction")["r_net"].sum().items()},
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def predictive(df: pd.DataFrame, feat: str, rng: np.random.Generator) -> dict:
    d = df[[feat, "r_net"]].dropna()
    x, y = d[feat].values, d["r_net"].values
    rho = spearman(x, y)
    shifts = rng.integers(len(x) // 10, len(x) - len(x) // 10, N_PERM)
    null = np.array([spearman(np.roll(x, s), y) for s in shifts])
    p = float((np.abs(null) >= abs(rho)).mean())
    q = pd.qcut(d[feat].rank(method="first"), 3, labels=["低", "中", "高"])
    terc = d.groupby(q, observed=True)["r_net"].agg(["count", "mean", "sum"])
    return {"n": int(len(d)), "spearman": round(rho, 4), "perm_p_circular": round(p, 4),
            "tercile_mean_r": {k: round(float(v), 3) for k, v in terc["mean"].items()},
            "tercile_sum_r": {k: round(float(v), 1) for k, v in terc["sum"].items()},
            "tercile_range": {k: [round(float(d[feat][q == k].min()), 3), round(float(d[feat][q == k].max()), 3)]
                              for k in ["低", "中", "高"]}}


def r_equity_stats(weights: np.ndarray, r: np.ndarray) -> dict:
    eq = np.cumsum(weights * r)
    peak = np.maximum.accumulate(np.concatenate([[0], eq]))[1:]
    return {"sum_r": round(float(eq[-1]), 1), "max_dd_r": round(float((peak - eq).max()), 1),
            "ratio": round(float(eq[-1] / max((peak - eq).max(), 1e-9)), 2)}


def main() -> int:
    df = load_trades()
    df = df.join(market_features(df)).join(history_features(df))
    months_all = (pd.Timestamp(END) - pd.Timestamp(START)).days / 30.44
    out: dict = {"generated_at": datetime.now().isoformat(), "period": [START, END], "n_trades": int(len(df)),
                 "note": "探索的な記述分析。ルール変更の根拠にはしない。", "structure": {}, "feature_levels": {},
                 "predictive": {}, "rules": {}}

    out["structure"]["全期間"] = structure(df, months_all)
    for name, (a, b) in PERIODS.items():
        mask = (df["entry_time"] >= a) & (df["entry_time"] <= pd.Timestamp(b) + pd.Timedelta(days=1))
        months = (pd.Timestamp(b) - pd.Timestamp(a)).days / 30.44
        out["structure"][name] = structure(df[mask], months)
        out["structure"][name + "以外"] = structure(df[~mask], months_all - months)
        out["feature_levels"][name] = {f: [round(float(df.loc[mask, f].median()), 3),
                                           round(float(df.loc[~mask, f].median()), 3)] for f in FEATURES}
    # 年別の構造
    out["structure_by_year"] = {str(y): structure(g, 12) for y, g in df.groupby(df["entry_time"].dt.year)}

    rng = np.random.default_rng(42)
    for f in FEATURES:
        out["predictive"][f] = predictive(df, f, rng)

    # 予測力が有意だった指標の頑健性: 全期間の三分位の境界で区切り、期間別・通貨別の平均Rを見る
    robust = {}
    for f in ("vol_ratio", "past_r_20"):
        d = df.dropna(subset=[f])
        cut = np.nanpercentile(d[f], [33.3, 66.7])
        lab = np.where(d[f] < cut[0], "低", np.where(d[f] < cut[1], "中", "高"))
        seg = pd.cut(d["entry_time"], [pd.Timestamp("2021-01-01"), pd.Timestamp("2023-11-01"),
                                       pd.Timestamp("2025-04-01"), pd.Timestamp("2026-01-01")],
                     labels=["拡張期間", "Train", "Validation"], right=False)
        robust[f] = {
            "by_segment": {str(k): {t: round(float(v), 3) for t, v in g.groupby(lab[g.index.map(d.index.get_loc)])["r_net"].mean().items()}
                           for k, g in d.groupby(seg, observed=True)},
            "by_pair": {k: {t: round(float(v), 3) for t, v in g.groupby(lab[g.index.map(d.index.get_loc)])["r_net"].mean().items()}
                        for k, g in d.groupby("pair")},
        }
    out["robustness"] = robust

    # 単純なルールの当てはめ（R単位・1件1Rを基準に、条件で0.5倍または0倍）
    r = df["r_net"].values
    base = r_equity_stats(np.ones(len(r)), r)
    rules = {"基準（全件1R）": base}
    pr = df["past_r_20"].values
    rules["直前20件の平均Rが負なら0.5R"] = r_equity_stats(np.where(pr < 0, 0.5, 1.0), r)
    rules["直前20件の平均Rが負なら見送り"] = r_equity_stats(np.where(pr < 0, 0.0, 1.0), r)
    for f in ("vol_ratio", "er_120"):
        lo = np.nanpercentile(df[f], 33.3)
        rules[f"{f} が下位1/3なら0.5R"] = r_equity_stats(np.where(df[f].values < lo, 0.5, 1.0), r)
    out["rules"] = rules

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("structure", "feature_levels")}, ensure_ascii=False, indent=1))
    print(json.dumps(out["structure_by_year"], ensure_ascii=False))
    for f, v in out["predictive"].items():
        print(f, v["spearman"], v["perm_p_circular"], v["tercile_mean_r"], v["tercile_range"])
    for k, v in rules.items():
        print(k, v)
    print(json.dumps(out["robustness"], ensure_ascii=False))
    print(f"[出力] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
