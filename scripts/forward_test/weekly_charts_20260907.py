"""SYS-FX012 週次レポート チャート生成 (2026-09-07 週).

(a) H1 分析チャート (4 通貨 × 1 ファイル)
(b) エントリーチャート (2x2 サブプロット)
(c) 決済チャート (2x2 サブプロット)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\Atsushi Hasebe\.minimax-agent\projects\minmax-fx-day-trading-lab")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from derive_vol_breakout_entry_params import to_h1  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DS1_JSON = PROJECT_ROOT / "data" / "curated" / "ds-1.json"
DS1_FORWARD_JSON = PROJECT_ROOT / "data" / "curated" / "ds-1-forward.json"
RAW_FORWARD_DIR = PROJECT_ROOT / "data" / "raw" / "ds-1-forward"
EVENTS_JSON = PROJECT_ROOT / "logs" / "forward_test_cycle" / "weekly_events_v4_2026-09-07.json"
SUMMARY_JSON = PROJECT_ROOT / "logs" / "forward_test_cycle" / "weekly_summary_v2_2026-09-07.json"
LEDGER_JSON = PROJECT_ROOT / "research" / "method-notes" / "sysfx012_forward_test_ledger.json"
CHART_DIR = PROJECT_ROOT / "logs" / "charts" / "2026-09-07"
CHART_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DATE = "2026-09-07"
CUTOFF = pd.Timestamp("2026-08-15 06:00:00")
LATEST = pd.Timestamp("2026-09-05 05:55:00")
PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]


def load_m5(pair: str) -> pd.DataFrame:
    frames = []
    with DS1_JSON.open(encoding="utf-8") as f:
        ds1 = json.load(f)
    if pair in ds1.get("pairs", {}):
        df = pd.DataFrame(ds1["pairs"][pair]["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        frames.append(df.set_index("timestamp"))
    for f in sorted(RAW_FORWARD_DIR.glob(f"ohlcv_{pair}_5min_*.csv")):
        df = pd.read_csv(f, parse_dates=["timestamp"])
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        frames.append(df.set_index("timestamp"))
    if DS1_FORWARD_JSON.exists():
        with DS1_FORWARD_JSON.open(encoding="utf-8") as f:
            dsf = json.load(f)
        if pair in dsf.get("pairs", {}):
            df = pd.DataFrame(dsf["pairs"][pair]["data"])
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            frames.append(df.set_index("timestamp"))
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def to_h1_local(m5: pd.DataFrame) -> pd.DataFrame:
    return m5.resample("1h", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()


# ----- (a) H1 分析チャート -----

def chart_h1_analysis(pair: str, events_by_pair: dict, m5_data: dict, n_breakout: float = 3.5):
    h1 = m5_data[pair]["h1"]
    h1w = h1[(h1.index >= CUTOFF) & (h1.index <= LATEST + pd.Timedelta(hours=1))]
    if h1w.empty:
        print(f"  [WARN] {pair}: no H1 data in window")
        return None

    events = events_by_pair.get(pair, [])
    up_vlines, down_vlines, excluded_vlines = [], [], []
    for e in events:
        ts = pd.Timestamp(e["time"])
        if ts not in h1w.index:
            continue
        if e["h1_trend"] is None:
            excluded_vlines.append(ts)
        elif e["direction"] == "UP":
            up_vlines.append(ts)
        else:
            down_vlines.append(ts)
    all_vlines = up_vlines + down_vlines + excluded_vlines
    all_colors = (["#2ca02c"] * len(up_vlines)
                  + ["#d62728"] * len(down_vlines)
                  + ["#888888"] * len(excluded_vlines))
    style = mpf.make_mpf_style(base_mpf_style="charles", gridstyle=":", y_on_right=True)
    title = f"{pair} H1 - Forward Test Week ({CUTOFF:%Y-%m-%d} - {REPORT_DATE})  N_events={len(events)}  N_BREAKOUT={n_breakout}"
    fig, axes = mpf.plot(
        h1w, type="candle", style=style,
        title=title, ylabel="Price",
        vlines=dict(vlines=all_vlines, colors=all_colors, linewidths=1.2, alpha=0.8),
        show_nontrading=False, returnfig=True,
        figsize=(13, 5.5),
    )
    ax = axes[0]
    for e in events:
        ts = pd.Timestamp(e["time"])
        if ts not in h1w.index:
            continue
        color = ("#2ca02c" if e["direction"] == "UP" and e["h1_trend"] is not None
                 else "#d62728" if e["direction"] == "DOWN" and e["h1_trend"] is not None
                 else "#888888")
        ax.annotate(f"{e['range_atr']:.1f}", xy=(ts, h1w.loc[ts, "high"]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8, color=color, fontweight="bold")
    out_path = CHART_DIR / f"chart_h1_analysis_{pair}_{REPORT_DATE}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return out_path


# ----- (b) エントリーチャート -----

def chart_entry(events_by_pair: dict, m5_data: dict, summary: dict):
    n_total = sum(len(events_by_pair.get(p, [])) for p in PAIRS)
    n_trend = sum(1 for p in PAIRS for e in events_by_pair.get(p, []) if e["h1_trend"] is not None and e["in_dedup"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=120)
    fig.suptitle(
        f"Entry chart - Forward Test Week ({CUTOFF:%Y-%m-%d} - {REPORT_DATE})\n"
        f"検出 {n_total} 件 (raw=13, dedup=9) → トレンド通過 {n_trend} → M5 エントリー 14 件 (EUR/GBP 集中)",
        fontsize=12, fontweight="bold")
    for ax, pair in zip(axes.flat, PAIRS):
        events = events_by_pair.get(pair, [])
        h1w = m5_data[pair]["h1w"]
        if h1w.empty:
            ax.text(0.5, 0.5, f"{pair}\n(no H1 data)", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        # H1 candle 背景
        mpf.plot(h1w, type="candle", ax=ax, style=mpf.make_mpf_style(base_mpf_style="charles"),
                 show_nontrading=False, axtitle=f"{pair} H1 + M5 close")
        # M5 close overlay
        m5 = m5_data[pair]["m5"]
        m5_in_window = m5[(m5.index >= h1w.index.min()) & (m5.index <= h1w.index.max() + pd.Timedelta(hours=1))]
        if not m5_in_window.empty:
            ax.plot(m5_in_window.index, m5_in_window["close"], color="#1f77b4", linewidth=0.5, alpha=0.4, label="M5 close")
        # 検出イベントマーカー
        n_ev = len(events)
        ax.text(0.99, 0.97, f"検出 {n_ev} 件", transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff4e5", edgecolor="#F97316"), fontsize=9)
        ax.legend(loc="lower left", fontsize=8)
    fig.text(0.5, 0.04,
             "※EUR_JPY / GBP_JPY で 9/2-9/4 の急落局面に複数のトレンド通過イベントが集中。USD_JPY は方向不一致で M5 エントリーなし。\n"
             "AUD_JPY は cutoff 期間中 raw 検出 0 件 (ATR が他 3 通貨より小さく N_BREAKOUT=3.5 未達)",
             ha="center", fontsize=9, color="#444")
    out_path = CHART_DIR / f"chart_entry_{REPORT_DATE}.png"
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return out_path


# ----- (c) 決済チャート -----

def chart_exit(ledger: dict, m5_data: dict):
    trades = ledger["backtest"]["trades"]
    n_trades = len(trades)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=120)
    fig.suptitle(f"Exit chart - Forward Test Week ({CUTOFF:%Y-%m-%d} - {REPORT_DATE})\n"
                 f"決済済みトレード: {n_trades} 件 (WIN {int(n_trades * ledger['backtest']['win_rate'])} / LOSS {n_trades - int(n_trades * ledger['backtest']['win_rate'])}, "
                 f"平均R {ledger['backtest']['mean_r_net']:+.3f}, 最終残高 ${ledger['backtest']['final_balance']:.2f})",
                 fontsize=12, fontweight="bold")
    for ax, pair in zip(axes.flat, PAIRS):
        h1w = m5_data[pair]["h1w"]
        if h1w.empty:
            ax.text(0.5, 0.5, f"{pair}\n(no H1 data)", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        p_trades = [t for t in trades if t["pair"] == pair]
        mpf.plot(h1w, type="candle", ax=ax, style=mpf.make_mpf_style(base_mpf_style="charles"),
                 show_nontrading=False, axtitle=f"{pair} (n={len(p_trades)})")
        m5 = m5_data[pair]["m5"]
        m5_in_window = m5[(m5.index >= h1w.index.min()) & (m5.index <= h1w.index.max() + pd.Timedelta(hours=1))]
        if not m5_in_window.empty:
            ax.plot(m5_in_window.index, m5_in_window["close"], color="#1f77b4", linewidth=0.5, alpha=0.4)
        # トレード exit markers
        for t in p_trades:
            et = pd.Timestamp(t["entry_time"])
            xt = pd.Timestamp(t["exit_time"])
            win = t["r_net"] > 0
            color = "#2ca02c" if win else "#d62728"
            ax.axvline(et, color="#F97316", linestyle="--", linewidth=0.8, alpha=0.7)  # エントリー
            ax.axvline(xt, color=color, linestyle="-", linewidth=1.2, alpha=0.9)  # 決済
            ax.scatter([et], [t["entry_price"]], marker="^", s=40, color=color, zorder=5,
                       edgecolors="black", linewidth=0.5)
            ax.annotate(f"{t['r_net']:+.2f}R", xy=(xt, t["entry_price"]),
                        xytext=(5, -8 if win else 8), textcoords="offset points",
                        fontsize=7, color=color, fontweight="bold")
        if p_trades:
            wr = sum(1 for t in p_trades if t["r_net"] > 0) / len(p_trades) * 100
            sum_r = sum(t["r_net"] for t in p_trades)
            ax.text(0.99, 0.97, f"n={len(p_trades)} WR={wr:.0f}% ΣR={sum_r:+.2f}",
                    transform=ax.transAxes, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff4e5", edgecolor="#F97316"),
                    fontsize=9)
        else:
            ax.text(0.5, 0.5, f"{pair}\n(トレードなし)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#666"))
    fig.text(0.5, 0.04,
             "全トレード方向 DOWN (EUR/GBP の 9/2-9/4 急落局面)、決済理由は全件 SL_INITIAL_NO_TP (trail-only 設計)。"
             " 緑=WIN, 赤=LOSS, オレンジ点線=エントリー, 実線=決済",
             ha="center", fontsize=9, color="#444")
    out_path = CHART_DIR / f"chart_exit_{REPORT_DATE}.png"
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return out_path


def main():
    print("=" * 78)
    print(f"チャート生成 ({REPORT_DATE})")
    print("=" * 78)
    with EVENTS_JSON.open(encoding="utf-8") as f:
        ev_data = json.load(f)
    with SUMMARY_JSON.open(encoding="utf-8") as f:
        summary = json.load(f)
    with LEDGER_JSON.open(encoding="utf-8") as f:
        ledger = json.load(f)
    events = ev_data["events"]
    events_by_pair = {p: [] for p in PAIRS}
    for e in events:
        events_by_pair[e["pair"]].append(e)

    m5_data = {}
    for pair in PAIRS:
        m5 = load_m5(pair)
        h1 = to_h1_local(m5)
        h1w = h1[(h1.index >= CUTOFF) & (h1.index <= LATEST + pd.Timedelta(hours=1))]
        m5_data[pair] = {"m5": m5, "h1": h1, "h1w": h1w}
        print(f"  {pair}: M5 {len(m5)}本, H1w {len(h1w)}本")

    print("\n--- (a) H1 分析チャート ---")
    for pair in PAIRS:
        chart_h1_analysis(pair, events_by_pair, m5_data)

    print("\n--- (b) エントリーチャート ---")
    chart_entry(events_by_pair, m5_data, summary)

    print("\n--- (c) 決済チャート ---")
    chart_exit(ledger, m5_data)

    print(f"\n=== 出力先: {CHART_DIR} ===")


if __name__ == "__main__":
    main()
