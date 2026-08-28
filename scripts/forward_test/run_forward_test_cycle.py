"""SYS-FX012(EXP-FX000006) フォワードテスト・サイクル実行.

司令塔の明示指示(2026-08-22、「このまま進める」)により、Train+Validation
合計で必須KPI13/18(実効n・ペイオフレシオ・permutation有意性が未達)・C品質
チーム未レビューの状態のまま、実際の値動きに対するフォワードテスト(ペーパー
トレード、実発注なし)を開始する。

設計は改善ループ第2試行で確立した最良候補を完全凍結して使う(新規パラメータ
なし): detect_candidate1(N_BREAKOUT=3.5単独) + H1ダウ理論トレンド判定不能
イベント除外フィルター(zigzag threshold_atr_h1=2.0)。対象は4通貨(USD/JPY・
EUR/JPY・GBP/JPY・AUD/JPY)。エントリー層・出口・コストモデルはT-13確定版を
そのまま流用。

cutoff = 2026-08-15 06:00 JST (既存data/curated/ds-1.jsonの4通貨全ての最終
バーの直後)。cutoff以前のH1バーはATR/zigzagの文脈計算にのみ使い、新規
トレンドイベントの検出には使わない(Train/Validation/Testいずれとも非重複)。

データは`data/raw/ds-1-forward/*.csv`(git管理・永続、`scripts/data/fetch_ds1_ohlcv.py`の
`fetch_pair()`が書き出す)を正の情報源として使う。本スクリプトは毎回、
cutoff〜最新取得バーまでの全期間を再計算する(差分ではなく全期間再計算、
状態管理を単純化するため)。

**2026-08-27修正(OBS000009 不具合2)**: 従来は`data/curated/ds-1-forward.json`
(.gitignore対象)のみに依存していたが、GitHub Actionsのランナーは実行ごとに
まっさらな環境のため「既存ファイルへの追記マージ」が実行間で永続せず、
2026-08-15のフォワードテスト開始以降、cutoff以降のデータが実質「直近2日分の
ローリング窓」でしか見えていなかった(2026-08-19 21:00 JSTの3件のN_BREAKOUT
イベントを取りこぼし)。`load_m5_forward()`を`data/raw/ds-1-forward/*.csv`
(git管理下、ワークフロー実行をまたいで永続)を読むよう修正し、
`data_coverage`診断(cutoff以降のバー数・最大ギャップ)をledger出力に追加した。

出力:
  research/method-notes/sysfx012_forward_test_ledger.json
  (毎回上書き。cutoff以降の全トレード・エクイティカーブ・可能ならKPI評価を含む)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from analyze_n_breakout_h1_dow_trend_alignment import h1_dow_trend_direction  # noqa: E402
from backtest_vol_breakout_dow_theory import (  # noqa: E402
    select_non_overlapping_breakout_events, simulate_dow_theory_trend,
)
from backtest_vol_breakout_dow_theory_4pairs import SELECTED_PAIRS  # noqa: E402
from backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd import (  # noqa: E402
    ATR_TRAIL_MULTIPLIER_M5, BREAKEVEN_TRIGGER_R, COMMISSION_RATE_ROUND_TRIP,
    INITIAL_CAPITAL_USD, MAX_LEVERAGE, RISK_PCT_PER_TRADE,
    SLIPPAGE_PIPS_MARKET_LEG, SLIPPAGE_PIPS_STOP_TRIGGERED, SPREAD_PIPS,
    STOP_BUFFER_ATR_M5, TP_CUM_FRACTION, TP_LEVELS_TRAILONLY, pip_size,
)
from backtest_vol_continuation_candidates_trendfilter_4pairs_trainonly import (  # noqa: E402
    detect_candidate1,
)
from derive_vol_breakout_entry_params import N_BREAKOUT, to_h1  # noqa: E402
from evaluate_vol_breakout_dow_theory_kpi import evaluate_period  # noqa: E402
from price_shock_filter import make_price_shock_check  # noqa: E402
from minmax_fx_dt.backtest.permutation import (  # noqa: E402
    permutation_test_block, permutation_test_clustered,
)
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

CUTOFF = pd.Timestamp("2026-08-15 06:00:00")  # JST、ds-1.jsonの4通貨最終バー(05:55)の直後
DS1_JSON = ROOT / "data" / "curated" / "ds-1.json"
DS1_FORWARD_JSON = ROOT / "data" / "curated" / "ds-1-forward.json"
RAW_FORWARD_DIR = ROOT / "data" / "raw" / "ds-1-forward"
OUT_PATH = ROOT / "research" / "method-notes" / "sysfx012_forward_test_ledger.json"

# ---------------------------------------------------------------------------
# 複数戦略のフォワードテスト対応 (2026-08-28)
#
# 司令塔判断「1のあと2へ」(EXP-FX000021 amendment-02) により、SYS-FX026 を
# フォワードテストへ投入する。本スクリプトは元々 SYS-FX012 専用だったため、
# **既定の挙動を一切変えずに** 戦略を差し替えられるようパラメータ化した。
#
# `--strategy sysfx012` (既定) は従来と完全に同一の結果を返す (回帰確認済み)。
#
# 2戦略の違いは「検出層・トレール幅・リスク率・出力先」の4点のみで、
# データ読み込み・被覆診断・コストモデル・複利エクイティ・KPI評価は共有する。
# ---------------------------------------------------------------------------


def _detect_sysfx012(h1: pd.DataFrame, atr_h1: pd.Series) -> tuple[pd.Series, pd.Series]:
    """SYS-FX012: 候補①(N_BREAKOUT=3.5単独)。H1トレンド判定不能除外は別途適用."""
    return detect_candidate1(h1, atr_h1)


def _detect_sysfx026(h1: pd.DataFrame, atr_h1: pd.Series) -> tuple[pd.Series, pd.Series]:
    """SYS-FX026: 素の N_BREAKOUT (H1トレンドフィルターなし)。

    `backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd.find_trades_for_period()`
    と同一の検出ロジック (ratio >= N_BREAKOUT、方向はブレイクバーの陽陰) を、
    フォワード側の (up, down) ブールSeries形式に合わせて表現したもの。
    """
    ratio = (h1["high"] - h1["low"]) / atr_h1
    is_break = ratio >= N_BREAKOUT
    bullish = h1["close"] > h1["open"]
    return (is_break & bullish).fillna(False), (is_break & ~bullish).fillna(False)


STRATEGIES: dict[str, dict] = {
    "sysfx012": {
        "label": "SYS-FX012",
        "detect": _detect_sysfx012,
        "apply_h1_trend_filter": True,   # H1ダウ理論トレンド判定不能イベントを除外
        "trail_mult_m5": ATR_TRAIL_MULTIPLIER_M5,   # = stop_buffer × 1.0 (T-02)
        "risk_pct": RISK_PCT_PER_TRADE,             # = 1.00%
        "out_path": ROOT / "research" / "method-notes" / "sysfx012_forward_test_ledger.json",
        "design": "SYS-FX012凍結最良候補(候補①N_BREAKOUT単独+H1トレンド判定不能除外フィルター、4通貨)。"
                  "00-spec.md「フォワードテスト仕様」節に事前登録済み。新規パラメータなし",
        "note": "司令塔の明示指示(2026-08-22)により、必須KPI13/18未達・C品質チーム未レビューのまま"
                "フォワードテスト(ペーパートレード、実発注なし)を実施中。cutoff以降の全期間を毎回再計算する",
    },
    "sysfx026": {
        "label": "SYS-FX026",
        "detect": _detect_sysfx026,
        "apply_h1_trend_filter": False,  # SYS-FX026 の基盤は SYS-FX011 系でトレンドフィルター無し
        "trail_mult_m5": STOP_BUFFER_ATR_M5 * 3.0,  # 親spec で事前登録した3.0倍
        "risk_pct": 0.0065,                          # amendment-01 §3 の機械的導出 (0.65%)
        "out_path": ROOT / "research" / "method-notes" / "sysfx026_forward_test_ledger.json",
        "design": "SYS-FX026凍結構成(素のN_BREAKOUT+価格反応型ショック抑制、トレール幅3.0倍、"
                  "risk_pct 0.65%、4通貨)。全パラメータはTrain単独由来で、Validation/Testからは"
                  "一切導出していない(EXP-FX000021 amendment-01 §3・amendment-02 §2)",
        "note": "司令塔判断「1のあと2へ」(2026-08-28)により投入。Test判定はB(部分的に再現、"
                "未達=K5m 2.56<3.0)で採用GOは未具申(amendment-02 §10)。フォワードテストの目的は、"
                "ゲートを一切緩めずに実効nを本物の未見データで積み増し、K5mが基準線のどちら側に"
                "落ち着くのかを判定すること。ペーパートレード、実発注なし",
    },
}


def load_m5_forward(pair: str) -> pd.DataFrame:
    """既存ds-1.json(凍結、warmup文脈用) + data/raw/ds-1-forward/*.csv(git管理・永続)
    + data/curated/ds-1-forward.json(同一実行内のみの一時ファイル、あれば追加で重ねる)を結合。

    2026-08-27修正(OBS000009 不具合2): 従来はds-1-forward.json(.gitignore対象)のみに
    依存していたが、GitHub Actionsのランナーは毎回まっさらな環境のため、そのファイルへの
    「既存に追記マージ」が実行間で全く永続せず、cutoff以降のデータが実質「直近2日分の
    ローリング窓」でしか見えていなかった(2026-08-19 21:00 JSTの3件のN_BREAKOUTイベントを
    取りこぼしていたことを`scripts/analyze_post_breakout_trend_visual_check.py`の副産物で
    発見)。`data/raw/ds-1-forward/*.csv`(`fetch_ds1_ohlcv.fetch_pair()`が書き出す、ds-1と
    同じ命名規則のCSV)はgit管理下にありワークフロー実行をまたいで永続するため、これを
    正の情報源とする。ds-1-forward.jsonは同一実行内で直前にfetchした最新データを追加で
    重ねるためだけに残す(なくても動く)。
    """
    frames = []
    with DS1_JSON.open(encoding="utf-8") as f:
        ds1 = json.load(f)
    if pair in ds1["pairs"]:
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
        if pair in dsf["pairs"]:
            df = pd.DataFrame(dsf["pairs"][pair]["data"])
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            frames.append(df.set_index("timestamp"))

    if not frames:
        raise FileNotFoundError(f"{pair}: ds-1.json・data/raw/ds-1-forward・ds-1-forward.json のいずれにもデータが無い")
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


MAX_NORMAL_GAP_HOURS = 60  # 週末クローズ(土06:00〜月07:00 JST、祝日で若干伸びる場合あり)を許容する上限


def compute_data_coverage(m5_by_pair: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> dict:
    """cutoff以降のM5データに、想定より大きい欠落(ギャップ)がないかを診断する.

    2026-08-27追加(OBS000009 不具合2の再発防止)。`n_events_raw: 0`が「相場が静かだった」
    のか「データに穴が開いている」のかを区別できなかったことが今回の発見を遅らせた。
    cutoff自体も仮想の先頭バーとしてギャップ計算に含める(cutoff直後にデータが
    存在しない「立ち上がりの穴」も検出対象にするため)。土日のFX市場クローズ
    (最大`MAX_NORMAL_GAP_HOURS`時間、祝日で若干伸びる場合を許容)を超えるギャップが
    あれば`has_gap: true`とし、その区間を明示する。
    """
    coverage = {}
    for pair, m5 in m5_by_pair.items():
        fwd = m5[m5.index >= cutoff]
        if len(fwd) == 0:
            coverage[pair] = {"n_bars": 0, "max_gap_minutes": None, "has_gap": True,
                               "note": "cutoff以降のデータが1本も無い"}
            continue
        idx_with_cutoff = pd.DatetimeIndex([cutoff]).append(fwd.index)
        gaps = idx_with_cutoff.to_series().diff().dropna()
        max_gap = gaps.max()
        max_gap_at = gaps.idxmax()
        coverage[pair] = {
            "n_bars": len(fwd),
            "first_bar": str(fwd.index.min()),
            "last_bar": str(fwd.index.max()),
            "max_gap_minutes": round(max_gap.total_seconds() / 60, 1),
            "max_gap_before": str(max_gap_at),
            "has_gap": bool(max_gap > pd.Timedelta(hours=MAX_NORMAL_GAP_HOURS)),
        }
    return coverage


def find_forward_trades(pair: str, m5: pd.DataFrame, shock_check, cutoff: pd.Timestamp,
                         cfg: dict | None = None):
    """凍結済みの戦略構成を、cutoff以降のイベントのみに適用。

    cfg 省略時は SYS-FX012 構成 (従来の既定挙動と完全に同一)。

    cutoff以前のバーはATR(H1)・zigzagピボットの文脈計算にのみ使う(先読みなし、
    Validationでのwarmupと同じ扱い)。新規トレンドイベントの検出(=positions)は
    cutoff以降のH1バーに限定する。
    """
    h1 = to_h1(m5)
    atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    atr_m5 = atr_ind(m5["high"], m5["low"], m5["close"], length=14)

    cfg = cfg or STRATEGIES["sysfx012"]
    up, down = cfg["detect"](h1, atr_h1)
    positions, directions = [], []
    for i in range(len(h1)):
        if h1.index[i] < cutoff:
            continue
        if bool(up.iloc[i]):
            positions.append(i)
            directions.append("UP")
        elif bool(down.iloc[i]):
            positions.append(i)
            directions.append("DOWN")

    dedup_positions = select_non_overlapping_breakout_events(h1.index, positions, directions)
    dedup_directions = {pos: d for pos, d in zip(positions, directions)}

    n_events_dedup = len(dedup_positions)
    n_events_trendfiltered = 0
    trades = []
    for pos in dedup_positions:
        if cfg["apply_h1_trend_filter"]:
            h1_trend = h1_dow_trend_direction(h1, atr_h1, pos)
            if h1_trend is None:
                continue
        n_events_trendfiltered += 1
        direction = dedup_directions[pos]
        trades.extend(simulate_dow_theory_trend(
            m5, atr_m5, h1, atr_h1, pos, direction, STOP_BUFFER_ATR_M5, cfg["trail_mult_m5"],
            blackout_check=shock_check, tp_levels=TP_LEVELS_TRAILONLY, skip_first_entry=False,
            atr_trail_series=atr_m5, m5_exit=True, breakeven_trigger_r=BREAKEVEN_TRIGGER_R))
    return trades, len(positions), n_events_dedup, n_events_trendfiltered


def run_forward_cycle(cfg: dict | None = None) -> dict:
    cfg = cfg or STRATEGIES["sysfx012"]
    m5_by_pair, h1_by_pair, atr_h1_by_pair = {}, {}, {}
    latest_bar_by_pair = {}
    for pair in SELECTED_PAIRS:
        m5 = load_m5_forward(pair)
        m5_by_pair[pair] = m5
        h1_by_pair[pair] = to_h1(m5)
        atr_h1_by_pair[pair] = atr_ind(h1_by_pair[pair]["high"], h1_by_pair[pair]["low"],
                                        h1_by_pair[pair]["close"], length=14)
        latest_bar_by_pair[pair] = str(m5.index.max())
    data_coverage = compute_data_coverage(m5_by_pair, CUTOFF)
    shock_check = make_price_shock_check(h1_by_pair, atr_h1_by_pair)

    all_trades: list[dict] = []
    still_open_trades: list[dict] = []
    n_raw_total = n_dedup_total = n_trendfiltered_total = 0
    for pair, m5 in m5_by_pair.items():
        trades, n_raw, n_dedup, n_trendfiltered = find_forward_trades(pair, m5, shock_check, CUTOFF, cfg)
        n_raw_total += n_raw
        n_dedup_total += n_dedup
        n_trendfiltered_total += n_trendfiltered
        spread = SPREAD_PIPS.get(pair, 0.5)
        pip = pip_size(pair)
        for sim in trades:
            # data_exhausted=True: 決済判定用の未来バーが単に足りないために
            # simulate_scaled_scheme()が強制決済(NO_M5_DATA_AFTER_ENTRY/MAX_HOLD)
            # を返したケース。Train/Validation/Testでは稀だが、フォワードテストは
            # データの先端=常に「現在」のため、直近72時間以内のエントリーは
            # 毎回これに該当しうる(2026-08-22、C品質チームレビューで発覚)。
            # まだ結果が確定していないポジションなので、r_net・コストモデル計算を
            # 行わず、複利計算・KPIから除外して「保有中」として別扱いする。
            # 十分な実データが蓄積されれば、翌週以降の全期間再計算で自然に
            # 決着済みトレードとして再評価される。
            if sim.get("data_exhausted"):
                still_open_trades.append({
                    "pair": pair, "direction": sim["direction"],
                    "entry_time": pd.Timestamp(sim["entry_time"]),
                    "entry_price": sim["entry_price"], "initial_risk": sim["initial_risk"],
                })
                continue
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
            all_trades.append({
                "pair": pair, "direction": sim["direction"],
                "entry_time": pd.Timestamp(sim["entry_time"]), "exit_time": sim["exit_time"],
                "entry_price": sim["entry_price"], "initial_risk": sim["initial_risk"],
                "exit_reason": sim["exit_reason"], "n_levels_hit": sim["n_levels_hit"],
                "fraction_via_tp": fraction_via_tp,
                "r_gross": sim["r"], "cost_r": cost_r, "commission_r": commission_r,
                "r_net": r_net, "leverage_ratio": leverage_ratio,
            })

    all_trades.sort(key=lambda t: t["entry_time"])
    events = []
    for idx, t in enumerate(all_trades):
        events.append((t["entry_time"], 0, idx, "ENTRY"))
        events.append((t["exit_time"], 1, idx, "EXIT"))
    events.sort(key=lambda e: (e[0], e[1]))

    balance = INITIAL_CAPITAL_USD
    ruined = False
    equity_curve = [{"time": str(CUTOFF), "balance": balance}]
    for time_, _order, idx, kind in events:
        t = all_trades[idx]
        if kind == "ENTRY":
            if ruined:
                t["risk_dollars"] = 0.0
                t["skipped_ruin"] = True
            else:
                risk_pct = cfg["risk_pct"]
                max_risk_pct = MAX_LEVERAGE / t["leverage_ratio"] if t["leverage_ratio"] > 0 else risk_pct
                effective_risk_pct = min(risk_pct, max_risk_pct)
                t["risk_dollars"] = balance * effective_risk_pct
                t["effective_risk_pct"] = effective_risk_pct
                t["skipped_ruin"] = False
        else:
            if t.get("skipped_ruin"):
                t["dollar_pnl"] = 0.0
            else:
                t["dollar_pnl"] = t["r_net"] * t["risk_dollars"]
                balance += t["dollar_pnl"]
                if balance <= 0:
                    balance = 0.0
                    ruined = True
            t["balance_after"] = balance
            equity_curve.append({"time": str(time_), "balance": balance})

    n_closed = len(all_trades)  # data_exhaustedはstill_open_tradesへ既に分離済みなので、残りは全て決着済み
    n_open = len(still_open_trades)
    n = n_closed + n_open
    r_values = [t["r_net"] for t in all_trades if not t.get("skipped_ruin")]
    pairs_for_perm = [t["pair"] for t in all_trades if not t.get("skipped_ruin")]
    day_clusters_for_perm = [t["entry_time"].strftime("%Y-%m-%d") for t in all_trades if not t.get("skipped_ruin")]
    perm_result = permutation_test_clustered(r_values, pairs_for_perm, seed=42) if len(r_values) >= 4 else None
    perm_result_block = permutation_test_block(r_values, day_clusters_for_perm, seed=42) if len(r_values) >= 4 else None
    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r < 0]
    profit_factor_val = (sum(wins) / abs(sum(losses))) if losses else None
    payoff_val = (float(np.mean(wins)) / abs(float(np.mean(losses)))) if wins and losses else None
    win_rate = (len(wins) / len(r_values)) if r_values else None
    mean_r_net = float(np.mean(r_values)) if r_values else None

    all_output_trades = sorted(all_trades + still_open_trades, key=lambda t: t["entry_time"])
    period_result = {
        "period": "forward_test", "cutoff": str(CUTOFF),
        "n_events_raw": n_raw_total, "n_events_dedup": n_dedup_total,
        "n_events_trendfiltered": n_trendfiltered_total,
        "n_trades_total": n, "n_trades_closed": n_closed, "n_trades_open": n_open,
        "win_rate": round(win_rate, 4) if win_rate else None,
        "mean_r_net": round(mean_r_net, 4) if mean_r_net else None,
        "profit_factor": round(profit_factor_val, 3) if profit_factor_val else None,
        "payoff_ratio": round(payoff_val, 3) if payoff_val else None,
        "perm_p_clustered": round(perm_result.p_value, 4) if perm_result else None,
        "perm_p_block": round(perm_result_block.p_value, 4) if perm_result_block else None,
        "final_balance": round(balance, 2),
        "trades": [
            {k: (str(v) if isinstance(v, pd.Timestamp) else (round(v, 6) if isinstance(v, float) else v))
             for k, v in t.items()}
            for t in all_output_trades
        ],
        "equity_curve": equity_curve,
    }

    kpi = None
    closed_trades_for_kpi = [t for t in period_result["trades"] if "dollar_pnl" in t]
    if len(closed_trades_for_kpi) > 0:
        kpi_input = dict(period_result)
        kpi_input["trades"] = closed_trades_for_kpi
        kpi = evaluate_period("forward_test", kpi_input, perm_p_field="perm_p_block",
                               apply_n_correlation_discount=False, apply_k3m_scale_invariant=True)

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "strategy": cfg["label"],
        "design": cfg["design"],
        "note": cfg["note"],
        "params": {"trail_mult_m5": cfg["trail_mult_m5"], "risk_pct": cfg["risk_pct"],
                   "apply_h1_trend_filter": cfg["apply_h1_trend_filter"]},
        "cutoff": str(CUTOFF),
        "latest_bar_by_pair": latest_bar_by_pair,
        "data_coverage": data_coverage,
        "backtest": period_result,
        "kpi": kpi,
    }
    cfg["out_path"].write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="フォワードテスト・サイクル実行")
    parser.add_argument("--strategy", default="sysfx012", choices=sorted(STRATEGIES),
                        help="対象戦略 (既定 sysfx012 = 従来と完全に同一の挙動)")
    args = parser.parse_args()
    cfg = STRATEGIES[args.strategy]

    print(f"=== {cfg['label']} フォワードテスト・サイクル実行 ===")
    print(f"cutoff = {CUTOFF} (JST)")
    print(f"パラメータ: trail_m5={cfg['trail_mult_m5']:.4f}  risk_pct={cfg['risk_pct']*100:.2f}%  "
          f"H1トレンドフィルター={cfg['apply_h1_trend_filter']}")
    out = run_forward_cycle(cfg)
    p = out["backtest"]
    print(f"\n最新バー: {out['latest_bar_by_pair']}")
    for pair, cov in out["data_coverage"].items():
        flag = " [GAP注意]" if cov.get("has_gap") else ""
        print(f"  データ被覆[{pair}]: {cov.get('n_bars')}本 {cov.get('first_bar')}〜{cov.get('last_bar')} "
              f"最大ギャップ={cov.get('max_gap_minutes')}分{flag}")
    filt_label = "判定不能除外後" if cfg["apply_h1_trend_filter"] else "フィルター無し"
    print(f"イベント={p['n_events_raw']}件(dedup後{p['n_events_dedup']}、{filt_label}{p['n_events_trendfiltered']})")
    print(f"トレード数={p['n_trades_total']}(決済済み{p['n_trades_closed']}、保有中{p['n_trades_open']})")
    print(f"最終残高=${p['final_balance']}  勝率={p['win_rate']}  PF={p['profit_factor']}  "
          f"ペイオフ={p['payoff_ratio']}  perm_p(block)={p['perm_p_block']}")
    if out["kpi"]:
        print(f"KPI: {out['kpi']['kpi_required_pass_count']}")
    else:
        print("KPI: 決済済みトレードなし、評価不可(想定通り。母数がまだ小さい初期段階)")
    print(f"\n[出力]: {cfg['out_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
