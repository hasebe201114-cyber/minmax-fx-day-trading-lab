"""SLタイト化版(k=0.5)の Train/Validation/Test 3期間・$1,000初期資金バックテスト.

背景: Q30(`analyze_tighter_stop_diagnostic.py`)でTrain期間のみを対象に、
TP価格水準固定・ストップのみk倍にタイト化する簡易診断(コスト無し・R
マルチプルのみ)を行い、k=0.5が平均R最良(ただし非有意)と判明した。本
スクリプトはこのk=0.5設計を、(1) Train/Validation/Testの3期間独立評価
(SYS-FX009 v2と同一の期間分割)、(2) $1,000初期資金からの複利ポジション
サイジング、(3) 実運用コスト(スプレッド・スリッページ・手数料)を反映した
形で再評価し、あとから統計分析できるようトレード単位のデータを収集する。

設計 (事前登録・結果を見る前に固定):
    - 対象通貨: 5通貨、対象期間: SYS-FX009 v2(`run_train_val_test_fx009.py`)
      と同一のTrain(2023-11-01〜2025-03-31)/Validation(2025-04-01〜
      2025-11-30)/Test(2025-12-01〜2026-08-15)。各期間は独立に評価し、
      D1/H1指標もその期間のデータのみから計算する(TVT分離の既存方針を踏襲)
    - パターン検出パラメータ: `double_pattern_params_h1.json`(Trainデータの
      みから導出済み、Validation/Testでも変更せずそのまま使用)
    - ストップ: k=0.5(Q30の最良候補)。stop_price = entry_price ∓ k*risk_original
      (risk_originalは元のパターン幾何ベースのストップ距離、Q30と同じ定義)
    - TP価格水準: entry_price ± {1,2,3}*risk_original (risk_originalベース、
      変更しない。Q30と同一設計)
    - ポジションサイジング: 各トレードのエントリー時点の口座残高(直近で
      決済済みのトレードまでを反映した残高、含み損益は考慮しない簡易複利)
      の1%をリスク額とする(risk_pct=1.0%。本PJに既存の定量的な取引あたり
      リスク%規定は無いため、月間DD上限10%(K2m)と整合する保守的な標準値
      として本シミュレーション用に新規に採用する)
    - 初期資金: $1,000 (各期間の開始時点でリセット。Train→Validation→Test
      を1本の連続口座として複利継続はしない。TVT分離の趣旨=各期間を独立
      した検証として扱う既存方針に合わせる)
    - コスト: スプレッド往復(通貨別、`run_train_val_test_fx009.py`と同一の
      SPREAD_PIPSテーブル)+ スリッページ往復1.0pip(0.5pip×2、既存の
      0.5pip前提を往復に適用)+ 手数料往復0.004%(0.002%×2、CLAUDE.md記載の
      GMO外国為替FX手数料率)。いずれもRマルチプル換算(cost_R = コスト価格幅 /
      (k*risk_original))してから口座残高ベースのドルP&Lへ変換する
      (通貨換算レートを別途取得せずに済む、通貨に依存しない定式化)
    - 複数通貨の同時ポジションは、決済順にリアライズド残高を更新する
      イベント駆動方式で処理(含み損益によるサイジングへの影響は考慮しない
      簡易モデル)。残高が0以下になった場合はそれ以降の新規エントリーを
      スキップする(破産ガード)
    - 統計的有意性: 通貨間相関を考慮した`permutation_test_clustered()`
      (提案5で新設済み)をRマルチプル(r_net_real、コスト・複利前のポジション
      サイズに依存しない正規化値)に対して各期間ごとに実行

出力: research/method-notes/tighter_stop_1000usd_backtest.json
       (トレード単位の全データ・期間別エクイティカーブ・サマリ統計を含む)

制約・簡略化 (正直に明記):
    - ロットサイズ最小単位(GMOの実運用制約、1,000通貨単位等)による丸めは
      行わない。連続サイジングの理想化モデル
    - 建玉あたりの証拠金/レバレッジ制約(最大25倍)は強制せず、参考情報として
      leverage_ratio(entry_price / (k*risk_original))の分布のみ記録する
    - 方向性エッジ自体は本セッションのIC分析(提案1〜3・6)で既に非有意と
      確定済み。本シミュレーションはコスト込みでの実運用イメージを掴む
      ための参考値であり、正式な採用判断の根拠には使わない
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

import analyze_scaled_exit_diagnostic as base  # noqa: E402
from minmax_fx_dt.backtest.permutation import permutation_test_clustered  # noqa: E402
from minmax_fx_dt.strategy.support_resistance import zigzag_pivots_typed  # noqa: E402

PAIRS = base.PAIRS
TP_LEVELS = base.TP_LEVELS
MAX_HOLD_BARS = base.MAX_HOLD_BARS

PERIODS = {
    "train":      ("2023-11-01", "2025-03-31"),
    "validation": ("2025-04-01", "2025-11-30"),
    "test":       ("2025-12-01", "2026-08-15"),
}

SPREAD_PIPS = {
    "USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6, "EUR_USD": 0.3,
}
SLIPPAGE_PIPS_ROUND_TRIP = 1.0  # 0.5pip x 2 (エントリー+エグジット)
COMMISSION_RATE_ROUND_TRIP = 0.00004  # 0.002% x 2 (CLAUDE.md記載のGMO手数料率)

STOP_K = 0.5  # Q30の最良候補
RISK_PCT_PER_TRADE = 0.01  # 新規採用: 口座残高の1%
INITIAL_CAPITAL_USD = 1000.0


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def load_m5_period(pair: str, start: str, end: str) -> pd.DataFrame:
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= start) & (df.index <= end)]


def find_entries_for_period(pair: str, m5: pd.DataFrame, params: dict) -> list[dict]:
    """`analyze_scaled_exit_diagnostic.find_continuation_entries`と同一ロジックだが、
    Train固定ではなく渡されたm5(期間スライス済み)から検出する。"""
    h1 = base.to_h1(m5)
    d1 = base.to_d1(m5)
    atr_h1 = base.atr_ind(h1["high"], h1["low"], h1["close"], length=14)
    lt_dir = base.lt_direction_series(d1)

    pivots = zigzag_pivots_typed(h1["high"], h1["low"], atr_h1, params["zigzag_threshold_atr"])
    triplets = base.alternating_triplets(pivots)
    tol = params["pattern_tolerance_atr"]
    buffer_atr = params["stop_buffer_atr"]
    cap = params["break_search_cap_bars"]

    entries = []
    for idx1, kind1, idx2, _k2, idx3, _k3 in triplets:
        required_lt = "DOWN" if kind1 == "HIGH" else "UP"
        atr_neckline = atr_h1.iloc[idx2]
        if pd.isna(atr_neckline) or atr_neckline <= 0:
            continue
        p1 = float(h1["high" if kind1 == "HIGH" else "low"].iloc[idx1])
        p2 = float(h1["high" if kind1 == "HIGH" else "low"].iloc[idx3])
        if abs(p1 - p2) / float(atr_neckline) > tol:
            continue
        neckline = float(h1["low" if kind1 == "HIGH" else "high"].iloc[idx2])
        pattern_extreme = max(p1, p2) if kind1 == "HIGH" else min(p1, p2)

        search_end = min(idx3 + 1 + cap, len(h1))
        for j in range(idx3 + 1, search_end):
            broke = (
                (kind1 == "HIGH" and float(h1["low"].iloc[j]) < neckline)
                or (kind1 == "LOW" and float(h1["high"].iloc[j]) > neckline)
            )
            if not broke:
                continue
            lt_at_break = lt_dir.asof(h1.index[j])
            if lt_at_break != required_lt:
                break
            atr_entry = atr_h1.iloc[j]
            if pd.isna(atr_entry) or atr_entry <= 0:
                break
            entry_price = float(h1["close"].iloc[j])
            direction = "DOWN" if kind1 == "HIGH" else "UP"
            buffer = buffer_atr * float(atr_entry)
            stop0 = pattern_extreme + buffer if direction == "DOWN" else pattern_extreme - buffer
            initial_risk = abs(entry_price - stop0)
            if initial_risk <= 0:
                break
            entries.append(dict(
                pair=pair, direction=direction, entry_idx=j, entry_time=h1.index[j],
                entry_price=entry_price, initial_risk=initial_risk,
            ))
            break
    return entries, h1, atr_h1


def simulate_trade(h1: pd.DataFrame, atr_h1: pd.Series, entry: dict, trail_mult: float, stop_k: float) -> dict:
    """k倍タイトストップ+新方式(40/35/25%)段階利確。r(Rマルチプル)はrisk_original
    基準(kによらず同一モノサシ)。exit_timeも返す。"""
    direction = entry["direction"]
    entry_price = entry["entry_price"]
    risk = entry["initial_risk"]
    stop = entry_price - stop_k * risk if direction == "UP" else entry_price + stop_k * risk
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
        if base.is_weekend_close_time(ts):
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
                    new_stop = o - trail_mult * float(atr_i)
                    stop = max(stop, new_stop)
                else:
                    new_stop = o + trail_mult * float(atr_i)
                    stop = min(stop, new_stop)
        if remaining_fraction <= 1e-9:
            return {"r": realized_r, "exit_reason": "TP_FULL", "n_levels_hit": 3, "exit_time": ts}
    ts_last = h1.index[end - 1]
    c = float(h1["close"].iloc[end - 1])
    exit_r = (c - entry_price) / risk if direction == "UP" else (entry_price - c) / risk
    n_levels_hit = sum(1 for lv in levels if lv[3])
    return {"r": realized_r + remaining_fraction * exit_r, "exit_reason": "MAX_HOLD",
            "n_levels_hit": n_levels_hit, "exit_time": ts_last}


def run_period(period_name: str, start: str, end: str, params: dict, trail_mult: float) -> dict:
    print(f"\n=== {period_name}: {start} 〜 {end} ===")
    all_trades: list[dict] = []
    for pair in PAIRS:
        m5 = load_m5_period(pair, start, end)
        if len(m5) < 1000:
            print(f"  [{pair}] データ不足 ({len(m5)}bars)、スキップ")
            continue
        entries, h1, atr_h1 = find_entries_for_period(pair, m5, params)
        spread = SPREAD_PIPS.get(pair, 0.5)
        pip = pip_size(pair)
        cost_price = (2 * spread + SLIPPAGE_PIPS_ROUND_TRIP) * pip
        for e in entries:
            sim = simulate_trade(h1, atr_h1, e, trail_mult, STOP_K)
            risk_real_price = STOP_K * e["initial_risk"]
            r_real_gross = sim["r"] / STOP_K
            cost_r = cost_price / risk_real_price
            leverage_ratio = e["entry_price"] / risk_real_price
            commission_r = COMMISSION_RATE_ROUND_TRIP * leverage_ratio
            r_net_real = r_real_gross - cost_r - commission_r
            all_trades.append({
                "pair": pair, "direction": e["direction"],
                "entry_time": e["entry_time"], "exit_time": sim["exit_time"],
                "entry_price": e["entry_price"], "initial_risk_original": e["initial_risk"],
                "stop_k": STOP_K, "risk_real_price": risk_real_price,
                "exit_reason": sim["exit_reason"], "n_levels_hit": sim["n_levels_hit"],
                "r_diagnostic": sim["r"], "r_real_gross": r_real_gross,
                "cost_r": cost_r, "commission_r": commission_r, "r_net_real": r_net_real,
                "leverage_ratio": leverage_ratio,
            })
        print(f"  [{pair}] エントリー={len(entries)}件")

    # イベント駆動: エントリー順にリスク額を記録、決済順に残高を更新
    all_trades.sort(key=lambda t: t["entry_time"])
    events = []
    for idx, t in enumerate(all_trades):
        events.append((t["entry_time"], 0, idx, "ENTRY"))  # ENTRYを同時刻ならEXITより先に処理しない
        events.append((t["exit_time"], 1, idx, "EXIT"))
    events.sort(key=lambda e: (e[0], e[1]))

    balance = INITIAL_CAPITAL_USD
    ruined = False
    equity_curve = [{"time": str(pd.Timestamp(start)), "balance": balance}]
    for time_, _order, idx, kind in events:
        t = all_trades[idx]
        if kind == "ENTRY":
            if ruined:
                t["risk_dollars"] = 0.0
                t["skipped_ruin"] = True
            else:
                t["risk_dollars"] = balance * RISK_PCT_PER_TRADE
                t["skipped_ruin"] = False
        else:  # EXIT
            if t.get("skipped_ruin"):
                t["dollar_pnl"] = 0.0
            else:
                t["dollar_pnl"] = t["r_net_real"] * t["risk_dollars"]
                balance += t["dollar_pnl"]
                if balance <= 0:
                    balance = 0.0
                    ruined = True
            t["balance_after"] = balance
            equity_curve.append({"time": str(time_), "balance": balance})

    n = len(all_trades)
    n_effective_trades = sum(1 for t in all_trades if not t.get("skipped_ruin"))
    r_values = [t["r_net_real"] for t in all_trades if not t.get("skipped_ruin")]
    pairs_for_perm = [t["pair"] for t in all_trades if not t.get("skipped_ruin")]
    dollar_pnls = [t["dollar_pnl"] for t in all_trades if not t.get("skipped_ruin")]

    n_wins = sum(1 for r in r_values if r > 0)
    win_rate = n_wins / len(r_values) if r_values else None
    mean_r_net = float(np.mean(r_values)) if r_values else None
    final_balance = balance
    total_return_pct = (final_balance / INITIAL_CAPITAL_USD - 1.0) * 100.0

    balances = [pt["balance"] for pt in equity_curve]
    running_max = np.maximum.accumulate(balances) if balances else np.array([INITIAL_CAPITAL_USD])
    drawdowns = [(b - m) / m * 100.0 if m > 0 else 0.0 for b, m in zip(balances, running_max)]
    max_dd_pct = min(drawdowns) if drawdowns else 0.0

    perm_result = None
    if len(r_values) >= 4:
        perm_result = permutation_test_clustered(r_values, pairs_for_perm, seed=42)

    print(f"  トレード数={n} (破産ガードでスキップ={n - n_effective_trades})  最終残高=${final_balance:.2f}  "
          f"総リターン={total_return_pct:.1f}%  最大DD={max_dd_pct:.1f}%")
    print(f"  勝率={win_rate:.3f}  平均r_net_real={mean_r_net:.4f}"
          f"{f'  perm_p(cluster)={perm_result.p_value:.4f}' if perm_result else ''}")

    return {
        "period": period_name, "start": start, "end": end,
        "n_trades": n, "n_effective_trades": n_effective_trades,
        "final_balance_usd": round(final_balance, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "mean_r_net_real": round(mean_r_net, 4) if mean_r_net is not None else None,
        "perm_p_clustered": round(perm_result.p_value, 4) if perm_result else None,
        "leverage_ratio_stats": {
            "median": round(float(np.median([t["leverage_ratio"] for t in all_trades])), 1) if all_trades else None,
            "max": round(float(np.max([t["leverage_ratio"] for t in all_trades])), 1) if all_trades else None,
        },
        "trades": [
            {k: (str(v) if isinstance(v, pd.Timestamp) else (round(v, 6) if isinstance(v, float) else v))
             for k, v in t.items()}
            for t in all_trades
        ],
        "equity_curve": equity_curve,
    }


def main() -> int:
    print("=== SLタイト化版(k=0.5) Train/Validation/Test $1,000バックテスト ===")
    with (ROOT / "research" / "EXP-FX000003" / "10-result" / "double_pattern_params_h1.json").open(encoding="utf-8") as f:
        params = json.load(f)
    trail_mult = params["atr_trail_multiplier"]
    print(f"事前登録: stop_k={STOP_K}, risk_pct_per_trade={RISK_PCT_PER_TRADE}, "
          f"initial_capital=${INITIAL_CAPITAL_USD}")

    period_results = {}
    for period_name, (start, end) in PERIODS.items():
        period_results[period_name] = run_period(period_name, start, end, params, trail_mult)

    print(f"\n=== サマリ ===")
    print(f"{'期間':<12}{'取引数':>8}{'最終残高':>12}{'総リターン':>10}{'最大DD':>8}{'勝率':>8}{'perm_p':>8}")
    for name, r in period_results.items():
        print(f"{name:<12}{r['n_trades']:>8}{'$'+str(r['final_balance_usd']):>12}"
              f"{str(r['total_return_pct'])+'%':>10}{str(r['max_drawdown_pct'])+'%':>8}"
              f"{r['win_rate']:>8}{r['perm_p_clustered']:>8}")

    out_path = ROOT / "research" / "method-notes" / "tighter_stop_1000usd_backtest.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "design": {
                "stop_k": STOP_K, "risk_pct_per_trade": RISK_PCT_PER_TRADE,
                "initial_capital_usd": INITIAL_CAPITAL_USD,
                "tp_levels": TP_LEVELS, "spread_pips": SPREAD_PIPS,
                "slippage_pips_round_trip": SLIPPAGE_PIPS_ROUND_TRIP,
                "commission_rate_round_trip": COMMISSION_RATE_ROUND_TRIP,
            },
            "periods": period_results,
            "_note": (
                "Q30で選定したk=0.5タイトストップ設計を、Train/Validation/Test"
                "3期間・$1,000初期資金・実運用コスト込みで再評価。各期間は独立に"
                "$1,000からリスタート(TVT分離の既存方針に合わせ複利は期間を跨がない)。"
                "方向性エッジ自体は既に非有意と確定済みのため、正式な採用判断の根拠"
                "ではなく、実運用イメージ把握とトレード単位データ収集が目的。"
            ),
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
