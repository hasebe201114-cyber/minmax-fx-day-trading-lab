"""両建て・グリッド系(CLAUDE.md記載の未検証カテゴリ)の探索的サニティチェック
(Train期間のみ、正式プロトコル外・spec編集なし).

## 位置づけ

司令塔との対話「エッジとは幻。エッジが無いことを前提にした戦略が大事」を受けた
戦略カテゴリ棚卸しで、CLAUDE.md記載の「両建て・グリッド系」が一度も検証されて
いないと判明した。本スクリプトは、正式なSYS-FX起票(prescreen)の前段階として、
最も単純な形のグリッド戦略が基本的な経済性(コスト込みで正の期待値の可能性が
あるか)を持つかを確認する探索的診断。

## 戦略設計(第一版、結果を見てからの調整はしない)

- 時間軸: H4(中期スイングという本PJスコープに合わせ、H1グリッドより粗い間隔)
- 再アンカー: REANCHOR_BARS(30本≈5日)ごとに、その時点の終値を中心(center)、
  ATR(H4,14)をグリッド間隔の基準として再設定する(グリッド自体は期間内固定、
  移動平均に追従させる設計は複雑化するため見送る)
- グリッド: 中心の上下にN_LEVELS段(既定5)、間隔=GRID_STEP_ATR_MULT×ATR
- エントリー: 価格が買いレベルを下抜けたら1ロット買い(未保有の場合のみ)、
  売りレベルを上抜けたら1ロット売り(同)
- 利確: 保有中の各ポジションは、価格が「エントリーレベル+1グリッド刻み」
  (買い)/「エントリーレベル-1グリッド刻み」(売り)に達したら決済、
  再度そのレベルで新規建てできる状態に戻す(グリッドの往復収益モデル)
- 損切り: 価格が最外レベル(N_LEVELS段目)からさらに1グリッド刻み外側へ抜けたら、
  そちら側の保有ポジション全てを成行決済し、その側は次の再アンカーまで新規停止
  (トレンド発生時の損失を打ち切る安全弁)
- 両建て: 買い側・売り側は独立に管理し、同時に保有しうる(K7m相当の証拠金消費率は
  別途集計する)

出力: research/method-notes/grid_strategy_sanity_check.json
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

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
REANCHOR_BARS = 30  # H4本数、約5日
N_LEVELS = 5
GRID_STEP_ATR_MULT = 1.0
LOT_RISK_DOLLARS = 10.0  # 1グリッド刻み分の値動きに対する固定ロット相当額(比較用の単純化)
SPREAD_PIPS = {"USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6}
SLIPPAGE_PIPS = 0.5
COMMISSION_RATE_ROUND_TRIP = 0.00004


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def load_m5(pair: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "ds-1" / f"ohlcv_{pair}_5min_*.csv")))
    frames = [pd.read_csv(f, parse_dates=["timestamp"]) for f in files]
    df = pd.concat(frames).drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    return df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]


def to_h4(m5: pd.DataFrame) -> pd.DataFrame:
    agg = [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]
    return pd.DataFrame({c: m5[c].resample("4h").agg(a) for c, a in agg}).dropna()


def simulate_pair(pair: str) -> dict:
    m5 = load_m5(pair)
    h4 = to_h4(m5)
    atr_h4 = atr_ind(h4["high"], h4["low"], h4["close"], length=14)
    pip = pip_size(pair)
    spread = SPREAD_PIPS[pair]

    trades: list[dict] = []
    exposure_snapshots: list[int] = []  # 各バーでの同時保有ポジション数(両側合計)

    i = 0
    n = len(h4)
    while i < n:
        anchor_pos = i
        if not np.isfinite(atr_h4.iloc[anchor_pos]) or atr_h4.iloc[anchor_pos] <= 0:
            i += 1
            continue
        center = float(h4["close"].iloc[anchor_pos])
        step = float(atr_h4.iloc[anchor_pos]) * GRID_STEP_ATR_MULT
        buy_levels = [center - k * step for k in range(1, N_LEVELS + 1)]
        sell_levels = [center + k * step for k in range(1, N_LEVELS + 1)]
        buy_stop = center - (N_LEVELS + 1) * step
        sell_stop = center + (N_LEVELS + 1) * step

        buy_open = [None] * N_LEVELS  # entry_priceを保持、Noneなら未保有
        sell_open = [None] * N_LEVELS
        buy_disabled = False
        sell_disabled = False

        period_end = min(anchor_pos + REANCHOR_BARS, n)
        for j in range(anchor_pos, period_end):
            hi, lo = float(h4["high"].iloc[j]), float(h4["low"].iloc[j])
            ts = h4.index[j]

            # 損切り判定(買い側)
            if not buy_disabled and lo <= buy_stop:
                for k in range(N_LEVELS):
                    if buy_open[k] is not None:
                        entry = buy_open[k]
                        gross_pips = (buy_stop - entry) / pip
                        cost_pips = spread + 2 * SLIPPAGE_PIPS
                        net_pips = gross_pips - cost_pips
                        trades.append({"pair": pair, "side": "buy", "outcome": "STOP",
                                       "entry_time": str(ts), "net_pips": net_pips})
                        buy_open[k] = None
                buy_disabled = True
            # 損切り判定(売り側)
            if not sell_disabled and hi >= sell_stop:
                for k in range(N_LEVELS):
                    if sell_open[k] is not None:
                        entry = sell_open[k]
                        gross_pips = (entry - sell_stop) / pip
                        cost_pips = spread + 2 * SLIPPAGE_PIPS
                        net_pips = gross_pips - cost_pips
                        trades.append({"pair": pair, "side": "sell", "outcome": "STOP",
                                       "entry_time": str(ts), "net_pips": net_pips})
                        sell_open[k] = None
                sell_disabled = True

            # 利確判定(買い側): エントリーレベルより1刻み上に達したら決済
            if not buy_disabled:
                for k in range(N_LEVELS):
                    if buy_open[k] is not None and hi >= buy_levels[k] + step:
                        entry = buy_open[k]
                        gross_pips = step / pip
                        cost_pips = spread + 2 * SLIPPAGE_PIPS
                        net_pips = gross_pips - cost_pips
                        trades.append({"pair": pair, "side": "buy", "outcome": "TP",
                                       "entry_time": str(ts), "net_pips": net_pips})
                        buy_open[k] = None
            if not sell_disabled:
                for k in range(N_LEVELS):
                    if sell_open[k] is not None and lo <= sell_levels[k] - step:
                        entry = sell_open[k]
                        gross_pips = step / pip
                        cost_pips = spread + 2 * SLIPPAGE_PIPS
                        net_pips = gross_pips - cost_pips
                        trades.append({"pair": pair, "side": "sell", "outcome": "TP",
                                       "entry_time": str(ts), "net_pips": net_pips})
                        sell_open[k] = None

            # 新規エントリー判定
            if not buy_disabled:
                for k in range(N_LEVELS):
                    if buy_open[k] is None and lo <= buy_levels[k]:
                        buy_open[k] = buy_levels[k]
            if not sell_disabled:
                for k in range(N_LEVELS):
                    if sell_open[k] is None and hi >= sell_levels[k]:
                        sell_open[k] = sell_levels[k]

            n_open = sum(1 for x in buy_open if x is not None) + sum(1 for x in sell_open if x is not None)
            exposure_snapshots.append(n_open)

        # 期間末: 残っているポジションは時価決済(マーク・トゥ・マーケット)
        last_close = float(h4["close"].iloc[period_end - 1])
        for k in range(N_LEVELS):
            if buy_open[k] is not None:
                gross_pips = (last_close - buy_open[k]) / pip
                cost_pips = spread + 2 * SLIPPAGE_PIPS
                trades.append({"pair": pair, "side": "buy", "outcome": "MARK",
                               "entry_time": str(h4.index[period_end - 1]), "net_pips": gross_pips - cost_pips})
            if sell_open[k] is not None:
                gross_pips = (sell_open[k] - last_close) / pip
                cost_pips = spread + 2 * SLIPPAGE_PIPS
                trades.append({"pair": pair, "side": "sell", "outcome": "MARK",
                               "entry_time": str(h4.index[period_end - 1]), "net_pips": gross_pips - cost_pips})

        i = period_end

    return {"trades": trades, "exposure_snapshots": exposure_snapshots}


def main() -> int:
    print("=== 両建て・グリッド系 探索的サニティチェック (Train期間のみ) ===\n")

    all_trades = []
    all_exposure = []
    per_pair = {}
    for pair in PAIRS:
        res = simulate_pair(pair)
        all_trades.extend(res["trades"])
        all_exposure.extend(res["exposure_snapshots"])
        pips = [t["net_pips"] for t in res["trades"]]
        n_tp = sum(1 for t in res["trades"] if t["outcome"] == "TP")
        n_stop = sum(1 for t in res["trades"] if t["outcome"] == "STOP")
        n_mark = sum(1 for t in res["trades"] if t["outcome"] == "MARK")
        per_pair[pair] = {
            "n_trades": len(res["trades"]), "n_tp": n_tp, "n_stop": n_stop, "n_mark": n_mark,
            "sum_net_pips": round(sum(pips), 2) if pips else None,
            "mean_net_pips": round(float(np.mean(pips)), 4) if pips else None,
            "max_concurrent_exposure": max(res["exposure_snapshots"]) if res["exposure_snapshots"] else 0,
        }
        print(f"[{pair}] トレード数={len(res['trades'])}(TP={n_tp}/STOP={n_stop}/MARK={n_mark}) "
              f"合計net_pips={per_pair[pair]['sum_net_pips']} 平均net_pips={per_pair[pair]['mean_net_pips']} "
              f"最大同時保有={per_pair[pair]['max_concurrent_exposure']}")

    pooled_pips = [t["net_pips"] for t in all_trades]
    wins = [p for p in pooled_pips if p > 0]
    losses = [p for p in pooled_pips if p < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else None
    win_rate = len(wins) / len(pooled_pips) if pooled_pips else None
    payoff = (float(np.mean(wins)) / abs(float(np.mean(losses)))) if wins and losses else None

    by_outcome = {}
    for outcome in ("TP", "STOP", "MARK"):
        sub = [t["net_pips"] for t in all_trades if t["outcome"] == outcome]
        if sub:
            by_outcome[outcome] = {
                "n": len(sub), "sum_pips": round(float(sum(sub)), 2),
                "mean_pips": round(float(np.mean(sub)), 4),
                "min_pips": round(float(min(sub)), 2), "max_pips": round(float(max(sub)), 2),
            }
    by_outcome_side = {}
    for side in ("buy", "sell"):
        for outcome in ("TP", "STOP", "MARK"):
            sub = [t["net_pips"] for t in all_trades if t["outcome"] == outcome and t["side"] == side]
            if sub:
                by_outcome_side[f"{side}_{outcome}"] = {
                    "n": len(sub), "sum_pips": round(float(sum(sub)), 2), "mean_pips": round(float(np.mean(sub)), 4),
                }

    result = {
        "generated_at": datetime.now().isoformat(),
        "status": "探索的サニティチェック(正式プロトコル外・spec編集なし)",
        "question": "両建て・グリッド戦略は、最も単純な設計でコスト込み基本的な経済性を持つか",
        "period": {"train_start": TRAIN_START, "train_end": TRAIN_END},
        "pairs": PAIRS,
        "params": {
            "timeframe": "H4", "reanchor_bars": REANCHOR_BARS, "n_levels": N_LEVELS,
            "grid_step_atr_mult": GRID_STEP_ATR_MULT,
            "cost_model": f"スプレッド(通貨別)+スリッページ{SLIPPAGE_PIPS}pip×2(往復)、手数料は本サニティチェックでは簡略化のため未計上",
        },
        "per_pair": per_pair,
        "pooled": {
            "n_trades": len(pooled_pips),
            "win_rate": round(win_rate, 4) if win_rate else None,
            "profit_factor": round(pf, 3) if pf else None,
            "payoff_ratio": round(payoff, 4) if payoff else None,
            "mean_net_pips": round(float(np.mean(pooled_pips)), 4) if pooled_pips else None,
            "sum_net_pips": round(float(sum(pooled_pips)), 2) if pooled_pips else None,
            "max_concurrent_exposure_any_pair": max(p["max_concurrent_exposure"] for p in per_pair.values()),
        },
        "by_outcome": by_outcome,
        "by_outcome_side": by_outcome_side,
        "caveats": [
            "手数料(COMMISSION_RATE_ROUND_TRIP)は本サニティチェックでは計上していない(概算のみ)。",
            "ポジションサイズはグリッド段ごとに固定(ロット均一)、複利サイジングは未実装。",
            "K7m(両建て証拠金消費率)は本チェックでは概算できていない(最大同時保有ポジション数のみ記録)。",
            "H4クローズ時点の高安のみで判定しており、H4バー内の高安到達順序は考慮していない"
            "(同一バー内で利確と損切りが両方発生しうる場合の順序依存性は未対応)。",
            "スワップポイントは計上していない(両建てで正味計算が必要、DS-7流用が必要)。",
            "Train期間のみ。正式なKPI評価・Validation/Testでの検証は未実施。",
        ],
    }

    out_path = ROOT / "research" / "method-notes" / "grid_strategy_sanity_check.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n--- プール全体 ---")
    print(f"トレード数={result['pooled']['n_trades']} 勝率={result['pooled']['win_rate']} "
          f"PF={result['pooled']['profit_factor']} ペイオフ={result['pooled']['payoff_ratio']} "
          f"平均net_pips={result['pooled']['mean_net_pips']} 合計net_pips={result['pooled']['sum_net_pips']}")
    print(f"最大同時保有(全通貨中最大)={result['pooled']['max_concurrent_exposure_any_pair']}")
    print("\n--- outcome別 ---")
    for k, v in by_outcome.items():
        print(f"  {k}: n={v['n']} sum={v['sum_pips']} mean={v['mean_pips']} min={v['min_pips']} max={v['max_pips']}")
    print("\n--- buy/sell × outcome ---")
    for k, v in by_outcome_side.items():
        print(f"  {k}: n={v['n']} sum={v['sum_pips']} mean={v['mean_pips']}")
    print(f"\n出力: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
