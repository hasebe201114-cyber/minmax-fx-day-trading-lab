"""逆張り(反発シグナル+50%TP)戦略の再設計: SL幅にコストフロアを課す(Train期間のみ・探索的).

## 位置づけ(HARKing防止のため明記)

- `scripts/analyze_counter_trend_reversal_entry.py`(v1)の結果を受けた再設計。
  v1は「浅めSL(0.1ATR)」でRRこそ良好(median 2.06)だったが、SL幅(pips換算で
  0.27〜0.79pips)に対して往復コスト(1.6〜2.4pips)が2.8〜5.9倍もあり、
  コスト込み期待値が全パターンでマイナスだった(-0.059〜-0.091R)。
- 本スクリプトは**正式な検証プロトコル外の探索的診断**であり、`00-spec.md`等の
  事前登録文書は一切編集しない。使用期間は**Train (2023-11-01〜2025-03-31)のみ**。
- v1のコンポーネント(反発シグナル・TP=50%戻し・コストモデル)はそのまま
  `analyze_counter_trend_reversal_entry`からimportして再利用し、SLの設計だけを変更する。

## 再設計の内容

「SL幅は往復コストの最低N倍を確保する」というフロアを課す:

    risk_structural = |entry_price - running_extreme|  (反発を確認した安値/高値そのもの、
                                                          v1の"buffer"を廃した最もタイトな
                                                          構造的ストップ)
    risk_used = max(risk_structural, floor_mult × round_trip_cost_price)
    SL価格 = risk_usedをentry_priceから逆算

floor_mult ∈ {0(フロアなし=構造的ストップそのまま), 3, 5, 7} で感度を見る。
TPは v1 と同じ「ブレイクバーレンジの50%戻し」で固定(構造的な水準のまま、
SLフロアとは独立)。

出力: research/method-notes/counter_trend_cost_floored_sl.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from minmax_fx_dt.strategy.indicators import atr as atr_ind
from analyze_counter_trend_reversal_entry import (  # noqa: E402
    ATR_LEN, MAX_HOLD_HOURS, N_BREAKOUT, PAIRS, PREP_MINUTES, SIGNAL_SEARCH_HOURS,
    SLIPPAGE_PIPS, SPREAD_PIPS, TP_FRAC, TRAIN_END, TRAIN_START, find_reversal_signal,
    load_m5, resample, simulate_trade,
)

FLOOR_MULTIPLES = [0.0, 3.0, 4.0, 5.0, 6.0, 7.0]
BOOTSTRAP_SEED = 20260827
N_BOOTSTRAP = 5000


def main() -> int:
    print("=== 逆張り再設計: SL幅にコストフロアを課す (Train期間のみ・探索的) ===\n")

    results: dict = {
        sw: {f"floor_{fm:g}x": [] for fm in FLOOR_MULTIPLES}
        for sw in SIGNAL_SEARCH_HOURS
    }
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
            d = int(body_dir[pos])
            confirmed_time = h1.index[pos] + pd.Timedelta(hours=1)
            search_start = confirmed_time + pd.Timedelta(minutes=PREP_MINUTES)

            if d < 0:
                tp_price = float(bar["low"] + TP_FRAC * rng)
                counter_dir = +1
            else:
                tp_price = float(bar["high"] - TP_FRAC * rng)
                counter_dir = -1

            for sw_name, sw_hours in SIGNAL_SEARCH_HOURS.items():
                search_end = confirmed_time + pd.Timedelta(hours=sw_hours)
                sig = find_reversal_signal(m5, atr_m5, search_start, search_end, d)
                if sig is None:
                    continue
                n_signal_found[sw_name] += 1
                entry_time = pd.Timestamp(sig["entry_time"])
                entry_price = sig["entry_price"]
                risk_structural = abs(entry_price - sig["running_extreme"])
                if risk_structural <= 0:
                    continue

                for fm in FLOOR_MULTIPLES:
                    key = f"floor_{fm:g}x"
                    risk_used = max(risk_structural, fm * rt_cost_price)
                    sl_price = entry_price - counter_dir * risk_used
                    sim = simulate_trade(m5, entry_time, entry_price, sl_price, tp_price,
                                          counter_dir, MAX_HOLD_HOURS)
                    reward = abs(tp_price - entry_price)
                    rr = reward / risk_used
                    r_gross = None
                    if sim["outcome"] == "TP":
                        r_gross = rr
                    elif sim["outcome"] == "SL":
                        r_gross = -1.0
                    elif sim["outcome"] == "TIMEOUT":
                        r_gross = counter_dir * (sim["exit_price"] - entry_price) / risk_used
                    cost_r = rt_cost_price / risk_used
                    r_net = (r_gross - cost_r) if r_gross is not None else None
                    results[sw_name][key].append({
                        "pair": pair, "outcome": sim["outcome"],
                        "risk_price": risk_used, "risk_structural": risk_structural,
                        "was_floored": bool(risk_used > risk_structural),
                        "reward_price": reward, "rr": rr,
                        "r_gross": r_gross, "cost_r": cost_r, "r_net": r_net,
                    })

    def summarize(trades: list[dict], pair_pip: dict[str, float], rng: np.random.Generator) -> dict:
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
        floored_frac = sum(1 for t in trades if t["was_floored"]) / n
        risk_pips = [t["risk_price"] / pair_pip.get(t["pair"], 0.0001) for t in trades]
        if len(r_net_list) >= 10:
            arr = np.asarray(r_net_list)
            boot = rng.choice(arr, size=(N_BOOTSTRAP, arr.size), replace=True).mean(axis=1)
            ci95 = [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]
        else:
            ci95 = None
        return {
            "n": n,
            "by_outcome": by_outcome,
            "win_rate_decided_TP_vs_SL": round(win_rate, 4) if win_rate is not None else None,
            "median_rr_reward_over_risk": round(float(np.median(rr_list)), 3),
            "mean_r_gross_before_cost": round(float(np.mean(r_gross_list)), 4) if r_gross_list else None,
            "mean_cost_r": round(float(np.mean(cost_r_list)), 4),
            "mean_r_net": round(float(np.mean(r_net_list)), 4) if r_net_list else None,
            "median_r_net": round(float(np.median(r_net_list)), 4) if r_net_list else None,
            "profit_factor_r_net": round(pf, 3) if pf else None,
            "positive_r_net_rate": round(sum(1 for r in r_net_list if r > 0) / len(r_net_list), 4) if r_net_list else None,
            "fraction_floored_by_cost": round(floored_frac, 4),
            "mean_risk_pips": round(float(np.mean(risk_pips)), 3),
            "mean_r_net_ci95_bootstrap": ci95,
        }

    pair_pip = {p: (0.01 if "JPY" in p else 0.0001) for p in PAIRS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    out = {
        "generated_at": datetime.now().isoformat(),
        "status": "探索的診断(正式プロトコル外・spec編集なし)。v1(counter_trend_reversal_entry.json)の再設計",
        "question": "SL幅に「往復コストの最低N倍」というフロアを課せば、逆張り(反発シグナル+50%TP)の"
                    "コスト込み期待値はプラスに転じるか",
        "period": {"train_start": TRAIN_START, "train_end": TRAIN_END},
        "pairs": PAIRS,
        "n_breakout_events_total": int(n_breakout_events_total),
        "n_signal_found_by_search_window": n_signal_found,
        "definitions": {
            "reversal_signal": "v1と同一(analyze_counter_trend_reversal_entry.find_reversal_signal)",
            "risk_structural": "|entry_price - running_extreme|。v1の0.1/0.5ATRバッファを廃した"
                               "最もタイトな構造的ストップ(反発を確認した安値/高値そのもの)",
            "sl_floor": "risk_used = max(risk_structural, floor_mult × 往復コスト)。"
                       f"floor_mult ∈ {FLOOR_MULTIPLES}(0=フロアなし)",
            "tp": "v1と同一: ブレイクバー(H1)のレンジの50%戻し水準(SLフロアとは独立、固定)",
            "cost_model": f"往復スプレッド×2+スリッページ{SLIPPAGE_PIPS}pip×2(v1と同一)",
            "max_hold_hours": MAX_HOLD_HOURS,
        },
        "results": {
            sw: {f"floor_{fm:g}x": summarize(results[sw][f"floor_{fm:g}x"], pair_pip, rng) for fm in FLOOR_MULTIPLES}
            for sw in SIGNAL_SEARCH_HOURS
        },
        "caveats": [
            "反発シグナルが探索窓内に出たイベントのみを対象(v1と同一の除外)。",
            "1トレンドイベントにつき1トレードのみ(ピラミッディングなし)。",
            "週末クローズ・両建て証拠金制約は考慮していない。",
            "TPはSLフロアと無関係に固定しているため、SLが広がるほどRRは機械的に縮む"
            "(フロアが効くほど『勝率は上がるがRRが下がる』というトレードオフになる)。",
            "Train期間のみ。Validation/Testは正式検証のために温存している。",
        ],
    }

    out_path = ROOT / "research" / "method-notes" / "counter_trend_cost_floored_sl.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"H1ブレイクイベント総数: {n_breakout_events_total}")
    for sw in SIGNAL_SEARCH_HOURS:
        print(f"\n--- 探索窓={sw} (シグナル成立 {n_signal_found[sw]}/{n_breakout_events_total} "
              f"= {n_signal_found[sw]/max(1,n_breakout_events_total):.1%}) ---")
        for fm in FLOOR_MULTIPLES:
            key = f"floor_{fm:g}x"
            s = out["results"][sw][key]
            print(f"  [{key}] n={s['n']} フロア発動率={s.get('fraction_floored_by_cost')} "
                  f"平均SL幅={s.get('mean_risk_pips')}pips "
                  f"win_rate={s.get('win_rate_decided_TP_vs_SL')} "
                  f"median_RR={s.get('median_rr_reward_over_risk')} "
                  f"r_gross={s.get('mean_r_gross_before_cost')} "
                  f"r_net={s.get('mean_r_net')} 95%CI={s.get('mean_r_net_ci95_bootstrap')} "
                  f"PF={s.get('profit_factor_r_net')} "
                  f"positive_rate={s.get('positive_r_net_rate')}")
    print(f"\n出力: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
