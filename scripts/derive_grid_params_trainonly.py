"""EXP-FX000018 フェーズゲート2: グリッドパラメータのデータ駆動導出 (Train期間のみ).

`research/EXP-FX000018/00-spec.md` §2 で事前登録した導出アルゴリズム D1〜D3 を
そのまま実装する。**損益・勝率・PF等の成績指標を一切参照せず、価格構造の統計量
(分散比・変位分布・価格エンベロープ)のみからグリッド段数N・間隔k・再アンカー周期R
を決める**(グリッド間隔を損益で最適化すると、それ自体がHARKingになるため)。

- D1: 再アンカー周期 R = argmin_{h in [6,120]} VR(h) の4通貨中央値を6の倍数に丸め、
      週末フラット制約により [6, 30] にクリップ
- D2: グリッド間隔 k = median |P_{t+h*} - P_t| / ATR_t、h* = argmin_{h in [1,30]} VR(h)
      の4通貨中央値。サニティ帯 [0.4, 2.5] にクリップ
- D3: グリッド段数 N = round(Env_80 / k) - 1、[3, 12] にクリップ。
      Env は R バー先までの ATR 正規化価格エンベロープ、80パーセンタイル

併せて §8 の通貨除外判定(損益非依存の構造的基準 cost_ratio_pair)も算出する。

出力: research/EXP-FX000018/10-result/grid_params.json
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

SPREAD_PIPS = {"USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6}
SLIPPAGE_PIPS_STOP = 1.0  # T-09 逆指値

# spec §2 で事前登録した探索レンジ・丸め・クリップ
R_SEARCH_MIN, R_SEARCH_MAX = 6, 120
R_ROUND_MULTIPLE = 6          # H4で1営業日
R_CLIP_MIN, R_CLIP_MAX = 6, 30  # 週末フラット制約 (1営業週 = H4 30本)
HSTAR_SEARCH_MIN, HSTAR_SEARCH_MAX = 1, 30
K_CLIP_MIN, K_CLIP_MAX = 0.4, 2.5
ENV_PERCENTILE = 80
N_CLIP_MIN, N_CLIP_MAX = 3, 12
ATR_LENGTH = 14
EXCLUSION_COST_RATIO_MULTIPLE = 2.0  # spec §8


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def load_m5_train(pair: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "ds-1" / f"ohlcv_{pair}_5min_*.csv")))
    if not files:
        raise FileNotFoundError(f"DS-1のM5 CSVが見つかりません: {pair}")
    frames = [pd.read_csv(f, parse_dates=["timestamp"]) for f in files]
    df = pd.concat(frames).drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    return df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]


def to_h4(m5: pd.DataFrame) -> pd.DataFrame:
    agg = [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]
    return pd.DataFrame({c: m5[c].resample("4h").agg(a) for c, a in agg}).dropna()


def variance_ratio_curve(close: pd.Series, h_max: int) -> np.ndarray:
    """VR(h) = Var(P_{t+h} - P_t) / (h * Var(P_{t+1} - P_t)) を h=1..h_max で返す (index 0 が h=1)."""
    p = close.to_numpy(dtype=float)
    var1 = float(np.var(np.diff(p), ddof=1))
    out = np.full(h_max, np.nan)
    if var1 <= 0:
        return out
    for h in range(1, h_max + 1):
        if len(p) <= h + 1:
            continue
        diffs = p[h:] - p[:-h]
        out[h - 1] = float(np.var(diffs, ddof=1)) / (h * var1)
    return out


def main() -> int:
    print("=== EXP-FX000018 フェーズゲート2: グリッドパラメータのデータ駆動導出 (Train期間のみ) ===")
    print(f"対象: {PAIRS}  期間: {TRAIN_START} 〜 {TRAIN_END}")
    print("※ 損益・勝率等の成績指標は一切参照しない (spec §2 事前登録)\n")

    h4_by_pair: dict[str, pd.DataFrame] = {}
    atr_by_pair: dict[str, pd.Series] = {}
    for pair in PAIRS:
        m5 = load_m5_train(pair)
        h4 = to_h4(m5)
        h4_by_pair[pair] = h4
        atr_by_pair[pair] = atr_ind(h4["high"], h4["low"], h4["close"], length=ATR_LENGTH)
        print(f"  [{pair}] M5={len(m5):,}bars  H4={len(h4):,}bars")

    # --- D1 / D2: 分散比曲線 ---
    per_pair: dict[str, dict] = {}
    print("\n--- D1/D2: 分散比 VR(h) ---")
    for pair in PAIRS:
        vr = variance_ratio_curve(h4_by_pair[pair]["close"], R_SEARCH_MAX)
        # D1: h in [6,120] で VR 最小
        seg = vr[R_SEARCH_MIN - 1:R_SEARCH_MAX]
        r_argmin = int(np.nanargmin(seg)) + R_SEARCH_MIN
        # D2: h in [1,30] で VR 最小
        seg_s = vr[HSTAR_SEARCH_MIN - 1:HSTAR_SEARCH_MAX]
        h_star = int(np.nanargmin(seg_s)) + HSTAR_SEARCH_MIN
        per_pair[pair] = {
            "vr_at_h1": round(float(vr[0]), 4),
            "vr_min_long_range": round(float(np.nanmin(seg)), 4),
            "r_argmin_raw": r_argmin,
            "vr_min_short_range": round(float(np.nanmin(seg_s)), 4),
            "h_star_raw": h_star,
        }
        print(f"  [{pair}] R候補(h in [{R_SEARCH_MIN},{R_SEARCH_MAX}])={r_argmin} (VR={np.nanmin(seg):.4f})  "
              f"h*(h in [{HSTAR_SEARCH_MIN},{HSTAR_SEARCH_MAX}])={h_star} (VR={np.nanmin(seg_s):.4f})")

    r_median = float(np.median([per_pair[p]["r_argmin_raw"] for p in PAIRS]))
    r_rounded = int(max(R_ROUND_MULTIPLE, round(r_median / R_ROUND_MULTIPLE) * R_ROUND_MULTIPLE))
    r_final = int(min(max(r_rounded, R_CLIP_MIN), R_CLIP_MAX))
    h_star_final = int(round(float(np.median([per_pair[p]["h_star_raw"] for p in PAIRS]))))
    print(f"\n  D1: R = median({[per_pair[p]['r_argmin_raw'] for p in PAIRS]}) = {r_median} "
          f"→ 6の倍数丸め {r_rounded} → 週末制約クリップ[{R_CLIP_MIN},{R_CLIP_MAX}] → **R = {r_final}**")
    print(f"  D2: h* = median({[per_pair[p]['h_star_raw'] for p in PAIRS]}) = {h_star_final}")

    # --- D2: k = median |P_{t+h*} - P_t| / ATR_t (4通貨プール) ---
    pooled_disp: list[float] = []
    for pair in PAIRS:
        close = h4_by_pair[pair]["close"].to_numpy(dtype=float)
        a = atr_by_pair[pair].to_numpy(dtype=float)
        n = len(close)
        idx = np.arange(n - h_star_final)
        disp = np.abs(close[idx + h_star_final] - close[idx]) / a[idx]
        disp = disp[np.isfinite(disp)]
        per_pair[pair]["displacement_median_atr"] = round(float(np.median(disp)), 4)
        per_pair[pair]["n_displacement_samples"] = int(len(disp))
        pooled_disp.extend(disp.tolist())
    k_raw = float(np.median(pooled_disp))
    k_final = round(min(max(k_raw, K_CLIP_MIN), K_CLIP_MAX), 2)
    k_clipped = not (K_CLIP_MIN <= round(k_raw, 2) <= K_CLIP_MAX)
    print(f"  D2: k_raw = median(4通貨プール {len(pooled_disp):,}件) = {k_raw:.4f} → **k = {k_final}**"
          f"{'  [サニティ帯でクリップ]' if k_clipped else ''}")

    # --- D3: N = round(Env_80 / k) - 1 ---
    pooled_env: list[float] = []
    for pair in PAIRS:
        h4 = h4_by_pair[pair]
        high = h4["high"].to_numpy(dtype=float)
        low = h4["low"].to_numpy(dtype=float)
        close = h4["close"].to_numpy(dtype=float)
        a = atr_by_pair[pair].to_numpy(dtype=float)
        # a..a+R の [a+1, a+R] における最高値・最安値 (アンカーバー自身は含めない)
        fwd_max = pd.Series(high).rolling(r_final).max().shift(-r_final).to_numpy()
        fwd_min = pd.Series(low).rolling(r_final).min().shift(-r_final).to_numpy()
        env = np.maximum(fwd_max - close, close - fwd_min) / a
        env = env[np.isfinite(env)]
        per_pair[pair]["env_p80_atr"] = round(float(np.percentile(env, ENV_PERCENTILE)), 4)
        per_pair[pair]["n_env_samples"] = int(len(env))
        pooled_env.extend(env.tolist())
    env_p80 = float(np.percentile(pooled_env, ENV_PERCENTILE))
    n_raw = round(env_p80 / k_final) - 1
    n_final = int(min(max(n_raw, N_CLIP_MIN), N_CLIP_MAX))
    print(f"  D3: Env_p{ENV_PERCENTILE} = {env_p80:.4f} ATR (4通貨プール {len(pooled_env):,}件) "
          f"→ N_raw = round({env_p80:.4f}/{k_final}) - 1 = {n_raw} → クリップ[{N_CLIP_MIN},{N_CLIP_MAX}] → **N = {n_final}**")
    print(f"      → グリッド全幅 (N+1)*k = {(n_final + 1) * k_final:.2f} ATR")

    # --- §8: 通貨除外判定 (損益非依存の構造的基準) ---
    print("\n--- §8: 通貨除外判定 (cost_ratio_pair、損益非依存) ---")
    cost_ratios: dict[str, float] = {}
    for pair in PAIRS:
        pip = pip_size(pair)
        median_atr_pips = float(np.nanmedian(atr_by_pair[pair].to_numpy(dtype=float))) / pip
        step_pips = k_final * median_atr_pips
        cost_pips = 2 * SPREAD_PIPS[pair] + SLIPPAGE_PIPS_STOP
        cost_ratios[pair] = cost_pips / step_pips
        per_pair[pair]["median_atr_h4_pips"] = round(median_atr_pips, 2)
        per_pair[pair]["grid_step_pips"] = round(step_pips, 2)
        per_pair[pair]["cost_pips_worst_case"] = round(cost_pips, 2)
        per_pair[pair]["cost_ratio"] = round(cost_ratios[pair], 5)
    median_cost_ratio = float(np.median(list(cost_ratios.values())))
    threshold = EXCLUSION_COST_RATIO_MULTIPLE * median_cost_ratio
    excluded = [p for p in PAIRS if cost_ratios[p] > threshold]
    for pair in PAIRS:
        mark = "  ← 除外候補" if pair in excluded else ""
        print(f"  [{pair}] 刻み={per_pair[pair]['grid_step_pips']}pips  コスト={per_pair[pair]['cost_pips_worst_case']}pips  "
              f"cost_ratio={cost_ratios[pair]:.5f} (刻みの{1/cost_ratios[pair]:.1f}倍){mark}")
    print(f"  中央値={median_cost_ratio:.5f}  除外閾値(中央値×{EXCLUSION_COST_RATIO_MULTIPLE})={threshold:.5f}")
    print(f"  → 構造的基準による除外通貨: {excluded if excluded else 'なし (4通貨すべてで判定する)'}")

    result = {
        "generated_at": datetime.now().isoformat(),
        "exp_id": "EXP-FX000018",
        "sys_id": "SYS-FX024",
        "phase_gate": "ゲート2 (パラメータ導出)",
        "spec_ref": "research/EXP-FX000018/00-spec.md §2 (D1〜D3)・§8",
        "period": {"train_start": TRAIN_START, "train_end": TRAIN_END},
        "pairs": PAIRS,
        "method": {
            "D1_reanchor_bars": f"argmin_(h in [{R_SEARCH_MIN},{R_SEARCH_MAX}]) VR(h) の4通貨中央値 → "
                                f"{R_ROUND_MULTIPLE}の倍数に丸め → [{R_CLIP_MIN},{R_CLIP_MAX}]にクリップ(週末フラット制約)",
            "D2_grid_step_atr_mult": f"h* = argmin_(h in [{HSTAR_SEARCH_MIN},{HSTAR_SEARCH_MAX}]) VR(h) の4通貨中央値、"
                                     f"k = median(|P_(t+h*) - P_t| / ATR_t) を4通貨プールで算出 → "
                                     f"サニティ帯[{K_CLIP_MIN},{K_CLIP_MAX}]",
            "D3_n_levels": f"Env_a = max(maxHigh[a+1..a+R] - C_a, C_a - minLow[a+1..a+R]) / ATR_a の"
                           f"4通貨プール{ENV_PERCENTILE}パーセンタイル Env_80 に対し "
                           f"N = round(Env_80 / k) - 1 → クリップ[{N_CLIP_MIN},{N_CLIP_MAX}]",
            "pnl_independence": "損益・勝率・PF・シャープ等の成績指標は本導出で一切参照していない",
        },
        "derived": {
            "reanchor_bars_h4": r_final,
            "reanchor_bars_raw_median": r_median,
            "reanchor_bars_before_weekend_clip": r_rounded,
            "h_star_bars": h_star_final,
            "grid_step_atr_mult": k_final,
            "grid_step_atr_mult_raw": round(k_raw, 4),
            "grid_step_clipped": k_clipped,
            "env_p80_atr": round(env_p80, 4),
            "n_levels": n_final,
            "n_levels_raw": int(n_raw),
            "grid_full_span_atr": round((n_final + 1) * k_final, 4),
        },
        "currency_exclusion_check": {
            "criterion": f"cost_ratio_pair = (2*spread + {SLIPPAGE_PIPS_STOP}) / (k * median ATR_H4[pips]) が"
                         f"4通貨中央値の{EXCLUSION_COST_RATIO_MULTIPLE}倍を超える通貨のみ除外候補",
            "cost_ratio_by_pair": {p: round(cost_ratios[p], 5) for p in PAIRS},
            "median_cost_ratio": round(median_cost_ratio, 5),
            "exclusion_threshold": round(threshold, 5),
            "excluded_pairs": excluded,
            "_note": "探索診断でAUD/JPYが弱かったことを理由とする除外は、結果を見てからの選択(cherry-picking)に"
                     "該当するため行わない (spec §8)",
        },
        "per_pair": per_pair,
    }

    out_dir = ROOT / "research" / "EXP-FX000018" / "10-result"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grid_params.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    print(f"\n=== 確定パラメータ: N={n_final}段  k={k_final}×ATR(H4,14)  R={r_final}本(H4) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
