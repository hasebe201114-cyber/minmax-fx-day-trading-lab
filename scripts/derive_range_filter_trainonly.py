"""EXP-FX000018 amendment-02 §2: レンジ判定フィルター(ER)の閾値をTrainから導出.

改善ループ第2試行で使う Kaufman の Efficiency Ratio の閾値を、**損益・勝率等の
成績指標を一切参照せずに** 決める(フェーズゲート2・amendment-01 W2 と同じ方針)。

  ER(a, W) = |C_a - C_(a-W)| / Σ|C_j - C_(j-1)|   (H4終値、W = R = 24、先読みなし)
  ER_MAX   = Train期間・4通貨プールの ER(24) 分布の中央値

新規の自由パラメータはゼロ: W は凍結済みの再アンカー周期 R をそのまま使い、
閾値は「トレンド寄りの半分の局面を除外する」という設計選択(中央値)で決まる。

併せて amendment-02 §4.1 の追加報告項目のうち「除外された局面と採用された局面で
その後 R バーの実現レンジ幅・実現ボラティリティがどう違ったか」を、**損益を使わずに
価格構造だけで**測っておく(フィルターが狙いどおり「伸びる局面」を除外できているかの
直接確認。Train損益を見る前に実行する)。

出力: research/EXP-FX000018/10-result/range_filter.json
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

from grid_portfolio_engine import load_m5, to_h4  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
ATR_LENGTH = 14


def efficiency_ratio(close: np.ndarray, window: int) -> np.ndarray:
    """Kaufman の Efficiency Ratio。index i は「i 本目までの確定情報のみ」を使う(先読みなし)."""
    n = len(close)
    out = np.full(n, np.nan)
    step = np.abs(np.diff(close, prepend=close[0]))
    cum = np.cumsum(step)
    for i in range(window, n):
        denom = cum[i] - cum[i - window]
        if denom > 0:
            out[i] = abs(close[i] - close[i - window]) / denom
    return out


def main() -> int:
    params = json.loads(
        (ROOT / "research" / "EXP-FX000018" / "10-result" / "grid_params.json").read_text(encoding="utf-8")
    )["derived"]
    window = params["reanchor_bars_h4"]  # W = R (新規パラメータではない)

    print("=== EXP-FX000018 amendment-02: レンジ判定フィルター(ER)の閾値導出 (Train期間のみ) ===")
    print(f"対象: {PAIRS}  期間: {TRAIN_START} 〜 {TRAIN_END}  W = R = {window}本(H4)")
    print("※ 損益・勝率等の成績指標は一切参照しない\n")

    pooled_er: list[float] = []
    per_pair: dict[str, dict] = {}
    cache: dict[str, dict] = {}
    for pair in PAIRS:
        h4 = to_h4(load_m5(pair, TRAIN_START, TRAIN_END))
        close = h4["close"].to_numpy(dtype=float)
        er = efficiency_ratio(close, window)
        atr = atr_ind(h4["high"], h4["low"], h4["close"], length=ATR_LENGTH).to_numpy(dtype=float)
        cache[pair] = {"h4": h4, "er": er, "atr": atr}
        valid = er[np.isfinite(er)]
        pooled_er.extend(valid.tolist())
        per_pair[pair] = {
            "n_bars": int(len(valid)),
            "er_p25": round(float(np.percentile(valid, 25)), 4),
            "er_median": round(float(np.median(valid)), 4),
            "er_p75": round(float(np.percentile(valid, 75)), 4),
        }
        print(f"  [{pair}] 有効バー={len(valid):,}  ER p25={per_pair[pair]['er_p25']}  "
              f"中央値={per_pair[pair]['er_median']}  p75={per_pair[pair]['er_p75']}")

    er_max = round(float(np.median(pooled_er)), 4)
    print(f"\n  → **ER_MAX = 4通貨プール({len(pooled_er):,}件)の中央値 = {er_max}**")
    print(f"     ER > {er_max} のアンカーではグリッドを張らない(トレンド寄りの半分を除外)\n")

    # amendment-02 §4.1: 除外/採用局面のその後 R バーの実現レンジ幅・実現ボラ (損益非依存)
    print("--- 除外局面 vs 採用局面 の「その後 R バー」の価格構造 (損益は一切使わない) ---")
    regime_check: dict[str, dict] = {}
    for pair in PAIRS:
        h4, er, atr = cache[pair]["h4"], cache[pair]["er"], cache[pair]["atr"]
        high = h4["high"].to_numpy(dtype=float)
        low = h4["low"].to_numpy(dtype=float)
        close = h4["close"].to_numpy(dtype=float)
        fwd_max = pd.Series(high).rolling(window).max().shift(-window).to_numpy()
        fwd_min = pd.Series(low).rolling(window).min().shift(-window).to_numpy()
        fwd_range = (fwd_max - fwd_min) / atr                      # 実現レンジ幅 (ATR単位)
        fwd_net = np.abs(np.roll(close, -window) - close) / atr    # 正味変位 (ATR単位)
        fwd_net[-window:] = np.nan
        ok = np.isfinite(er) & np.isfinite(fwd_range) & np.isfinite(fwd_net)
        keep = ok & (er <= er_max)   # フィルター通過 (レンジ判定)
        drop = ok & (er > er_max)    # フィルターで除外 (トレンド判定)
        regime_check[pair] = {
            "n_keep": int(keep.sum()), "n_drop": int(drop.sum()),
            "keep_fwd_range_median_atr": round(float(np.median(fwd_range[keep])), 3),
            "drop_fwd_range_median_atr": round(float(np.median(fwd_range[drop])), 3),
            "keep_fwd_net_displacement_median_atr": round(float(np.median(fwd_net[keep])), 3),
            "drop_fwd_net_displacement_median_atr": round(float(np.median(fwd_net[drop])), 3),
        }
        d = regime_check[pair]
        print(f"  [{pair}] 採用局面(n={d['n_keep']}): その後レンジ幅中央値={d['keep_fwd_range_median_atr']}ATR "
              f"正味変位={d['keep_fwd_net_displacement_median_atr']}ATR")
        print(f"           除外局面(n={d['n_drop']}): その後レンジ幅中央値={d['drop_fwd_range_median_atr']}ATR "
              f"正味変位={d['drop_fwd_net_displacement_median_atr']}ATR")

    result = {
        "generated_at": datetime.now().isoformat(),
        "exp_id": "EXP-FX000018", "sys_id": "SYS-FX024",
        "spec_ref": "research/EXP-FX000018/00-spec-amendment-02.md §2",
        "period": {"train_start": TRAIN_START, "train_end": TRAIN_END},
        "pairs": PAIRS,
        "method": {
            "indicator": "Kaufman Efficiency Ratio (H4終値、無方向・先読みなし)",
            "lookback_window": window,
            "lookback_window_source": "凍結済みの再アンカー周期 R をそのまま使用(新規パラメータではない)",
            "threshold_rule": "Train期間・4通貨プールの ER 分布の中央値(トレンド寄りの半分を除外)",
            "pnl_independence": "損益・勝率等の成績指標は本導出で一切参照していない",
        },
        "derived": {"er_max": er_max, "lookback_window": window},
        "per_pair": per_pair,
        "regime_forward_structure_check": regime_check,
    }
    out = ROOT / "research" / "EXP-FX000018" / "10-result" / "range_filter.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[出力]: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
