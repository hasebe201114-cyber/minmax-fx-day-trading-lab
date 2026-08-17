"""EXP-FX000003 (SYS-FX009 v2): ダブルトップ/ボトム検出パラメータをデータ駆動で導出.

00-prescreen.md §不採用パターンの妥当性で挙げた3つのHARKingリスクに対応する
パラメータを、結果(バックテストのKPI)を見る前にTrainデータの実測分布から導出する:

    pattern_tolerance_atr        = pooled(|P(t) - P(t+2)| / ATR_at_neckline) の p25
                                    (H4 ZigZag転換点の交互3点、閾値2.0xATRはOBS000006と同一)
    stop_buffer_atr               = pooled((H4のhigh-low)/ATR) の p25
                                    (通常のバーの値幅ノイズでパターン無効化水準の
                                    僅か外側を突かれて即損切りされることを避けるためのバッファ)
    max_bars_since_second_pivot   = round_to_standard(
                                        pooled(pattern_tolerance_atr以内で一致した
                                        交互3点について、2本目の転換点確定からネックライン
                                        を実際に割り込むまでのバー数)のp90
                                     )

LT(D1)のSMA短期/長期は、SYS-FX008 (EXP-FX000002) で既に同一手法
(`scripts/derive_trend_follow_params.py`、ZigZag転換点間隔中央値ベース)で
導出済み・全ペアMA(10,20)に収束した値をそのまま再利用する
(00-prescreen.md「SYS-FX007/008で既にテスト済みの手法を再利用」)。

出力: research/EXP-FX000003/10-result/double_pattern_params.json
      (この値は Train/Validation/Test いずれの結果を見た後も変更しない)

Usage:
    python scripts/derive_double_pattern_params.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from minmax_fx_dt.strategy.indicators import atr as atr_ind
from minmax_fx_dt.strategy.support_resistance import zigzag_pivots_typed

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"

# 事前登録 (結果を見る前に確定)
ZIGZAG_THRESHOLD_ATR = 2.0  # OBS000006追記6・EXP-FX000001/2と同一の「意味あるスイング」閾値
TOLERANCE_PERCENTILE = 25  # 「ほぼ同水準」とみなす許容誤差 = 分布下位25%点 (選択的であるべきため)
BUFFER_PERCENTILE = 25  # ストップバッファ = 1本のバー値幅(ATR比)の下位25%点
STALENESS_PERCENTILE = 90  # ネックライン割れまでの遅延 = 分布上位90%点をカバー
BREAK_SEARCH_CAP_BARS = 60  # ネックライン割れ探索の打ち切り (これを超えたら「割れなかった」扱い)
STALENESS_CANDIDATES = [10, 15, 20, 30, 40]
TRAIL_MFE_HORIZON_H4_BARS = 30  # ネックライン割れ後のフォロースルーを測る窓 (H4本数、約5日)
LT_SMA_SHORT, LT_SMA_LONG = 10, 20  # SYS-FX008 (EXP-FX000002) で既に導出済み、全ペアMA(10,20)に収束


def load_m5(pair: str) -> pd.DataFrame:
    ds1_path = ROOT / "data" / "curated" / "ds-1.json"
    with ds1_path.open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]


def to_h4(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "open": m5["open"].resample("4h").first(),
        "high": m5["high"].resample("4h").max(),
        "low": m5["low"].resample("4h").min(),
        "close": m5["close"].resample("4h").last(),
    }).dropna()


def to_d1(m5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "open": m5["open"].resample("D").first(),
        "high": m5["high"].resample("D").max(),
        "low": m5["low"].resample("D").min(),
        "close": m5["close"].resample("D").last(),
    }).dropna()


def lt_direction_series(d1: pd.DataFrame) -> pd.Series:
    """SMAクロスのみによるLT方向 (SYS-FX008 lt_direction_only()と同一ロジック)."""
    sma_short = d1["close"].rolling(LT_SMA_SHORT, min_periods=LT_SMA_SHORT).mean()
    sma_long = d1["close"].rolling(LT_SMA_LONG, min_periods=LT_SMA_LONG).mean()
    direction = pd.Series("NONE", index=d1.index, dtype="object")
    direction = direction.mask(sma_short > sma_long, "UP")
    direction = direction.mask(sma_short < sma_long, "DOWN")
    return direction


def round_to_standard(value: float, candidates: list[int]) -> int:
    return min(candidates, key=lambda c: abs(c - value))


def alternating_triplets(pivots: list[tuple[int, str]]) -> list[tuple[int, str, int, str, int, str]]:
    """直近ではなく全履歴から、交互(山→谷→山 / 谷→山→谷)の3点組を全て抽出."""
    triplets = []
    for i in range(2, len(pivots)):
        idx1, kind1 = pivots[i - 2]
        idx2, kind2 = pivots[i - 1]
        idx3, kind3 = pivots[i]
        if kind1 == kind3 and kind1 != kind2:
            triplets.append((idx1, kind1, idx2, kind2, idx3, kind3))
    return triplets


def main() -> int:
    print("=== EXP-FX000003 (SYS-FX009 v2): ダブルパターン検出パラメータ導出 ===\n")
    print(f"事前登録: zigzag_threshold_atr={ZIGZAG_THRESHOLD_ATR} (OBS000006既存手法を流用), "
          f"tolerance_percentile=p{TOLERANCE_PERCENTILE}, buffer_percentile=p{BUFFER_PERCENTILE}, "
          f"staleness_percentile=p{STALENESS_PERCENTILE}\n")

    pooled_delta_atr: list[float] = []
    pooled_bar_range_atr: list[float] = []
    pair_stats: dict[str, dict] = {}

    for pair in PAIRS:
        m5 = load_m5(pair)
        h4 = to_h4(m5)
        atr_h4 = atr_ind(h4["high"], h4["low"], h4["close"], length=14)

        pivots = zigzag_pivots_typed(h4["high"], h4["low"], atr_h4, ZIGZAG_THRESHOLD_ATR)
        triplets = alternating_triplets(pivots)

        pair_delta_atr: list[float] = []
        for idx1, kind1, idx2, _kind2, idx3, _kind3 in triplets:
            atr_neckline = atr_h4.iloc[idx2]
            if pd.isna(atr_neckline) or atr_neckline <= 0:
                continue
            p1 = float(h4["high" if kind1 == "HIGH" else "low"].iloc[idx1])
            p2 = float(h4["high" if kind1 == "HIGH" else "low"].iloc[idx3])
            pair_delta_atr.append(abs(p1 - p2) / float(atr_neckline))

        bar_range_atr = ((h4["high"] - h4["low"]) / atr_h4).replace([np.inf, -np.inf], np.nan).dropna()

        pair_stats[pair] = {
            "n_alternating_triplets": len(triplets),
            "delta_atr_median": round(float(np.median(pair_delta_atr)), 3) if pair_delta_atr else None,
        }
        print(f"[{pair}] 交互3点組={len(triplets)}件  "
              f"ΔP/ATR中央値={pair_stats[pair]['delta_atr_median']}")

        pooled_delta_atr.extend(pair_delta_atr)
        pooled_bar_range_atr.extend(bar_range_atr.tolist())

    pattern_tolerance_atr = round(float(np.percentile(pooled_delta_atr, TOLERANCE_PERCENTILE)), 3)
    stop_buffer_atr = round(float(np.percentile(pooled_bar_range_atr, BUFFER_PERCENTILE)), 3)
    print(f"\npooled n_triplets={len(pooled_delta_atr)}  "
          f"pattern_tolerance_atr(p{TOLERANCE_PERCENTILE})={pattern_tolerance_atr}")
    print(f"pooled n_bars={len(pooled_bar_range_atr)}  "
          f"stop_buffer_atr(p{BUFFER_PERCENTILE})={stop_buffer_atr}")

    # staleness + トレーリング幅: 導出したtoleranceで実際に「一致」判定される3点組について、
    # 2本目(ネックラインを挟む対側ピボット)確定後、何本でネックラインを割り込むか。
    # さらに、LT方向がパターン方向と一致するブレイク (=実際にエントリーが成立する
    # 想定シグナル) について、ブレイク後horizon本でのMFE(ATR倍数)からトレーリング幅を導出する。
    pooled_lags: list[int] = []
    pooled_mfe: list[float] = []
    for pair in PAIRS:
        m5 = load_m5(pair)
        h4 = to_h4(m5)
        d1 = to_d1(m5)
        atr_h4 = atr_ind(h4["high"], h4["low"], h4["close"], length=14)
        lt_dir = lt_direction_series(d1)
        pivots = zigzag_pivots_typed(h4["high"], h4["low"], atr_h4, ZIGZAG_THRESHOLD_ATR)
        triplets = alternating_triplets(pivots)

        pair_mfe: list[float] = []
        for idx1, kind1, idx2, _kind2, idx3, _kind3 in triplets:
            atr_neckline = atr_h4.iloc[idx2]
            if pd.isna(atr_neckline) or atr_neckline <= 0:
                continue
            p1 = float(h4["high" if kind1 == "HIGH" else "low"].iloc[idx1])
            p2 = float(h4["high" if kind1 == "HIGH" else "low"].iloc[idx3])
            if abs(p1 - p2) / float(atr_neckline) > pattern_tolerance_atr:
                continue
            neckline = float(h4["low" if kind1 == "HIGH" else "high"].iloc[idx2])
            required_lt = "DOWN" if kind1 == "HIGH" else "UP"
            search_end = min(idx3 + 1 + BREAK_SEARCH_CAP_BARS, len(h4))
            for j in range(idx3 + 1, search_end):
                broke = (
                    (kind1 == "HIGH" and float(h4["low"].iloc[j]) < neckline)
                    or (kind1 == "LOW" and float(h4["high"].iloc[j]) > neckline)
                )
                if not broke:
                    continue
                pooled_lags.append(j - idx3)
                lt_at_break = lt_dir.asof(h4.index[j])
                atr_entry = atr_h4.iloc[j]
                if lt_at_break != required_lt or pd.isna(atr_entry) or atr_entry <= 0:
                    break
                entry_price = float(h4["close"].iloc[j])
                window_high = h4["high"].iloc[j + 1: j + 1 + TRAIL_MFE_HORIZON_H4_BARS]
                window_low = h4["low"].iloc[j + 1: j + 1 + TRAIL_MFE_HORIZON_H4_BARS]
                if len(window_high) == 0:
                    break
                if kind1 == "HIGH":
                    mfe = (entry_price - float(window_low.min())) / float(atr_entry)
                else:
                    mfe = (float(window_high.max()) - entry_price) / float(atr_entry)
                pair_mfe.append(mfe)
                break

        if pair_mfe:
            print(f"[{pair}] LT一致シグナル={len(pair_mfe)}件  MFE中央値={np.median(pair_mfe):.2f}xATR")
        pooled_mfe.extend(pair_mfe)

    staleness_raw = float(np.percentile(pooled_lags, STALENESS_PERCENTILE)) if pooled_lags else 20.0
    max_bars_since_second_pivot = round_to_standard(staleness_raw, STALENESS_CANDIDATES)
    print(f"\nネックライン割れまでの遅延 (許容誤差内で一致した{len(pooled_lags)}件、"
          f"打ち切り{BREAK_SEARCH_CAP_BARS}本): p{STALENESS_PERCENTILE}={staleness_raw:.1f}本 "
          f"→ max_bars_since_second_pivot={max_bars_since_second_pivot}")

    atr_trail_multiplier = round(float(np.median(pooled_mfe)), 2) if pooled_mfe else 2.0
    print(f"\npooled LT一致シグナル={len(pooled_mfe)}件  MFE中央値(pooled、horizon={TRAIL_MFE_HORIZON_H4_BARS}本)="
          f"{atr_trail_multiplier}xATR → atr_trail_multiplier = {atr_trail_multiplier}")

    out_dir = ROOT / "research" / "EXP-FX000003" / "10-result"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "double_pattern_params.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "train_period": [TRAIN_START, TRAIN_END],
            "zigzag_threshold_atr": ZIGZAG_THRESHOLD_ATR,
            "pattern_tolerance_atr": pattern_tolerance_atr,
            "stop_buffer_atr": stop_buffer_atr,
            "max_bars_since_second_pivot": max_bars_since_second_pivot,
            "staleness_raw_p90_bars": round(staleness_raw, 2),
            "atr_trail_multiplier": atr_trail_multiplier,
            "trail_mfe_horizon_h4_bars": TRAIL_MFE_HORIZON_H4_BARS,
            "lt_sma_short": LT_SMA_SHORT,
            "lt_sma_long": LT_SMA_LONG,
            "lt_sma_source": "EXP-FX000002/10-result/trend_follow_params.json (全ペアMA(10,20)に収束、再利用)",
            "pooled_n_triplets": len(pooled_delta_atr),
            "pooled_n_break_lags": len(pooled_lags),
            "pooled_n_lt_matched_signals": len(pooled_mfe),
            "_note": "この値はTrain/Validation/Testいずれの結果を見た後も変更しない",
            "pairs": pair_stats,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
