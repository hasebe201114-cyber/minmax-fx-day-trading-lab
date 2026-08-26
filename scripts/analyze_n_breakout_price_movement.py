"""N_BREAKOUT発生後の値動き(MFE/MAE/終値ベースの推移)を複数の経過時間で実測する.

司令塔からの依頼「N_BREAKOUT発生後の値動きの分析をしたい」への対応。既存の
`analyze_vol_breakout_retrace_window.py`(確定後30分〜3時間の戻り幅分布のみ)・
`analyze_vol_breakout_frequency.py`(48H1バー窓の戻り幅の粗い分布)・
`analyze_n_breakout_h1_dow_trend_alignment.py`(戦略シミュレーション後の勝率)
のいずれとも異なり、**戦略ロジック(エントリー確定・SL/TP)を介さない生の値動き**を、
複数の経過時間ポイントでまとめて可視化する。

事前登録: 本スクリプトは基礎統計の実測・可視化材料出しのみを行う探索的診断であり、
00-spec.md等の正式仕様やKPI判定には影響しない(採否判断とは無関係)。

検出定義(N=3.5・重複排除)は`backtest_vol_breakout_dow_theory.py`の確立済みロジックを
そのまま再利用する(独自の閾値・重複判定は導入しない)。

計測する経過時間ポイント: 30分(準備期間の終端)・1h・3h(既存retrace_window確定窓の終端)・
6h・12h・24h・48h(frequency.pyの既定窓)・72h(atr_trail_multiplier導出窓)。
各時点で、ブレイクバー確定時刻からその時点までのM5データを使い、
  - mfe_atr: ブレイク方向への最大値幅(ATR(H1)単位)
  - mae_atr: ブレイク方向と逆行した最大値幅(ATR(H1)単位)
  - close_atr: その時点に最も近いM5終値のブレイク方向への変化量(ATR(H1)単位、符号あり)
を算出する。ATR(H1)はブレイク確定バー時点の値で正規化する(以降の時点でATRが
再計算されるとイベント間比較がぶれるため固定)。

出力: research/method-notes/n_breakout_price_movement.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from backtest_vol_breakout_dow_theory import select_non_overlapping_breakout_events  # noqa: E402
from derive_vol_breakout_entry_params import N_BREAKOUT, PAIRS, TRAIN_START, TRAIN_END, load_m5, to_h1  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

HORIZONS_HOURS = [0.5, 1, 3, 6, 12, 24, 48, 72]


def candle_shape(break_bar: pd.Series, direction: str) -> dict:
    """ブレイクバー自体のローソク形状(実体/ヒゲの比率)を算出する."""
    o, h, lo, c = float(break_bar["open"]), float(break_bar["high"]), float(break_bar["low"]), float(break_bar["close"])
    rng = h - lo
    if rng <= 0:
        return {"body_frac": None, "upper_wick_frac": None, "lower_wick_frac": None}
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - lo
    return {
        "body_frac": round(body / rng, 3),
        "upper_wick_frac": round(upper_wick / rng, 3),
        "lower_wick_frac": round(lower_wick / rng, 3),
        "direction_matches_body": (c > o) == (direction == "UP"),
    }


def measure_event(h1: pd.DataFrame, m5: pd.DataFrame, atr_h1: pd.Series, break_idx: int,
                   direction: str) -> dict | None:
    """`break_idx`のH1バーをN_BREAKOUT検知バーとして、確定後の値動きを計測する.

    重要な注意(本分析で新規発見): `h1.index[break_idx]`はpandas `resample('1h')`の
    既定ラベル付け(label='left')により**バーの開始時刻**であり、終了(確定)時刻ではない
    (例: ラベル10:00のH1バーはM5データ[10:00,11:00)を集約し、実際にバーが確定する
    のは11:00時点)。EXP-FX000005系列の既存スクリプト(`derive_vol_breakout_entry_params.py`
    ・`backtest_vol_breakout_dow_theory.py`等)はこの`h1.index[break_idx]`をそのまま
    「ブレイクバーが確定した時刻」として扱い、そこから+30分/+3時間等の探索窓を計算して
    いるため、既存の全窓定義は実際にはバー確定(=次のH1バー開始)より最大1時間early-shift
    している可能性がある(窓の前半がブレイクバー自身の未確定な形成過程のデータを含む)。
    本スクリプトはこの先読みを避けるため、`h1.index[break_idx+1]`(＝次のH1バー開始
    ＝実質的なバー確定タイミング)を基準点として使う。既存スクリプトとの直接比較を
    したい場合は`h1.index[break_idx]`基準の結果と差分を取ること。
    """
    break_bar = h1.iloc[break_idx]
    break_time = h1.index[break_idx]
    confirm_time = h1.index[break_idx + 1] if break_idx + 1 < len(h1.index) else break_time + pd.Timedelta(hours=1)
    atr_at_break = float(atr_h1.iloc[break_idx])
    if not np.isfinite(atr_at_break) or atr_at_break <= 0:
        return None
    ref_close = float(break_bar["close"])
    break_range = float(break_bar["high"] - break_bar["low"])
    sign = 1.0 if direction == "UP" else -1.0

    m5_after = m5[m5.index >= confirm_time]
    if len(m5_after) == 0:
        return None

    by_horizon: dict[str, dict] = {}
    for hours in HORIZONS_HOURS:
        window_end = confirm_time + pd.Timedelta(hours=hours)
        win = m5_after[m5_after.index <= window_end]
        if len(win) == 0:
            continue
        if direction == "UP":
            mfe = float(win["high"].max()) - ref_close
            mae = ref_close - float(win["low"].min())
        else:
            mfe = ref_close - float(win["low"].min())
            mae = float(win["high"].max()) - ref_close
        mfe = max(0.0, mfe)
        mae = max(0.0, mae)
        close_move = sign * (float(win["close"].iloc[-1]) - ref_close)
        by_horizon[str(hours)] = {
            "mfe_atr": mfe / atr_at_break,
            "mae_atr": mae / atr_at_break,
            "close_atr": close_move / atr_at_break,
            # 「戻し」がブレイクバー自体のレンジ全体を飲み込んだか(=発生バー始値水準まで押し戻された相当)
            "full_reversal": mae >= break_range if break_range > 0 else None,
            # 終値ベースで見て、ブレイク方向にブレイクバー1本分以上さらに進んだか(強い継続)
            "strong_continuation": close_move >= break_range if break_range > 0 else None,
        }
    if not by_horizon:
        return None
    return {
        "break_time": break_time.isoformat(),
        "confirm_time": confirm_time.isoformat(),
        "direction": direction,
        "range_atr": break_range / atr_at_break,
        "candle_shape": candle_shape(break_bar, direction),
        "by_horizon": by_horizon,
    }


def summarize_horizon(events: list[dict], hours: str, field: str) -> dict | None:
    vals = [e["by_horizon"][hours][field] for e in events if hours in e["by_horizon"]]
    if not vals:
        return None
    arr = np.array(vals)
    return {
        "n": len(arr),
        "median": round(float(np.median(arr)), 3),
        "p25": round(float(np.percentile(arr, 25)), 3),
        "p75": round(float(np.percentile(arr, 75)), 3),
        "mean": round(float(arr.mean()), 3),
    }


def rate_horizon(events: list[dict], hours: str, field: str) -> dict | None:
    vals = [e["by_horizon"][hours][field] for e in events if hours in e["by_horizon"]
            and e["by_horizon"][hours][field] is not None]
    if not vals:
        return None
    return {"n": len(vals), "rate": round(float(np.mean(vals)), 3)}


def summarize_candle_shape(events: list[dict]) -> dict:
    shapes = [e["candle_shape"] for e in events if e["candle_shape"]["body_frac"] is not None]
    range_atr = np.array([e["range_atr"] for e in events])
    body = np.array([s["body_frac"] for s in shapes])
    upper = np.array([s["upper_wick_frac"] for s in shapes])
    lower = np.array([s["lower_wick_frac"] for s in shapes])
    dir_match_rate = float(np.mean([s["direction_matches_body"] for s in shapes])) if shapes else None

    def q(arr):
        return {"median": round(float(np.median(arr)), 3), "p25": round(float(np.percentile(arr, 25)), 3),
                "p75": round(float(np.percentile(arr, 75)), 3)} if len(arr) else None

    return {
        "n": len(events),
        "range_atr": q(range_atr),
        "body_frac": q(body),
        "upper_wick_frac": q(upper),
        "lower_wick_frac": q(lower),
        "direction_matches_body_rate": round(dir_match_rate, 3) if dir_match_rate is not None else None,
    }


def main() -> int:
    print(f"=== N_BREAKOUT(N={N_BREAKOUT})発生後の値動き(MFE/MAE/終値) 基礎統計 (Train期間) ===\n")

    all_events: list[dict] = []
    per_pair_events: dict[str, list[dict]] = {}

    for pair in PAIRS:
        m5 = load_m5(pair)
        h1 = to_h1(m5)
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
        idxs = np.where(ratio.values >= N_BREAKOUT)[0]
        positions = [h1.index.get_loc(ratio.index[i]) for i in idxs]
        directions = ["UP" if h1.iloc[pos]["close"] > h1.iloc[pos]["open"] else "DOWN" for pos in positions]
        dedup_positions = select_non_overlapping_breakout_events(h1.index, positions, directions)
        dedup_directions = {pos: d for pos, d in zip(positions, directions)}

        pair_events = []
        for pos in dedup_positions:
            ev = measure_event(h1, m5, atr_h1, pos, dedup_directions[pos])
            if ev is not None:
                pair_events.append(ev)
        per_pair_events[pair] = pair_events
        all_events.extend(pair_events)
        n_up = sum(1 for e in pair_events if e["direction"] == "UP")
        n_down = len(pair_events) - n_up
        print(f"[{pair}] イベント数={len(pair_events)} (UP={n_up} / DOWN={n_down})")

    print(f"\n全体プール(5通貨): n={len(all_events)}イベント\n")

    candle_shape_summary = summarize_candle_shape(all_events)
    print("--- 発生時のローソク形状・値幅(ブレイクバー自身) ---")
    print(f"レンジ/ATR比: 中央値={candle_shape_summary['range_atr']['median']:.2f} "
          f"(p25={candle_shape_summary['range_atr']['p25']:.2f} p75={candle_shape_summary['range_atr']['p75']:.2f})")
    print(f"実体比率(body/range): 中央値={candle_shape_summary['body_frac']['median']:.2f}  "
          f"上ヒゲ比率: 中央値={candle_shape_summary['upper_wick_frac']['median']:.2f}  "
          f"下ヒゲ比率: 中央値={candle_shape_summary['lower_wick_frac']['median']:.2f}")
    print(f"陽線/陰線の向きとブレイク方向が一致した割合: {candle_shape_summary['direction_matches_body_rate']:.1%}\n")

    header = (f"{'経過時間':<8}{'MFE中央値':>10}{'MFE p75':>10}{'MAE中央値':>10}{'MAE p75':>10}"
              f"{'終値変化中央値':>14}{'完全戻し率':>10}{'強い継続率':>10}")
    print(header)

    pooled_by_horizon: dict[str, dict] = {}
    for hours in HORIZONS_HOURS:
        key = str(hours)
        mfe = summarize_horizon(all_events, key, "mfe_atr")
        mae = summarize_horizon(all_events, key, "mae_atr")
        close = summarize_horizon(all_events, key, "close_atr")
        full_reversal = rate_horizon(all_events, key, "full_reversal")
        strong_cont = rate_horizon(all_events, key, "strong_continuation")
        pooled_by_horizon[key] = {"mfe_atr": mfe, "mae_atr": mae, "close_atr": close,
                                   "full_reversal_rate": full_reversal, "strong_continuation_rate": strong_cont}
        if mfe and mae and close:
            fr = f"{full_reversal['rate']:.1%}" if full_reversal else "n/a"
            sc = f"{strong_cont['rate']:.1%}" if strong_cont else "n/a"
            print(f"{key+'h':<8}{mfe['median']:>10.3f}{mfe['p75']:>10.3f}{mae['median']:>10.3f}"
                  f"{mae['p75']:>10.3f}{close['median']:>14.3f}{fr:>10}{sc:>10}")

    per_pair_summary = {}
    for pair, events in per_pair_events.items():
        per_pair_summary[pair] = {
            "candle_shape": summarize_candle_shape(events),
            "by_horizon": {
                str(hours): {
                    "mfe_atr": summarize_horizon(events, str(hours), "mfe_atr"),
                    "mae_atr": summarize_horizon(events, str(hours), "mae_atr"),
                    "close_atr": summarize_horizon(events, str(hours), "close_atr"),
                    "full_reversal_rate": rate_horizon(events, str(hours), "full_reversal"),
                    "strong_continuation_rate": rate_horizon(events, str(hours), "strong_continuation"),
                }
                for hours in HORIZONS_HOURS
            },
        }

    result = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "purpose": "N_BREAKOUT発生後の値動き(MFE/MAE/終値変化、ATR(H1)単位)を複数経過時間で実測する探索的診断",
        "caveat": "戦略ロジック(エントリー確定・SL/TP・コスト)を介さない生の値動き。"
                   "正式プロトコル外であり00-spec.md等・KPI判定には影響しない",
        "method": {
            "n_breakout": N_BREAKOUT,
            "dedup": "select_non_overlapping_breakout_events()(backtest_vol_breakout_dow_theory.py)を流用",
            "atr_normalization": "ブレイク確定バー時点のATR(H1,14,Wilder)で固定(以降のATR再計算はしない)",
            "horizons_hours": HORIZONS_HOURS,
            "mfe_mae_definition": "ブレイクバー終値を基準に、ブレイク方向への最大値幅=MFE、"
                                   "逆方向への最大値幅=MAE(いずれもM5高値/安値ベース)",
            "confirm_time_fix": (
                "新規発見: h1.index[break_idx]はpandas resample('1h')既定ラベル(label='left')"
                "によりH1バーの開始時刻であり、確定(終了)時刻ではない。EXP-FX000005系列の既存"
                "スクリプトはbreak_time(=バー開始時刻)をそのまま「確定時刻」として+30分/+3時間等の"
                "窓を計算しており、窓の前半が当該ブレイクバー自身の未確定な形成過程を含みうる"
                "(最大1時間の先読み相当)。本スクリプトはh1.index[break_idx+1](次のH1バー開始"
                "=実質的な確定タイミング)を基準に修正して計測している。既存メソドロジーへの"
                "影響は本分析の対象外(採否判断への影響は未評価)、別途確認を推奨"
            ),
        },
        "train_period": [TRAIN_START, TRAIN_END],
        "pairs": PAIRS,
        "n_events_pooled": len(all_events),
        "candle_shape_pooled": candle_shape_summary,
        "pooled_by_horizon": pooled_by_horizon,
        "per_pair": per_pair_summary,
        "detection_window_note": (
            "N_BREAKOUTはH1バーの確定(=1時間足の終値確定)を待って判定するため、検知そのものに"
            "本質的に最大1時間の内在的な遅延がある(バー確定前の判定は先読みになるため不可)。"
            "その後「発生直後の戻し」を判定する探索窓は、EXP-FX000005で司令塔確認済みの"
            "確定後30分(準備期間)〜3時間(retrace_ratio導出窓)が実務的な目安。"
            "本スクリプトのhorizons_hours別の結果を見ると、MAE(戻し)は概ね3h時点で大部分が"
            "出そろい6h以降の追加拡大は緩やかなのに対し、MFE/close_atr(継続方向への値動き)は"
            "24h〜72hにかけてなお緩やかに拡大し続ける傾向がある場合、"
            "「戻り確認は短い窓(30分〜3時間)、トレンド継続の判定は長い窓(24〜72時間)」という"
            "非対称な時間幅設計が妥当である可能性を示唆する(具体的な数値はpooled_by_horizonを参照)。"
        ),
    }
    out_path = ROOT / "research" / "method-notes" / "n_breakout_price_movement.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
