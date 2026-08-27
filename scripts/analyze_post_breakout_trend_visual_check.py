"""司令塔の視覚的観察「N_BREAKOUT後の戻りでトレンドが形成されているように見える」の
定量的な当否を、Train期間のみで探索的に確認する診断スクリプト.

## 位置づけ（HARKing防止のため明記）

- 本スクリプトは**正式な検証プロトコル外の探索的診断**であり、`00-spec.md` 等の
  事前登録文書は一切編集しない（前例: `n_breakout_h1_dow_trend_alignment.json`、
  `h1_continuation_resume_reverify.json`）。
- 使用期間は **Train (2023-11-01〜2025-03-31) のみ**。Validation/Test は
  将来の正式検証のために温存する。
- 新規パラメータの導出・選択は行わない。N=3.5・ATR(14) は既存の確定値を流用する。

## 検証したい命題

司令塔がTradingViewのN_BREAKOUTインジケータでチャートを目視した結果得た印象:
「ブレーク後の反発（戻り）でいずれもトレンドが形成されているように見える」

これを反証可能な形に分解する:

- Q1: ブレイク方向へのその後の値動きは、**普通のバーの方向**に比べて有意に継続的か？
      （＝「大きいバーだから続く」のか、「どのバーでも方向は多少続く」だけなのか）
- Q2: 「トレンド形成」を MFE/MAE の非対称性（±1ATRのバリア到達レース）で定義したとき、
      ブレイク後は普通のバー後より有利か？
- Q3: 司令塔の言う「戻ってから」に限定した場合（確定後30分〜3時間で50%以上戻した群）、
      その後のトレンド形成率は、戻さなかった群より高いか？

## 対照群（ここが目視では絶対に見えない部分）

- CTRL_ALL: 同期間の**全H1バー**を、そのバー自身の陽線/陰線方向でアンカーにする
- CTRL_NORMAL: レンジ/ATR が 1.0〜1.5 の「ごく普通のバー」だけをアンカーにする

出力: research/method-notes/post_breakout_trend_visual_check.json
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

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"
N_BREAKOUT = 3.5
ATR_LEN = 14
HORIZONS_H1 = [3, 6, 12, 24, 72]
BARRIER_ATR = 1.0
BARRIER_MAX_BARS = 72
# 「戻り」判定窓は vol_breakout_retrace_window.json で確定済みの定義を流用
RETRACE_WINDOW_START_MIN = 30
RETRACE_WINDOW_END_HOURS = 3
RETRACE_SPLIT = 0.5
RNG_SEED = 20260827


def load_m5(pair: str, root: Path, subdir: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / "data" / "raw" / subdir / f"ohlcv_{pair}_5min_*.csv")))
    if not files:
        raise FileNotFoundError(f"{pair} の M5 CSV が見つからない: {subdir}")
    frames = [pd.read_csv(f, parse_dates=["timestamp"]) for f in files]
    df = pd.concat(frames).drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    return df


def resample(m5: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]
    return pd.DataFrame({c: m5[c].resample(rule).agg(a) for c, a in agg}).dropna()


def forward_metrics(h1: pd.DataFrame, atr_h1: pd.Series) -> dict[int, pd.DataFrame]:
    """各H1バーをアンカーとしたときの、先のHバー分の終値変化・MFE・MAEを事前計算する.

    アンカーバー自身は含めない（i+1..i+H）ため先読みは発生しない。
    """
    out = {}
    for h in HORIZONS_H1:
        fwd_close = h1["close"].shift(-h)
        fwd_high = h1["high"].rolling(h).max().shift(-h)
        fwd_low = h1["low"].rolling(h).min().shift(-h)
        out[h] = pd.DataFrame({
            "fwd_close": fwd_close,
            "fwd_high": fwd_high,
            "fwd_low": fwd_low,
        })
    return out


def collect(h1: pd.DataFrame, atr_h1: pd.Series, fwd: dict[int, pd.DataFrame],
            positions: np.ndarray, directions: np.ndarray) -> dict[int, dict[str, list]]:
    """アンカー位置群について、ATR正規化した順行リターン・MFE・MAEを集計する."""
    close = h1["close"].to_numpy()
    atr_v = atr_h1.to_numpy()
    res: dict[int, dict[str, list]] = {h: {"ret": [], "mfe": [], "mae": []} for h in HORIZONS_H1}
    for h in HORIZONS_H1:
        fc = fwd[h]["fwd_close"].to_numpy()
        fh = fwd[h]["fwd_high"].to_numpy()
        fl = fwd[h]["fwd_low"].to_numpy()
        for pos, d in zip(positions, directions):
            a = atr_v[pos]
            if not np.isfinite(a) or a <= 0 or not np.isfinite(fc[pos]):
                continue
            c0 = close[pos]
            res[h]["ret"].append(float(d * (fc[pos] - c0) / a))
            if d > 0:
                res[h]["mfe"].append(float((fh[pos] - c0) / a))
                res[h]["mae"].append(float((fl[pos] - c0) / a))
            else:
                res[h]["mfe"].append(float((c0 - fl[pos]) / a))
                res[h]["mae"].append(float((c0 - fh[pos]) / a))
    return res


def barrier_race(h1: pd.DataFrame, atr_h1: pd.Series, pos: int, direction: int) -> str | None:
    """アンカー終値から ±BARRIER_ATR のどちらに先に触れたかを判定する（パス依存）."""
    a = atr_h1.iloc[pos]
    if not np.isfinite(a) or a <= 0:
        return None
    c0 = float(h1["close"].iloc[pos])
    up_b = c0 + direction * BARRIER_ATR * a
    dn_b = c0 - direction * BARRIER_ATR * a
    hi = h1["high"].to_numpy()
    lo = h1["low"].to_numpy()
    end = min(pos + BARRIER_MAX_BARS, len(h1) - 1)
    if end <= pos:
        return None
    for j in range(pos + 1, end + 1):
        if direction > 0:
            hit_win = hi[j] >= up_b
            hit_lose = lo[j] <= dn_b
        else:
            hit_win = lo[j] <= up_b
            hit_lose = hi[j] >= dn_b
        if hit_win and hit_lose:
            return "AMBIGUOUS"  # 同一バー内で両側到達（H1粒度では順序不明）
        if hit_win:
            return "WIN"
        if hit_lose:
            return "LOSE"
    return "NEITHER"


def retrace_frac(h1: pd.DataFrame, m15: pd.DataFrame, pos: int, direction: int,
                 anchor: str) -> tuple[float, pd.Timestamp] | None:
    """戻り幅（ブレイクバーのレンジに対する比率）と、判定窓の終端時刻を返す.

    anchor="confirmed": ブレイクバーが**確定した時刻**（＝バーオープン+1h、
        `00-spec.md` §「探索窓M」の文言どおり）を起点にする。実運用可能。
    anchor="bar_open": 現行実装（`backtest_vol_breakout_dow_theory.py`
        `simulate_dow_theory_trend()` の `break_time = h1.index[break_idx]`）と
        同じくバー**オープン時刻**を起点にする。窓の前半30分がブレイクバー自身の
        後半と重なるため、シグナル成立前のデータを見ていることになる（比較用）。
    """
    bar = h1.iloc[pos]
    rng = float(bar["high"] - bar["low"])
    if rng <= 0:
        return None
    origin = h1.index[pos] + (pd.Timedelta(hours=1) if anchor == "confirmed" else pd.Timedelta(0))
    win_start = origin + pd.Timedelta(minutes=RETRACE_WINDOW_START_MIN)
    win_end = origin + pd.Timedelta(hours=RETRACE_WINDOW_END_HOURS)
    win = m15[(m15.index > win_start) & (m15.index <= win_end)]
    if len(win) == 0:
        return None
    if direction > 0:
        worst = float(win["low"].min())
        frac = max(0.0, (float(bar["high"]) - worst) / rng)
    else:
        worst = float(win["high"].max())
        frac = max(0.0, (worst - float(bar["low"])) / rng)
    return frac, win_end


def reanchored_race(h1: pd.DataFrame, atr_h1: pd.Series, m15: pd.DataFrame,
                    pos: int, direction: int, win_end: pd.Timestamp) -> tuple[str, float] | None:
    """戻り判定窓が終わった**あと**の値動きだけでバリアレースを判定する.

    戻り幅の測定窓とレースの窓が重ならないため、「戻った群は負ける」という
    自己参照（測定窓の中でバリアに触れているのを数えてしまう）が構造的に起きない。
    """
    a = atr_h1.iloc[pos]
    if not np.isfinite(a) or a <= 0:
        return None
    m15_at = m15[m15.index <= win_end]
    if len(m15_at) == 0:
        return None
    entry = float(m15_at["close"].iloc[-1])
    start = int(h1.index.searchsorted(win_end, side="left"))
    end = min(start + BARRIER_MAX_BARS, len(h1) - 1)
    if start >= len(h1) or end <= start:
        return None
    win_b = entry + direction * BARRIER_ATR * a
    lose_b = entry - direction * BARRIER_ATR * a
    hi = h1["high"].to_numpy()
    lo = h1["low"].to_numpy()
    outcome = "NEITHER"
    for j in range(start, end + 1):
        if direction > 0:
            hit_w, hit_l = hi[j] >= win_b, lo[j] <= lose_b
        else:
            hit_w, hit_l = lo[j] <= win_b, hi[j] >= lose_b
        if hit_w and hit_l:
            outcome = "AMBIGUOUS"
            break
        if hit_w:
            outcome = "WIN"
            break
        if hit_l:
            outcome = "LOSE"
            break
    fwd = float(direction * (float(h1["close"].iloc[end]) - entry) / a)
    return outcome, fwd


def summarize(values: list[float], rng: np.random.Generator, n_boot: int = 2000) -> dict:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    boot = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        "positive_rate": round(float((arr > 0).mean()), 4),
    }


def main() -> int:
    rng = np.random.default_rng(RNG_SEED)
    print("=== N_BREAKOUT後の「トレンド形成」目視印象の定量チェック (Train期間のみ・探索的) ===\n")

    groups = {"BREAKOUT": [], "CTRL_ALL": [], "CTRL_NORMAL": []}
    pooled: dict[str, dict[int, dict[str, list]]] = {
        g: {h: {"ret": [], "mfe": [], "mae": []} for h in HORIZONS_H1} for g in groups
    }
    races: dict[str, list[str]] = {g: [] for g in groups}
    ANCHORS = ("confirmed", "bar_open")
    SPLITS = ("RETRACED", "NOT_RETRACED")
    leaky_split = {a: {k: [] for k in SPLITS} for a in ANCHORS}
    clean_split = {a: {k: [] for k in SPLITS} for a in ANCHORS}
    clean_fwd: dict[str, dict[str, list[float]]] = {a: {k: [] for k in SPLITS} for a in ANCHORS}
    retrace_fracs: dict[str, list[float]] = {a: [] for a in ANCHORS}
    per_pair_events = {}
    atr_cost_context = {}
    SPREAD_PIPS = {"USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6, "EUR_USD": 0.3}
    SLIPPAGE_PIPS = 0.5

    for pair in PAIRS:
        m5 = load_m5(pair, ROOT, "ds-1")
        m5 = m5[(m5.index >= TRAIN_START) & (m5.index <= TRAIN_END)]
        h1 = resample(m5, "1h")
        m15 = resample(m5, "15min")
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=ATR_LEN)
        ratio = (h1["high"] - h1["low"]) / atr_h1
        fwd = forward_metrics(h1, atr_h1)

        body_dir = np.sign((h1["close"] - h1["open"]).to_numpy())
        valid = np.isfinite(ratio.to_numpy()) & (body_dir != 0)

        is_break = valid & (ratio.to_numpy() >= N_BREAKOUT)
        is_normal = valid & (ratio.to_numpy() >= 1.0) & (ratio.to_numpy() < 1.5)

        sets = {
            "BREAKOUT": np.where(is_break)[0],
            "CTRL_ALL": np.where(valid)[0],
            "CTRL_NORMAL": np.where(is_normal)[0],
        }
        per_pair_events[pair] = {
            "n_h1_bars": int(len(h1)),
            "n_breakout": int(is_break.sum()),
            "n_ctrl_all": int(valid.sum()),
            "n_ctrl_normal": int(is_normal.sum()),
        }
        pip = 0.01 if "JPY" in pair else 0.0001
        atr_pips = float(np.nanmedian(atr_h1.to_numpy())) / pip
        rt_cost_pips = 2 * SPREAD_PIPS[pair] + 2 * SLIPPAGE_PIPS
        atr_cost_context[pair] = {
            "median_atr_h1_pips": round(atr_pips, 2),
            "round_trip_cost_pips": round(rt_cost_pips, 2),
            "barrier_1atr_over_cost": round(atr_pips / rt_cost_pips, 2),
        }
        print(f"{pair}: H1バー {len(h1)}本 / N_BREAKOUT {int(is_break.sum())}件 "
              f"({is_break.sum() / max(1, valid.sum()):.2%}) / ATR(H1)中央値 {atr_pips:.1f}pips "
              f"= 往復コストの {atr_pips / rt_cost_pips:.1f}倍")

        for gname, positions in sets.items():
            dirs = body_dir[positions].astype(int)
            got = collect(h1, atr_h1, fwd, positions, dirs)
            for h in HORIZONS_H1:
                for k in ("ret", "mfe", "mae"):
                    pooled[gname][h][k].extend(got[h][k])
            # バリアレースは重いので対照群はサンプリングする
            if gname == "BREAKOUT":
                sample_pos, sample_dir = positions, dirs
            else:
                take = min(len(positions), 1500)
                idx = rng.choice(len(positions), size=take, replace=False)
                sample_pos, sample_dir = positions[idx], dirs[idx]
            for pos, d in zip(sample_pos, sample_dir):
                r = barrier_race(h1, atr_h1, int(pos), int(d))
                if r:
                    races[gname].append(r)

        # Q3: 「戻ってから」に限定した層別
        for pos in sets["BREAKOUT"]:
            d = int(body_dir[pos])
            for anchor in ("confirmed", "bar_open"):
                got_rf = retrace_frac(h1, m15, int(pos), d, anchor)
                if got_rf is None:
                    continue
                rf, win_end = got_rf
                key = "RETRACED" if rf >= RETRACE_SPLIT else "NOT_RETRACED"
                # (a) リーク版: ブレイクバー終値を起点にレースを判定するため、
                #     戻り幅の測定窓とレース窓が重なる
                leaky = barrier_race(h1, atr_h1, int(pos), d)
                if leaky:
                    leaky_split[anchor][key].append(leaky)
                # (b) リークなし版: 戻り判定窓が終わったあとの値動きだけで判定
                clean = reanchored_race(h1, atr_h1, m15, int(pos), d, win_end)
                if clean:
                    clean_split[anchor][key].append(clean[0])
                    clean_fwd[anchor][key].append(clean[1])
                retrace_fracs[anchor].append(rf)

    def counter(rs: list[str]) -> list[str]:
        """±1ATRの対称バリアなので、同じ建て直し地点から逆方向に入った場合の
        勝敗は WIN/LOSE をそのまま入れ替えたものに一致する（AMBIGUOUSは不変）。"""
        flip = {"WIN": "LOSE", "LOSE": "WIN"}
        return [flip.get(r, r) for r in rs]

    def race_summary(rs: list[str]) -> dict:
        n = len(rs)
        if n == 0:
            return {"n": 0}
        c = {k: rs.count(k) for k in ("WIN", "LOSE", "AMBIGUOUS", "NEITHER")}
        decided = c["WIN"] + c["LOSE"]
        return {
            "n": n,
            **{k.lower(): c[k] for k in c},
            "win_rate_decided": round(c["WIN"] / decided, 4) if decided else None,
        }

    result = {
        "generated_at": datetime.now().isoformat(),
        "status": "探索的診断（正式プロトコル外・spec編集なし）",
        "question": "N_BREAKOUT後の戻りでトレンドが形成されているという目視印象は、対照群に対して優位か",
        "period": {"train_start": TRAIN_START, "train_end": TRAIN_END},
        "pairs": PAIRS,
        "definitions": {
            "n_breakout": f"H1レンジ/ATR({ATR_LEN},プロジェクト実装=TRのSMA) >= {N_BREAKOUT}",
            "direction": "ブレイクバー自身の close>open なら UP、close<open なら DOWN",
            "anchor": "ブレイクバーの終値時点（アンカーバー自身は先の窓に含めない=先読みなし）",
            "horizons_h1_bars": HORIZONS_H1,
            "horizon_note": "バー本数ベースのため週末ギャップを跨ぐ。時間ベースではない",
            "normalization": "全てアンカー時点のATR(H1,14)で割った単位",
            "barrier_race": f"アンカー終値から順行+{BARRIER_ATR}ATR と 逆行-{BARRIER_ATR}ATR の"
                            f"どちらに先に触れるかを最大{BARRIER_MAX_BARS}H1バーまで追跡。"
                            "同一H1バー内で両側到達した場合はAMBIGUOUS（H1粒度では順序判定不能）",
            "ctrl_all": "同期間の全H1バーを、そのバー自身の陽線/陰線方向でアンカーにした対照群",
            "ctrl_normal": "レンジ/ATRが1.0〜1.5の平凡なバーだけをアンカーにした対照群",
            "retrace_split": f"確定後{RETRACE_WINDOW_START_MIN}分〜{RETRACE_WINDOW_END_HOURS}時間(M15)で"
                             f"ブレイクバーレンジの{RETRACE_SPLIT:.0%}以上戻したか",
        },
        "per_pair_events": per_pair_events,
        "forward_move_by_group": {
            g: {
                str(h): {
                    "ret_atr": summarize(pooled[g][h]["ret"], rng),
                    "mfe_atr": summarize(pooled[g][h]["mfe"], rng),
                    "mae_atr": summarize(pooled[g][h]["mae"], rng),
                }
                for h in HORIZONS_H1
            }
            for g in groups
        },
        "barrier_race_by_group": {g: race_summary(races[g]) for g in groups},
        "retrace_stratified": {
            a: {
                "retrace_frac": summarize(retrace_fracs[a], rng),
                "leaky_same_window": {
                    k: {
                        "barrier_race_breakout_direction": race_summary(leaky_split[a][k]),
                        "barrier_race_counter_direction": race_summary(counter(leaky_split[a][k])),
                    }
                    for k in SPLITS
                },
                "clean_after_window": {
                    k: {
                        "barrier_race_breakout_direction": race_summary(clean_split[a][k]),
                        "barrier_race_counter_direction": race_summary(counter(clean_split[a][k])),
                        "fwd_ret_atr_at_72bars_breakout_direction": summarize(clean_fwd[a][k], rng),
                        "fwd_ret_atr_at_72bars_counter_direction": summarize(
                            [-v for v in clean_fwd[a][k]], rng),
                    }
                    for k in SPLITS
                },
            }
            for a in ANCHORS
        },
        "leakage_note": (
            "leaky_same_window は、戻り幅を測る窓（ブレイク後30分〜3時間）と "
            "±1ATRバリアレースの窓（ブレイクバー終値起点）が重なっているため、"
            "『大きく戻った＝すでに逆行バリアに触れている』という同義反復になる。"
            "clean_after_window は戻り判定窓の終端で建て直してから判定するため、"
            "この自己参照がない。両者の差が、目視で『戻ってからトレンドになる』と"
            "見えてしまう錯覚の大きさそのものを表す。"
        ),
        "anchor_note": (
            "confirmed = ブレイクH1バーが確定した時刻(バーオープン+1h)を起点。"
            "00-spec.md §『探索窓M』の文言どおりで実運用可能。"
            "bar_open = 現行実装(simulate_dow_theory_trend の break_time = h1.index[i])と"
            "同じバーオープン時刻起点。判定窓の前半がブレイクバー自身の後半と重なる。"
        ),
        "atr_vs_cost_context": atr_cost_context,
        "counter_entry_note": (
            "司令塔の仮説「ブレーク後の戻りでブレーク方向とは逆にエントリーすれば勝率が上がるのでは」"
            "への直接の回答は clean_after_window の barrier_race_counter_direction。"
            "±1ATRの対称バリアなので、順張りの勝率と逆張りの勝率は必ず足して1になる"
            "（＝どちらかが有利ならもう一方は不利）。leaky_same_window の逆張り勝率が高く見えるのは、"
            "戻り幅を測る窓の中で既に逆行バリアに触れているのを数えているためで、実運用では取れない。"
        ),
        "caveats": [
            "コスト（スプレッド・スリッページ・スワップ）は一切考慮していない生の値動きの統計。",
            "週末クローズ制約・エントリー執行ルールも考慮していないため、戦略の期待値ではない。",
            "対照群のバリアレースはサンプリング（各通貨最大1500件）で算出している。",
            "Train期間のみ。Validation/Testは正式検証のために温存している。",
        ],
    }

    out = ROOT / "research" / "method-notes" / "post_breakout_trend_visual_check.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n出力: {out}")

    print("\n--- 順行リターン(ATR単位) 平均 ---")
    for h in HORIZONS_H1:
        row = " | ".join(
            f"{g}: {result['forward_move_by_group'][g][str(h)]['ret_atr'].get('mean')}"
            f" (n={result['forward_move_by_group'][g][str(h)]['ret_atr'].get('n')})"
            for g in groups
        )
        print(f"+{h}H1バー: {row}")
    print("\n--- ±1ATRバリアレース 勝率(決着分のみ) ---")
    for g in groups:
        print(f"{g}: {result['barrier_race_by_group'][g]}")
    print("\n--- 戻り層別 (confirmed起点) ---")
    for k in SPLITS:
        rs = result["retrace_stratified"]["confirmed"]
        lk = rs["leaky_same_window"][k]
        cl = rs["clean_after_window"][k]
        print(f"  [{k}] リーク版      順張り{lk['barrier_race_breakout_direction']['win_rate_decided']} / "
              f"逆張り{lk['barrier_race_counter_direction']['win_rate_decided']} "
              f"(n={lk['barrier_race_breakout_direction']['n']})")
        print(f"  [{k}] リークなし版  順張り{cl['barrier_race_breakout_direction']['win_rate_decided']} / "
              f"逆張り{cl['barrier_race_counter_direction']['win_rate_decided']} "
              f"(n={cl['barrier_race_breakout_direction']['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
