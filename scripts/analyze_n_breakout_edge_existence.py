"""Phase 2: 確定後N_BREAKOUTに方向性エッジは存在するか(戦略ロジック非依存の存在検定).

先読みバグ(`analyze_h1_confirm_time_lookahead_impact.py`)により、SYS-FX011/SYS-FX012の
Train・Validationとも先読み除去後に平均Rがゼロ〜マイナスへ落ちることが判明した。
そこで「どう改善するか」より前に、**そもそも確定後のN_BREAKOUTに方向性エッジが
存在するのか**を、戦略ロジック(M5ダウ理論エントリー・SL/TP・コスト)を一切介さずに
検定する。土台が無ければ、どんなエントリー層設計・パラメータ再導出も無意味になる。

## 事前登録(結果を見る前に確定。以下は実行前にコミットされた内容)

### 検出定義(既存の凍結設計をそのまま流用、新規パラメータなし)
- N_BREAKOUT = 3.5、H1バーのレンジ/ATR(14,Wilder)比
- 方向: バーが陽線ならUP、陰線ならDOWN
- 重複除去: `select_non_overlapping_breakout_events()`(同一方向72時間窓)

### 先読みの排除(本分析の核心)
H1バーのラベルは**開始時刻**であり確定時刻ではない。ラベルTのバーは時刻T+1hに確定する。
したがって:
- 基準価格 = ブレイクバー(index i)の終値 = 確定時刻T+1hに初めて既知になる値
- ホライズンh(バー数)後の価格 = `close[i+h]`(時刻T+1h+h に既知)
- 前方リターン = sign × (close[i+h] - close[i]) / ATR[i]   (sign: UP=+1, DOWN=-1)
正の値=ブレイク方向への継続、負の値=反転を意味する。

ホライズンは**バー数**で数える(カレンダー時間ではない)。両データセットとも週末バーが
存在しないため、バー数で揃える方が2データセット間の比較として一貫する。

### データセット(3パネル、独立に検定して再現性を見る)
| パネル | 期間 | 位置づけ |
|---|---|---|
| GMO Train | 2023-11-01〜2025-03-31 | 既にフィッティング済み |
| GMO Validation | 2025-04-01〜2025-11-30 | 修正版も参照済み |
| **Dukascopy** | 2018-11-01〜2023-10-26 | **戦略フィッティング未使用(最もクリーン)** |
Testは凍結中(T-05)のため本分析では一切参照しない。
Dukascopyは元からH1のためリサンプル不要。GMOはM5→H1へリサンプル(既存`to_h1`と同一)。
両者とも「バーのラベル=開始時刻」という同一の規約で扱う。

### 検定
- 統計量: 符号付き前方リターン(ATR単位)の平均
- 検定: `permutation_test_block()`(日付クラスタ単位の符号シャッフル、T-06で確立済み)
- **両側p値**を使う。系統的な反転(平均が負)も「エッジ」であり(逆張りすればよい)、
  片側検定では見落とすため
- ホライズン: [1, 3, 6, 12, 24, 72] バー

### 多重検定補正と判定基準(結果を見る前に固定)
- **主検定(Primary)**: 層別化なしのプール("all")× 6ホライズン = 6検定。
  Bonferroni補正 α = 0.05/6 = 0.008333
- **副検定(Secondary)**: 下記の層別化 × 6ホライズン。Bonferroni補正は副検定の総数で行う。
  層(事前に固定、結果を見て追加・削除しない):
    direction(UP/DOWN)、H1ダウ理論整合(aligned/counter/undetermined)、
    range/ATR強度(中央値以下/超)、実体比率body_frac(中央値以下/超)、通貨ペア別(5)
  中央値は各パネル内で算出する。
- **再現性要件**: あるホライズン/層が「エッジあり」と認められるのは、
  **3パネル中2つ以上でBonferroni補正後も有意** かつ **3パネルすべてで平均の符号が一致**
  する場合に限る。1パネルのみの有意は多重検定・偶然として扱う。
- **対照群**: 各パネルで、ブレイクバー以外のH1バーから同数をランダム抽出し
  (方向は同じルール=陽線/陰線で決定)、同一の検定にかける。対照群で有意判定が
  ほとんど出ないことをサニティチェックとする。

### 結論の書き方(事前登録)
- 上記の再現性要件を満たす組み合わせが**0件** → 「確定後N_BREAKOUTに方向性エッジは
  検出できない」と結論する。
- 1件以上生き残った場合 → その層を明示し、Phase 3(再設計・パラメータ再導出)の
  対象候補として司令塔へ提示する。本スクリプト内では再導出は行わない。

## 制約
正式プロトコル外の探索的診断だが、上記のとおり評価基準は実行前に固定している。
共有本番コード(`src/`・既存`scripts/backtest_*.py`)は一切変更しない。
パラメータ再導出は行わない(司令塔指示「パラメータ再導出は後回し」)。

出力: research/method-notes/n_breakout_edge_existence.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_n_breakout_h1_dow_trend_alignment import h1_dow_trend_direction  # noqa: E402
from backtest_vol_breakout_dow_theory import select_non_overlapping_breakout_events  # noqa: E402
from derive_vol_breakout_entry_params import N_BREAKOUT  # noqa: E402
from minmax_fx_dt.backtest.permutation import permutation_test_block  # noqa: E402
from minmax_fx_dt.strategy.indicators import atr as atr_ind  # noqa: E402

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
HORIZONS = [1, 3, 6, 12, 24, 72]  # H1バー数
ALPHA = 0.05
RANDOM_SEED = 42
DUKA_DIR = ROOT / "data" / "raw" / "dukascopy"
DUKA_TAG = "2018-11_2023-10"
DUKA_START, DUKA_END = "2018-11-01", "2023-10-26"

PANELS = {
    "gmo_train": ("gmo", "2023-11-01", "2025-03-31"),
    "gmo_validation": ("gmo", "2025-04-01", "2025-11-30"),
    "dukascopy_2018_2023": ("dukascopy", DUKA_START, DUKA_END),
}


def load_h1_gmo(pair: str, start: str, end: str) -> pd.DataFrame:
    with (ROOT / "data" / "curated" / "ds-1.json").open(encoding="utf-8") as f:
        ds1 = json.load(f)
    df = pd.DataFrame(ds1["pairs"][pair]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    m5 = df.set_index("timestamp").sort_index()
    m5 = m5[(m5.index >= start) & (m5.index <= end)]
    return pd.DataFrame({c: m5[c].resample("1h").agg(a) for c, a in
                          [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]}).dropna()


def load_h1_dukascopy(pair: str, start: str, end: str) -> pd.DataFrame:
    path = DUKA_DIR / f"ohlcv_{pair}_h1_{DUKA_TAG}.csv"
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[(df.index >= start) & (df.index <= end)]
    return df[["open", "high", "low", "close"]].dropna()


def body_fraction(bar: pd.Series) -> float | None:
    rng = float(bar["high"] - bar["low"])
    if rng <= 0:
        return None
    return abs(float(bar["close"]) - float(bar["open"])) / rng


def measure(h1: pd.DataFrame, atr_h1: pd.Series, idx: int, direction: str) -> dict | None:
    """ブレイクバーidx(確定時刻=ラベル+1h)を起点に、符号付き前方リターンを算出する."""
    a = float(atr_h1.iloc[idx])
    if not np.isfinite(a) or a <= 0:
        return None
    ref = float(h1["close"].iloc[idx])
    sign = 1.0 if direction == "UP" else -1.0
    bar = h1.iloc[idx]
    rng = float(bar["high"] - bar["low"])
    out = {
        "time": h1.index[idx],
        "direction": direction,
        "range_atr": rng / a,
        "body_frac": body_fraction(bar),
        "fwd": {},
    }
    for h in HORIZONS:
        j = idx + h
        if j >= len(h1):
            continue
        out["fwd"][h] = sign * (float(h1["close"].iloc[j]) - ref) / a
    return out if out["fwd"] else None


def detect_panel(kind: str, start: str, end: str, rng_seed: int) -> tuple[list[dict], list[dict]]:
    """1パネル分のブレイクイベントとランダム対照群を作る."""
    events: list[dict] = []
    controls: list[dict] = []
    rs = np.random.RandomState(rng_seed)
    for pair in PAIRS:
        h1 = load_h1_gmo(pair, start, end) if kind == "gmo" else load_h1_dukascopy(pair, start, end)
        if len(h1) < 200:
            continue
        atr_h1 = atr_ind(h1["high"], h1["low"], h1["close"], length=14)
        ratio = ((h1["high"] - h1["low"]) / atr_h1).dropna()
        hits = np.where(ratio.values >= N_BREAKOUT)[0]
        positions = [h1.index.get_loc(ratio.index[i]) for i in hits]
        directions = ["UP" if h1["close"].iloc[p] > h1["open"].iloc[p] else "DOWN" for p in positions]
        dedup = select_non_overlapping_breakout_events(h1.index, positions, directions)
        dirmap = dict(zip(positions, directions, strict=True))

        breakout_set = set(positions)
        for pos in dedup:
            rec = measure(h1, atr_h1, pos, dirmap[pos])
            if rec is None:
                continue
            rec["pair"] = pair
            rec["dow_trend"] = h1_dow_trend_direction(h1, atr_h1, pos)
            events.append(rec)

        # 対照群: ブレイクバー以外から同数をランダム抽出(方向は同じルールで決定)
        valid = [i for i in range(20, len(h1) - max(HORIZONS) - 1) if i not in breakout_set]
        n_pick = min(len(dedup), len(valid))
        if n_pick > 0:
            for pos in rs.choice(valid, size=n_pick, replace=False):
                pos = int(pos)
                d = "UP" if h1["close"].iloc[pos] > h1["open"].iloc[pos] else "DOWN"
                rec = measure(h1, atr_h1, pos, d)
                if rec is not None:
                    rec["pair"] = pair
                    controls.append(rec)
    return events, controls


def build_strata(events: list[dict]) -> dict[str, list[dict]]:
    """事前登録した層別化を適用する(中央値はこのパネル内で算出)."""
    strata: dict[str, list[dict]] = {"all": events}
    for d in ("UP", "DOWN"):
        strata[f"direction={d}"] = [e for e in events if e["direction"] == d]
    for label, want in (("aligned", "same"), ("counter", "diff"), ("undetermined", None)):
        if want is None:
            sel = [e for e in events if e["dow_trend"] is None]
        elif want == "same":
            sel = [e for e in events if e["dow_trend"] is not None and e["dow_trend"] == e["direction"]]
        else:
            sel = [e for e in events if e["dow_trend"] is not None and e["dow_trend"] != e["direction"]]
        strata[f"dow_{label}"] = sel
    rng_med = float(np.median([e["range_atr"] for e in events])) if events else 0.0
    strata[f"range_atr<=median({rng_med:.2f})"] = [e for e in events if e["range_atr"] <= rng_med]
    strata[f"range_atr>median({rng_med:.2f})"] = [e for e in events if e["range_atr"] > rng_med]
    bodies = [e["body_frac"] for e in events if e["body_frac"] is not None]
    body_med = float(np.median(bodies)) if bodies else 0.0
    strata[f"body_frac<=median({body_med:.2f})"] = [
        e for e in events if e["body_frac"] is not None and e["body_frac"] <= body_med]
    strata[f"body_frac>median({body_med:.2f})"] = [
        e for e in events if e["body_frac"] is not None and e["body_frac"] > body_med]
    for pair in PAIRS:
        strata[f"pair={pair}"] = [e for e in events if e["pair"] == pair]
    return strata


def test_group(events: list[dict], horizon: int) -> dict | None:
    vals, clusters = [], []
    for e in events:
        if horizon in e["fwd"]:
            vals.append(e["fwd"][horizon])
            clusters.append(pd.Timestamp(e["time"]).strftime("%Y-%m-%d"))
    if len(vals) < 10:
        return {"n": len(vals), "mean": None, "p_two_sided": None, "note": "n<10のため検定しない"}
    res = permutation_test_block(vals, clusters, seed=RANDOM_SEED)
    arr = np.array(vals)
    return {
        "n": len(vals),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "win_rate": round(float((arr > 0).mean()), 4),
        "p_two_sided": round(float(res.p_value_two_sided), 4),
    }


def run_panel(name: str, kind: str, start: str, end: str) -> dict:
    print(f"\n########## {name} ({kind} {start}〜{end}) ##########")
    events, controls = detect_panel(kind, start, end, RANDOM_SEED)
    print(f"検出イベント数(dedup後、5通貨プール)={len(events)}  対照群={len(controls)}")
    if not events:
        return {"n_events": 0}

    strata = build_strata(events)
    primary = {str(h): test_group(strata["all"], h) for h in HORIZONS}
    secondary = {
        sname: {str(h): test_group(sel, h) for h in HORIZONS}
        for sname, sel in strata.items() if sname != "all"
    }
    control = {str(h): test_group(controls, h) for h in HORIZONS}

    n_secondary = sum(1 for s in secondary.values() for r in s.values()
                      if r and r.get("p_two_sided") is not None)
    alpha_primary = ALPHA / len(HORIZONS)
    alpha_secondary = ALPHA / n_secondary if n_secondary else ALPHA

    print(f"\n--- 主検定(層別化なし、Bonferroni α={alpha_primary:.5f}) ---")
    print(f"{'ホライズン':<12}{'n':>6}{'平均(ATR単位)':>14}{'勝率':>8}{'両側p':>10}{'判定':>8}")
    for h in HORIZONS:
        r = primary[str(h)]
        if r and r.get("p_two_sided") is not None:
            sig = "**有意**" if r["p_two_sided"] < alpha_primary else "-"
            print(f"{str(h)+'バー':<12}{r['n']:>6}{r['mean']:>14.4f}{r['win_rate']:>8.3f}"
                  f"{r['p_two_sided']:>10.4f}{sig:>8}")

    sig_secondary = [(s, h, r) for s, hs in secondary.items() for h, r in hs.items()
                     if r and r.get("p_two_sided") is not None and r["p_two_sided"] < alpha_secondary]
    print(f"\n--- 副検定(層別化、Bonferroni α={alpha_secondary:.6f}、{n_secondary}検定) ---")
    print(f"補正後も有意な層×ホライズン: {len(sig_secondary)}件")
    for s, h, r in sig_secondary:
        print(f"    {s} / {h}バー: n={r['n']} 平均={r['mean']} p={r['p_two_sided']}")

    sig_control = [h for h in HORIZONS if control[str(h)] and control[str(h)].get("p_two_sided") is not None
                   and control[str(h)]["p_two_sided"] < alpha_primary]
    print(f"\n--- 対照群(ランダム時刻、サニティチェック) --- 補正後有意: {len(sig_control)}/{len(HORIZONS)}件")

    return {
        "n_events": len(events), "n_controls": len(controls),
        "alpha_primary": alpha_primary, "alpha_secondary": alpha_secondary,
        "n_secondary_tests": n_secondary,
        "primary": primary, "secondary": secondary, "control": control,
        "significant_secondary": [{"stratum": s, "horizon": h, **r} for s, h, r in sig_secondary],
        "significant_control_horizons": sig_control,
    }


def main() -> int:
    print("=== Phase 2: 確定後N_BREAKOUTに方向性エッジは存在するか ===")
    print(f"事前登録: N={N_BREAKOUT}、ホライズン={HORIZONS}バー、両側permutation(日付ブロック)、"
          f"Bonferroni補正、3パネル中2つ以上で有意かつ符号一致を要件とする\n")

    panels = {name: run_panel(name, kind, s, e) for name, (kind, s, e) in PANELS.items()}

    # --- 事前登録した再現性要件の適用 ---
    print("\n\n########## 統合判定(事前登録した再現性要件) ##########")
    names = [n for n in PANELS if panels[n].get("n_events")]
    replicated = []
    print(f"\n--- 主検定: 各ホライズンの3パネル比較 ---")
    print(f"{'ホライズン':<10}" + "".join(f"{n:>26}" for n in names) + f"{'再現':>10}")
    for h in HORIZONS:
        cells, sigs, signs = [], 0, set()
        for n in names:
            r = panels[n]["primary"].get(str(h))
            if r and r.get("p_two_sided") is not None:
                is_sig = r["p_two_sided"] < panels[n]["alpha_primary"]
                sigs += int(is_sig)
                signs.add(np.sign(r["mean"]))
                cells.append(f"{r['mean']:+.4f}(p={r['p_two_sided']:.3f}){'*' if is_sig else ''}")
            else:
                cells.append("n/a")
        ok = sigs >= 2 and len(signs) == 1
        if ok:
            replicated.append({"family": "primary", "stratum": "all", "horizon": h})
        print(f"{str(h)+'バー':<10}" + "".join(f"{c:>26}" for c in cells) + f"{'YES' if ok else 'no':>10}")

    # 副検定の再現性(いずれかのパネルで有意だった層のみ突き合わせ)
    cand = {(d["stratum"], d["horizon"]) for n in names for d in panels[n]["significant_secondary"]}
    for stratum, h in sorted(cand):
        sigs, signs = 0, set()
        for n in names:
            r = panels[n]["secondary"].get(stratum, {}).get(str(h))
            if r and r.get("p_two_sided") is not None:
                sigs += int(r["p_two_sided"] < panels[n]["alpha_secondary"])
                signs.add(np.sign(r["mean"]))
        if sigs >= 2 and len(signs) == 1:
            replicated.append({"family": "secondary", "stratum": stratum, "horizon": int(h)})

    print(f"\n=== 事前登録の再現性要件を満たした組み合わせ: {len(replicated)}件 ===")
    for r in replicated:
        print(f"    {r}")
    conclusion = (
        "確定後N_BREAKOUTに、事前登録した基準を満たす方向性エッジは検出できなかった"
        if not replicated else
        f"事前登録の基準を満たす組み合わせが{len(replicated)}件生き残った(Phase 3の候補)"
    )
    print(f"\n結論: {conclusion}")

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "purpose": "戦略ロジックを介さず、確定後N_BREAKOUTに方向性エッジが存在するかを検定する(Phase 2)",
        "preregistration": {
            "n_breakout": N_BREAKOUT,
            "horizons_bars": HORIZONS,
            "return_definition": "sign * (close[i+h] - close[i]) / ATR[i]。ホライズンはバー数。"
                                  "基準はブレイクバー終値(=確定時刻に初めて既知)で先読みなし",
            "test": "permutation_test_block(日付クラスタ単位の符号シャッフル)、両側p値",
            "correction": f"主検定 Bonferroni α=0.05/{len(HORIZONS)}、副検定は副検定総数でBonferroni",
            "replication_requirement": "3パネル中2つ以上で補正後有意、かつ3パネルすべてで平均の符号が一致",
            "panels": {k: {"source": v[0], "start": v[1], "end": v[2]} for k, v in PANELS.items()},
            "note": "Testは凍結中(T-05)のため一切参照していない。パラメータ再導出は行っていない",
        },
        "panels": panels,
        "replicated": replicated,
        "conclusion": conclusion,
    }
    out_path = ROOT / "research" / "method-notes" / "n_breakout_edge_existence.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
