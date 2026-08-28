"""EXP-FX000020: 拡張期間（Dukascopy + GMO）の統合データ供給と保守的コストモデルの適用.

`research/EXP-FX000020/00-spec.md` §1・§2 の実装。既存の戦略パイプラインを**一切書き換えずに**
拡張期間で走らせるため、各モジュールのローダとスプレッド定数を差し替えるユーティリティを提供する。

- `load_m5_extended(pair, start, end)`: Dukascopy M5(2021-11〜2023-10) と DS-1(2023-11〜) を
  結合する。重複期間は **GMO(DS-1) を優先**する（コストモデルが GMO 較正であるため）
- `patch_pipelines(cost_level)`: 各モジュールの `load_m5_period` / `load_m5` を差し替え、
  `SPREAD_PIPS` 辞書を**インプレースで更新**して保守的スプレッドを全パイプラインへ波及させる
  （`from ... import SPREAD_PIPS` で参照を共有している呼び出し側にも同時に効く）

**既知の限界（結果に必ず併記すること）**:

- Dukascopy は出来高0バーを除外する仕様のため M5 バーが GMO より約2割少ない。低流動性帯の
  執行判定機会が減る方向であり、損切り検出がわずかに楽観側へぶれる
- **DS-7（スワップ）は 2023-11 以降しか存在しない**ため、拡張期間ではスワップが計上されない。
  両建てグリッドの正味スワップは負（ブローカースプレッド分）なので、この欠落は楽観側に働く。
  ただし実測寄与は 17ヶ月・4通貨で −$13 と小さく、結論を左右する規模ではない
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

DUKA_M5_TAG = "2021-11_2023-10"
EXTENDED_TRAIN = ("2021-11-01", "2025-03-31")   # spec §1: 単一の連続ブロック（41ヶ月）
CURRENT_TRAIN = ("2023-11-01", "2025-03-31")    # 比較用（現行17ヶ月）
COST_MODEL_JSON = ROOT / "research" / "EXP-FX000020" / "10-result" / "conservative_cost_model.json"

_cache: dict[str, pd.DataFrame] = {}


def load_m5_extended(pair: str, start: str, end: str) -> pd.DataFrame:
    """Dukascopy M5 + DS-1 M5 を結合して返す（重複は GMO 優先）."""
    if pair not in _cache:
        frames: list[pd.DataFrame] = []
        duka = ROOT / "data" / "raw" / "dukascopy" / f"ohlcv_{pair}_5min_{DUKA_M5_TAG}.csv"
        if duka.exists():
            frames.append(pd.read_csv(duka, parse_dates=["timestamp"]))
        for f in sorted(glob.glob(str(ROOT / "data" / "raw" / "ds-1" / f"ohlcv_{pair}_5min_*.csv"))):
            frames.append(pd.read_csv(f, parse_dates=["timestamp"]))
        if not frames:
            raise FileNotFoundError(f"M5データが見つかりません: {pair}")
        df = pd.concat(frames)
        # keep="last": 後に置いた DS-1(GMO) が重複期間で勝つ
        df = df.drop_duplicates(subset="timestamp", keep="last").set_index("timestamp").sort_index()
        _cache[pair] = df[["open", "high", "low", "close"]]
    df = _cache[pair]
    return df[(df.index >= start) & (df.index <= end)]


def coverage_report(pairs: list[str]) -> dict:
    """拡張データの網羅状況（spec §1.1 の品質ゲート用、損益非依存）."""
    out = {}
    for pair in pairs:
        full = load_m5_extended(pair, "2000-01-01", "2100-01-01")
        ext = full[(full.index >= EXTENDED_TRAIN[0]) & (full.index <= "2023-10-31")]
        cur = full[(full.index >= CURRENT_TRAIN[0]) & (full.index <= CURRENT_TRAIN[1])]
        by_day_ext = ext.groupby(ext.index.date).size()
        by_day_cur = cur.groupby(cur.index.date).size()
        med_cur = float(by_day_cur.median()) if len(by_day_cur) else 0.0
        thin = int((by_day_ext < med_cur * 0.5).sum()) if med_cur else 0
        out[pair] = {
            "n_bars_extension": int(len(ext)), "n_bars_current_train": int(len(cur)),
            "n_days_extension": int(len(by_day_ext)), "n_days_current_train": int(len(by_day_cur)),
            "median_bars_per_day_extension": float(by_day_ext.median()) if len(by_day_ext) else 0.0,
            "median_bars_per_day_current": med_cur,
            "thin_days_extension": thin,
            "thin_days_pct": round(thin / len(by_day_ext) * 100, 2) if len(by_day_ext) else None,
            "density_vs_current_pct": round(
                (float(by_day_ext.median()) / med_cur * 100), 1) if med_cur else None,
            "extension_first": str(ext.index[0]) if len(ext) else None,
            "extension_last": str(ext.index[-1]) if len(ext) else None,
        }
    return out


def junction_continuity(pairs: list[str]) -> dict:
    """接続部（Dukascopy 末 → GMO 頭）の価格連続性（spec §1.1 Q6）."""
    out = {}
    for pair in pairs:
        full = load_m5_extended(pair, "2000-01-01", "2100-01-01")
        before = full[full.index < "2023-11-01"]
        after = full[full.index >= "2023-11-01"]
        if not len(before) or not len(after):
            continue
        last_duka = float(before["close"].iloc[-1])
        first_gmo = float(after["open"].iloc[0])
        pip = 0.01 if "JPY" in pair else 0.0001
        out[pair] = {
            "last_dukascopy_close": last_duka, "first_gmo_open": first_gmo,
            "gap_pips": round((first_gmo - last_duka) / pip, 2),
            "gap_pct": round((first_gmo - last_duka) / last_duka * 100, 4),
        }
    return out


def load_cost_model(level: str = "base_era_ratio") -> dict[str, float]:
    """保守的スプレッド（spec §2）。level: base_era_ratio / sensitivity_x1.5 / sensitivity_x2.0."""
    if not COST_MODEL_JSON.exists():
        raise FileNotFoundError(f"コストモデル未導出: {COST_MODEL_JSON}（先に measure_dukascopy_spread.py）")
    data = json.loads(COST_MODEL_JSON.read_text(encoding="utf-8"))
    return data["sensitivity_levels"][level]


def patch_pipelines(cost_level: str = "base_era_ratio") -> dict[str, float]:
    """既存パイプラインのローダとスプレッドを拡張期間用に差し替える（インプレース）."""
    import backtest_sysfx018_breakeven_sweep_trainonly as fx018
    import backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd as v7
    import grid_portfolio_engine as gpe

    spreads = load_cost_model(cost_level)

    # ローダ差し替え: v7 側の定義と、それを自モジュールへ import 済みの fx018 側の両方
    v7.load_m5_period = load_m5_extended
    fx018.load_m5_period = load_m5_extended
    gpe.load_m5 = load_m5_extended

    # SPREAD_PIPS はインプレース更新（`from ... import SPREAD_PIPS` の参照共有先にも波及する）
    for pair, val in spreads.items():
        if pair in v7.SPREAD_PIPS:
            v7.SPREAD_PIPS[pair] = val
        if pair in gpe.SPREAD_PIPS:
            gpe.SPREAD_PIPS[pair] = val
    assert fx018.SPREAD_PIPS is v7.SPREAD_PIPS, "SPREAD_PIPS の参照共有が崩れています"
    return spreads
