"""通貨ペア間の相対価値（統計裁定）に構造があるかの探索診断 — Train期間のみ、損益非依存.

## なぜこの診断か

この project は SYS-FX007〜024 の約20戦略をすべて不採用にしてきた。IC分析
(`signal_ic*.json`、GMO期間+Dukascopy 5年で再現) が示す「標準的テクニカル指標に
無条件・線形の方向性予測力なし」という結論と整合的に、方向性を当てにいく設計は
ことごとく失敗した。方向性に依存しない収益源としては

  (a) レンジ内の価格往復  → SYS-FX024 で期待値が負と判明
  (b) スワップキャリー    → SYS-FX010 で価格変動依存が排除できず不採用
  (c) ニュース窓のボラ    → SYS-FX023 でイベント数不足

を検証済みだが、**(d) 通貨ペア間の相対価値（2資産の共和分関係に依拠する平均回帰）
は CLAUDE.md の戦略カテゴリ一覧にも SYSTEMS.md にも一度も登場しておらず、完全に
未検証**である。(a) の「戻る根拠」が価格の物理（レンジ）だったのに対し、(d) の
戻る根拠は「2通貨の相対価格が経済的に結びついている」という別のメカニズムであり、
SYS-FX024 の否定結果は (d) を否定しない。

## 本診断で測ること（損益・勝率・PF等の成績指標は一切参照しない）

ペア (i, j) の対数価格スプレッド  s_t = log(P_i) - β·log(P_j)  について:

  1. β (Train全体のOLS) と決定係数 R^2
  2. 平均回帰の半減期 (AR(1)係数から。単位: H4本 → 営業日換算)
  3. Dickey-Fuller の t 統計量 (定数項あり・ラグなし。5%臨界値 ≈ -2.86)
  4. スプレッドの標準偏差 σ (対数 → %)
  5. **capture_ratio = σ(%) / 往復コスト(%)** ← 最重要
  6. ±1σ 到達回数 (年あたり。サンプル数が稼げるか)

**5 が決定的**である。SYS-FX007〜023 の敗因は一貫して「粗利益が往復コストの数倍
しかない」ことだった。統計裁定は 2レグ建てるためコストが2倍かかるので、ここで
足切りされる可能性が高い。**戦略を組む前に、まずこの比率を見る。**

## 本診断の限界（正直な記録）

- β は Train 全体の OLS であり in-sample。**構造の存在確認であって戦略ではない。**
  戦略化する場合はローリング推定にして先読みを排除する必要がある
- 半減期・DF統計量も Train 全体で測っており、多重比較の補正をしていない
  (8通貨から28組を総当たりするため、偶然有意に見える組が出る)。**採否の根拠には
  使わず、「見るべき組があるか」の絞り込みにのみ使う**

出力: research/method-notes/relative_value_structure.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from grid_portfolio_engine import load_m5, pip_size, to_h4  # noqa: E402

TRAIN_START, TRAIN_END = "2023-11-01", "2025-03-31"

# Train期間をカバーする全8通貨 (DS-1)
PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY", "CAD_JPY", "CHF_JPY", "EUR_USD"]
SPREAD_PIPS = {
    "USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6, "EUR_USD": 0.3,   # 既存確定値
    "NZD_JPY": 1.2, "CAD_JPY": 1.0, "CHF_JPY": 1.5,                                    # SYS-FX016 確定値
}
SLIPPAGE_PIPS = 0.5      # 片道 (T-09 の一般成行)
H4_BARS_PER_DAY = 6


def ols_beta(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """y = a + b*x の OLS。(a, b, R^2) を返す."""
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    b = float(((x - xm) * (y - ym)).sum() / sxx) if sxx > 0 else 0.0
    a = float(ym - b * xm)
    resid = y - (a + b * x)
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return a, b, r2


def dickey_fuller(s: np.ndarray) -> tuple[float, float]:
    """Δs_t = α + ρ·s_(t-1) + ε の OLS から (t統計量, 半減期[bars]) を返す (ラグなしDF)."""
    y = np.diff(s)
    x = s[:-1]
    a, rho, _ = ols_beta(y, x)
    resid = y - (a + rho * x)
    n = len(y)
    dof = n - 2
    if dof <= 0 or rho >= 0:
        return 0.0, float("inf")
    sigma2 = float((resid ** 2).sum()) / dof
    sxx = float(((x - x.mean()) ** 2).sum())
    se = float(np.sqrt(sigma2 / sxx)) if sxx > 0 else float("inf")
    t_stat = rho / se if se > 0 else 0.0
    phi = 1.0 + rho
    half_life = -np.log(2) / np.log(phi) if 0 < phi < 1 else float("inf")
    return float(t_stat), float(half_life)


def main() -> int:
    print("=== 通貨ペア間の相対価値（統計裁定）の構造診断 — Train期間のみ・損益非依存 ===")
    print(f"対象8通貨: {PAIRS}")
    print(f"期間: {TRAIN_START} 〜 {TRAIN_END}  足: H4")
    print("※ 損益・勝率・PF等の成績指標は一切参照しない。β・半減期は in-sample であり構造の存在確認のみ\n")

    closes: dict[str, np.ndarray] = {}
    price_level: dict[str, float] = {}
    index = None
    for pair in PAIRS:
        h4 = to_h4(load_m5(pair, TRAIN_START, TRAIN_END))
        closes[pair] = h4["close"].to_numpy(dtype=float)
        price_level[pair] = float(np.median(closes[pair]))
        index = h4.index if index is None else index
        if len(closes[pair]) != len(closes[PAIRS[0]]):
            raise ValueError(f"H4本数が一致しません: {pair}")
    n_bars = len(closes[PAIRS[0]])
    years = n_bars / (H4_BARS_PER_DAY * 260)
    print(f"  H4本数={n_bars:,} (約{years:.2f}年)\n")

    def leg_cost_pct(pair: str) -> float:
        """1レグ・片道のコスト (スプレッド+スリッページ) を価格に対する%で返す."""
        return (SPREAD_PIPS[pair] + SLIPPAGE_PIPS) * pip_size(pair) / price_level[pair] * 100.0

    rows = []
    for a, b in combinations(PAIRS, 2):
        la, lb = np.log(closes[a]), np.log(closes[b])
        _c, beta, r2 = ols_beta(la, lb)
        spread = la - beta * lb
        t_stat, half_life = dickey_fuller(spread)
        sigma_pct = float(spread.std(ddof=1)) * 100.0
        # 往復コスト: 2レグ × 往復2回。第2レグはβでヘッジ比率が変わるためβを掛ける
        cost_pct = 2 * leg_cost_pct(a) + 2 * abs(beta) * leg_cost_pct(b)
        capture = sigma_pct / cost_pct if cost_pct > 0 else 0.0
        z = (spread - spread.mean()) / spread.std(ddof=1)
        crossings = int((np.abs(z[:-1]) < 1.0).sum() and np.sum((np.abs(z[:-1]) < 1.0) & (np.abs(z[1:]) >= 1.0)))
        rows.append({
            "pair_a": a, "pair_b": b, "beta": round(beta, 4), "r2": round(r2, 4),
            "half_life_bars": round(half_life, 1) if np.isfinite(half_life) else None,
            "half_life_days": round(half_life / H4_BARS_PER_DAY, 2) if np.isfinite(half_life) else None,
            "df_t_stat": round(t_stat, 3),
            "df_stationary_5pct": bool(t_stat < -2.86),
            "sigma_pct": round(sigma_pct, 4),
            "roundtrip_cost_pct": round(cost_pct, 4),
            "capture_ratio": round(capture, 2),
            "sigma_crossings_per_year": round(crossings / years, 1),
        })

    rows.sort(key=lambda r: -r["capture_ratio"])
    print(f"{'ペア':<22}{'β':>7}{'R2':>7}{'半減期(日)':>11}{'DF-t':>8}{'定常':>6}"
          f"{'σ(%)':>8}{'往復ｺｽﾄ(%)':>11}{'σ/ｺｽﾄ':>9}{'±1σ/年':>9}")
    for r in rows:
        hl = f"{r['half_life_days']}" if r["half_life_days"] is not None else "-"
        print(f"{r['pair_a']}/{r['pair_b']:<12}{r['beta']:>7.3f}{r['r2']:>7.3f}{hl:>11}"
              f"{r['df_t_stat']:>8.2f}{'○' if r['df_stationary_5pct'] else '×':>6}"
              f"{r['sigma_pct']:>8.3f}{r['roundtrip_cost_pct']:>11.4f}"
              f"{r['capture_ratio']:>9.1f}{r['sigma_crossings_per_year']:>9.1f}")

    stationary = [r for r in rows if r["df_stationary_5pct"]]
    print(f"\n  DF検定で5%有意に定常なペア: {len(stationary)}/{len(rows)}組")
    print("  ※ 28組の総当たりのため多重比較の補正なし。偶然有意に見える組が数組出るのは想定内")

    # 参考: 比較対象として SYS-FX024 グリッドのコスト比率 (刻み/往復コスト) は 27.9〜54.3倍
    print("\n  [比較] SYS-FX024 グリッドの σ/コスト 相当 = 27.9〜54.3倍 (それでも期待値は負だった)")
    print("  [比較] SYS-FX007〜023 の K5m スプレッドコスト倍率 = おおむね 2〜3倍 (基準3.0に届かず不採用が続いた)")

    best = rows[0]
    print(f"\n  最良の capture_ratio: {best['pair_a']}/{best['pair_b']} = {best['capture_ratio']}倍 "
          f"(半減期{best['half_life_days']}日, DF-t={best['df_t_stat']}, ±1σ到達{best['sigma_crossings_per_year']}回/年)")

    out = {
        "generated_at": datetime.now().isoformat(),
        "status": "探索診断（正式プロトコル外・spec編集なし・損益非依存）",
        "question": "通貨ペア間の相対価値に、コストを上回る規模の平均回帰構造が存在するか",
        "period": {"train_start": TRAIN_START, "train_end": TRAIN_END, "bars_h4": n_bars, "years": round(years, 2)},
        "pairs": PAIRS,
        "spread_pips": SPREAD_PIPS,
        "slippage_pips_per_leg": SLIPPAGE_PIPS,
        "method": {
            "spread": "s_t = log(P_a) - beta*log(P_b)、beta は Train 全体の OLS (in-sample)",
            "half_life": "Δs_t = α + ρ·s_(t-1) + ε の AR(1) から -ln2/ln(1+ρ)",
            "stationarity": "Dickey-Fuller t統計量 (定数項あり・ラグなし)、5%臨界値 ≈ -2.86",
            "cost": "往復コスト% = 2×(スプレッド+スリッページ)/価格 を両レグ分 (第2レグはβ倍)",
            "capture_ratio": "σ(%) / 往復コスト(%)。±1σでエントリーし平均へ回帰する想定の粗利益比率",
            "pnl_independence": "損益・勝率・PF等の成績指標は本診断で一切参照していない",
        },
        "caveats": [
            "beta・半減期・DF統計量はすべて Train 全体の in-sample 推定。構造の存在確認であって戦略ではない",
            "8通貨28組の総当たりで多重比較の補正をしていない。偶然有意に見える組が出るのは想定内であり、"
            "採否の根拠には使わず「見るべき組があるか」の絞り込みにのみ使う",
            "2レグ建てのためコストが単一ペア戦略の約2倍かかる。capture_ratio はその前提で計算済み",
            "GMOが直接扱うクロス(例: AUD/NZD)なら1レグで済みコストは半減しうるが、本診断は保守側の2レグ建てで評価",
        ],
        "results": rows,
        "n_stationary_5pct": len(stationary),
    }
    path = ROOT / "research" / "method-notes" / "relative_value_structure.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[出力]: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
