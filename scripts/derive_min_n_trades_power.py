"""min_n_trades (60→66) の再導出: permutation testの検出力(power)から統計的に定める.

背景: 従来の閾値「60」→「66」は、統計的な検出力計算に基づくものではなく、
(1) 60は初期specでの経験則的な値(導出根拠が文書化されていない)、
(2) 66はSYS-FX007ベースラインプリセットの実測トレード数がたまたま66件だった、
という2つの「その場しのぎの数字」が、SYS-FX008・SYS-FX009へ「同じ基準を使う」
という理由だけでそのまま踏襲されてきた。司令塔の指摘を受け、実際に使っている
`backtest.permutation.permutation_test()`(符号シャッフル検定)そのものを使った
検出力シミュレーションで、min_n_tradesを再導出する。

方法:
    1. 実測トレード損益の絶対値分布(値幅)を経験分布としてブートストラップ
       (SYS-FX009 Train pooled、n=179の実測値を使用)
    2. 「真の勝率エッジ」p (=各トレードの符号が+1になる確率)を複数設定し、
       各(p, n)について実際のpermutation_test()関数でp値を計算し、
       p<0.05になった割合(検出力)を推定する
    3. nについては指数探索+二分探索で「検出力80%を達成する最小n」を効率的に求める
    4. K5m(1トレード期待値 > スプレッド往復コスト×3)を満たす最小限のエッジを
       勝率換算し、それを検出するのに必要なnも参考値として算出する

出力: research/method-notes/min_n_trades_power_analysis.json
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

from minmax_fx_dt.backtest.permutation import permutation_test

N_REPS = 150  # 検出力シミュレーションの反復回数 (計算コストとのバランス、±5pt程度のノイズを許容)
N_PERMUTATIONS = 300  # permutation_test 内部のシャッフル回数 (spec既定1000より削減、探索用)
ALPHA = 0.05
TARGET_POWER = 0.80
N_MIN, N_MAX = 10, 4000
WIN_RATE_EDGES = [0.55, 0.60, 0.65, 0.70, 0.7654]  # 0.7654 = SYS-FX009 Train pooled実測勝率


def load_empirical_magnitudes() -> np.ndarray:
    """SYS-FX009 Train pooled (5通貨) の実測トレード損益から値幅(絶対値)分布を取得."""
    pairs = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
    result_dir = ROOT / "research" / "EXP-FX000003" / "10-result" / "train_val_test"
    all_pnls: list[float] = []
    for pair in pairs:
        with (result_dir / f"tvt_{pair}_train.json").open(encoding="utf-8") as f:
            d = json.load(f)
        all_pnls.extend(d["trade_pnls"])
    return np.abs(np.array(all_pnls, dtype=float))


def simulate_power(magnitudes: np.ndarray, n: int, win_rate: float, rng: np.random.Generator) -> float:
    """指定の(n, 真の勝率)で、実際のpermutation_test()関数を使った検出力を推定."""
    hits = 0
    for _ in range(N_REPS):
        sampled_mag = rng.choice(magnitudes, size=n, replace=True)
        signs = rng.choice([-1.0, 1.0], size=n, p=[1 - win_rate, win_rate])
        pnls = signs * sampled_mag
        result = permutation_test(pnls, n_permutations=N_PERMUTATIONS, seed=None)
        if result.p_value < ALPHA:
            hits += 1
    return hits / N_REPS


def find_required_n(magnitudes: np.ndarray, win_rate: float, rng: np.random.Generator) -> tuple[int, dict]:
    """指数探索→二分探索で検出力80%を達成する最小nを求める."""
    trace: dict[int, float] = {}

    def power_at(n: int) -> float:
        if n not in trace:
            trace[n] = simulate_power(magnitudes, n, win_rate, rng)
        return trace[n]

    # 指数探索でpower>=targetとなる上限を見つける
    n = N_MIN
    while power_at(n) < TARGET_POWER and n < N_MAX:
        n = min(n * 2, N_MAX)
    if power_at(n) < TARGET_POWER:
        return -1, trace  # N_MAXでも届かない

    lo = max(N_MIN, n // 2)
    hi = n
    # 二分探索 (10ステップ程度で十分収束)
    for _ in range(8):
        if hi - lo <= 5:
            break
        mid = (lo + hi) // 2
        if power_at(mid) >= TARGET_POWER:
            hi = mid
        else:
            lo = mid + 1
    return hi, trace


def main() -> int:
    print("=== min_n_trades 再導出: permutation testの検出力シミュレーション ===\n")
    magnitudes = load_empirical_magnitudes()
    mean_magnitude = float(magnitudes.mean())
    print(f"実測値幅分布 (SYS-FX009 Train pooled, n={len(magnitudes)}): "
          f"平均値幅={mean_magnitude:.1f}円, 中央値={np.median(magnitudes):.1f}円\n")

    rng = np.random.default_rng(42)

    spread_round_trip_jpy = 6.0  # USD/JPY (research/EXP-FX000003 Train結果より)
    k5m_min_edge_jpy = spread_round_trip_jpy * 3.0
    k5m_win_rate = 0.5 + k5m_min_edge_jpy / (2 * mean_magnitude)
    print(f"K5m最小エッジ(スプレッド往復×3) = {k5m_min_edge_jpy:.1f}円/トレード "
          f"→ 勝率換算 {k5m_win_rate*100:.2f}% (五分五分からわずか+{(k5m_win_rate-0.5)*100:.2f}pt)\n")

    edges_to_test = sorted(set(WIN_RATE_EDGES + [round(k5m_win_rate, 4)]))

    results: dict[str, dict] = {}
    for win_rate in edges_to_test:
        print(f"--- 真の勝率 {win_rate*100:.2f}% (エッジ {(win_rate-0.5)*200:.1f}pt) ---")
        required_n, trace = find_required_n(magnitudes, win_rate, rng)
        for n in sorted(trace):
            print(f"  n={n:>5}  power={trace[n]:.3f}")
        print(f"  → 検出力{TARGET_POWER*100:.0f}%達成の最小n: "
              f"{required_n if required_n > 0 else f'>{N_MAX}(未到達)'}\n")
        results[f"win_rate_{win_rate:.4f}"] = {
            "win_rate": win_rate,
            "edge_pt": round((win_rate - 0.5) * 200, 2),
            "power_trace": {str(k): v for k, v in sorted(trace.items())},
            "required_n_for_target_power": required_n if required_n > 0 else None,
        }

    out_dir = ROOT / "research" / "method-notes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "min_n_trades_power_analysis.json"
    out_path.write_text(
        json.dumps({
            "generated_at": datetime.now().isoformat(),
            "method": "empirical magnitude bootstrap + sign-flip permutation_test() (project's actual function)",
            "magnitude_source": "EXP-FX000003 (SYS-FX009) Train pooled trade_pnls, n=179",
            "mean_magnitude_jpy": mean_magnitude,
            "n_reps_per_power_estimate": N_REPS,
            "n_permutations_per_test": N_PERMUTATIONS,
            "alpha": ALPHA,
            "target_power": TARGET_POWER,
            "k5m_min_edge_jpy": k5m_min_edge_jpy,
            "k5m_implied_win_rate": k5m_win_rate,
            "results": results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
