"""トレード損益に対する permutation test (統計的有意性検定).

背景 (OBS000005-検証プロセス構造-adversarialレビュー 差し戻し5):
    spec (research/EXP-FX000001/00-spec.md §パラメータ空間) は「各候補で 1000 回の
    ラベルシャッフルで p 値計算」を要求しているが、これはエントリー条件の判定ラベルを
    シャッフルしてバックテストエンジンをフル再実行する方式を想定した記述である。
    本PJの実測では 1 セル (1 通貨 × 1 期間) のバックテストが数百秒かかるため、
    1000 回のフル再実行は 5 通貨 × 3 期間だけでも非現実的な計算コストになる。

    そのため本実装では、確定済みのトレード損益列 (エンジンを 1 回だけ実行した結果) に
    対する符号シャッフル (sign-flip permutation test) を採用する。これは「観測された
    トレードの勝敗方向がコイン投げで説明できるか」を検定するもので、真のラベルシャッフル
    (エントリー条件そのものを乱数で置き換えて再実行する方式) より弱い検定ではあるが、
    計算コストゼロで「サンプル数が少なすぎて有意差を主張できない」ことを機械的に検出できる。
    この方式選択の妥当性・限界は OBS000005 に明記し、司令塔判断で採用した。

帰無仮説 H0: 各トレードの勝敗方向 (符号) はエッジと無関係な五分五分のコイン投げであり、
    観測された平均損益は偶然の産物である (絶対値=値幅はエッジの有無に関わらず同じという仮定)。
対立仮説 H1: 観測された平均損益はコイン投げでは説明できないほど大きい (=正のエッジがある)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

DEFAULT_N_PERMUTATIONS = 1000  # spec 既定


@dataclass
class PermutationTestResult:
    """permutation test の結果."""

    n_trades: int
    n_permutations: int
    observed_statistic: float  # 観測された平均損益
    null_mean: float           # 帰無分布の平均
    null_std: float            # 帰無分布の標準偏差
    p_value: float             # 片側検定 (正のエッジがあるか)
    p_value_two_sided: float   # 両側検定
    method: str = "sign_flip_mean_pnl"

    def to_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "n_permutations": self.n_permutations,
            "observed_statistic": round(self.observed_statistic, 4),
            "null_mean": round(self.null_mean, 4),
            "null_std": round(self.null_std, 4),
            "p_value": round(self.p_value, 4),
            "p_value_two_sided": round(self.p_value_two_sided, 4),
            "method": self.method,
        }


def permutation_test(
    trade_pnls: Sequence[float],
    *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int | None = None,
) -> PermutationTestResult:
    """トレード損益列に対する符号シャッフル permutation test.

    各トレードの損益の符号を独立に 50% でランダム反転させ (値幅=絶対値は固定)、
    平均損益を n_permutations 回再計算して帰無分布を作る。p 値は帰無分布が観測値以上
    (片側) / 観測値の絶対値以上 (両側) になった割合 (+1 補正、Davison & Hinkley 1997
    の標準的な permutation p 値推定式: (count + 1) / (n_permutations + 1))。

    Args:
        trade_pnls: 各トレードの損益 (円 or pips、符号付き)。0 件・1 件でも動作する
            (検定力はほぼゼロになるが、エラーにはしない — サンプル不足の可視化が目的)。
        n_permutations: シャッフル回数 (spec 既定 1000)。
        seed: 乱数シード。None ならプロセスごとに変動 (再現性が必要な場合は固定値を渡す)。

    Returns:
        PermutationTestResult。n_trades=0 の場合は p_value=1.0 (有意差なし) を返す。
    """
    n = len(trade_pnls)
    if n == 0:
        return PermutationTestResult(
            n_trades=0,
            n_permutations=n_permutations,
            observed_statistic=0.0,
            null_mean=0.0,
            null_std=0.0,
            p_value=1.0,
            p_value_two_sided=1.0,
        )

    pnls = np.asarray(trade_pnls, dtype=float)
    magnitudes = np.abs(pnls)
    observed = float(pnls.mean())

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n))
    null_stats = (signs * magnitudes).mean(axis=1)

    p_value = float((np.sum(null_stats >= observed) + 1) / (n_permutations + 1))
    p_value_two_sided = float(
        (np.sum(np.abs(null_stats) >= abs(observed)) + 1) / (n_permutations + 1)
    )

    return PermutationTestResult(
        n_trades=n,
        n_permutations=n_permutations,
        observed_statistic=observed,
        null_mean=float(null_stats.mean()),
        null_std=float(null_stats.std()),
        p_value=p_value,
        p_value_two_sided=p_value_two_sided,
    )


__all__ = ["PermutationTestResult", "permutation_test", "DEFAULT_N_PERMUTATIONS"]
