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

追記 (2026-08-18、提案5「統計基盤の補正」対応):
    上記の符号シャッフルは各トレードの符号を完全に独立に反転させるが、5通貨プール評価
    (複数通貨のトレードを1本の配列に結合して1回のpermutation_test()に渡す運用)では、
    異なる通貨ペアのトレードも独立とみなしてしまう。実測 (research/method-notes/
    market_character.json、D1リターンのTrain期間相関) では、5通貨平均相関0.486
    (実効独立数1.70)、JPYクロス4通貨限定では平均相関0.808 (実効独立数1.17) と、
    通貨ペア間に無視できない相関が存在する。独立性を仮定した符号シャッフルは
    帰無分布の分散を過小評価し、p値を楽観側(有意判定されやすい側)に歪める。

    この歪みを補正するため `permutation_test_clustered()` を追加した。単一通貨での
    検証や、通貨ラベルを持たない既存の診断スクリプトは従来通り `permutation_test()`
    を使用してよい (後方互換のため変更しない)。5通貨プール評価にのみ新関数を使う。

    設計変更の経緯 (2026-08-18、実装中に自己検証で発見): 当初は通貨を
    JPYクロス/EUR_USDの2クラスタに分割し、1回のpermutation試行につきクラスタ数分
    (2個)の符号のみを独立に引く設計だったが、過去3戦略のTVT結果で検証したところ
    帰無分布が実質4通り(2クラスタ×±1)にしか分岐せず、p値が0.25刻みに近い粗い
    離散値しか取れない欠陥が判明した(検証: `scripts/verify_cluster_correction_conclusions.py`)。
    通貨ペアごとの実測相関行列(`PAIR_CORRELATION_MATRIX`)をガウス・コピュラの
    相関としてそのまま使い、通貨ペア単位(5ペア→最大32通りの符号組み合わせ)で
    相関した符号を生成する方式に修正し、粗い離散化を解消した。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

DEFAULT_N_PERMUTATIONS = 1000  # spec 既定

# 通貨間相関行列 (research/method-notes/market_character.json より、D1リターン・Train期間)
PAIR_CORRELATION_MATRIX: dict[str, dict[str, float]] = {
    "USD_JPY": {"USD_JPY": 1.0, "EUR_JPY": 0.7728, "GBP_JPY": 0.7871, "AUD_JPY": 0.6811, "EUR_USD": -0.4428},
    "EUR_JPY": {"USD_JPY": 0.7728, "EUR_JPY": 1.0, "GBP_JPY": 0.9183, "AUD_JPY": 0.8384, "EUR_USD": 0.2249},
    "GBP_JPY": {"USD_JPY": 0.7871, "EUR_JPY": 0.9183, "GBP_JPY": 1.0, "AUD_JPY": 0.8516, "EUR_USD": 0.0902},
    "AUD_JPY": {"USD_JPY": 0.6811, "EUR_JPY": 0.8384, "GBP_JPY": 0.8516, "AUD_JPY": 1.0, "EUR_USD": 0.138},
    "EUR_USD": {"USD_JPY": -0.4428, "EUR_JPY": 0.2249, "GBP_JPY": 0.0902, "AUD_JPY": 0.138, "EUR_USD": 1.0},
}

def effective_pair_count(pairs: Sequence[str]) -> float:
    """指定した通貨サブセットの実効独立数を算出する.

    N_eff = k / (1 + (k-1)*rho_bar) (analyze_market_character.py の
    effective_independent_count() と同一の式・同一の相関行列を使用)。
    """
    unique_pairs = sorted(set(pairs))
    k = len(unique_pairs)
    if k <= 1:
        return float(k)
    off_diag = [
        PAIR_CORRELATION_MATRIX[a][b]
        for i, a in enumerate(unique_pairs)
        for b in unique_pairs[i + 1:]
    ]
    rho_bar = sum(off_diag) / len(off_diag)
    denom = 1.0 + (k - 1) * rho_bar
    return k / denom if denom != 0 else float(k)


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


def permutation_test_clustered(
    trade_pnls: Sequence[float],
    pairs: Sequence[str],
    *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int | None = None,
    correlation_matrix: Mapping[str, Mapping[str, float]] | None = None,
) -> PermutationTestResult:
    """通貨間相関を考慮した permutation test (ガウス・コピュラによる相関符号反転版).

    `permutation_test()` は各トレードの符号を独立に反転させるが、これは複数通貨の
    トレードをプールする際に「異なる通貨ペアのトレードも独立」という誤った前提を
    含む (モジュール docstring の追記参照)。本関数は通貨ペアごとに、実測相関行列
    (`PAIR_CORRELATION_MATRIX`) をガウス・コピュラの相関として使った相関乱数から
    符号を生成する: 1回のpermutation試行につき通貨ペアの数だけ相関した標準正規
    乱数を引き (相関行列のCholesky分解によるガウス・コピュラ)、閾値0で符号化して、
    同一通貨ペアの全トレードへ一括適用する。高相関な通貨ペア (例: EUR/JPY・GBP/JPY
    は相関0.918) はほぼ同じ符号を引きやすくなり、無相関/逆相関の通貨ペア (例:
    EUR/USD) はより独立に近い形で符号が決まる — 実測された相関構造を反映した、
    独立シャッフルより保守的な (広い) 帰無分布を与える。

    簡略化の注記: 生成される符号どうしの相関は、二値変数の性質上、入力したガウス
    相関よりもやや弱くなる (Corr(sign(X),sign(Y)) = (2/π)・arcsin(ρ) の関係)。厳密に
    一致させるには逆変換 ρ_gauss = sin(ρ_target・π/2) が必要だが、実測相関行列に
    この逆変換を適用すると半正定値性が崩れる (実装時に数値確認済み) ため、本関数は
    実測相関をそのままガウス相関として使う単純化を採用している。この単純化は符号
    相関をやや過小評価する方向 (=補正をやや弱める方向) に働くが、無相関を仮定する
    `permutation_test()` よりは常に保守的である。

    Args:
        trade_pnls: 各トレードの損益 (円 or pips、符号付き)。
        pairs: trade_pnls と同じ長さの、各トレードの通貨ペア名 (例: "USD_JPY")。
        n_permutations: シャッフル回数。
        seed: 乱数シード。
        correlation_matrix: 通貨ペア間相関行列 (pair→pair→correlation)。省略時は
            `PAIR_CORRELATION_MATRIX` (market_character.json のTrain期間実測値) を
            使用。行列に無い通貨ペアが混在すると KeyError。

    Returns:
        PermutationTestResult。method フィールドに使用した通貨ペア数を明記する。
    """
    n = len(trade_pnls)
    if len(pairs) != n:
        raise ValueError(f"trade_pnls({n}件)とpairs({len(pairs)}件)の長さが一致しません")
    if n == 0:
        return PermutationTestResult(
            n_trades=0,
            n_permutations=n_permutations,
            observed_statistic=0.0,
            null_mean=0.0,
            null_std=0.0,
            p_value=1.0,
            p_value_two_sided=1.0,
            method="gaussian_copula_pair_sign_flip",
        )

    corr_map = correlation_matrix if correlation_matrix is not None else PAIR_CORRELATION_MATRIX
    unique_pairs = sorted(set(pairs))
    k = len(unique_pairs)
    pair_index = {p: i for i, p in enumerate(unique_pairs)}
    trade_pair_idx = np.array([pair_index[p] for p in pairs])

    pnls = np.asarray(trade_pnls, dtype=float)
    magnitudes = np.abs(pnls)
    observed = float(pnls.mean())

    rng = np.random.default_rng(seed)
    if k == 1:
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n))
    else:
        sigma = np.array([[corr_map[a][b] for b in unique_pairs] for a in unique_pairs])
        sigma = sigma + np.eye(k) * 1e-8  # Cholesky分解の数値安定化用の微小nugget
        chol = np.linalg.cholesky(sigma)
        z = rng.standard_normal(size=(n_permutations, k))
        y = z @ chol.T
        pair_signs = np.where(y >= 0.0, 1.0, -1.0)  # (n_permutations, k)
        signs = pair_signs[:, trade_pair_idx]  # (n_permutations, n)

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
        method=f"gaussian_copula_pair_sign_flip(k_pairs={k})",
    )


__all__ = [
    "PermutationTestResult",
    "permutation_test",
    "permutation_test_clustered",
    "effective_pair_count",
    "PAIR_CORRELATION_MATRIX",
    "DEFAULT_N_PERMUTATIONS",
]
