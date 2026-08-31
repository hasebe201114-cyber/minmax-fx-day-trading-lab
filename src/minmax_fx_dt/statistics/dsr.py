"""Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado (2014).

起源:
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
    Overfitting and Non-Normality" — Journal of Portfolio Management 40(5), 94-107.

    本実装は minmax-fx-eval-framework v0.2 (https://github.com/hasebe201114-cyber/
    minmax-fx-eval-framework) から Phase 1 マージ (2026-08-29) で取り込み。
    親 PJ では参考値扱い (必須 KPI 化は Phase 2 で対応予定)。

目的:
    選択バイアスを補正した「真の Sharpe がゼロより大きい確率」を算出する。
    観測 Sharpe が高く見えても、それが多数の試行中の最大値であるなら
    真のエッジではない可能性が高い。DSR はこれを定量化する。

理論:
    DSR = Φ( (SR - E[max SR*]) · √(T - 1) / √(1 - γ₃·SR + ((γ₄ - 1)/4)·SR²) )

    E[max SR*] = (1 - γ_E) · Φ⁻¹(1 - 1/N) + γ_E · Φ⁻¹(1 - 1/(N·e))

    ここで:
    - SR: 観測された年率 Sharpe
    - T: リターン観測数
    - γ₃: リターンの標本歪度 (skewness)
    - γ₄: リターンの標本尖度 (kurtosis, 生・3 がガウス)
    - Φ: 標準正規 CDF
    - γ_E: Euler-Mascheroni 定数 ≈ 0.5772
    - e: 自然対数の底 ≈ 2.71828
    - N: 試行数（独立戦略・パラメータ組み合わせの数）

使用例 (親 PJ 内):
    >>> import numpy as np
    >>> from minmax_fx_dt.statistics.dsr import deflated_sharpe_ratio
    >>> np.random.seed(42)
    >>> returns = np.random.normal(0.001, 0.01, 100)
    >>> result = deflated_sharpe_ratio(returns, n_trials=7, periods_per_year=252)
    >>> print(f"DSR = {result.dsr:.4f}, threshold = {result.threshold}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

# 物理定数
EULER_MASCHERONI = 0.5772156649015329  # γ_E
EULER_E = math.e  # e

# DSR 必須ゲート閾値 (v0.3 必須化)
#
# ⚠️ 2026-09-01 重要訂正 (claude code 環境 C 査読 30-claudecode-c-review.md B-1 反映)
#
# 以前 (commit fed71a4) は Bailey 2014 論文に「Table 1 (p.96)」「Figure 2 (p.98)」
# および引用符付き "We recommend a minimum DSR of 0.95..." の記述があるかのように
# 注釈していたが、claude code 環境 C が実物 PDF を全文検索した結果、
# これらの図表名・章番号・引用文は論文中に存在しないことが判明した
# (citation hallucination)。これは Mavis 環境 C 査読 m-S3 への対応で
# 同一 LLM による自己査読では見抜けなかった典型例である。
#
# 実態として 0.95 という閾値は:
# 1. Bailey & Lopez de Prado (2014) 論文本体では 0.95 を「規範的推奨」として
#    明示する箇所はない。論文中の数値は p.16 付近の例示で 95% 信頼水準を
#    言及しているのみ。
# 2. しかし DSR の二次資料 (daru.finance, quanterlab.com, metricgate.com 等) では
#    0.95 が「strong evidence of genuine alpha」の閾値として広く流布している。
# 3. **本 PJ での 0.95 採用は「95% 信頼水準の統計的慣習を DSR に適用した
#    本 PJ 独自の設計判断」** として位置づける。
#
# 参照: Bailey, D. H. & López de Prado, M. (2014).
#   "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting,
#   and Non-Normality." Journal of Portfolio Management 40(5), 94-107.
#   doi:10.3905/jpm.2014.40.5.094
DSR_REQUIRED_THRESHOLD = 0.95  # 本 PJ 独自の設計判断 (95% 信頼水準の慣習)


@dataclass
class DeflatedSharpeRatioResult:
    """Deflated Sharpe Ratio 計算結果."""

    sharpe_observed: float
    expected_max_sharpe: float
    skewness: float
    kurtosis: float
    n_observations: int
    n_trials: int
    dsr: float  # 真の Sharpe > 0 の確率 (DSR)
    psr: float  # 比較用: ゼロを閾値とした PSR
    z_statistic: float
    passes_threshold: bool  # DSR ≥ 0.95 なら True
    threshold: float

    def to_dict(self) -> dict:
        return {
            "sharpe_observed": round(self.sharpe_observed, 4),
            "expected_max_sharpe": round(self.expected_max_sharpe, 4),
            "skewness": round(self.skewness, 4),
            "kurtosis": round(self.kurtosis, 4),
            "n_observations": self.n_observations,
            "n_trials": self.n_trials,
            "dsr": round(self.dsr, 4),
            "psr": round(self.psr, 4),
            "z_statistic": round(self.z_statistic, 4),
            "passes_threshold": self.passes_threshold,
            "threshold": self.threshold,
        }


def expected_max_sharpe_ratio(n_trials: int) -> float:
    """N 試行の帰無仮説下での最大 Sharpe 期待値 (E[max SR*]).

    Bailey & Lopez de Prado (2014) 式 4:
        E[max SR*] = (1 - γ_E) · Φ⁻¹(1 - 1/N) + γ_E · Φ⁻¹(1 - 1/(N·e))

    Args:
        n_trials: 試行数（独立な戦略・パラメータ組み合わせの数）。1 以上。

    Returns:
        期待最大 Sharpe。n_trials=1 のとき 0 を返す（後述 Note 参照）。

    Note (v0.3 m-D1 対応):
        N=1 の特別扱い — Bailey 2014 公式では Φ⁻¹(1-1/N) = Φ⁻¹(0) = -∞ となり
        数値的に NaN を返す。本実装は便宜的に 0 を返すが、これは **N=1 が PSR
        (Probabilistic Sharpe Ratio) と同等**（benchmark=0 の単一試行に帰着）
        という解釈に基づく。N=1 の DSR は selection bias 補正が不要なので、
        probabilistic_sharpe_ratio() にフォールバックしても安全。
        Bailey 論文自体は N=1 を前提としない（事前登録で N>=2 が理想）が、
        本 PJ では SYS-FX009 v2 のような「1 試行で停止したケース」を扱えるよう
        N=1 を許容している。
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_trials == 1:
        # N=1: 1 試行の max = 単一値 = 0。PSR と同等（benchmark=0）として扱う。
        # → probabilistic_sharpe_ratio() にフォールバックしても安全。
        return 0.0
    # Φ⁻¹(1 - 1/N) — 上側分位
    z_upper = stats.norm.ppf(1.0 - 1.0 / n_trials)
    # Φ⁻¹(1 - 1/(N·e)) — Euler 補正項
    z_euler = stats.norm.ppf(1.0 - 1.0 / (n_trials * EULER_E))
    return (1.0 - EULER_MASCHERONI) * z_upper + EULER_MASCHERONI * z_euler


def probabilistic_sharpe_ratio(
    sharpe_observed: float,
    *,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (PSR) — Bailey & Lopez de Prado (2012, 2014).

    観測 Sharpe が、与えられた benchmark SR を有意水準的に上回る確率を返す。
    補正項に γ₃ (skewness) と γ₄ (kurtosis) を含む（非ガウス対応）。

    PSR = Φ( (SR - SR_benchmark) · √(T - 1) / √(1 - γ₃·SR + ((γ₄ - 1)/4)·SR²) )

    Args:
        sharpe_observed: 観測 Sharpe。
        n_observations: リターン観測数。1 以上。
        skewness: リターンの標本歪度。デフォルト 0（ガウス）。
        kurtosis: リターンの標本尖度（生・3 がガウス）。デフォルト 3.0。
        benchmark_sharpe: 比較対象の Sharpe。デフォルト 0。

    Returns:
        0〜1 の確率値。

    Raises:
        ValueError: n_observations < 1、kurtosis < 1、または分母が非正。
    """
    if n_observations < 1:
        raise ValueError(f"n_observations must be >= 1, got {n_observations}")
    if kurtosis < 1.0:
        raise ValueError(f"kurtosis must be >= 1 (raw kurtosis), got {kurtosis}")

    # 分母: √(1 - γ₃·SR + ((γ₄ - 1)/4)·SR²)
    denom_sq = 1.0 - skewness * sharpe_observed + ((kurtosis - 1.0) / 4.0) * sharpe_observed**2
    if denom_sq <= 0.0:
        raise ValueError(
            f"denominator squared is non-positive ({denom_sq}); "
            f"sharpe={sharpe_observed} may be extreme for skewness={skewness}, "
            f"kurtosis={kurtosis}"
        )
    denom = math.sqrt(denom_sq)

    z = (sharpe_observed - benchmark_sharpe) * math.sqrt(n_observations - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: np.ndarray | list[float],
    *,
    n_trials: int,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    threshold: float = DSR_REQUIRED_THRESHOLD,
) -> DeflatedSharpeRatioResult:
    """Deflated Sharpe Ratio (DSR) を計算する.

    Bailey & Lopez de Prado (2014) 公式の完全実装。観測リターン列と試行数 N から
    選択バイアス補正済み Sharpe 確率を返す。

    Args:
        returns: 期間リターン列（例: 日次リターン）。最低 2 観測必要。
        n_trials: 試行数（独立な戦略・パラメータ組み合わせの数）。
        risk_free_rate: リスクフリーレート（期間あたり、デフォルト 0）。
        periods_per_year: 年率換算のための期間数（デフォルト 252 営業日）。
        threshold: 必須ゲートの閾値（デフォルト 0.95 = Bailey 2014 標準）。

    Returns:
        DeflatedSharpeRatioResult インスタンス。

    Raises:
        ValueError: returns 長が < 2、または n_trials < 1。

    Note:
        親 PJ の文脈では、月次リターンを渡して periods_per_year=12 とする想定。
        あるいは日次リターンで periods_per_year=252。
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1:
        raise ValueError(f"returns must be 1D, got shape {r.shape}")
    n = len(r)
    if n < 2:
        raise ValueError(f"returns must have >= 2 observations, got {n}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    excess = r - risk_free_rate
    mean = float(excess.mean())
    std = float(excess.std(ddof=1))  # 標本標準偏差（n-1）
    if std <= 0.0 or not math.isfinite(std):
        # Sharpe 計算不能（リスクゼロまたは非有限）→ 自動的に REJECT
        return DeflatedSharpeRatioResult(
            sharpe_observed=0.0,
            expected_max_sharpe=expected_max_sharpe_ratio(n_trials),
            skewness=0.0,
            kurtosis=3.0,
            n_observations=n,
            n_trials=n_trials,
            dsr=0.0,
            psr=0.0,
            z_statistic=0.0,
            passes_threshold=False,
            threshold=threshold,
        )

    # 年率 Sharpe
    sharpe = mean / std * math.sqrt(periods_per_year)
    # 標本歪度・尖度（scipy の bias-corrected 推定量）
    skewness = float(stats.skew(r, bias=False))
    kurtosis = float(stats.kurtosis(r, fisher=False, bias=False))  # 生（+3 込み）

    e_max = expected_max_sharpe_ratio(n_trials)

    # PSR（比較用: benchmark=0）
    psr = probabilistic_sharpe_ratio(
        sharpe,
        n_observations=n,
        skewness=skewness,
        kurtosis=kurtosis,
        benchmark_sharpe=0.0,
    )

    # DSR（benchmark=E[max SR*]）
    dsr = probabilistic_sharpe_ratio(
        sharpe,
        n_observations=n,
        skewness=skewness,
        kurtosis=kurtosis,
        benchmark_sharpe=e_max,
    )

    # Z 値（参考）
    z = (sharpe - e_max) * math.sqrt(n - 1) / math.sqrt(
        1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2
    )

    return DeflatedSharpeRatioResult(
        sharpe_observed=sharpe,
        expected_max_sharpe=e_max,
        skewness=skewness,
        kurtosis=kurtosis,
        n_observations=n,
        n_trials=n_trials,
        dsr=dsr,
        psr=psr,
        z_statistic=z,
        passes_threshold=(dsr >= threshold),
        threshold=threshold,
    )


__all__ = [
    "DeflatedSharpeRatioResult",
    "EULER_MASCHERONI",
    "EULER_E",
    "DSR_REQUIRED_THRESHOLD",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe_ratio",
]
