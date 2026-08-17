"""レジスタンス / サポートライン検出 (SYS-FX007 v2 中核).

アルゴリズム:
1. Williams Fractal で極値検出 (N 本窓の中央足が最高値/最安値)
2. クラスタリング: 近い位置 (±ATR * cluster_threshold) のフラクタルを統合
3. 強度スコアリング: 接触回数 × 最近のテスト × 範囲接触率
4. フィルタリング: 接触回数 ≥ 3、直近 1 年 ATR * 0.5% 以内の接触のみ

Usage:
    sr_lines = detect_support_resistance(
        high=df["high"],
        low=df["low"],
        atr=df["atr_h4"],
        fractal_window=5,
        min_touches=3,
        cluster_threshold=0.5,
    )
    # sr_lines: list[SRLevel]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class SRLevel:
    """レジスタンス / サポートライン."""

    price: float           # ライン価格
    kind: str              # "RESISTANCE" / "SUPPORT"
    touches: int           # 接触回数
    first_touch: pd.Timestamp  # 最初の接触日
    last_touch: pd.Timestamp   # 最後の接触日
    strength: float        # 強度スコア (0.0 - 1.0)
    fractal_count: int     # 統合されたフラクタル数


def detect_fractals(
    high: pd.Series,
    low: pd.Series,
    window: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """Williams Fractal で極値を検出.

    Args:
        high: 高値系列.
        low: 安値系列.
        window: フラクタル窓幅 (奇数、デフォルト 5).

    Returns:
        (fractal_highs, fractal_lows) - True/False のマスク.
        fractal_highs[i] = True ⇔ high[i] が ±(window-1)/2 範囲内で最大.
    """
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")
    half = window // 2

    # 中央足が最高値/最安値
    highs_arr = high.to_numpy()
    lows_arr = low.to_numpy()
    n = len(highs_arr)

    is_fractal_high = np.zeros(n, dtype=bool)
    is_fractal_low = np.zeros(n, dtype=bool)

    for i in range(half, n - half):
        center_high = highs_arr[i]
        center_low = lows_arr[i]
        # 窓内の最大/最小
        window_highs = highs_arr[i - half : i + half + 1]
        window_lows = lows_arr[i - half : i + half + 1]
        if center_high == window_highs.max() and (window_highs == center_high).sum() == 1:
            is_fractal_high[i] = True
        if center_low == window_lows.min() and (window_lows == center_low).sum() == 1:
            is_fractal_low[i] = True

    return (
        pd.Series(is_fractal_high, index=high.index),
        pd.Series(is_fractal_low, index=low.index),
    )


def zigzag_pivots_typed(
    high: pd.Series,
    low: pd.Series,
    atr_series: pd.Series,
    threshold_atr: float,
) -> list[tuple[int, str]]:
    """ATR正規化したZigZagで「意味のあるスイング」の転換点を種別付きで抽出する.

    OBS000006追記6: Williams Fractal(window=5) はノイズレベルの微小な上下動でも
    極値と判定してしまい、「レンジ/トレンドの持続期間」の代理指標としては使えない
    (実測: 全ペアでギャップ中央値が3本前後に張り付き、フラクタル窓幅5の機械的な
    制約に支配されているだけと判明)。ZigZagはATR比例の閾値以上の反転のみを転換点
    として残すため、この問題を回避する。

    Returns:
        (バーのインデックス, 種別) のリスト。種別は "HIGH"（山）/ "LOW"（谷）で、
        隣接する要素は必ず交互になる。EXP-FX000003（ダブルトップ/ボトム検出）向け
        にzigzag_pivot_indices()を拡張したもの。
    """
    n = len(high)
    pivots: list[tuple[int, str]] = []
    if n == 0:
        return pivots
    trend = 0  # 0=未確定, 1=上昇追跡中, -1=下降追跡中
    running_max_idx, running_max = 0, float(high.iloc[0])
    running_min_idx, running_min = 0, float(low.iloc[0])
    for i in range(1, n):
        atr_i = atr_series.iloc[i]
        if pd.isna(atr_i) or atr_i <= 0:
            continue
        thresh = threshold_atr * float(atr_i)
        h_i, l_i = float(high.iloc[i]), float(low.iloc[i])
        if trend >= 0:
            if h_i > running_max:
                running_max, running_max_idx = h_i, i
            if running_max - l_i >= thresh:
                pivots.append((running_max_idx, "HIGH"))
                trend = -1
                running_min, running_min_idx = l_i, i
                running_max, running_max_idx = h_i, i
                continue
        if trend <= 0:
            if l_i < running_min:
                running_min, running_min_idx = l_i, i
            if h_i - running_min >= thresh:
                pivots.append((running_min_idx, "LOW"))
                trend = 1
                running_max, running_max_idx = h_i, i
                running_min, running_min_idx = l_i, i
    return pivots


def zigzag_pivot_indices(
    high: pd.Series,
    low: pd.Series,
    atr_series: pd.Series,
    threshold_atr: float,
) -> list[int]:
    """ATR正規化したZigZagで「意味のあるスイング」の転換点だけを抽出する.

    種別（山/谷）が不要な場合の簡易版。実装は`zigzag_pivots_typed()`に委譲。

    Returns:
        転換点となったバーのインデックス（位置番号）のリスト。
    """
    return [idx for idx, _kind in zigzag_pivots_typed(high, low, atr_series, threshold_atr)]


def cluster_fractals(
    fractals: pd.Series,
    values: pd.Series,
    cluster_threshold_pct: float = 0.5,
) -> list[tuple[float, int, list[pd.Timestamp]]]:
    """近い位置のフラクタルをクラスタリング.

    Args:
        fractals: True/False マスク.
        values: 価格系列 (high または low).
        cluster_threshold_pct: クラスタリング閾値 (% of mean, デフォルト 0.5%).

    Returns:
        list of (cluster_center_price, fractal_count, touch_dates).
    """
    if fractals.sum() == 0:
        return []

    mask = fractals.values
    indices = np.where(mask)[0]
    prices = values.iloc[indices].to_numpy()
    dates = values.index[indices].to_list()

    if len(prices) == 0:
        return []

    # 価格順にソート
    order = np.argsort(prices)
    prices = prices[order]
    dates = [dates[i] for i in order]

    clusters: list[list[float]] = []
    cluster_dates: list[list] = []
    current_cluster: list[float] = []
    current_dates: list = []
    mean_price = float(np.mean(prices))

    for p, d in zip(prices, dates):
        if not current_cluster:
            current_cluster.append(p)
            current_dates.append(d)
        else:
            cluster_center = float(np.mean(current_cluster))
            threshold = cluster_center * (cluster_threshold_pct / 100.0)
            if abs(p - cluster_center) <= threshold:
                current_cluster.append(p)
                current_dates.append(d)
            else:
                clusters.append(current_cluster)
                cluster_dates.append(current_dates)
                current_cluster = [p]
                current_dates = [d]
    if current_cluster:
        clusters.append(current_cluster)
        cluster_dates.append(current_dates)

    result: list[tuple[float, int, list]] = []
    for cluster_prices, cluster_dts in zip(clusters, cluster_dates):
        center = float(np.mean(cluster_prices))
        result.append((center, len(cluster_prices), cluster_dts))
    return result


def count_touches(
    high: pd.Series,
    low: pd.Series,
    price: float,
    tolerance_pct: float = 0.005,
) -> int:
    """価格が指定ラインに接触した回数をカウント.

    Args:
        high: 高値系列.
        low: 安値系列.
        price: ライン価格.
        tolerance_pct: 接触とみなす許容範囲 (% of price, デフォルト 0.5%).

    Returns:
        接触回数.
    """
    tolerance = price * tolerance_pct
    touched = (high >= (price - tolerance)) & (low <= (price + tolerance))
    return int(touched.sum())


def compute_strength(
    touches: int,
    fractal_count: int,
    last_touch: pd.Timestamp,
    reference_date: pd.Timestamp,
    decay_days: int = 90,
) -> float:
    """S/R ラインの強度スコア (0.0 - 1.0).

    スコア = min(1.0, 接触スコア + フラクタルスコア) × 鮮度スコア

    Args:
        touches: 接触回数.
        fractal_count: 統合されたフラクタル数.
        last_touch: 最後の接触日.
        reference_date: 評価基準日.
        decay_days: 鮮度半減期 (デフォルト 90 日).

    Returns:
        強度スコア (0.0 - 1.0).
    """
    # 接触スコア (接触 5 回で満点)
    touch_score = min(1.0, touches / 5.0)
    # フラクタルスコア (3 つで満点)
    fractal_score = min(1.0, fractal_count / 3.0)
    base = min(1.0, (touch_score + fractal_score) / 1.5)

    # 鮮度スコア (減衰)
    if isinstance(last_touch, pd.Timestamp) and isinstance(reference_date, pd.Timestamp):
        days_old = (reference_date - last_touch).days
        if days_old < 0:
            days_old = 0
        freshness = 0.5 ** (days_old / decay_days)
    else:
        freshness = 0.5
    return base * freshness


def detect_support_resistance(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    fractal_window: int = 5,
    min_touches: int = 3,
    cluster_threshold_pct: float = 0.5,
    touch_tolerance_pct: float = 0.5,
    lookback_days: int = 365,
) -> list[SRLevel]:
    """レジスタンス / サポートラインを検出 (SYS-FX007 v2 の中核).

    Args:
        high: 高値系列 (H4 想定).
        low: 安値系列.
        close: 終値系列 (鮮度判定に使用).
        fractal_window: フラクタル窓幅 (奇数, デフォルト 5).
        min_touches: 最低接触回数 (デフォルト 3).
        cluster_threshold_pct: クラスタリング閾値 (% of cluster center, デフォルト 0.5%).
        touch_tolerance_pct: 接触判定の許容範囲 (% of price, デフォルト 0.5%).
        lookback_days: 評価対象期間 (日, デフォルト 365).

    Returns:
        検出された S/R ラインのリスト (強度スコア降順).
    """
    # lookback でデータを切り出し
    if not isinstance(high.index, pd.DatetimeIndex):
        raise ValueError("high.index must be DatetimeIndex")
    if len(high) == 0:
        return []

    ref_date = high.index[-1]
    cutoff = ref_date - pd.Timedelta(days=lookback_days)
    mask_recent = high.index >= cutoff
    high_recent = high[mask_recent]
    low_recent = low[mask_recent]
    close_recent = close[mask_recent]

    # 1. フラクタル検出
    fractal_highs, fractal_lows = detect_fractals(high_recent, low_recent, window=fractal_window)

    # 2. クラスタリング
    res_clusters = cluster_fractals(
        fractal_highs, high_recent, cluster_threshold_pct=cluster_threshold_pct
    )
    sup_clusters = cluster_fractals(
        fractal_lows, low_recent, cluster_threshold_pct=cluster_threshold_pct
    )

    levels: list[SRLevel] = []

    # 3. レジスタンスライン
    for center_price, fractal_count, dates in res_clusters:
        # count_touches() は tolerance_pct を「比率」(0.005=0.5%) で受け取るのに対し、
        # touch_tolerance_pct は cluster_threshold_pct と同じ「パーセント数値」(0.5=0.5%)
        # の規約。/100 を忘れると許容誤差が100倍(価格の50%)に膨張し、ほぼ全バーが
        # 「接触」判定されてしまう単位不一致バグがあった (OBS000007 追記8で発見)。
        touches = count_touches(high_recent, low_recent, center_price, touch_tolerance_pct / 100.0)
        if touches < min_touches:
            continue
        last_touch = max(dates)
        first_touch = min(dates)
        strength = compute_strength(touches, fractal_count, last_touch, ref_date)
        levels.append(
            SRLevel(
                price=center_price,
                kind="RESISTANCE",
                touches=touches,
                first_touch=first_touch,
                last_touch=last_touch,
                strength=strength,
                fractal_count=fractal_count,
            )
        )

    # 4. サポートライン
    for center_price, fractal_count, dates in sup_clusters:
        # count_touches() は tolerance_pct を「比率」(0.005=0.5%) で受け取るのに対し、
        # touch_tolerance_pct は cluster_threshold_pct と同じ「パーセント数値」(0.5=0.5%)
        # の規約。/100 を忘れると許容誤差が100倍(価格の50%)に膨張し、ほぼ全バーが
        # 「接触」判定されてしまう単位不一致バグがあった (OBS000007 追記8で発見)。
        touches = count_touches(high_recent, low_recent, center_price, touch_tolerance_pct / 100.0)
        if touches < min_touches:
            continue
        last_touch = max(dates)
        first_touch = min(dates)
        strength = compute_strength(touches, fractal_count, last_touch, ref_date)
        levels.append(
            SRLevel(
                price=center_price,
                kind="SUPPORT",
                touches=touches,
                first_touch=first_touch,
                last_touch=last_touch,
                strength=strength,
                fractal_count=fractal_count,
            )
        )

    # 5. 強度スコア降順でソート
    levels.sort(key=lambda x: x.strength, reverse=True)
    return levels


def find_nearest_sr(
    price: float,
    levels: list[SRLevel],
    kind: str | None = None,
) -> SRLevel | None:
    """指定価格に最も近い S/R ラインを返す.

    Args:
        price: 基準価格.
        levels: 検出済み S/R ライン.
        kind: "RESISTANCE" / "SUPPORT" / None (両方).

    Returns:
        最も近い S/R ライン. なければ None.
    """
    candidates = [l for l in levels if kind is None or l.kind == kind]
    if not candidates:
        return None
    return min(candidates, key=lambda l: abs(l.price - price))


def filter_by_strength(
    levels: list[SRLevel],
    min_strength: float = 0.5,
) -> list[SRLevel]:
    """強度スコアでフィルタリング."""
    return [l for l in levels if l.strength >= min_strength]


if __name__ == "__main__":
    # サンプル実行: 合成データで S/R 検出
    import numpy as np
    from datetime import datetime, timedelta

    np.random.seed(42)
    n = 1000
    dates = pd.date_range(end=datetime(2025, 1, 1), periods=n, freq="4h")
    # 100 ± 5 のレンジ + ノイズ
    base = 100.0
    price = base + 5 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 0.5, n)
    high = pd.Series(price + np.abs(np.random.normal(0.3, 0.1, n)), index=dates)
    low = pd.Series(price - np.abs(np.random.normal(0.3, 0.1, n)), index=dates)
    close = pd.Series(price, index=dates)

    levels = detect_support_resistance(high, low, close, min_touches=2)
    print(f"検出された S/R ライン: {len(levels)} 本")
    for lv in levels[:5]:
        print(
            f"  {lv.kind:<11} price={lv.price:>8.3f} touches={lv.touches:>2} "
            f"fractals={lv.fractal_count} strength={lv.strength:.3f}"
        )
