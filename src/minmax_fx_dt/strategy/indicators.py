"""テクニカル指標 (手書き実装、pandas-ta 不要).

SYS-FX007 で使う指標を純粋な pandas/numpy で実装。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---- 移動平均 ----

def sma(close: pd.Series, length: int = 50) -> pd.Series:
    """単純移動平均線 (SMA)."""
    return close.rolling(window=length, min_periods=length).mean()


def ema(close: pd.Series, length: int = 50) -> pd.Series:
    """指数移動平均線 (EMA)."""
    return close.ewm(span=length, adjust=False, min_periods=length).mean()


# ---- ボラティリティ ----

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=length, min_periods=length).mean()


# ---- レンジ / チャネル ----

def donchian(
    high: pd.Series,
    low: pd.Series,
    lower_length: int = 20,
    upper_length: int = 20,
) -> pd.DataFrame:
    """Donchian Channel.

    Returns:
        DataFrame with columns: DCL (lower), DCM (mid), DCU (upper).
    """
    dcl = low.rolling(window=lower_length, min_periods=lower_length).min()
    dcu = high.rolling(window=upper_length, min_periods=upper_length).max()
    dcm = (dcl + dcu) / 2.0
    return pd.DataFrame({"DCL": dcl, "DCM": dcm, "DCU": dcu})


def bbands(
    close: pd.Series,
    length: int = 20,
    std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands.

    Returns:
        DataFrame with columns: BBL, BBM, BBU, BBB (bandwidth), BBP (%B).
    """
    mid = close.rolling(window=length, min_periods=length).mean()
    sd = close.rolling(window=length, min_periods=length).std(ddof=0)
    bbl = mid - std * sd
    bbu = mid + std * sd
    bandwidth = (bbu - bbl) / mid
    pct_b = (close - bbl) / (bbu - bbl)
    return pd.DataFrame({
        f"BBL_{length}_{std}": bbl,
        f"BBM_{length}_{std}": mid,
        f"BBU_{length}_{std}": bbu,
        f"BBB_{length}_{std}": bandwidth,
        f"BBP_{length}_{std}": pct_b,
    })


# ---- トレンド強度 ----

def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.DataFrame:
    """Average Directional Index (ADX) 手書き実装.

    Returns:
        DataFrame with columns: ADX_<length>, DMP_<length>, DMN_<length>.
    """
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    tr = atr(high, low, close, length=1)  # True Range (no smoothing)
    atr_smooth = tr.rolling(window=length, min_periods=length).sum()

    plus_di = 100 * (plus_dm.rolling(window=length, min_periods=length).sum() / atr_smooth)
    minus_di = 100 * (minus_dm.rolling(window=length, min_periods=length).sum() / atr_smooth)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.rolling(window=length, min_periods=length).mean()
    return pd.DataFrame({
        f"ADX_{length}": adx_val,
        f"DMP_{length}": plus_di,
        f"DMN_{length}": minus_di,
    })


# ---- オシレーター ----

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---- 長期方向判定 (LT レイヤ) ----

def classify_lt_direction(
    close: pd.Series,
    sma_short: pd.Series,
    sma_long: pd.Series,
    adx_value: pd.Series,
    adx_threshold: float = 25.0,
) -> pd.Series:
    """LT 方向のラベル付け.

    Returns:
        pd.Series with values "UP" / "DOWN" / "RANGE".
    """
    direction = pd.Series("RANGE", index=close.index, dtype="object")
    trending = adx_value >= adx_threshold
    above = (close > sma_short) & (sma_short > sma_long)
    below = (close < sma_short) & (sma_short < sma_long)
    direction = direction.mask(trending & above, "UP")
    direction = direction.mask(trending & below, "DOWN")
    return direction


__all__ = [
    "sma", "ema", "atr", "donchian", "bbands", "adx", "rsi",
    "classify_lt_direction",
]
