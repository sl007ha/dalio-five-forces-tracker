from __future__ import annotations

from math import erf, sqrt
from typing import Any

import numpy as np
import pandas as pd


def apply_transform(raw: pd.Series, transform: str, direction: str = "positive") -> pd.Series:
    """Transform a raw series so higher values mean more of the named force."""
    series = pd.to_numeric(raw, errors="coerce").astype("float64")
    transform = (transform or "level").lower()

    if transform in {"level", "identity"}:
        transformed = series
    elif transform == "yoy_pct_change":
        transformed = series.pct_change(365) * 100.0
    elif transform.startswith("momentum_") or transform.startswith("relative_momentum_"):
        window = _window_from_name(transform)
        transformed = series.pct_change(window) * 100.0
    elif transform.startswith("diff_"):
        window = _window_from_name(transform)
        transformed = series.diff(window)
    elif transform.startswith("drawdown_"):
        window = _window_from_name(transform)
        rolling_high = series.rolling(window=window, min_periods=max(20, window // 4)).max()
        transformed = ((series / rolling_high) - 1.0) * -100.0
        transformed = transformed.clip(lower=0.0)
    elif transform.startswith("volatility_"):
        window = _window_from_name(transform)
        transformed = series.pct_change().rolling(window=window, min_periods=max(5, window // 3)).std() * np.sqrt(252) * 100.0
    else:
        raise ValueError(f"Unsupported transform: {transform}")

    if (direction or "positive").lower() in {"negative", "inverse", "inverted"}:
        transformed = transformed * -1.0
    return transformed


def _window_from_name(name: str) -> int:
    suffix = name.rsplit("_", maxsplit=1)[-1].lower()
    if suffix.endswith("d"):
        suffix = suffix[:-1]
    return int(suffix)


def rolling_percentile(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Percentile rank of the current value inside its rolling window."""
    min_periods = min_periods or min(window, 90)

    def percentile(values: np.ndarray) -> float:
        clean = pd.Series(values).dropna()
        if clean.empty or pd.isna(clean.iloc[-1]):
            return np.nan
        return float((clean <= clean.iloc[-1]).mean() * 100.0)

    return series.rolling(window=window, min_periods=min_periods).apply(percentile, raw=True)


def rolling_mad(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or min(window, 90)

    def mad(values: np.ndarray) -> float:
        clean = pd.Series(values).dropna()
        if clean.empty:
            return np.nan
        median = clean.median()
        return float((clean - median).abs().median())

    return series.rolling(window=window, min_periods=min_periods).apply(mad, raw=True)


def z_score(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or min(window, 90)
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def robust_z_score(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or min(window, 90)
    median = series.rolling(window=window, min_periods=min_periods).median()
    mad = rolling_mad(series, window=window, min_periods=min_periods) * 1.4826
    return (series - median) / mad.replace(0, np.nan)


def robust_z_to_score(robust_z: pd.Series) -> pd.Series:
    """Convert robust z-score to a 0-100 score using the normal CDF."""

    def convert(value: Any) -> float:
        if pd.isna(value):
            return np.nan
        return float(100.0 * 0.5 * (1.0 + erf(float(value) / sqrt(2.0))))

    return robust_z.map(convert).clip(lower=0.0, upper=100.0)


def coalesce_score(*series: pd.Series) -> pd.Series:
    """Use the first non-null score among robust and percentile alternatives."""
    if not series:
        return pd.Series(dtype="float64")
    result = series[0].copy()
    for fallback in series[1:]:
        result = result.fillna(fallback)
    return result.clip(lower=0.0, upper=100.0)
