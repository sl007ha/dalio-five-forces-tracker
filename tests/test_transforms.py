from __future__ import annotations

import pandas as pd

from src.scoring.transforms import apply_transform, robust_z_to_score, rolling_percentile


def test_negative_direction_inverts_level() -> None:
    raw = pd.Series([1.0, 2.0, 3.0])
    transformed = apply_transform(raw, "level", "negative")
    assert transformed.tolist() == [-1.0, -2.0, -3.0]


def test_momentum_transform_uses_pct_change_window() -> None:
    raw = pd.Series([100.0] * 21 + [110.0])
    transformed = apply_transform(raw, "momentum_21d", "positive")
    assert round(float(transformed.iloc[-1]), 2) == 10.0


def test_robust_z_score_conversion_is_monotonic() -> None:
    scores = robust_z_to_score(pd.Series([-1.0, 0.0, 1.0]))
    assert scores.iloc[0] < scores.iloc[1] < scores.iloc[2]


def test_rolling_percentile_scores_last_value() -> None:
    series = pd.Series([1, 2, 3, 4, 5], dtype="float64")
    percentile = rolling_percentile(series, window=5, min_periods=5)
    assert percentile.iloc[-1] == 100.0
