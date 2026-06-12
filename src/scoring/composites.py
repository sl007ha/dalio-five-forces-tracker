from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def weighted_average(values: Mapping[str, float] | pd.Series, weights: Mapping[str, float] | None = None) -> float:
    """NaN-aware weighted average."""
    if isinstance(values, pd.Series):
        value_map = values.to_dict()
    else:
        value_map = dict(values)
    weights = weights or {key: 1.0 for key in value_map}

    numerator = 0.0
    denominator = 0.0
    for key, value in value_map.items():
        weight = float(weights.get(key, 0.0))
        if weight <= 0 or pd.isna(value):
            continue
        numerator += float(value) * weight
        denominator += weight
    if denominator == 0:
        return np.nan
    return numerator / denominator


def series_weighted_average(frame: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """NaN-aware weighted average across dataframe columns."""
    aligned_weights = pd.Series(weights, dtype="float64")
    available_cols = [col for col in frame.columns if col in aligned_weights.index]
    if not available_cols:
        return pd.Series(index=frame.index, dtype="float64")
    sub = frame[available_cols].astype("float64")
    w = aligned_weights.loc[available_cols]
    numerator = sub.mul(w, axis=1).sum(axis=1, skipna=True)
    denominator = sub.notna().mul(w, axis=1).sum(axis=1)
    return numerator / denominator.replace(0, np.nan)


def compute_composite_columns(force_frame: pd.DataFrame, scoring_config: dict) -> pd.DataFrame:
    """Add macro/productivity/market setup composites."""
    result = force_frame.copy()
    weights = scoring_config.get("force_weights", {})
    macro_weights = weights.get("macro_fragility", {})
    productivity_weights = weights.get("productivity_upside", {})

    if macro_weights:
        result["macro_fragility_score"] = series_weighted_average(result, macro_weights)
    if productivity_weights:
        result["productivity_upside_score"] = series_weighted_average(result, productivity_weights)

    if {"productivity_upside_score", "macro_fragility_score"}.issubset(result.columns):
        result["market_setup_score"] = result["productivity_upside_score"] - result["macro_fragility_score"]
    if {"tech_productivity_score", "tech_fragility_score"}.issubset(result.columns):
        result["net_tech_setup"] = result["tech_productivity_score"] - result["tech_fragility_score"]
    return result
