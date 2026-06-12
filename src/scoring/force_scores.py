from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.scoring.composites import compute_composite_columns, series_weighted_average, weighted_average
from src.scoring.transforms import (
    apply_transform,
    coalesce_score,
    robust_z_score,
    robust_z_to_score,
    rolling_percentile,
    z_score,
)
from src.utils.dates import daily_index

LOGGER = logging.getLogger(__name__)


def build_indicator_panel(
    raw_data: pd.DataFrame,
    indicators: dict[str, dict[str, Any]],
    scoring_config: dict[str, Any],
    start_date: str,
    as_of_date: str,
) -> pd.DataFrame:
    """Convert heterogeneous raw observations into a standardized indicator panel."""
    frames: list[pd.DataFrame] = []
    raw_data = _normalize_raw_data(raw_data)

    for indicator_id, meta in indicators.items():
        indicator_raw = raw_data[raw_data["indicator_id"] == indicator_id] if not raw_data.empty else pd.DataFrame()
        if indicator_raw.empty or pd.to_numeric(indicator_raw.get("raw_value"), errors="coerce").dropna().empty:
            frames.append(_missing_indicator_row(indicator_id, meta, as_of_date, indicator_raw))
            continue

        try:
            frames.append(_score_one_indicator(indicator_id, meta, indicator_raw, scoring_config, start_date, as_of_date))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Scoring failed for %s: %s", indicator_id, exc)
            frames.append(_missing_indicator_row(indicator_id, meta, as_of_date, indicator_raw))

    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values(["indicator_id", "date"]).reset_index(drop=True)


def latest_indicator_scores(indicator_panel: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    if indicator_panel.empty:
        return indicator_panel
    as_of = pd.to_datetime(as_of_date)
    latest_rows: list[pd.Series] = []
    for _, frame in indicator_panel[indicator_panel["date"] <= as_of].groupby("indicator_id", sort=False):
        latest_rows.append(frame.sort_values("date").iloc[-1])
    return pd.DataFrame(latest_rows).reset_index(drop=True)


def build_force_scores(
    indicator_panel: pd.DataFrame,
    indicators: dict[str, dict[str, Any]],
    scoring_config: dict[str, Any],
) -> pd.DataFrame:
    """Aggregate indicator scores into force, component, underpricing, and composite series."""
    if indicator_panel.empty:
        return pd.DataFrame()

    score_matrix = indicator_panel.pivot_table(index="date", columns="indicator_id", values="score_0_100", aggfunc="last")
    force_frame = pd.DataFrame(index=score_matrix.index)

    force_frame["debt_money_score"] = _score_for_filter(score_matrix, indicators, force="debt_money")
    force_frame["debt_money_market_pricing"] = _score_for_filter(
        score_matrix, indicators, force="debt_money", market_pricing=True
    )
    force_frame["debt_money_underpriced_signal"] = force_frame["debt_money_score"] - force_frame["debt_money_market_pricing"]

    force_frame["internal_disorder_real_world"] = _score_for_filter(
        score_matrix, indicators, force="internal_disorder", component="real_world"
    )
    force_frame["internal_disorder_market_pricing"] = _score_for_filter(
        score_matrix, indicators, force="internal_disorder", component="market_pricing"
    )
    force_frame["internal_disorder_score"] = force_frame["internal_disorder_real_world"]
    force_frame["internal_disorder_underpriced_signal"] = (
        force_frame["internal_disorder_real_world"] - force_frame["internal_disorder_market_pricing"]
    )

    force_frame["geopolitical_real_world"] = _score_for_filter(
        score_matrix, indicators, force="geopolitical_conflict", component="real_world"
    )
    force_frame["geopolitical_economic_transmission"] = _score_for_filter(
        score_matrix, indicators, force="geopolitical_conflict", component="economic_transmission"
    )
    force_frame["geopolitical_market_pricing"] = _score_for_filter(
        score_matrix, indicators, force="geopolitical_conflict", component="market_pricing"
    )
    force_frame["geopolitical_conflict_score"] = series_weighted_average(
        force_frame[["geopolitical_real_world", "geopolitical_economic_transmission"]],
        {"geopolitical_real_world": 0.65, "geopolitical_economic_transmission": 0.35},
    )
    force_frame["geopolitical_underpriced_signal"] = (
        force_frame["geopolitical_real_world"]
        + force_frame["geopolitical_economic_transmission"]
        - force_frame["geopolitical_market_pricing"]
    )

    force_frame["nature_shock_tactical_score"] = _score_for_filter(
        score_matrix, indicators, force="nature", component="tactical"
    )
    force_frame["nature_pressure_structural_score"] = _score_for_filter(
        score_matrix, indicators, force="nature", component="structural"
    )
    force_frame["nature_market_pricing"] = _score_for_filter(score_matrix, indicators, force="nature", market_pricing=True)
    nature_weights = scoring_config.get("component_weights", {}).get("nature", {})
    force_frame["nature_shock_score"] = series_weighted_average(
        force_frame[["nature_shock_tactical_score", "nature_pressure_structural_score"]],
        {
            "nature_shock_tactical_score": nature_weights.get("tactical", 0.7),
            "nature_pressure_structural_score": nature_weights.get("structural", 0.3),
        },
    )
    force_frame["nature_underpriced_signal"] = force_frame["nature_shock_score"] - force_frame["nature_market_pricing"]

    force_frame["tech_productivity_score"] = _score_for_filter(
        score_matrix, indicators, force="technology", component="productivity"
    )
    force_frame["tech_fragility_score"] = _score_for_filter(score_matrix, indicators, force="technology", component="fragility")
    force_frame["technology_market_pricing"] = force_frame["tech_fragility_score"]
    force_frame["technology_underpriced_signal"] = force_frame["tech_productivity_score"] - force_frame["tech_fragility_score"]

    force_frame = compute_composite_columns(force_frame, scoring_config)
    return force_frame.reset_index().sort_values("date").reset_index(drop=True)


def latest_scores_table(
    force_scores: pd.DataFrame,
    latest_indicators: pd.DataFrame,
    indicators: dict[str, dict[str, Any]],
    scoring_config: dict[str, Any],
    as_of_date: str,
) -> pd.DataFrame:
    """Create machine-readable latest force score rows."""
    if force_scores.empty:
        return pd.DataFrame()
    force_scores = force_scores.sort_values("date")
    latest = force_scores[force_scores["date"] <= pd.to_datetime(as_of_date)].tail(1)
    if latest.empty:
        latest = force_scores.tail(1)
    latest_row = latest.iloc[0]
    historical = force_scores.set_index("date")

    rows = [
        _row("Debt / Money / Economic Cycle", "debt_money_score", "debt_money_market_pricing", "debt_money_underpriced_signal", latest_row, historical, latest_indicators, indicators, scoring_config, "debt_money"),
        _row("Internal Order / Disorder", "internal_disorder_score", "internal_disorder_market_pricing", "internal_disorder_underpriced_signal", latest_row, historical, latest_indicators, indicators, scoring_config, "internal_disorder"),
        _row("Great Power Conflict", "geopolitical_conflict_score", "geopolitical_market_pricing", "geopolitical_underpriced_signal", latest_row, historical, latest_indicators, indicators, scoring_config, "geopolitical_conflict"),
        _row("Acts of Nature", "nature_shock_score", "nature_market_pricing", "nature_underpriced_signal", latest_row, historical, latest_indicators, indicators, scoring_config, "nature"),
        _row("Technology Productivity", "tech_productivity_score", "technology_market_pricing", "technology_underpriced_signal", latest_row, historical, latest_indicators, indicators, scoring_config, "technology", component="productivity"),
        _row("Technology Fragility", "tech_fragility_score", "technology_market_pricing", "technology_underpriced_signal", latest_row, historical, latest_indicators, indicators, scoring_config, "technology", component="fragility"),
        _composite_row("Macro Fragility", "macro_fragility_score", latest_row, historical),
        _composite_row("Productivity Upside", "productivity_upside_score", latest_row, historical),
        _composite_row("Market Setup", "market_setup_score", latest_row, historical),
        _composite_row("Net Tech Setup", "net_tech_setup", latest_row, historical),
    ]
    result = pd.DataFrame(rows)
    result["data_timestamp"] = pd.Timestamp.utcnow().isoformat()
    result["as_of_date"] = pd.to_datetime(latest_row["date"]).date().isoformat()
    return result


def confidence_for_group(
    latest_indicators: pd.DataFrame,
    indicators: dict[str, dict[str, Any]],
    scoring_config: dict[str, Any],
    force: str,
    component: str | None = None,
) -> float:
    selected_ids = [
        indicator_id
        for indicator_id, meta in indicators.items()
        if meta.get("force") == force and (component is None or meta.get("component") == component)
    ]
    if not selected_ids:
        return np.nan

    subset = latest_indicators[latest_indicators["indicator_id"].isin(selected_ids)].copy()
    available = subset["score_0_100"].notna() if not subset.empty else pd.Series(dtype="bool")
    available_count = int(available.sum()) if not subset.empty else 0
    total_count = len(selected_ids)
    availability_score = 100.0 * available_count / total_count if total_count else 0.0

    source_quality = scoring_config.get("source_quality", {})
    qualities = [source_quality.get(indicators[indicator_id].get("source"), 50) for indicator_id in selected_ids]
    source_score = float(np.mean(qualities)) if qualities else 0.0

    if not subset.empty and available_count:
        stale = pd.to_numeric(subset.loc[available, "data_staleness_days"], errors="coerce").fillna(999.0)
        freshness = float(np.maximum(0.0, 100.0 - stale.clip(lower=0).mean()))
    else:
        freshness = 0.0

    confidence_cfg = scoring_config.get("confidence", {})
    confidence = (
        availability_score * confidence_cfg.get("availability_weight", 0.45)
        + source_score * confidence_cfg.get("source_quality_weight", 0.30)
        + freshness * confidence_cfg.get("freshness_weight", 0.25)
    )

    missing_required = [
        indicator_id
        for indicator_id in selected_ids
        if indicators[indicator_id].get("required", False)
        and (subset.empty or subset.loc[subset["indicator_id"] == indicator_id, "score_0_100"].isna().all())
    ]
    missing_optional = [
        indicator_id
        for indicator_id in selected_ids
        if not indicators[indicator_id].get("required", False)
        and (subset.empty or subset.loc[subset["indicator_id"] == indicator_id, "score_0_100"].isna().all())
    ]
    confidence -= len(missing_required) * confidence_cfg.get("missing_required_penalty", 12)
    confidence -= len(missing_optional) * confidence_cfg.get("missing_optional_penalty", 3)
    return float(np.clip(confidence, 0.0, 100.0))


def _normalize_raw_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    if raw_data is None or raw_data.empty:
        return pd.DataFrame(columns=["date", "raw_value", "indicator_id", "source", "source_series_id", "last_updated_date"])
    result = raw_data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["raw_value"] = pd.to_numeric(result["raw_value"], errors="coerce")
    if "last_updated_date" not in result.columns:
        result["last_updated_date"] = result["date"]
    result["last_updated_date"] = pd.to_datetime(result["last_updated_date"], errors="coerce")
    for column in ["source", "source_series_id"]:
        if column not in result.columns:
            result[column] = ""
    return result.dropna(subset=["date"])


def _score_one_indicator(
    indicator_id: str,
    meta: dict[str, Any],
    indicator_raw: pd.DataFrame,
    scoring_config: dict[str, Any],
    start_date: str,
    as_of_date: str,
) -> pd.DataFrame:
    calendar_cfg = scoring_config.get("calendar", {})
    scoring_cfg = scoring_config.get("scoring", {})
    fill_limits = calendar_cfg.get("daily_forward_fill_limit", {})
    frequency = str(meta.get("frequency", "daily"))
    fill_limit = int(fill_limits.get(frequency, 10))

    idx = daily_index(start_date, as_of_date)
    raw_series = (
        indicator_raw.groupby("date")["raw_value"]
        .mean()
        .sort_index()
        .reindex(idx)
        .ffill(limit=fill_limit)
    )
    observed = indicator_raw.dropna(subset=["raw_value"]).groupby("date")["date"].max().sort_index()
    last_observed = observed.reindex(idx).ffill()
    source = str(meta.get("source", indicator_raw["source"].dropna().iloc[-1] if "source" in indicator_raw else ""))
    source_series_id = str(meta.get("ticker_or_series_id", ""))
    source_quality = scoring_config.get("source_quality", {}).get(source, 50)

    transformed = apply_transform(raw_series, str(meta.get("transform", "level")), str(meta.get("direction", "positive")))
    z_window = int(scoring_cfg.get("z_window_days", 756))
    robust_window = int(scoring_cfg.get("robust_window_days", 756))
    percentile_windows = scoring_cfg.get("percentile_windows", {})
    change_windows = scoring_cfg.get("change_windows", {})

    z = z_score(transformed, z_window)
    rz = robust_z_score(transformed, robust_window)
    p1 = rolling_percentile(transformed, int(percentile_windows.get("percentile_1y", 252)))
    p5 = rolling_percentile(transformed, int(percentile_windows.get("percentile_5y", 1260)))
    p10 = rolling_percentile(transformed, int(percentile_windows.get("percentile_10y", 2520)))
    score = coalesce_score(robust_z_to_score(rz), p5, p1, p10)

    frame = pd.DataFrame(
        {
            "date": idx,
            "indicator_id": indicator_id,
            "force": meta.get("force"),
            "component": meta.get("component"),
            "raw_value": raw_series.values,
            "transformed_value": transformed.values,
            "score_0_100": score.values,
            "z_score": z.values,
            "robust_z": rz.values,
            "percentile_1y": p1.values,
            "percentile_5y": p5.values,
            "percentile_10y": p10.values,
            "change_1m": transformed.diff(int(change_windows.get("change_1m", 21))).values,
            "change_3m": transformed.diff(int(change_windows.get("change_3m", 63))).values,
            "last_updated_date": last_observed.values,
            "source": source,
            "source_series_id": source_series_id,
            "confidence": source_quality,
        }
    )
    frame["data_staleness_days"] = (frame["date"] - pd.to_datetime(frame["last_updated_date"])).dt.days
    frame.loc[frame["last_updated_date"].isna(), "data_staleness_days"] = np.nan
    frame.loc[frame["score_0_100"].isna(), "confidence"] = 0.0
    frame.loc[frame["data_staleness_days"] > fill_limit, "confidence"] = (
        frame.loc[frame["data_staleness_days"] > fill_limit, "confidence"] * 0.5
    )
    return frame


def _missing_indicator_row(
    indicator_id: str,
    meta: dict[str, Any],
    as_of_date: str,
    indicator_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    last_updated = pd.NaT
    if indicator_raw is not None and not indicator_raw.empty and "last_updated_date" in indicator_raw.columns:
        last_updated = pd.to_datetime(indicator_raw["last_updated_date"], errors="coerce").dropna()
        last_updated = last_updated.max() if not last_updated.empty else pd.NaT
    as_of = pd.to_datetime(as_of_date)
    staleness = (as_of - last_updated).days if pd.notna(last_updated) else np.nan
    return pd.DataFrame(
        [
            {
                "date": as_of,
                "indicator_id": indicator_id,
                "force": meta.get("force"),
                "component": meta.get("component"),
                "raw_value": np.nan,
                "transformed_value": np.nan,
                "score_0_100": np.nan,
                "z_score": np.nan,
                "robust_z": np.nan,
                "percentile_1y": np.nan,
                "percentile_5y": np.nan,
                "percentile_10y": np.nan,
                "change_1m": np.nan,
                "change_3m": np.nan,
                "last_updated_date": last_updated,
                "data_staleness_days": staleness,
                "source": meta.get("source"),
                "source_series_id": meta.get("ticker_or_series_id"),
                "confidence": 0.0,
            }
        ]
    )


def _score_for_filter(
    score_matrix: pd.DataFrame,
    indicators: dict[str, dict[str, Any]],
    force: str,
    component: str | None = None,
    market_pricing: bool | None = None,
) -> pd.Series:
    selected = {}
    for indicator_id, meta in indicators.items():
        if meta.get("force") != force:
            continue
        if component is not None and meta.get("component") != component:
            continue
        if market_pricing is not None and bool(meta.get("market_pricing_flag", False)) != market_pricing:
            continue
        if indicator_id in score_matrix.columns:
            selected[indicator_id] = float(meta.get("weight", 1.0))
    if not selected:
        return pd.Series(index=score_matrix.index, dtype="float64")
    return series_weighted_average(score_matrix[list(selected)], selected)


def _change_from_history(historical: pd.DataFrame, metric: str, latest_date: pd.Timestamp, days: int) -> float:
    if metric not in historical.columns:
        return np.nan
    target = latest_date - pd.Timedelta(days=days)
    prior = historical[historical.index <= target]
    if prior.empty:
        return np.nan
    current = historical.loc[latest_date, metric]
    previous = prior.iloc[-1][metric]
    if pd.isna(current) or pd.isna(previous):
        return np.nan
    return float(current - previous)


def _row(
    label: str,
    score_metric: str,
    market_metric: str,
    underpriced_metric: str,
    latest_row: pd.Series,
    historical: pd.DataFrame,
    latest_indicators: pd.DataFrame,
    indicators: dict[str, dict[str, Any]],
    scoring_config: dict[str, Any],
    force: str,
    component: str | None = None,
) -> dict[str, Any]:
    latest_date = pd.to_datetime(latest_row["date"])
    return {
        "force": label,
        "metric": score_metric,
        "score": _safe_get(latest_row, score_metric),
        "change_1m": _change_from_history(historical, score_metric, latest_date, 21),
        "change_3m": _change_from_history(historical, score_metric, latest_date, 63),
        "market_pricing": _safe_get(latest_row, market_metric),
        "underpriced_signal": _safe_get(latest_row, underpriced_metric),
        "confidence": confidence_for_group(latest_indicators, indicators, scoring_config, force, component=component),
    }


def _composite_row(label: str, metric: str, latest_row: pd.Series, historical: pd.DataFrame) -> dict[str, Any]:
    latest_date = pd.to_datetime(latest_row["date"])
    return {
        "force": label,
        "metric": metric,
        "score": _safe_get(latest_row, metric),
        "change_1m": _change_from_history(historical, metric, latest_date, 21),
        "change_3m": _change_from_history(historical, metric, latest_date, 63),
        "market_pricing": np.nan,
        "underpriced_signal": np.nan,
        "confidence": np.nan,
    }


def _safe_get(row: pd.Series, key: str) -> float:
    value = row.get(key, np.nan)
    return float(value) if pd.notna(value) else np.nan
