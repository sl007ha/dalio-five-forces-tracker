from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.asset_mapping import DEFAULT_EVENT_DEFINITIONS


@dataclass(frozen=True)
class EventWindow:
    name: str
    days: int


DEFAULT_WINDOWS = [
    EventWindow("1m", 21),
    EventWindow("3m", 63),
    EventWindow("6m", 126),
]


def detect_events(
    force_scores: pd.DataFrame,
    event_definitions: dict[str, tuple[str, str, float]] | None = None,
) -> pd.DataFrame:
    """Return event dates using only same-day or prior force scores."""
    event_definitions = event_definitions or DEFAULT_EVENT_DEFINITIONS
    if force_scores.empty:
        return pd.DataFrame(columns=["date", "event_name", "metric", "threshold", "value"])

    frame = force_scores.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    rows: list[dict[str, object]] = []
    for event_name, (metric, operator, threshold) in event_definitions.items():
        if metric not in frame.columns:
            continue
        values = pd.to_numeric(frame[metric], errors="coerce")
        if operator == ">":
            mask = values > threshold
        elif operator == "<":
            mask = values < threshold
        else:
            raise ValueError(f"Unsupported event operator: {operator}")
        for _, row in frame[mask].iterrows():
            rows.append(
                {
                    "date": row["date"],
                    "event_name": event_name,
                    "metric": metric,
                    "threshold": threshold,
                    "value": row[metric],
                }
            )
    return pd.DataFrame(rows)


def run_event_study(
    force_scores: pd.DataFrame,
    asset_prices: pd.DataFrame,
    windows: list[EventWindow] | None = None,
) -> pd.DataFrame:
    """Calculate forward outcomes after events without using future data to trigger events."""
    windows = windows or DEFAULT_WINDOWS
    events = detect_events(force_scores)
    if events.empty or asset_prices.empty:
        return pd.DataFrame()

    prices = asset_prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    returns = prices.pct_change()

    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        event_date = pd.to_datetime(event["date"])
        if event_date not in prices.index:
            available = prices.index[prices.index <= event_date]
            if available.empty:
                continue
            event_date = available[-1]
        start_prices = prices.loc[event_date]
        for window in windows:
            forward_dates = prices.index[(prices.index > event_date) & (prices.index <= event_date + pd.Timedelta(days=window.days))]
            if forward_dates.empty:
                continue
            window_prices = prices.loc[forward_dates]
            forward_return = window_prices.iloc[-1] / start_prices - 1.0
            drawdown = window_prices.divide(start_prices).divide(window_prices.divide(start_prices).cummax()) - 1.0
            realized_vol = returns.loc[forward_dates].std() * np.sqrt(252)
            for asset in prices.columns:
                rows.append(
                    {
                        "event_name": event["event_name"],
                        "event_date": event_date,
                        "asset": asset,
                        "window": window.name,
                        "forward_return": forward_return.get(asset),
                        "forward_max_drawdown": drawdown[asset].min(),
                        "forward_realized_volatility": realized_vol.get(asset),
                    }
                )
    return pd.DataFrame(rows)


def summarize_event_study(event_results: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    if event_results.empty:
        return pd.DataFrame()
    grouped = event_results.groupby(["event_name", "asset", "window"], dropna=False)
    summary = grouped.agg(
        number_of_events=("forward_return", "count"),
        median_return=("forward_return", "median"),
        p25_return=("forward_return", lambda x: x.quantile(0.25)),
        p75_return=("forward_return", lambda x: x.quantile(0.75)),
        hit_rate=("forward_return", lambda x: (x > 0).mean()),
        median_forward_max_drawdown=("forward_max_drawdown", "median"),
        median_forward_realized_volatility=("forward_realized_volatility", "median"),
    ).reset_index()

    if events is not None and not events.empty:
        overlap = _average_event_overlap(events)
        summary = summary.merge(overlap, on="event_name", how="left")
    else:
        summary["average_overlap_with_other_force_events"] = np.nan
    return summary


def _average_event_overlap(events: pd.DataFrame) -> pd.DataFrame:
    event_counts = events.groupby("date")["event_name"].nunique().rename("events_on_date")
    merged = events.merge(event_counts, on="date", how="left")
    merged["overlap"] = merged["events_on_date"] - 1
    return (
        merged.groupby("event_name")["overlap"]
        .mean()
        .rename("average_overlap_with_other_force_events")
        .reset_index()
    )
