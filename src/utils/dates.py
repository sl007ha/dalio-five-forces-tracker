from __future__ import annotations

from datetime import date, datetime

import pandas as pd


def parse_date(value: str | date | datetime | None, default: date | None = None) -> date:
    """Parse a flexible date value into a date."""
    if value is None or value == "":
        if default is None:
            return date.today()
        return default
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def daily_index(start_date: str | date, end_date: str | date) -> pd.DatetimeIndex:
    """Return a daily calendar index, including weekends for daily report continuity."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    return pd.date_range(start=start, end=end, freq="D")


def days_between(later: str | date | datetime, earlier: str | date | datetime) -> int:
    """Return whole days between two dates."""
    return int((pd.to_datetime(later).normalize() - pd.to_datetime(earlier).normalize()).days)
