from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


class NatureConnector:
    """Placeholder for slow-moving nature-pressure data with explicit staleness."""

    source_name = "nature"

    def __init__(self, indicators: dict[str, dict[str, Any]], data_dir: Path) -> None:
        self.indicators = indicators
        self.data_dir = data_dir
        self.raw: pd.DataFrame = pd.DataFrame()
        self.processed: pd.DataFrame = pd.DataFrame()

    def _items(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (indicator_id, meta)
            for indicator_id, meta in self.indicators.items()
            if meta.get("source") == self.source_name
        ]

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        as_of = pd.to_datetime(end_date)
        placeholder_update = as_of - pd.Timedelta(days=365)
        for indicator_id, meta in self._items():
            LOGGER.warning(
                "Nature structural source %s is a placeholder; official disaster/climate data is not yet wired",
                indicator_id,
            )
            rows.append(
                {
                    "date": as_of,
                    "raw_value": pd.NA,
                    "indicator_id": indicator_id,
                    "source": self.source_name,
                    "source_series_id": meta.get("ticker_or_series_id", "structural_placeholder"),
                    "last_updated_date": placeholder_update,
                }
            )
        self.raw = pd.DataFrame(rows)
        return self.raw

    def clean(self) -> pd.DataFrame:
        self.processed = self.raw.copy()
        return self.processed

    def save_raw(self) -> Path | None:
        if self.raw.empty:
            return None
        path = self.data_dir / "raw" / "nature_raw.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw.to_csv(path, index=False)
        return path

    def save_processed(self) -> Path | None:
        if self.processed.empty:
            return None
        path = self.data_dir / "processed" / "nature_processed.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.processed.to_csv(path, index=False)
        return path
