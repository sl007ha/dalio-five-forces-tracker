from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)


class FredConnector:
    """Fetch FRED series through the public graph CSV endpoint."""

    source_name = "fred"

    def __init__(self, indicators: dict[str, dict[str, Any]], data_dir: Path, timeout: int = 20) -> None:
        self.indicators = indicators
        self.data_dir = data_dir
        self.timeout = timeout
        self.raw: pd.DataFrame = pd.DataFrame()
        self.processed: pd.DataFrame = pd.DataFrame()

    def _items(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (indicator_id, meta)
            for indicator_id, meta in self.indicators.items()
            if meta.get("source") == self.source_name
        ]

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for indicator_id, meta in self._items():
            series_id = str(meta.get("ticker_or_series_id", "")).strip()
            if not series_id:
                LOGGER.warning("FRED indicator %s has no series id", indicator_id)
                continue

            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                series_df = pd.read_csv(StringIO(response.text))
            except Exception as exc:  # noqa: BLE001 - source failures should degrade gracefully.
                LOGGER.warning("FRED fetch failed for %s (%s): %s", indicator_id, series_id, exc)
                continue

            if "observation_date" not in series_df.columns or series_id not in series_df.columns:
                LOGGER.warning("FRED response for %s did not contain expected columns", series_id)
                continue

            series_df = series_df.rename(columns={"observation_date": "date", series_id: "raw_value"})
            series_df["date"] = pd.to_datetime(series_df["date"], errors="coerce")
            series_df["raw_value"] = pd.to_numeric(series_df["raw_value"].replace(".", pd.NA), errors="coerce")
            series_df = series_df.dropna(subset=["date"])
            series_df = series_df[(series_df["date"] >= pd.to_datetime(start_date)) & (series_df["date"] <= pd.to_datetime(end_date))]

            if series_df.empty:
                LOGGER.warning("FRED series %s returned no rows in requested window", series_id)
                continue

            last_valid = series_df.dropna(subset=["raw_value"])["date"].max()
            frame = series_df[["date", "raw_value"]].copy()
            frame["indicator_id"] = indicator_id
            frame["source"] = self.source_name
            frame["source_series_id"] = series_id
            frame["last_updated_date"] = last_valid
            frames.append(frame)

        self.raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self.raw

    def clean(self) -> pd.DataFrame:
        self.processed = self.raw.copy()
        return self.processed

    def save_raw(self) -> Path | None:
        if self.raw.empty:
            return None
        path = self.data_dir / "raw" / "fred_raw.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw.to_csv(path, index=False)
        return path

    def save_processed(self) -> Path | None:
        if self.processed.empty:
            return None
        path = self.data_dir / "processed" / "fred_processed.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.processed.to_csv(path, index=False)
        return path
