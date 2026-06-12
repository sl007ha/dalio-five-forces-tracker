from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)


class GdeltConnector:
    """Fetch daily news-intensity proxies from the GDELT DOC 2.0 API."""

    source_name = "gdelt"
    endpoint = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, indicators: dict[str, dict[str, Any]], data_dir: Path, timeout: int = 30) -> None:
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

    @staticmethod
    def _chunks(start_date: str, end_date: str, days: int = 90) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + pd.Timedelta(days=days - 1), end)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end + pd.Timedelta(days=1)
        return chunks

    def _fetch_query(self, query: str, start_date: str, end_date: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for start, end in self._chunks(start_date, end_date):
            params = {
                "query": query,
                "mode": "timelinevolraw",
                "format": "json",
                "startdatetime": start.strftime("%Y%m%d000000"),
                "enddatetime": end.strftime("%Y%m%d235959"),
            }
            url = f"{self.endpoint}?{urlencode(params)}"
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("GDELT query failed for %s to %s: %s", start.date(), end.date(), exc)
                continue

            timeline = payload.get("timeline") or payload.get("Timeline") or []
            rows: list[dict[str, Any]] = []
            for item in timeline:
                raw_date = item.get("date") or item.get("datetime") or item.get("Date")
                raw_value = item.get("value") or item.get("norm") or item.get("Volume") or item.get("count")
                if raw_date is None:
                    continue
                rows.append({"date": raw_date, "raw_value": raw_value})
            if rows:
                frame = pd.DataFrame(rows)
                frame["date"] = pd.to_datetime(frame["date"].astype(str).str.slice(0, 8), format="%Y%m%d", errors="coerce")
                frame["raw_value"] = pd.to_numeric(frame["raw_value"], errors="coerce")
                frames.append(frame.dropna(subset=["date"]))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "raw_value"])

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for indicator_id, meta in self._items():
            query = meta.get("query")
            if not query:
                LOGGER.warning("GDELT indicator %s has no query", indicator_id)
                continue
            frame = self._fetch_query(str(query), start_date, end_date)
            if frame.empty:
                LOGGER.warning("GDELT returned no rows for %s", indicator_id)
                continue
            frame = frame.groupby("date", as_index=False)["raw_value"].mean()
            frame["indicator_id"] = indicator_id
            frame["source"] = self.source_name
            frame["source_series_id"] = indicator_id
            frame["last_updated_date"] = frame.dropna(subset=["raw_value"])["date"].max()
            frames.append(frame)

        self.raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self.raw

    def clean(self) -> pd.DataFrame:
        self.processed = self.raw.copy()
        return self.processed

    def save_raw(self) -> Path | None:
        if self.raw.empty:
            return None
        path = self.data_dir / "raw" / "gdelt_raw.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw.to_csv(path, index=False)
        return path

    def save_processed(self) -> Path | None:
        if self.processed.empty:
            return None
        path = self.data_dir / "processed" / "gdelt_processed.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.processed.to_csv(path, index=False)
        return path
