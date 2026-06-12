from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

GPR_URLS = [
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xlsx",
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls",
]


class GprConnector:
    """Fetch the Caldara-Iacoviello GPR spreadsheet when available."""

    source_name = "gpr"

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

    @staticmethod
    def _read_first_available() -> pd.DataFrame:
        last_error: Exception | None = None
        for url in GPR_URLS:
            try:
                return pd.read_excel(url)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error is not None:
            raise last_error
        return pd.DataFrame()

    @staticmethod
    def _normalize_gpr_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        columns = {str(col).strip(): col for col in frame.columns}
        date_col = None
        for name, original in columns.items():
            lowered = name.lower()
            if "date" in lowered or "month" in lowered:
                date_col = original
                break
        value_col = None
        preferred = ["GPR", "GPRD", "GPRH"]
        for candidate in preferred:
            if candidate in columns:
                value_col = columns[candidate]
                break
        if value_col is None:
            for name, original in columns.items():
                if "gpr" in name.lower():
                    value_col = original
                    break
        if date_col is None or value_col is None:
            raise ValueError("Could not identify date and GPR columns")
        normalized = frame[[date_col, value_col]].rename(columns={date_col: "date", value_col: "raw_value"})
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
        normalized["raw_value"] = pd.to_numeric(normalized["raw_value"], errors="coerce")
        return normalized.dropna(subset=["date"]).sort_values("date")

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        if not self._items():
            self.raw = pd.DataFrame()
            return self.raw
        try:
            source_frame = self._normalize_gpr_frame(self._read_first_available())
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("GPR fetch failed: %s", exc)
            self.raw = pd.DataFrame()
            return self.raw

        source_frame = source_frame[
            (source_frame["date"] >= pd.to_datetime(start_date)) & (source_frame["date"] <= pd.to_datetime(end_date))
        ]
        if source_frame.empty:
            LOGGER.warning("GPR source returned no rows in requested window")
            self.raw = pd.DataFrame()
            return self.raw

        frames: list[pd.DataFrame] = []
        for indicator_id, meta in self._items():
            frame = source_frame.copy()
            frame["indicator_id"] = indicator_id
            frame["source"] = self.source_name
            frame["source_series_id"] = meta.get("ticker_or_series_id", "GPR")
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
        path = self.data_dir / "raw" / "gpr_raw.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw.to_csv(path, index=False)
        return path

    def save_processed(self) -> Path | None:
        if self.processed.empty:
            return None
        path = self.data_dir / "processed" / "gpr_processed.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.processed.to_csv(path, index=False)
        return path
