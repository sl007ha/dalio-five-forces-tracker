from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


class TechnologyConnector:
    """Placeholder for non-market technology fundamentals.

    Market technology proxies are handled by MarketConnector. This class reserves a
    clean connector boundary for future sources such as patent, capex, or AI-index
    datasets.
    """

    source_name = "technology"

    def __init__(self, indicators: dict[str, dict[str, Any]], data_dir: Path) -> None:
        self.indicators = indicators
        self.data_dir = data_dir
        self.raw: pd.DataFrame = pd.DataFrame()
        self.processed: pd.DataFrame = pd.DataFrame()

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        LOGGER.info("Technology fundamentals connector has no v0.1 external source; market proxies are used")
        self.raw = pd.DataFrame(columns=["date", "raw_value", "indicator_id", "source", "source_series_id", "last_updated_date"])
        return self.raw

    def clean(self) -> pd.DataFrame:
        self.processed = self.raw.copy()
        return self.processed

    def save_raw(self) -> Path | None:
        return None

    def save_processed(self) -> Path | None:
        return None
