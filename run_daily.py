from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

import pandas as pd

from src.data_sources.fred import FredConnector
from src.data_sources.gdelt import GdeltConnector
from src.data_sources.gpr import GprConnector
from src.data_sources.market import MarketConnector
from src.data_sources.nature import NatureConnector
from src.data_sources.technology import TechnologyConnector
from src.reporting.daily_report import generate_daily_report
from src.scoring.force_scores import build_force_scores, build_indicator_panel, latest_indicator_scores, latest_scores_table
from src.utils.io import ensure_directories, load_yaml, write_dataframe
from src.utils.logging import setup_logging

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> int:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    ensure_directories(PROJECT_ROOT)
    warning_collector = setup_logging(
        os.getenv("LOG_LEVEL", "INFO"),
        PROJECT_ROOT / "data" / "processed" / "run_daily.log",
    )

    indicators_config = load_yaml(PROJECT_ROOT / "config" / "indicators.yml")
    scoring_config = load_yaml(PROJECT_ROOT / "config" / "scoring.yml")
    indicators = indicators_config.get("indicators", {})
    start_date = os.getenv("START_DATE") or scoring_config.get("calendar", {}).get("default_start_date", "2015-01-01")
    as_of_date = os.getenv("AS_OF_DATE") or date.today().isoformat()

    LOGGER.info("Running Dalio five-forces tracker from %s to %s", start_date, as_of_date)
    frames: list[pd.DataFrame] = []

    connectors = []
    if env_flag("ENABLE_FRED", True):
        connectors.append(FredConnector(indicators, PROJECT_ROOT / "data"))
    if env_flag("ENABLE_MARKET", True):
        connectors.append(MarketConnector(indicators, PROJECT_ROOT / "data"))
    if env_flag("ENABLE_GPR", True):
        connectors.append(GprConnector(indicators, PROJECT_ROOT / "data"))
    if env_flag("ENABLE_GDELT", True):
        connectors.append(GdeltConnector(indicators, PROJECT_ROOT / "data"))
    if env_flag("ENABLE_NATURE_PLACEHOLDER", True):
        connectors.append(NatureConnector(indicators, PROJECT_ROOT / "data"))
    connectors.append(TechnologyConnector(indicators, PROJECT_ROOT / "data"))

    for connector in connectors:
        LOGGER.info("Fetching %s", connector.__class__.__name__)
        try:
            raw = connector.fetch(start_date, as_of_date)
            connector.clean()
            connector.save_raw()
            connector.save_processed()
            if raw is not None and not raw.empty:
                frames.append(raw)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("%s failed and will be skipped: %s", connector.__class__.__name__, exc)

    raw_data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw_data.empty:
        LOGGER.warning("No external source returned usable raw data; outputs will contain missing/low-confidence scores")

    indicator_panel = build_indicator_panel(raw_data, indicators, scoring_config, start_date, as_of_date)
    latest_indicators = latest_indicator_scores(indicator_panel, as_of_date)
    force_scores = build_force_scores(indicator_panel, indicators, scoring_config)
    latest_scores = latest_scores_table(force_scores, latest_indicators, indicators, scoring_config, as_of_date)

    _safe_write(indicator_panel, PROJECT_ROOT / "data" / "processed" / "indicator_panel.parquet")
    _safe_write(force_scores, PROJECT_ROOT / "data" / "processed" / "force_scores.parquet")
    _safe_write(latest_scores, PROJECT_ROOT / "data" / "processed" / "latest_scores.csv")
    _safe_write(latest_scores, PROJECT_ROOT / "data" / "processed" / "latest_scores.json")

    report_path = PROJECT_ROOT / "data" / "reports" / f"{as_of_date}_dalio_five_forces_report.md"
    generate_daily_report(
        as_of_date,
        latest_scores,
        latest_indicators,
        force_scores,
        warning_collector.messages,
        report_path,
        max_drivers=int(scoring_config.get("reporting", {}).get("max_key_drivers", 5)),
    )

    print(f"Generated report: {report_path}")
    print("Latest scores:")
    if latest_scores.empty:
        print("No latest scores generated.")
    else:
        print(latest_scores[["force", "score", "market_pricing", "underpriced_signal", "confidence"]].to_string(index=False))
    if warning_collector.messages:
        print("Warnings:")
        for warning in warning_collector.messages[-20:]:
            print(f"- {warning}")
    return 0


def _safe_write(frame: pd.DataFrame, path: Path) -> None:
    try:
        write_dataframe(frame, path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not write %s: %s", path.name, exc)
        if path.suffix == ".parquet":
            fallback = path.with_suffix(".csv")
            LOGGER.warning("Writing CSV fallback for %s at %s", path.name, fallback)
            frame.to_csv(fallback, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
