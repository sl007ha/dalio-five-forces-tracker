from __future__ import annotations

import logging
from pathlib import Path


class WarningCollector(logging.Handler):
    """Collect warning-and-above messages for report data-quality notes."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> WarningCollector:
    collector = WarningCollector()
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    collector.setFormatter(formatter)

    handlers: list[logging.Handler] = [logging.StreamHandler(), collector]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    return collector
