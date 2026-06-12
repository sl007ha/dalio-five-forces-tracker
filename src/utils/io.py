from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_directories(root: Path) -> None:
    """Create the project data directories used by the daily run."""
    for rel in [
        "data/raw",
        "data/processed",
        "data/reports",
        "notebooks",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def write_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=index)
    elif suffix == ".json":
        df.to_json(path, orient="records", indent=2, date_format="iso")
    elif suffix == ".parquet":
        df.to_parquet(path, index=index)
    else:
        raise ValueError(f"Unsupported dataframe output type: {path}")


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
