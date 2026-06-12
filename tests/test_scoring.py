from __future__ import annotations

import pandas as pd

from src.scoring.composites import weighted_average
from src.scoring.force_scores import build_force_scores, build_indicator_panel, latest_indicator_scores, latest_scores_table


def test_weighted_average_ignores_missing_values() -> None:
    value = weighted_average({"a": 10.0, "b": float("nan"), "c": 30.0}, {"a": 1.0, "b": 10.0, "c": 1.0})
    assert value == 20.0


def test_basic_indicator_to_force_pipeline() -> None:
    indicators = {
        "test_indicator": {
            "force": "debt_money",
            "component": "real_world",
            "source": "fred",
            "frequency": "daily",
            "ticker_or_series_id": "TEST",
            "transform": "level",
            "direction": "positive",
            "weight": 1.0,
            "market_pricing_flag": False,
            "required": True,
        }
    }
    scoring_config = {
        "calendar": {"daily_forward_fill_limit": {"daily": 3}},
        "scoring": {
            "robust_window_days": 5,
            "z_window_days": 5,
            "percentile_windows": {"percentile_1y": 5, "percentile_5y": 5, "percentile_10y": 5},
            "change_windows": {"change_1m": 2, "change_3m": 3},
        },
        "source_quality": {"fred": 95},
        "force_weights": {"macro_fragility": {"debt_money_score": 1.0}, "productivity_upside": {}},
        "confidence": {
            "availability_weight": 0.45,
            "source_quality_weight": 0.30,
            "freshness_weight": 0.25,
            "missing_required_penalty": 12,
            "missing_optional_penalty": 3,
        },
    }
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "raw_value": range(10),
            "indicator_id": "test_indicator",
            "source": "fred",
            "source_series_id": "TEST",
            "last_updated_date": pd.date_range("2024-01-01", periods=10, freq="D"),
        }
    )
    panel = build_indicator_panel(raw, indicators, scoring_config, "2024-01-01", "2024-01-10")
    latest = latest_indicator_scores(panel, "2024-01-10")
    force_scores = build_force_scores(panel, indicators, scoring_config)
    latest_scores = latest_scores_table(force_scores, latest, indicators, scoring_config, "2024-01-10")
    assert "score_0_100" in panel.columns
    assert "debt_money_score" in force_scores.columns
    assert not latest_scores.empty
