from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate_daily_report(
    report_date: str,
    latest_scores: pd.DataFrame,
    latest_indicators: pd.DataFrame,
    force_scores: pd.DataFrame,
    warnings: list[str],
    output_path: Path,
    max_drivers: int = 5,
) -> Path:
    """Write the daily Markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score_map = _score_map(latest_scores)
    indicator_map = latest_indicators.set_index("indicator_id").to_dict("index") if not latest_indicators.empty else {}

    sections = [
        "# Dalio Five Forces Daily Tracker",
        f"Date: {report_date}",
        "",
        "## Executive Summary",
        *_executive_summary(score_map, latest_scores),
        "",
        "## Scoreboard",
        _scoreboard(latest_scores),
        "",
        _force_section(
            "Force 1: Debt / Money / Economic Cycle",
            "debt_money_score",
            "debt_money",
            score_map,
            latest_indicators,
            max_drivers,
            _debt_implication(score_map),
        ),
        "",
        _force_section(
            "Force 2: Internal Order / Disorder",
            "internal_disorder_score",
            "internal_disorder",
            score_map,
            latest_indicators,
            max_drivers,
            _internal_implication(score_map),
        ),
        "",
        _force_section(
            "Force 3: Great Power Conflict",
            "geopolitical_conflict_score",
            "geopolitical_conflict",
            score_map,
            latest_indicators,
            max_drivers,
            _geo_implication(score_map),
        ),
        "",
        _nature_section(score_map, latest_indicators, indicator_map, max_drivers),
        "",
        _technology_section(score_map, latest_indicators, max_drivers),
        "",
        "## Cross-Asset Implications",
        *_cross_asset_implications(score_map),
        "",
        "## Watchlist",
        *_watchlist(latest_indicators),
        "",
        "## Data Quality Notes",
        *_data_quality_notes(latest_scores, latest_indicators, warnings),
    ]

    output_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return output_path


def _score_map(latest_scores: pd.DataFrame) -> dict[str, float]:
    if latest_scores.empty:
        return {}
    return {
        str(row["metric"]): float(row["score"])
        for _, row in latest_scores.iterrows()
        if pd.notna(row.get("score"))
    } | {
        f"{row['metric']}__market_pricing": float(row["market_pricing"])
        for _, row in latest_scores.iterrows()
        if pd.notna(row.get("market_pricing"))
    } | {
        f"{row['metric']}__underpriced_signal": float(row["underpriced_signal"])
        for _, row in latest_scores.iterrows()
        if pd.notna(row.get("underpriced_signal"))
    }


def _executive_summary(score_map: dict[str, float], latest_scores: pd.DataFrame) -> list[str]:
    macro = score_map.get("macro_fragility_score")
    productivity = score_map.get("productivity_upside_score")
    market_setup = score_map.get("market_setup_score")
    summary = [
        f"- Overall market setup: {_market_setup_text(macro, productivity, market_setup)}",
        f"- Biggest force-level changes: {_biggest_changes(latest_scores)}",
        f"- Biggest underpriced risks: {_biggest_underpriced(latest_scores)}",
        "- Portfolio implications: Use these scores as a risk-management dashboard, not as personalized financial advice.",
    ]
    return summary


def _market_setup_text(macro: float | None, productivity: float | None, market_setup: float | None) -> str:
    if _is_missing(macro) or _is_missing(productivity):
        return "Insufficient data to classify the setup."
    if macro > 80 and productivity > 70:
        return "High-productivity but high-fragility environment; this may suggest keeping growth exposure disciplined while adding hedges."
    if macro > 80:
        return "Macro fragility is high; historically this is consistent with tighter risk budgets and more attention to liquidity."
    if productivity > 70 and market_setup is not None and market_setup > 20:
        return "Productivity upside is strong relative to macro fragility; this may favor pro-growth exposure with valuation discipline."
    if market_setup is not None and market_setup < -20:
        return "Macro fragility outweighs productivity upside; risk management implication is lower beta and closer stress monitoring."
    return "Mixed or moderate setup; no single force dominates the dashboard."


def _scoreboard(latest_scores: pd.DataFrame) -> str:
    if latest_scores.empty:
        return "No latest score rows were generated."
    rows = ["| Force | Score | 1m Change | 3m Change | Market Pricing | Underpriced Signal | Confidence |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in latest_scores.iterrows():
        if str(row.get("force", "")).startswith(("Macro", "Productivity", "Market", "Net")):
            continue
        rows.append(
            "| {force} | {score} | {change_1m} | {change_3m} | {market} | {underpriced} | {confidence} |".format(
                force=row.get("force", ""),
                score=_fmt(row.get("score")),
                change_1m=_fmt(row.get("change_1m")),
                change_3m=_fmt(row.get("change_3m")),
                market=_fmt(row.get("market_pricing")),
                underpriced=_fmt(row.get("underpriced_signal")),
                confidence=_fmt(row.get("confidence")),
            )
        )
    return "\n".join(rows)


def _force_section(
    title: str,
    metric: str,
    force: str,
    score_map: dict[str, float],
    latest_indicators: pd.DataFrame,
    max_drivers: int,
    implication: str,
) -> str:
    score = score_map.get(metric)
    drivers = _key_drivers(latest_indicators, force, max_drivers)
    return "\n".join(
        [
            f"## {title}",
            f"- Latest score: {_fmt(score)}",
            f"- Key drivers: {drivers}",
            f"- Indicators moving most: {_moving_most(latest_indicators, force, max_drivers)}",
            f"- Investment implications: {implication}",
        ]
    )


def _nature_section(
    score_map: dict[str, float],
    latest_indicators: pd.DataFrame,
    indicator_map: dict[str, dict[str, Any]],
    max_drivers: int,
) -> str:
    structural = score_map.get("nature_pressure_structural_score")
    tactical = score_map.get("nature_shock_tactical_score")
    nature_rows = latest_indicators[latest_indicators["force"] == "nature"] if not latest_indicators.empty else pd.DataFrame()
    staleness = pd.to_numeric(nature_rows["data_staleness_days"], errors="coerce").max() if not nature_rows.empty else np.nan
    confidence = pd.to_numeric(nature_rows["confidence"], errors="coerce").mean() if not nature_rows.empty else np.nan
    return "\n".join(
        [
            "## Force 4: Acts of Nature",
            f"- Latest score: {_fmt(score_map.get('nature_shock_score'))}",
            f"- Tactical score: {_fmt(tactical)}",
            f"- Structural pressure score: {_fmt(structural)}",
            f"- Nature data staleness days: {_fmt(staleness)}",
            f"- Nature confidence score: {_fmt(confidence)}",
            f"- Key drivers: {_key_drivers(latest_indicators, 'nature', max_drivers)}",
            "- Investment implications: Nature-risk data is low-confidence in v0.1; use it mainly as a watchlist for commodity, insurance, utility, and news-intensity stress.",
        ]
    )


def _technology_section(score_map: dict[str, float], latest_indicators: pd.DataFrame, max_drivers: int) -> str:
    productivity = score_map.get("tech_productivity_score")
    fragility = score_map.get("tech_fragility_score")
    net = score_map.get("net_tech_setup")
    implication = _tech_implication(score_map)
    return "\n".join(
        [
            "## Force 5: Technology",
            f"- Tech productivity score: {_fmt(productivity)}",
            f"- Tech fragility score: {_fmt(fragility)}",
            f"- Net tech setup: {_fmt(net)}",
            f"- Key productivity drivers: {_key_drivers(latest_indicators, 'technology', max_drivers, component='productivity')}",
            f"- Key fragility drivers: {_key_drivers(latest_indicators, 'technology', max_drivers, component='fragility')}",
            f"- Investment implications: {implication}",
        ]
    )


def _cross_asset_implications(score_map: dict[str, float]) -> list[str]:
    macro = score_map.get("macro_fragility_score")
    geo_under = score_map.get("geopolitical_conflict_score__underpriced_signal")
    tech_prod = score_map.get("tech_productivity_score")
    tech_frag = score_map.get("tech_fragility_score")
    debt = score_map.get("debt_money_score")
    nature = score_map.get("nature_shock_score")
    return [
        f"- Equities: {_equity_text(macro, tech_prod, tech_frag)}",
        f"- Rates: {_rates_text(debt)}",
        f"- Credit: {_credit_text(debt, macro)}",
        f"- Commodities: {_commodity_text(geo_under, nature)}",
        f"- FX: Elevated dollar/liquidity pressure may suggest watching USD-sensitive assets closely when debt/money scores are high.",
        f"- QQQ / Nasdaq: {_tech_implication(score_map)}",
        f"- Gold: {_gold_text(geo_under, macro)}",
        f"- Energy: {_energy_text(geo_under)}",
        f"- Defense: {_defense_text(geo_under)}",
        "- Cash / short-duration bonds: Higher macro fragility may make optionality and liquidity more valuable.",
    ]


def _watchlist(latest_indicators: pd.DataFrame) -> list[str]:
    if latest_indicators.empty:
        return ["- No indicator watchlist available."]
    rows = []
    extremes = latest_indicators[pd.to_numeric(latest_indicators["score_0_100"], errors="coerce") >= 80]
    movers = latest_indicators[pd.to_numeric(latest_indicators["change_1m"], errors="coerce").abs() >= 15]
    stale = latest_indicators[pd.to_numeric(latest_indicators["data_staleness_days"], errors="coerce") > 30]
    rows.append(f"- Indicators near extreme levels: {_indicator_names(extremes)}")
    rows.append(f"- Indicators with sharp recent deterioration or improvement: {_indicator_names(movers)}")
    rows.append(f"- Missing or stale data: {_indicator_names(stale)}")
    return rows


def _data_quality_notes(latest_scores: pd.DataFrame, latest_indicators: pd.DataFrame, warnings: list[str]) -> list[str]:
    notes: list[str] = []
    missing = latest_indicators[latest_indicators["score_0_100"].isna()] if not latest_indicators.empty else pd.DataFrame()
    low_conf = latest_scores[pd.to_numeric(latest_scores["confidence"], errors="coerce") < 50] if not latest_scores.empty else pd.DataFrame()
    notes.append(f"- Missing sources: {_indicator_names(missing)}")
    notes.append(f"- Low-confidence scores: {_score_names(low_conf)}")
    if warnings:
        notes.append("- Connector warnings:")
        notes.extend([f"  - {warning}" for warning in warnings[:20]])
    else:
        notes.append("- Connector warnings: None captured.")
    notes.append("- No-investment-advice note: This report is a research and monitoring artifact, not personalized financial advice.")
    return notes


def _key_drivers(latest_indicators: pd.DataFrame, force: str, max_drivers: int, component: str | None = None) -> str:
    if latest_indicators.empty:
        return "none available"
    subset = latest_indicators[latest_indicators["force"] == force].copy()
    if component is not None:
        subset = subset[subset["component"] == component]
    subset["driver_rank"] = (pd.to_numeric(subset["score_0_100"], errors="coerce") - 50.0).abs()
    subset = subset.sort_values("driver_rank", ascending=False).head(max_drivers)
    return _indicator_names(subset)


def _moving_most(latest_indicators: pd.DataFrame, force: str, max_drivers: int) -> str:
    if latest_indicators.empty:
        return "none available"
    subset = latest_indicators[latest_indicators["force"] == force].copy()
    subset["move_rank"] = pd.to_numeric(subset["change_1m"], errors="coerce").abs()
    subset = subset.sort_values("move_rank", ascending=False).head(max_drivers)
    return _indicator_names(subset)


def _indicator_names(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "none"
    values = []
    for _, row in frame.iterrows():
        score = row.get("score_0_100")
        suffix = f" ({_fmt(score)})" if pd.notna(score) else ""
        values.append(f"{row.get('indicator_id')}{suffix}")
    return ", ".join(values)


def _score_names(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "none"
    return ", ".join(str(value) for value in frame["force"].head(10))


def _biggest_changes(latest_scores: pd.DataFrame) -> str:
    if latest_scores.empty:
        return "none available"
    frame = latest_scores.copy()
    frame["rank"] = pd.to_numeric(frame["change_1m"], errors="coerce").abs()
    frame = frame.dropna(subset=["rank"]).sort_values("rank", ascending=False).head(3)
    if frame.empty:
        return "insufficient history"
    return ", ".join(f"{row['force']} ({_fmt(row['change_1m'])})" for _, row in frame.iterrows())


def _biggest_underpriced(latest_scores: pd.DataFrame) -> str:
    if latest_scores.empty:
        return "none available"
    frame = latest_scores.dropna(subset=["underpriced_signal"]).sort_values("underpriced_signal", ascending=False).head(3)
    if frame.empty:
        return "none available"
    return ", ".join(f"{row['force']} ({_fmt(row['underpriced_signal'])})" for _, row in frame.iterrows())


def _debt_implication(score_map: dict[str, float]) -> str:
    debt = score_map.get("debt_money_score")
    if not _is_missing(debt) and debt > 80:
        return "Debt/money pressure is high; long-duration equities and credit-sensitive assets may be more vulnerable."
    return "Debt/money conditions are not extreme in the current score; monitor rates, inflation, and credit spreads for deterioration."


def _internal_implication(score_map: dict[str, float]) -> str:
    under = score_map.get("internal_disorder_score__underpriced_signal")
    if not _is_missing(under) and under > 50:
        return "Internal disorder risk appears high relative to market pricing; this may suggest closer hedging and liquidity monitoring."
    return "Internal disorder does not screen as sharply underpriced, though news-based proxies should be treated with caution."


def _geo_implication(score_map: dict[str, float]) -> str:
    under = score_map.get("geopolitical_conflict_score__underpriced_signal")
    if not _is_missing(under) and under > 70:
        return "Geopolitical risk appears underpriced relative to market hedges; consider monitoring energy, gold, defense, and portfolio beta."
    return "Geopolitical risk is not clearly underpriced by this rule set, but real-world and market-pricing layers should be monitored separately."


def _tech_implication(score_map: dict[str, float]) -> str:
    productivity = score_map.get("tech_productivity_score")
    fragility = score_map.get("tech_fragility_score")
    if not _is_missing(productivity) and not _is_missing(fragility) and productivity > 80 and fragility > 80:
        return "AI/technology impulse is strong, but crowding and valuation fragility are elevated."
    if not _is_missing(productivity) and productivity > 70:
        return "Technology productivity impulse is strong; risk management implication is to pair exposure with crowding and drawdown checks."
    if not _is_missing(fragility) and fragility > 70:
        return "Technology fragility is elevated; momentum and drawdown risks may deserve extra attention."
    return "Technology signals are mixed or moderate."


def _equity_text(macro: float | None, tech_prod: float | None, tech_frag: float | None) -> str:
    if not _is_missing(macro) and macro > 80:
        return "High macro fragility may suggest lower beta, better balance-sheet quality, and stricter drawdown controls."
    if not _is_missing(tech_prod) and tech_prod > 70 and (_is_missing(tech_frag) or tech_frag < 70):
        return "Strong tech productivity with moderate fragility may be supportive, while still requiring valuation discipline."
    return "Mixed equity setup; force dispersion matters more than the composite alone."


def _rates_text(debt: float | None) -> str:
    if not _is_missing(debt) and debt > 80:
        return "High debt/money pressure may be historically consistent with more rate volatility and duration sensitivity."
    return "Rates implications are moderate unless debt/money pressure accelerates."


def _credit_text(debt: float | None, macro: float | None) -> str:
    if (not _is_missing(debt) and debt > 75) or (not _is_missing(macro) and macro > 75):
        return "Credit risk may deserve tighter spread and liquidity monitoring."
    return "Credit stress does not screen as extreme in the current composite."


def _commodity_text(geo_under: float | None, nature: float | None) -> str:
    if (not _is_missing(geo_under) and geo_under > 70) or (not _is_missing(nature) and nature > 70):
        return "Commodity hedges may become more relevant when geopolitical or nature-shock signals rise."
    return "Commodity implications are not dominant in the current dashboard."


def _gold_text(geo_under: float | None, macro: float | None) -> str:
    if (not _is_missing(geo_under) and geo_under > 70) or (not _is_missing(macro) and macro > 80):
        return "Gold may function as a hedge candidate when conflict or macro-fragility risk looks underpriced."
    return "Gold signal is watchlist-level rather than dominant."


def _energy_text(geo_under: float | None) -> str:
    if not _is_missing(geo_under) and geo_under > 70:
        return "Energy may be a transmission hedge when conflict risk appears underpriced."
    return "Energy implication depends on oil and gas momentum confirmation."


def _defense_text(geo_under: float | None) -> str:
    if not _is_missing(geo_under) and geo_under > 70:
        return "Defense relative strength may be worth monitoring as a pricing confirmation."
    return "Defense is a confirmation proxy rather than a standalone signal here."


def _fmt(value: Any) -> str:
    if _is_missing(value):
        return "n/a"
    return f"{float(value):.1f}"


def _is_missing(value: Any) -> bool:
    return value is None or pd.isna(value)
