from __future__ import annotations

ASSET_MAP = {
    "SPY": "US equities",
    "QQQ": "Nasdaq / growth equities",
    "TLT": "Long-duration Treasuries",
    "GLD": "Gold",
    "XLE": "Energy equities",
    "ITA": "Defense equities",
    "XAR": "Defense equities alternate",
    "HYG": "High yield credit",
    "UUP": "US dollar ETF proxy",
    "^VIX": "Equity volatility index",
}


DEFAULT_EVENT_DEFINITIONS = {
    "macro_fragility_gt_80": ("macro_fragility_score", ">", 80),
    "debt_money_gt_80": ("debt_money_score", ">", 80),
    "internal_disorder_gt_80": ("internal_disorder_score", ">", 80),
    "geopolitical_conflict_gt_80": ("geopolitical_conflict_score", ">", 80),
    "nature_shock_gt_80": ("nature_shock_score", ">", 80),
    "tech_productivity_gt_80": ("tech_productivity_score", ">", 80),
    "tech_fragility_gt_80": ("tech_fragility_score", ">", 80),
    "geopolitical_underpriced_gt_70": ("geopolitical_underpriced_signal", ">", 70),
    "net_tech_setup_gt_50": ("net_tech_setup", ">", 50),
}
