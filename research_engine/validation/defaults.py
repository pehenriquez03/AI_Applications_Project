"""Defaults for details a paper leaves open.

Each default is applied with status "assumed" in the adaptation log and the
run continues. The implementation-reliability flag turns amber when more than
AMBER_MAX_ASSUMED material fields are assumed.
"""
from __future__ import annotations

from research_engine.contracts import (
    MIN_FORMATION_DAYS, MIN_HISTORY_DAYS, N_GROUPS_DEFAULT, TOP_PERCENT_DEFAULT,
)

AMBER_MAX_ASSUMED = 3

DEFAULTS: dict[str, dict] = {
    "price_floor":         {"value": None, "text": "no price filter unless stated"},
    "min_history_days":    {"value": MIN_HISTORY_DAYS,
                            "text": "one year of history to be eligible (or the signal window if longer)"},
    "min_formation_days":  {"value": MIN_FORMATION_DAYS, "text": "15 trading days minimum in a formation month"},
    "skip_months":         {"value": 0, "text": "no skip"},
    "holding_months":      {"value": 1, "text": "one-month hold"},
    "rebalance":           {"value": "monthly", "text": "monthly rebalance"},
    "sort":                {"value": "deciles", "text": "deciles with equal-count breakpoints"},
    "n_groups":            {"value": N_GROUPS_DEFAULT, "text": "ten groups"},
    "breakpoints":         {"value": "equal-count", "text": "equal-count breakpoints over eligible names"},
    "top_percent":         {"value": TOP_PERCENT_DEFAULT, "text": "30% for top/bottom-percent sorts"},
    "favored":             {"value": None,
                            "text": "favored group from the paper's headline table, else all groups run and "
                                    "the rubric marks 'investigate further'"},
    "weighting":           {"value": "equal", "text": "equal weight"},
    "expected_sign":       {"value": "favored_group_higher", "text": "favored group expected to earn more"},
    "vol_returns":         {"value": "weekly_friday_simple", "text": "simple weekly returns from Friday closes for volatility"},
    "high_price":          {"value": "raw_split_adjusted_high", "text": "raw split-adjusted daily high for the 52-week high"},
    "momentum_formation_end": {"value": "prior_month_last_close",
                               "text": "momentum formation ends at the prior month's last close"},
    "frog_window":         {"value": {"months": 11, "skip": 1}, "text": "frog-in-the-pan window t-12 to t-2"},
    "ineligible_mid_hold": {"value": "sell_at_last_price", "text": "ineligible mid-hold sold at last price"},
    "cash_yield":          {"value": 0.0, "text": "cash earns zero"},
}

# Per-strategy fallbacks when the paper's favored side or windows cannot be parsed.
STRATEGY_FALLBACKS: dict[str, dict] = {
    "max":         {"favored": "lowest",  "formation_months": 1,  "holding_months": 1},
    "momentum":    {"favored": "highest", "formation_months": 6,  "holding_months": 6},
    "seasonality": {"favored": "highest", "formation_months": 1,  "holding_months": 1},
    "lowvol":      {"favored": "lowest",  "formation_months": 1,  "holding_months": 1},
    "high52":      {"favored": "highest", "formation_months": 12, "holding_months": 6},
    "frog":        {"favored": "highest", "formation_months": 11, "holding_months": 6},
    "oil":         {"favored": "highest", "formation_months": 36, "holding_months": 1},
}
