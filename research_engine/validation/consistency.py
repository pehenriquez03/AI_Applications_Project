"""Python checks on a StrategySpec before adaptation.

check_consistency(spec) -> list of issues, each
{"field", "tag", "severity": "block"|"warn"|"info", "message"}.
Only tag "missing_input" (severity block) stops a run; the rest are reported.
"""
from __future__ import annotations

import re
from typing import Optional

from research_engine.contracts import OIL_PRICES, StrategySpec

_WORDNUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "eighteen": 18, "twenty-four": 24,
    "twenty four": 24, "thirty-six": 36, "thirty six": 36, "forty-eight": 48, "sixty": 60,
    "fifty-two": 52, "fifty two": 52,
}
_UNIT_MONTHS = {"month": 1.0, "mo": 1.0, "week": 12 / 52, "wk": 12 / 52, "year": 12.0, "yr": 12.0,
                "day": 1 / 21, "trading day": 1 / 21}
_NOT_STATED = re.compile(r"^\s*(not stated|not specified|unspecified|unclear|unknown|n/?a|not reported|none stated)\b", re.I)


def is_not_stated(value: Optional[str]) -> bool:
    return value is None or not str(value).strip() or bool(_NOT_STATED.match(str(value)))


def _words_to_digits(s: str) -> str:
    for w in sorted(_WORDNUM, key=len, reverse=True):
        s = re.sub(rf"\b{w}\b", str(_WORDNUM[w]), s)
    return s


def parse_window_lags(text: str) -> Optional[tuple[int, int]]:
    """'t-12 to t-2' -> (12, 2)."""
    m = re.search(r"t\s*[-−]\s*(\d+)\s*(?:to|through|until|and|-|,|–)\s*t\s*[-−]\s*(\d+)", text or "", re.I)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return (max(a, b), min(a, b))


def parse_months(text: Optional[str]) -> Optional[float]:
    """Parse a period phrase to months. None if not parseable or not stated."""
    if is_not_stated(text):
        return None
    s = _words_to_digits(str(text).lower().replace("–", "-"))
    lags = parse_window_lags(s)
    if lags:
        return float(lags[0] - lags[1] + 1)
    if re.match(r"^\s*(none|no\b|zero\b|0\b|immediately|not skipped)", s) or re.search(r"\bno (skip|gap|lag)\b", s):
        return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*[- ]?\s*(trading day|month|mo|week|wk|year|yr|day)s?\b", s)
    if m:
        return float(m.group(1)) * _UNIT_MONTHS[m.group(2)]
    if re.search(r"\b(previous|prior|past|last|preceding|a|each|the|single|one)\s+(calendar\s+)?month\b", s) \
            or re.search(r"\bmonthly\b", s):
        return 1.0
    if re.search(r"\b(a|one|the|previous|prior)\s+week\b|\bweekly\b", s):
        return 12 / 52
    if re.search(r"\b(a|one|the|previous|prior)\s+year\b|\bannual(ly)?\b", s):
        return 12.0
    if re.search(r"\bquarter(ly)?\b", s):
        return 3.0
    if re.search(r"\b(a|one|the)\s+day\b|\bdaily\b", s):
        return 1 / 21
    return None


# ----------------------------------------------------------------------------
# direction words
# ----------------------------------------------------------------------------
_LOW = re.compile(r"\b(low|lowest|lower|bottom|smallest|least|loser|losers|minimum|decile 1|portfolio 1|negative)\b", re.I)
_HIGH = re.compile(r"\b(high|highest|higher|top|largest|greatest|most|winner|winners|decile 10|portfolio 10|positive)\b", re.I)


def group_direction(text: Optional[str]) -> Optional[str]:
    """'lowest MAX decile' -> 'lowest'; ambiguous or empty -> None."""
    if is_not_stated(text):
        return None
    lo, hi = bool(_LOW.search(text)), bool(_HIGH.search(text))
    if lo and not hi:
        return "lowest"
    if hi and not lo:
        return "highest"
    if lo and hi:  # e.g. "lowest-volatility (high-quality) stocks": first mention wins
        return "lowest" if _LOW.search(text).start() < _HIGH.search(text).start() else "highest"
    return None


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------
_ALWAYS_MISSING = [
    (re.compile(r"book|earnings|accounting|fundamental|profit|total assets|\bassets\b|sales|revenue|cash ?flow|"
                r"balance[- ]sheet|compustat|leverage|accrual|gross margin|cost of goods", re.I), "fundamental data"),
    (re.compile(r"futures|forward contract|option|bond|treasury yield|currenc|\bfx\b|foreign exchange|"
                r"commodit|swap|credit default", re.I), "non-equity instruments"),
    (re.compile(r"intraday|tick|high[- ]frequency|order book|quote data|\btaq\b", re.I), "intraday data"),
    (re.compile(r"analyst|forecast|i/b/e/s|institutional ownership|short interest|credit rating|13f", re.I),
     "data not in the snapshot"),
]
_CAP = re.compile(r"market cap|market value|market equity|\bsize\b|shares outstanding|\bme\b|capitali[sz]ation", re.I)
_VOLUME = re.compile(r"volume|turnover|illiquidity|bid[- ]ask|spread|amihud", re.I)
_OIL = re.compile(r"\boil\b|crude|\bwti\b|brent", re.I)
_AVAILABLE = re.compile(r"price|return|close|crsp|market index|market return|market portfolio|s&p|index|"
                        r"factor|fama|french|risk[- ]free|t-?bill|treasury bill|\bhigh\b|exchange|share code|"
                        r"listing|stock", re.I)
_SECONDARY_PURPOSE = re.compile(r"weight|breakpoint|filter|screen|exclu|control|robust|subsample|size group|"
                                r"nyse", re.I)


def classify_input(item: str, spec: Optional[StrategySpec] = None) -> tuple[str, str]:
    """-> (kind, note), kind in available | adaptable | missing | unknown.

    Market cap and volume block only when the signal itself uses them; when they
    serve weighting, breakpoints or filters we adapt (equal weight, equal-count
    breakpoints, no size filter)."""
    signal_text = ""
    if spec is not None:
        signal_text = f"{spec.signal_formula_words.value} {spec.signal_formula_symbols.value}"
    if _OIL.search(item) and not re.search(r"option", item, re.I):
        if OIL_PRICES.exists():
            return "available", "oil price series (data/oil_prices.csv)"
        return "missing", "oil price series not found"
    for rx, label in _ALWAYS_MISSING:
        if rx.search(item):
            return "missing", label
    for rx, label in ((_CAP, "market capitalization"), (_VOLUME, "volume or liquidity data")):
        if rx.search(item):
            if rx.search(signal_text):
                return "missing", f"{label} used by the signal"
            if _SECONDARY_PURPOSE.search(item) or spec is not None:
                return "adaptable", f"{label} used only for weighting, breakpoints or filters"
            return "missing", label
    if _AVAILABLE.search(item):
        return "available", ""
    return "unknown", "input not recognised; assumed derivable from daily prices"


def check_consistency(spec: StrategySpec) -> list[dict]:
    issues: list[dict] = []

    def add(field, tag, severity, message):
        issues.append({"field": field, "tag": tag, "severity": severity, "message": message})

    for field in ("formation_window", "holding_period", "skip_window"):
        v = getattr(spec, field).value
        if is_not_stated(v):
            add(field, "not_stated", "info", f"{field} not stated; default applies")
        elif parse_months(v) is None:
            add(field, "unparseable_period", "warn", f"cannot parse {field} '{v}' to months")
        elif parse_months(v) < 0:
            add(field, "nonpositive_period", "warn", f"{field} '{v}' is negative")
    for field in ("formation_window", "holding_period"):
        m = parse_months(getattr(spec, field).value)
        if m is not None and m == 0:
            add(field, "nonpositive_period", "warn", f"{field} parses to zero months")

    fav, dis = spec.favored_group.value, spec.disfavored_group.value
    if not is_not_stated(fav) and fav.strip().casefold() == (dis or "").strip().casefold():
        add("favored_group", "groups_identical", "warn", "favored and disfavored groups are the same")
    fd, dd = group_direction(fav), group_direction(dis)
    if fd and dd and fd == dd:
        add("favored_group", "groups_same_direction", "warn",
            f"favored ('{fav}') and disfavored ('{dis}') point the same way")
    if fd is None and not is_not_stated(fav):
        add("favored_group", "favored_direction_unclear", "warn", f"cannot tell high/low side from '{fav}'")
    if spec.expected_sign == "favored_group_lower":
        add("expected_sign", "sign_inconsistent", "warn",
            "expected_sign says the favored group earns less; the favored group is defined as the one earning the premium")
    elif spec.expected_sign == "unclear":
        add("expected_sign", "sign_unclear", "info", "expected sign unclear; default favored_group_higher applies")

    for item in spec.required_inputs.value:
        kind, note = classify_input(item, spec)
        if kind == "missing":
            add("required_inputs", "missing_input", "block", f"'{item}': {note}")
        elif kind == "adaptable":
            add("required_inputs", "adaptable_input", "info", f"'{item}': {note}")
        elif kind == "unknown":
            add("required_inputs", "unknown_input", "warn", f"'{item}': {note}")

    if not spec.can_implement:
        add("can_implement", "cannot_implement", "info",
            "extraction says the paper cannot be implemented: " + "; ".join(spec.cannot_implement_reasons))
    return issues
