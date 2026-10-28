"""Contracts every module codes against. Change here = change everywhere.

Sections: paths and constants; experiment conventions; Pydantic schemas for
the StrategySpec (what the paper says), ExperimentSpec (what we test), the
adaptation log, the progress file, and results.json; run-folder file names.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "snapshot_2026-10-28"
UNIVERSE_FILE = ROOT / "data" / "universe" / "constituents.csv"
OIL_PRICES = ROOT / "data" / "oil_prices.csv"
RUNS_DIR = ROOT / "runs"
CACHE_DIR = ROOT / "cache" / "llm"
PAPERS_DIR = ROOT / "papers"

MODEL = "claude-sonnet-5-5"
PROMPT_VERSION = "v1"
SCHEMA_VERSION = "1"

WARMUP_START = "2012-01-01"
EVAL_START = "2015-01-01"
EVAL_END = "2025-12-31"
SUBPERIOD_SPLIT = "2020-01-01"

COST_BPS_BASE = 10
COST_BPS_GRID = (0, 10, 25)
MIN_NAMES = 20                 # single sorts
MIN_NAMES_DOUBLE_SORT = 15     # double sorts
MIN_FORMATION_DAYS = 15        # trading days with data in a formation month
MIN_HISTORY_DAYS = 252         # eligibility: at least one year of daily data
N_GROUPS_DEFAULT = 10
TOP_PERCENT_DEFAULT = 0.30

BENCHMARKS = ("spy", "ew_universe")
ECON_BENCHMARK = "ew_universe"
# Data-quality exclusions applied on top of the snapshot (reason in BUILD_LOG.md / data_notes.md)
EXCLUDE_TICKERS = ("PARA",)
# Single-day adjusted-close corrections: on these dates Yahoo mis-applied a corporate action; the
# adjusted return is replaced by the split-adjusted close return (data_notes.md has the evidence).
ADJ_CLOSE_PATCHES = {("DHR", "2016-07-05"): "Fortive spin-off applied twice by Yahoo (+61.2% vs +3.6% close return)"}  # Yahoo adjusted close broken (82% drop Dec 2023, +645% jump Jun 2025)  # incremental fees are always vs the EW universe fund

FUND = dict(starting_aum=100_000_000.0, management_fee_rate=0.02,
            performance_fee_rate=0.20, hurdle_rate=0.05, hurdle_type="hard",
            high_water_mark=True, crystallization_frequency="annual",
            horizon_years=5, discount_rate=0.10)
RESEARCH_COST = dict(dev_review_hours=80, maintenance_hours_per_year=10,
                     hourly_rate=150.0)  # model usage is added from measured tokens
MONTE_CARLO = dict(n_paths=1000, mean_block_days=126, block_sensitivity_days=(63, 126, 252),
                   seed=20261028)

RECOMMENDATIONS = ("pursue", "investigate further", "deprioritize", "unable to evaluate")

# ----------------------------------------------------------------------------
# Run folder file names (all relative to runs/<run_id>/)
# ----------------------------------------------------------------------------
class RunFiles:
    paper_meta = "paper_meta.json"          # sha256, title, authors, year, n_pages, selected_experiment, source path
    paper_text = "paper_text.json"          # [{"page": int, "text": str}]
    spec_raw = "spec_raw_{k}.json"          # k = 1..3, StrategySpec as returned by the model
    check_pass = "check_pass_{k}.json"      # CheckPassResult per extraction
    spec_consensus = "spec_consensus.json"  # StrategySpec chosen among the k runs (majority per field) + agreement
    experiment_spec = "experiment_spec.json"
    adaptation_log = "adaptation_log.json"  # list[AdaptationEntry]
    plan = "plan.json"                      # compiled list of registry ops with params
    eligibility = "eligibility.parquet"     # date x ticker bool at month-ends
    signals = "signals.parquet"             # long: date, ticker, signal, group, selected, weight, cohort
    holdings = "holdings.parquet"           # long: date, ticker, shares, price, value, weight
    trades = "trades.parquet"               # long: date, ticker, shares, price, value, cost, reason
    returns = "returns.parquet"             # wide daily: strategy_gross, strategy_net, spy, ew_universe (+ group_1..group_N net)
    portfolio_value = "portfolio_value.parquet"  # daily: gross_value, net_value, cash, n_holdings, turnover (on rebalance days)
    run_log = "run.log"
    validation = "validation.json"      # evidence check, consistency issues, check-pass results
    data_check = "data_check.json"
    scorecard_strategy = "scorecard_strategy.json"
    scorecard_interpretation = "scorecard_interpretation.json"
    montecarlo = "montecarlo.parquet"       # one row per path: ending wealth, max dd, beats_spy, beats_ew, fees..., for strategy and benchmark
    economics = "economics.json"
    results = "results.json"                # everything the dossier needs; numbers the narrative may cite
    strategy_summary = "strategy_summary.json"  # extraction-only runs: {summary, data_needs, implementation_note, source}
    narrative = "narrative.json"            # {"summary": str, "sections": {...}, "verified": bool, "unverified_numbers": [...]}
    dossier = "dossier.html"
    progress = "progress.json"              # Progress model, rewritten by the pipeline after every stage change
    usage = "llm_usage.json"                # list of {call, model, input_tokens, output_tokens, cached: bool}


# ----------------------------------------------------------------------------
# StrategySpec: what the paper says. Filled by the model. No adaptations here.
# ----------------------------------------------------------------------------
class Evidence(BaseModel):
    page: int = Field(description="1-based page number where the quote appears")
    quote: str = Field(description="verbatim quote from that page supporting the value")


class EvidencedStr(BaseModel):
    value: str
    evidence: Evidence


class EvidencedList(BaseModel):
    value: list[str]
    evidence: Evidence


SortRule = Literal["deciles", "quintiles", "terciles", "top_bottom_percent", "continuous", "other"]
Weighting = Literal["equal", "value", "rank", "other"]
ExpectedSign = Literal["favored_group_higher", "favored_group_lower", "unclear"]


class StrategySpec(BaseModel):
    """The paper's strategy, as stated. Material fields carry evidence."""
    paper_title: str
    authors: list[str]
    year: int
    selected_experiment: str = Field(description="which table/experiment of the paper is being described")

    can_implement: bool = Field(description="false if the strategy needs accounting data, futures or non-equity instruments, or several asset classes")
    cannot_implement_reasons: list[str]

    economic_rationale: EvidencedStr = Field(description="the paper's stated reason the effect exists")
    universe_and_filters: EvidencedStr
    required_inputs: EvidencedList = Field(description="data needed, e.g. daily prices, market cap, fundamentals")
    signal_formula_words: EvidencedStr
    signal_formula_symbols: EvidencedStr
    formation_window: EvidencedStr = Field(description="e.g. '1 month of daily data', '6 months'")
    skip_window: EvidencedStr = Field(description="gap between formation and holding, e.g. 'none', '1 month'")
    holding_period: EvidencedStr
    rebalance_frequency: EvidencedStr
    sort_rule: SortRule
    sort_rule_evidence: Evidence
    breakpoints: EvidencedStr = Field(description="how groups are cut, e.g. equal-count deciles, NYSE breakpoints")
    favored_group: EvidencedStr = Field(description="the ranked group the paper says earns the premium, e.g. 'lowest MAX decile'")
    disfavored_group: EvidencedStr
    weighting: Weighting
    weighting_evidence: Evidence
    expected_sign: ExpectedSign
    expected_sign_evidence: Evidence
    headline_result: EvidencedStr

    unresolved_details: list[str] = Field(description="details the paper leaves open; do not guess them")


MATERIAL_FIELDS = [
    "economic_rationale", "universe_and_filters", "required_inputs", "signal_formula_words",
    "signal_formula_symbols", "formation_window", "skip_window", "holding_period",
    "rebalance_frequency", "sort_rule", "breakpoints", "favored_group", "disfavored_group",
    "weighting", "expected_sign", "headline_result",
]


class CheckPassItem(BaseModel):
    field: str
    issue: str
    contradicting_quote: str
    page: int


class CheckPassResult(BaseModel):
    agrees: bool
    items: list[CheckPassItem]


# ----------------------------------------------------------------------------
# ExperimentSpec: what we actually test. Produced by validation/adapt.py.
# ----------------------------------------------------------------------------
FieldStatus = Literal["stated", "assumed", "adapted", "unresolved"]


class AdaptationEntry(BaseModel):
    field: str
    paper_value: Optional[str]
    our_value: str
    status: FieldStatus
    reason: str


class SignalOp(BaseModel):
    op: str                       # registry operation name
    params: dict = Field(default_factory=dict)


class ExperimentSpec(BaseModel):
    run_id: str
    paper_sha256: str
    strategy_slug: str            # e.g. "max", "momentum"
    status: Literal["ready", "blocked"]
    blocked_reasons: list[str] = Field(default_factory=list)

    # construction
    signal_ops: list[SignalOp]    # ordered time-series ops producing the signal panel
    selection: dict               # {"sort": "deciles"|"quintiles"|"terciles"|"top_bottom_percent", "n_groups": int, "percent": float|None, "favored": "lowest"|"highest", "double_sort": {...}|None}
    weighting: Literal["equal"] = "equal"
    holding_months: int = 1
    skip_months: int = 0
    min_names: int = MIN_NAMES
    price_floor: Optional[float] = None
    min_history_days: int = MIN_HISTORY_DAYS
    min_formation_days: int = MIN_FORMATION_DAYS
    expected_sign: ExpectedSign = "favored_group_higher"

    # experiment
    universe: str = "415 names from snapshot; eligible from S&P 500 add date; survivor universe"
    snapshot: str = SNAPSHOT_DIR.name
    warmup_start: str = WARMUP_START
    eval_start: str = EVAL_START
    eval_end: str = EVAL_END
    execution: str = "signal through prior month's last close; execute at next month's first trading-day close; new weights earn only after execution"
    cost_bps: int = COST_BPS_BASE
    benchmarks: tuple[str, str] = BENCHMARKS
    fund: dict = Field(default_factory=lambda: dict(FUND))
    research_cost: dict = Field(default_factory=lambda: dict(RESEARCH_COST))
    monte_carlo: dict = Field(default_factory=lambda: dict(MONTE_CARLO))

    adaptations: list[AdaptationEntry] = Field(default_factory=list)
    n_assumed: int = 0
    n_adapted: int = 0


# ----------------------------------------------------------------------------
# Progress file: the app polls this. Rewritten after every stage change.
# ----------------------------------------------------------------------------
STAGES = [
    (1, "Read paper"), (2, "Extract strategy"), (3, "Validate and adapt"),
    (4, "Check data"), (5, "Construct portfolios"), (6, "Backtest"),
    (7, "Evaluate"), (8, "Monte Carlo"), (9, "Dossier"),
]
StageStatus = Literal["pending", "running", "done", "blocked", "failed", "skipped"]   # skipped: extraction-only run


class StageState(BaseModel):
    n: int
    name: str
    status: StageStatus = "pending"
    detail: str = ""              # one-line fact when done, progress line while running, reason when blocked/failed
    started: Optional[str] = None
    finished: Optional[str] = None


class Progress(BaseModel):
    run_id: str
    paper_title: str = ""
    strategy_slug: str = ""
    stages: list[StageState]
    recommendation: Optional[str] = None
    summary: str = ""             # the narrative summary paragraph, filled at stage 9
    error: Optional[str] = None

    @classmethod
    def new(cls, run_id: str) -> "Progress":
        return cls(run_id=run_id, stages=[StageState(n=n, name=name) for n, name in STAGES])


def write_progress(run_dir: Path, p: Progress) -> None:
    (run_dir / RunFiles.progress).write_text(p.model_dump_json(indent=1))


def read_progress(run_dir: Path) -> Progress:
    return Progress.model_validate_json((run_dir / RunFiles.progress).read_text())
