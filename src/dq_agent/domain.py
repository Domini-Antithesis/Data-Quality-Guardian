"""Typed models shared by every layer.

The governing idea, borrowed from good refactoring tools: **the LLM never
writes code that touches your data.** It chooses one strategy from a fixed
menu and explains why. Every transformation is a tested pandas function.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IssueType(str, Enum):
    MISSING_VALUES = "missing_values"
    DUPLICATE_ROWS = "duplicate_rows"
    MIXED_TYPES = "mixed_types"
    INCONSISTENT_DATES = "inconsistent_dates"
    OUTLIERS = "outliers"
    WHITESPACE = "whitespace"
    INCONSISTENT_CASING = "inconsistent_casing"
    INVALID_FORMAT = "invalid_format"
    CONSTANT_COLUMN = "constant_column"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> float:
        return {"low": 1.0, "medium": 2.5, "high": 5.0, "critical": 10.0}[self.value]


class FixStrategy(str, Enum):
    """The complete menu. An LLM may pick one of these and nothing else."""

    NONE = "none"  # leave it alone, on purpose
    DROP_ROWS = "drop_rows"
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_MODE = "fill_mode"
    FILL_CONSTANT = "fill_constant"
    FORWARD_FILL = "forward_fill"
    DROP_DUPLICATES = "drop_duplicates"
    COERCE_NUMERIC = "coerce_numeric"
    NORMALIZE_DATES = "normalize_dates"
    TRIM_WHITESPACE = "trim_whitespace"
    STANDARDIZE_CASE_TITLE = "standardize_case_title"
    STANDARDIZE_CASE_UPPER = "standardize_case_upper"
    STANDARDIZE_CASE_LOWER = "standardize_case_lower"
    CLIP_OUTLIERS = "clip_outliers"
    REMOVE_OUTLIER_ROWS = "remove_outlier_rows"
    BLANK_INVALID = "blank_invalid"
    DROP_COLUMN = "drop_column"


# Which strategies are even legal for which issue. Checked before an LLM is
# asked and again before anything is applied, so a hallucinated strategy can
# never reach the data.
ALLOWED_STRATEGIES: dict[IssueType, tuple[FixStrategy, ...]] = {
    IssueType.MISSING_VALUES: (
        FixStrategy.NONE,
        FixStrategy.DROP_ROWS,
        FixStrategy.FILL_MEAN,
        FixStrategy.FILL_MEDIAN,
        FixStrategy.FILL_MODE,
        FixStrategy.FILL_CONSTANT,
        FixStrategy.FORWARD_FILL,
    ),
    IssueType.DUPLICATE_ROWS: (FixStrategy.NONE, FixStrategy.DROP_DUPLICATES),
    IssueType.MIXED_TYPES: (FixStrategy.NONE, FixStrategy.COERCE_NUMERIC, FixStrategy.BLANK_INVALID),
    IssueType.INCONSISTENT_DATES: (FixStrategy.NONE, FixStrategy.NORMALIZE_DATES),
    IssueType.OUTLIERS: (FixStrategy.NONE, FixStrategy.CLIP_OUTLIERS, FixStrategy.REMOVE_OUTLIER_ROWS),
    IssueType.WHITESPACE: (FixStrategy.NONE, FixStrategy.TRIM_WHITESPACE),
    IssueType.INCONSISTENT_CASING: (
        FixStrategy.NONE,
        FixStrategy.STANDARDIZE_CASE_TITLE,
        FixStrategy.STANDARDIZE_CASE_UPPER,
        FixStrategy.STANDARDIZE_CASE_LOWER,
    ),
    IssueType.INVALID_FORMAT: (FixStrategy.NONE, FixStrategy.BLANK_INVALID, FixStrategy.DROP_ROWS),
    IssueType.CONSTANT_COLUMN: (FixStrategy.NONE, FixStrategy.DROP_COLUMN),
}

# What each strategy does, in one plain sentence. Shown in the UI and given to
# the LLM so it chooses from described options rather than bare names.
STRATEGY_HELP: dict[FixStrategy, str] = {
    FixStrategy.NONE: "Leave this as it is and record the decision.",
    FixStrategy.DROP_ROWS: "Delete the affected rows entirely.",
    FixStrategy.FILL_MEAN: "Replace blanks with the column's average.",
    FixStrategy.FILL_MEDIAN: (
        "Replace blanks with the column's middle value (safer than the average when there are extremes)."
    ),
    FixStrategy.FILL_MODE: "Replace blanks with the most common value in the column.",
    FixStrategy.FILL_CONSTANT: "Replace blanks with a fixed value you choose, such as 'Unknown' or 0.",
    FixStrategy.FORWARD_FILL: (
        "Carry the previous row's value down into the blank (only sensible for ordered data)."
    ),
    FixStrategy.DROP_DUPLICATES: "Keep the first copy of each duplicated row and delete the rest.",
    FixStrategy.COERCE_NUMERIC: "Convert the column to numbers; anything that is not a number becomes blank.",
    FixStrategy.NORMALIZE_DATES: "Rewrite every date in the same YYYY-MM-DD format.",
    FixStrategy.TRIM_WHITESPACE: "Remove spaces at the start and end of every value.",
    FixStrategy.STANDARDIZE_CASE_TITLE: "Rewrite text values in Title Case.",
    FixStrategy.STANDARDIZE_CASE_UPPER: "Rewrite text values in UPPER CASE.",
    FixStrategy.STANDARDIZE_CASE_LOWER: "Rewrite text values in lower case.",
    FixStrategy.CLIP_OUTLIERS: "Pull extreme values back to the normal range instead of deleting them.",
    FixStrategy.REMOVE_OUTLIER_ROWS: "Delete the rows that contain extreme values.",
    FixStrategy.BLANK_INVALID: "Blank out only the values that fail the check and keep the rest of the row.",
    FixStrategy.DROP_COLUMN: "Remove the whole column from the dataset.",
}

DESTRUCTIVE_STRATEGIES: frozenset[FixStrategy] = frozenset(
    {
        FixStrategy.DROP_ROWS,
        FixStrategy.DROP_DUPLICATES,
        FixStrategy.REMOVE_OUTLIER_ROWS,
        FixStrategy.DROP_COLUMN,
    }
)


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    non_null: int
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: list[str] = Field(default_factory=list)
    numeric_like_pct: float = 0.0
    date_like_pct: float = 0.0
    min_value: str | None = None
    max_value: str | None = None
    mean_value: float | None = None


class DatasetProfile(BaseModel):
    rows: int
    columns: int
    column_profiles: list[ColumnProfile] = Field(default_factory=list)
    memory_kb: float = 0.0


class Issue(BaseModel):
    """One problem found by one detector. Deterministic: same file, same issues."""

    id: str
    issue_type: IssueType
    column: str | None  # None means the issue is about whole rows
    severity: Severity
    title: str
    detail: str
    affected_rows: int
    affected_pct: float
    evidence: list[str] = Field(default_factory=list)
    allowed_strategies: list[FixStrategy] = Field(default_factory=list)
    default_strategy: FixStrategy = FixStrategy.NONE
    metrics: dict[str, Any] = Field(default_factory=dict)

    @property
    def penalty(self) -> float:
        """How much this issue costs the dataset's quality score."""
        return self.severity.weight * min(1.0, self.affected_pct / 100 * 3 + 0.15)


class Advice(BaseModel):
    """The LLM's recommendation for one issue. It chooses and explains; it
    never writes a transformation."""

    issue_id: str
    strategy: FixStrategy
    reasoning: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    fill_value: str | None = None
    source: str = "llm"  # "llm" | "rule" (deterministic fallback)


class Decision(BaseModel):
    """What the human approved. This, and only this, is what gets applied."""

    issue_id: str
    approved: bool
    strategy: FixStrategy
    fill_value: str | None = None


class AppliedFix(BaseModel):
    issue_id: str
    issue_type: IssueType
    column: str | None
    strategy: FixStrategy
    applied: bool
    rows_changed: int = 0
    rows_removed: int = 0
    columns_removed: int = 0
    detail: str = ""


class QualityScore(BaseModel):
    """0 to 100. 100 means no detector found anything."""

    score: float
    completeness: float
    uniqueness: float
    validity: float
    consistency: float
    issue_count: int
    penalty: float

    @property
    def grade(self) -> str:
        if self.score >= 95:
            return "Excellent"
        if self.score >= 85:
            return "Good"
        if self.score >= 70:
            return "Fair"
        if self.score >= 50:
            return "Poor"
        return "Critical"


class RunStatus(str, Enum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    AWAITING_APPROVAL = "awaiting_approval"
    APPLYING = "applying"
    COMPLETED = "completed"
    FAILED = "failed"


class RunRecord(BaseModel):
    id: int
    filename: str
    status: RunStatus
    created_at: str
    updated_at: str | None = None
    rows_before: int = 0
    rows_after: int = 0
    columns_before: int = 0
    columns_after: int = 0
    score_before: float = 0.0
    score_after: float = 0.0
    issues_found: int = 0
    fixes_applied: int = 0
    provider: str = ""
    model: str = ""
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    advice_fallbacks: int = 0
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
