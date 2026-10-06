"""Column and dataset profiling. Pure pandas, no LLM, no I/O."""

from __future__ import annotations

import re

import pandas as pd

from dq_agent.domain import ColumnProfile, DatasetProfile

# Dates people actually paste into spreadsheets.
_DATE_PATTERNS = (
    r"^\d{4}-\d{1,2}-\d{1,2}$",
    r"^\d{1,2}/\d{1,2}/\d{4}$",
    r"^\d{1,2}-\d{1,2}-\d{4}$",
    r"^\d{4}/\d{1,2}/\d{1,2}$",
    r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$",
    r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$",
    r"^\d{4}-\d{1,2}-\d{1,2}[ T]\d{1,2}:\d{2}",
)
_DATE_RE = re.compile("|".join(_DATE_PATTERNS))
_NUMERIC_RE = re.compile(r"^-?[\d,]*\.?\d+%?$")


def is_date_like(value: str) -> bool:
    return bool(_DATE_RE.match(value.strip()))


def is_numeric_like(value: str) -> bool:
    v = value.strip().replace("$", "").replace("€", "").replace("£", "")
    return bool(v) and bool(_NUMERIC_RE.match(v))


# Spellings of "no value here" that people actually type into spreadsheets.
# These count as blanks, but they are still reported separately so a user can
# see that a column uses placeholder text rather than real empty cells.
PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {"", "na", "n/a", "n.a.", "n/a.", "nan", "null", "none", "nil", "-", "--", "?", "#n/a"}
)


def is_placeholder(value: str) -> bool:
    return value.strip().lower() in PLACEHOLDER_TOKENS


def is_text_column(series: pd.Series) -> bool:
    """True for a column holding text.

    pandas 2 gives such a column dtype `object`; pandas 3 gives it `str`.
    Comparing against `object` alone silently skipped three detectors on
    pandas 3, so every text check goes through this helper instead.
    """
    return not (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    )


def as_text(series: pd.Series) -> pd.Series:
    """The column as plain Python strings with missing values as "".

    `series.astype(str)` is not enough: on pandas 3 a missing value stays NA
    rather than becoming the string "nan", so any `.map()` over the result is
    handed a float and raises. Every text operation goes through this.
    """
    return series.astype(object).where(series.notna(), "").astype(str)


def non_blank(series: pd.Series) -> pd.Series:
    """Values that are genuinely present. An empty string and '  ' are blanks,
    which `pandas.isna` alone does not catch."""
    s = series.dropna()
    if is_text_column(s):
        s = s[~as_text(s).map(is_placeholder)]
    return s


def blank_mask(series: pd.Series) -> pd.Series:
    """True where the value is missing, empty or whitespace-only."""
    mask = series.isna()
    if is_text_column(series):
        mask = mask | as_text(series).map(is_placeholder)
    return mask


def placeholder_mask(series: pd.Series) -> pd.Series:
    """Blank-but-written-out values, such as the literal text "N/A"."""
    if not is_text_column(series):
        return pd.Series(False, index=series.index)
    text = as_text(series)
    return text.map(is_placeholder) & (text.str.strip() != "")


def profile_column(series: pd.Series) -> ColumnProfile:
    present = non_blank(series)
    total = len(series)
    null_count = total - len(present)
    as_text = present.astype(str)

    numeric_like = float(as_text.map(is_numeric_like).mean() * 100) if len(present) else 0.0
    date_like = float(as_text.map(is_date_like).mean() * 100) if len(present) else 0.0

    min_value = max_value = None
    mean_value = None
    if pd.api.types.is_numeric_dtype(series) and len(present):
        min_value, max_value = str(present.min()), str(present.max())
        mean_value = round(float(present.mean()), 4)
    elif len(present):
        ordered = sorted(as_text.unique())
        min_value, max_value = ordered[0][:40], ordered[-1][:40]

    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        non_null=len(present),
        null_count=int(null_count),
        null_pct=round(100 * null_count / total, 2) if total else 0.0,
        unique_count=int(present.nunique()),
        sample_values=[v[:60] for v in as_text.head(5).tolist()],
        numeric_like_pct=round(numeric_like, 2),
        date_like_pct=round(date_like, 2),
        min_value=min_value,
        max_value=max_value,
        mean_value=mean_value,
    )


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    return DatasetProfile(
        rows=len(df),
        columns=len(df.columns),
        column_profiles=[profile_column(df[c]) for c in df.columns],
        memory_kb=round(df.memory_usage(deep=True).sum() / 1024, 1),
    )
