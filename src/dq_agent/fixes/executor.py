"""The only code that ever modifies your data.

One small pandas function per strategy. No LLM output is executed, formatted
into code, or passed to `eval`. The LLM can only choose which of these named
functions runs, and an illegal choice is rejected before it gets here.
"""

from __future__ import annotations

import pandas as pd

from dq_agent.detectors.profile import as_text, blank_mask, is_text_column, non_blank
from dq_agent.detectors.rules import EMAIL_RE, PHONE_RE
from dq_agent.domain import (
    ALLOWED_STRATEGIES,
    AppliedFix,
    Decision,
    FixStrategy,
    Issue,
    IssueType,
)


class FixRejected(ValueError):
    """The requested fix is not legal for this issue. Never applied."""


def _coerce_series_numeric(series: pd.Series) -> pd.Series:
    cleaned = as_text(series).str.replace(r"[$€£,\s]", "", regex=True).str.rstrip("%")
    return pd.to_numeric(cleaned, errors="coerce")


def looks_day_first(text: pd.Series) -> bool:
    """Whether "06/01/2026" in this column means 6 January or 1 June.

    Guessing wrong silently corrupts every ambiguous date, so the answer comes
    from evidence: if any value in the column has a first number above 12, the
    column must be day-first. With no such value the column is ambiguous and
    the month-first reading is used, which is what pandas assumes.
    """
    parts = text.str.extract(r"^(\d{1,2})[/-](\d{1,2})[/-]\d{2,4}$")
    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")
    if first.notna().sum() == 0:
        return False
    # A first component above 12 can only be a day.
    return bool((first > 12).any() and not (second > 12).any())


def _fill_masked(series: pd.Series, mask: pd.Series, value) -> pd.Series:
    """Write `value` where `mask` is True, widening the column if it has to.

    pandas 3 refuses to put a float into a `str` column, so a column that
    cannot hold the value is converted to a plain object column rather than
    letting the fix fail.
    """
    try:
        out = series.copy()
        out[mask] = value
        return out
    except (TypeError, ValueError):
        return series.astype(object).mask(mask, value)


def validate(issue: Issue, decision: Decision) -> None:
    """Re-derive what is allowed from the issue itself rather than trusting
    whatever arrived with the request."""
    if decision.strategy not in ALLOWED_STRATEGIES[issue.issue_type]:
        raise FixRejected(f"'{decision.strategy.value}' is not a valid fix for {issue.issue_type.value}.")
    if decision.strategy == FixStrategy.FILL_CONSTANT and decision.fill_value is None:
        raise FixRejected("Filling with a constant needs a value to fill in.")
    averages = {FixStrategy.FILL_MEAN, FixStrategy.FILL_MEDIAN}
    if issue.column is not None and decision.strategy in averages and not issue.metrics.get("numeric", True):
        raise FixRejected(
            f"'{issue.column}' holds text, so an average or median cannot be computed. "
            "Use the most common value or a placeholder instead."
        )


def apply_fix(df: pd.DataFrame, issue: Issue, decision: Decision) -> tuple[pd.DataFrame, AppliedFix]:
    """Return a NEW DataFrame plus a record of exactly what changed."""
    validate(issue, decision)
    strategy = decision.strategy
    col = issue.column
    rows_before, cols_before = len(df), len(df.columns)
    out = df.copy()
    changed = 0
    detail = ""

    if strategy == FixStrategy.NONE:
        return df, AppliedFix(
            issue_id=issue.id,
            issue_type=issue.issue_type,
            column=col,
            strategy=strategy,
            applied=False,
            detail="Left unchanged by choice.",
        )

    if strategy == FixStrategy.DROP_DUPLICATES:
        out = out.drop_duplicates(keep="first").reset_index(drop=True)
        detail = f"Removed {rows_before - len(out)} duplicate rows, kept the first copy of each."

    elif strategy == FixStrategy.DROP_ROWS:
        if col is None:
            raise FixRejected("Dropping rows needs a column to test.")
        if issue.issue_type == IssueType.INVALID_FORMAT:
            keep = _valid_format_mask(out[col], issue) | blank_mask(out[col])
            out = out[keep].reset_index(drop=True)
        else:
            out = out[~blank_mask(out[col])].reset_index(drop=True)
        detail = f"Removed {rows_before - len(out)} rows because of '{col}'."

    elif strategy in {
        FixStrategy.FILL_MEAN,
        FixStrategy.FILL_MEDIAN,
        FixStrategy.FILL_MODE,
        FixStrategy.FILL_CONSTANT,
        FixStrategy.FORWARD_FILL,
    }:
        if col is None:
            raise FixRejected("Filling needs a column.")
        mask = blank_mask(out[col])
        changed = int(mask.sum())
        if strategy == FixStrategy.FORWARD_FILL:
            out[col] = out[col].where(~mask).ffill()
            remaining = int(blank_mask(out[col]).sum())
            changed -= remaining
            detail = f"Carried the previous value into {changed} blanks in '{col}'."
            if remaining:
                detail += f" {remaining} blanks at the top had nothing above them to copy."
        elif strategy in {FixStrategy.FILL_MEAN, FixStrategy.FILL_MEDIAN}:
            # An average only exists for numbers, so the column becomes numeric
            # as part of the fix. Leaving it as text and writing "10.0" into it
            # would look filled while still being unusable in a sum.
            numeric = _coerce_series_numeric(out[col]) if is_text_column(out[col]) else out[col]
            present = non_blank(numeric)
            if present.empty:
                raise FixRejected(f"'{col}' has no numeric values, so an average cannot be computed.")
            value = round(float(present.mean() if strategy == FixStrategy.FILL_MEAN else present.median()), 4)
            out[col] = numeric.fillna(value)
            detail = (
                f"Filled {changed} blanks in '{col}' with {value} "
                f"({'average' if strategy == FixStrategy.FILL_MEAN else 'median'}) "
                "and converted the column to numbers."
            )
        else:
            if strategy == FixStrategy.FILL_MODE:
                modes = non_blank(out[col]).mode()
                if modes.empty:
                    raise FixRejected(f"'{col}' is entirely empty, so there is no common value to use.")
                value = modes.iloc[0]
            else:
                value = decision.fill_value
            out[col] = _fill_masked(out[col], mask, value)
            detail = f"Filled {changed} blanks in '{col}' with {value!r}."

    elif strategy == FixStrategy.COERCE_NUMERIC:
        if col is None:
            raise FixRejected("Converting to numbers needs a column.")
        converted = _coerce_series_numeric(out[col])
        changed = int((converted.isna() & ~blank_mask(out[col])).sum())
        out[col] = converted
        detail = f"Converted '{col}' to numbers; {changed} unconvertible values became blank."

    elif strategy == FixStrategy.NORMALIZE_DATES:
        if col is None:
            raise FixRejected("Normalising dates needs a column.")
        original = as_text(out[col])
        day_first = looks_day_first(original)
        parsed = pd.to_datetime(out[col], errors="coerce", format="mixed", dayfirst=day_first)
        formatted = parsed.dt.strftime("%Y-%m-%d")
        changed = int((formatted.notna() & (formatted != original)).sum())
        unparsed = int(parsed.isna().sum() - blank_mask(out[col]).sum())
        out[col] = formatted.where(parsed.notna(), out[col])
        order = "day/month/year" if day_first else "month/day/year"
        detail = f"Rewrote {changed} dates in '{col}' as YYYY-MM-DD, reading slashed dates as {order}."
        if unparsed > 0:
            detail += f" {unparsed} values could not be read as a date and were left alone."

    elif strategy == FixStrategy.TRIM_WHITESPACE:
        if col is None:
            raise FixRejected("Trimming needs a column.")
        before = as_text(out[col])
        trimmed = before.str.strip()
        changed = int((before != trimmed).sum())
        out[col] = out[col].where(blank_mask(out[col]), trimmed)
        detail = f"Trimmed spaces from {changed} values in '{col}'."

    elif strategy in {
        FixStrategy.STANDARDIZE_CASE_TITLE,
        FixStrategy.STANDARDIZE_CASE_UPPER,
        FixStrategy.STANDARDIZE_CASE_LOWER,
    }:
        if col is None:
            raise FixRejected("Standardising case needs a column.")
        before = as_text(out[col])
        if strategy == FixStrategy.STANDARDIZE_CASE_TITLE:
            after = before.str.strip().str.title()
        elif strategy == FixStrategy.STANDARDIZE_CASE_UPPER:
            after = before.str.strip().str.upper()
        else:
            after = before.str.strip().str.lower()
        changed = int((before != after).sum())
        out[col] = out[col].where(blank_mask(out[col]), after)
        detail = f"Rewrote {changed} values in '{col}' in a consistent case."

    elif strategy in {FixStrategy.CLIP_OUTLIERS, FixStrategy.REMOVE_OUTLIER_ROWS}:
        if col is None:
            raise FixRejected("Handling outliers needs a column.")
        low = issue.metrics.get("lower_bound")
        high = issue.metrics.get("upper_bound")
        if low is None or high is None:
            raise FixRejected("This issue has no recorded normal range, so outliers cannot be handled.")
        numeric = pd.to_numeric(out[col], errors="coerce")
        outside = (numeric < low) | (numeric > high)
        if strategy == FixStrategy.CLIP_OUTLIERS:
            changed = int(outside.sum())
            out[col] = numeric.clip(lower=low, upper=high).where(numeric.notna(), out[col])
            detail = f"Pulled {changed} extreme values in '{col}' back to {low:,.2f}-{high:,.2f}."
        else:
            out = out[~outside.fillna(False)].reset_index(drop=True)
            detail = f"Removed {rows_before - len(out)} rows with extreme values in '{col}'."

    elif strategy == FixStrategy.BLANK_INVALID:
        if col is None:
            raise FixRejected("Blanking invalid values needs a column.")
        if issue.issue_type == IssueType.INVALID_FORMAT:
            bad = ~_valid_format_mask(out[col], issue) & ~blank_mask(out[col])
        else:
            bad = _coerce_series_numeric(out[col]).isna() & ~blank_mask(out[col])
        changed = int(bad.sum())
        out.loc[bad, col] = None
        detail = f"Blanked {changed} invalid values in '{col}' and kept the rest of each row."

    elif strategy == FixStrategy.DROP_COLUMN:
        if col is None:
            raise FixRejected("Dropping a column needs a column.")
        out = out.drop(columns=[col])
        detail = f"Removed the column '{col}'."

    else:  # pragma: no cover - the enum is exhaustive above
        raise FixRejected(f"No handler for strategy '{strategy.value}'.")

    return out, AppliedFix(
        issue_id=issue.id,
        issue_type=issue.issue_type,
        column=col,
        strategy=strategy,
        applied=True,
        rows_changed=changed,
        rows_removed=max(0, rows_before - len(out)),
        columns_removed=max(0, cols_before - len(out.columns)),
        detail=detail,
    )


def _valid_format_mask(series: pd.Series, issue: Issue) -> pd.Series:
    pattern = EMAIL_RE if issue.metrics.get("expected") == "email address" else PHONE_RE
    return as_text(series).str.strip().map(lambda v: bool(pattern.match(v)))


def apply_all(
    df: pd.DataFrame, issues: list[Issue], decisions: list[Decision]
) -> tuple[pd.DataFrame, list[AppliedFix]]:
    """Apply approved decisions in a safe order and collect the audit trail.

    Order matters: tidy values first (trim, case, types, dates), then fill
    blanks, then remove rows or columns. Deduplicating before trimming would
    miss duplicates that differ only by a trailing space.
    """
    order = {
        FixStrategy.TRIM_WHITESPACE: 0,
        FixStrategy.STANDARDIZE_CASE_TITLE: 1,
        FixStrategy.STANDARDIZE_CASE_UPPER: 1,
        FixStrategy.STANDARDIZE_CASE_LOWER: 1,
        FixStrategy.COERCE_NUMERIC: 2,
        FixStrategy.NORMALIZE_DATES: 2,
        FixStrategy.BLANK_INVALID: 3,
        FixStrategy.FILL_MEAN: 4,
        FixStrategy.FILL_MEDIAN: 4,
        FixStrategy.FILL_MODE: 4,
        FixStrategy.FILL_CONSTANT: 4,
        FixStrategy.FORWARD_FILL: 4,
        FixStrategy.CLIP_OUTLIERS: 5,
        FixStrategy.DROP_DUPLICATES: 6,
        FixStrategy.REMOVE_OUTLIER_ROWS: 7,
        FixStrategy.DROP_ROWS: 8,
        FixStrategy.DROP_COLUMN: 9,
        FixStrategy.NONE: 10,
    }
    by_id = {i.id: i for i in issues}
    approved = [d for d in decisions if d.approved and d.issue_id in by_id]
    approved.sort(key=lambda d: order.get(d.strategy, 5))

    out = df
    applied: list[AppliedFix] = []
    dropped_columns: set[str] = set()
    for decision in approved:
        issue = by_id[decision.issue_id]
        if issue.column and (issue.column in dropped_columns or issue.column not in out.columns):
            applied.append(
                AppliedFix(
                    issue_id=issue.id,
                    issue_type=issue.issue_type,
                    column=issue.column,
                    strategy=decision.strategy,
                    applied=False,
                    detail=f"Skipped: the column '{issue.column}' is no longer in the data.",
                )
            )
            continue
        try:
            out, record = apply_fix(out, issue, decision)
        except FixRejected as exc:
            record = AppliedFix(
                issue_id=issue.id,
                issue_type=issue.issue_type,
                column=issue.column,
                strategy=decision.strategy,
                applied=False,
                detail=f"Not applied: {exc}",
            )
        if record.applied and decision.strategy == FixStrategy.DROP_COLUMN and issue.column:
            dropped_columns.add(issue.column)
        applied.append(record)
    return out, applied
