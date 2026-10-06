"""The nine detectors.

Every one is a pure function of (DataFrame, Settings). Same file in, same
issues out, no LLM involved. That is what makes the before/after score
meaningful and the test-suite able to check correctness against fixtures.
"""

from __future__ import annotations

import re

import pandas as pd

from dq_agent.config import Settings
from dq_agent.detectors.profile import (
    as_text,
    blank_mask,
    is_date_like,
    is_numeric_like,
    is_text_column,
    non_blank,
    placeholder_mask,
)
from dq_agent.domain import ALLOWED_STRATEGIES, FixStrategy, Issue, IssueType, Severity

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"^[+()\-.\s\d]{7,20}$")


def _issue(
    issue_type: IssueType,
    column: str | None,
    severity: Severity,
    title: str,
    detail: str,
    affected_rows: int,
    total_rows: int,
    default_strategy: FixStrategy,
    evidence: list[str] | None = None,
    metrics: dict | None = None,
) -> Issue:
    slug = (column or "rows").replace(" ", "_").lower()
    return Issue(
        id=f"{issue_type.value}:{slug}",
        issue_type=issue_type,
        column=column,
        severity=severity,
        title=title,
        detail=detail,
        affected_rows=affected_rows,
        affected_pct=round(100 * affected_rows / total_rows, 2) if total_rows else 0.0,
        evidence=(evidence or [])[:5],
        allowed_strategies=list(ALLOWED_STRATEGIES[issue_type]),
        default_strategy=default_strategy,
        metrics=metrics or {},
    )


def looks_numeric(series: pd.Series, threshold: float = 0.8) -> bool:
    """Whether a column holds numbers, judged from its values rather than its
    dtype. Files are deliberately loaded as text so the mixed-type detector can
    see the evidence, which makes `is_numeric_dtype` useless here."""
    if pd.api.types.is_numeric_dtype(series):
        return True
    present = non_blank(series)
    if present.empty:
        return False
    return bool(as_text(present).map(is_numeric_like).mean() >= threshold)


def detect_missing_values(df: pd.DataFrame, s: Settings) -> list[Issue]:
    issues = []
    for col in df.columns:
        mask = blank_mask(df[col])
        count = int(mask.sum())
        if count == 0:
            continue
        pct = 100 * count / len(df)
        if pct < s.missing_warn_pct:
            continue
        numeric = looks_numeric(df[col])
        written_out = int(placeholder_mask(df[col]).sum())
        placeholder_note = (
            f" {written_out} of them {'is' if written_out == 1 else 'are'} written out"
            " as text such as 'N/A' rather than left empty."
            if written_out
            else ""
        )
        if pct >= s.missing_critical_pct:
            severity = Severity.CRITICAL
        elif pct >= 20:
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM
        issues.append(
            _issue(
                IssueType.MISSING_VALUES,
                col,
                severity,
                f"{count} blank values in '{col}'",
                f"{pct:.1f}% of this column is empty."
                + placeholder_note
                + " "
                + (
                    "It holds numbers, so filling with the median keeps the distribution steady."
                    if numeric
                    else "It holds text, so filling with the most common value or a placeholder is usual."
                ),
                count,
                len(df),
                FixStrategy.FILL_MEDIAN if numeric else FixStrategy.FILL_MODE,
                metrics={"null_pct": round(pct, 2), "numeric": numeric},
            )
        )
    return issues


def detect_duplicate_rows(df: pd.DataFrame, s: Settings) -> list[Issue]:
    dupes = df.duplicated(keep="first")
    count = int(dupes.sum())
    if count == 0:
        return []
    pct = 100 * count / len(df)
    evidence = [
        " | ".join(f"{c}={str(v)[:20]}" for c, v in row.items()) for _, row in df[dupes].head(3).iterrows()
    ]
    return [
        _issue(
            IssueType.DUPLICATE_ROWS,
            None,
            Severity.HIGH if pct >= 5 else Severity.MEDIUM,
            f"{count} duplicate rows",
            f"{pct:.1f}% of rows are exact copies of an earlier row. "
            "Duplicates inflate every count and average computed downstream.",
            count,
            len(df),
            FixStrategy.DROP_DUPLICATES,
            evidence=evidence,
        )
    ]


def detect_mixed_types(df: pd.DataFrame, s: Settings) -> list[Issue]:
    issues = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        present = non_blank(df[col])
        if len(present) < 5:
            continue
        as_text = present.astype(str)
        numeric_mask = as_text.map(is_numeric_like)
        numeric_pct = 100 * numeric_mask.mean()
        # A column that is mostly numbers but stored as text, with a few
        # stragglers, is the classic "total" column with "N/A" in it.
        if not (s.mixed_type_tolerance_pct < 100 - numeric_pct <= 35 and numeric_pct >= 60):
            continue
        offenders = as_text[~numeric_mask]
        count = int(len(offenders))
        issues.append(
            _issue(
                IssueType.MIXED_TYPES,
                col,
                Severity.HIGH,
                f"'{col}' mixes numbers and text",
                f"{numeric_pct:.0f}% of the values are numbers but {count} are not, "
                "so the whole column is stored as text and cannot be summed or averaged.",
                count,
                len(df),
                FixStrategy.COERCE_NUMERIC,
                evidence=sorted(offenders.unique())[:5],
                metrics={"numeric_pct": round(numeric_pct, 2)},
            )
        )
    return issues


def _date_shape(value: str) -> str:
    """Collapse a date to its layout so different values in the same format
    compare equal: runs of digits become #, runs of letters become A. Without
    this, '5 January 2026' and '29 February 2026' would look like two formats."""
    return re.sub(r"[A-Za-z]+", "A", re.sub(r"\d+", "#", value.strip()))


def detect_inconsistent_dates(df: pd.DataFrame, s: Settings) -> list[Issue]:
    issues = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        present = non_blank(df[col])
        if len(present) < 5:
            continue
        as_text = present.astype(str)
        date_mask = as_text.map(is_date_like)
        if date_mask.mean() < 0.6:
            continue
        # A run of digits collapses to one "#", so 5 January and 29 January are the
        # same shape while 05/01/2026 and 2026-01-05 stay different.
        shapes = as_text[date_mask].map(_date_shape)
        distinct = shapes.value_counts()
        if len(distinct) < 2:
            continue
        minority = int(distinct.iloc[1:].sum())
        issues.append(
            _issue(
                IssueType.INCONSISTENT_DATES,
                col,
                Severity.HIGH,
                f"'{col}' uses {len(distinct)} different date formats",
                "Mixed date formats sort wrongly and break any downstream date filter. "
                f"Most common shape is {distinct.index[0]} ({int(distinct.iloc[0])} rows).",
                minority,
                len(df),
                FixStrategy.NORMALIZE_DATES,
                evidence=[str(v) for v in as_text[date_mask].head(5)],
                metrics={"formats": int(len(distinct))},
            )
        )
    return issues


def detect_outliers(df: pd.DataFrame, s: Settings) -> list[Issue]:
    issues = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < s.outlier_min_rows:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        low, high = q1 - s.outlier_iqr_multiplier * iqr, q3 + s.outlier_iqr_multiplier * iqr
        mask = (series < low) | (series > high)
        count = int(mask.sum())
        if count == 0:
            continue
        issues.append(
            _issue(
                IssueType.OUTLIERS,
                col,
                Severity.MEDIUM,
                f"{count} extreme values in '{col}'",
                f"Values outside {low:,.2f} to {high:,.2f} sit far beyond the normal spread. "
                "They may be genuine, or a typo such as an extra zero.",
                count,
                len(df),
                FixStrategy.NONE,  # outliers are often real: never touch them without a decision
                evidence=[f"{v:,.2f}" for v in series[mask].head(5)],
                metrics={"lower_bound": round(float(low), 4), "upper_bound": round(float(high), 4)},
            )
        )
    return issues


def detect_whitespace(df: pd.DataFrame, s: Settings) -> list[Issue]:
    issues = []
    for col in df.columns:
        if not is_text_column(df[col]):
            continue
        present = non_blank(df[col]).astype(str)
        if present.empty:
            continue
        mask = present != present.str.strip()
        count = int(mask.sum())
        if count == 0:
            continue
        issues.append(
            _issue(
                IssueType.WHITESPACE,
                col,
                Severity.LOW,
                f"{count} values in '{col}' have stray spaces",
                "Leading or trailing spaces make two identical-looking values compare as different, "
                "which silently splits groups and joins.",
                count,
                len(df),
                FixStrategy.TRIM_WHITESPACE,
                evidence=[repr(v) for v in present[mask].head(5)],
            )
        )
    return issues


def detect_inconsistent_casing(df: pd.DataFrame, s: Settings) -> list[Issue]:
    issues = []
    for col in df.columns:
        if not is_text_column(df[col]):
            continue
        present = non_blank(df[col]).astype(str).str.strip()
        if len(present) < 5 or present.nunique() > len(present) * s.high_cardinality_ratio:
            continue
        grouped = present.groupby(present.str.lower()).nunique()
        clashing = grouped[grouped > 1]
        if clashing.empty:
            continue
        affected = int(present[present.str.lower().isin(clashing.index)].shape[0])
        examples = []
        for key in clashing.index[:3]:
            variants = sorted(present[present.str.lower() == key].unique())
            examples.append(" / ".join(variants[:4]))
        issues.append(
            _issue(
                IssueType.INCONSISTENT_CASING,
                col,
                Severity.MEDIUM,
                f"'{col}' has {len(clashing)} values written in different cases",
                "The same value spelled with different capitalisation is counted as several "
                "different categories in any group-by.",
                affected,
                len(df),
                FixStrategy.STANDARDIZE_CASE_TITLE,
                evidence=examples,
                metrics={"clashing_groups": int(len(clashing))},
            )
        )
    return issues


def detect_invalid_format(df: pd.DataFrame, s: Settings) -> list[Issue]:
    """Only for columns whose name says what they should contain, so the rule
    can be stated plainly rather than guessed."""
    issues = []
    for col in df.columns:
        if not is_text_column(df[col]):
            continue
        lowered = str(col).lower()
        if "email" in lowered or "e-mail" in lowered:
            pattern, label = EMAIL_RE, "email address"
        elif "phone" in lowered or "mobile" in lowered or "contact number" in lowered:
            pattern, label = PHONE_RE, "phone number"
        else:
            continue
        present = non_blank(df[col]).astype(str).str.strip()
        if present.empty:
            continue
        mask = ~present.map(lambda v, _p=pattern: bool(_p.match(v)))
        count = int(mask.sum())
        if count == 0:
            continue
        issues.append(
            _issue(
                IssueType.INVALID_FORMAT,
                col,
                Severity.HIGH,
                f"{count} values in '{col}' are not a valid {label}",
                f"This column is named like a {label} but some values do not look like one. "
                "They will fail validation in any system you send them to.",
                count,
                len(df),
                FixStrategy.BLANK_INVALID,
                evidence=[str(v)[:40] for v in present[mask].head(5)],
                metrics={"expected": label},
            )
        )
    return issues


def detect_constant_columns(df: pd.DataFrame, s: Settings) -> list[Issue]:
    issues = []
    for col in df.columns:
        present = non_blank(df[col])
        if len(present) == 0 or present.nunique() != 1:
            continue
        issues.append(
            _issue(
                IssueType.CONSTANT_COLUMN,
                col,
                Severity.LOW,
                f"'{col}' has the same value in every row",
                f"Every row reads '{str(present.iloc[0])[:40]}'. The column carries no information "
                "and only adds noise and file size.",
                len(df),
                len(df),
                FixStrategy.NONE,
                evidence=[str(present.iloc[0])[:60]],
            )
        )
    return issues


DETECTORS = (
    detect_missing_values,
    detect_duplicate_rows,
    detect_mixed_types,
    detect_inconsistent_dates,
    detect_outliers,
    detect_whitespace,
    detect_inconsistent_casing,
    detect_invalid_format,
    detect_constant_columns,
)


def detect_all(df: pd.DataFrame, settings: Settings) -> list[Issue]:
    """Run every detector. Ordered worst-first so the UI shows what matters."""
    issues: list[Issue] = []
    for detector in DETECTORS:
        issues.extend(detector(df, settings))
    issues.sort(key=lambda i: (-i.severity.weight, -i.affected_pct, i.id))
    return issues
