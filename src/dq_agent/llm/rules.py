"""The deterministic advisor.

Two jobs, same as its counterpart in the sibling project: it is the default
when no API key is configured, and it is the safety net when a real provider
fails. Its recommendations are the conservative, textbook choice for each
issue type, and every one is marked `source="rule"` so the UI can say so.
"""

from __future__ import annotations

from dq_agent.domain import Advice, FixStrategy, Issue, IssueType
from dq_agent.llm.base import AdviceResult, FixAdvisor


def advise_issue(issue: Issue) -> Advice:
    t, pct = issue.issue_type, issue.affected_pct
    numeric = bool(issue.metrics.get("numeric"))

    if t == IssueType.MISSING_VALUES:
        if pct >= 60:
            strategy = FixStrategy.NONE
            why = (
                f"{pct:.0f}% of this column is empty. Filling that much would invent most of the "
                "data. Consider whether the column is usable at all before changing it."
            )
        elif pct <= 5 and not numeric:
            strategy = FixStrategy.FILL_MODE
            why = f"Only {pct:.1f}% is missing, so the most common value is a safe, low-impact filler."
        elif numeric:
            strategy = FixStrategy.FILL_MEDIAN
            why = (
                "This column holds numbers. The median is robust to extreme values, so filling with "
                "it keeps the distribution closer to the truth than the average would."
            )
        else:
            strategy = FixStrategy.FILL_MODE
            why = "This column holds text, so the most common value is the conventional filler."
    elif t == IssueType.DUPLICATE_ROWS:
        strategy = FixStrategy.DROP_DUPLICATES
        why = "Exact duplicate rows double-count everything downstream. Keeping the first copy is standard."
    elif t == IssueType.MIXED_TYPES:
        strategy = FixStrategy.COERCE_NUMERIC
        why = (
            f"{issue.metrics.get('numeric_pct', 0):.0f}% of values are already numbers. Converting the "
            "column makes it usable in sums and averages; the few non-numeric entries become blank "
            "and show up as a missing-value issue you can then handle explicitly."
        )
    elif t == IssueType.INCONSISTENT_DATES:
        strategy = FixStrategy.NORMALIZE_DATES
        why = "One ISO format (YYYY-MM-DD) sorts correctly and is what every downstream tool expects."
    elif t == IssueType.OUTLIERS:
        strategy = FixStrategy.NONE
        why = (
            "Extreme values are often genuine, not errors. Review the examples first; clip them only "
            "if you are confident they are data-entry mistakes."
        )
    elif t == IssueType.WHITESPACE:
        strategy = FixStrategy.TRIM_WHITESPACE
        why = "Trimming is lossless and removes a common cause of failed joins and split groups."
    elif t == IssueType.INCONSISTENT_CASING:
        strategy = FixStrategy.STANDARDIZE_CASE_TITLE
        why = "Title Case reads well for names and categories and merges the duplicate groups."
    elif t == IssueType.INVALID_FORMAT:
        strategy = FixStrategy.BLANK_INVALID
        why = (
            "Blanking only the invalid values keeps the rest of each row, which is almost always "
            "preferable to deleting whole records over one bad field."
        )
    elif t == IssueType.CONSTANT_COLUMN:
        strategy = FixStrategy.NONE
        why = (
            "A single-value column is harmless. Drop it only if you are sure nothing downstream "
            "expects the column to exist."
        )
    else:  # pragma: no cover
        strategy = FixStrategy.NONE
        why = "No rule for this issue type."

    return Advice(issue_id=issue.id, strategy=strategy, reasoning=why, confidence=0.6, source="rule")


class RuleFixAdvisor(FixAdvisor):
    name = "mock"
    model = "rule-based-v1"

    def advise(self, issues: list[Issue], context: str) -> AdviceResult:
        return AdviceResult(advice=[advise_issue(i) for i in issues], llm_calls=0)
