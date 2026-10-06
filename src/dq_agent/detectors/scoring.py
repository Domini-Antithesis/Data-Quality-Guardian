"""The data-quality score: one number from 0 to 100 plus four dimensions.

Deliberately simple and fully explainable, because a score nobody can explain
is a score nobody trusts. Each issue costs points according to its severity
and how much of the dataset it touches; the dimension breakdown says which
kind of problem is dragging the number down.
"""

from __future__ import annotations

from dq_agent.domain import Issue, IssueType, QualityScore

# Which quality dimension each detector speaks to.
DIMENSION_OF: dict[IssueType, str] = {
    IssueType.MISSING_VALUES: "completeness",
    IssueType.DUPLICATE_ROWS: "uniqueness",
    IssueType.MIXED_TYPES: "validity",
    IssueType.INCONSISTENT_DATES: "consistency",
    IssueType.OUTLIERS: "validity",
    IssueType.WHITESPACE: "consistency",
    IssueType.INCONSISTENT_CASING: "consistency",
    IssueType.INVALID_FORMAT: "validity",
    IssueType.CONSTANT_COLUMN: "completeness",
}

DIMENSIONS = ("completeness", "uniqueness", "validity", "consistency")


def score_dataset(issues: list[Issue]) -> QualityScore:
    penalty = sum(i.penalty for i in issues)
    per_dimension = {d: 0.0 for d in DIMENSIONS}
    for issue in issues:
        per_dimension[DIMENSION_OF[issue.issue_type]] += issue.penalty

    def as_score(points: float) -> float:
        return round(max(0.0, 100.0 - points * 2.5), 1)

    return QualityScore(
        score=round(max(0.0, 100.0 - penalty), 1),
        completeness=as_score(per_dimension["completeness"]),
        uniqueness=as_score(per_dimension["uniqueness"]),
        validity=as_score(per_dimension["validity"]),
        consistency=as_score(per_dimension["consistency"]),
        issue_count=len(issues),
        penalty=round(penalty, 2),
    )
