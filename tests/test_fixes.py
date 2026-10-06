"""The executor: does each fix do exactly what it claims, and nothing else?"""

from __future__ import annotations

import pandas as pd
import pytest

from dq_agent.config import Settings
from dq_agent.detectors.rules import detect_all
from dq_agent.detectors.scoring import score_dataset
from dq_agent.domain import ALLOWED_STRATEGIES, Decision, FixStrategy, IssueType
from dq_agent.fixes.executor import FixRejected, apply_all, apply_fix
from dq_agent.sources import load_dataframe


def issue_of(df, kind: IssueType, column: str | None = None):
    issues = detect_all(df, Settings())
    for i in issues:
        if i.issue_type == kind and (column is None or i.column == column):
            return i
    raise AssertionError(f"no {kind.value} issue for column {column}: {[i.id for i in issues]}")


def decide(issue, strategy: FixStrategy, fill_value=None) -> Decision:
    return Decision(issue_id=issue.id, approved=True, strategy=strategy, fill_value=fill_value)


def test_trim_whitespace_only_touches_padded_values(messy_df):
    issue = issue_of(messy_df, IssueType.WHITESPACE, "City")
    out, fix = apply_fix(messy_df, issue, decide(issue, FixStrategy.TRIM_WHITESPACE))
    assert fix.applied and fix.rows_changed == 1
    assert list(out["City"]) == [c.strip() for c in messy_df["City"]]
    assert len(out) == len(messy_df), "trimming must not remove rows"


def test_standardize_case_merges_the_variants(messy_df):
    issue = issue_of(messy_df, IssueType.INCONSISTENT_CASING, "City")
    out, _ = apply_fix(messy_df, issue, decide(issue, FixStrategy.STANDARDIZE_CASE_TITLE))
    assert set(out["City"]) == {"Mumbai", "Delhi", "Pune"}


def test_drop_duplicates_keeps_the_first_copy(messy_df):
    issue = issue_of(messy_df, IssueType.DUPLICATE_ROWS)
    out, fix = apply_fix(messy_df, issue, decide(issue, FixStrategy.DROP_DUPLICATES))
    assert len(out) == len(messy_df) - 1 and fix.rows_removed == 1
    assert list(out["OrderID"]).count("10") == 1


def test_normalize_dates_rewrites_only_the_odd_ones(messy_df):
    issue = issue_of(messy_df, IssueType.INCONSISTENT_DATES, "OrderDate")
    out, fix = apply_fix(messy_df, issue, decide(issue, FixStrategy.NORMALIZE_DATES))
    assert fix.applied
    assert all(len(v) == 10 and v[4] == "-" for v in out["OrderDate"])
    # "06/01/2026" is ambiguous on its own. The column also contains no value
    # with a first component above 12, so it is read month-first, as pandas does.
    assert out["OrderDate"].iloc[1] == "2026-06-01"


def test_day_first_dates_are_detected_from_the_data():
    """A column containing 29/01 can only be day/month, so 06/01 in the same
    column must be 6 January, not 1 June."""
    csv = b"id,when\n1,29/01/2026\n2,06/01/2026\n3,2026-01-07\n4,2026-01-08\n5,2026-01-09\n6,2026-01-10\n"
    df = load_dataframe("d.csv", csv)
    issue = issue_of(df, IssueType.INCONSISTENT_DATES, "when")
    out, fix = apply_fix(df, issue, decide(issue, FixStrategy.NORMALIZE_DATES))
    assert out["when"].iloc[0] == "2026-01-29"
    assert out["when"].iloc[1] == "2026-01-06"
    assert "day/month/year" in fix.detail


def test_coerce_numeric_blanks_only_the_unconvertible(messy_df):
    issue = issue_of(messy_df, IssueType.MIXED_TYPES, "Amount")
    out, fix = apply_fix(messy_df, issue, decide(issue, FixStrategy.COERCE_NUMERIC))
    assert pd.api.types.is_numeric_dtype(out["Amount"])
    assert fix.rows_changed == 1  # the single "pending"
    assert out["Amount"].iloc[0] == 100.50


def test_blank_invalid_keeps_the_rest_of_the_row(messy_df):
    issue = issue_of(messy_df, IssueType.INVALID_FORMAT, "CustomerEmail")
    out, fix = apply_fix(messy_df, issue, decide(issue, FixStrategy.BLANK_INVALID))
    assert fix.rows_changed == 1
    assert len(out) == len(messy_df)
    assert pd.isna(out["CustomerEmail"].iloc[2])
    assert out["OrderID"].iloc[2] == "3", "the rest of the row must survive"


def test_fill_median_converts_the_column_and_fills():
    df = load_dataframe("x.csv", b"id,amount\n1,10\n2,20\n3,\n4,30\n5,\n6,40\n")
    issue = issue_of(df, IssueType.MISSING_VALUES, "amount")
    out, fix = apply_fix(df, issue, decide(issue, FixStrategy.FILL_MEDIAN))
    assert pd.api.types.is_numeric_dtype(out["amount"])
    assert out["amount"].isna().sum() == 0
    assert out["amount"].iloc[2] == 25.0
    assert "median" in fix.detail


def test_fill_constant_needs_a_value(messy_df):
    issue = issue_of(messy_df, IssueType.MISSING_VALUES, "Notes")
    with pytest.raises(FixRejected, match="needs a value"):
        apply_fix(messy_df, issue, decide(issue, FixStrategy.FILL_CONSTANT))
    out, fix = apply_fix(messy_df, issue, decide(issue, FixStrategy.FILL_CONSTANT, "Unknown"))
    assert (out["Notes"] == "Unknown").sum() == 9


def test_an_average_is_refused_on_a_text_column():
    df = load_dataframe("t.csv", b"id,city\n1,Mumbai\n2,\n3,Delhi\n4,\n5,Pune\n6,Agra\n")
    issue = issue_of(df, IssueType.MISSING_VALUES, "city")
    with pytest.raises(FixRejected, match="holds text"):
        apply_fix(df, issue, decide(issue, FixStrategy.FILL_MEAN))


def test_a_strategy_outside_the_menu_is_rejected(messy_df):
    """The last line of defence: even if a bad choice reached the executor,
    it is re-checked against the issue's own allowed list."""
    issue = issue_of(messy_df, IssueType.DUPLICATE_ROWS)
    with pytest.raises(FixRejected, match="not a valid fix"):
        apply_fix(messy_df, issue, decide(issue, FixStrategy.NORMALIZE_DATES))


def test_none_changes_nothing_but_is_recorded(messy_df):
    issue = issue_of(messy_df, IssueType.DUPLICATE_ROWS)
    out, fix = apply_fix(messy_df, issue, decide(issue, FixStrategy.NONE))
    assert out is messy_df and not fix.applied and "by choice" in fix.detail


def test_every_issue_type_has_a_working_menu(messy_df):
    """Every strategy the UI can offer must actually run or refuse cleanly,
    never raise something unexpected."""
    for issue in detect_all(messy_df, Settings()):
        for strategy in ALLOWED_STRATEGIES[issue.issue_type]:
            fill = "X" if strategy == FixStrategy.FILL_CONSTANT else None
            try:
                out, fix = apply_fix(messy_df, issue, decide(issue, strategy, fill))
            except FixRejected:
                continue  # a clean, explained refusal is a valid outcome
            assert isinstance(out, pd.DataFrame)
            assert fix.strategy == strategy


def test_fixes_run_in_a_safe_order(messy_df):
    """Deduplicating before trimming would miss rows that differ only by a
    trailing space, so the executor sorts the approved fixes itself."""
    issues = detect_all(messy_df, Settings())
    by_type = {i.issue_type: i for i in issues}
    decisions = [
        decide(by_type[IssueType.DUPLICATE_ROWS], FixStrategy.DROP_DUPLICATES),
        decide(by_type[IssueType.WHITESPACE], FixStrategy.TRIM_WHITESPACE),
        decide(by_type[IssueType.INCONSISTENT_CASING], FixStrategy.STANDARDIZE_CASE_TITLE),
    ]
    _, applied = apply_all(messy_df, issues, decisions)
    order = [f.strategy for f in applied]
    assert order.index(FixStrategy.TRIM_WHITESPACE) < order.index(FixStrategy.DROP_DUPLICATES)


def test_rejected_decisions_are_not_applied(messy_df):
    issues = detect_all(messy_df, Settings())
    issue = next(i for i in issues if i.issue_type == IssueType.DUPLICATE_ROWS)
    decisions = [Decision(issue_id=issue.id, approved=False, strategy=FixStrategy.DROP_DUPLICATES)]
    out, applied = apply_all(messy_df, issues, decisions)
    assert len(out) == len(messy_df) and applied == []


def _as_drop_column_issue(issue):
    return issue.model_copy(
        update={
            "issue_type": IssueType.CONSTANT_COLUMN,
            "id": "constant_column:city",
            "allowed_strategies": list(ALLOWED_STRATEGIES[IssueType.CONSTANT_COLUMN]),
        }
    )


def test_dropping_a_column_runs_after_fixes_to_that_column(messy_df):
    """Approving both "tidy this column" and "delete this column" must not
    crash. The ordering runs the tidy first, so neither fix errors."""
    issues = detect_all(messy_df, Settings())
    city_ws = next(i for i in issues if i.issue_type == IssueType.WHITESPACE)
    drop = _as_drop_column_issue(city_ws)
    decisions = [
        Decision(issue_id=drop.id, approved=True, strategy=FixStrategy.DROP_COLUMN),
        decide(city_ws, FixStrategy.TRIM_WHITESPACE),
    ]
    out, applied = apply_all(messy_df, issues + [drop], decisions)
    assert "City" not in out.columns
    assert [f.strategy for f in applied] == [FixStrategy.TRIM_WHITESPACE, FixStrategy.DROP_COLUMN]
    assert all(f.applied for f in applied)


def test_a_fix_on_a_missing_column_is_skipped_not_crashed(messy_df):
    """The defensive guard itself: a fix whose column is gone reports why it
    was skipped instead of raising a KeyError."""
    issues = detect_all(messy_df, Settings())
    city_ws = next(i for i in issues if i.issue_type == IssueType.WHITESPACE)
    without_city = messy_df.drop(columns=["City"])
    out, applied = apply_all(without_city, issues, [decide(city_ws, FixStrategy.TRIM_WHITESPACE)])
    assert len(applied) == 1 and not applied[0].applied
    assert "no longer in the data" in applied[0].detail


def test_applying_fixes_raises_the_quality_score(messy_df):
    """The end-to-end promise: the score is re-measured, not assumed."""
    issues = detect_all(messy_df, Settings())
    before = score_dataset(issues).score
    decisions = [decide(i, i.default_strategy) for i in issues if i.default_strategy != FixStrategy.NONE]
    cleaned, applied = apply_all(messy_df, issues, decisions)
    after = score_dataset(detect_all(cleaned, Settings())).score
    assert after > before, f"{before} -> {after}"
    assert any(f.applied for f in applied)
