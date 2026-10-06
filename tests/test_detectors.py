"""Detector correctness against a fixture with known, planted flaws."""

from __future__ import annotations

import pandas as pd
import pytest

from dq_agent.config import Settings
from dq_agent.detectors.profile import as_text, is_date_like, is_numeric_like, profile_dataset
from dq_agent.detectors.rules import detect_all, looks_numeric
from dq_agent.detectors.scoring import DIMENSION_OF, score_dataset
from dq_agent.domain import IssueType
from dq_agent.sources import load_dataframe


def types_found(df, settings=None) -> set[IssueType]:
    return {i.issue_type for i in detect_all(df, settings or Settings())}


def test_every_planted_flaw_is_found(messy_df):
    found = types_found(messy_df)
    for expected in (
        IssueType.MISSING_VALUES,
        IssueType.DUPLICATE_ROWS,
        IssueType.MIXED_TYPES,
        IssueType.INCONSISTENT_DATES,
        IssueType.WHITESPACE,
        IssueType.INCONSISTENT_CASING,
        IssueType.INVALID_FORMAT,
        IssueType.CONSTANT_COLUMN,
    ):
        assert expected in found, f"{expected.value} was not detected"


def test_a_clean_dataset_reports_nothing():
    csv = "id,name,amount,day\n"
    csv += "".join(f"{i},Name {i},{100 + i}.00,2026-01-{i:02d}\n" for i in range(1, 13))
    df = load_dataframe("clean.csv", csv.encode())
    issues = detect_all(df, Settings())
    assert issues == [], [i.title for i in issues]
    assert score_dataset(issues).score == 100.0


def test_detectors_are_deterministic(messy_df):
    first = [i.model_dump() for i in detect_all(messy_df, Settings())]
    second = [i.model_dump() for i in detect_all(messy_df, Settings())]
    assert first == second


def test_missing_value_counts_and_severity(messy_df):
    issue = next(
        i
        for i in detect_all(messy_df, Settings())
        if i.issue_type == IssueType.MISSING_VALUES and i.column == "Notes"
    )
    assert issue.affected_rows == 9  # nine blank Notes in the fixture
    assert issue.severity.value == "critical"  # 82% > the 40% critical threshold


def test_duplicate_detection_counts_only_the_extra_copy(messy_df):
    issue = next(i for i in detect_all(messy_df, Settings()) if i.issue_type == IssueType.DUPLICATE_ROWS)
    assert issue.affected_rows == 1
    assert issue.column is None


def test_casing_and_whitespace_are_separate_issues(messy_df):
    issues = {i.issue_type: i for i in detect_all(messy_df, Settings())}
    assert issues[IssueType.WHITESPACE].column == "City"
    assert issues[IssueType.INCONSISTENT_CASING].column == "City"
    assert "Mumbai" in " ".join(issues[IssueType.INCONSISTENT_CASING].evidence)


def test_invalid_email_only_flags_columns_named_like_emails():
    csv = "email,nickname\nnot-an-email,not-an-email\nb@x.com,bob\nc@x.com,carol\nd@x.com,dave\ne@x.com,eve\n"
    df = load_dataframe("e.csv", csv.encode())
    issues = [i for i in detect_all(df, Settings()) if i.issue_type == IssueType.INVALID_FORMAT]
    assert [i.column for i in issues] == ["email"], "a column not named like an email must not be judged"


def test_missing_threshold_is_configurable(messy_df):
    relaxed = Settings(missing_warn_pct=95)
    assert IssueType.MISSING_VALUES not in types_found(messy_df, relaxed)


def test_outlier_sensitivity_is_configurable():
    """A mild outlier is caught by a sensitive setting and ignored by a
    conservative one; that dial is the whole point of the setting."""
    # 1..40 spread gives an inter-quartile range of 19.5, so 200 is far out at
    # a sensitive setting and comfortably inside at a conservative one.
    df = pd.DataFrame({"amount": [str(i) for i in range(1, 41)] + ["200"]})
    assert IssueType.OUTLIERS in types_found(df, Settings(outlier_iqr_multiplier=1.5))
    assert IssueType.OUTLIERS not in types_found(df, Settings(outlier_iqr_multiplier=10))


def test_a_flat_column_produces_no_outliers():
    """With no spread the IQR is zero, so the rule declines to guess rather
    than declaring every row an outlier."""
    df = pd.DataFrame({"amount": ["10"] * 40 + ["10000"]})
    assert IssueType.OUTLIERS not in types_found(df, Settings(outlier_iqr_multiplier=1.5))


def test_placeholder_text_counts_as_missing_and_is_called_out():
    df = load_dataframe("p.csv", b"id,total\n1,1\n2,N/A\n3,3\n4,null\n5,5\n6,6\n7,7\n8,8\n")
    issue = next(i for i in detect_all(df, Settings()) if i.issue_type == IssueType.MISSING_VALUES)
    assert issue.affected_rows == 2, "'N/A' and 'null' are blanks, not values"
    assert "written out" in issue.detail


def test_outliers_need_enough_rows_to_be_meaningful():
    df = pd.DataFrame({"amount": ["1", "2", "3", "900"]})
    assert IssueType.OUTLIERS not in types_found(df, Settings())


def test_date_shape_ignores_the_actual_day_and_month_name():
    csv = "when\n" + "".join(f"{d} January 2026\n" for d in range(1, 9))
    df = load_dataframe("d.csv", csv.encode())
    assert IssueType.INCONSISTENT_DATES not in types_found(df), (
        "one format written 8 ways is still one format"
    )


def test_a_single_column_file_keeps_its_header():
    """pandas' own delimiter sniffer splits the header "total" on the letter
    "t". The loader counts real delimiters instead."""
    df = load_dataframe("one.csv", b"total\n10\n20\n30\n")
    assert list(df.columns) == ["total"]
    assert list(df["total"]) == ["10", "20", "30"]


def test_semicolon_and_tab_files_load():
    assert list(load_dataframe("s.csv", b"a;b\n1;2\n").columns) == ["a", "b"]
    assert list(load_dataframe("t.tsv", b"a\tb\n1\t2\n").columns) == ["a", "b"]


def test_looks_numeric_uses_values_not_dtype():
    assert looks_numeric(pd.Series(["1", "2", "3", "4", "5"]))
    assert looks_numeric(pd.Series(["1,200.50", "$300", "17%"]))
    assert not looks_numeric(pd.Series(["apple", "pear", "1"]))


@pytest.mark.parametrize(
    "value, is_date, is_number",
    [
        ("2026-01-05", True, False),
        ("05/01/2026", True, False),
        ("5 January 2026", True, False),
        ("1,200.50", False, True),
        ("$300", False, True),
        ("N/A", False, False),
        ("", False, False),
    ],
)
def test_value_classifiers(value, is_date, is_number):
    assert is_date_like(value) is is_date
    assert is_numeric_like(value) is is_number


def test_as_text_turns_missing_values_into_empty_strings():
    s = pd.Series(["a", None, "c"])
    assert list(as_text(s)) == ["a", "", "c"]


def test_profile_reports_nulls_and_samples(messy_df):
    profile = profile_dataset(messy_df)
    assert profile.rows == 11 and profile.columns == 7
    notes = next(c for c in profile.column_profiles if c.name == "Notes")
    assert notes.null_count == 9
    amount = next(c for c in profile.column_profiles if c.name == "Amount")
    assert amount.numeric_like_pct > 80


def test_score_falls_with_severity_and_maps_to_dimensions(messy_df):
    issues = detect_all(messy_df, Settings())
    score = score_dataset(issues)
    assert 0 <= score.score < 100
    assert score.issue_count == len(issues)
    assert score.grade in {"Excellent", "Good", "Fair", "Poor", "Critical"}
    for issue in issues:
        assert issue.issue_type in DIMENSION_OF
