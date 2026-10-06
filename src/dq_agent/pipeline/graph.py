"""The agent workflow as two LangGraph state machines with a human in between.

    ANALYSE   profile -> detect -> score_before -> advise -> END (awaiting approval)
                                                      |
                                            [ a human approves each fix ]
                                                      |
    APPLY     apply_fixes -> re-detect -> score_after -> report -> END

Splitting the workflow at the approval point is the whole design, not a
convenience. The brief calls for a human-in-the-loop step, and a person cannot
be expected to answer inside a single blocking graph run: they may take an
hour, or never come back. So the analyse phase ends by handing over a list of
proposals, and the apply phase starts from what they actually approved.

Both phases share the same node functions and the same state type, so the
LangSmith trace of a full session reads as one workflow in two acts.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from dq_agent.config import Settings
from dq_agent.detectors.profile import profile_dataset
from dq_agent.detectors.rules import detect_all
from dq_agent.detectors.scoring import score_dataset
from dq_agent.domain import (
    Advice,
    AppliedFix,
    DatasetProfile,
    Decision,
    Issue,
    QualityScore,
)
from dq_agent.fixes.executor import apply_all
from dq_agent.llm.base import FixAdvisor, ProviderAuthError, ProviderError
from dq_agent.llm.rules import advise_issue


class DQState(TypedDict, total=False):
    # Context
    settings: Settings
    advisor: FixAdvisor
    filename: str
    df: pd.DataFrame

    # Analyse phase
    profile: DatasetProfile
    issues: list[Issue]
    score_before: QualityScore
    advice: list[Advice]
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    advice_fallbacks: int
    warnings: list[str]

    # Apply phase
    decisions: list[Decision]
    cleaned_df: pd.DataFrame
    applied: list[AppliedFix]
    issues_after: list[Issue]
    score_after: QualityScore


def _context_line(profile: DatasetProfile, filename: str) -> str:
    columns = ", ".join(f"{c.name} ({c.dtype})" for c in profile.column_profiles[:12])
    more = "" if len(profile.column_profiles) <= 12 else f" and {len(profile.column_profiles) - 12} more"
    return f"'{filename}' with {profile.rows:,} rows and {profile.columns} columns: {columns}{more}."


# --- analyse phase nodes ---------------------------------------------------


def profile_node(state: DQState) -> dict[str, Any]:
    """Profile Agent: shape, types, null counts, samples per column."""
    return {"profile": profile_dataset(state["df"]), "warnings": []}


def detect_node(state: DQState) -> dict[str, Any]:
    """Detector Agent: nine deterministic scans, worst issues first."""
    return {"issues": detect_all(state["df"], state["settings"])}


def score_before_node(state: DQState) -> dict[str, Any]:
    """Validator, first pass: the score the dataset arrives with."""
    return {"score_before": score_dataset(state["issues"])}


def advise_node(state: DQState) -> dict[str, Any]:
    """Fixer Agent: the LLM picks one allowed strategy per issue and explains it.

    On a recoverable provider failure every issue falls back to the rule
    advisor and the run carries a visible warning. A rejected key or an
    unavailable model is re-raised, because falling back would hide a
    one-click configuration problem.
    """
    settings, advisor = state["settings"], state["advisor"]
    issues = state["issues"][: settings.max_issues_advised]
    skipped = state["issues"][settings.max_issues_advised :]
    warnings = list(state.get("warnings", []))

    if not issues:
        return {
            "advice": [],
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "advice_fallbacks": 0,
        }

    context = _context_line(state["profile"], state["filename"])
    fallbacks = 0
    try:
        result = advisor.advise(issues, context)
        advice = result.advice
        llm_calls, p_tokens, c_tokens = result.llm_calls, result.prompt_tokens, result.completion_tokens
        warnings.extend(result.warnings)
    except ProviderAuthError:
        raise
    except ProviderError as exc:
        advice = [advise_issue(i) for i in issues]
        fallbacks = len(advice)
        llm_calls = p_tokens = c_tokens = 0
        warnings.append(f"{len(advice)} recommendations came from the built-in rules: {exc}")

    if skipped:
        advice.extend(advise_issue(i) for i in skipped)
        warnings.append(
            f"{len(skipped)} lower-priority issues were advised by the built-in rules to stay within "
            f"the limit of {settings.max_issues_advised} AI-advised issues per run."
        )

    return {
        "advice": advice,
        "llm_calls": llm_calls,
        "prompt_tokens": p_tokens,
        "completion_tokens": c_tokens,
        "advice_fallbacks": fallbacks,
        "warnings": warnings,
    }


# --- apply phase nodes -----------------------------------------------------


def apply_node(state: DQState) -> dict[str, Any]:
    """Executor: runs only the approved decisions, with plain pandas."""
    cleaned, applied = apply_all(state["df"], state["issues"], state["decisions"])
    return {"cleaned_df": cleaned, "applied": applied}


def revalidate_node(state: DQState) -> dict[str, Any]:
    """Validator, second pass: re-run every detector on the cleaned data.

    Re-running the same detectors rather than assuming the fixes worked is the
    point. A fix that did not help shows up here as an issue that is still
    present, and the score refuses to move.
    """
    issues_after = detect_all(state["cleaned_df"], state["settings"])
    return {"issues_after": issues_after, "score_after": score_dataset(issues_after)}


def build_analyse_graph():
    g = StateGraph(DQState)
    g.add_node("profile", profile_node)
    g.add_node("detect", detect_node)
    g.add_node("score_before", score_before_node)
    g.add_node("advise", advise_node)
    g.add_edge(START, "profile")
    g.add_edge("profile", "detect")
    g.add_edge("detect", "score_before")
    g.add_edge("score_before", "advise")
    g.add_edge("advise", END)
    return g.compile()


def build_apply_graph():
    g = StateGraph(DQState)
    g.add_node("apply_fixes", apply_node)
    g.add_node("revalidate", revalidate_node)
    g.add_edge(START, "apply_fixes")
    g.add_edge("apply_fixes", "revalidate")
    g.add_edge("revalidate", END)
    return g.compile()


def run_analysis(*, df: pd.DataFrame, filename: str, settings: Settings, advisor: FixAdvisor) -> DQState:
    return build_analyse_graph().invoke(
        {"df": df, "filename": filename, "settings": settings, "advisor": advisor}
    )


def run_apply(
    *, df: pd.DataFrame, issues: list[Issue], decisions: list[Decision], settings: Settings
) -> DQState:
    return build_apply_graph().invoke(
        {"df": df, "issues": issues, "decisions": decisions, "settings": settings}
    )
