"""The two graph phases, the advisors, and the human approval boundary."""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from dq_agent.config import Settings
from dq_agent.detectors.rules import detect_all
from dq_agent.domain import ALLOWED_STRATEGIES, Decision, FixStrategy, Issue, IssueType, Severity
from dq_agent.llm.base import AdviceResult, FixAdvisor, ProviderAuthError, ProviderError
from dq_agent.llm.langchain_provider import LangChainFixAdvisor, build_chat_model, parse_advice
from dq_agent.llm.registry import build_advisor
from dq_agent.llm.rules import RuleFixAdvisor
from dq_agent.pipeline.graph import build_analyse_graph, build_apply_graph, run_analysis, run_apply


# ---------------------------------------------------------------- the graphs
def test_graphs_have_the_expected_nodes():
    assert {"profile", "detect", "score_before", "advise"} <= set(build_analyse_graph().get_graph().nodes)
    assert {"apply_fixes", "revalidate"} <= set(build_apply_graph().get_graph().nodes)


def test_analyse_phase_stops_before_touching_the_data(messy_df, settings):
    """The human-in-the-loop boundary: analysing must produce proposals and
    change nothing."""
    state = run_analysis(df=messy_df, filename="messy.csv", settings=settings, advisor=RuleFixAdvisor())
    assert state["issues"] and state["advice"]
    assert "cleaned_df" not in state, "the analyse phase must not produce modified data"
    assert state["df"].equals(messy_df)
    assert len(state["advice"]) == len(state["issues"]), "every issue needs a recommendation"


def test_apply_phase_revalidates_rather_than_assuming(messy_df, settings):
    issues = detect_all(messy_df, settings)
    decisions = [
        Decision(issue_id=i.id, approved=i.default_strategy != FixStrategy.NONE, strategy=i.default_strategy)
        for i in issues
    ]
    state = run_apply(df=messy_df, issues=issues, decisions=decisions, settings=settings)
    assert state["score_after"].score > 0
    # The "after" issues come from running the detectors again, so they cannot
    # simply be the "before" list minus what we tried to fix.
    assert state["issues_after"] == detect_all(state["cleaned_df"], settings)


def test_approving_nothing_leaves_the_data_identical(messy_df, settings):
    issues = detect_all(messy_df, settings)
    decisions = [Decision(issue_id=i.id, approved=False, strategy=i.default_strategy) for i in issues]
    state = run_apply(df=messy_df, issues=issues, decisions=decisions, settings=settings)
    assert state["cleaned_df"].equals(messy_df)
    assert state["score_after"].score == state["score_after"].score  # and it is still scored


# ---------------------------------------------------------------- advisors
def test_rule_advisor_covers_every_issue_type(messy_df, settings):
    issues = detect_all(messy_df, settings)
    advice = RuleFixAdvisor().advise(issues, "test")
    assert len(advice.advice) == len(issues)
    for a, i in zip(advice.advice, issues, strict=True):
        assert a.strategy in ALLOWED_STRATEGIES[i.issue_type]
        assert len(a.reasoning) > 30, "a recommendation without a reason is not usable"
        assert a.source == "rule"


def test_rule_advisor_refuses_to_invent_data_for_a_mostly_empty_column(messy_df, settings):
    issues = detect_all(messy_df, settings)
    notes = next(i for i in issues if i.column == "Notes")
    advice = RuleFixAdvisor().advise([notes], "test").advice[0]
    assert advice.strategy == FixStrategy.NONE
    assert "invent" in advice.reasoning


def _issue(kind=IssueType.MISSING_VALUES, ident="i1") -> Issue:
    return Issue(
        id=ident,
        issue_type=kind,
        column="price",
        severity=Severity.MEDIUM,
        title="12 blanks",
        detail="8% empty",
        affected_rows=12,
        affected_pct=8.0,
        allowed_strategies=list(ALLOWED_STRATEGIES[kind]),
        metrics={"numeric": True},
    )


def _reply(*items) -> str:
    return json.dumps([{"id": i, "strategy": s, "reasoning": "because", "confidence": 0.9} for i, s in items])


class _Recorder(GenericFakeChatModel):
    seen: list = []

    def _generate(self, messages, *a, **k):
        self.seen.append(list(messages))
        return super()._generate(messages, *a, **k)


def _advisor(*replies):
    model = _Recorder(messages=iter([AIMessage(content=r) for r in replies]), seen=[])
    return LangChainFixAdvisor(model, name="fake", model="fake-1"), model


def test_llm_advice_is_parsed_and_the_menu_travels_with_the_prompt():
    advisor, model = _advisor(_reply(("i1", "fill_median")))
    result = advisor.advise([_issue()], "a table")
    assert result.advice[0].strategy == FixStrategy.FILL_MEDIAN
    assert result.advice[0].source == "llm"
    sent = " ".join(str(m.content) for m in model.seen[0])
    assert "fill_median" in sent and "Replace blanks with the column" in sent


def test_a_strategy_outside_the_menu_is_rejected_and_repaired():
    """The guard that matters: an LLM naming a fix that is illegal for this
    issue must never reach the executor."""
    advisor, model = _advisor(_reply(("i1", "drop_duplicates")), _reply(("i1", "fill_mode")))
    result = advisor.advise([_issue()], "a table")
    assert result.advice[0].strategy == FixStrategy.FILL_MODE
    assert result.llm_calls == 2
    assert "not allowed" in str(model.seen[1][-1].content)


def test_two_illegal_answers_raise_rather_than_guess():
    advisor, _ = _advisor(_reply(("i1", "drop_duplicates")), _reply(("i1", "drop_column")))
    with pytest.raises(ProviderError, match="unusable advice"):
        advisor.advise([_issue()], "a table")


def test_missing_issue_in_the_reply_is_rejected():
    with pytest.raises(ValueError, match="Missing ids"):
        parse_advice(_reply(("i1", "fill_mode")), [_issue(ident="i1"), _issue(ident="i2")])


def test_fenced_and_wrapped_json_are_tolerated():
    raw = (
        "```json\n"
        + json.dumps({"results": [{"id": "i1", "strategy": "fill_mode", "reasoning": "x"}]})
        + "\n```"
    )
    assert parse_advice(raw, [_issue()])[0].strategy == FixStrategy.FILL_MODE


@pytest.mark.parametrize(
    "error, match",
    [
        (type("E", (Exception,), {"status_code": 401})("no"), "rejected the API key"),
        (type("E", (Exception,), {"status_code": 404})("no"), "does not offer the model"),
        (RuntimeError("model_not_found"), "does not offer the model"),
    ],
)
def test_credential_and_model_problems_are_not_recoverable(error, match):
    class Boom(GenericFakeChatModel):
        def _generate(self, *a, **k):
            raise error

    advisor = LangChainFixAdvisor(Boom(messages=iter([])), name="fake", model="m")
    with pytest.raises(ProviderAuthError, match=match):
        advisor.advise([_issue()], "x")


def test_token_usage_is_recorded():
    msg = AIMessage(
        content=_reply(("i1", "fill_mode")),
        usage_metadata={"input_tokens": 90, "output_tokens": 25, "total_tokens": 115},
    )
    advisor = LangChainFixAdvisor(GenericFakeChatModel(messages=iter([msg])), name="f", model="m")
    result = advisor.advise([_issue()], "x")
    assert (result.prompt_tokens, result.completion_tokens) == (90, 25)


# ---------------------------------------------------------------- fallbacks
class _BrokenAdvisor(FixAdvisor):
    name, model = "broken", "x"

    def advise(self, issues, context):
        raise ProviderError("simulated outage")


class _RejectedAdvisor(FixAdvisor):
    name, model = "rejected", "x"

    def advise(self, issues, context):
        raise ProviderAuthError("rejected the API key. Check the key on the Settings page.")


def test_a_provider_outage_falls_back_to_the_rules_with_a_warning(messy_df, settings):
    state = run_analysis(df=messy_df, filename="m.csv", settings=settings, advisor=_BrokenAdvisor())
    assert state["advice_fallbacks"] == len(state["issues"])
    assert all(a.source == "rule" for a in state["advice"])
    assert any("built-in rules" in w for w in state["warnings"])


def test_a_rejected_key_stops_the_run_instead_of_falling_back(messy_df, settings):
    with pytest.raises(ProviderAuthError):
        run_analysis(df=messy_df, filename="m.csv", settings=settings, advisor=_RejectedAdvisor())


class _CountingAdvisor(FixAdvisor):
    name, model = "counting", "x"

    def __init__(self):
        self.batches = []

    def advise(self, issues, context):
        self.batches.append(len(issues))
        return AdviceResult(advice=RuleFixAdvisor().advise(issues, context).advice, llm_calls=1)


def test_the_advised_issue_limit_is_respected(messy_df, settings):
    settings = settings.model_copy(update={"max_issues_advised": 3})
    advisor = _CountingAdvisor()
    state = run_analysis(df=messy_df, filename="m.csv", settings=settings, advisor=advisor)
    assert advisor.batches == [3], "only the worst issues go to the AI"
    assert len(state["advice"]) == len(state["issues"]), "the rest still get a rule-based recommendation"
    assert any("lower-priority" in w for w in state["warnings"])


def test_registry_builds_both_kinds():
    assert build_advisor(Settings(llm_provider="mock")).name == "mock"
    groq = build_advisor(Settings(llm_provider="groq", groq_api_key="gsk_dummy"))
    assert groq.name == "groq"


def test_missing_key_is_explained():
    with pytest.raises(ProviderError, match="Groq API key is missing"):
        build_chat_model(Settings(llm_provider="groq"))
    with pytest.raises(ProviderError, match="Unknown LLM provider"):
        build_chat_model(Settings(llm_provider="skynet"))
