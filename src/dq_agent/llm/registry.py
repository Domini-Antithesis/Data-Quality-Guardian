"""The only place a concrete advisor is chosen by name."""

from __future__ import annotations

from dq_agent.config import Settings
from dq_agent.llm.base import FixAdvisor
from dq_agent.llm.rules import RuleFixAdvisor


def build_advisor(settings: Settings) -> FixAdvisor:
    if settings.llm_provider == "mock":
        return RuleFixAdvisor()
    from dq_agent.llm.langchain_provider import LangChainFixAdvisor, build_chat_model

    return LangChainFixAdvisor(
        build_chat_model(settings), name=settings.llm_provider, model=settings.resolved_model()
    )
