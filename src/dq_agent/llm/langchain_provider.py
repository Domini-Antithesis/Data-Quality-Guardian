"""Real LLM advisors, through LangChain chat models so LangSmith tracing works
identically for Groq, Gemini, OpenAI and Ollama.

The model is given each issue plus the exact menu of strategies that are legal
for it, and must return a strategy from that menu with a reason. Its answer is
validated, and any strategy outside the menu is rejected rather than trusted.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from dq_agent.config import Settings
from dq_agent.domain import STRATEGY_HELP, Advice, FixStrategy, Issue
from dq_agent.llm.base import AdviceResult, FixAdvisor, ProviderAuthError, ProviderError

SYSTEM_PROMPT = """You are a senior data engineer advising on data-quality fixes.

For EVERY issue you are given, return one JSON object with exactly these keys:
- "id": the issue id you were given, copied exactly
- "strategy": one value chosen from that issue's own "allowed" list. Never invent one.
- "reasoning": two or three plain sentences a non-expert can follow, saying what the fix does
  to this specific column and why it beats the alternatives here. Mention the numbers you were given.
- "confidence": a number from 0.0 to 1.0
- "fill_value": only when strategy is "fill_constant"; the exact value to write, otherwise null

Rules:
- Return ONLY a JSON array of these objects, one per issue, in the same order.
- No prose, no markdown fences.
- Prefer the least destructive fix that actually solves the problem. Deleting rows or columns
  throws information away, so choose it only when nothing milder works.
- "none" is a legitimate answer when changing the data would be a guess, for example when most of
  a column is missing or when extreme values are probably genuine.
- Never invent an id and never skip an issue.
"""


class _Item(BaseModel):
    id: str
    strategy: str
    reasoning: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    fill_value: str | None = None


def _extract_json_array(text: str) -> Any:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def issue_payload(issue: Issue) -> dict[str, Any]:
    """What the model sees. The allowed menu travels with each issue, so the
    model is choosing from described options rather than guessing names."""
    return {
        "id": issue.id,
        "problem": issue.title,
        "explanation": issue.detail,
        "column": issue.column,
        "severity": issue.severity.value,
        "affected_rows": issue.affected_rows,
        "affected_percent": issue.affected_pct,
        "examples": issue.evidence,
        "allowed": [{"strategy": s.value, "means": STRATEGY_HELP[s]} for s in issue.allowed_strategies],
    }


def parse_advice(raw_text: str, issues: list[Issue]) -> list[Advice]:
    """Validate the model's answer against what each issue actually allows.
    Raises ValueError with a message precise enough to feed back to the model."""
    try:
        data = _extract_json_array(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Output was not valid JSON: {exc}") from exc
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        raise ValueError("Output must be a JSON array of objects.")

    by_id = {i.id: i for i in issues}
    parsed: dict[str, _Item] = {}
    errors: list[str] = []
    for entry in data:
        try:
            item = _Item.model_validate(entry)
        except ValidationError as exc:
            errors.append(str(exc.errors()[0].get("msg", "invalid item")))
            continue
        issue = by_id.get(item.id)
        if issue is None:
            errors.append(f"Unknown issue id '{item.id}'")
            continue
        allowed = {s.value for s in issue.allowed_strategies}
        if item.strategy not in allowed:
            errors.append(f"'{item.strategy}' is not allowed for {item.id}; choose one of {sorted(allowed)}")
            continue
        parsed[item.id] = item

    missing = [i.id for i in issues if i.id not in parsed]
    if missing:
        errors.append(f"Missing ids: {missing}")
    if errors:
        raise ValueError("; ".join(errors[:4]))

    return [
        Advice(
            issue_id=i.id,
            strategy=FixStrategy(parsed[i.id].strategy),
            reasoning=parsed[i.id].reasoning.strip()[:800],
            confidence=round(parsed[i.id].confidence, 3),
            fill_value=parsed[i.id].fill_value,
            source="llm",
        )
        for i in issues
    ]


def _usage(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)


_AUTH_MARKERS = (
    "invalid api key",
    "invalid_api_key",
    "api key not valid",
    "incorrect api key",
    "unauthorized",
    "authentication",
    "permission denied",
)


def _is_auth_error(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) in (401, 403):
        return True
    text = str(exc).lower()
    return any(m in text for m in _AUTH_MARKERS)


def _is_model_error(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 404:
        return True
    text = str(exc).lower()
    return "model_not_found" in text or "does not exist or you do not have access" in text


class LangChainFixAdvisor(FixAdvisor):
    def __init__(self, chat_model: BaseChatModel, name: str, model: str):
        self.chat_model = chat_model
        self.name = name
        self.model = model

    def advise(self, issues: list[Issue], context: str) -> AdviceResult:
        if not issues:
            return AdviceResult(advice=[])
        payload = [issue_payload(i) for i in issues]
        messages: list[Any] = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Dataset: {context}\n\nIssues to advise on:\n"
                + json.dumps(payload, ensure_ascii=False, indent=1)
            ),
        ]
        prompt_tokens = completion_tokens = 0
        last_error = ""
        for attempt in range(2):
            try:
                reply = self.chat_model.invoke(messages)
            except Exception as exc:
                if _is_model_error(exc):
                    raise ProviderAuthError(
                        f"{self.name} does not offer the model '{self.model}' on this account "
                        f"({exc}). Pick another model on the Settings page."
                    ) from exc
                if _is_auth_error(exc):
                    raise ProviderAuthError(
                        f"{self.name} rejected the API key ({exc}). Check the key on the Settings page."
                    ) from exc
                raise ProviderError(f"{self.name} request failed: {exc}") from exc
            calls = attempt + 1
            p, c = _usage(reply)
            prompt_tokens += p
            completion_tokens += c
            content = reply.content if isinstance(reply.content, str) else json.dumps(reply.content)
            try:
                return AdviceResult(
                    advice=parse_advice(content, issues),
                    llm_calls=calls,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except ValueError as exc:
                last_error = str(exc)
                if attempt == 0:
                    messages = messages + [
                        reply,
                        HumanMessage(
                            content=(
                                f"Your previous answer was rejected: {last_error}. "
                                "Reply again with ONLY the corrected JSON array, covering every id, "
                                "using only each issue's own allowed strategies."
                            )
                        ),
                    ]
        raise ProviderError(f"{self.name} returned unusable advice after a repair attempt: {last_error}")


def build_chat_model(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider
    model = settings.resolved_model()
    if provider == "groq":
        if not settings.groq_api_key:
            raise ProviderError("Groq API key is missing. Add it on the Settings page.")
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, api_key=settings.groq_api_key, temperature=0, max_retries=2)
    if provider == "gemini":
        if not settings.google_api_key:
            raise ProviderError("Google API key is missing. Add it on the Settings page.")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ProviderError("Gemini support is not installed: pip install -e .[gemini]") from exc
        return ChatGoogleGenerativeAI(model=model, google_api_key=settings.google_api_key, temperature=0)
    if provider == "openai":
        if not settings.openai_api_key:
            raise ProviderError("OpenAI API key is missing. Add it on the Settings page.")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ProviderError("OpenAI support is not installed: pip install -e .[openai]") from exc
        return ChatOpenAI(model=model, api_key=settings.openai_api_key, temperature=0, max_retries=2)
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ProviderError("Ollama support is not installed: pip install -e .[ollama]") from exc
        return ChatOllama(model=model, base_url=settings.ollama_base_url, temperature=0)
    raise ProviderError(f"Unknown LLM provider '{provider}'.")
