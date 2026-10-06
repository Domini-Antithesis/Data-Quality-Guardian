"""The interface every fix-advisor implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from dq_agent.domain import Advice, Issue


@dataclass
class AdviceResult:
    advice: list[Advice]
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    warnings: list[str] = field(default_factory=list)


class ProviderError(RuntimeError):
    """Recoverable: rate limit, network, unusable output. The pipeline falls
    back to the deterministic rule advisor for the affected issues."""


class ProviderAuthError(ProviderError):
    """Not recoverable: the key was rejected or the model is unavailable.
    Falling back would hide a setting the user can fix in one click."""


class FixAdvisor(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def advise(self, issues: list[Issue], context: str) -> AdviceResult:
        """Recommend one allowed strategy per issue, with a reason."""

    def test_connection(self) -> str:
        from dq_agent.domain import ALLOWED_STRATEGIES, IssueType, Severity

        probe = Issue(
            id="probe",
            issue_type=IssueType.MISSING_VALUES,
            column="price",
            severity=Severity.MEDIUM,
            title="12 blank values in 'price'",
            detail="8% of this column is empty. It holds numbers.",
            affected_rows=12,
            affected_pct=8.0,
            allowed_strategies=list(ALLOWED_STRATEGIES[IssueType.MISSING_VALUES]),
            metrics={"numeric": True},
        )
        result = self.advise([probe], "A 150-row product table.")
        picked = result.advice[0].strategy.value
        return f"OK: {self.name} ({self.model}) answered - recommended '{picked}' for a sample issue."
