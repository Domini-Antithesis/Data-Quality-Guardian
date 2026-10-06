"""Settings loading, precedence, secret handling and documentation drift."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from dq_agent.config import (
    DEFAULT_MODELS,
    PROVIDERS,
    SECRET_FIELDS,
    Settings,
    apply_tracing_env,
    load_settings,
    save_settings,
)
from dq_agent.domain import ALLOWED_STRATEGIES, STRATEGY_HELP, FixStrategy, IssueType, Severity

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_env_example_documents_every_setting():
    """A setting nobody can discover is a setting nobody can use."""
    documented = set(re.findall(r"^([A-Z_0-9]+)=", (REPO_ROOT / ".env.example").read_text(), re.M))
    fields = {f.upper() for f in Settings.model_fields}
    assert not (fields - documented), f"undocumented: {sorted(fields - documented)}"
    assert not (documented - fields), f"documented but not real: {sorted(documented - fields)}"


def test_every_provider_has_a_default_model():
    assert set(PROVIDERS) == set(DEFAULT_MODELS)
    assert all(DEFAULT_MODELS.values())


def test_demo_mode_never_claims_a_real_model():
    s = Settings(llm_provider="mock", llm_model="openai/gpt-oss-120b")
    assert s.resolved_model() == "rule-based-v1"


def test_precedence_env_then_saved_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSING_WARN_PCT", "42")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    s = load_settings(tmp_path, env_file=None)
    assert s.missing_warn_pct == 42 and s.llm_provider == "openai"
    save_settings(s, {"missing_warn_pct": 7})
    assert load_settings(tmp_path, env_file=None).missing_warn_pct == 7


def test_a_bad_env_value_falls_back_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTLIER_MIN_ROWS", "lots")
    assert load_settings(tmp_path, env_file=None).outlier_min_rows == 20


def test_public_view_never_leaks_a_secret():
    view = Settings(groq_api_key="gsk_secret", langchain_api_key="lsv2_x").public_view()
    for field in SECRET_FIELDS:
        assert field not in view and f"{field}_set" in view
    assert "gsk_secret" not in json.dumps(view)


def test_saved_settings_never_store_the_data_dir(tmp_path):
    save_settings(Settings(data_dir=str(tmp_path)), {"llm_provider": "mock"})
    assert "data_dir" not in json.loads((tmp_path / "settings.json").read_text())


def test_tracing_env_is_applied_and_cleared():
    import os

    apply_tracing_env(Settings(langchain_tracing_v2=True, langchain_api_key="lsv2_x", langchain_project="p"))
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true" and os.environ["LANGCHAIN_PROJECT"] == "p"
    apply_tracing_env(Settings(langchain_tracing_v2=True, langchain_api_key=""))
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


@pytest.mark.parametrize("value", [0.5, 11])
def test_outlier_multiplier_is_bounded(value):
    with pytest.raises(ValueError):
        Settings(outlier_iqr_multiplier=value)


def test_every_issue_type_has_a_fix_menu_and_none_is_always_offered():
    for kind in IssueType:
        allowed = ALLOWED_STRATEGIES[kind]
        assert allowed, kind
        assert FixStrategy.NONE in allowed, f"{kind.value} must allow leaving it alone"


def test_every_strategy_is_explained_in_plain_language():
    """The UI and the LLM both read these, so a missing one is a silent gap."""
    for strategy in FixStrategy:
        assert STRATEGY_HELP.get(strategy), strategy
        assert len(STRATEGY_HELP[strategy]) > 20


def test_severity_weights_increase():
    weights = [s.weight for s in (Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)]
    assert weights == sorted(weights) and len(set(weights)) == 4
