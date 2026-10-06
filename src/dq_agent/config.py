"""Configuration: defaults, then .env / environment, then data/settings.json.

Same precedence everywhere, and secrets are never handed back out by the API.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

DEFAULT_MODELS: dict[str, str] = {
    "mock": "rule-based-v1",
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
}

PROVIDERS: tuple[str, ...] = ("mock", "groq", "gemini", "openai", "ollama")

SECRET_FIELDS: frozenset[str] = frozenset(
    {"groq_api_key", "google_api_key", "openai_api_key", "langchain_api_key"}
)


class Settings(BaseModel):
    llm_provider: str = "groq"
    llm_model: str = ""
    groq_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Detector thresholds. Every one is adjustable from the Settings page.
    missing_warn_pct: float = Field(default=5.0, ge=0, le=100)
    missing_critical_pct: float = Field(default=40.0, ge=0, le=100)
    outlier_iqr_multiplier: float = Field(default=3.0, ge=1.0, le=10.0)
    outlier_min_rows: int = Field(default=20, ge=5, le=100000)
    mixed_type_tolerance_pct: float = Field(default=2.0, ge=0, le=100)
    high_cardinality_ratio: float = Field(default=0.5, ge=0, le=1)

    auto_approve_safe_fixes: bool = False
    max_issues_advised: int = Field(default=40, ge=1, le=200)

    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "data-quality-agent"

    app_host: str = "127.0.0.1"
    app_port: int = 8401
    data_dir: str = "./data"

    def resolved_model(self) -> str:
        # Demo mode ignores any model name: showing a real model beside the
        # built-in rules would claim an AI ran when none did.
        if self.llm_provider == "mock":
            return DEFAULT_MODELS["mock"]
        return self.llm_model or DEFAULT_MODELS.get(self.llm_provider, "")

    def public_view(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, value in self.model_dump().items():
            if name in SECRET_FIELDS:
                out[f"{name}_set"] = bool(value)
            else:
                out[name] = value
        out["resolved_model"] = self.resolved_model()
        out["default_models"] = DEFAULT_MODELS
        out["providers"] = list(PROVIDERS)
        return out


def _coerce(field: str, raw: str) -> Any:
    kind = Settings.model_fields[field].annotation
    if kind is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if kind is int:
        return int(raw)
    if kind is float:
        return float(raw)
    return raw


def settings_file(data_dir: str | Path | None = None) -> Path:
    return Path(data_dir or os.environ.get("DATA_DIR", "./data")) / "settings.json"


def load_settings(data_dir: str | Path | None = None, *, env_file: str | Path | None = ".env") -> Settings:
    if env_file and Path(env_file).exists():
        load_dotenv(env_file, override=False)

    values: dict[str, Any] = {}
    for field in Settings.model_fields:
        raw = os.environ.get(field.upper())
        if raw:
            with contextlib.suppress(ValueError):
                values[field] = _coerce(field, raw)

    if data_dir is not None:
        values["data_dir"] = str(data_dir)
    path = settings_file(values.get("data_dir"))
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                values.update(
                    {k: v for k, v in saved.items() if k in Settings.model_fields and k != "data_dir"}
                )
        except (OSError, json.JSONDecodeError):
            pass
    return Settings(**values)


def save_settings(current: Settings, updates: dict[str, Any]) -> Settings:
    merged = current.model_dump()
    for key, value in updates.items():
        if key not in Settings.model_fields or key == "data_dir":
            continue
        if key in SECRET_FIELDS and (value is None or value == ""):
            continue
        merged[key] = value
    new = Settings(**merged)
    path = settings_file(new.data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({k: v for k, v in new.model_dump().items() if k != "data_dir"}, indent=2),
        encoding="utf-8",
    )
    return new


def apply_tracing_env(settings: Settings) -> None:
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        for key in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
            os.environ[key] = "true"
        for key in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"):
            os.environ[key] = settings.langchain_api_key
        for key in ("LANGCHAIN_PROJECT", "LANGSMITH_PROJECT"):
            os.environ[key] = settings.langchain_project
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
