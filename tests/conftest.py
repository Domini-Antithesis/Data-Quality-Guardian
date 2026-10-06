from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from dq_agent.config import Settings
from dq_agent.storage.db import Database


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Tests never read the developer's real .env or environment."""
    for key in list(os.environ):
        if key.startswith(("GROQ_", "GOOGLE_", "OPENAI_", "LANGCHAIN_", "LANGSMITH_", "LLM_", "MISSING_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(llm_provider="mock", data_dir=str(tmp_path))


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.sqlite3")


# A tiny fixture with one planted flaw per detector, so ground truth is exact.
MESSY_CSV = (
    "OrderID,OrderDate,City,CustomerEmail,Amount,Currency,Notes\n"
    "1,2026-01-05,Mumbai,a@example.com,100.50,INR,ok\n"
    "2,06/01/2026,  mumbai ,b@example.com,200.00,INR,\n"
    "3,2026-01-07,MUMBAI,not-an-email,pending,INR,\n"
    "4,2026-01-08,Delhi,d@example.com,N/A,INR,\n"
    "5,2026-01-09,Delhi,e@example.com,250.00,INR,\n"
    "6,2026-01-10,Delhi,f@example.com,,INR,\n"
    "7,2026-01-11,Pune,g@example.com,275.00,INR,note\n"
    "8,2026-01-12,Pune,h@example.com,225.00,INR,\n"
    "9,2026-01-13,Pune,i@example.com,260.00,INR,\n"
    "10,2026-01-14,Pune,j@example.com,240.00,INR,\n"
    "10,2026-01-14,Pune,j@example.com,240.00,INR,\n"
)


@pytest.fixture
def messy_csv() -> str:
    return MESSY_CSV


@pytest.fixture
def messy_df(messy_csv: str) -> pd.DataFrame:
    from dq_agent.sources import load_dataframe

    return load_dataframe("messy.csv", messy_csv.encode())
