"""Glue between the web API and the two graph phases.

Uploaded files live on disk under `data/uploads/<run id>/`, so a run survives
a server restart and the cleaned file can be downloaded later without keeping
anything in memory.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from dq_agent.config import Settings, apply_tracing_env
from dq_agent.domain import Decision, RunStatus, utc_now_iso
from dq_agent.llm.base import ProviderError
from dq_agent.llm.registry import build_advisor
from dq_agent.pipeline.graph import run_analysis, run_apply
from dq_agent.sources import load_dataframe
from dq_agent.storage.db import Database


class RunService:
    def __init__(self, db: Database, settings_provider: Callable[[], Settings], data_dir: Path):
        self.db = db
        self._settings_provider = settings_provider
        self.data_dir = Path(data_dir)
        self._threads: dict[int, threading.Thread] = {}

    @property
    def settings(self) -> Settings:
        return self._settings_provider()

    # --- files -------------------------------------------------------------
    def run_dir(self, run_id: int) -> Path:
        path = self.data_dir / "uploads" / str(run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def original_path(self, run_id: int, filename: str | None = None) -> Path:
        """The uploaded file is kept byte-for-byte. Re-reading it is exactly
        reproducible and needs no extra storage format, so the "before" data a
        fix is applied to is provably the data the user uploaded."""
        if filename is None:
            existing = [p for p in self.run_dir(run_id).glob("original.*")]
            if not existing:
                raise FileNotFoundError(f"The uploaded file for run {run_id} is no longer on disk.")
            return existing[0]
        suffix = Path(filename).suffix.lower() or ".csv"
        return self.run_dir(run_id) / f"original{suffix}"

    def cleaned_path(self, run_id: int) -> Path:
        return self.run_dir(run_id) / "cleaned.csv"

    def load_original(self, run_id: int) -> pd.DataFrame:
        path = self.original_path(run_id)
        return load_dataframe(path.name, path.read_bytes())

    # --- phase 1: analyse --------------------------------------------------
    def start_analysis(self, filename: str, data: bytes) -> int:
        settings = self.settings
        df = load_dataframe(filename, data)
        run_id = self.db.create_run(filename, settings.llm_provider, settings.resolved_model())
        self.original_path(run_id, filename).write_bytes(data)
        self.db.update_run(run_id, rows_before=len(df), columns_before=len(df.columns))
        self.db.audit(run_id, "human", "uploaded", f"{filename}: {len(df):,} rows, {len(df.columns)} columns")

        def _target() -> None:
            try:
                self._analyse(run_id, df, filename)
            except Exception as exc:
                self.db.update_run(run_id, status=RunStatus.FAILED, error=str(exc)[:1000])
                self.db.audit(run_id, "system", "analysis failed", str(exc)[:500])
            finally:
                self._threads.pop(run_id, None)

        thread = threading.Thread(target=_target, name=f"analyse-{run_id}", daemon=True)
        self._threads[run_id] = thread
        thread.start()
        return run_id

    def _analyse(self, run_id: int, df: pd.DataFrame, filename: str) -> None:
        settings = self.settings
        apply_tracing_env(settings)
        self.db.update_run(run_id, status=RunStatus.ANALYZING)
        try:
            advisor = build_advisor(settings)
        except ProviderError as exc:
            self.db.update_run(run_id, status=RunStatus.FAILED, error=str(exc))
            self.db.audit(run_id, "system", "analysis failed", str(exc)[:500])
            return

        state = run_analysis(df=df, filename=filename, settings=settings, advisor=advisor)
        issues = state["issues"]
        self.db.save_issues(run_id, issues, phase="before")
        self.db.save_advice(run_id, state["advice"])
        self.db.audit(run_id, "detector", "scanned", f"{len(issues)} issues found across 9 checks")
        source = "built-in rules" if state["advice_fallbacks"] else f"{advisor.name} ({advisor.model})"
        self.db.audit(
            run_id, "ai", "recommended fixes", f"{len(state['advice'])} recommendations from {source}"
        )
        self.db.update_run(
            run_id,
            status=RunStatus.AWAITING_APPROVAL,
            profile=state["profile"],
            score_before=state["score_before"].score,
            score_before_detail=state["score_before"],
            issues_found=len(issues),
            llm_calls=state["llm_calls"],
            prompt_tokens=state["prompt_tokens"],
            completion_tokens=state["completion_tokens"],
            advice_fallbacks=state["advice_fallbacks"],
            warnings=state["warnings"],
        )

    # --- phase 2: apply ----------------------------------------------------
    def apply_decisions(self, run_id: int, decisions: list[Decision]) -> None:
        settings = self.settings
        issues = self.db.load_issues(run_id, "before")
        df = self.load_original(run_id)
        self.db.save_decisions(run_id, decisions)
        approved = [d for d in decisions if d.approved]
        self.db.audit(
            run_id,
            "human",
            "approved fixes",
            f"{len(approved)} of {len(decisions)} proposed fixes approved",
        )
        self.db.update_run(run_id, status=RunStatus.APPLYING)

        state = run_apply(df=df, issues=issues, decisions=decisions, settings=settings)
        cleaned = state["cleaned_df"]
        applied = state["applied"]
        cleaned.to_csv(self.cleaned_path(run_id), index=False)

        self.db.save_applied(run_id, applied)
        self.db.save_issues(run_id, state["issues_after"], phase="after")
        for fix in applied:
            self.db.audit(
                run_id,
                "system",
                "applied" if fix.applied else "skipped",
                f"{fix.issue_id} -> {fix.strategy.value}: {fix.detail}",
            )
        self.db.update_run(
            run_id,
            status=RunStatus.COMPLETED,
            rows_after=len(cleaned),
            columns_after=len(cleaned.columns),
            score_after=state["score_after"].score,
            score_after_detail=state["score_after"],
            fixes_applied=sum(1 for f in applied if f.applied),
        )
        before = self.db.get_run(run_id)
        self.db.audit(
            run_id,
            "system",
            "validated",
            f"Quality score {before.score_before:.1f} -> {state['score_after'].score:.1f}, "
            f"{len(state['issues_after'])} issues remaining",
        )

    def wait(self, run_id: int, timeout: float | None = None) -> None:
        thread = self._threads.get(run_id)
        if thread:
            thread.join(timeout)


__all__ = ["RunService", "utc_now_iso"]
