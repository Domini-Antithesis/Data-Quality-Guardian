"""The web API and the static frontend in one FastAPI app."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dq_agent import __version__
from dq_agent.config import Settings, apply_tracing_env, load_settings, save_settings
from dq_agent.domain import (
    STRATEGY_HELP,
    Decision,
    FixStrategy,
    RunStatus,
)
from dq_agent.llm.base import ProviderError
from dq_agent.llm.registry import build_advisor
from dq_agent.service import RunService
from dq_agent.sources import LoadError, load_dataframe, preview
from dq_agent.storage.db import Database, score_from_json

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "resources" / "samples"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class DecisionIn(BaseModel):
    issue_id: str
    approved: bool = True
    strategy: str
    fill_value: str | None = None


class ApplyBody(BaseModel):
    decisions: list[DecisionIn]


def create_app(data_dir: str | Path | None = None, *, env_file: str | None = ".env") -> FastAPI:
    settings = load_settings(data_dir, env_file=env_file)
    apply_tracing_env(settings)
    data_path = Path(settings.data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    db = Database(data_path / "dq.sqlite3")
    holder: dict[str, Settings] = {"settings": settings}
    service = RunService(db, lambda: holder["settings"], data_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    app = FastAPI(
        title="Data Quality Auto-Fixing & Validation Agent",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.db = db
    app.state.service = service
    app.state.settings_holder = holder

    def current() -> Settings:
        return holder["settings"]

    def _run_or_404(run_id: int):
        run = db.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    # --- health & settings ------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        s = current()
        key_field = {"groq": "groq_api_key", "gemini": "google_api_key", "openai": "openai_api_key"}.get(
            s.llm_provider
        )
        return {
            "status": "ok",
            "version": __version__,
            "provider": s.llm_provider,
            "model": s.resolved_model(),
            "llm_ready": s.llm_provider in ("mock", "ollama") or bool(getattr(s, key_field or "", "")),
            "tracing": bool(s.langchain_tracing_v2 and s.langchain_api_key),
        }

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return current().public_view()

    @app.put("/api/settings")
    def put_settings(updates: dict[str, Any]) -> dict[str, Any]:
        try:
            new = save_settings(current(), updates)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        holder["settings"] = new
        apply_tracing_env(new)
        return new.public_view()

    @app.post("/api/settings/test/llm")
    def test_llm() -> dict[str, Any]:
        try:
            return {"ok": True, "message": build_advisor(current()).test_connection()}
        except ProviderError as exc:
            return {"ok": False, "message": str(exc)}
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "message": f"Unexpected error: {exc}"}

    @app.get("/api/strategies")
    def strategies() -> dict[str, str]:
        """What each fix does, in plain language. The UI shows these as help."""
        return {s.value: STRATEGY_HELP[s] for s in FixStrategy}

    # --- analysis ---------------------------------------------------------
    @app.post("/api/analyze")
    async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File is larger than 25 MB.")
        if not data:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        try:
            run_id = service.start_analysis(file.filename or "upload.csv", data)
        except LoadError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"run_id": run_id}

    @app.post("/api/samples/{name}/analyze")
    def analyze_sample(name: str) -> dict[str, Any]:
        path = SAMPLES_DIR / Path(name).name
        if not path.exists() or path.suffix != ".csv":
            raise HTTPException(status_code=404, detail="Sample not found")
        return {"run_id": service.start_analysis(path.name, path.read_bytes())}

    @app.get("/api/samples")
    def samples() -> list[dict[str, str]]:
        if not SAMPLES_DIR.exists():
            return []
        return [{"name": p.name} for p in sorted(SAMPLES_DIR.glob("*.csv"))]

    @app.get("/api/samples/{name}")
    def sample_file(name: str) -> FileResponse:
        path = SAMPLES_DIR / Path(name).name
        if not path.exists() or path.suffix != ".csv":
            raise HTTPException(status_code=404, detail="Sample not found")
        return FileResponse(path, media_type="text/csv", filename=path.name)

    # --- runs -------------------------------------------------------------
    @app.get("/api/runs")
    def list_runs(limit: int = 50) -> list[dict[str, Any]]:
        return [r.model_dump() for r in db.list_runs(min(limit, 200))]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: int) -> dict[str, Any]:
        run = _run_or_404(run_id)
        raw = db.get_run_raw(run_id) or {}
        issues = db.load_issues(run_id, "before")
        advice = db.load_advice(run_id)
        decisions = {d.issue_id: d for d in db.load_decisions(run_id)}
        return {
            "run": run.model_dump(),
            "profile": json.loads(raw.get("profile") or "{}"),
            "score_before": (
                s.model_dump() if (s := score_from_json(raw.get("score_before_detail", ""))) else None
            ),
            "score_after": (
                s.model_dump() if (s := score_from_json(raw.get("score_after_detail", ""))) else None
            ),
            "issues": [
                {
                    **i.model_dump(),
                    "advice": advice[i.id].model_dump() if i.id in advice else None,
                    "decision": decisions[i.id].model_dump() if i.id in decisions else None,
                }
                for i in issues
            ],
            "issues_after": [i.model_dump() for i in db.load_issues(run_id, "after")],
            "applied": [a.model_dump() for a in db.load_applied(run_id)],
            "audit": db.audit_log(run_id),
        }

    @app.get("/api/runs/{run_id}/preview")
    def run_preview(run_id: int, which: str = "original", rows: int = 20) -> dict[str, Any]:
        run = _run_or_404(run_id)
        try:
            if which == "cleaned":
                path = service.cleaned_path(run_id)
                if not path.exists():
                    raise HTTPException(status_code=409, detail="No cleaned file yet. Apply fixes first.")
                df = load_dataframe(path.name, path.read_bytes())
            else:
                df = service.load_original(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        return {"filename": run.filename, "which": which, **preview(df, min(rows, 200))}

    @app.post("/api/runs/{run_id}/apply")
    def apply(run_id: int, body: ApplyBody) -> dict[str, Any]:
        run = _run_or_404(run_id)
        if run.status not in (RunStatus.AWAITING_APPROVAL, RunStatus.COMPLETED):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"This run is '{run.status.value}'. Fixes can only be applied once analysis has finished."
                ),
            )
        known = {i.id for i in db.load_issues(run_id, "before")}
        decisions = []
        for d in body.decisions:
            if d.issue_id not in known:
                raise HTTPException(status_code=400, detail=f"Unknown issue '{d.issue_id}' for this run.")
            try:
                strategy = FixStrategy(d.strategy)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Unknown fix '{d.strategy}'.") from exc
            decisions.append(
                Decision(
                    issue_id=d.issue_id,
                    approved=d.approved,
                    strategy=strategy,
                    fill_value=d.fill_value,
                )
            )
        try:
            service.apply_decisions(run_id, decisions)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        return get_run(run_id)

    @app.get("/api/runs/{run_id}/download")
    def download(run_id: int) -> FileResponse:
        run = _run_or_404(run_id)
        path = service.cleaned_path(run_id)
        if not path.exists():
            raise HTTPException(status_code=409, detail="No cleaned file yet. Apply fixes first.")
        stem = Path(run.filename).stem
        return FileResponse(path, media_type="text/csv", filename=f"{stem}_cleaned.csv")

    @app.get("/api/runs/{run_id}/audit.md")
    def audit_report(run_id: int) -> PlainTextResponse:
        run = _run_or_404(run_id)
        applied = db.load_applied(run_id)
        advice = db.load_advice(run_id)
        issues = {i.id: i for i in db.load_issues(run_id, "before")}
        lines = [
            f"# Data quality audit report: {run.filename}",
            "",
            f"- Run: #{run.id}",
            f"- Started: {run.created_at}",
            f"- Finished: {run.updated_at or 'not finished'}",
            f"- Status: {run.status.value}",
            f"- Advisor: {run.provider} ({run.model}), {run.llm_calls} AI calls, "
            f"{run.prompt_tokens + run.completion_tokens} tokens",
            "",
            "## Result",
            "",
            "| | Before | After |",
            "|---|---|---|",
            f"| Quality score | {run.score_before:.1f} | {run.score_after:.1f} |",
            f"| Rows | {run.rows_before:,} | {run.rows_after:,} |",
            f"| Columns | {run.columns_before} | {run.columns_after} |",
            f"| Issues | {run.issues_found} | {len(db.load_issues(run_id, 'after'))} |",
            "",
            "## Fixes",
            "",
        ]
        if applied:
            for fix in applied:
                issue = issues.get(fix.issue_id)
                rec = advice.get(fix.issue_id)
                lines.append(f"### {issue.title if issue else fix.issue_id}")
                lines.append("")
                lines.append(f"- Fix chosen: `{fix.strategy.value}`")
                lines.append(f"- Applied: {'yes' if fix.applied else 'no'}")
                lines.append(f"- Effect: {fix.detail}")
                if rec:
                    lines.append(f"- Recommended by: {rec.source}")
                    lines.append(f"- Reasoning: {rec.reasoning}")
                lines.append("")
        else:
            lines.append("_No fixes have been applied yet._\n")
        lines.append("## Full audit trail")
        lines.append("")
        lines.append("| When | Who | Event | Detail |")
        lines.append("|---|---|---|---|")
        for entry in db.audit_log(run_id):
            detail = entry["detail"].replace("|", "\\|")
            lines.append(f"| {entry['at']} | {entry['actor']} | {entry['event']} | {detail} |")
        report = "\n".join(lines)
        return PlainTextResponse(
            report,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="dq-audit-run-{run_id}.md"'},
        )

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: int) -> dict[str, Any]:
        if not db.delete_run(run_id):
            raise HTTPException(status_code=404, detail="Run not found")
        import shutil

        shutil.rmtree(service.run_dir(run_id), ignore_errors=True)
        return {"deleted": run_id}

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        return db.overview()

    # --- frontend ---------------------------------------------------------
    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")
    else:  # pragma: no cover

        @app.get("/", include_in_schema=False)
        def index_missing() -> PlainTextResponse:
            return PlainTextResponse("Frontend files not found; the JSON API is under /api.")

    @app.exception_handler(LoadError)
    async def _load_error(_, exc: LoadError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


def get_app() -> FastAPI:
    """Entry point for `uvicorn --factory`. Nothing happens at import time."""
    return create_app()
