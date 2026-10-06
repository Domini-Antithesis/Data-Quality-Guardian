"""SQLite persistence: runs, issues, advice, decisions and the audit log.

The brief asks for a full audit trail. Every issue found, every recommendation
with its reasoning, every human decision and every applied change is stored,
so a run can be explained months later without re-running anything.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from dq_agent.domain import (
    Advice,
    AppliedFix,
    Decision,
    Issue,
    QualityScore,
    RunRecord,
    RunStatus,
    utc_now_iso,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    rows_before INTEGER DEFAULT 0,
    rows_after INTEGER DEFAULT 0,
    columns_before INTEGER DEFAULT 0,
    columns_after INTEGER DEFAULT 0,
    score_before REAL DEFAULT 0,
    score_after REAL DEFAULT 0,
    score_before_detail TEXT DEFAULT '{}',
    score_after_detail TEXT DEFAULT '{}',
    profile TEXT DEFAULT '{}',
    issues_found INTEGER DEFAULT 0,
    fixes_applied INTEGER DEFAULT 0,
    provider TEXT DEFAULT '',
    model TEXT DEFAULT '',
    llm_calls INTEGER DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    advice_fallbacks INTEGER DEFAULT 0,
    warnings TEXT DEFAULT '[]',
    error TEXT
);

CREATE TABLE IF NOT EXISTS issues (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    issue_id TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT 'before',   -- before | after
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, issue_id, phase)
);

CREATE TABLE IF NOT EXISTS advice (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    issue_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, issue_id)
);

CREATE TABLE IF NOT EXISTS decisions (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    issue_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    PRIMARY KEY (run_id, issue_id)
);

CREATE TABLE IF NOT EXISTS applied_fixes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    at TEXT NOT NULL,
    actor TEXT NOT NULL,       -- system | detector | ai | human
    event TEXT NOT NULL,
    detail TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_audit_run ON audit(run_id);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # --- runs --------------------------------------------------------------
    def create_run(self, filename: str, provider: str, model: str) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (filename, status, created_at, provider, model) VALUES (?, ?, ?, ?, ?)",
                (filename, RunStatus.QUEUED.value, utc_now_iso(), provider, model),
            )
            return int(cur.lastrowid)

    def update_run(self, run_id: int, **fields: Any) -> None:
        if not fields:
            return
        encoded: dict[str, Any] = {"updated_at": utc_now_iso()}
        for key, value in fields.items():
            if isinstance(value, RunStatus):
                value = value.value
            elif isinstance(value, bool):
                value = int(value)
            elif hasattr(value, "model_dump"):
                value = json.dumps(value.model_dump())
            elif isinstance(value, (list, dict)):
                value = json.dumps(
                    [v.model_dump() if hasattr(v, "model_dump") else v for v in value]
                    if isinstance(value, list)
                    else value
                )
            encoded[key] = value
        assignments = ", ".join(f"{k} = ?" for k in encoded)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE runs SET {assignments} WHERE id = ?", (*encoded.values(), run_id))

    def get_run(self, run_id: int) -> RunRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return _row_to_run(row) if row else None

    def get_run_raw(self, run_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [_row_to_run(r) for r in rows]

    def delete_run(self, run_id: int) -> bool:
        with self._lock, self._connect() as conn:
            for table in ("issues", "advice", "decisions", "applied_fixes", "audit"):
                conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
            return conn.execute("DELETE FROM runs WHERE id = ?", (run_id,)).rowcount > 0

    # --- issues, advice, decisions ----------------------------------------
    def save_issues(self, run_id: int, issues: list[Issue], phase: str = "before") -> None:
        rows = [(run_id, i.id, phase, json.dumps(i.model_dump())) for i in issues]
        with self._lock, self._connect() as conn:
            # Replace the phase's list as a whole. Fixes can be applied again
            # with different decisions, and an issue that no longer exists must
            # not linger from the previous pass beside a score that says it is gone.
            conn.execute("DELETE FROM issues WHERE run_id = ? AND phase = ?", (run_id, phase))
            conn.executemany(
                "INSERT OR REPLACE INTO issues (run_id, issue_id, phase, payload) VALUES (?, ?, ?, ?)",
                rows,
            )

    def load_issues(self, run_id: int, phase: str = "before") -> list[Issue]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM issues WHERE run_id = ? AND phase = ?", (run_id, phase)
            ).fetchall()
        issues = [Issue(**json.loads(r["payload"])) for r in rows]
        issues.sort(key=lambda i: (-i.severity.weight, -i.affected_pct, i.id))
        return issues

    def save_advice(self, run_id: int, advice: list[Advice]) -> None:
        rows = [(run_id, a.issue_id, json.dumps(a.model_dump())) for a in advice]
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO advice (run_id, issue_id, payload) VALUES (?, ?, ?)", rows
            )

    def load_advice(self, run_id: int) -> dict[str, Advice]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM advice WHERE run_id = ?", (run_id,)).fetchall()
        out = {}
        for r in rows:
            a = Advice(**json.loads(r["payload"]))
            out[a.issue_id] = a
        return out

    def save_decisions(self, run_id: int, decisions: list[Decision]) -> None:
        now = utc_now_iso()
        rows = [(run_id, d.issue_id, json.dumps(d.model_dump()), now) for d in decisions]
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO decisions (run_id, issue_id, payload, decided_at)"
                " VALUES (?, ?, ?, ?)",
                rows,
            )

    def load_decisions(self, run_id: int) -> list[Decision]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM decisions WHERE run_id = ?", (run_id,)).fetchall()
        return [Decision(**json.loads(r["payload"])) for r in rows]

    def save_applied(self, run_id: int, applied: list[AppliedFix]) -> None:
        rows = [(run_id, json.dumps(a.model_dump())) for a in applied]
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM applied_fixes WHERE run_id = ?", (run_id,))
            conn.executemany("INSERT INTO applied_fixes (run_id, payload) VALUES (?, ?)", rows)

    def load_applied(self, run_id: int) -> list[AppliedFix]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM applied_fixes WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        return [AppliedFix(**json.loads(r["payload"])) for r in rows]

    # --- audit -------------------------------------------------------------
    def audit(self, run_id: int, actor: str, event: str, detail: str = "") -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO audit (run_id, at, actor, event, detail) VALUES (?, ?, ?, ?, ?)",
                (run_id, utc_now_iso(), actor, event, detail[:2000]),
            )

    def audit_log(self, run_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT at, actor, event, detail FROM audit WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def overview(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS runs,"
                " COALESCE(SUM(issues_found),0) AS issues,"
                " COALESCE(SUM(fixes_applied),0) AS fixes,"
                " COALESCE(AVG(CASE WHEN status='completed' THEN score_after - score_before END),0) AS lift"
                " FROM runs"
            ).fetchone()
        return {
            "runs": row["runs"],
            "issues_found": row["issues"],
            "fixes_applied": row["fixes"],
            "average_score_lift": round(float(row["lift"]), 1),
        }


def _row_to_run(row: sqlite3.Row) -> RunRecord:
    d = dict(row)
    return RunRecord(
        id=d["id"],
        filename=d["filename"],
        status=RunStatus(d["status"]),
        created_at=d["created_at"],
        updated_at=d.get("updated_at"),
        rows_before=d["rows_before"],
        rows_after=d["rows_after"],
        columns_before=d["columns_before"],
        columns_after=d["columns_after"],
        score_before=d["score_before"],
        score_after=d["score_after"],
        issues_found=d["issues_found"],
        fixes_applied=d["fixes_applied"],
        provider=d["provider"] or "",
        model=d["model"] or "",
        llm_calls=d["llm_calls"],
        prompt_tokens=d["prompt_tokens"],
        completion_tokens=d["completion_tokens"],
        advice_fallbacks=d["advice_fallbacks"],
        warnings=json.loads(d["warnings"] or "[]"),
        error=d.get("error"),
    )


def score_from_json(raw: str) -> QualityScore | None:
    try:
        data = json.loads(raw or "{}")
        return QualityScore(**data) if data else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
