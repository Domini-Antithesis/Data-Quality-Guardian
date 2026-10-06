"""The HTTP surface, including the approval boundary and the audit trail."""

from __future__ import annotations

import json
import os
import re
import time

import pytest
from fastapi.testclient import TestClient

from dq_agent.api.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    with TestClient(create_app(tmp_path, env_file=None)) as c:
        yield c


def wait_for(client, run_id, statuses=("awaiting_approval", "completed", "failed"), timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/runs/{run_id}").json()
        if payload["run"]["status"] in statuses:
            return payload
        time.sleep(0.05)
    raise AssertionError("run did not finish in time")


def analyse_sample(client, name="messy_sales_data.csv"):
    r = client.post(f"/api/samples/{name}/analyze")
    assert r.status_code == 200, r.text
    return wait_for(client, r.json()["run_id"])


def test_health_index_and_static(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["provider"] == "mock" and h["llm_ready"] is True
    assert "Data Quality Agent" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_strategy_help_covers_every_strategy(client):
    from dq_agent.domain import FixStrategy

    help_text = client.get("/api/strategies").json()
    assert {s.value for s in FixStrategy} == set(help_text)
    assert all(len(v) > 10 for v in help_text.values())


def test_settings_roundtrip_hides_secrets(client, tmp_path):
    body = client.put("/api/settings", json={"groq_api_key": "gsk_secret", "missing_warn_pct": 12}).json()
    assert body["groq_api_key_set"] is True and "groq_api_key" not in body
    assert body["missing_warn_pct"] == 12
    assert json.loads((tmp_path / "settings.json").read_text())["groq_api_key"] == "gsk_secret"
    client.put("/api/settings", json={"groq_api_key": ""})
    assert json.loads((tmp_path / "settings.json").read_text())["groq_api_key"] == "gsk_secret"
    assert client.put("/api/settings", json={"outlier_iqr_multiplier": 99}).status_code == 400


def test_analysis_stops_for_approval_and_changes_nothing(client):
    payload = analyse_sample(client)
    run = payload["run"]
    assert run["status"] == "awaiting_approval"
    assert run["issues_found"] == 11
    assert run["rows_after"] == 0, "nothing has been produced yet"
    assert payload["score_before"]["score"] < 100
    assert payload["score_after"] is None
    assert len(payload["issues"]) == 11
    assert all(i["advice"] for i in payload["issues"]), "every issue arrives with a recommendation"
    assert all(i["allowed_strategies"] for i in payload["issues"])
    # No cleaned file exists until fixes are approved.
    assert client.get(f"/api/runs/{run['id']}/download").status_code == 409


def test_approving_fixes_improves_the_score_and_records_everything(client):
    payload = analyse_sample(client)
    run_id = payload["run"]["id"]
    decisions = [
        {
            "issue_id": i["id"],
            "approved": i["advice"]["strategy"] != "none",
            "strategy": i["advice"]["strategy"],
            "fill_value": None,
        }
        for i in payload["issues"]
    ]
    result = client.post(f"/api/runs/{run_id}/apply", json={"decisions": decisions}).json()
    run = result["run"]
    assert run["status"] == "completed"
    assert run["score_after"] > run["score_before"]
    assert run["rows_after"] < run["rows_before"], "duplicates were removed"
    assert run["fixes_applied"] >= 5
    assert len(result["issues_after"]) < run["issues_found"]
    assert all(i["decision"] for i in result["issues"]), "every decision is stored"

    events = [a["event"] for a in result["audit"]]
    for expected in ("uploaded", "scanned", "recommended fixes", "approved fixes", "validated"):
        assert expected in events, events
    assert {a["actor"] for a in result["audit"]} >= {"human", "detector", "ai", "system"}

    assert client.get(f"/api/runs/{run_id}/download").status_code == 200
    report = client.get(f"/api/runs/{run_id}/audit.md").text
    assert "# Data quality audit report" in report and "Quality score" in report


def test_rejecting_everything_still_produces_a_file_but_no_changes(client):
    payload = analyse_sample(client)
    run_id = payload["run"]["id"]
    decisions = [
        {"issue_id": i["id"], "approved": False, "strategy": i["advice"]["strategy"], "fill_value": None}
        for i in payload["issues"]
    ]
    result = client.post(f"/api/runs/{run_id}/apply", json={"decisions": decisions}).json()
    assert result["run"]["fixes_applied"] == 0
    assert result["run"]["rows_after"] == result["run"]["rows_before"]
    assert result["run"]["score_after"] == result["run"]["score_before"]
    assert len(result["issues_after"]) == result["run"]["issues_found"]

    # Applying again, this time approving the fixes, must replace the list of
    # remaining issues rather than leave the first pass's entries beside it.
    approve = [
        {**d, "approved": d["strategy"] != "none"} for d in decisions
    ]
    again = client.post(f"/api/runs/{run_id}/apply", json={"decisions": approve}).json()
    assert len(again["issues_after"]) == again["score_after"]["issue_count"]
    assert len(again["issues_after"]) < len(result["issues_after"])


def test_a_fix_the_issue_does_not_allow_is_refused(client):
    payload = analyse_sample(client)
    run_id = payload["run"]["id"]
    dupes = next(i for i in payload["issues"] if i["issue_type"] == "duplicate_rows")
    result = client.post(
        f"/api/runs/{run_id}/apply",
        json={"decisions": [{"issue_id": dupes["id"], "approved": True, "strategy": "normalize_dates"}]},
    ).json()
    applied = result["applied"][0]
    assert not applied["applied"] and "not a valid fix" in applied["detail"]


def test_unknown_issue_and_unknown_strategy_are_rejected(client):
    payload = analyse_sample(client)
    run_id = payload["run"]["id"]
    real = payload["issues"][0]["id"]
    assert (
        client.post(
            f"/api/runs/{run_id}/apply",
            json={"decisions": [{"issue_id": "made-up", "approved": True, "strategy": "none"}]},
        ).status_code
        == 400
    )
    assert (
        client.post(
            f"/api/runs/{run_id}/apply",
            json={"decisions": [{"issue_id": real, "approved": True, "strategy": "delete_everything"}]},
        ).status_code
        == 400
    )


def test_preview_shows_original_then_cleaned(client):
    payload = analyse_sample(client)
    run_id = payload["run"]["id"]
    original = client.get(f"/api/runs/{run_id}/preview?which=original&rows=5").json()
    assert len(original["rows"]) == 5 and original["total_rows"] == 249
    assert client.get(f"/api/runs/{run_id}/preview?which=cleaned").status_code == 409

    dupes = next(i for i in payload["issues"] if i["issue_type"] == "duplicate_rows")
    client.post(
        f"/api/runs/{run_id}/apply",
        json={"decisions": [{"issue_id": dupes["id"], "approved": True, "strategy": "drop_duplicates"}]},
    )
    cleaned = client.get(f"/api/runs/{run_id}/preview?which=cleaned&rows=5").json()
    assert cleaned["total_rows"] == 240


def test_a_clean_file_reports_no_issues(client):
    payload = analyse_sample(client, "clean_orders.csv")
    assert payload["run"]["issues_found"] == 0
    assert payload["score_before"]["score"] == 100.0
    assert payload["issues"] == []


def test_bad_uploads_are_explained(client):
    assert (
        client.post("/api/analyze", files={"file": ("x.pdf", b"%PDF", "application/pdf")}).status_code == 400
    )
    assert client.post("/api/analyze", files={"file": ("empty.csv", b"", "text/csv")}).status_code == 400
    assert client.post("/api/samples/nope.csv/analyze").status_code == 404
    assert client.get("/api/runs/999").status_code == 404


def test_history_and_deletion(client):
    payload = analyse_sample(client)
    run_id = payload["run"]["id"]
    assert client.get("/api/overview").json()["runs"] == 1
    assert client.get("/api/runs").json()[0]["id"] == run_id
    assert client.delete(f"/api/runs/{run_id}").status_code == 200
    assert client.get(f"/api/runs/{run_id}").status_code == 404


def test_uploaded_file_is_kept_byte_for_byte(client, tmp_path, messy_csv):
    r = client.post("/api/analyze", files={"file": ("mine.csv", messy_csv.encode(), "text/csv")})
    run_id = r.json()["run_id"]
    wait_for(client, run_id)
    stored = next((tmp_path / "uploads" / str(run_id)).glob("original.*"))
    assert stored.read_bytes() == messy_csv.encode()


def test_importing_the_module_has_no_side_effects(tmp_path):
    """Importing the app must not create a database: the CLI sets DATA_DIR
    just before the server starts."""
    import subprocess
    import sys

    probe = tmp_path / "nope"
    result = subprocess.run(
        [sys.executable, "-c", "import dq_agent.api.app"],
        cwd=tmp_path,
        env={**os.environ, "DATA_DIR": str(probe)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not probe.exists() and not (tmp_path / "data").exists()


def test_audit_report_escapes_table_characters(client):
    """A pipe inside a value would otherwise break the Markdown table."""
    csv = b"id,note\n1,a|b\n2,\n3,\n4,\n5,\n6,\n"
    run_id = client.post("/api/analyze", files={"file": ("pipes.csv", csv, "text/csv")}).json()["run_id"]
    wait_for(client, run_id)
    report = client.get(f"/api/runs/{run_id}/audit.md").text
    for line in report.splitlines():
        if line.startswith("| 20") or line.startswith("| 2026"):
            assert not re.search(r"(?<!\\)\|.*(?<!\\)\|.*(?<!\\)\|.*(?<!\\)\|.*(?<!\\)\|", line[1:-1])
