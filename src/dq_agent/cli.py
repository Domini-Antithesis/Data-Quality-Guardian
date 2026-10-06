"""Command line entry points.

dq-agent serve                  # start the web app
dq-agent scan data.csv          # detect issues and print a report
dq-agent clean data.csv -o out.csv   # detect, apply the recommended fixes, save
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from dq_agent.config import load_settings


def find_free_port(host: str, preferred: int, attempts: int = 20) -> int:
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"No free port found between {preferred} and {preferred + attempts - 1}.")


def _open_browser_when_ready(url: str, timeout: float = 30.0) -> None:
    def _wait() -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{url}/api/health", timeout=2):
                    webbrowser.open(url)
                    return
            except OSError:
                time.sleep(0.5)

    threading.Thread(target=_wait, daemon=True).start()


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = load_settings(args.data_dir)
    host = args.host or settings.app_host
    wanted = args.port or settings.app_port
    try:
        port = find_free_port(host, wanted)
    except OSError as exc:
        print(f"Cannot start: {exc} Set another port with --port or APP_PORT.", file=sys.stderr)
        return 1
    if port != wanted:
        print(f"Port {wanted} is already in use by another program, so using port {port} instead.")
    if args.data_dir:
        os.environ["DATA_DIR"] = str(Path(args.data_dir).resolve())
    url = f"http://{host}:{port}"
    print(f"Data Quality Agent -> {url}")
    if args.open:
        _open_browser_when_ready(url)
    uvicorn.run("dq_agent.api.app:get_app", factory=True, host=host, port=port, reload=False)
    return 0


def _report(df_name: str, issues, score) -> None:
    print(f"\n{df_name}: quality score {score.score}/100 ({score.grade})")
    print(
        f"  completeness {score.completeness}  uniqueness {score.uniqueness}  "
        f"validity {score.validity}  consistency {score.consistency}"
    )
    if not issues:
        print("  No issues found.")
        return
    print(f"\n  {len(issues)} issues:")
    for i in issues:
        print(f"   [{i.severity.value:8}] {i.title}  ({i.affected_rows} rows, {i.affected_pct}%)")
        print(f"              {i.detail}")


def _cmd_scan(args: argparse.Namespace) -> int:
    from dq_agent.detectors.rules import detect_all
    from dq_agent.detectors.scoring import score_dataset
    from dq_agent.sources import load_dataframe

    settings = load_settings(args.data_dir)
    path = Path(args.file)
    df = load_dataframe(path.name, path.read_bytes())
    issues = detect_all(df, settings)
    _report(path.name, issues, score_dataset(issues))
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    from dq_agent.detectors.rules import detect_all
    from dq_agent.detectors.scoring import score_dataset
    from dq_agent.domain import Decision
    from dq_agent.fixes.executor import apply_all
    from dq_agent.llm.registry import build_advisor
    from dq_agent.sources import load_dataframe

    settings = load_settings(args.data_dir)
    if args.provider:
        settings = settings.model_copy(update={"llm_provider": args.provider})
    path = Path(args.file)
    df = load_dataframe(path.name, path.read_bytes())
    issues = detect_all(df, settings)
    before = score_dataset(issues)
    _report(path.name, issues, before)
    if not issues:
        return 0

    advisor = build_advisor(settings)
    print(f"\nAsking {advisor.name} ({advisor.model}) to recommend fixes...")
    advice = {a.issue_id: a for a in advisor.advise(issues, f"{path.name}, {len(df)} rows").advice}

    decisions = [
        Decision(
            issue_id=i.id,
            approved=advice[i.id].strategy.value != "none",
            strategy=advice[i.id].strategy,
            fill_value=advice[i.id].fill_value,
        )
        for i in issues
        if i.id in advice
    ]
    cleaned, applied = apply_all(df, issues, decisions)
    print("\nApplied:")
    for fix in applied:
        print(f"   {'OK  ' if fix.applied else 'skip'} {fix.strategy.value:26} {fix.detail}")

    after_issues = detect_all(cleaned, settings)
    after = score_dataset(after_issues)
    print(f"\nQuality score {before.score} -> {after.score} ({after.grade})")
    print(f"Rows {len(df):,} -> {len(cleaned):,}   Columns {len(df.columns)} -> {len(cleaned.columns)}")
    if args.output:
        out = Path(args.output)
        cleaned.to_csv(out, index=False)
        print(f"Wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dq-agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default=None, help="Where the database and settings live")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start the web app")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--open", action="store_true", help="Open a browser once it is up")
    serve.set_defaults(func=_cmd_serve)

    scan = sub.add_parser("scan", help="Detect issues and print a report (no LLM, no changes)")
    scan.add_argument("file")
    scan.set_defaults(func=_cmd_scan)

    clean = sub.add_parser("clean", help="Detect, apply the recommended fixes, and save the result")
    clean.add_argument("file")
    clean.add_argument("-o", "--output", default=None, help="Where to write the cleaned CSV")
    clean.add_argument("--provider", default=None, choices=["mock", "groq", "gemini", "openai", "ollama"])
    clean.set_defaults(func=_cmd_clean)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
