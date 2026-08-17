from __future__ import annotations

import argparse
import json
import os

from chatgpt_plugins_github_bridge import (
    GitHubIssueClient,
    latest_runs_by_task,
    parse_job_issue,
    parse_status_comment,
)

from ..config import load_accounts
from .common import load_github_event, write_github_output

_TERMINAL = {"succeeded", "failed", "cancelled", "blocked"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts-file", required=True)
    args = parser.parse_args(argv)
    event = load_github_event()
    issue = event.get("issue") or {}
    issue_number = int(issue["number"])
    job = parse_job_issue(str(issue.get("title") or ""), str(issue.get("body") or ""))
    repository = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    client = GitHubIssueClient(repository, token)
    comments = client.list_comments(issue_number)
    bodies = [comment.body for comment in comments]
    runs = latest_runs_by_task(bodies, job.job_id)

    latest_status: dict[str, str] = {}
    for body in bodies:
        try:
            status = parse_status_comment(body)
        except (ValueError, json.JSONDecodeError, TypeError):
            continue
        if status.job_id == job.job_id:
            latest_status[status.task_id] = status.state

    accounts = {cfg.descriptor.account_id: cfg for cfg in load_accounts(args.accounts_file)}
    matrix: list[dict] = []
    missing: list[str] = []
    for task in job.tasks:
        run = runs.get(task.task_id)
        if run is None:
            missing.append(task.task_id)
            continue
        if latest_status.get(task.task_id) in _TERMINAL:
            continue
        account = accounts.get(run.account_id)
        if account is None or not account.descriptor.enabled:
            raise ValueError(f"run references unknown/disabled account: {run.account_id}")
        if account.descriptor.secret_scope != run.account_environment:
            raise ValueError(
                "run record account environment does not match trusted account registry"
            )
        if run.integrity_hash != job.integrity_hash():
            raise ValueError("job Issue body changed after submission; integrity hash mismatch")
        matrix.append(
            {
                "job_id": run.job_id,
                "task_id": run.task_id,
                "account_id": run.account_id,
                "account_environment": run.account_environment,
                "kernel_ref": run.kernel_ref,
                "source_commit": run.source_commit,
                "github_run_id": run.github_run_id,
            }
        )

    write_github_output("matrix", json.dumps({"include": matrix}, separators=(",", ":")))
    write_github_output("has_tasks", "true" if matrix else "false")
    write_github_output("missing_tasks", json.dumps(missing))
    print(
        json.dumps(
            {"job_id": job.job_id, "active_tasks": len(matrix), "missing_tasks": missing}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
