from __future__ import annotations

import argparse
import json

from chatgpt_plugins_github_bridge import parse_job_issue

from ..config import load_accounts, load_profiles, load_repo_policy
from .common import load_github_event, write_github_output


def build_matrix(title: str, body: str, accounts_file: str, profiles_file: str, repo_policy_file: str | None = None) -> tuple[dict, list[dict]]:
    job = parse_job_issue(title, body)
    if repo_policy_file:
        load_repo_policy(repo_policy_file).require(job.source.repository)
    accounts = {cfg.descriptor.account_id: cfg for cfg in load_accounts(accounts_file)}
    profiles = load_profiles(profiles_file)
    matrix: list[dict] = []
    for task in job.tasks:
        account = accounts.get(task.account_id)
        if account is None or not account.descriptor.enabled:
            raise ValueError(f"account {task.account_id!r} is unknown or disabled")
        profile = profiles.get(task.profile)
        if profile is None:
            raise ValueError(f"profile {task.profile!r} is unknown")
        if not profile.capabilities.issubset(account.descriptor.capabilities):
            raise ValueError(
                f"account {task.account_id!r} lacks capabilities required by profile {task.profile!r}"
            )
        accelerator = task.accelerator or account.default_accelerator or ""
        matrix.append(
            {
                "job_id": job.job_id,
                "task_id": task.task_id,
                "profile": task.profile,
                "account_id": account.descriptor.account_id,
                "account_environment": account.descriptor.secret_scope,
                "kaggle_owner": account.owner_slug,
                "accelerator": accelerator,
                "source_repository": job.source.repository,
                "source_commit": job.source.commit,
                "parameters_json": json.dumps(task.parameters, sort_keys=True, separators=(",", ":")),
                "job_integrity_hash": job.integrity_hash(),
            }
        )
    public = {
        "job_id": job.job_id,
        "source_repository": job.source.repository,
        "source_commit": job.source.commit,
        "task_count": len(job.tasks),
        "integrity_hash": job.integrity_hash(),
    }
    return public, matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts-file", required=True)
    parser.add_argument("--profiles-file", required=True)
    parser.add_argument("--repo-policy-file", required=True)
    args = parser.parse_args(argv)
    event = load_github_event()
    issue = event.get("issue") or {}
    public, matrix = build_matrix(
        str(issue.get("title") or ""),
        str(issue.get("body") or ""),
        args.accounts_file,
        args.profiles_file,
        args.repo_policy_file,
    )
    write_github_output("job", json.dumps(public, sort_keys=True))
    write_github_output("matrix", json.dumps({"include": matrix}, separators=(",", ":")))
    print(json.dumps(public, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
