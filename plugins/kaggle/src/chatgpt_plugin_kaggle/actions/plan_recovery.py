from __future__ import annotations

import argparse
import json
import re

from ..config import load_accounts
from .common import load_github_event, write_github_output

RECOVERY_SCHEMA_V1 = "chatgpt.compute.recovery/v1"
RECOVERY_MARKER = "<!-- chatgpt-plugins-recovery:v1 -->"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_SAFE_KERNEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_SAFE_ARTIFACT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
_JSON_FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.I | re.S)
_DEFAULT_PATTERN = r".*(result\.json|job\.log)$"


def _extract_payload(body: str) -> dict:
    marker_pos = body.find(RECOVERY_MARKER)
    if marker_pos < 0:
        raise ValueError(f"required marker not found: {RECOVERY_MARKER}")
    match = _JSON_FENCE.search(body, marker_pos + len(RECOVERY_MARKER))
    if not match:
        raise ValueError("JSON code fence not found after recovery marker")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("recovery payload must be an object")
    return payload


def build_recovery_matrix(body: str, accounts_file: str) -> tuple[dict, list[dict]]:
    payload = _extract_payload(body or "")
    allowed = {"schema", "recovery_id", "runs", "artifact_names"}
    if set(payload) - allowed:
        raise ValueError("recovery payload contains unsupported fields")
    if payload.get("schema") != RECOVERY_SCHEMA_V1:
        raise ValueError("unsupported recovery schema")

    recovery_id = str(payload.get("recovery_id") or "")
    if not _SAFE_ID.fullmatch(recovery_id):
        raise ValueError("recovery_id contains unsupported characters")

    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or not 1 <= len(raw_runs) <= 16:
        raise ValueError("recovery requires between 1 and 16 runs")

    artifact_names = payload.get("artifact_names") or []
    if not isinstance(artifact_names, list) or len(artifact_names) > 10:
        raise ValueError("artifact_names must be a list with at most 10 entries")
    normalized_artifacts: list[str] = []
    for value in artifact_names:
        name = str(value)
        if not _SAFE_ARTIFACT.fullmatch(name):
            raise ValueError(f"unsafe recovery artifact name: {name!r}")
        normalized_artifacts.append(name)
    if normalized_artifacts:
        alternation = "|".join(re.escape(name) for name in normalized_artifacts)
        file_pattern = rf".*(?:{alternation}).*"
    else:
        file_pattern = _DEFAULT_PATTERN

    accounts = {cfg.descriptor.account_id: cfg for cfg in load_accounts(accounts_file)}
    matrix: list[dict] = []
    seen_tasks: set[str] = set()
    seen_accounts: set[str] = set()
    for raw in raw_runs:
        required = {"task_id", "account_id", "kernel_ref"}
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError(
                "each recovery run must contain task_id, account_id, and kernel_ref"
            )
        task_id = str(raw["task_id"])
        account_id = str(raw["account_id"])
        kernel_ref = str(raw["kernel_ref"])
        if not _SAFE_ID.fullmatch(task_id):
            raise ValueError(f"unsafe recovery task_id: {task_id!r}")
        if task_id in seen_tasks:
            raise ValueError(f"duplicate recovery task_id: {task_id}")
        if account_id in seen_accounts:
            raise ValueError(f"duplicate recovery account_id: {account_id}")
        seen_tasks.add(task_id)
        seen_accounts.add(account_id)

        account = accounts.get(account_id)
        if account is None or not account.descriptor.enabled:
            raise ValueError(f"recovery account {account_id!r} is unknown or disabled")
        parts = kernel_ref.split("/", 1)
        if len(parts) != 2 or not _SAFE_KERNEL.fullmatch(parts[1]):
            raise ValueError(f"invalid kernel_ref: {kernel_ref!r}")
        if parts[0].casefold() != account.owner_slug.casefold():
            raise ValueError(
                f"kernel owner {parts[0]!r} does not match account "
                f"{account_id!r} owner metadata"
            )
        matrix.append(
            {
                "recovery_id": recovery_id,
                "task_id": task_id,
                "account_id": account_id,
                "account_environment": account.descriptor.secret_scope,
                "kernel_ref": kernel_ref,
                "file_pattern": file_pattern,
            }
        )

    public = {
        "schema": RECOVERY_SCHEMA_V1,
        "recovery_id": recovery_id,
        "run_count": len(matrix),
        "artifact_names": normalized_artifacts,
    }
    return public, matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts-file", required=True)
    args = parser.parse_args(argv)
    event = load_github_event()
    issue = event.get("issue") or {}
    title = str(issue.get("title") or "")
    if not title.startswith("[KAGGLE-RECOVER]"):
        raise ValueError("recovery issue title must start with [KAGGLE-RECOVER]")
    public, matrix = build_recovery_matrix(
        str(issue.get("body") or ""),
        args.accounts_file,
    )
    write_github_output("recovery", json.dumps(public, sort_keys=True))
    write_github_output(
        "matrix",
        json.dumps({"include": matrix}, separators=(",", ":")),
    )
    print(json.dumps(public, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
