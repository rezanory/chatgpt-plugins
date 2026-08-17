from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .api_pool import KaggleApiPool, _jsonable

CONTROL_MARKER = "<!-- chatgpt-plugins-kaggle-control:v1 -->"
CLAIM_MARKER = "<!-- chatgpt-plugins-kaggle-control-claim:v1"
RECEIPT_MARKER = "<!-- chatgpt-plugins-kaggle-control-receipt:v1"
FAILURE_MARKER = "<!-- chatgpt-plugins-kaggle-control-failure:v1"
CONTROL_SCHEMA = "chatgpt.kaggle.control/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_KERNEL_REF = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


@dataclass(frozen=True, slots=True)
class ControlCommand:
    schema: str
    action: str
    job_id: str
    account_id: str
    kernel_ref: str

    @classmethod
    def from_issue(cls, title: str, body: str) -> ControlCommand:
        if not title.startswith("[KAGGLE-RUN]"):
            raise ValueError("control issue title must start with [KAGGLE-RUN]")
        if len(body) > 50_000 or CONTROL_MARKER not in body:
            raise ValueError("control issue marker is missing or body is too large")
        payload_text = body.split(CONTROL_MARKER, 1)[1].strip()
        if payload_text.startswith("```json"):
            payload_text = payload_text[7:].strip()
            if payload_text.endswith("```"):
                payload_text = payload_text[:-3].strip()
        payload = json.loads(payload_text)
        if not isinstance(payload, dict):
            raise ValueError("control payload must be a JSON object")
        allowed = {"schema", "action", "job_id", "account_id", "kernel_ref"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unsupported control fields: {sorted(unknown)}")
        command = cls(
            schema=str(payload.get("schema", "")),
            action=str(payload.get("action", "")),
            job_id=str(payload.get("job_id", "")),
            account_id=str(payload.get("account_id", "")),
            kernel_ref=str(payload.get("kernel_ref", "")),
        )
        command.validate()
        return command

    def validate(self) -> None:
        if self.schema != CONTROL_SCHEMA:
            raise ValueError("unsupported control schema")
        if self.action != "rerun_existing":
            raise ValueError("V0.1 control only supports rerun_existing")
        if not _SAFE_ID.fullmatch(self.job_id):
            raise ValueError("invalid job_id")
        if not _SAFE_ID.fullmatch(self.account_id):
            raise ValueError("invalid account_id")
        if not _KERNEL_REF.fullmatch(self.kernel_ref):
            raise ValueError("kernel_ref must use owner/slug form")


def verify_github_signature(secret: str, body: bytes, signature: str) -> bool:
    if not secret or not signature.startswith("sha256="):
        return False
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={digest}", signature)


def _github_api(path: str, token: str, *, method: str = "GET", payload: Any = None) -> Any:
    url = f"https://api.github.com{path}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "chatgpt-plugins-kaggle-gateway/0.1",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=15) as response:  # noqa: S310 - fixed GitHub API origin
        return json.loads(response.read().decode("utf-8") or "null")


class GitHubIssueJournal:
    def __init__(self, repository: str, token: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("invalid control repository")
        if not token:
            raise RuntimeError("CGP_GITHUB_TOKEN is required for write control")
        self.repository = repository
        self.token = token

    def comments(self, issue_number: int) -> list[dict[str, Any]]:
        value = _github_api(
            f"/repos/{self.repository}/issues/{issue_number}/comments?per_page=100",
            self.token,
        )
        return value if isinstance(value, list) else []

    def has_job_marker(self, issue_number: int, job_id: str) -> bool:
        needles = (
            f"{CLAIM_MARKER} job_id={job_id} -->",
            f"{RECEIPT_MARKER} job_id={job_id} -->",
        )
        for comment in self.comments(issue_number):
            body = str(comment.get("body") or "")
            if any(needle in body for needle in needles):
                return True
        return False

    def comment(self, issue_number: int, body: str) -> None:
        _github_api(
            f"/repos/{self.repository}/issues/{issue_number}/comments",
            self.token,
            method="POST",
            payload={"body": body[:60_000]},
        )


def rerun_existing(pool: KaggleApiPool, command: ControlCommand) -> dict[str, Any]:
    pool._validate_kernel_owner(command.account_id, command.kernel_ref)
    slot = pool._slot(command.account_id)
    with tempfile.TemporaryDirectory(prefix=f"cgp-rerun-{command.job_id}-") as directory:
        folder = Path(directory)
        with slot.lock:
            slot.api.kernels_pull(command.kernel_ref, path=str(folder), metadata=True, quiet=True)
            metadata_path = folder / "kernel-metadata.json"
            if not metadata_path.is_file():
                raise RuntimeError("Kaggle pull did not return kernel-metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            pulled_id = str(metadata.get("id") or "")
            if pulled_id and pulled_id.casefold() != command.kernel_ref.casefold():
                raise RuntimeError("pulled kernel identity does not match requested kernel_ref")
            result = slot.api.kernels_push(str(folder))
    return _jsonable(result)


def _receipt_result(result: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "ref",
        "url",
        "version_number",
        "versionNumber",
        "error",
        "status",
    }
    return {key: result[key] for key in allowed if key in result}


class ControlDispatcher:
    def __init__(self, pool: KaggleApiPool) -> None:
        self.pool = pool
        self.repository = os.getenv("CGP_CONTROL_REPOSITORY", "rezanory/chatgpt-plugins")
        self.journal = GitHubIssueJournal(self.repository, os.getenv("CGP_GITHUB_TOKEN", ""))
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def accept(self, issue_number: int, command: ControlCommand) -> bool:
        with self._lock:
            if command.job_id in self._active:
                return False
            if self.journal.has_job_marker(issue_number, command.job_id):
                return False
            self._active.add(command.job_id)
            self.journal.comment(
                issue_number,
                f"{CLAIM_MARKER} job_id={command.job_id} -->\n"
                f"Accepted `{command.action}` for `{command.account_id}` / "
                f"`{command.kernel_ref}`. Execution uses direct `KaggleApi()`.",
            )
        thread = threading.Thread(
            target=self._run,
            args=(issue_number, command),
            daemon=True,
            name=f"kaggle-control-{command.job_id}",
        )
        thread.start()
        return True

    def _run(self, issue_number: int, command: ControlCommand) -> None:
        try:
            result = rerun_existing(self.pool, command)
            receipt = {
                "schema": "chatgpt.kaggle.control.receipt/v1",
                "job_id": command.job_id,
                "status": "submitted",
                "account_id": command.account_id,
                "kernel_ref": command.kernel_ref,
                "provider_result": _receipt_result(result),
            }
            self.journal.comment(
                issue_number,
                f"{RECEIPT_MARKER} job_id={command.job_id} -->\n"
                "Direct Kaggle API submission accepted.\n\n"
                f"```json\n{json.dumps(receipt, sort_keys=True)}\n```",
            )
        except BaseException as exc:
            safe = self.pool.safe_error(command.account_id, exc)
            failure = {
                "schema": "chatgpt.kaggle.control.failure/v1",
                "job_id": command.job_id,
                "status": "failed",
                "account_id": command.account_id,
                "kernel_ref": command.kernel_ref,
                "error_type": exc.__class__.__name__,
                "error": safe,
            }
            with contextlib.suppress(Exception):
                self.journal.comment(
                    issue_number,
                    f"{FAILURE_MARKER} job_id={command.job_id} -->\n"
                    f"```json\n{json.dumps(failure, sort_keys=True)}\n```",
                )
        finally:
            with self._lock:
                self._active.discard(command.job_id)
