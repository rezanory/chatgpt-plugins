from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from chatgpt_plugins_core import ComputeJobSpec

JOB_TITLE_PREFIX = "[KAGGLE-JOB]"
JOB_MARKER = "<!-- chatgpt-plugins-job:v1 -->"
RUN_MARKER = "<!-- chatgpt-plugins-run:v1 -->"
STATUS_MARKER = "<!-- chatgpt-plugins-status:v1 -->"
_JSON_FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.I | re.S)


@dataclass(frozen=True, slots=True)
class RunRecord:
    schema: str
    job_id: str
    task_id: str
    account_id: str
    account_environment: str
    kernel_ref: str
    dataset_ref: str
    source_repository: str
    source_commit: str
    profile: str
    state: str
    github_run_id: str
    integrity_hash: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunRecord:
        required = {
            "schema",
            "job_id",
            "task_id",
            "account_id",
            "account_environment",
            "kernel_ref",
            "dataset_ref",
            "source_repository",
            "source_commit",
            "profile",
            "state",
            "github_run_id",
            "integrity_hash",
        }
        if set(value) != required:
            missing = required - set(value)
            unknown = set(value) - required
            raise ValueError(f"invalid run record keys; missing={sorted(missing)} unknown={sorted(unknown)}")
        return cls(**{key: str(value[key]) for key in required})


@dataclass(frozen=True, slots=True)
class StatusRecord:
    schema: str
    job_id: str
    task_id: str
    account_id: str
    kernel_ref: str
    state: str
    failure_category: str | None = None
    failure_fingerprint: str | None = None
    summary: str = ""
    artifact_name: str | None = None
    github_run_id: str | None = None


def _extract_json(text: str, marker: str) -> dict[str, Any]:
    marker_pos = text.find(marker)
    if marker_pos < 0:
        raise ValueError(f"required marker not found: {marker}")
    match = _JSON_FENCE.search(text, marker_pos + len(marker))
    if not match:
        raise ValueError("JSON code fence not found after marker")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("protocol payload must be a JSON object")
    return payload


def parse_job_issue(title: str, body: str) -> ComputeJobSpec:
    if not title.startswith(JOB_TITLE_PREFIX):
        raise ValueError(f"issue title must start with {JOB_TITLE_PREFIX}")
    return ComputeJobSpec.from_dict(_extract_json(body or "", JOB_MARKER))


def render_job_issue(job: ComputeJobSpec, *, human_summary: str = "") -> str:
    summary = human_summary.strip()
    prefix = f"{summary}\n\n" if summary else ""
    payload = json.dumps(job.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
    return f"{prefix}{JOB_MARKER}\n```json\n{payload}\n```\n"


def render_run_comment(record: RunRecord, *, note: str = "Kaggle task submitted.") -> str:
    payload = json.dumps(asdict(record), indent=2, sort_keys=True)
    return f"{note}\n\n{RUN_MARKER}\n```json\n{payload}\n```\n"


def parse_run_comment(body: str) -> RunRecord:
    return RunRecord.from_dict(_extract_json(body or "", RUN_MARKER))


def render_status_comment(record: StatusRecord) -> str:
    payload = json.dumps(asdict(record), indent=2, sort_keys=True)
    return f"Kaggle status update: **{record.state}**\n\n{STATUS_MARKER}\n```json\n{payload}\n```\n"


def parse_status_comment(body: str) -> StatusRecord:
    payload = _extract_json(body or "", STATUS_MARKER)
    allowed = {
        "schema",
        "job_id",
        "task_id",
        "account_id",
        "kernel_ref",
        "state",
        "failure_category",
        "failure_fingerprint",
        "summary",
        "artifact_name",
        "github_run_id",
    }
    if set(payload) != allowed:
        raise ValueError("invalid status record keys")
    return StatusRecord(**payload)


def latest_runs_by_task(comments: Iterable[str], job_id: str) -> dict[str, RunRecord]:
    result: dict[str, RunRecord] = {}
    for body in comments:
        try:
            record = parse_run_comment(body)
        except (ValueError, json.JSONDecodeError, TypeError):
            continue
        if record.job_id == job_id:
            result[record.task_id] = record
    return result
