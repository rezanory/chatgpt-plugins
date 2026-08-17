from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

JOB_SCHEMA_V1 = "chatgpt.compute.job/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_COMMIT = re.compile(r"^[0-9a-fA-F]{40,64}$")


class JobStatus(StrEnum):
    QUEUED = "queued"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


TERMINAL_STATUSES = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.BLOCKED}
ACTIVE_STATUSES = {JobStatus.QUEUED, JobStatus.DISPATCHED, JobStatus.RUNNING}


class FailureCategory(StrEnum):
    DEPENDENCY = "dependency"
    IMPORT_ERROR = "import_error"
    SYNTAX = "syntax"
    TEST_FAILURE = "test_failure"
    DJANGO_CHECK = "django_check"
    DJANGO_MIGRATION = "django_migration"
    OOM = "oom"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    PROVIDER_QUOTA = "provider_quota"
    API_RATE_LIMIT = "api_rate_limit"
    TRANSPORT = "transport"
    NETWORK = "network"
    INFRASTRUCTURE = "infrastructure"
    POLICY = "policy"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class AccountDescriptor:
    account_id: str
    provider: str
    secret_scope: str
    enabled: bool = True
    max_parallel: int = 1
    capabilities: frozenset[str] = frozenset({"cpu"})
    labels: frozenset[str] = frozenset()
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.account_id):
            raise ValueError("account_id contains unsupported characters")
        if not self.provider.strip():
            raise ValueError("provider must not be empty")
        if not self.secret_scope.strip():
            raise ValueError("secret_scope must not be empty")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        if self.weight <= 0:
            raise ValueError("weight must be > 0")


@dataclass(frozen=True, slots=True)
class SourceRef:
    repository: str
    ref: str

    def __post_init__(self) -> None:
        parts = self.repository.split("/")
        if len(parts) != 2 or not all(_SAFE_ID.fullmatch(part) for part in parts):
            raise ValueError("repository must be in safe owner/name form")
        if not self.ref.strip():
            raise ValueError("ref must not be empty")


@dataclass(frozen=True, slots=True)
class ImmutableSourceRef:
    """GitHub source used by compute jobs.

    V0.1 intentionally requires an immutable commit-like hash instead of a branch name so every
    external run is reproducible and cannot silently move underneath a queued job.
    """

    repository: str
    commit: str

    def __post_init__(self) -> None:
        SourceRef(self.repository, self.commit)
        if not _COMMIT.fullmatch(self.commit):
            raise ValueError("commit must be a full hexadecimal commit id (40-64 chars)")


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    profile: str
    account_id: str
    accelerator: str | None = None
    parameters: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.task_id):
            raise ValueError("task_id contains unsupported characters")
        if not _SAFE_ID.fullmatch(self.profile):
            raise ValueError("profile contains unsupported characters")
        if not _SAFE_ID.fullmatch(self.account_id):
            raise ValueError("account_id contains unsupported characters")
        if self.accelerator is not None and not _SAFE_ID.fullmatch(self.accelerator):
            raise ValueError("accelerator contains unsupported characters")
        if len(self.parameters) > 64:
            raise ValueError("task parameters exceed V0.1 limit")
        for key, value in self.parameters.items():
            if not _SAFE_ID.fullmatch(str(key)):
                raise ValueError(f"unsupported parameter key: {key!r}")
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"unsupported parameter type for {key!r}")
            if isinstance(value, str) and len(value) > 500:
                raise ValueError(f"parameter {key!r} is too long")


@dataclass(frozen=True, slots=True)
class RepairPolicy:
    enabled: bool = True
    max_attempts: int = 2

    def __post_init__(self) -> None:
        if not 0 <= self.max_attempts <= 5:
            raise ValueError("max_attempts must be between 0 and 5")


@dataclass(frozen=True, slots=True)
class ComputeJobSpec:
    schema: str
    job_id: str
    source: ImmutableSourceRef
    tasks: tuple[TaskSpec, ...]
    repair_policy: RepairPolicy = RepairPolicy()
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != JOB_SCHEMA_V1:
            raise ValueError(f"unsupported schema: {self.schema!r}")
        if not _SAFE_ID.fullmatch(self.job_id):
            raise ValueError("job_id contains unsupported characters")
        if not self.tasks or len(self.tasks) > 16:
            raise ValueError("V0.1 requires between 1 and 16 tasks")
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task_id values must be unique")
        account_ids = [task.account_id for task in self.tasks]
        if len(account_ids) != len(set(account_ids)):
            raise ValueError(
                "V0.1 parallel issues require unique account_id values; "
                "split sequential work into later jobs"
            )
        if len(self.metadata) > 32:
            raise ValueError("metadata exceeds V0.1 limit")
        for key, value in self.metadata.items():
            if not _SAFE_ID.fullmatch(key):
                raise ValueError(f"unsupported metadata key: {key!r}")
            if len(value) > 1000:
                raise ValueError(f"metadata value too long for {key!r}")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ComputeJobSpec:
        if not isinstance(value, dict):
            raise ValueError("job envelope must be an object")
        allowed = {"schema", "job_id", "source", "tasks", "repair_policy", "metadata"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown job fields: {sorted(unknown)}")
        source = value.get("source")
        if not isinstance(source, dict) or set(source) != {"repository", "commit"}:
            raise ValueError("source must contain exactly repository and commit")
        tasks_raw = value.get("tasks")
        if not isinstance(tasks_raw, list):
            raise ValueError("tasks must be a list")
        tasks: list[TaskSpec] = []
        for item in tasks_raw:
            if not isinstance(item, dict):
                raise ValueError("each task must be an object")
            allowed_task = {"task_id", "profile", "account_id", "accelerator", "parameters"}
            unknown_task = set(item) - allowed_task
            if unknown_task:
                raise ValueError(f"unknown task fields: {sorted(unknown_task)}")
            tasks.append(
                TaskSpec(
                    task_id=str(item["task_id"]),
                    profile=str(item["profile"]),
                    account_id=str(item["account_id"]),
                    accelerator=(str(item["accelerator"]) if item.get("accelerator") else None),
                    parameters=dict(item.get("parameters") or {}),
                )
            )
        repair_raw = value.get("repair_policy") or {}
        if not isinstance(repair_raw, dict):
            raise ValueError("repair_policy must be an object")
        if set(repair_raw) - {"enabled", "max_attempts"}:
            raise ValueError("repair_policy contains unsupported fields")
        metadata = value.get("metadata") or {}
        if not isinstance(metadata, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
        ):
            raise ValueError("metadata must be a string-to-string object")
        return cls(
            schema=str(value.get("schema", "")),
            job_id=str(value.get("job_id", "")),
            source=ImmutableSourceRef(str(source["repository"]), str(source["commit"])),
            tasks=tuple(tasks),
            repair_policy=RepairPolicy(
                enabled=bool(repair_raw.get("enabled", True)),
                max_attempts=int(repair_raw.get("max_attempts", 2)),
            ),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def integrity_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class JobRequest:
    """Legacy/provider-neutral scheduler request retained for the direct-MCP transport."""

    job_id: str
    source: SourceRef
    profile: str
    required_capabilities: frozenset[str] = frozenset({"cpu"})
    required_labels: frozenset[str] = frozenset()
    preferred_account_ids: tuple[str, ...] = ()
    max_repair_attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("job_id must not be empty")
        if not self.profile.strip():
            raise ValueError("profile must not be empty")
        if self.max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be >= 0")


@dataclass(frozen=True, slots=True)
class FailureReport:
    category: FailureCategory
    summary: str
    log_excerpt: str = ""
    retryable: bool = False
    source_paths: tuple[str, ...] = ()
    confidence: float = 1.0
    fingerprint: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    job_id: str
    provider: str
    status: JobStatus
    source: SourceRef
    profile: str
    account_id: str | None = None
    execution_id: str | None = None
    resolved_commit_sha: str | None = None
    dataset_ref: str | None = None
    kernel_ref: str | None = None
    repair_attempt: int = 0
    max_repair_attempts: int = 0
    failure: FailureReport | None = None
    artifacts: tuple[str, ...] = ()
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
