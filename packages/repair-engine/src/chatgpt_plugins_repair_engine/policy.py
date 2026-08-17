from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from chatgpt_plugins_core import FailureCategory, FailureReport, JobSnapshot


class RepairAction(StrEnum):
    RETRY = "retry"
    REPAIR_SOURCE = "repair_source"
    WAIT = "wait"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class RepairDecision:
    action: RepairAction
    reason: str
    next_attempt: int | None = None


# V0.1 is intentionally conservative. OOM is repairable only via a reviewed, allow-listed patch;
# auth/quota/network/provider failures never trigger source modification.
_SOURCE_CATEGORIES = {
    FailureCategory.DEPENDENCY,
    FailureCategory.IMPORT_ERROR,
    FailureCategory.SYNTAX,
    FailureCategory.TEST_FAILURE,
    FailureCategory.DJANGO_CHECK,
    FailureCategory.DJANGO_MIGRATION,
    FailureCategory.OOM,
}
_WAIT_CATEGORIES = {
    FailureCategory.PROVIDER_QUOTA,
    FailureCategory.API_RATE_LIMIT,
}
_STOP_CATEGORIES = {
    FailureCategory.AUTHENTICATION,
    FailureCategory.POLICY,
}


def decide_repair(job: JobSnapshot, failure: FailureReport) -> RepairDecision:
    if failure.category in _WAIT_CATEGORIES:
        return RepairDecision(RepairAction.WAIT, "provider capacity/rate limit must recover first")
    if failure.category in _STOP_CATEGORIES:
        return RepairDecision(RepairAction.STOP, "operator/provider policy action is required")
    if job.repair_attempt >= job.max_repair_attempts:
        return RepairDecision(RepairAction.STOP, "repair attempt budget exhausted")
    next_attempt = job.repair_attempt + 1
    if failure.retryable and failure.category not in _SOURCE_CATEGORIES:
        return RepairDecision(
            RepairAction.RETRY, "failure is classified as transient", next_attempt
        )
    if failure.category in _SOURCE_CATEGORIES:
        return RepairDecision(
            RepairAction.REPAIR_SOURCE,
            "source-level failure may be proposed as a patch and must pass the repair gate",
            next_attempt,
        )
    return RepairDecision(RepairAction.STOP, "unknown/non-retryable failure requires review")
