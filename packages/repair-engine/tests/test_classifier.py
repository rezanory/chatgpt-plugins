from chatgpt_plugins_core import FailureCategory, JobSnapshot, JobStatus, SourceRef
from chatgpt_plugins_repair_engine import RepairAction, classify_failure, decide_repair


def job(max_attempts=2, attempt=0):
    return JobSnapshot(
        job_id="j",
        provider="kaggle",
        status=JobStatus.FAILED,
        source=SourceRef("o/r", "sha"),
        profile="tests",
        repair_attempt=attempt,
        max_repair_attempts=max_attempts,
    )


def test_oom_is_repairable_but_gated():
    failure = classify_failure("CUDA out of memory")
    assert failure.category == FailureCategory.OOM
    assert failure.fingerprint
    assert decide_repair(job(), failure).action == RepairAction.REPAIR_SOURCE


def test_import_error_requests_source_repair():
    failure = classify_failure("ModuleNotFoundError: No module named 'x'")
    assert failure.category == FailureCategory.IMPORT_ERROR
    assert decide_repair(job(), failure).action == RepairAction.REPAIR_SOURCE


def test_auth_never_requests_code_repair():
    failure = classify_failure("Unauthorized: invalid token")
    assert decide_repair(job(), failure).action == RepairAction.STOP


def test_attempt_budget_stops_loop():
    failure = classify_failure("Connection reset by peer")
    assert decide_repair(job(max_attempts=1, attempt=1), failure).action == RepairAction.STOP
