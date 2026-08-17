from chatgpt_plugins_core import JOB_SCHEMA_V1, ComputeJobSpec
from chatgpt_plugins_github_bridge import (
    RunRecord,
    latest_runs_by_task,
    parse_job_issue,
    parse_run_comment,
    render_job_issue,
    render_run_comment,
)


def spec():
    return ComputeJobSpec.from_dict(
        {
            "schema": JOB_SCHEMA_V1,
            "job_id": "job-1",
            "source": {"repository": "o/r", "commit": "b" * 40},
            "tasks": [{"task_id": "t1", "profile": "tests", "account_id": "kg-01"}],
        }
    )


def test_issue_round_trip():
    job = spec()
    body = render_job_issue(job, human_summary="Run this.")
    parsed = parse_job_issue("[KAGGLE-JOB] test", body)
    assert parsed == job


def test_run_record_round_trip_and_latest():
    record = RunRecord(
        schema="chatgpt.compute.run/v1",
        job_id="job-1",
        task_id="t1",
        account_id="kg-01",
        account_environment="kaggle-01",
        kernel_ref="owner/kernel",
        dataset_ref="owner/dataset",
        source_repository="o/r",
        source_commit="b" * 40,
        profile="tests",
        state="submitted",
        github_run_id="123",
        integrity_hash="c" * 64,
    )
    body = render_run_comment(record)
    assert parse_run_comment(body) == record
    assert latest_runs_by_task(["noise", body], "job-1")["t1"] == record
