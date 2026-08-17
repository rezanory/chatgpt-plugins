import pytest
from chatgpt_plugins_core import JOB_SCHEMA_V1, ComputeJobSpec


def valid_job():
    return {
        "schema": JOB_SCHEMA_V1,
        "job_id": "job-001",
        "source": {"repository": "owner/repo", "commit": "a" * 40},
        "tasks": [
            {"task_id": "t1", "profile": "python-tests", "account_id": "kg-01"},
            {"task_id": "t2", "profile": "python-tests", "account_id": "kg-02"},
        ],
        "repair_policy": {"enabled": True, "max_attempts": 2},
    }


def test_job_contract_is_canonical_and_hashable():
    job = ComputeJobSpec.from_dict(valid_job())
    assert job.integrity_hash() == job.integrity_hash()
    assert len(job.integrity_hash()) == 64


def test_job_contract_rejects_moving_ref_and_duplicate_parallel_account():
    value = valid_job()
    value["source"]["commit"] = "main"
    with pytest.raises(ValueError):
        ComputeJobSpec.from_dict(value)

    value = valid_job()
    value["tasks"][1]["account_id"] = "kg-01"
    with pytest.raises(ValueError):
        ComputeJobSpec.from_dict(value)
