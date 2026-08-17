import json
from pathlib import Path

from chatgpt_plugin_kaggle.actions.plan_issue import build_matrix
from chatgpt_plugins_core import JOB_SCHEMA_V1, ComputeJobSpec
from chatgpt_plugins_github_bridge import render_job_issue


def test_plan_maps_only_trusted_account_environment(tmp_path: Path):
    accounts = tmp_path / "accounts.json"
    accounts.write_text(
        json.dumps(
            [
                {
                    "account_id": "kg-01",
                    "owner_slug": "owner",
                    "secret_scope": "kaggle-01",
                    "capabilities": ["cpu", "gpu"],
                    "enabled": True,
                }
            ]
        )
    )
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "tests": {
                    "steps": [{"argv": ["python", "-m", "pytest"]}],
                    "capabilities": ["cpu"],
                }
            }
        )
    )
    job = ComputeJobSpec.from_dict(
        {
            "schema": JOB_SCHEMA_V1,
            "job_id": "job-1",
            "source": {"repository": "o/r", "commit": "a" * 40},
            "tasks": [{"task_id": "t1", "profile": "tests", "account_id": "kg-01"}],
        }
    )
    public, matrix = build_matrix(
        "[KAGGLE-JOB] test",
        render_job_issue(job),
        str(accounts),
        str(profiles),
    )
    assert public["job_id"] == "job-1"
    assert matrix[0]["account_environment"] == "kaggle-01"
    assert matrix[0]["kaggle_owner"] == "owner"
