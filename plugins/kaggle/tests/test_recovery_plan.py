import json
from pathlib import Path

import pytest

from chatgpt_plugin_kaggle.actions.plan_recovery import (
    RECOVERY_MARKER,
    RECOVERY_SCHEMA_V1,
    build_recovery_matrix,
)


def _accounts(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "account_id": "kg-01",
                    "owner_slug": "owner-one",
                    "secret_scope": "kaggle-01",
                    "enabled": True,
                    "max_parallel": 1,
                    "capabilities": ["cpu", "gpu"],
                },
                {
                    "account_id": "kg-02",
                    "owner_slug": "owner-two",
                    "secret_scope": "kaggle-02",
                    "enabled": True,
                    "max_parallel": 1,
                    "capabilities": ["cpu", "gpu"],
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


def _body(payload: dict) -> str:
    return f"{RECOVERY_MARKER}\n```json\n{json.dumps(payload)}\n```\n"


def test_recovery_maps_trusted_environment_and_escapes_artifact_filter(tmp_path: Path):
    accounts = _accounts(tmp_path / "accounts.json")
    payload = {
        "schema": RECOVERY_SCHEMA_V1,
        "recovery_id": "pneumonia-recovery-01",
        "artifact_names": ["KAGGLE_EXECUTION_V62_2", "fingerprint.json"],
        "runs": [
            {
                "task_id": "s01",
                "account_id": "kg-01",
                "kernel_ref": "owner-one/pneumonia-v6-2-2-s01",
            },
            {
                "task_id": "s02",
                "account_id": "kg-02",
                "kernel_ref": "owner-two/pneumonia-v6-2-2-s02",
            },
        ],
    }

    public, matrix = build_recovery_matrix(_body(payload), str(accounts))

    assert public["run_count"] == 2
    assert matrix[0]["account_environment"] == "kaggle-01"
    assert matrix[1]["account_environment"] == "kaggle-02"
    assert matrix[0]["kernel_ref"] == "owner-one/pneumonia-v6-2-2-s01"
    pattern = matrix[0]["file_pattern"]
    assert "KAGGLE_EXECUTION_V62_2" in pattern
    assert r"fingerprint\.json" in pattern


def test_recovery_rejects_kernel_owner_that_does_not_match_account(tmp_path: Path):
    accounts = _accounts(tmp_path / "accounts.json")
    payload = {
        "schema": RECOVERY_SCHEMA_V1,
        "recovery_id": "pneumonia-recovery-02",
        "runs": [
            {
                "task_id": "s01",
                "account_id": "kg-01",
                "kernel_ref": "owner-two/pneumonia-v6-2-2-s01",
            }
        ],
    }

    with pytest.raises(ValueError, match="does not match account"):
        build_recovery_matrix(_body(payload), str(accounts))
