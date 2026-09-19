from __future__ import annotations

import pytest

import m07_runtime_contract as contract


def test_r384_first_attempt_does_not_require_state_attachment() -> None:
    assert contract.runtime_dataset_sources("M07_RUNTIME_M07_R384_A01") == [
        contract.RAW_DATASET_HANDLE,
        contract.PERSISTENCE_HANDLE,
    ]


def test_r384_continuation_attempt_attaches_exact_state_dataset() -> None:
    assert contract.runtime_dataset_sources("M07_RUNTIME_M07_R384_A03") == [
        contract.RAW_DATASET_HANDLE,
        contract.PERSISTENCE_HANDLE,
        contract.R384_STATE_HANDLE,
    ]


def test_r320_runtime_preserves_exact_bridge_attachment() -> None:
    assert contract.runtime_dataset_sources("M07_RUNTIME_M07_R320_A07") == [
        contract.RAW_DATASET_HANDLE,
        contract.PERSISTENCE_HANDLE,
        contract.R320_BRIDGE_DATASET,
    ]


def _ready_state() -> dict:
    return {
        "ref": contract.R384_STATE_HANDLE,
        "ownerRef": "trickermark",
        "currentVersionNumber": 1,
        "versions": [{"versionNumber": 1, "status": "Ready"}],
    }


def test_r384_state_dataset_gate_accepts_exact_ready_version() -> None:
    report = contract.validate_runtime_state_dataset(
        _ready_state(), token="M07_RUNTIME_M07_R384_A03"
    )
    assert report == {
        "status": "PASS",
        "required": True,
        "dataset_ref": contract.R384_STATE_HANDLE,
        "current_version_number": 1,
    }


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda d: d.update(ref="trickermark/wrong"), "ref mismatch"),
        (lambda d: d.update(currentVersionNumber=0), "no published version"),
        (lambda d: d["versions"][0].update(status="Creating"), "not Ready"),
    ],
)
def test_r384_state_dataset_gate_rejects_unusable_state(mutator, match: str) -> None:
    payload = _ready_state()
    mutator(payload)
    with pytest.raises(contract.ContractError, match=match):
        contract.validate_runtime_state_dataset(
            payload, token="M07_RUNTIME_M07_R384_A03"
        )
