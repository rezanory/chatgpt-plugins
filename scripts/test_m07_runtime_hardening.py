from __future__ import annotations

import json
from pathlib import Path

import pytest

import m07_runtime_contract as contract


FRAGMENT = Path(__file__).with_name("m07_r320_bridge_extracted_compat.pyfrag")


def _load_fragment_namespace() -> dict:
    namespace = {
        "Path": Path,
        "json": json,
        "CGP_RUNTIME_SOURCE_STATE_HANDLE": contract.R320_BRIDGE_HANDLE,
    }
    exec(compile(FRAGMENT.read_text(encoding="utf-8"), str(FRAGMENT), "exec"), namespace)
    return namespace


def _bridge_receipt() -> dict:
    return {
        "schema": "m07.r320.producer.bridge.v2",
        "status": "COMPLETE",
        "target_dataset": "trickermark/m07-r320-producer-v4-bridge-e3884dd1",
        "campaign_receipt_sha256": contract._M07_R320_CAMPAIGN_RECEIPT
        if hasattr(contract, "_M07_R320_CAMPAIGN_RECEIPT")
        else "e3884dd10e323d2f0925660599a420964781111d850664e8c88a830324460c77",
        "split_fingerprint": contract.SPLIT_FINGERPRINT,
        "recipe_fingerprint": contract.RECIPE_FINGERPRINT,
        "sealed_fold_ids": [1, 2, 3, 4],
        "missing_fold_ids": [5],
        "locked_test_used_for_training": False,
        "external_used_for_training": False,
        "training_performed": False,
        "mutable_dataset_versions_5_6_used": False,
    }


def _write_bridge_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "BRIDGE_RECEIPT.json").write_text(
        json.dumps(_bridge_receipt()), encoding="utf-8"
    )
    (root / "CAMPAIGN_STATE.json").write_text("{}", encoding="utf-8")


@pytest.mark.parametrize("relative", [Path("."), Path("download") / "nested" / "bridge"])
def test_bridge_root_resolver_accepts_direct_and_nested_layout(tmp_path: Path, relative: Path) -> None:
    namespace = _load_fragment_namespace()
    root = tmp_path / relative
    _write_bridge_root(root)
    resolved = namespace["_m07_resolve_exact_bridge_root"]("M07", 320, tmp_path)
    assert resolved == root.resolve()


def test_bridge_root_resolver_rejects_ambiguous_exact_roots(tmp_path: Path) -> None:
    namespace = _load_fragment_namespace()
    _write_bridge_root(tmp_path / "a")
    _write_bridge_root(tmp_path / "b")
    with pytest.raises(ValueError, match="exactly one sealed root"):
        namespace["_m07_resolve_exact_bridge_root"]("M07", 320, tmp_path)


def test_bridge_root_resolver_rejects_missing_exact_root(tmp_path: Path) -> None:
    namespace = _load_fragment_namespace()
    (tmp_path / "CAMPAIGN_STATE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one sealed root"):
        namespace["_m07_resolve_exact_bridge_root"]("M07", 320, tmp_path)


def test_bridge_root_resolver_rejects_corrupt_identity(tmp_path: Path) -> None:
    namespace = _load_fragment_namespace()
    bad = _bridge_receipt()
    bad["recipe_fingerprint"] = "0" * 64
    (tmp_path / "BRIDGE_RECEIPT.json").write_text(json.dumps(bad), encoding="utf-8")
    (tmp_path / "CAMPAIGN_STATE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one sealed root"):
        namespace["_m07_resolve_exact_bridge_root"]("M07", 320, tmp_path)


def _runtime_notebook(attempt: int) -> dict:
    runtime = {
        "enabled": True,
        "resolution": 320,
        "attempt": attempt,
        "account_id": "kg-05",
        "owner": "trickermark",
        "source_persistence_handle": contract.PERSISTENCE_HANDLE,
        "source_phase2_state_handle": contract.R320_BRIDGE_HANDLE,
        "max_new_folds": 1,
        "execution_scope": "SETUP_DEFINITIONS_EXACT_RUNTIME_STAGE_ONLY",
        "pre_runtime_analysis_cells_removed": True,
    }
    source = "\n".join(
        [
            "CGP_MAX_NEW_FOLDS_PER_RUN = 1",
            "CGP_RUNTIME_READONLY_PREDECESSOR = True",
            "Runtime predecessor persistence is read-only",
            f"CGP_RUNTIME_SOURCE_STATE_HANDLE = {contract.R320_BRIDGE_HANDLE!r}",
            "def _m07_resolve_exact_bridge_root(model_id, resolution, temp): pass",
            "restore_root = _m07_resolve_exact_bridge_root(model_id, resolution, temp)",
            "M07_R320_BRIDGE_LAYOUT_RESOLVED",
            "M07_R320_BRIDGE_EXTRACTED_RESTORE_VERIFIED",
        ]
    )
    return {
        "metadata": {
            "cgp_continuation": {
                "campaign_id": contract.CAMPAIGN_ID,
                "persistence_handle": contract.PERSISTENCE_HANDLE,
                "locked_test_used": False,
                "external_used": False,
                "phase2_unlocked": True,
                "runtime_partition": runtime,
            }
        },
        "cells": [{"cell_type": "code", "source": source.splitlines(keepends=True)}],
    }


def _write_runtime_artifact(tmp_path: Path, attempt: int, run_id: str = "35299999999"):
    notebook = tmp_path / "runtime.ipynb"
    notebook.write_text(json.dumps(_runtime_notebook(attempt)), encoding="utf-8")
    sha_file = tmp_path / "m07_runtime_notebook.sha256"
    sha_file.write_text(contract.sha256_file(notebook) + "\n", encoding="utf-8")
    slug_file = tmp_path / "m07_kernel_slug.txt"
    slug_file.write_text(
        f"trickermark/m07-runtime-r320-a{attempt:02d}-20260917-{run_id}",
        encoding="utf-8",
    )
    title_file = tmp_path / "m07_kernel_title.txt"
    title_file.write_text(
        f"M07 Runtime R320 A{attempt:02d} 20260917 {run_id}", encoding="utf-8"
    )
    return notebook, sha_file, slug_file, title_file


@pytest.mark.parametrize("attempt", [7, 8, 9])
def test_dynamic_runtime_artifact_contract_accepts_current_attempts(
    tmp_path: Path, attempt: int
) -> None:
    run_id = "35299999999"
    notebook, sha_file, slug_file, title_file = _write_runtime_artifact(
        tmp_path, attempt, run_id
    )
    report = contract.validate_runtime_artifact(
        token=f"M07_RUNTIME_M07_R320_A{attempt:02d}",
        run_id=run_id,
        notebook_path=notebook,
        sha_path=sha_file,
        slug_path=slug_file,
        title_path=title_file,
    )
    assert report["status"] == "PASS"
    assert report["attempt"] == attempt
    assert report["source_phase2_state_handle"] == contract.R320_BRIDGE_HANDLE


def test_historical_r320_attempts_are_blocked() -> None:
    with pytest.raises(contract.ContractError, match="immutable"):
        contract.parse_runtime_token("M07_RUNTIME_M07_R320_A06")


def _sealed_runtime_receipt(attempt: int = 9) -> dict:
    body = {
        "status": "PARTIAL_FOLD_UNIT_COMPLETE",
        "model_id": "M07",
        "resolution": 320,
        "attempt": attempt,
        "restored_folds": 4,
        "max_new_folds": 1,
        "completed_fold": 5,
        "new_folds_completed": 1,
        "five_fold_ready": True,
        "next_action": "VALIDATE_FIVE_FOLD_STATE_BEFORE_LOCKED_TEST",
        "locked_test_started": False,
        "persistence_handle": contract.PERSISTENCE_HANDLE,
        "run_contract": {
            "stage": "phase2_campaign",
            "model_id": "M07",
            "resolution": 320,
            "split_fingerprint": contract.SPLIT_FINGERPRINT,
            "extra": {"recipe_fingerprint": contract.RECIPE_FINGERPRINT},
        },
        "artifact_sha256": {"campaign_state": "a" * 64},
    }
    return {**body, "receipt_sha256": contract.canonical_fingerprint(body)}


def test_runtime_scientific_receipt_passes_exact_contract() -> None:
    receipt = _sealed_runtime_receipt()
    report = contract.validate_runtime_receipt(
        receipt, token="M07_RUNTIME_M07_R320_A09"
    )
    assert report["status"] == "PASS"
    assert report["restored_folds"] == 4


def test_runtime_scientific_receipt_rejects_tampering() -> None:
    receipt = _sealed_runtime_receipt()
    receipt["restored_folds"] = 3
    with pytest.raises(contract.ContractError, match="seal mismatch"):
        contract.validate_runtime_receipt(receipt, token="M07_RUNTIME_M07_R320_A09")


def test_extract_session_state_reads_status_field_only() -> None:
    assert contract.extract_session_state({"status": "ERROR", "message": "COMPLETE"}) == "ERROR"
    assert contract.extract_session_state({"message": "ERROR"}) is None


def test_runtime_scientific_receipt_rejects_locked_test_boundary_crossing() -> None:
    receipt = _sealed_runtime_receipt()
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    body["locked_test_started"] = True
    body["artifact_sha256"] = {"campaign_state": "a" * 64, "final_report": "b" * 64}
    tampered = {**body, "receipt_sha256": contract.canonical_fingerprint(body)}
    with pytest.raises(contract.ContractError, match="Locked-Test boundary"):
        contract.validate_runtime_receipt(tampered, token="M07_RUNTIME_M07_R320_A09")


def test_runtime_scientific_receipt_requires_fold5_ready_boundary() -> None:
    receipt = _sealed_runtime_receipt()
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    body["five_fold_ready"] = False
    tampered = {**body, "receipt_sha256": contract.canonical_fingerprint(body)}
    with pytest.raises(contract.ContractError, match="five-fold-ready"):
        contract.validate_runtime_receipt(tampered, token="M07_RUNTIME_M07_R320_A09")


def test_runtime_artifact_rejects_pre_fold5_locked_test_cell(tmp_path: Path) -> None:
    run_id = "35299999999"
    notebook, sha_file, slug_file, title_file = _write_runtime_artifact(tmp_path, 9, run_id)
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    payload["cells"].append({
        "cell_type": "code",
        "source": ["# 12) ONE-TIME LOCKED TEST — forbidden in Fold5 recovery\n"],
    })
    notebook.write_text(json.dumps(payload), encoding="utf-8")
    sha_file.write_text(contract.sha256_file(notebook) + "\n", encoding="utf-8")
    with pytest.raises(contract.ContractError, match="forbidden pre-Fold5"):
        contract.validate_runtime_artifact(
            token="M07_RUNTIME_M07_R320_A09",
            run_id=run_id,
            notebook_path=notebook,
            sha_path=sha_file,
            slug_path=slug_file,
            title_path=title_file,
        )


def _kernel(ref: str, last: str = "2026-09-17T13:00:00Z") -> dict:
    return {"ref": ref, "lastRunTime": last}


def test_runtime_inventory_allows_next_attempt_after_terminal_errors() -> None:
    a07 = "trickermark/m07-runtime-r320-a07-20260916-35066913586"
    a08 = "trickermark/m07-runtime-r320-a08-20260917-35224996849"
    report = contract.assess_runtime_inventory(
        token="M07_RUNTIME_M07_R320_A09",
        kernels=[_kernel(a08), _kernel(a07)],
        statuses={a07: {"status": "ERROR"}, a08: {"status": "ERROR"}},
    )
    assert report["status"] == "PASS"
    assert report["candidate_count"] == 2


def test_runtime_inventory_blocks_attempt_reuse_even_after_error() -> None:
    a09 = "trickermark/m07-runtime-r320-a09-20260917-35299999999"
    with pytest.raises(contract.ContractError, match="ATTEMPT_ALREADY_USED"):
        contract.assess_runtime_inventory(
            token="M07_RUNTIME_M07_R320_A09",
            kernels=[_kernel(a09)],
            statuses={a09: {"status": "ERROR"}},
        )


def test_runtime_inventory_blocks_overlapping_active_gpu() -> None:
    a08 = "trickermark/m07-runtime-r320-a08-20260917-35224996849"
    with pytest.raises(contract.ContractError, match="ACTIVE_RUNTIME_EXISTS"):
        contract.assess_runtime_inventory(
            token="M07_RUNTIME_M07_R320_A09",
            kernels=[_kernel(a08)],
            statuses={a08: {"status": "RUNNING"}},
        )


def test_runtime_inventory_blocks_stale_attempt_when_newer_exists() -> None:
    a10 = "trickermark/m07-runtime-r320-a10-20260917-35300000000"
    with pytest.raises(contract.ContractError, match="NEWER_ATTEMPT_ALREADY_EXISTS"):
        contract.assess_runtime_inventory(
            token="M07_RUNTIME_M07_R320_A09",
            kernels=[_kernel(a10)],
            statuses={a10: {"status": "ERROR"}},
        )
