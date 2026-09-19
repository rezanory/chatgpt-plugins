from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import pytest

import m07_runtime_contract as contract


FRAGMENT = Path(__file__).with_name("m07_r384_state_expanded_compat.pyfrag")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_fragment_namespace() -> dict:
    namespace = {
        "Path": Path,
        "json": json,
        "re": re,
        "shutil": shutil,
        "zipfile": zipfile,
        "CGP_RUNTIME_ATTEMPT": 5,
    }
    exec(
        compile(FRAGMENT.read_text(encoding="utf-8"), str(FRAGMENT), "exec"),
        namespace,
    )
    namespace.update(
        {
            "phase2_handle": lambda model_id, resolution: contract.R384_STATE_HANDLE,
            "phase2_contract": lambda model_id, resolution, **kwargs: {
                "model_id": model_id,
                "resolution": int(resolution),
                **kwargs,
            },
            "validate_receipt": lambda payload, contract_value, artifacts=None, allowed_statuses=None: payload,
        }
    )
    return namespace


def _campaign_payload(folds=(1,)) -> dict:
    return {
        "schema": "phase2.state.v2",
        "status": "COMPLETE",
        "model_id": "M07",
        "resolution": 384,
        "run_contract": {
            "split_fingerprint": contract.SPLIT_FINGERPRINT,
            "extra": {"recipe_fingerprint": contract.RECIPE_FINGERPRINT},
        },
        "artifact_sha256": {
            f"FOLD_{fold}_RECOVERY.zip": "a" * 64 for fold in folds
        },
        "receipt_sha256": "b" * 64,
    }


def _write_expanded_root(root: Path, folds=(1,)) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "CAMPAIGN_STATE.json").write_text(
        json.dumps(_campaign_payload(folds)), encoding="utf-8"
    )
    for fold in folds:
        fold_root = root / f"FOLD_{fold}_RECOVERY" / "FOLDS" / f"fold_{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        (fold_root / "COMPLETED.json").write_text(
            json.dumps({"fold": fold}), encoding="utf-8"
        )
        (fold_root / "artifact.bin").write_bytes(f"fold-{fold}".encode())


def test_r384_mount_resolver_accepts_unique_preattached_state(tmp_path: Path) -> None:
    namespace = _load_fragment_namespace()
    candidate = (
        tmp_path
        / "datasets"
        / "trickermark"
        / "m07-gate-r384-state-v1-7"
    )
    _write_expanded_root(candidate, (1,))
    resolved = namespace["_m07_resolve_r384_state_mount"](
        "M07", 384, tmp_path
    )
    assert resolved == candidate.resolve()


def test_r384_mount_resolver_rejects_ambiguous_roots(tmp_path: Path) -> None:
    namespace = _load_fragment_namespace()
    for prefix in ("a", "b"):
        _write_expanded_root(
            tmp_path / prefix / "m07-gate-r384-state-v1-7", (1,)
        )
    with pytest.raises(RuntimeError, match="valid_candidate_count.*2"):
        namespace["_m07_resolve_r384_state_mount"]("M07", 384, tmp_path)


def test_r384_campaign_rejects_fold_sequence_gap() -> None:
    namespace = _load_fragment_namespace()
    with pytest.raises(ValueError, match="FOLD_SEQUENCE_GAP"):
        namespace["_m07_r384_campaign_fold_ids"](
            _campaign_payload((1, 3))
        )


def test_r384_expanded_restore_reseals_local_zip_state(tmp_path: Path) -> None:
    namespace = _load_fragment_namespace()
    candidate = (
        tmp_path
        / "input"
        / "datasets"
        / "trickermark"
        / "m07-gate-r384-state-v1-7"
    )
    _write_expanded_root(candidate, (1,))

    out = tmp_path / "out"
    state = tmp_path / "state"
    out.mkdir()
    state.mkdir()

    namespace.update(
        {
            "phase2_output_root": lambda model_id, resolution: out,
            "phase2_state_root": lambda model_id, resolution: state,
            "phase2_validate_fold": lambda model_id, resolution, fold, fold_dir: (
                {"fold": fold},
                None,
            ),
            "run_file_sha256": _sha256,
            "PHASE2_RESTORE_VERIFIED": set(),
            "PHASE2_REMOTE_FOLD_RECEIPTS": {},
        }
    )

    def seal(model_id, resolution):
        (state / "CAMPAIGN_STATE.json").write_text(
            json.dumps({"receipt_sha256": "c" * 64}), encoding="utf-8"
        )

    namespace["phase2_seal_persistence"] = seal
    restored = namespace["_m07_restore_r384_expanded_state"](
        "M07", 384, input_root=tmp_path / "input"
    )

    assert restored == 1
    assert (out / "FOLDS" / "fold_1" / "COMPLETED.json").is_file()
    assert (state / "FOLD_1_RECOVERY.zip").is_file()
    assert ("M07", 384) in namespace["PHASE2_RESTORE_VERIFIED"]


def _write_runtime_artifact(
    tmp_path: Path, attempt: int = 5, run_id: str = "35449999999"
):
    runtime = {
        "enabled": True,
        "resolution": 384,
        "attempt": attempt,
        "account_id": "kg-05",
        "owner": "trickermark",
        "source_persistence_handle": contract.PERSISTENCE_HANDLE,
        "source_phase2_state_handle": None,
        "max_new_folds": 1,
        "execution_scope": "SETUP_DEFINITIONS_EXACT_RUNTIME_STAGE_ONLY",
        "pre_runtime_analysis_cells_removed": True,
        "legacy_r224_preflight_removed": True,
        "bridge_restore_mode": None,
    }
    source = "\n".join(
        [
            "CGP_MAX_NEW_FOLDS_PER_RUN = 1",
            "CGP_RUNTIME_READONLY_PREDECESSOR = True",
            "Runtime predecessor persistence is read-only",
            contract.R384_STATE_HANDLE,
            "def _m07_resolve_r384_state_mount(model_id, resolution, input_root=None): pass",
            "def _m07_restore_r384_expanded_state(model_id, resolution, input_root=None): pass",
            "r384_restored = _m07_restore_r384_expanded_state(model_id, resolution)",
            "M07_R384_EXPANDED_STATE_RESTORE_VERIFIED",
        ]
    )
    payload = {
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
        "cells": [
            {
                "cell_type": "code",
                "source": source.splitlines(keepends=True),
            }
        ],
    }
    notebook = tmp_path / "runtime.ipynb"
    notebook.write_text(json.dumps(payload), encoding="utf-8")
    sha_file = tmp_path / "m07_runtime_notebook.sha256"
    sha_file.write_text(
        contract.sha256_file(notebook) + "\n", encoding="utf-8"
    )
    slug_file = tmp_path / "m07_kernel_slug.txt"
    slug_file.write_text(
        f"trickermark/m07-runtime-r384-a{attempt:02d}-20260919-{run_id}",
        encoding="utf-8",
    )
    title_file = tmp_path / "m07_kernel_title.txt"
    title_file.write_text(
        f"M07 Runtime R384 A{attempt:02d} 20260919 {run_id}",
        encoding="utf-8",
    )
    return notebook, sha_file, slug_file, title_file


def test_r384_artifact_requires_expanded_state_restore_markers(
    tmp_path: Path,
) -> None:
    notebook, sha_file, slug_file, title_file = _write_runtime_artifact(
        tmp_path
    )
    report = contract.validate_runtime_artifact(
        token="M07_RUNTIME_M07_R384_A05",
        run_id="35449999999",
        notebook_path=notebook,
        sha_path=sha_file,
        slug_path=slug_file,
        title_path=title_file,
    )
    assert report["status"] == "PASS"

    payload = json.loads(notebook.read_text(encoding="utf-8"))
    payload["cells"][0]["source"] = [
        line
        for line in payload["cells"][0]["source"]
        if "M07_R384_EXPANDED_STATE_RESTORE_VERIFIED" not in line
    ]
    notebook.write_text(json.dumps(payload), encoding="utf-8")
    sha_file.write_text(
        contract.sha256_file(notebook) + "\n", encoding="utf-8"
    )
    with pytest.raises(
        contract.ContractError, match="expanded-state evidence marker"
    ):
        contract.validate_runtime_artifact(
            token="M07_RUNTIME_M07_R384_A05",
            run_id="35449999999",
            notebook_path=notebook,
            sha_path=sha_file,
            slug_path=slug_file,
            title_path=title_file,
        )
