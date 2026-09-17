from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest

import m07_runtime_contract as contract


FRAGMENT = Path(__file__).with_name("m07_r320_bridge_extracted_compat.pyfrag")
ARCHIVES = {
    "FOLD_1_RECOVERY.zip": "cd18468bf9305bb801f809098f8917462b16fb69fe9313b8bd17f2a95a7d1531",
    "FOLD_2_RECOVERY.zip": "a75967888cec5cea6133a2ed457bb998edb9ada01442be88b2cc35e970e6f72d",
    "FOLD_3_RECOVERY.zip": "4b04f657ddfcab61dc720f17ba4a8a5a8f234cd2bb7ed10b64e7357e049a9ad6",
    "FOLD_4_RECOVERY.zip": "74a7c69c572a66a139beb99dfa375185ce2f25e96689642eac2a1e56928deb93",
}
CAMPAIGN_RECEIPT = "e3884dd10e323d2f0925660599a420964781111d850664e8c88a830324460c77"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _namespace() -> dict:
    ns = {
        "Path": Path,
        "json": json,
        "shutil": shutil,
        "zipfile": zipfile,
        "run_file_sha256": _sha256,
        "CGP_RUNTIME_SOURCE_STATE_HANDLE": contract.R320_BRIDGE_HANDLE,
    }
    exec(compile(FRAGMENT.read_text(encoding="utf-8"), str(FRAGMENT), "exec"), ns)
    return ns


def _campaign() -> dict:
    return {
        "schema": "phase2.state.v2",
        "status": "COMPLETE",
        "model_id": "M07",
        "resolution": 320,
        "receipt_sha256": CAMPAIGN_RECEIPT,
        "run_contract": {
            "split_fingerprint": contract.SPLIT_FINGERPRINT,
            "extra": {"recipe_fingerprint": contract.RECIPE_FINGERPRINT},
        },
        "artifact_sha256": dict(ARCHIVES),
    }


def _write_bridge_root(root: Path, *, with_folds: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fold_receipts = {}
    if with_folds:
        for fold in range(1, 5):
            fold_dir = root / f"FOLD_{fold}_RECOVERY" / "FOLDS" / f"fold_{fold}"
            fold_dir.mkdir(parents=True)
            artifact = fold_dir / "validation_predictions.csv"
            artifact.write_text(f"fold,{fold}\n", encoding="utf-8")
            artifact_manifest = {artifact.name: _sha256(artifact)}
            completed = {
                "schema": "pneumonia.phase2.fold.v1.7",
                "status": "COMPLETED",
                "model_id": "M07",
                "resolution": 320,
                "fold_id": fold,
                "confirmed_m07_recipe_fingerprint": contract.RECIPE_FINGERPRINT,
                "split_fingerprint": contract.SPLIT_FINGERPRINT,
                "locked_test_used_for_training": False,
                "external_used_for_training": False,
                "receipt_sha256": f"receipt-{fold}",
                "run_fingerprint": f"run-{fold}",
                "artifact_sha256": artifact_manifest,
            }
            (fold_dir / "COMPLETED.json").write_text(json.dumps(completed), encoding="utf-8")
            fold_receipts[str(fold)] = {
                "receipt_sha256": completed["receipt_sha256"],
                "run_fingerprint": completed["run_fingerprint"],
                "artifact_sha256": artifact_manifest,
            }
    bridge = {
        "schema": "m07.r320.producer.bridge.v2",
        "status": "COMPLETE",
        "target_dataset": "trickermark/m07-r320-producer-v4-bridge-e3884dd1",
        "campaign_receipt_sha256": CAMPAIGN_RECEIPT,
        "split_fingerprint": contract.SPLIT_FINGERPRINT,
        "recipe_fingerprint": contract.RECIPE_FINGERPRINT,
        "archive_sha256": dict(ARCHIVES),
        "sealed_fold_ids": [1, 2, 3, 4],
        "missing_fold_ids": [5],
        "locked_test_used_for_training": False,
        "external_used_for_training": False,
        "training_performed": False,
        "mutable_dataset_versions_5_6_used": False,
        "fold_receipts": fold_receipts,
    }
    (root / "BRIDGE_RECEIPT.json").write_text(json.dumps(bridge), encoding="utf-8")
    (root / "CAMPAIGN_STATE.json").write_text(json.dumps(_campaign()), encoding="utf-8")


@pytest.mark.parametrize(
    "relative",
    [
        Path("m07-r320-producer-v4-bridge-e3884dd1"),
        Path("datasets") / "trickermark" / "m07-r320-producer-v4-bridge-e3884dd1",
    ],
)
def test_bridge_mount_accepts_direct_and_nested_layout(tmp_path: Path, relative: Path) -> None:
    ns = _namespace()
    candidate = tmp_path / relative
    _write_bridge_root(candidate)
    resolved = ns["_m07_resolve_exact_bridge_mount"]("M07", 320, tmp_path)
    assert resolved == candidate.resolve()


def test_bridge_mount_rejects_ambiguous_exact_roots(tmp_path: Path) -> None:
    ns = _namespace()
    _write_bridge_root(tmp_path / "a" / "m07-r320-producer-v4-bridge-e3884dd1")
    _write_bridge_root(tmp_path / "b" / "m07-r320-producer-v4-bridge-e3884dd1")
    with pytest.raises(ValueError, match="valid_candidate_count.*2"):
        ns["_m07_resolve_exact_bridge_mount"]("M07", 320, tmp_path)


def test_bridge_mount_rejects_corrupt_campaign_identity(tmp_path: Path) -> None:
    ns = _namespace()
    candidate = tmp_path / "m07-r320-producer-v4-bridge-e3884dd1"
    _write_bridge_root(candidate)
    campaign = json.loads((candidate / "CAMPAIGN_STATE.json").read_text(encoding="utf-8"))
    campaign["receipt_sha256"] = "0" * 64
    (candidate / "CAMPAIGN_STATE.json").write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="valid_candidate_count.*0"):
        ns["_m07_resolve_exact_bridge_mount"]("M07", 320, tmp_path)


def test_bridge_restore_reads_mount_but_writes_only_to_work_root(tmp_path: Path) -> None:
    ns = _namespace()
    source = tmp_path / "input" / "m07-r320-producer-v4-bridge-e3884dd1"
    work = tmp_path / "work"
    work.mkdir(parents=True)
    _write_bridge_root(source, with_folds=True)
    before = sorted(path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file())

    staged, archives, receipt = ns["_m07_restore_exact_bridge_extracted"](
        "M07", 320, source, _campaign(), work_root=work
    )

    assert staged == work / "_validated_output"
    assert receipt == source / "BRIDGE_RECEIPT.json"
    assert set(archives) == set(ARCHIVES)
    assert all(path.parent == work / "_bridge_repacked_state" for path in archives.values())
    assert not (source / "_validated_output").exists()
    assert not (source / "_bridge_repacked_state").exists()
    after = sorted(path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file())
    assert after == before


def test_fragment_keeps_download_fallback_but_exact_runtime_has_mount_resolver() -> None:
    source = FRAGMENT.read_text(encoding="utf-8")
    assert "def _m07_resolve_exact_bridge_mount" in source
    assert "M07_R320_BRIDGE_MOUNT_VERIFIED" in source
    assert "work_root = source_root if work_root is None else Path(work_root)" in source
    assert 'staged = work_root / "_validated_output"' in source
    assert 'repacked = work_root / "_bridge_repacked_state"' in source
