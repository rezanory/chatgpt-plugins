"""Clone the verified R320 source Version 4 into an immutable kg-05 dataset.

This runs on the trusted self-hosted runner, outside Kaggle's non-interactive
notebook environment.  The clone is intentionally a new handle with one
version so subsequent kernels can attach the unversioned dataset source while
the notebook requests ``/versions/1`` deterministically.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
from typing import Any

import kagglehub


SOURCE_HANDLE = os.environ.get(
    "SOURCE_HANDLE", "rezanory/m07-gate-r320-state-v1-7/versions/4"
).strip()
TARGET_HANDLE = os.environ.get(
    "TARGET_HANDLE", "trickermark/m07-gate-r320-state-v1-7-immutable-v4"
).strip()
RUN_ID = os.environ.get("GITHUB_RUN_ID", "local").strip()
RUNNER_TEMP = pathlib.Path(os.environ.get("RUNNER_TEMP", pathlib.Path.cwd()))
ROOT = RUNNER_TEMP / f"m07-source-v4-{RUN_ID}"
DOWNLOAD_ROOT = ROOT / "download"
STAGED = ROOT / "staged"
RECEIPT_PATH = ROOT / "m07-source-freeze-receipt.json"


def fail(message: str) -> None:
    raise SystemExit(message)


def assert_target_absent() -> None:
    """Fail closed if the fixed clone handle already exists.

    A missing dataset is the only condition that permits creating Version 1.
    Any non-404 response (including auth or transport failures) is a blocker,
    not permission to create a second version.
    """

    probe = ROOT / "target-probe"
    try:
        kagglehub.dataset_download(
            TARGET_HANDLE,
            path="dataset-metadata.json",
            output_dir=str(probe),
            force_download=True,
        )
    except Exception as exc:  # noqa: BLE001 - preserve provider diagnosis
        detail = str(exc)
        lowered = detail.lower()
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
        if status_code == 404 or "404" in lowered or "not found" in lowered or "does not exist" in lowered:
            return
        fail(
            "BLOCKED_VALIDATION_INFRASTRUCTURE: target preflight failed: "
            f"{type(exc).__name__} status={status_code or 'UNKNOWN'}"
        )
    fail("BLOCKED_IMMUTABLE_SOURCE_HYGIENE: target clone handle already exists")


def main() -> int:
    if SOURCE_HANDLE != "rezanory/m07-gate-r320-state-v1-7/versions/4":
        fail("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: source handle drift")
    if TARGET_HANDLE != "trickermark/m07-gate-r320-state-v1-7-immutable-v4":
        fail("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: target handle drift")

    ROOT.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    STAGED.mkdir(parents=True, exist_ok=True)
    probe_only = os.environ.get('PROBE_ONLY', '').strip() == '1'

    downloaded = pathlib.Path(
        kagglehub.dataset_download(
            SOURCE_HANDLE,
            output_dir=str(DOWNLOAD_ROOT),
            force_download=True,
        )
    )
    if not downloaded.is_dir():
        fail("BLOCKED_EXACT_OBJECT_IDENTITY_MISMATCH: source download is not a directory")

    required = [
        "CAMPAIGN_STATE.json",
        "FOLD_4_RECOVERY/FOLDS/fold_4/COMPLETED.json",
    ]
    source_files = {
        path.relative_to(downloaded).as_posix(): path
        for path in downloaded.rglob("*")
        if path.is_file() and path.name not in {".complete", "dataset-metadata.json"}
    }
    missing = [name for name in required if name not in source_files]
    if missing:
        fail(f"BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: missing={missing}")

    manifest: list[dict[str, Any]] = []
    for name in sorted(source_files):
        source = source_files[name]
        data = source.read_bytes()
        manifest.append(
            {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
        destination = STAGED / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    manifest_blob = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode()
    manifest_sha = hashlib.sha256(manifest_blob).hexdigest()
    metadata = {
        "title": "M07 Gate R320 State V1.7 Immutable V4",
        "id": TARGET_HANDLE,
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            "Immutable fix-forward clone of "
            f"{SOURCE_HANDLE}; source Version 4 is frozen and must not be replaced."
        ),
    }
    (STAGED / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    receipt: dict[str, Any] = {
        "schema": "m07.d260914d.source.freeze.v1",
        "source_handle": SOURCE_HANDLE,
        "target_handle": TARGET_HANDLE,
        "source_version": 4,
        "required_files": required,
        "file_count": len(manifest),
        "manifest_sha256": manifest_sha,
        "manifest": manifest,
        "github_run_id": RUN_ID,
        "source_sha": os.environ.get("GITHUB_SHA", "").strip(),
    }
    RECEIPT_PATH.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_handle": SOURCE_HANDLE,
                "target_handle": TARGET_HANDLE,
                "source_version": 4,
                "required_files": {name: name in source_files for name in required},
                "file_count": len(manifest),
                "manifest_sha256": manifest_sha,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    if probe_only:
        receipt['upload_status'] = 'PROBE_SOURCE_VERIFIED_NO_MUTATION'
        RECEIPT_PATH.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'probe_only': True, 'source_handle': SOURCE_HANDLE, 'manifest_sha256': manifest_sha, 'file_count': len(manifest)}, sort_keys=True))
        return 0

    assert_target_absent()

    kagglehub.dataset_upload(
        TARGET_HANDLE,
        str(STAGED),
        version_notes="d260914d immutable clone of rezanory/m07-gate-r320-state-v1-7 Version 4",
    )
    receipt["upload_status"] = "SUBMITTED_FIRST_VERSION"
    RECEIPT_PATH.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "target_handle": TARGET_HANDLE,
                "upload_status": receipt["upload_status"],
                "manifest_sha256": manifest_sha,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
