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

from kagglesdk.blobs.types.blob_api_service import ApiBlobType
from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest

import kagglehub
from kagglehub.clients import build_kaggle_client
from kagglehub.datasets import DEFAULT_IGNORE_PATTERNS
from kagglehub.exceptions import handle_mutate_call
from kagglehub.gcs_upload import normalize_patterns, upload_files_and_directories
from kagglehub.handle import parse_dataset_handle


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


def create_target_dataset_once() -> None:
    """Create the immutable clone as Version 1 only; never create a version.

    ``kagglehub.dataset_upload`` intentionally falls back to creating a new
    version when CreateDataset reports a conflict.  That behavior is unsafe
    for this immutable source clone: a pre-existing target must be a hard
    failure, not a reason to append Version 2.  Use the same official upload
    transport but call CreateDataset directly so a conflict remains fatal.
    """

    handle = parse_dataset_handle(TARGET_HANDLE)
    uploaded = upload_files_and_directories(
        str(STAGED),
        item_type=ApiBlobType.DATASET,
        ignore_patterns=normalize_patterns(default=DEFAULT_IGNORE_PATTERNS, additional=None),
    )
    proto = uploaded.to_proto()
    request = ApiCreateDatasetRequest()
    request.owner_slug = handle.owner
    request.slug = handle.dataset
    request.title = "M07 Gate R320 State V1.7 Immutable V4"
    request.files = proto.files
    request.directories = proto.directories
    request.is_private = True
    with build_kaggle_client() as api_client:
        handle_mutate_call(
            lambda: api_client.datasets.dataset_api_client.create_dataset(request)
        )

def main() -> int:
    if SOURCE_HANDLE != "rezanory/m07-gate-r320-state-v1-7/versions/4":
        fail("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: source handle drift")
    if TARGET_HANDLE != "trickermark/m07-gate-r320-state-v1-7-immutable-v4":
        fail("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: target handle drift")

    ROOT.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    STAGED.mkdir(parents=True, exist_ok=True)

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

    create_target_dataset_once()
    receipt["upload_status"] = "CREATE_DATASET_ACCEPTED_FIRST_VERSION"
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
