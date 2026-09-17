"""Clone verified R320 source Version 4 into an immutable owner dataset shared read-only with kg-05.

This runs on the trusted self-hosted runner, outside Kaggle's non-interactive
notebook environment.  The clone is intentionally a new handle with one
version so subsequent kernels can attach the unversioned dataset source while
the notebook requests ``/versions/1`` deterministically.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
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


def _kaggle_authorization_header() -> str:
    """Build an auth header without ever logging credential material."""

    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        fail("BLOCKED_VALIDATION_INFRASTRUCTURE: KAGGLE_API_TOKEN is unavailable")
    if token.startswith("KGAT_"):
        return f"Bearer {token}"
    username = os.environ.get("KAGGLE_USERNAME", "rezanory").strip() or "rezanory"
    encoded = base64.b64encode(f"{username}:{token}".encode("utf-8")).decode("ascii")
    return f"Basic {encoded}"


def assert_target_absent() -> None:
    """Fail closed unless the exact target is absent from the owner's dataset list.

    ``dataset_download`` is intentionally not used for an absence check: Kaggle
    can return HTTP 403 for a missing/private target, which is ambiguous. The
    authenticated datasets/list endpoint is read-only and distinguishes an
    existing own dataset from an absent one without attempting an attachment.
    """

    owner, slug = TARGET_HANDLE.split("/", 1)
    if owner != "trickermark":
        fail("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: target owner drift")

    auth = _kaggle_authorization_header()
    base = "https://www.kaggle.com/api/v1/datasets/list"
    exact_ref = TARGET_HANDLE.lower()
    seen: set[str] = set()

    for page in range(1, 11):
        query = urllib.parse.urlencode(
            {
                "group": "my",
                "sortBy": "updated",
                "size": "all",
                "filetype": "all",
                "license": "all",
                "tagids": "",
                "search": slug,
                "user": "",
                "page": page,
            }
        )
        request = urllib.request.Request(
            f"{base}?{query}",
            headers={
                "Authorization": auth,
                "User-Agent": "radlina-m07-source-freeze/1.0",
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200:
                    fail(
                        "BLOCKED_VALIDATION_INFRASTRUCTURE: target list preflight "
                        f"HTTP {response.status}"
                    )
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            fail(
                "BLOCKED_VALIDATION_INFRASTRUCTURE: target list preflight "
                f"HTTP {exc.code}"
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            fail(
                "BLOCKED_VALIDATION_INFRASTRUCTURE: target list preflight failed: "
                f"{type(exc).__name__}"
            )

        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("datasets"), list):
            rows = payload["datasets"]
        else:
            fail("BLOCKED_VALIDATION_INFRASTRUCTURE: unexpected datasets/list response shape")

        for row in rows:
            if isinstance(row, dict):
                ref = str(row.get("ref", "")).strip().lower()
                if ref:
                    seen.add(ref)
        if exact_ref in seen:
            fail("BLOCKED_IMMUTABLE_SOURCE_HYGIENE: target clone handle already exists")
        if len(rows) < 20:
            break

    print(
        json.dumps(
            {
                "target_absence_probe": "PASS",
                "target_handle": TARGET_HANDLE,
                "matching_refs_seen": len(seen),
            },
            sort_keys=True,
        )
    )


def main() -> int:
    if SOURCE_HANDLE != "rezanory/m07-gate-r320-state-v1-7/versions/4":
        fail("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: source handle drift")
    if TARGET_HANDLE != "trickermark/m07-gate-r320-state-v1-7-immutable-v4":
        fail("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: target handle drift")

    ROOT.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    STAGED.mkdir(parents=True, exist_ok=True)
    probe_only = os.environ.get("PROBE_ONLY", "").strip() == "1"

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
        "isPrivate": True,
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
        receipt["upload_status"] = "PROBE_SOURCE_VERIFIED_NO_MUTATION"
        RECEIPT_PATH.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"probe_only": True, "source_handle": SOURCE_HANDLE, "target_handle": TARGET_HANDLE, "file_count": len(manifest), "manifest_sha256": manifest_sha}, sort_keys=True))
        return 0

    assert_target_absent()
    kagglehub.dataset_upload(
        TARGET_HANDLE,
        str(STAGED),
        version_notes="d260914d immutable same-owner clone of rezanory/m07-gate-r320-state-v1-7 Version 4",
    )
    receipt["upload_status"] = "SUBMITTED_FIRST_VERSION"

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    receipt["owner_status"] = "trickermark"

    deadline = time.time() + 300
    status_payload: dict[str, Any] = {}
    while time.time() < deadline:
        status_payload = json.loads(
            api.dataset_status(
                TARGET_HANDLE,
                format="json(status,current_version_number)",
            )
        )
        status = str(status_payload.get("status", "")).lower()
        version = int(status_payload.get("current_version_number") or 0)
        if status == "ready" and version == 1:
            break
        if status in {"failed", "error"}:
            fail(
                "BLOCKED_VALIDATION_INFRASTRUCTURE: immutable target processing "
                f"failed status={status} version={version}"
            )
        time.sleep(10)
    else:
        fail(
            "BLOCKED_VALIDATION_INFRASTRUCTURE: immutable target did not become "
            f"Ready Version 1: {status_payload}"
        )

    receipt["target_status"] = "READY"
    receipt["target_version"] = 1
    RECEIPT_PATH.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "target_handle": TARGET_HANDLE,
                "upload_status": receipt["upload_status"],
                "owner_status": receipt["owner_status"],
                "target_status": receipt["target_status"],
                "target_version": receipt["target_version"],
                "manifest_sha256": manifest_sha,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
