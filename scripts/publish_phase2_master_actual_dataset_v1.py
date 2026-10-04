from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

import phase2_final_evidence_extract_v1 as broker

OWNER = "azadka"
DATASET_SLUG = "pneumonia-phase2-master-actual-v2"
DATASET_REF = f"{OWNER}/{DATASET_SLUG}"
FILE_NAME = "PNEUMONIA_PHASE2_MASTER_ACTUAL_KAGGLE_SOURCES_V2.ipynb"
EXPECTED_SHA256 = "72cec18366b769a3753d70be879d5d458c39cc09bdbf106204f53ccf51d79daf"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_scalar(value, names: set[str]):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in names and item not in (None, ""):
                return item
        for item in value.values():
            found = find_scalar(item, names)
            if found not in (None, ""):
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_scalar(item, names)
            if found not in (None, ""):
                return found
    return None


def action_call(token: str, request_id: str, service: str, method: str, body: dict, purpose: str) -> dict:
    return broker.post_json(
        broker.ACTION_ENDPOINT,
        token,
        {
            "request_id": request_id,
            "provider": "kaggle",
            "operation_class": "write",
            "account_id": "master",
            "purpose": purpose,
            "service": service,
            "method": method,
            "body": body,
        },
        timeout=300,
    )


def read_call(token: str, service: str, method: str, body: dict) -> dict:
    return broker.post_json(
        broker.READ_ENDPOINT,
        token,
        {
            "action": "raw_read",
            "account_id": "master",
            "service": service,
            "method": method,
            "body": body,
        },
        timeout=180,
    )


def start_blob_upload(action_token: str, path: pathlib.Path) -> tuple[str, str, dict]:
    common = {
        "name": FILE_NAME,
        "contentType": "application/json",
        "contentLength": path.stat().st_size,
        "lastModifiedEpochSeconds": int(path.stat().st_mtime),
    }
    attempts = [
        ("ApiStartBlobUpload", "DATASET"),
        ("ApiStartBlobUpload", 1),
        ("StartBlobUpload", "DATASET"),
        ("StartBlobUpload", 1),
    ]
    history = []
    for idx, (method, blob_type) in enumerate(attempts, 1):
        response = action_call(
            action_token,
            f"phase2-master-dataset-blob-{os.environ['GITHUB_RUN_ID']}-{idx}",
            "blobs.BlobApiService",
            method,
            {**common, "type": blob_type},
            "Start a Kaggle dataset blob upload for the exact sealed Phase-2 master notebook; no model compute.",
        )
        history.append({"method": method, "type": blob_type, "response": response})
        if not response.get("ok"):
            continue
        upload_token = find_scalar(response, {"token"})
        create_url = find_scalar(response, {"createUrl", "create_url"})
        if isinstance(upload_token, str) and len(upload_token) >= 8 and isinstance(create_url, str) and create_url.startswith("http"):
            return upload_token, create_url, {"attempts": history}
    raise RuntimeError("START_BLOB_UPLOAD_FAILED=" + json.dumps(history)[:12000])


def put_blob(create_url: str, raw: bytes) -> int:
    request = urllib.request.Request(
        create_url,
        data=raw,
        method="PUT",
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Range": f"bytes 0-{len(raw)-1}/{len(raw)}",
            "Content-Length": str(len(raw)),
            "User-Agent": "phase2-master-actual-dataset-v1/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            status = int(response.status)
            if status not in (200, 201):
                raise RuntimeError(f"BLOB_PUT_HTTP_{status}")
            return status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:3000]
        raise RuntimeError(f"BLOB_PUT_HTTP_{exc.code}:{body}") from exc


def create_or_version_dataset(action_token: str, upload_token: str) -> tuple[str, dict]:
    create_body = {
        "ownerSlug": OWNER,
        "slug": DATASET_SLUG,
        "title": "Pneumonia Phase2 Master Actual V2",
        "isPrivate": True,
        "files": [{"token": upload_token}],
        "directories": [],
    }
    create = action_call(
        action_token,
        f"phase2-master-dataset-create-{os.environ['GITHUB_RUN_ID']}",
        "datasets.DatasetApiService",
        "CreateDataset",
        create_body,
        "Create a private master-account dataset containing the exact sealed Phase-2 integrated notebook.",
    )
    if create.get("ok") and not find_scalar(create, {"error"}):
        return "created", create

    error_text = json.dumps(create).lower()
    if not any(word in error_text for word in ("already", "exist", "in use", "conflict", "409")):
        raise RuntimeError("CREATE_DATASET_FAILED=" + json.dumps(create)[:12000])

    version = action_call(
        action_token,
        f"phase2-master-dataset-version-{os.environ['GITHUB_RUN_ID']}",
        "datasets.DatasetApiService",
        "CreateDatasetVersion",
        {
            "ownerSlug": OWNER,
            "datasetSlug": DATASET_SLUG,
            "body": {
                "versionNotes": "Exact lossless Phase-2 master notebook from 36 completed Kaggle kernels",
                "files": [{"token": upload_token}],
                "directories": [],
            },
        },
        "datasets.DatasetApiService",
    )
    if not version.get("ok") or find_scalar(version, {"error"}):
        raise RuntimeError("CREATE_DATASET_VERSION_FAILED=" + json.dumps(version)[:12000])
    return "versioned", version


def wait_dataset(read_token: str) -> tuple[int, dict]:
    last = {}
    for attempt in range(1, 61):
        response = read_call(
            read_token,
            "datasets.DatasetApiService",
            "GetDataset",
            {"ownerSlug": OWNER, "datasetSlug": DATASET_SLUG},
        )
        last = response
        if response.get("ok"):
            payload = broker._read_payload(response)
            version = int(payload.get("currentVersionNumber") or payload.get("current_version_number") or 0)
            if version > 0:
                return version, payload
        print("MASTER_DATASET_STATUS", attempt, "WAIT", flush=True)
        time.sleep(5)
    raise RuntimeError("DATASET_NOT_READY=" + json.dumps(last)[:6000])


def download_exact(action_token: str, version: int) -> tuple[bytes, dict]:
    response = action_call(
        action_token,
        f"phase2-master-dataset-download-{os.environ['GITHUB_RUN_ID']}",
        "datasets.DatasetApiService",
        "DownloadDataset",
        {
            "ownerSlug": OWNER,
            "datasetSlug": DATASET_SLUG,
            "fileName": FILE_NAME,
            "datasetVersionNumber": version,
            "raw": True,
        },
        "Verify the exact private master-account dataset file by read-only download and SHA-256 comparison.",
    )
    if not response.get("ok"):
        raise RuntimeError("DOWNLOAD_DATASET_BROKER_FAILED=" + json.dumps(response)[:8000])
    url = find_scalar(response, {"url"})
    if not isinstance(url, str) or not url.startswith("https://"):
        raise RuntimeError("DOWNLOAD_DATASET_URL_MISSING=" + json.dumps(response)[:8000])
    request = urllib.request.Request(url, headers={"User-Agent": "phase2-master-dataset-verify/1.0"})
    with urllib.request.urlopen(request, timeout=600) as download:
        raw = download.read()
    return raw, response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    read_token = os.environ["CGP_READ_OIDC_TOKEN"].strip()
    action_token = os.environ["CGP_ACTION_OIDC_TOKEN"].strip()
    if len(read_token) < 100 or len(action_token) < 100:
        raise SystemExit("OIDC_TOKEN_INVALID")

    raw = args.notebook.read_bytes()
    actual_sha = sha256_bytes(raw)
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError(f"LOCAL_MASTER_SHA_MISMATCH:{actual_sha}")
    parsed = json.loads(raw)
    if len(parsed.get("cells") or []) != 162:
        raise RuntimeError("LOCAL_MASTER_CELL_COUNT_INVALID")
    if sum(cell.get("cell_type") == "code" for cell in parsed["cells"]) != 98:
        raise RuntimeError("LOCAL_MASTER_CODE_CELL_COUNT_INVALID")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    upload_token, create_url, start_meta = start_blob_upload(action_token, args.notebook)
    put_status = put_blob(create_url, raw)
    dataset_action, dataset_response = create_or_version_dataset(action_token, upload_token)
    version, dataset_payload = wait_dataset(read_token)
    downloaded, download_response = download_exact(action_token, version)

    downloaded_sha = sha256_bytes(downloaded)
    if downloaded_sha != EXPECTED_SHA256:
        raise RuntimeError(
            f"MASTER_DATASET_READBACK_SHA_MISMATCH:{EXPECTED_SHA256}:{downloaded_sha}:{len(downloaded)}"
        )

    receipt = {
        "schema": "pneumonia.phase2.master.actual-kaggle-dataset.publish.v1",
        "status": "PASS",
        "dataset_ref": DATASET_REF,
        "dataset_version_number": version,
        "file_name": FILE_NAME,
        "bytes": len(raw),
        "source_sha256": actual_sha,
        "downloaded_bytes": len(downloaded),
        "downloaded_sha256": downloaded_sha,
        "cells": 162,
        "code_cells": 98,
        "real_kernel_count": 36,
        "original_cell_occurrences": 575,
        "quick_save_kernel_attempted": False,
        "training_performed": False,
        "inference_performed": False,
        "dataset_action": dataset_action,
        "blob_put_http_status": put_status,
    }
    (args.output_dir / "KAGGLE_MASTER_ACTUAL_DATASET_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "KAGGLE_MASTER_ACTUAL_DATASET_START.json").write_text(
        json.dumps(start_meta, indent=2), encoding="utf-8"
    )
    (args.output_dir / "KAGGLE_MASTER_ACTUAL_DATASET_CREATE.json").write_text(
        json.dumps(dataset_response, indent=2), encoding="utf-8"
    )
    (args.output_dir / "KAGGLE_MASTER_ACTUAL_DATASET_GET.json").write_text(
        json.dumps(dataset_payload, indent=2), encoding="utf-8"
    )
    (args.output_dir / "KAGGLE_MASTER_ACTUAL_DATASET_DOWNLOAD.json").write_text(
        json.dumps(download_response, indent=2), encoding="utf-8"
    )
    print("PHASE2_MASTER_DATASET_PUBLISH_PASS " + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
