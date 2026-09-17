from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

OWNER = "rezanory"
DATASET_SLUG = "m07-gate-r320-state-v1-7"
READER = "trickermark"
ACTION_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"


def fail(message: str) -> None:
    raise SystemExit(message)


def call(method: str, body: dict) -> dict:
    oidc = os.environ.get("CGP_ACTION_OIDC_TOKEN", "").strip()
    if len(oidc) < 100:
        fail("BLOCKED_VALIDATION_INFRASTRUCTURE: trusted action OIDC token missing")
    payload = {
        "request_id": f"m07-r320-share-{method.lower()}-{os.environ.get('GITHUB_RUN_ID','local')}",
        "provider": "kaggle",
        "operation_class": "write",
        "account_id": "kg-03",
        "purpose": "Permission-only repair: grant trickermark READER access to exact M07 R320 V1.7 source; no data upload, no training, no Locked Test.",
        "service": "datasets.DatasetApiService",
        "method": method,
        "body": body,
    }
    req = urllib.request.Request(
        ACTION_ENDPOINT,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {oidc}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "radlina-m07-r320-share/2.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            envelope = json.loads(response.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read(1400).decode("utf-8", "replace")
        fail(f"BLOCKED_VALIDATION_INFRASTRUCTURE: broker {method} HTTP {exc.code}: {detail[:900]}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        fail(f"BLOCKED_VALIDATION_INFRASTRUCTURE: broker {method} failed: {type(exc).__name__}")
    if not isinstance(envelope, dict) or not envelope.get("ok"):
        fail(f"BLOCKED_VALIDATION_INFRASTRUCTURE: broker {method} rejected: {json.dumps(envelope)[:900]}")
    result = envelope.get("result")
    if not isinstance(result, dict):
        fail(f"BLOCKED_VALIDATION_INFRASTRUCTURE: broker {method} result missing")
    return result


def normalize_collaborators(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    rows: list[dict] = []
    for row in value:
        if isinstance(row, dict) and str(row.get("username", "")).strip():
            rows.append(dict(row))
    return rows


def main() -> int:
    dataset = call("GetDataset", {"ownerSlug": OWNER, "datasetSlug": DATASET_SLUG})
    versions = dataset.get("versions") or []
    v4 = next((row for row in versions if isinstance(row, dict) and int(row.get("versionNumber") or 0) == 4), None)
    if not v4 or str(v4.get("status", "")).upper() != "READY":
        fail("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: source Version 4 is not READY")
    if "completed fold 4" not in str(v4.get("versionNotes", "")).lower():
        fail("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: source Version 4 note drift")

    metadata = call("GetDatasetMetadata", {"ownerSlug": OWNER, "datasetSlug": DATASET_SLUG})
    info = metadata.get("info") if isinstance(metadata.get("info"), dict) else {}
    collaborators = normalize_collaborators(info.get("collaborators"))
    matched = False
    for row in collaborators:
        if str(row.get("username", "")).strip().lower() == READER:
            row["role"] = 1
            matched = True
    if not matched:
        collaborators.append({"username": READER, "role": 1})

    settings = {
        "title": str(info.get("title") or DATASET_SLUG),
        "description": str(info.get("description") or "M07 V1.7 R320 sealed state"),
        "isPrivate": True,
        "collaborators": collaborators,
    }
    if isinstance(info.get("licenses"), list):
        settings["licenses"] = info["licenses"]
    if isinstance(info.get("keywords"), list):
        settings["keywords"] = info["keywords"]

    update = call("UpdateDatasetMetadata", {"ownerSlug": OWNER, "datasetSlug": DATASET_SLUG, "settings": settings})
    errors = update.get("errors") or []
    if isinstance(errors, list) and any(str(item).strip() for item in errors):
        fail("BLOCKED_VALIDATION_INFRASTRUCTURE: collaborator update validation errors")

    after = call("GetDatasetMetadata", {"ownerSlug": OWNER, "datasetSlug": DATASET_SLUG})
    after_info = after.get("info") if isinstance(after.get("info"), dict) else {}
    after_collaborators = normalize_collaborators(after_info.get("collaborators"))
    verified = any(
        str(row.get("username", "")).strip().lower() == READER
        and (row.get("role") == 1 or str(row.get("role", "")).upper() == "READER")
        for row in after_collaborators
    )
    if not verified:
        fail("BLOCKED_VALIDATION_INFRASTRUCTURE: trickermark READER verification failed")

    print(json.dumps({
        "status": "PASS",
        "dataset_ref": f"{OWNER}/{DATASET_SLUG}",
        "source_version": 4,
        "source_v4_status": "READY",
        "reader": READER,
        "role": "READER",
        "permission_only": True,
        "data_version_created": False,
        "training": False,
        "locked_test": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
