from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request

ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
ACCOUNT_ID = "kg-03"
OWNER = "rezanory"
DATASET_SLUG = "m07-gate-r320-state-v1-7"
VERSION = 4
FILES = [
    "CAMPAIGN_STATE.json",
    "FOLD_1_RECOVERY/FOLDS/fold_1/COMPLETED.json",
    "FOLD_2_RECOVERY/FOLDS/fold_2/COMPLETED.json",
    "FOLD_3_RECOVERY/FOLDS/fold_3/COMPLETED.json",
    "FOLD_4_RECOVERY/FOLDS/fold_4/COMPLETED.json",
    "FOLD_1_RECOVERY/FOLDS/fold_1/best.weights.h5.state.json",
    "FOLD_2_RECOVERY/FOLDS/fold_2/best.weights.h5.state.json",
    "FOLD_3_RECOVERY/FOLDS/fold_3/best.weights.h5.state.json",
    "FOLD_4_RECOVERY/FOLDS/fold_4/best.weights.h5.state.json",
]


def _broker_url(file_name: str) -> str:
    oidc = os.environ.get("CGP_ACTION_OIDC_TOKEN", "").strip()
    if len(oidc) < 100:
        raise RuntimeError("trusted action OIDC token missing")
    request_id = (
        "m07-r320-v4-download-"
        + hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:16]
        + "-"
        + os.environ.get("GITHUB_RUN_ID", "local")
    )
    payload = {
        "request_id": request_id,
        "provider": "kaggle",
        "operation_class": "privileged",
        "account_id": ACCOUNT_ID,
        "purpose": (
            "Read-only exact V1.7 R320 Version 4 receipt/hash reconciliation from the "
            "canonical kg-03/rezanory persistence owner; no upload, no training, no Locked Test."
        ),
        "service": "datasets.DatasetApiService",
        "method": "DownloadDataset",
        "body": {
            "ownerSlug": OWNER,
            "datasetSlug": DATASET_SLUG,
            "fileName": file_name,
            "datasetVersionNumber": VERSION,
            "raw": True,
        },
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {oidc}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "m07-r320-v4-receipt-probe/1.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            envelope = json.loads(response.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", "replace")
        raise RuntimeError(f"broker {file_name} HTTP {exc.code}: {detail[:800]}") from exc
    if not isinstance(envelope, dict) or not envelope.get("ok"):
        raise RuntimeError(f"broker rejected {file_name}: {json.dumps(envelope)[:800]}")
    result = envelope.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"broker result missing for {file_name}")
    url = str(result.get("url") or "")
    if not url.startswith("https://"):
        raise RuntimeError(f"broker redirect missing for {file_name}")
    return url


def _hash_fields(value: object, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, (dict, list)):
                out.update(_hash_fields(item, path))
            elif any(token in str(key).lower() for token in ("sha", "hash", "digest")) and isinstance(
                item, (str, int, float, bool)
            ):
                out[path] = item
    elif isinstance(value, list):
        for index, item in enumerate(value):
            out.update(_hash_fields(item, f"{prefix}[{index}]"))
    return out


def main() -> int:
    summary: list[dict[str, object]] = []
    for file_name in FILES:
        url = _broker_url(file_name)
        print(f"::add-mask::{url}")
        req = urllib.request.Request(url, headers={"User-Agent": "m07-r320-v4-receipt-probe/1.1"})
        with urllib.request.urlopen(req, timeout=120) as response:
            data = response.read(2_000_000)
        if len(data) >= 2_000_000:
            raise RuntimeError(f"bounded file too large: {file_name}")
        try:
            value = json.loads(data.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"invalid JSON: {file_name}") from exc
        record = value if isinstance(value, dict) else {}
        summary.append(
            {
                "file": file_name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "status": record.get("status"),
                "fold": record.get("fold"),
                "fold_index": record.get("fold_index"),
                "completed": record.get("completed"),
                "hash_fields": _hash_fields(value),
            }
        )
    print(
        "M07_R320_V4_RECEIPT_PROBE="
        + json.dumps(
            {
                "account_id": ACCOUNT_ID,
                "source": f"{OWNER}/{DATASET_SLUG}/versions/{VERSION}",
                "files": summary,
                "upload": False,
                "training": False,
                "locked_test": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
