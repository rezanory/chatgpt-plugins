from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
A06_ACCOUNT_ID = "kg-05"
A06_OWNER = "trickermark"
A06_KERNEL_SLUG = "m07-runtime-r320-a06-20260916-35097401885"
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
_ALLOWED_DOWNLOAD_HOSTS = {"api.kaggle.com", "www.kaggle.com", "storage.googleapis.com"}


def _safe_error_text(value: str) -> str:
    text = str(value or "")[:1200]
    for marker in ("KGAT_", "Bearer ", "Basic "):
        if marker in text:
            return "<redacted diagnostic containing credential-shaped material>"
    return text


def _broker_call(
    *,
    account_id: str,
    service: str,
    method: str,
    body: dict[str, object],
    request_tag: str,
    purpose: str,
) -> dict[str, object]:
    oidc = os.environ.get("CGP_ACTION_OIDC_TOKEN", "").strip()
    if len(oidc) < 100:
        raise RuntimeError("trusted action OIDC token missing")
    request_id = f"m07-{request_tag}-{os.environ.get('GITHUB_RUN_ID', 'local')}"
    payload = {
        "request_id": request_id,
        "provider": "kaggle",
        "operation_class": "privileged",
        "account_id": account_id,
        "purpose": purpose,
        "service": service,
        "method": method,
        "body": body,
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {oidc}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "m07-r320-v4-receipt-probe/1.2",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            envelope = json.loads(response.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as exc:
        detail = _safe_error_text(exc.read(4000).decode("utf-8", "replace"))
        raise RuntimeError(f"broker {method} HTTP {exc.code}: {detail}") from exc
    if not isinstance(envelope, dict) or not envelope.get("ok"):
        detail = _safe_error_text(json.dumps(envelope, ensure_ascii=True))
        raise RuntimeError(f"broker rejected {method}: {detail}")
    result = envelope.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"broker result missing for {method}")
    return result


def _live_a06_status() -> dict[str, object]:
    result = _broker_call(
        account_id=A06_ACCOUNT_ID,
        service="kernels.KernelsApiService",
        method="GetKernelSessionStatus",
        body={"userName": A06_OWNER, "kernelSlug": A06_KERNEL_SLUG},
        request_tag="a06-live-status",
        purpose=(
            "Read-only exact live status for legacy M07 R320 A06 under V1.7 reconciliation; "
            "no SaveKernel, no upload, no training, no Locked Test."
        ),
    )
    print(
        "M07_A06_LIVE_STATUS="
        + json.dumps(
            {
                "account_id": A06_ACCOUNT_ID,
                "kernel_ref": f"{A06_OWNER}/{A06_KERNEL_SLUG}",
                "result": result,
                "compute_launched": False,
                "upload": False,
                "training": False,
                "locked_test": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return result


def _list_exact_v4_files() -> set[str]:
    result = _broker_call(
        account_id=ACCOUNT_ID,
        service="datasets.DatasetApiService",
        method="ListDatasetFiles",
        body={
            "ownerSlug": OWNER,
            "datasetSlug": DATASET_SLUG,
            "datasetVersionNumber": VERSION,
            "pageSize": 100,
        },
        request_tag="r320-v4-list",
        purpose=(
            "Read-only exact V1.7 R320 Version 4 file reconciliation from canonical kg-03/rezanory; "
            "no upload, no training, no Locked Test."
        ),
    )
    raw_files = result.get("datasetFiles")
    if not isinstance(raw_files, list):
        raw_files = result.get("files")
    if not isinstance(raw_files, list):
        raise RuntimeError("ListDatasetFiles returned no datasetFiles/files list")
    names: set[str] = set()
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        name = str(item.get("ref") or item.get("name") or item.get("fileName") or "").strip()
        if name:
            names.add(name)
    missing = [name for name in FILES if name not in names]
    if missing:
        raise RuntimeError("exact v4 receipt files missing: " + json.dumps(missing, separators=(",", ":")))
    print(
        "M07_R320_V4_LISTING_OK="
        + json.dumps(
            {
                "account_id": ACCOUNT_ID,
                "source": f"{OWNER}/{DATASET_SLUG}/versions/{VERSION}",
                "required_file_count": len(FILES),
                "all_required_present": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return names


def _download_url(file_name: str) -> str:
    result = _broker_call(
        account_id=ACCOUNT_ID,
        service="datasets.DatasetApiService",
        method="DownloadDataset",
        body={
            "ownerSlug": OWNER,
            "datasetSlug": DATASET_SLUG,
            "fileName": file_name,
            "datasetVersionNumber": VERSION,
            "raw": True,
        },
        request_tag="r320-v4-download-" + hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:16],
        purpose=(
            "Read-only exact V1.7 R320 Version 4 receipt/hash reconciliation from canonical kg-03/rezanory; "
            "no upload, no training, no Locked Test."
        ),
    )
    url = str(result.get("url") or "")
    if not url.startswith("https://"):
        raise RuntimeError(f"broker redirect missing for {file_name}")
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        host in _ALLOWED_DOWNLOAD_HOSTS
        or host.endswith(".kaggleusercontent.com")
        or host.endswith(".googleusercontent.com")
    ):
        raise RuntimeError(f"download URL host is not allowlisted for {file_name}")
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
    try:
        _live_a06_status()
    except Exception as exc:
        print(
            "M07_A06_LIVE_STATUS_ERROR="
            + json.dumps(
                {
                    "error": _safe_error_text(repr(exc)),
                    "account_id": A06_ACCOUNT_ID,
                    "kernel_ref": f"{A06_OWNER}/{A06_KERNEL_SLUG}",
                    "compute_launched": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        raise

    try:
        _list_exact_v4_files()
        summary: list[dict[str, object]] = []
        for file_name in FILES:
            url = _download_url(file_name)
            print(f"::add-mask::{url}")
            req = urllib.request.Request(url, headers={"User-Agent": "m07-r320-v4-receipt-probe/1.2"})
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
    except Exception as exc:
        print(
            "M07_R320_V4_RECEIPT_PROBE_ERROR="
            + json.dumps(
                {
                    "error": _safe_error_text(repr(exc)),
                    "account_id": ACCOUNT_ID,
                    "source": f"{OWNER}/{DATASET_SLUG}/versions/{VERSION}",
                    "upload": False,
                    "training": False,
                    "locked_test": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
