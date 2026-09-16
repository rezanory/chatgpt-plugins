"""Read-only inventory of the cancelled M07 R320 producer output.

Never prints signed URLs, credentials, or the kernel log body. The only output is
safe file metadata needed to locate immutable Fold-4 state evidence.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.error
import urllib.request

OWNER = "rezanory"
KERNEL_SLUG = "m07-phase2-unlock-20260915"
ENDPOINT = "https://api.kaggle.com/v1/kernels.KernelsApiService/ListKernelSessionOutput"
OUT = pathlib.Path(os.environ.get("RUNNER_TEMP", ".")) / "m07-producer-output-manifest.json"


def auth_header() -> str:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("BLOCKED_VALIDATION_INFRASTRUCTURE: KAGGLE_API_TOKEN missing")
    if token.startswith("KGAT_"):
        return f"Bearer {token}"
    raw = base64.b64encode(f"{OWNER}:{token}".encode()).decode()
    return f"Basic {raw}"


def post(body: dict) -> dict:
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "Authorization": auth_header(),
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "m07-producer-output-inventory/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read(1200).decode("utf-8", "replace")
        raise SystemExit(f"PRODUCER_OUTPUT_HTTP_{exc.code}: {detail[:600]}") from exc


def main() -> int:
    files: list[dict] = []
    token = ""
    log_chars = None
    for page in range(20):
        body = {"userName": OWNER, "kernelSlug": KERNEL_SLUG, "pageSize": 100}
        if token:
            body["pageToken"] = token
        payload = post(body)
        if log_chars is None:
            log_chars = len(str(payload.get("log") or ""))
        rows = payload.get("files") or []
        for item in rows:
            if not isinstance(item, dict):
                continue
            name = str(item.get("fileName") or "").replace("\\", "/").strip()
            if not name:
                continue
            size = item.get("totalBytes")
            if size is None:
                size = item.get("bytes")
            try:
                size = int(size or 0)
            except (TypeError, ValueError):
                size = 0
            files.append({"file_name": name, "bytes": size})
        next_token = str(payload.get("nextPageToken") or payload.get("next_page_token") or "").strip()
        if not next_token or next_token == token:
            break
        token = next_token
    else:
        raise SystemExit("BLOCKED_VALIDATION_INFRASTRUCTURE: producer output pagination exceeded")

    files.sort(key=lambda row: row["file_name"])
    state_markers = [
        row for row in files
        if any(
            marker in row["file_name"]
            for marker in (
                "CAMPAIGN_STATE.json",
                "FOLD_1_RECOVERY.zip",
                "FOLD_2_RECOVERY.zip",
                "FOLD_3_RECOVERY.zip",
                "FOLD_4_RECOVERY.zip",
                "FOLD_5_RECOVERY.zip",
                "FOLD_4_RECOVERY/FOLDS/fold_4/COMPLETED.json",
                "FOLDS/fold_4/COMPLETED.json",
                "FINAL_REPORT.json",
                "PARTIAL_RUN_RECEIPT.json",
            )
        )
    ]
    result = {
        "schema": "m07.d260914d.producer.output.inventory.v1",
        "kernel_ref": f"{OWNER}/{KERNEL_SLUG}",
        "read_only": True,
        "file_count": len(files),
        "log_chars_not_emitted": log_chars,
        "state_markers": state_markers,
        "files": files,
        "signed_urls_returned": False,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "kernel_ref": result["kernel_ref"],
        "file_count": result["file_count"],
        "state_marker_count": len(state_markers),
        "state_markers": state_markers,
        "log_chars_not_emitted": log_chars,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
