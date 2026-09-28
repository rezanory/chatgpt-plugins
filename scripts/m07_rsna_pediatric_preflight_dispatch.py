from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

ACTION_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
READ_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/read/kaggle"
ACCOUNT = "kg-03"
OWNER = "rezanory"
DATASET_REF = "nih-chest-xrays/data"
EXPECTED_MANIFEST_SHA256 = "3450a101e626a09535d98e5aa9ca52dbf64ded584c2952f7613e3c22981cf5cd"
EXPECTED_COUNTS = {
    "expanded_images": 1099,
    "expanded_patients": 553,
    "expanded_negative": 578,
    "expanded_positive": 521,
    "primary_images": 282,
    "primary_patients": 158,
    "primary_negative": 146,
    "primary_positive": 136,
}


def post(endpoint: str, token: str, payload: dict, timeout: int = 240) -> dict:
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "m07-rsna-pediatric-preflight/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            return json.loads(raw or "{}")
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "http_status": exc.code,
            "error": exc.read(20000).decode("utf-8", "replace"),
        }


def extract_status(value):
    if isinstance(value, dict):
        for key in ("status", "kernelStatus", "state"):
            item = value.get(key)
            if isinstance(item, str) and item:
                upper = item.upper()
                if upper in {"QUEUED", "RUNNING", "COMPLETE", "ERROR", "CANCELLED"}:
                    return upper
        for item in value.values():
            status = extract_status(item)
            if status:
                return status
    elif isinstance(value, list):
        for item in value:
            status = extract_status(item)
            if status:
                return status
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not str(args.run_id).isdigit():
        raise SystemExit("RSNA_PREFLIGHT_RUN_ID_INVALID")

    action_token = os.environ["CGP_ACTION_OIDC_TOKEN"].strip()
    read_token = os.environ["CGP_READ_OIDC_TOKEN"].strip()
    if len(action_token) < 100 or len(read_token) < 100:
        raise SystemExit("RSNA_PREFLIGHT_OIDC_INVALID")

    repo = pathlib.Path(__file__).resolve().parents[1]
    temp = pathlib.Path(os.environ["RUNNER_TEMP"])
    manifest_path = repo / "evidence" / "m07_external" / "rsna_pediatric_v1" / "manifest.json"
    manifest = manifest_path.read_bytes()
    source = (repo / "scripts" / "m07_rsna_pediatric_preflight.py").read_text(encoding="utf-8")

    rows = json.loads(manifest.decode("utf-8"))
    canonical = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise SystemExit("RSNA_PREFLIGHT_MANIFEST_SHA_DRIFT")

    encoded = base64.b64encode(zlib.compress(manifest, 9)).decode("ascii")
    future = "from __future__ import annotations\n"
    if not source.startswith(future):
        raise SystemExit("RSNA_PREFLIGHT_SCRIPT_FUTURE_IMPORT_DRIFT")
    code = (
        future
        + f"EMBEDDED_MANIFEST_B64={encoded!r}\n"
        + source[len(future):]
    )

    slug = f"m07-rsna-pediatric-preflight-{args.run_id}"
    kernel_ref = f"{OWNER}/{slug}"
    payload = {
        "request_id": f"m07-rsna-pediatric-preflight-{args.run_id}",
        "provider": "kaggle",
        "operation_class": "compute",
        "account_id": ACCOUNT,
        "purpose": (
            "CPU-only integrity preflight for frozen-M07 secondary external validation "
            "on the pre-specified RSNA pediatric cohort. No GPU, training, inference, "
            "HPO, adaptation, calibration, or external threshold tuning."
        ),
        "service": "kernels.KernelsApiService",
        "method": "SaveKernel",
        "body": {
            "slug": kernel_ref,
            "newTitle": f"M07 RSNA Pediatric External Preflight {args.run_id}",
            "text": code,
            "language": "python",
            "kernelType": "script",
            "kernelExecutionType": "SAVE_AND_RUN_ALL",
            "isPrivate": True,
            "enableGpu": False,
            "enableTpu": False,
            "enableInternet": False,
            "kernelDataSources": [],
            "datasetDataSources": [DATASET_REF],
            "competitionDataSources": [],
            "modelDataSources": [],
        },
    }
    launch = post(ACTION_ENDPOINT, action_token, payload)
    (temp / "m07-rsna-preflight-launch.json").write_text(
        json.dumps(launch, indent=2), encoding="utf-8"
    )
    print("RSNA_PREFLIGHT_LAUNCH", json.dumps(launch, sort_keys=True), flush=True)
    provider_result = launch.get("result") if isinstance(launch.get("result"), dict) else {}
    provider_error = str(provider_result.get("error") or launch.get("error") or "")
    if not launch.get("ok") or provider_error:
        raise SystemExit("RSNA_PREFLIGHT_SUBMISSION_REJECTED")

    provider_ref_raw = str(launch.get("provider_ref") or provider_result.get("ref") or kernel_ref)
    provider_ref = provider_ref_raw.strip()
    if provider_ref.startswith("/code/"):
        provider_ref = provider_ref[len("/code/"):]
    provider_ref = provider_ref.strip("/")
    if provider_ref.count("/") != 1:
        raise SystemExit("RSNA_PREFLIGHT_PROVIDER_REF_INVALID=" + provider_ref_raw)
    history = []
    terminal = ""
    for attempt in range(1, 121):
        row = post(
            READ_ENDPOINT,
            read_token,
            {"action": "resolved_kernel_status", "account_id": ACCOUNT, "kernel_ref": provider_ref},
            timeout=120,
        )
        history.append(row)
        terminal = extract_status(row)
        print("RSNA_PREFLIGHT_STATUS", attempt, terminal or "UNKNOWN", flush=True)
        if terminal in {"COMPLETE", "ERROR", "CANCELLED"}:
            break
        time.sleep(15)

    (temp / "m07-rsna-preflight-status.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    if terminal != "COMPLETE":
        raise SystemExit("RSNA_PREFLIGHT_TERMINAL_" + (terminal or "TIMEOUT"))

    owner, kernel_slug = provider_ref.split("/", 1)
    items = []
    for page in range(1, 21):
        listing = post(
            READ_ENDPOINT,
            read_token,
            {
                "action": "raw_read",
                "account_id": ACCOUNT,
                "service": "kernels.KernelsApiService",
                "method": "ListKernelSessionOutput",
                "body": {
                    "userName": owner,
                    "kernelSlug": kernel_slug,
                    "page": page,
                    "pageSize": 100,
                },
            },
            timeout=120,
        )
        if not listing.get("ok"):
            raise SystemExit("RSNA_PREFLIGHT_OUTPUT_LIST_FAILED")
        files = (listing.get("result") or {}).get("files") or []
        items.extend(files)
        if len(files) < 100:
            break

    target = next(
        (
            item
            for item in items
            if str(item.get("fileName", "")).endswith("M07_RSNA_PEDIATRIC_PREFLIGHT.json")
        ),
        None,
    )
    if not target or not target.get("url"):
        raise SystemExit("RSNA_PREFLIGHT_RECEIPT_NOT_FOUND")
    url = str(target["url"])
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.kaggleusercontent.com":
        raise SystemExit("RSNA_PREFLIGHT_UNTRUSTED_OUTPUT_URL")

    request = urllib.request.Request(
        url, headers={"User-Agent": "m07-rsna-pediatric-preflight/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
    output = temp / "M07_RSNA_PEDIATRIC_PREFLIGHT.json"
    output.write_bytes(data)
    receipt = json.loads(data.decode("utf-8"))

    expected = {
        "schema": "m07.external.rsna_pediatric.preflight.v1",
        "status": "PASS_RSNA_PEDIATRIC_SCHEMA",
        "dataset_ref": DATASET_REF,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "counts": EXPECTED_COUNTS,
        "duplicate_image_sha256_count": 0,
        "training_performed": False,
        "hpo_performed": False,
        "inference_started": False,
        "external_threshold_tuning": False,
        "external_adaptation": False,
    }
    mismatches = {
        key: {"expected": value, "actual": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise SystemExit("RSNA_PREFLIGHT_RECEIPT_MISMATCH=" + json.dumps(mismatches, sort_keys=True))

    claimed = str(receipt.get("receipt_sha256") or "")
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if claimed != actual:
        raise SystemExit("RSNA_PREFLIGHT_RECEIPT_SHA_MISMATCH")

    print("M07_RSNA_PEDIATRIC_PREFLIGHT_PASS", json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
