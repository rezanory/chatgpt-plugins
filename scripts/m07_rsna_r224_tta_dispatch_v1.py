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
DATASET_REF = "nih-chest-xrays/data"
EXPECTED_MANIFEST_SHA256 = "3450a101e626a09535d98e5aa9ca52dbf64ded584c2952f7613e3c22981cf5cd"
EXPECTED_SPLIT = "896491de87f9dc2a1d7d63548b7c5c22206da11f27a37efece8efc8e1557c8a9"
EXPECTED_RECIPE = "02c77dd257612e866756b16270f71816e61fb9f2403e14126aaacb9f01a8da0b"
EXPECTED_COUNTS = {
    "primary_n": 282,
    "primary_normal": 146,
    "primary_lung_opacity": 136,
    "expanded_n": 1099,
    "expanded_normal": 578,
    "expanded_lung_opacity": 521,
}

CONTRACTS = {
    "M07_EXTERNAL_RSNA_R224_TTA_V1": {
        "resolution": 224,
        "account_id": "kg-03",
        "owner": "rezanory",
        "state_handle": "rezanory/m07-final-5fold-fix2-d260914d",
        "batch_size": 12,
    },
}


def write_json(path: pathlib.Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def post_json(endpoint: str, token: str, payload: dict, timeout: int = 240) -> dict:
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "m07-rsna-external-v1/1.0",
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


def issue_oidc(audience: str) -> str:
    base = os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
    bearer = os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]
    sep = "&" if "?" in base else "?"
    url = base + sep + urllib.parse.urlencode({"audience": audience})
    request = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + bearer}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        token = json.load(response)["value"]
    if len(token) < 100:
        raise RuntimeError("OIDC_TOKEN_UNAVAILABLE=" + audience)
    print("::add-mask::" + token)
    return token


def recursive_status(value):
    if isinstance(value, dict):
        for key in ("status", "kernelStatus", "state"):
            item = value.get(key)
            if isinstance(item, str):
                upper = item.upper()
                if upper in {"QUEUED", "RUNNING", "COMPLETE", "ERROR", "CANCELLED"}:
                    return upper
        for item in value.values():
            status = recursive_status(item)
            if status:
                return status
    elif isinstance(value, list):
        for item in value:
            status = recursive_status(item)
            if status:
                return status
    return ""


def normalize_kernel_ref(value: str, fallback: str, expected_owner: str) -> str:
    raw = str(value or fallback).strip()
    ref = raw
    if ref.startswith("/code/"):
        ref = ref[len("/code/"):]
    ref = ref.strip("/")
    if ref.count("/") != 1:
        raise RuntimeError("RSNA_PROVIDER_REF_INVALID=" + raw)
    if ref.split("/", 1)[0].casefold() != expected_owner.casefold():
        raise RuntimeError("RSNA_PROVIDER_REF_OWNER_MISMATCH=" + raw)
    return ref


def safe_download(url: str, target: pathlib.Path):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.kaggleusercontent.com":
        raise RuntimeError("UNTRUSTED_KAGGLE_OUTPUT_URL")
    request = urllib.request.Request(
        url, headers={"User-Agent": "m07-rsna-external-v1/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        target.write_bytes(response.read())


def verify_receipt(receipt: dict, contract: dict):
    expected = {
        "schema": "m07.external.rsna_pediatric.r224_tta.terminal.v1",
        "status": "PASS_TTA_ANALYSIS",
        "resolution": 224,
        "expanded_n": EXPECTED_COUNTS["expanded_n"],
        "expanded_patients": 553,
        "primary_n": EXPECTED_COUNTS["primary_n"],
        "training_performed": False,
        "hpo_performed": False,
        "external_threshold_tuning": False,
        "external_model_adaptation": False,
        "external_calibration_fitting": False,
        "test_time_augmentation": True,
        "raw_external_result_preserved": True,
        "canonical_replay_confusion": {"tn": 400, "fp": 178, "fn": 37, "tp": 484},
        "source_state": contract["state_handle"],
        "split_fingerprint": EXPECTED_SPLIT,
        "hpo_recipe_fingerprint": EXPECTED_RECIPE,
        "external_dataset": DATASET_REF,
        "external_manifest_sha256": EXPECTED_MANIFEST_SHA256,
    }
    mismatches = {
        key: {"expected": value, "actual": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "M07_RSNA_TTA_TERMINAL_RECEIPT_MISMATCH="
            + json.dumps(mismatches, sort_keys=True)
        )
    views = receipt.get("tta_views")
    expected_views = [
        "canonical",
        "gamma_085",
        "gamma_115",
        "contrast_085",
        "contrast_115",
        "brightness_m10",
        "brightness_p10",
    ]
    if views != expected_views:
        raise RuntimeError("M07_RSNA_TTA_VIEWS_MISMATCH")
    claimed = str(receipt.get("receipt_sha256") or "")
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if claimed != actual:
        raise RuntimeError("M07_RSNA_TTA_TERMINAL_RECEIPT_SHA_MISMATCH")
    artifacts = receipt.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("M07_RSNA_TTA_TERMINAL_ARTIFACT_MANIFEST_MISSING")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.token not in CONTRACTS:
        raise SystemExit("M07_RSNA_EXTERNAL_TOKEN_INVALID")
    if not str(args.run_id).isdigit():
        raise SystemExit("M07_RSNA_GITHUB_RUN_ID_INVALID")

    contract = CONTRACTS[args.token]
    resolution = int(contract["resolution"])
    account_id = str(contract["account_id"])
    owner = str(contract["owner"])
    state_handle = str(contract["state_handle"])
    batch_size = int(contract["batch_size"])

    action_token = os.environ["CGP_ACTION_OIDC_TOKEN"].strip()
    read_token = os.environ["CGP_READ_OIDC_TOKEN"].strip()
    if len(action_token) < 100 or len(read_token) < 100:
        raise SystemExit("M07_RSNA_OIDC_TOKEN_INVALID")

    root = pathlib.Path(__file__).resolve().parents[1]
    runner_temp = pathlib.Path(os.environ["RUNNER_TEMP"])
    source = (
        root / "scripts" / "m07_rsna_r224_tta_v1.py"
    ).read_text(encoding="utf-8")
    manifest_path = (
        root / "evidence" / "m07_external" / "rsna_pediatric_v1" / "manifest.json"
    )
    manifest = manifest_path.read_bytes()
    rows = json.loads(manifest.decode("utf-8"))
    canonical = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise SystemExit("M07_RSNA_MANIFEST_SHA_DRIFT")
    encoded = base64.b64encode(zlib.compress(manifest, 9)).decode("ascii")

    future = "from __future__ import annotations\n"
    if not source.startswith(future):
        raise SystemExit("M07_RSNA_FUTURE_IMPORT_DRIFT")
    preamble = (
        future
        + "import os\n"
        + f"EMBEDDED_MANIFEST_B64={encoded!r}\n"
        + f"os.environ['M07_EXTERNAL_RESOLUTION']={str(resolution)!r}\n"
        + f"os.environ['M07_EXTERNAL_STATE_HANDLE']={state_handle!r}\n"
        + f"os.environ['M07_EXTERNAL_BATCH_SIZE']={str(batch_size)!r}\n"
    )
    code = preamble + source[len(future):]

    slug = f"m07-rsna-r224-tta-v1-{args.run_id}"
    kernel_ref = f"{owner}/{slug}"
    launch_path = runner_temp / "m07-rsna-r224-tta-launch.json"

    # Idempotent recovery gate: reuse the exact run if it already exists.
    existing = post_json(
        READ_ENDPOINT,
        read_token,
        {
            "action": "resolved_kernel_status",
            "account_id": account_id,
            "kernel_ref": kernel_ref,
        },
        timeout=120,
    )
    existing_status = recursive_status(existing)
    transient_probe_status = int(existing.get("http_status") or 0)
    if transient_probe_status in {408, 425, 429, 500, 502, 503, 504}:
        write_json(
            launch_path,
            {
                "ok": False,
                "reused_existing": False,
                "existing_probe": existing,
                "kernel_ref": kernel_ref,
            },
        )
        raise SystemExit(
            f"M07_RSNA_EXISTING_KERNEL_PROBE_TRANSIENT_HTTP_{transient_probe_status}"
        )

    if existing_status in {"ERROR", "CANCELLED"}:
        write_json(
            launch_path,
            {
                "ok": False,
                "reused_existing": True,
                "existing_status": existing_status,
                "kernel_ref": kernel_ref,
            },
        )
        raise SystemExit("M07_RSNA_EXISTING_KERNEL_TERMINAL_" + existing_status)

    if existing_status in {"QUEUED", "RUNNING", "COMPLETE"}:
        provider_ref = kernel_ref
        compute_mode = "REUSED_EXISTING"
        launch = {
            "ok": True,
            "reused_existing": True,
            "existing_status": existing_status,
            "provider_ref": provider_ref,
            "account_id": account_id,
            "resolution": resolution,
            "cgp_compute_mode": compute_mode,
        }
        write_json(launch_path, launch)
        print(
            "M07_RSNA_EXTERNAL_REUSE",
            json.dumps(launch, sort_keys=True),
            flush=True,
        )
    else:
        launch_payload = {
            "request_id": f"m07-rsna-r224-tta-v1-{args.run_id}",
            "provider": "kaggle",
            "operation_class": "compute",
            "account_id": account_id,
            "purpose": (
                "Frozen M07 R224 secondary RSNA TTA analysis. Inference only; seven "
                "predefined deterministic intensity transforms; no retraining, model-weight "
                "adaptation, calibration fitting, or external threshold tuning."
            ),
            "service": "kernels.KernelsApiService",
            "method": "SaveKernel",
            "body": {
                "slug": kernel_ref,
                "newTitle": f"M07 RSNA R224 TTA V1 {args.run_id}",
                "text": code,
                "language": "python",
                "kernelType": "script",
                "kernelExecutionType": "SAVE_AND_RUN_ALL",
                "isPrivate": True,
                "enableGpu": True,
                "enableTpu": False,
                "enableInternet": False,
                "kernelDataSources": [],
                "datasetDataSources": [state_handle, DATASET_REF],
                "competitionDataSources": [],
                "modelDataSources": [],
            },
        }
        compute_mode = "GPU"
        launch = post_json(ACTION_ENDPOINT, action_token, launch_payload)
        provider_result = (
            launch.get("result")
            if isinstance(launch.get("result"), dict)
            else {}
        )
        provider_error = str(
            provider_result.get("error") or launch.get("error") or ""
        )
        quota_error = "maximum weekly gpu quota" in provider_error.casefold()
        if quota_error:
            write_json(launch_path, launch)
            raise SystemExit("M07_RSNA_TTA_GPU_QUOTA_EXHAUSTED")
        launch["cgp_compute_mode"] = compute_mode
        write_json(launch_path, launch)
        print(
            "M07_RSNA_EXTERNAL_LAUNCH",
            json.dumps(launch, sort_keys=True),
            flush=True,
        )

        if not launch.get("ok") or provider_error:
            raise SystemExit("M07_RSNA_EXTERNAL_SUBMISSION_REJECTED")

        provider_ref = normalize_kernel_ref(
            launch.get("provider_ref") or provider_result.get("ref") or kernel_ref,
            kernel_ref,
            owner,
        )

    history = []
    terminal = ""
    last_refresh = time.monotonic()
    for attempt in range(1, 481):
        if time.monotonic() - last_refresh > 300:
            read_token = issue_oidc("cgp-control-plane-v3")
            last_refresh = time.monotonic()
        row = post_json(
            READ_ENDPOINT,
            read_token,
            {
                "action": "resolved_kernel_status",
                "account_id": account_id,
                "kernel_ref": provider_ref,
            },
            timeout=120,
        )
        if (
            row.get("http_status") == 403
            and "jwt expired" in str(row.get("error") or "").casefold()
        ):
            read_token = issue_oidc("cgp-control-plane-v3")
            last_refresh = time.monotonic()
            row = post_json(
                READ_ENDPOINT,
                read_token,
                {
                    "action": "resolved_kernel_status",
                    "account_id": account_id,
                    "kernel_ref": provider_ref,
                },
                timeout=120,
            )
        history.append(row)
        terminal = recursive_status(row)
        print(
            "M07_RSNA_EXTERNAL_STATUS",
            json.dumps(
                {"resolution": resolution, "attempt": attempt, "status": terminal or "UNKNOWN"},
                sort_keys=True,
            ),
            flush=True,
        )
        if terminal in {"COMPLETE", "ERROR", "CANCELLED"}:
            break
        time.sleep(20)

    status_path = runner_temp / "m07-rsna-r224-tta-status.json"
    write_json(status_path, history)
    if terminal != "COMPLETE":
        raise SystemExit("M07_RSNA_EXTERNAL_TERMINAL_" + (terminal or "TIMEOUT"))

    if time.monotonic() - last_refresh > 240:
        read_token = issue_oidc("cgp-control-plane-v3")

    items = []
    kernel_owner, kernel_slug = provider_ref.split("/", 1)
    for page in range(1, 31):
        listing = post_json(
            READ_ENDPOINT,
            read_token,
            {
                "action": "raw_read",
                "account_id": account_id,
                "service": "kernels.KernelsApiService",
                "method": "ListKernelSessionOutput",
                "body": {
                    "userName": kernel_owner,
                    "kernelSlug": kernel_slug,
                    "page": page,
                    "pageSize": 100,
                },
            },
            timeout=120,
        )
        if not listing.get("ok"):
            if (
                listing.get("http_status") == 403
                and "jwt expired" in str(listing.get("error") or "").casefold()
            ):
                read_token = issue_oidc("cgp-control-plane-v3")
                listing = post_json(
                    READ_ENDPOINT,
                    read_token,
                    {
                        "action": "raw_read",
                        "account_id": account_id,
                        "service": "kernels.KernelsApiService",
                        "method": "ListKernelSessionOutput",
                        "body": {
                            "userName": kernel_owner,
                            "kernelSlug": kernel_slug,
                            "page": page,
                            "pageSize": 100,
                        },
                    },
                    timeout=120,
                )
            if not listing.get("ok"):
                raise SystemExit("M07_RSNA_OUTPUT_LIST_FAILED")
        files = (listing.get("result") or {}).get("files") or []
        items.extend(files)
        if len(files) < 100:
            break

    receipt_suffix = "M07_RSNA_R224_TTA_TERMINAL_RECEIPT.json"
    report_suffix = "M07_RSNA_R224_TTA_REPORT.json"
    zip_suffix = "M07_RSNA_PEDIATRIC_R224_TTA_V1_COMPLETE.zip"
    receipt_item = next(
        (x for x in items if str(x.get("fileName", "")).endswith(receipt_suffix)),
        None,
    )
    report_item = next(
        (x for x in items if str(x.get("fileName", "")).endswith(report_suffix)),
        None,
    )
    zip_item = next(
        (x for x in items if str(x.get("fileName", "")).endswith(zip_suffix)),
        None,
    )
    if not receipt_item or not receipt_item.get("url"):
        raise SystemExit("M07_RSNA_TTA_TERMINAL_RECEIPT_NOT_FOUND")
    if not report_item or not report_item.get("url"):
        raise SystemExit("M07_RSNA_TTA_REPORT_NOT_FOUND")
    if not zip_item or not zip_item.get("url"):
        raise SystemExit("M07_RSNA_TTA_COMPLETE_ZIP_NOT_FOUND")

    receipt_path = runner_temp / "M07_RSNA_R224_TTA_TERMINAL_RECEIPT.json"
    report_path = runner_temp / "M07_RSNA_R224_TTA_REPORT.json"
    zip_path = runner_temp / zip_suffix
    safe_download(str(receipt_item["url"]), receipt_path)
    safe_download(str(report_item["url"]), report_path)
    safe_download(str(zip_item["url"]), zip_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verify_receipt(receipt, contract)
    if report.get("status") != "PASS_TTA_ANALYSIS":
        raise RuntimeError("M07_RSNA_TTA_REPORT_STATUS_INVALID")

    zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    summary = {
        "schema": "m07.external.rsna_pediatric.r224_tta.github_handoff.v1",
        "status": "PASS_TTA_ANALYSIS",
        "analysis_classification": "SECONDARY_EXTERNAL_EXPLORATORY_TTA",
        "resolution": resolution,
        "account_id": account_id,
        "owner": owner,
        "kernel_ref": provider_ref,
        "state_handle": state_handle,
        "compute_mode": compute_mode,
        "external_dataset": DATASET_REF,
        "external_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "terminal_receipt_sha256": receipt["receipt_sha256"],
        "complete_zip_sha256": zip_sha,
        "expanded_n": EXPECTED_COUNTS["expanded_n"],
        "primary_n": EXPECTED_COUNTS["primary_n"],
        "training_performed": False,
        "hpo_performed": False,
        "external_threshold_tuning": False,
        "external_model_adaptation": False,
        "external_calibration_fitting": False,
        "test_time_augmentation": True,
        "raw_external_result_preserved": True,
        "canonical_metrics": (report.get("canonical_replay_expanded") or {}).get("metrics"),
        "tta_metrics": (report.get("tta_equal_expanded") or {}).get("metrics"),
        "delta_tta_minus_canonical": report.get("delta_tta_minus_canonical"),
        "canonical_view_position_metrics": report.get("canonical_view_position_metrics"),
        "tta_view_position_metrics": report.get("tta_view_position_metrics"),
    }
    summary_path = runner_temp / "M07_RSNA_R224_TTA_GITHUB_HANDOFF.json"
    write_json(summary_path, summary)
    print("M07_RSNA_R224_TTA_PASS", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
