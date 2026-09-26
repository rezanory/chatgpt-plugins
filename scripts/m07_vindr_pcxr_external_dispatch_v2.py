from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request

ACTION_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
READ_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/read/kaggle"
COMPETITION = "pediatric-cxr-analysis-challenge"
EXPECTED_SPLIT = "896491de87f9dc2a1d7d63548b7c5c22206da11f27a37efece8efc8e1557c8a9"
EXPECTED_RECIPE = "02c77dd257612e866756b16270f71816e61fb9f2403e14126aaacb9f01a8da0b"
EXPECTED_COUNTS = {"primary_n": 1077, "normal": 907, "pneumonia_family": 170}

CONTRACTS = {
    "M07_EXTERNAL_VINDR_R224_V2": {
        "resolution": 224,
        "account_id": "kg-03",
        "owner": "rezanory",
        "state_handle": "rezanory/m07-final-5fold-fix2-d260914d",
        "batch_size": 12,
    },
    "M07_EXTERNAL_VINDR_R320_V2": {
        "resolution": 320,
        "account_id": "kg-05",
        "owner": "trickermark",
        "state_handle": "trickermark/m07-gate-r320-state-v1-7",
        "batch_size": 6,
    },
    "M07_EXTERNAL_VINDR_R384_V2": {
        "resolution": 384,
        "account_id": "kg-05",
        "owner": "trickermark",
        "state_handle": "trickermark/m07-gate-r384-state-v1-7",
        "batch_size": 4,
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
            "User-Agent": "m07-vindr-external-v2/1.0",
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


def safe_download(url: str, target: pathlib.Path):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.kaggleusercontent.com":
        raise RuntimeError("UNTRUSTED_KAGGLE_OUTPUT_URL")
    request = urllib.request.Request(url, headers={"User-Agent": "m07-vindr-external-v2/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        target.write_bytes(response.read())


def verify_receipt(receipt: dict, contract: dict):
    resolution = int(contract["resolution"])
    expected = {
        "schema": "m07.external.vindr_pcxr.terminal.v2",
        "status": "SCIENTIFIC_RECEIPT_PASS",
        "resolution": resolution,
        "primary_n": EXPECTED_COUNTS["primary_n"],
        "normal": EXPECTED_COUNTS["normal"],
        "pneumonia_family": EXPECTED_COUNTS["pneumonia_family"],
        "training_performed": False,
        "hpo_performed": False,
        "external_threshold_tuning": False,
        "external_adaptation": False,
        "source_state": contract["state_handle"],
        "split_fingerprint": EXPECTED_SPLIT,
        "hpo_recipe_fingerprint": EXPECTED_RECIPE,
    }
    mismatches = {
        key: {"expected": value, "actual": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise RuntimeError("M07_VINDR_TERMINAL_RECEIPT_MISMATCH=" + json.dumps(mismatches, sort_keys=True))
    claimed = str(receipt.get("receipt_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", claimed):
        raise RuntimeError("M07_VINDR_TERMINAL_RECEIPT_SHA_MISSING")
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual != claimed:
        raise RuntimeError("M07_VINDR_TERMINAL_RECEIPT_SHA_MISMATCH")
    artifacts = receipt.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("M07_VINDR_TERMINAL_ARTIFACT_MANIFEST_MISSING")
    for name, digest in artifacts.items():
        if not isinstance(name, str) or not re.fullmatch(r"[0-9a-f]{64}", str(digest or "")):
            raise RuntimeError("M07_VINDR_TERMINAL_ARTIFACT_MANIFEST_INVALID")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.token not in CONTRACTS:
        raise SystemExit("M07_VINDR_EXTERNAL_TOKEN_INVALID")
    if not str(args.run_id).isdigit():
        raise SystemExit("M07_VINDR_GITHUB_RUN_ID_INVALID")

    contract = CONTRACTS[args.token]
    resolution = int(contract["resolution"])
    account_id = str(contract["account_id"])
    owner = str(contract["owner"])
    state_handle = str(contract["state_handle"])
    batch_size = int(contract["batch_size"])

    action_token = os.environ["CGP_ACTION_OIDC_TOKEN"].strip()
    read_token = os.environ["CGP_READ_OIDC_TOKEN"].strip()
    if len(action_token) < 100 or len(read_token) < 100:
        raise SystemExit("M07_VINDR_OIDC_TOKEN_INVALID")

    root = pathlib.Path(__file__).resolve().parents[1]
    runner_temp = pathlib.Path(os.environ["RUNNER_TEMP"])
    source = (root / "scripts" / "m07_vindr_pcxr_external_v2.py").read_text(encoding="utf-8")
    future = "from __future__ import annotations\n"
    if not source.startswith(future):
        raise SystemExit("M07_VINDR_V2_FUTURE_IMPORT_DRIFT")
    preamble = (
        future
        + "import os\n"
        + f"os.environ['M07_EXTERNAL_RESOLUTION']={str(resolution)!r}\n"
        + f"os.environ['M07_EXTERNAL_STATE_HANDLE']={state_handle!r}\n"
        + f"os.environ['M07_EXTERNAL_BATCH_SIZE']={str(batch_size)!r}\n"
    )
    code = preamble + source[len(future):]

    slug = f"m07-vindr-pcxr-r{resolution}-v2-20260927-{args.run_id}"
    kernel_ref = f"{owner}/{slug}"
    launch_payload = {
        "request_id": f"m07-vindr-pcxr-r{resolution}-v2-{args.run_id}",
        "provider": "kaggle",
        "operation_class": "compute",
        "account_id": account_id,
        "purpose": (
            f"Frozen M07 R{resolution} external validation on VinDr-PCXR. "
            "Inference only; no HPO, retraining, adaptation, external threshold tuning, or model selection."
        ),
        "service": "kernels.KernelsApiService",
        "method": "SaveKernel",
        "body": {
            "slug": kernel_ref,
            "newTitle": f"M07 VinDr-PCXR External R{resolution} V2 {args.run_id}",
            "text": code,
            "language": "python",
            "kernelType": "script",
            "kernelExecutionType": "SAVE_AND_RUN_ALL",
            "isPrivate": True,
            "enableGpu": True,
            "enableTpu": False,
            "enableInternet": True,
            "kernelDataSources": [],
            "datasetDataSources": [state_handle],
            "competitionDataSources": [COMPETITION],
            "modelDataSources": [],
        },
    }
    launch = post_json(ACTION_ENDPOINT, action_token, launch_payload)
    launch_path = runner_temp / f"m07-vindr-r{resolution}-launch.json"
    write_json(launch_path, launch)
    print("M07_VINDR_EXTERNAL_LAUNCH", json.dumps(launch, sort_keys=True), flush=True)

    provider_result = launch.get("result") if isinstance(launch.get("result"), dict) else {}
    provider_error = str(provider_result.get("error") or launch.get("error") or "")
    if not launch.get("ok") or provider_error:
        if "accept this competition's rules" in provider_error.lower():
            raise SystemExit("M07_VINDR_COMPETITION_RULES_NOT_ACCEPTED")
        raise SystemExit("M07_VINDR_EXTERNAL_SUBMISSION_REJECTED")

    provider_ref = str(
        launch.get("provider_ref")
        or provider_result.get("ref")
        or kernel_ref
    )
    if "/" not in provider_ref or provider_ref.split("/", 1)[0].casefold() != owner.casefold():
        raise SystemExit("M07_VINDR_PROVIDER_REF_IDENTITY_MISMATCH")

    history = []
    terminal = ""
    for attempt in range(1, 481):
        row = post_json(
            READ_ENDPOINT,
            read_token,
            {"action": "kernel_status", "account_id": account_id, "kernel_ref": provider_ref},
            timeout=120,
        )
        history.append(row)
        terminal = recursive_status(row)
        print(
            "M07_VINDR_EXTERNAL_STATUS",
            json.dumps(
                {"resolution": resolution, "attempt": attempt, "status": terminal or "UNKNOWN"},
                sort_keys=True,
            ),
            flush=True,
        )
        if terminal in {"COMPLETE", "ERROR", "CANCELLED"}:
            break
        time.sleep(20)

    status_path = runner_temp / f"m07-vindr-r{resolution}-status.json"
    write_json(status_path, history)
    if terminal != "COMPLETE":
        raise SystemExit("M07_VINDR_EXTERNAL_TERMINAL_" + (terminal or "TIMEOUT"))

    items = []
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
                    "userName": provider_ref.split("/", 1)[0],
                    "kernelSlug": provider_ref.split("/", 1)[1],
                    "page": page,
                    "pageSize": 100,
                },
            },
            timeout=120,
        )
        if not listing.get("ok"):
            raise SystemExit("M07_VINDR_OUTPUT_LIST_FAILED")
        files = (listing.get("result") or {}).get("files") or []
        items.extend(files)
        if len(files) < 100:
            break

    receipt_suffix = "M07_VINDR_EXTERNAL_TERMINAL_RECEIPT.json"
    zip_suffix = f"M07_VINDR_PCXR_EXTERNAL_R{resolution}_V2_COMPLETE.zip"
    receipt_item = next(
        (x for x in items if str(x.get("fileName", "")).endswith(receipt_suffix)),
        None,
    )
    zip_item = next(
        (x for x in items if str(x.get("fileName", "")).endswith(zip_suffix)),
        None,
    )
    if not receipt_item or not receipt_item.get("url"):
        raise SystemExit("M07_VINDR_TERMINAL_RECEIPT_NOT_FOUND")
    if not zip_item or not zip_item.get("url"):
        raise SystemExit("M07_VINDR_COMPLETE_ZIP_NOT_FOUND")

    receipt_path = runner_temp / f"M07_VINDR_R{resolution}_TERMINAL_RECEIPT.json"
    zip_path = runner_temp / zip_suffix
    safe_download(str(receipt_item["url"]), receipt_path)
    safe_download(str(zip_item["url"]), zip_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verify_receipt(receipt, contract)

    zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    summary = {
        "schema": "m07.external.vindr_pcxr.github_handoff.v2",
        "status": "SCIENTIFIC_RECEIPT_PASS",
        "resolution": resolution,
        "account_id": account_id,
        "owner": owner,
        "kernel_ref": provider_ref,
        "state_handle": state_handle,
        "terminal_receipt_sha256": receipt["receipt_sha256"],
        "complete_zip_sha256": zip_sha,
        "primary_n": receipt["primary_n"],
        "training_performed": False,
        "hpo_performed": False,
        "external_threshold_tuning": False,
        "external_adaptation": False,
    }
    summary_path = runner_temp / f"M07_VINDR_R{resolution}_GITHUB_HANDOFF.json"
    write_json(summary_path, summary)
    print("M07_VINDR_EXTERNAL_SCIENTIFIC_PASS", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
