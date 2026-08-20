from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path

REPO = os.environ.get("GITHUB_REPOSITORY", "rezanory/chatgpt-plugins")
ISSUE = 18
EXPECTED_NOTEBOOK_SHA = "5abff78b9b10a13792b6d1b6cd24df780a64d16b7ec891d2ed74628dbb0b910b"
EXPECTED_SOURCE_ZIP_SHA = "ce3623b87ca221a52bcf886836070ebaebc490cf7574f6f3ce4db78095a32cde"
RAW_DATASET = "paultimothymooney/chest-xray-pneumonia"
KAGGLE_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService"
TARGET_SLUG = "pneumonia-v6-2-3-p0-sandbox"
TARGET_TITLE = "PNEUMONIA V6.2.3 P0 SINGLE MODEL SANDBOX"
ACTIVE = {"RUNNING", "QUEUED", "STARTING", "PENDING"}
FAILURE = {"ERROR", "FAILED", "CANCELLED", "CANCELED", "DEAD"}

ACCOUNTS = [
    ("kg-02", "radlinaradlina", "K_KG02"),
    ("kg-03", "rezanory", "K_KG03"),
    ("kg-04", "reyhanehazad", "K_KG04"),
    ("kg-05", "trickermark", "K_KG05"),
    ("kg-06", "msdenis", "K_KG06"),
    ("kg-07", "nisabulutmark", "K_KG07"),
    ("master", "azadka", "K_MASTER"),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def auth_header(username: str, token: str) -> str:
    if token.startswith("KGAT_"):
        return f"Bearer {token}"
    return "Basic " + base64.b64encode(f"{username}:{token}".encode()).decode()


def post_json(url: str, username: str, token: str, payload: dict, timeout: int = 60) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": auth_header(username, token),
            "Content-Type": "application/json",
            "User-Agent": "v623-p0-sandbox-control/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        value = json.loads(raw) if raw else {}
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}: {str(value)[:500]}")
        if isinstance(value, dict) and isinstance(value.get("code"), int) and value["code"] >= 400:
            raise RuntimeError(str(value.get("message", value))[:500])
        return value


def seconds(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return float(value.get("seconds", 0) or 0) + float(value.get("nanos", 0) or 0) / 1e9
    text = str(value).strip()
    m = re.fullmatch(r"(-?[0-9]+(?:\.[0-9]+)?)s", text)
    if m:
        return float(m.group(1))
    try:
        return float(text)
    except Exception:
        return None


def quota_remaining_hours(username: str, token: str) -> float | None:
    value = post_json(f"{KAGGLE_ROOT}/GetAcceleratorQuotaStatistics", username, token, {})
    q = value.get("gpuQuota") or value.get("gpu_quota")
    if not isinstance(q, dict):
        return None
    used = seconds(q.get("timeUsed") if "timeUsed" in q else q.get("time_used"))
    total = seconds(q.get("totalTimeAllowed") if "totalTimeAllowed" in q else q.get("total_time_allowed"))
    if used is None or total is None:
        return None
    return max(0.0, (total - used) / 3600.0)


def normalized_status(value: dict) -> str:
    session = value.get("session") if isinstance(value.get("session"), dict) else {}
    for candidate in (
        value.get("status"), value.get("statusName"), value.get("status_name"), value.get("state"),
        value.get("sessionStatus"), value.get("kernelSessionStatus"), session.get("status"), session.get("statusName"), session.get("state")
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().upper()
    return "UNKNOWN"


def read_source(source_dir: Path) -> tuple[bytes, str, str]:
    notebook = source_dir / "w16-kernel.ipynb"
    if not notebook.is_file():
        raise RuntimeError(f"Missing recovered W16 notebook: {notebook}")
    notebook_bytes = notebook.read_bytes()
    notebook_sha = sha256(notebook_bytes)
    if notebook_sha != EXPECTED_NOTEBOOK_SHA:
        raise RuntimeError(f"W16 notebook SHA drift: {notebook_sha}")
    value = json.loads(notebook_bytes.decode("utf-8"))
    code = "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
        for cell in value.get("cells", []) if isinstance(cell, dict) and cell.get("cell_type") == "code"
    )
    match = re.search(r"ARCHIVE_B64\s*=\s*['\"]([A-Za-z0-9+/=]+)['\"]", code, flags=re.S)
    if not match:
        raise RuntimeError("ARCHIVE_B64 not found in recovered W16 source")
    source_zip = base64.b64decode(match.group(1))
    source_sha = sha256(source_zip)
    if source_sha != EXPECTED_SOURCE_ZIP_SHA:
        raise RuntimeError(f"W16 source ZIP SHA drift: {source_sha}")
    return source_zip, notebook_sha, source_sha


def choose_account() -> tuple[str, str, str, float | None, list[dict]]:
    rows = []
    candidates = []
    for account_id, owner, env_name in ACCOUNTS:
        token = (os.environ.get(env_name) or "").strip()
        if not token:
            rows.append({"account_id": account_id, "owner": owner, "status": "TOKEN_MISSING"})
            continue
        try:
            remaining = quota_remaining_hours(owner, token)
            rows.append({"account_id": account_id, "owner": owner, "status": "OK", "gpu_remaining_hours": remaining})
            score = remaining if remaining is not None else -1.0
            if account_id == "master":
                score -= 0.05
            candidates.append((score, account_id, owner, token, remaining))
        except Exception as exc:
            rows.append({"account_id": account_id, "owner": owner, "status": "QUOTA_ERROR", "error": f"{type(exc).__name__}: {exc}"[:300]})
    if not candidates:
        raise RuntimeError("No Kaggle credential available for P0")
    candidates.sort(reverse=True, key=lambda x: x[0])
    _, account_id, owner, token, remaining = candidates[0]
    return account_id, owner, token, remaining, rows


def find_existing(owner: str, token: str) -> dict | None:
    value = post_json(f"{KAGGLE_ROOT}/ListKernels", owner, token, {
        "group": "PROFILE", "sortBy": "DATE_RUN", "pageSize": 100, "search": TARGET_SLUG,
    })
    for item in value.get("kernels", []) if isinstance(value.get("kernels"), list) else []:
        if isinstance(item, dict) and str(item.get("ref", "")).lower() == f"{owner}/{TARGET_SLUG}".lower():
            return item
    return None


def get_status(owner: str, token: str) -> str:
    value = post_json(f"{KAGGLE_ROOT}/GetKernelSessionStatus", owner, token, {"userName": owner, "kernelSlug": TARGET_SLUG})
    return normalized_status(value)


def save_kernel(owner: str, token: str, script: str) -> dict:
    request = {
        "slug": f"{owner}/{TARGET_SLUG}",
        "newTitle": TARGET_TITLE,
        "text": script,
        "language": "python",
        "kernelType": "script",
        "kernelExecutionType": "SAVE_AND_RUN_ALL",
        "isPrivate": True,
        "enableGpu": True,
        "enableTpu": False,
        "enableInternet": True,
        "kernelDataSources": [],
        "datasetDataSources": [RAW_DATASET],
        "competitionDataSources": [],
        "modelDataSources": [],
    }
    value = post_json(f"{KAGGLE_ROOT}/SaveKernel", owner, token, request, timeout=180)
    bad_d = value.get("invalidDatasetSources") if isinstance(value.get("invalidDatasetSources"), list) else []
    bad_k = value.get("invalidKernelSources") if isinstance(value.get("invalidKernelSources"), list) else []
    if bad_d or bad_k:
        raise RuntimeError(f"SaveKernel rejected sources: datasets={bad_d} kernels={bad_k}")
    return value


def list_outputs(owner: str, token: str) -> list[dict]:
    value = post_json(f"{KAGGLE_ROOT}/ListKernelSessionOutput", owner, token, {"userName": owner, "kernelSlug": TARGET_SLUG, "pageSize": 100}, timeout=120)
    files = value.get("files") if isinstance(value.get("files"), list) else []
    return [x for x in files if isinstance(x, dict)]


def fetch_output_json(files: list[dict], suffix: str) -> dict:
    matches = [x for x in files if str(x.get("fileName", "")).endswith(suffix) and isinstance(x.get("url"), str)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one output {suffix}, found {len(matches)}")
    url = matches[0]["url"]
    parsed = urllib.parse.urlparse(url)
    allowed = (
        parsed.scheme == "https" and (
            parsed.hostname in {"api.kaggle.com", "www.kaggle.com", "storage.googleapis.com"}
            or str(parsed.hostname).endswith(".kaggleusercontent.com")
            or str(parsed.hostname).endswith(".googleusercontent.com")
        )
    )
    if not allowed:
        raise RuntimeError(f"Output URL host not allowlisted: {parsed.hostname}")
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def post_issue(markdown: str) -> None:
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("GITHUB_TOKEN missing")
    body = json.dumps({"body": markdown}).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/issues/{ISSUE}/comments",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "v623-p0-sandbox-control/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        if response.status not in (200, 201):
            raise RuntimeError(f"Issue write failed HTTP {response.status}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-artifact-dir", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    args = ap.parse_args()

    source_zip, notebook_sha, source_sha = read_source(args.source_artifact_dir)
    template = args.template.read_text(encoding="utf-8")
    if "__SOURCE_ZIP_B64__" not in template:
        raise RuntimeError("P0 template placeholder missing")
    script = template.replace("__SOURCE_ZIP_B64__", base64.b64encode(source_zip).decode())
    if "__SOURCE_ZIP_B64__" in script:
        raise RuntimeError("P0 template replacement incomplete")
    script_sha = sha256(script.encode("utf-8"))

    account_id, owner, token, remaining, quota_rows = choose_account()
    target_ref = f"{owner}/{TARGET_SLUG}"

    existing = find_existing(owner, token)
    if existing is not None:
        status = get_status(owner, token)
        if status in ACTIVE:
            launch = {"action": "existing_active", "target_kernel_ref": target_ref, "status": status, "submitted": False}
        elif status == "COMPLETE":
            files = list_outputs(owner, token)
            try:
                receipt = fetch_output_json(files, "P0_RECEIPT.json")
                launch = {"action": "existing_complete", "target_kernel_ref": target_ref, "status": status, "submitted": False, "receipt": receipt}
            except Exception:
                raise RuntimeError("Existing COMPLETE P0 kernel lacks validated receipt; refusing duplicate launch")
        else:
            raise RuntimeError(f"Existing P0 kernel is terminal/non-active ({status}); refusing automatic rerun")
    else:
        result = save_kernel(owner, token, script)
        launch = {"action": "created_and_submitted", "target_kernel_ref": target_ref, "submitted": True, "result": result}

    launch_receipt = {
        "project": "PNEUMONIA V6.2.3-P0",
        "stage": "P0_LAUNCH",
        "github_run_id": int(os.environ.get("GITHUB_RUN_ID", "0") or 0),
        "git_commit_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "selected_account_id": account_id,
        "selected_owner": owner,
        "gpu_remaining_hours_before_launch": remaining,
        "quota_observation": quota_rows,
        "target_kernel_ref": target_ref,
        "script_sha256": script_sha,
        "w16_notebook_sha256": notebook_sha,
        "w16_source_zip_sha256": source_sha,
        "raw_dataset": RAW_DATASET,
        "locked_test_used": False,
        "external_data_used": False,
        "launch": launch,
    }
    print(json.dumps(launch_receipt, indent=2), flush=True)

    if "receipt" not in launch:
        terminal = None
        last = ""
        for attempt in range(1, 721):
            try:
                status = get_status(owner, token)
                print(f"P0 status {attempt}/720: {status}", flush=True)
                if status == "COMPLETE":
                    terminal = status
                    break
                if status in FAILURE:
                    raise RuntimeError(f"P0 Kaggle compute terminal failure: {status}")
            except Exception as exc:
                if "terminal failure" in str(exc):
                    raise
                last = f"{type(exc).__name__}: {exc}"
                print(f"P0 transport/status warning {attempt}/720: {last}", flush=True)
            time.sleep(30)
        if terminal != "COMPLETE":
            raise RuntimeError(f"P0 did not reach COMPLETE; last={last}")
        files = list_outputs(owner, token)
        receipt = fetch_output_json(files, "P0_RECEIPT.json")
    else:
        receipt = launch["receipt"]

    if receipt.get("project") != "PNEUMONIA V6.2.3-P0" or receipt.get("status") != "PASS":
        raise RuntimeError(f"P0 receipt identity/status mismatch: {receipt}")
    if receipt.get("locked_test_used") is not False or receipt.get("external_data_used") is not False or receipt.get("test_partition_enumerated") is not False:
        raise RuntimeError("P0 receipt indicates forbidden evidence access")
    counts = receipt.get("sample_counts") or {}
    if counts != {"train": 480, "cal": 120, "shadow": 120}:
        raise RuntimeError(f"P0 sample count mismatch: {counts}")

    metrics = receipt.get("aggregate_metrics") or {}
    body = (
        "## V6.2.3-P0 SINGLE-MODEL SANDBOX RECEIPT\n\n"
        f"- GitHub run: `{os.environ.get('GITHUB_RUN_ID','')}`\n"
        f"- Kaggle kernel: `{target_ref}`\n"
        f"- Kaggle account: `{account_id}` / `{owner}`\n"
        f"- Model: `M06 + ConvNeXt-Tiny @ 224`\n"
        "- Development-only split: `480 train / 120 calibration / 120 shadow`\n"
        "- Locked 624 used: `false`\n"
        "- test/ partition enumerated by P0 script: `false`\n"
        "- CheXpert/NIH used: `false / false`\n"
        f"- Verdict: `{receipt.get('verdict')}`\n"
        f"- Min shadow balanced accuracy across seeds: `{metrics.get('shadow_balanced_accuracy_min')}`\n"
        f"- Mean shadow balanced accuracy: `{metrics.get('shadow_balanced_accuracy_mean')}`\n"
        f"- Min shadow MCC: `{metrics.get('shadow_mcc_min')}`\n"
        f"- Min Normal recall: `{metrics.get('shadow_recall_normal_min')}`\n"
        f"- Min Pneumonia recall: `{metrics.get('shadow_recall_pneumonia_min')}`\n"
        f"- Threshold spread: `{metrics.get('threshold_spread')}`\n"
        f"- Worst stress BA: `{metrics.get('worst_stress_balanced_accuracy')}`\n"
        f"- Script SHA-256: `{script_sha}`\n"
        f"- Aggregate SHA-256: `{receipt.get('aggregate_sha256')}`\n"
        f"- Next action: `{receipt.get('next_action')}`\n"
        "- Status: `NON_OFFICIAL_DIRECTIONAL_ONLY`\n"
    )
    post_issue(body)
    print(json.dumps({"p0_complete": True, "kernel_ref": target_ref, "receipt": receipt}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
