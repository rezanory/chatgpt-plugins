import json
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

ACTION_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
READ_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/read/kaggle"
ACCOUNT_ID = "kg-05"
KERNEL_REF = "trickermark/m07-vindr-pcxr-preflight-20260927"
OWNER, SLUG = KERNEL_REF.split("/", 1)
ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER_TEMP = pathlib.Path(os.environ["RUNNER_TEMP"])
LAUNCH_PATH = RUNNER_TEMP / "m07-vindr-preflight-launch.json"
STATUS_PATH = RUNNER_TEMP / "m07-vindr-preflight-status.json"
RECEIPT_PATH = RUNNER_TEMP / "M07_VINDR_PCXR_PREFLIGHT.json"
SUMMARY_PATH = RUNNER_TEMP / "m07-vindr-preflight-summary.json"


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def post_json(url, token, payload, timeout=240):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "m07-vindr-external/1.2",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            return json.loads(raw or "{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read(12000).decode("utf-8", "replace")
        return {
            "ok": False,
            "http_status": exc.code,
            "error": raw,
        }


def fail_summary(status, next_action, detail=None, exit_code=2):
    payload = {
        "status": status,
        "next_action": next_action,
        "kernel_ref": KERNEL_REF,
        "gpu_used": False,
        "training_started": False,
        "inference_started": False,
    }
    if detail:
        payload["detail"] = str(detail)[:4000]
    write_json(SUMMARY_PATH, payload)
    print("M07_VINDR_PREFLIGHT_SUMMARY", json.dumps(payload, sort_keys=True), flush=True)
    raise SystemExit(exit_code)


action_token = os.environ["CGP_ACTION_OIDC_TOKEN"].strip()
read_token = os.environ["CGP_READ_OIDC_TOKEN"].strip()
if len(action_token) < 100 or len(read_token) < 100:
    fail_summary("OIDC_TOKEN_INVALID", "REPAIR_CONTROL_PLANE_AUTH", exit_code=10)

kernel_code = (ROOT / "scripts" / "m07_vindr_pcxr_preflight.py").read_text(encoding="utf-8")
payload = {
    "request_id": "m07-vindr-pcxr-preflight-kg05-20260927-v3",
    "provider": "kaggle",
    "operation_class": "compute",
    "account_id": ACCOUNT_ID,
    "purpose": (
        "CPU-only access/schema/patient-identity preflight for approved M07 external "
        "validation on VinDr-PCXR. No GPU, training, inference, or threshold tuning."
    ),
    "service": "kernels.KernelsApiService",
    "method": "SaveKernel",
    "body": {
        "slug": KERNEL_REF,
        "newTitle": "M07 VinDr-PCXR External Preflight 20260927",
        "text": kernel_code,
        "language": "python",
        "kernelType": "script",
        "kernelExecutionType": "SAVE_AND_RUN_ALL",
        "isPrivate": True,
        "enableGpu": False,
        "enableTpu": False,
        "enableInternet": True,
        "kernelDataSources": [],
        "datasetDataSources": [],
        "competitionDataSources": ["pediatric-cxr-analysis-challenge"],
        "modelDataSources": [],
    },
}
launch = post_json(ACTION_ENDPOINT, action_token, payload)
write_json(LAUNCH_PATH, launch)
print("M07_VINDR_LAUNCH", json.dumps(launch, sort_keys=True), flush=True)

if not launch.get("ok"):
    fail_summary(
        "SUBMISSION_REJECTED",
        "RESOLVE_KAGGLE_COMPETITION_ACCESS_OR_PAYLOAD",
        detail=launch.get("error") or launch,
        exit_code=20,
    )

history = []
terminal = None
for attempt in range(1, 61):
    status_payload = {
        "action": "kernel_status",
        "account_id": ACCOUNT_ID,
        "kernel_ref": KERNEL_REF,
    }
    result = post_json(READ_ENDPOINT, read_token, status_payload, timeout=120)
    history.append(result)
    node = result.get("result") or {}
    status = str(node.get("status") or (node.get("result") or {}).get("status") or "").upper()
    print("VINDR_PREFLIGHT_STATUS", attempt, status, flush=True)
    if status in {"COMPLETE", "ERROR", "CANCELLED"}:
        terminal = status
        break
    time.sleep(20)

write_json(STATUS_PATH, history)
if terminal != "COMPLETE":
    fail_summary(
        "KAGGLE_TERMINAL_" + str(terminal),
        "INSPECT_KAGGLE_PREFLIGHT_TERMINAL",
        exit_code=30,
    )

items = []
for page in range(1, 20):
    output_payload = {
        "action": "raw_read",
        "account_id": ACCOUNT_ID,
        "service": "kernels.KernelsApiService",
        "method": "ListKernelSessionOutput",
        "body": {
            "userName": OWNER,
            "kernelSlug": SLUG,
            "page": page,
            "pageSize": 100,
        },
    }
    response = post_json(READ_ENDPOINT, read_token, output_payload, timeout=120)
    if not response.get("ok"):
        fail_summary(
            "OUTPUT_LIST_FAILED",
            "INSPECT_KAGGLE_OUTPUT_API",
            detail=response.get("error") or response,
            exit_code=40,
        )
    files = (response.get("result") or {}).get("files") or []
    items.extend(files)
    if len(files) < 100:
        break

target = next(
    (
        item
        for item in items
        if str(item.get("fileName", "")).endswith("M07_VINDR_PCXR_PREFLIGHT.json")
    ),
    None,
)
if not target or not target.get("url"):
    fail_summary(
        "RECEIPT_NOT_FOUND",
        "INSPECT_KAGGLE_PREFLIGHT_OUTPUT",
        exit_code=41,
    )

url = str(target["url"])
parsed = urllib.parse.urlparse(url)
if parsed.scheme != "https" or parsed.hostname != "www.kaggleusercontent.com":
    fail_summary(
        "UNTRUSTED_OUTPUT_URL",
        "INSPECT_KAGGLE_OUTPUT_URL",
        exit_code=42,
    )

with urllib.request.urlopen(
    urllib.request.Request(url, headers={"User-Agent": "m07-vindr-external/1.2"}),
    timeout=120,
) as response:
    data = response.read()
RECEIPT_PATH.write_bytes(data)
receipt = json.loads(data.decode("utf-8"))

summary = {
    "status": receipt.get("status"),
    "next_action": receipt.get("next_action"),
    "kernel_ref": KERNEL_REF,
    "patient_identity_available": receipt.get("patient_identity_available"),
    "label_tables_found": receipt.get("label_tables_found"),
    "file_count": receipt.get("file_count"),
    "csv_count": receipt.get("csv_count"),
    "dicom_count": receipt.get("dicom_count"),
    "gpu_used": False,
    "training_started": False,
    "inference_started": False,
}
write_json(SUMMARY_PATH, summary)
print("M07_VINDR_PREFLIGHT_RECEIPT", json.dumps(receipt, sort_keys=True), flush=True)
print("M07_VINDR_PREFLIGHT_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)

if receipt.get("status") != "PASS_DATASET_SCHEMA":
    raise SystemExit(50)
if not receipt.get("patient_identity_available"):
    raise SystemExit(51)
