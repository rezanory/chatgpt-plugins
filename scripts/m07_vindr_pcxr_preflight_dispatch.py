from __future__ import annotations
import json, os, pathlib, time, urllib.error, urllib.parse, urllib.request

ACTION_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
READ_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/read/kaggle"
ACCOUNT = "kg-05"
KERNEL_REF = "trickermark/m07-vindr-pcxr-preflight-20260927"
OWNER, SLUG = KERNEL_REF.split("/", 1)
ROOT = pathlib.Path(os.environ["RUNNER_TEMP"])
SCRIPT = pathlib.Path("scripts/m07_vindr_pcxr_preflight.py").read_text(encoding="utf-8")

def post(endpoint: str, token: str, payload: dict, timeout: int = 240) -> dict:
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "m07-vindr-external/2.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            return json.loads(raw or "{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read(20000).decode("utf-8", "replace")
        return {"ok": False, "http_status": exc.code, "error": raw}

action_token = os.environ["CGP_ACTION_OIDC_TOKEN"]
read_token = os.environ["CGP_READ_OIDC_TOKEN"]

payload = {
    "request_id": "m07-vindr-pcxr-preflight-kg05-20260927-v2",
    "provider": "kaggle",
    "operation_class": "compute",
    "account_id": ACCOUNT,
    "purpose": "CPU-only VinDr-PCXR access and schema preflight for approved frozen-M07 external validation. No GPU, training, inference, or threshold tuning.",
    "service": "kernels.KernelsApiService",
    "method": "SaveKernel",
    "body": {
        "slug": KERNEL_REF,
        "newTitle": "M07 VinDr-PCXR External Preflight 20260927 V2",
        "text": SCRIPT,
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
launch = post(ACTION_ENDPOINT, action_token, payload)
(ROOT / "m07-vindr-preflight-launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
print("VINDR_PREFLIGHT_LAUNCH", json.dumps(launch, sort_keys=True), flush=True)
if not launch.get("ok"):
    raise SystemExit("VINDR_PREFLIGHT_SUBMISSION_REJECTED")

def extract_status(value):
    if isinstance(value, dict):
        for key in ("status", "kernelStatus", "state"):
            v = value.get(key)
            if isinstance(v, str) and v:
                return v.upper()
        for v in value.values():
            s = extract_status(v)
            if s:
                return s
    elif isinstance(value, list):
        for v in value:
            s = extract_status(v)
            if s:
                return s
    return ""

history = []
terminal = ""
for attempt in range(1, 81):
    row = post(
        READ_ENDPOINT,
        read_token,
        {"action": "kernel_status", "account_id": ACCOUNT, "kernel_ref": KERNEL_REF},
        timeout=120,
    )
    history.append(row)
    status = extract_status(row)
    print("VINDR_PREFLIGHT_STATUS", attempt, status or "UNKNOWN", flush=True)
    if status in {"COMPLETE", "ERROR", "CANCELLED"}:
        terminal = status
        break
    time.sleep(15)

(ROOT / "m07-vindr-preflight-status.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
if terminal != "COMPLETE":
    raise SystemExit("VINDR_PREFLIGHT_TERMINAL_" + (terminal or "TIMEOUT"))

items = []
for page in range(1, 21):
    row = post(
        READ_ENDPOINT,
        read_token,
        {
            "action": "raw_read",
            "account_id": ACCOUNT,
            "service": "kernels.KernelsApiService",
            "method": "ListKernelSessionOutput",
            "body": {"userName": OWNER, "kernelSlug": SLUG, "page": page, "pageSize": 100},
        },
        timeout=120,
    )
    result = row.get("result") or {}
    files = result.get("files") or []
    items.extend(files)
    if len(files) < 100:
        break

target = next(
    (x for x in items if str(x.get("fileName", "")).endswith("M07_VINDR_PCXR_PREFLIGHT.json")),
    None,
)
if not target or not target.get("url"):
    raise SystemExit("VINDR_PREFLIGHT_RECEIPT_NOT_FOUND")
url = str(target["url"])
parsed = urllib.parse.urlparse(url)
if parsed.scheme != "https" or parsed.hostname != "www.kaggleusercontent.com":
    raise SystemExit("VINDR_PREFLIGHT_UNTRUSTED_OUTPUT_URL")

with urllib.request.urlopen(
    urllib.request.Request(url, headers={"User-Agent": "m07-vindr-external/2.0"}),
    timeout=120,
) as response:
    data = response.read()
out = ROOT / "M07_VINDR_PCXR_PREFLIGHT.json"
out.write_bytes(data)
receipt = json.loads(data.decode("utf-8"))
print("M07_VINDR_PREFLIGHT_RECEIPT", json.dumps(receipt, sort_keys=True), flush=True)
if receipt.get("status") != "PASS_DATASET_SCHEMA":
    raise SystemExit("VINDR_PREFLIGHT_DATASET_SCHEMA_BLOCKED")
