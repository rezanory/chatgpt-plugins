from __future__ import annotations
import base64, gzip, hashlib, json, os, pathlib, urllib.request, urllib.error

ROOT = pathlib.Path(__file__).resolve().parent
RUNNER = ROOT / "soilbin_v6_23_runner.py"
MODEL_B64 = ROOT / "all_features_51.csv.gz.b64"
CORROBORATION_B64 = ROOT / "external_literature_corroboration_v3.csv.gz.b64"
REGISTRY_B64 = ROOT / "external_validation_registry.json.b64"

ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
KERNEL_REF = "mylovevpn1/soilbin-q1-q2-v6-23-extension-20260928"
TITLE = "SoilBin Q1 Q2 V6 23 Item Extension 20260928"
EXPECTED = {
    "model": "dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059",
    "corroboration": "6d28c811f1ae1dfcf868ec4cea110608178986f6133aaf6220f70e41926a9f40",
    "registry": "d4c54e2fee3df55e0155c6ed9c5ca82296ff0653bd53cabe6fb2da3abdddf5d1",
}

def verify_gz_b64(path: pathlib.Path, expected: str) -> str:
    text = path.read_text(encoding="utf-8").strip()
    raw = gzip.decompress(base64.b64decode(text, validate=True))
    got = hashlib.sha256(raw).hexdigest()
    if got != expected:
        raise SystemExit(f"fingerprint mismatch {path.name}: {got}")
    return text

def verify_b64(path: pathlib.Path, expected: str) -> str:
    text = path.read_text(encoding="utf-8").strip()
    raw = base64.b64decode(text, validate=True)
    got = hashlib.sha256(raw).hexdigest()
    if got != expected:
        raise SystemExit(f"fingerprint mismatch {path.name}: {got}")
    return text

runner = RUNNER.read_text(encoding="utf-8")
compile(runner, str(RUNNER), "exec")
for required in [
    "EXPECTED_23_ITEMS_GOT", "post_lock_exploratory", "does_not_supersede_v5",
    "History-shuffle negative control", "Multi-horizon / early-history forecasting",
    "Graph spectral analysis", "Information-optimal experimental subset selection",
    "TSP for experimental execution order", "random_split_used"
]:
    if required not in runner:
        raise SystemExit("required V6 guard missing: " + required)

model_b64 = verify_gz_b64(MODEL_B64, EXPECTED["model"])
corroboration_b64 = verify_gz_b64(CORROBORATION_B64, EXPECTED["corroboration"])
registry_b64 = verify_b64(REGISTRY_B64, EXPECTED["registry"])

source = (
    "MODEL_CSV_GZ_B64 = " + repr(model_b64) + "\n" +
    "CORROBORATION_CSV_GZ_B64 = " + repr(corroboration_b64) + "\n" +
    "EXTERNAL_REGISTRY_B64 = " + repr(registry_b64) + "\n" +
    runner
)
notebook = {
    "cells": [
        {"cell_type":"markdown","metadata":{},"source":[
            "# SoilBin Q1/Q2 V6 — 23-item exploratory extension\n",
            "Post-lock exploratory compute. Does not supersede frozen V5 claims.\n",
            "CPU-only; grouped validation; no random split; external labels do not tune models.\n",
        ]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source.splitlines(keepends=True)}
    ],
    "metadata": {
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3"},
        "soilbin":{
            "schema":"soilbin.q1.v6.23-extension",
            "cpu_only":True,
            "post_lock_exploratory":True,
            "does_not_supersede_v5":True,
            "external_labels_used_for_tuning":False,
            "random_split_used":False,
            "response_status":"calibrated vertical force proxy N; stress calibration pending",
            "depth_mapping_status":"provisional"
        }
    },
    "nbformat":4,"nbformat_minor":5
}
text = json.dumps(notebook, ensure_ascii=False, indent=1)
sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
print("SOILBIN_V6_23_NOTEBOOK_VALIDATED", sha, flush=True)

token = os.environ.get("CGP_ACTION_OIDC_TOKEN", "")
if len(token) < 100:
    raise SystemExit("CGP_ACTION_OIDC_TOKEN unavailable")

payload = {
    "request_id":"soilbin-q1-v6-23-extension-kg09-20260928-r1",
    "provider":"kaggle",
    "operation_class":"compute",
    "account_id":"kg-09",
    "purpose":"SoilBin post-lock V6 exploratory execution of all 23 registered development items: calibration audit, cumulative history, trajectory/memory/pass tests, recursive/reconstruction/ablation/negative controls, graph/spectral/information design and setup-order analyses. V5 remains frozen.",
    "service":"kernels.KernelsApiService",
    "method":"SaveKernel",
    "body":{
        "slug":KERNEL_REF,
        "newTitle":TITLE,
        "text":text,
        "language":"PYTHON",
        "kernelType":"NOTEBOOK",
        "kernelExecutionType":"SAVE_AND_RUN_ALL",
        "isPrivate":True,
        "enableGpu":False,
        "enableInternet":True,
    },
}
req = urllib.request.Request(
    ENDPOINT,
    data=json.dumps(payload,separators=(",",":")).encode(),
    method="POST",
    headers={
        "Authorization":"Bearer "+token,
        "Content-Type":"application/json",
        "Accept":"application/json",
        "User-Agent":"soilbin-q1-v6-23/1.0",
    },
)
try:
    with urllib.request.urlopen(req, timeout=240) as response:
        result = json.loads(response.read().decode("utf-8","replace") or "{}")
except urllib.error.HTTPError as error:
    raise SystemExit(f"Cloudflare action HTTP {error.code}: {error.read(5000).decode('utf-8','replace')}")

if not result.get("ok"):
    raise SystemExit("Cloudflare action returned ok=false: "+json.dumps(result)[:2500])

provider_ref = result.get("provider_ref") or (result.get("result") or {}).get("ref") or KERNEL_REF
receipt = {
    "accepted":True,
    "account_id":"kg-09",
    "owner":"mylovevpn1",
    "kernel_ref":provider_ref,
    "notebook_sha256":sha,
    "runner_sha256":hashlib.sha256(runner.encode("utf-8")).hexdigest(),
    "model_csv_sha256":EXPECTED["model"],
    "corroboration_csv_sha256":EXPECTED["corroboration"],
    "external_registry_sha256":EXPECTED["registry"],
    "cpu_only":True,
    "gpu":False,
    "post_lock_exploratory":True,
    "does_not_supersede_v5":True,
    "external_labels_used_for_tuning":False,
    "broker_run_id":result.get("broker_run_id"),
    "broker_sha":result.get("broker_sha"),
}
print("SOILBIN_V6_23_LAUNCH_ACCEPTED", json.dumps(receipt, ensure_ascii=False), flush=True)
path = pathlib.Path(os.environ.get("RUNNER_TEMP", ".")) / "soilbin_v6_23_launch_receipt.json"
path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as handle:
    handle.write(f"SOILBIN_V6_23_RECEIPT_PATH={path}\nSOILBIN_V6_23_KERNEL_REF={provider_ref}\n")
