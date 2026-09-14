from __future__ import annotations
import hashlib, json, os, pathlib, urllib.request, urllib.error

ROOT = pathlib.Path(__file__).resolve().parent
RUNNER = ROOT / "soilbin_q1_runner.py"
COND = ROOT / "conditions_51.csv"
TARG = ROOT / "targets_lc1_lc5_51.csv"
ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
KERNEL_REF = "mylovevpn1/soilbin-q1-q2-evidence-v1-20260914"

runner = RUNNER.read_text(encoding="utf-8")
cond = COND.read_text(encoding="utf-8-sig")
targ = TARG.read_text(encoding="utf-8-sig")
compile(runner, str(RUNNER), "exec")
assert cond.count("\n") >= 51 and targ.count("\n") >= 51
for required in ("'ExtraTrees'", "'MLP'", "GroupKFold", "B_history", "C_delta", "DEPTH_MAP"):
    assert required in runner, required
source = "CONDITIONS_CSV = " + repr(cond) + "\nTARGETS_CSV = " + repr(targ) + "\n" + runner
notebook = {
    "cells": [
        {"cell_type":"markdown","metadata":{},"source":["# SoilBin Q1/Q2 evidence benchmark v1\n","CPU-only; nested grouped validation; provisional depth mapping and force-proxy guardrails.\n"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"},"soilbin":{"schema":"soilbin.q1.v1","cpu_only":True,"external_validation":False,"depth_mapping_status":"provisional"}},
    "nbformat":4,"nbformat_minor":5,
}
text = json.dumps(notebook, ensure_ascii=False, indent=1)
sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
print("SOILBIN_NOTEBOOK_VALIDATED", sha, flush=True)
token = os.environ.get("CGP_ACTION_OIDC_TOKEN", "")
if len(token) < 100:
    raise SystemExit("CGP_ACTION_OIDC_TOKEN unavailable")
payload = {
    "request_id":"soilbin-q1-v1-kg09-20260914-r2","provider":"kaggle","operation_class":"compute","account_id":"kg-09",
    "purpose":"PhD/Q1-Q2 SoilBin CPU benchmark: 9 model families, nested grouped CV, history ablation, depth analysis, robustness, uncertainty and reproducible evidence package. No GPU.",
    "service":"kernels.KernelsApiService","method":"SaveKernel",
    "body":{"slug":KERNEL_REF,"newTitle":"SoilBin Q1 Q2 Evidence V1 20260914","text":text,"language":"PYTHON","kernelType":"NOTEBOOK","kernelExecutionType":"SAVE_AND_RUN_ALL","isPrivate":True,"enableGpu":False,"enableInternet":True},
}
req = urllib.request.Request(ENDPOINT, data=json.dumps(payload,separators=(",",":")).encode(), method="POST", headers={"Authorization":"Bearer "+token,"Content-Type":"application/json","Accept":"application/json","User-Agent":"soilbin-q1-v1/1.0"})
try:
    with urllib.request.urlopen(req, timeout=240) as response:
        result = json.loads(response.read().decode("utf-8","replace") or "{}")
except urllib.error.HTTPError as error:
    raise SystemExit(f"Cloudflare action HTTP {error.code}: {error.read(4000).decode('utf-8','replace')}")
if not result.get("ok"):
    raise SystemExit("Cloudflare action returned ok=false: "+json.dumps(result)[:1800])
provider_ref = result.get("provider_ref") or (result.get("result") or {}).get("ref") or KERNEL_REF
receipt = {"accepted":True,"account_id":"kg-09","owner":"mylovevpn1","kernel_ref":provider_ref,"notebook_sha256":sha,"cpu_only":True,"gpu":False,"broker_run_id":result.get("broker_run_id"),"broker_sha":result.get("broker_sha")}
print("SOILBIN_LAUNCH_ACCEPTED", json.dumps(receipt, ensure_ascii=False), flush=True)
path = pathlib.Path(os.environ.get("RUNNER_TEMP", ".")) / "soilbin_launch_receipt.json"
path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as handle:
    handle.write(f"SOILBIN_RECEIPT_PATH={path}\nSOILBIN_KERNEL_REF={provider_ref}\n")
