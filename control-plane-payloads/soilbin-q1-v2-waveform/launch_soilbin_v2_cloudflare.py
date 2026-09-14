from __future__ import annotations
import base64, gzip, hashlib, json, os, pathlib, urllib.request, urllib.error

ROOT = pathlib.Path(__file__).resolve().parent
RUNNER = ROOT / "soilbin_waveform_v2_runner.py"
DATA_B64 = ROOT / "all_features_51.csv.gz.b64"
ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
KERNEL_REF = "mylovevpn1/soilbin-q1-q2-waveform-v2-20260914"
EXPECTED_GZ_SHA = "29913b884d01a6ca68880450c222b9c35114d7e270aca13a713df1087a20b623"
EXPECTED_CSV_SHA = "dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"

runner = RUNNER.read_text(encoding="utf-8")
compile(runner, str(RUNNER), "exec")
b64 = DATA_B64.read_text(encoding="utf-8").strip()
gz = base64.b64decode(b64, validate=True)
if hashlib.sha256(gz).hexdigest() != EXPECTED_GZ_SHA:
    raise SystemExit("waveform-derived payload gzip SHA mismatch")
csv_bytes = gzip.decompress(gz)
if hashlib.sha256(csv_bytes).hexdigest() != EXPECTED_CSV_SHA:
    raise SystemExit("waveform-derived payload CSV SHA mismatch")
for required in ("C_prev_waveform_coupled", "D_delta_waveform_coupled", "GroupKFold", "Current-run waveform descriptors are forbidden", "uncertainty_calibration_v2.json"):
    if required not in runner:
        raise SystemExit(f"required V2 guard missing: {required}")
source = "MODEL_CSV_GZ_B64 = " + repr(b64) + "\n" + runner
notebook = {
    "cells": [
        {"cell_type":"markdown","metadata":{},"source":["# SoilBin Q1/Q2 waveform-derived history benchmark V2\n","CPU-only; prior-pass LC1/LC5 waveform morphology; nested grouped validation; no current-run waveform leakage.\n"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"},"soilbin":{"schema":"soilbin.q1.waveform.v2","cpu_only":True,"external_validation":False,"current_waveform_predictor":False,"depth_mapping_status":"provisional"}},
    "nbformat":4,"nbformat_minor":5,
}
text = json.dumps(notebook, ensure_ascii=False, indent=1)
sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
print("SOILBIN_V2_NOTEBOOK_VALIDATED", sha, flush=True)
token = os.environ.get("CGP_ACTION_OIDC_TOKEN", "")
if len(token) < 100:
    raise SystemExit("CGP_ACTION_OIDC_TOKEN unavailable")
payload = {
    "request_id":"soilbin-q1-v2-waveform-kg09-20260914-r1","provider":"kaggle","operation_class":"compute","account_id":"kg-09",
    "purpose":"PhD/Q1-Q2 SoilBin V2 CPU benchmark: leakage-safe previous-pass LC1/LC5 waveform morphology, coupled two-sensor prediction, delta ablation, nested grouped CV, robustness, QC sensitivity and group-calibrated uncertainty. No GPU; no external-validation claim.",
    "service":"kernels.KernelsApiService","method":"SaveKernel",
    "body":{"slug":KERNEL_REF,"newTitle":"SoilBin Q1 Q2 Waveform V2 20260914","text":text,"language":"PYTHON","kernelType":"NOTEBOOK","kernelExecutionType":"SAVE_AND_RUN_ALL","isPrivate":True,"enableGpu":False,"enableInternet":True},
}
req = urllib.request.Request(ENDPOINT, data=json.dumps(payload,separators=(",",":")).encode(), method="POST", headers={"Authorization":"Bearer "+token,"Content-Type":"application/json","Accept":"application/json","User-Agent":"soilbin-q1-v2/1.0"})
try:
    with urllib.request.urlopen(req, timeout=240) as response:
        result = json.loads(response.read().decode("utf-8","replace") or "{}")
except urllib.error.HTTPError as error:
    raise SystemExit(f"Cloudflare action HTTP {error.code}: {error.read(4000).decode('utf-8','replace')}")
if not result.get("ok"):
    raise SystemExit("Cloudflare action returned ok=false: "+json.dumps(result)[:1800])
provider_ref = result.get("provider_ref") or (result.get("result") or {}).get("ref") or KERNEL_REF
receipt = {"accepted":True,"account_id":"kg-09","owner":"mylovevpn1","kernel_ref":provider_ref,"notebook_sha256":sha,"data_csv_sha256":EXPECTED_CSV_SHA,"cpu_only":True,"gpu":False,"broker_run_id":result.get("broker_run_id"),"broker_sha":result.get("broker_sha")}
print("SOILBIN_V2_LAUNCH_ACCEPTED", json.dumps(receipt, ensure_ascii=False), flush=True)
path = pathlib.Path(os.environ.get("RUNNER_TEMP", ".")) / "soilbin_v2_launch_receipt.json"
path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as handle:
    handle.write(f"SOILBIN_V2_RECEIPT_PATH={path}\nSOILBIN_V2_KERNEL_REF={provider_ref}\n")
