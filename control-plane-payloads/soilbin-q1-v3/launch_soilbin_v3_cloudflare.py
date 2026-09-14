from __future__ import annotations
import base64, gzip, hashlib, json, os, pathlib, urllib.request, urllib.error

ROOT = pathlib.Path(__file__).resolve().parent
RUNNER = ROOT / "soilbin_v3_runner.py"
MODEL_B64 = ROOT / "all_features_51.csv.gz.b64"
CANADA_B64 = ROOT / "external_canada_ayetan_2026.csv.gz.b64"
ERDC_B64 = ROOT / "external_erdc_crrel_2009.csv.gz.b64"
CORROBORATION_B64 = ROOT / "external_literature_corroboration_v3.csv.gz.b64"
REGISTRY_B64 = ROOT / "external_validation_registry.json.b64"
ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
KERNEL_REF = "mylovevpn1/soilbin-q1-q2-v3-external-model-selection-20260914"
TITLE = "SoilBin Q1 Q2 V3 External Model Selection 20260914"
EXPECTED = {
    "model": {"gz":"29913b884d01a6ca68880450c222b9c35114d7e270aca13a713df1087a20b623","raw":"dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"},
    "canada": {"gz":"02477cfe6409f669806dc2ea59da88e5e8a8a6bd7fe1cf8f7efd4d93a3b8aad5","raw":"20ed0fd6b51c0fca7bb5e78e50cd7901e13d1ded3b03b3e13bff5df829592407"},
    "erdc": {"gz":"c20eaf2e81c0d07f91fed87a6d0ee7d24d03e41cb79611fae7ef5fbc81956ae9","raw":"9a445ba5080fa637eb89b4992e4c3c0b7eaa789ada0d1dfca2f2516f7fc316a6"},
    "corroboration": {"gz":"eab59c39c1bd48d52d13bd0b312c50ecbecfb2b000fd4e228e1c0c32e85e1f0e","raw":"6d28c811f1ae1dfcf868ec4cea110608178986f6133aaf6220f70e41926a9f40"},
    "registry_raw":"d4c54e2fee3df55e0155c6ed9c5ca82296ff0653bd53cabe6fb2da3abdddf5d1",
}

def load_gz_payload(path: pathlib.Path, label: str) -> str:
    b64 = path.read_text(encoding="utf-8").strip()
    gz = base64.b64decode(b64, validate=True)
    if hashlib.sha256(gz).hexdigest() != EXPECTED[label]["gz"]:
        raise SystemExit(f"{label} gzip SHA mismatch")
    raw = gzip.decompress(gz)
    if hashlib.sha256(raw).hexdigest() != EXPECTED[label]["raw"]:
        raise SystemExit(f"{label} raw SHA mismatch")
    return b64

runner = RUNNER.read_text(encoding="utf-8")
compile(runner, str(RUNNER), "exec")
model_b64 = load_gz_payload(MODEL_B64, "model")
canada_b64 = load_gz_payload(CANADA_B64, "canada")
erdc_b64 = load_gz_payload(ERDC_B64, "erdc")
corroboration_b64 = load_gz_payload(CORROBORATION_B64, "corroboration")
registry_b64 = REGISTRY_B64.read_text(encoding="utf-8").strip()
registry_raw = base64.b64decode(registry_b64, validate=True)
if hashlib.sha256(registry_raw).hexdigest() != EXPECTED["registry_raw"]:
    raise SystemExit("external registry raw SHA mismatch")
for required in (
    "D_delta_prev_peak", "E_delta_prev_peak_waveform", "exact_group_signflip_one_sided_p",
    "external_labels_used_for_tuning", "absolute_external_validation", "GroupKFold",
    "No current-run waveform", "external_literature_corroboration_v3.csv",
):
    if required not in runner:
        raise SystemExit(f"required V3 guard missing: {required}")
source = (
    "MODEL_CSV_GZ_B64 = " + repr(model_b64) + "\n" +
    "CANADA_CSV_GZ_B64 = " + repr(canada_b64) + "\n" +
    "ERDC_CSV_GZ_B64 = " + repr(erdc_b64) + "\n" +
    "CORROBORATION_CSV_GZ_B64 = " + repr(corroboration_b64) + "\n" +
    "EXTERNAL_REGISTRY_B64 = " + repr(registry_b64) + "\n" +
    runner
)
notebook = {
    "cells": [
        {"cell_type":"markdown","metadata":{},"source":[
            "# SoilBin Q1/Q2 V3 — External Validation + Complete Delta/Waveform Ablation\n",
            "CPU-only. Internal selection uses nested grouped CV only; external labels are test/corroboration evidence only and are never used for tuning.\n",
            "Primary requested ablation: Delta + previous peak versus Delta + previous peak + previous-pass waveform morphology.\n",
        ]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source.splitlines(keepends=True)},
    ],
    "metadata": {
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3"},
        "soilbin":{
            "schema":"soilbin.q1.v3","cpu_only":True,"gpu":False,
            "external_labels_used_for_tuning":False,"absolute_external_validation":False,
            "response":"calibrated vertical force proxy N","depth_mapping_status":"provisional",
        },
    },
    "nbformat":4,"nbformat_minor":5,
}
text = json.dumps(notebook, ensure_ascii=False, indent=1)
sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
print("SOILBIN_V3_NOTEBOOK_VALIDATED", sha, flush=True)
token = os.environ.get("CGP_ACTION_OIDC_TOKEN", "")
if len(token) < 100:
    raise SystemExit("CGP_ACTION_OIDC_TOKEN unavailable")
payload = {
    "request_id":"soilbin-q1-v3-external-model-selection-kg09-20260914-r1",
    "provider":"kaggle","operation_class":"compute","account_id":"kg-09",
    "purpose":"PhD/Q1-Q2 SoilBin V3 CPU benchmark: complete delta+previous-peak vs delta+previous-peak+waveform ablation, expanded model catalog, nested grouped CV, uncertainty, robustness, independent external numeric trend/physics validation, and structured literature corroboration. External labels never tune models; no absolute N-to-kPa claim.",
    "service":"kernels.KernelsApiService","method":"SaveKernel",
    "body":{
        "slug":KERNEL_REF,"newTitle":TITLE,"text":text,"language":"PYTHON","kernelType":"NOTEBOOK",
        "kernelExecutionType":"SAVE_AND_RUN_ALL","isPrivate":True,"enableGpu":False,"enableInternet":True,
    },
}
req = urllib.request.Request(
    ENDPOINT, data=json.dumps(payload,separators=(",",":")).encode(), method="POST",
    headers={"Authorization":"Bearer "+token,"Content-Type":"application/json","Accept":"application/json","User-Agent":"soilbin-q1-v3/1.0"},
)
try:
    with urllib.request.urlopen(req, timeout=240) as response:
        result = json.loads(response.read().decode("utf-8","replace") or "{}")
except urllib.error.HTTPError as error:
    raise SystemExit(f"Cloudflare action HTTP {error.code}: {error.read(4000).decode('utf-8','replace')}")
if not result.get("ok"):
    raise SystemExit("Cloudflare action returned ok=false: "+json.dumps(result)[:1800])
provider_ref = result.get("provider_ref") or (result.get("result") or {}).get("ref") or KERNEL_REF
receipt = {
    "accepted":True,"account_id":"kg-09","owner":"mylovevpn1","kernel_ref":provider_ref,
    "notebook_sha256":sha,"runner_sha256":hashlib.sha256(runner.encode("utf-8")).hexdigest(),
    "model_csv_sha256":EXPECTED["model"]["raw"],"canada_csv_sha256":EXPECTED["canada"]["raw"],
    "erdc_csv_sha256":EXPECTED["erdc"]["raw"],"corroboration_csv_sha256":EXPECTED["corroboration"]["raw"],
    "external_registry_sha256":EXPECTED["registry_raw"],"cpu_only":True,"gpu":False,
    "external_labels_used_for_tuning":False,"absolute_external_validation":False,
    "broker_run_id":result.get("broker_run_id"),"broker_sha":result.get("broker_sha"),
}
print("SOILBIN_V3_LAUNCH_ACCEPTED", json.dumps(receipt, ensure_ascii=False), flush=True)
path = pathlib.Path(os.environ.get("RUNNER_TEMP", ".")) / "soilbin_v3_launch_receipt.json"
path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as handle:
    handle.write(f"SOILBIN_V3_RECEIPT_PATH={path}\nSOILBIN_V3_KERNEL_REF={provider_ref}\n")
