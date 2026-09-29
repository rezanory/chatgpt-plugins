from __future__ import annotations
import base64, gzip, hashlib, json, os, pathlib, urllib.request, urllib.error

ROOT=pathlib.Path(__file__).resolve().parent
RUNNER=ROOT/"soilbin_v8_runner.py"
MODEL_B64=ROOT/"all_features_51.csv.gz.b64"
ENDPOINT="https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
KERNEL_REF="mylovevpn1/soilbin-q1-q2-v8-state-error-20260928"
TITLE="SoilBin Q1 Q2 V8 State Error 20260928"
EXPECTED_MODEL_SHA="dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"

def verify(path):
    text=path.read_text(encoding="utf-8").strip()
    raw=gzip.decompress(base64.b64decode(text,validate=True))
    got=hashlib.sha256(raw).hexdigest()
    if got!=EXPECTED_MODEL_SHA:
        raise SystemExit(f"model fingerprint mismatch: {got}")
    return text

runner=RUNNER.read_text(encoding="utf-8")
compile(runner,str(RUNNER),"exec")
for token in [
    "STATE_EWMA50","STATE_MULTISCALE","STATE_PATH","STATE_COMPACT","STATE_FULL",
    "RESIDUAL_RIDGE_COMPACT","RESIDUAL_RIDGE_FULL","HARD_GROUP_DIAGNOSTICS",
    "V5_EXACT_REPRODUCTION_FAILED","RESULTS_V8.json","does_not_supersede_v5"
]:
    if token not in runner:
        raise SystemExit("required V8 guard missing: "+token)

model_b64=verify(MODEL_B64)
source="MODEL_CSV_GZ_B64 = "+repr(model_b64)+"\n"+runner
notebook={
    "cells":[
        {"cell_type":"markdown","metadata":{},"source":[
            "# SoilBin V8 — latent soil-state memory + hard-group diagnostics\n",
            "Post-lock exploratory challenge. Frozen V5 must reproduce exactly before V8 is interpreted.\n",
            "Nested leave-one-VxW-group-out validation; no random split; only observed past-state predictors.\n"
        ]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source.splitlines(keepends=True)}
    ],
    "metadata":{
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3"},
        "soilbin":{
            "schema":"soilbin.q1.v8.state-error",
            "cpu_only":True,
            "post_lock_exploratory":True,
            "does_not_supersede_v5":True,
            "random_split_used":False,
            "external_labels_used_for_selection":False
        }
    },
    "nbformat":4,"nbformat_minor":5
}
text=json.dumps(notebook,ensure_ascii=False,indent=1)
sha=hashlib.sha256(text.encode("utf-8")).hexdigest()
print("SOILBIN_V8_NOTEBOOK_VALIDATED",sha,flush=True)

token=os.environ.get("CGP_ACTION_OIDC_TOKEN","")
if len(token)<100:
    raise SystemExit("CGP_ACTION_OIDC_TOKEN unavailable")

payload={
    "request_id":"soilbin-q1-v8-state-error-kg09-20260928-r1",
    "provider":"kaggle",
    "operation_class":"compute",
    "account_id":"kg-09",
    "purpose":"SoilBin V8 post-lock exploratory challenge: exact V5 reproduction, latent past-state memory summaries, training-only residual correction, and hard-group error diagnostics under nested grouped validation.",
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
        "enableInternet":True
    }
}
req=urllib.request.Request(
    ENDPOINT,
    data=json.dumps(payload,separators=(",",":")).encode(),
    method="POST",
    headers={
        "Authorization":"Bearer "+token,
        "Content-Type":"application/json",
        "Accept":"application/json",
        "User-Agent":"soilbin-q1-v8/1.0"
    }
)
try:
    with urllib.request.urlopen(req,timeout=240) as r:
        result=json.loads(r.read().decode("utf-8","replace") or "{}")
except urllib.error.HTTPError as e:
    raise SystemExit(f"Cloudflare action HTTP {e.code}: {e.read(5000).decode('utf-8','replace')}")
if not result.get("ok"):
    raise SystemExit("Cloudflare action returned ok=false: "+json.dumps(result)[:2500])

provider_ref=result.get("provider_ref") or (result.get("result") or {}).get("ref") or KERNEL_REF
receipt={
    "accepted":True,
    "account_id":"kg-09",
    "owner":"mylovevpn1",
    "kernel_ref":provider_ref,
    "notebook_sha256":sha,
    "runner_sha256":hashlib.sha256(runner.encode("utf-8")).hexdigest(),
    "model_csv_sha256":EXPECTED_MODEL_SHA,
    "cpu_only":True,
    "gpu":False,
    "post_lock_exploratory":True,
    "does_not_supersede_v5":True,
    "broker_run_id":result.get("broker_run_id"),
    "broker_sha":result.get("broker_sha")
}
print("SOILBIN_V8_LAUNCH_ACCEPTED",json.dumps(receipt,ensure_ascii=False),flush=True)
path=pathlib.Path(os.environ.get("RUNNER_TEMP","."))/"soilbin_v8_launch_receipt.json"
path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
with open(os.environ["GITHUB_ENV"],"a",encoding="utf-8") as h:
    h.write(f"SOILBIN_V8_RECEIPT_PATH={path}\nSOILBIN_V8_KERNEL_REF={provider_ref}\n")
