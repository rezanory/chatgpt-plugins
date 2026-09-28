from __future__ import annotations
import base64, gzip, hashlib, json, os, pathlib, urllib.error, urllib.request

ROOT=pathlib.Path(__file__).resolve().parent
RUNNER=ROOT/"soilbin_v16_runner.py"
MODEL_B64=ROOT/"all_features_51.csv.gz.b64"
ENDPOINT="https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
KERNEL_REF="mylovevpn1/soilbin-q1-q2-v16-functional-pass-dynamics-20260928"
TITLE="SoilBin V16 Functional Pass Dynamics 20260928"
EXPECTED_MODEL_SHA="dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"

def verify_model(path):
    text=path.read_text(encoding="utf-8").strip()
    raw=gzip.decompress(base64.b64decode(text,validate=True))
    got=hashlib.sha256(raw).hexdigest()
    if got!=EXPECTED_MODEL_SHA: raise SystemExit("model fingerprint mismatch: "+got)
    return text

runner=RUNNER.read_text(encoding="utf-8")
compile(runner,str(RUNNER),"exec")
required=[
 "functional-pass-dynamics","LINEAR","LOGARITHMIC","POWER_FREE","EXPONENTIAL_FREE",
 "STRETCHED_EXP_DECAY","PIECEWISE_LOG","DECAY_REBOUND","NESTED_SELECTED",
 "RESULTS_V16.json","RUN_MANIFEST_V16.json","collapse_confirmed",
]
for token in required:
    if token not in runner: raise SystemExit("required V16 guard missing: "+token)
model_b64=verify_model(MODEL_B64)
future="from __future__ import annotations"
if runner.count(future)!=1: raise SystemExit("runner future-import contract changed")
source=runner.replace(future,future+"\nMODEL_CSV_GZ_B64 = "+repr(model_b64),1)
compile(source,"soilbin_v16_notebook_source.py","exec")
notebook={
 "cells":[
  {"cell_type":"markdown","metadata":{},"source":[
   "# SoilBin V16 — Functional Pass-Dynamics Challenge\n",
   "Post-lock exploratory. No functional family is privileged a priori.\n",
   "Linear, logarithmic, power, exponential, hyperbolic, saturating, stretched-exponential, piecewise and decay/rebound candidates compete under group-held-out validation.\n",
   "T6 is evaluated only as a candidate regime transition; collapse/fatigue/shakedown class is not asserted.\n",
  ]},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source.splitlines(keepends=True)}
 ],
 "metadata":{
  "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
  "language_info":{"name":"python","version":"3"},
  "soilbin":{"schema":"soilbin.q1.v16.functional-pass-dynamics","cpu_only":True,
             "post_lock_exploratory":True,"does_not_supersede_v5":True,
             "random_split_used":False,"target_modified":False}
 },
 "nbformat":4,"nbformat_minor":5
}
text=json.dumps(notebook,ensure_ascii=False,indent=1)
sha=hashlib.sha256(text.encode()).hexdigest()
print("SOILBIN_V16_NOTEBOOK_VALIDATED",sha,flush=True)
token=os.environ.get("CGP_ACTION_OIDC_TOKEN","")
if len(token)<100: raise SystemExit("CGP_ACTION_OIDC_TOKEN unavailable")
payload={
 "request_id":"soilbin-q1-v16-functional-pass-dynamics-kg09-20260928-r1",
 "provider":"kaggle","operation_class":"compute","account_id":"kg-09",
 "purpose":"SoilBin V16 post-lock functional pass-dynamics challenge with group-held-out comparison of non-preferred functional families and candidate T6 regime-transition evidence.",
 "service":"kernels.KernelsApiService","method":"SaveKernel",
 "body":{"slug":KERNEL_REF,"newTitle":TITLE,"text":text,"language":"PYTHON","kernelType":"NOTEBOOK",
         "kernelExecutionType":"SAVE_AND_RUN_ALL","isPrivate":True,"enableGpu":False,"enableInternet":True}
}
req=urllib.request.Request(ENDPOINT,data=json.dumps(payload,separators=(",",":")).encode(),method="POST",
 headers={"Authorization":"Bearer "+token,"Content-Type":"application/json","Accept":"application/json","User-Agent":"soilbin-v16/1.0"})
try:
    with urllib.request.urlopen(req,timeout=240) as resp:
        result=json.loads(resp.read().decode("utf-8","replace") or "{}")
except urllib.error.HTTPError as exc:
    raise SystemExit(f"Cloudflare action HTTP {exc.code}: {exc.read(5000).decode('utf-8','replace')}") from exc
if not result.get("ok"): raise SystemExit("Cloudflare action returned ok=false: "+json.dumps(result)[:2500])
provider_ref=result.get("provider_ref") or (result.get("result") or {}).get("ref") or KERNEL_REF
receipt={"accepted":True,"account_id":"kg-09","owner":"mylovevpn1","kernel_ref":provider_ref,
 "notebook_sha256":sha,"runner_sha256":hashlib.sha256(runner.encode()).hexdigest(),
 "model_csv_sha256":EXPECTED_MODEL_SHA,"cpu_only":True,"gpu":False,
 "post_lock_exploratory":True,"does_not_supersede_v5":True,
 "broker_run_id":result.get("broker_run_id"),"broker_sha":result.get("broker_sha")}
print("SOILBIN_V16_LAUNCH_ACCEPTED",json.dumps(receipt,ensure_ascii=False),flush=True)
path=pathlib.Path(os.environ.get("RUNNER_TEMP","."))/"soilbin_v16_launch_receipt.json"
path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
with open(os.environ["GITHUB_ENV"],"a",encoding="utf-8") as h:
    h.write(f"SOILBIN_V16_RECEIPT_PATH={path}\nSOILBIN_V16_KERNEL_REF={provider_ref}\n")
