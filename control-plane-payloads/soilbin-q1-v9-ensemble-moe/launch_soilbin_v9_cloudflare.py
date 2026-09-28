from __future__ import annotations
import base64,gzip,hashlib,json,os,pathlib,urllib.request,urllib.error
ROOT=pathlib.Path(__file__).resolve().parent
RUNNER=ROOT/"soilbin_v9_runner.py"; MODEL=ROOT/"all_features_51.csv.gz.b64"
ENDPOINT="https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
KERNEL_REF="mylovevpn1/soilbin-q1-q2-v9-ensemble-moe-20260928"
EXPECTED="dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"
text=MODEL.read_text(encoding="utf-8").strip()
raw=gzip.decompress(base64.b64decode(text,validate=True))
if hashlib.sha256(raw).hexdigest()!=EXPECTED: raise SystemExit("model fingerprint mismatch")
runner=RUNNER.read_text(encoding="utf-8"); compile(runner,str(RUNNER),"exec")
for x in ["CHANNEL_REGIME_BLEND2","GREEDY_BLEND","STACK_RIDGE10","PERSISTENCE","oracle_headroom","V5_EXACT_REPRODUCTION_FAILED","RESULTS_V9.json"]:
    if x not in runner: raise SystemExit("missing V9 guard: "+x)
source="MODEL_CSV_GZ_B64 = "+repr(text)+"\n"+runner
nb={"cells":[{"cell_type":"markdown","metadata":{},"source":["# SoilBin V9 Ensemble/MoE\n","Leakage-safe post-lock ensemble ablation; frozen V5 reference.\n"]},
             {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source.splitlines(keepends=True)}],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"},
                "soilbin":{"schema":"soilbin.q1.v9.ensemble-moe","cpu_only":True,"post_lock_exploratory":True,"does_not_supersede_v5":True,
                           "outer_test_used_for_weight_selection":False,"random_split_used":False}},
    "nbformat":4,"nbformat_minor":5}
nbtxt=json.dumps(nb,ensure_ascii=False,indent=1); sha=hashlib.sha256(nbtxt.encode()).hexdigest()
print("SOILBIN_V9_NOTEBOOK_VALIDATED",sha,flush=True)
token=os.environ.get("CGP_ACTION_OIDC_TOKEN","")
if len(token)<100: raise SystemExit("CGP_ACTION_OIDC_TOKEN unavailable")
payload={"request_id":"soilbin-q1-v9-ensemble-moe-kg09-20260928-r1","provider":"kaggle","operation_class":"compute","account_id":"kg-09",
 "purpose":"SoilBin V9 post-lock leakage-safe ensemble and mixture-of-experts ablation using V5/V7/V8 representations; weights/gates learned only from outer-training cross-fitted predictions.",
 "service":"kernels.KernelsApiService","method":"SaveKernel","body":{"slug":KERNEL_REF,"newTitle":"SoilBin Q1 Q2 V9 Ensemble MoE 20260928",
 "text":nbtxt,"language":"PYTHON","kernelType":"NOTEBOOK","kernelExecutionType":"SAVE_AND_RUN_ALL","isPrivate":True,"enableGpu":False,"enableInternet":True}}
req=urllib.request.Request(ENDPOINT,data=json.dumps(payload,separators=(",",":")).encode(),method="POST",
 headers={"Authorization":"Bearer "+token,"Content-Type":"application/json","Accept":"application/json","User-Agent":"soilbin-v9/1.0"})
try:
    with urllib.request.urlopen(req,timeout=240) as r: result=json.loads(r.read().decode("utf-8","replace") or "{}")
except urllib.error.HTTPError as e:
    raise SystemExit(f"Cloudflare action HTTP {e.code}: {e.read(5000).decode('utf-8','replace')}")
if not result.get("ok"): raise SystemExit("Cloudflare action returned ok=false: "+json.dumps(result)[:2500])
ref=result.get("provider_ref") or (result.get("result") or {}).get("ref") or KERNEL_REF
receipt={"accepted":True,"account_id":"kg-09","owner":"mylovevpn1","kernel_ref":ref,"notebook_sha256":sha,
 "runner_sha256":hashlib.sha256(runner.encode()).hexdigest(),"model_csv_sha256":EXPECTED,"cpu_only":True,"gpu":False,
 "post_lock_exploratory":True,"does_not_supersede_v5":True,"broker_run_id":result.get("broker_run_id"),"broker_sha":result.get("broker_sha")}
print("SOILBIN_V9_LAUNCH_ACCEPTED",json.dumps(receipt),flush=True)
with open(os.environ["GITHUB_ENV"],"a",encoding="utf-8") as h: h.write("SOILBIN_V9_KERNEL_REF="+ref+"\n")
