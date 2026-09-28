"""SoilBin V26 single-lane safe Kaggle controller."""
from __future__ import annotations
import base64,gzip,hashlib,json,os,pathlib,threading,time,urllib.error,urllib.parse,urllib.request

ROOT=pathlib.Path(__file__).resolve().parent
READ="https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/read/kaggle"
WRITE="https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
POOL={"kg-02":"radlinaradlina","kg-03":"rezanory","kg-04":"reyhanehazad","kg-06":"msdenis",
      "kg-07":"nisabulutmark","kg-08":"azadkk","kg-09":"mylovevpn1","kg-11":"jobreza1"}
TERMINAL={"COMPLETE","COMPLETED","ERROR","FAILED","CANCELLED","CANCELED","CANCEL_ACKNOWLEDGED"}
EXPECTED="dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"
SLUG="soilbin-v26-lc5-transition-aware-20260928-r1"
CAMPAIGN="soilbin-v26-lc5-transition-aware-20260928-r1"
TOK={};LOCK=threading.Lock()
EVID=pathlib.Path(os.environ.get("RUNNER_TEMP","."))/"soilbin-v26-lc5-transition-aware"
EVID.mkdir(parents=True,exist_ok=True)

def dump(name,obj):
    (EVID/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

def token(kind):
    with LOCK:
        x=TOK.get(kind)
        if x and time.time()-x[0]<180:return x[1]
        aud="cgp-control-plane-v3-action" if kind=="write" else "cgp-control-plane-v3"
        u=os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
        u+=("&" if "?" in u else "?")+"audience="+urllib.parse.quote(aud,safe="")
        q=urllib.request.Request(u,headers={"Authorization":"Bearer "+os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]})
        with urllib.request.urlopen(q,timeout=30) as r:v=json.load(r)["value"]
        if len(v)<100:raise RuntimeError("OIDC_UNAVAILABLE")
        print("::add-mask::"+v,flush=True);TOK[kind]=(time.time(),v);return v

def call(body,kind="read",timeout=120):
    endpoint=WRITE if kind=="write" else READ
    for a in range(3):
        q=urllib.request.Request(endpoint,data=json.dumps(body,separators=(",",":")).encode(),method="POST",
            headers={"Authorization":"Bearer "+token(kind),"Content-Type":"application/json","Accept":"application/json",
                     "User-Agent":"soilbin-v26/1.0"})
        try:
            with urllib.request.urlopen(q,timeout=timeout) as r:val=json.load(r)
            if val.get("ok") is not True:raise RuntimeError("BROKER_REJECTED:"+json.dumps(val)[:1200])
            if kind=="read" and val.get("read_only") is not True:raise RuntimeError("READ_ONLY_GUARD")
            return val
        except urllib.error.HTTPError as e:
            detail=e.read(3000).decode("utf-8","replace")
            if e.code==403 and "JWT expired" in detail and a<2:
                with LOCK:TOK.pop(kind,None)
                continue
            if e.code in (429,502,503,504) and a<2:
                time.sleep(3*(a+1));continue
            raise RuntimeError(f"BROKER_HTTP_{e.code}:{detail}") from e
    raise RuntimeError("BROKER_RETRIES_EXHAUSTED")

def raw(account,method,body):
    return call({"action":"raw_read","account_id":account,"service":"kernels.KernelsApiService",
                 "method":method,"body":body})["result"]

def norm(ref):
    ref=str(ref or "").strip()
    if "kaggle.com/" in ref:ref=ref.split("kaggle.com/",1)[1]
    return ref.removeprefix("/code/").removeprefix("code/").lstrip("/")

def status(account,ref):
    owner,slug=ref.split("/",1)
    x=raw(account,"GetKernelSessionStatus",{"userName":owner,"kernelSlug":slug})
    return str(x.get("status") or "PENDING").upper()

def inventory(account):
    owner=POOL[account];res=raw(account,"ListKernels",{"group":"PROFILE","sortBy":"DATE_RUN","pageSize":100})
    ks=res.get("kernels",[]);refs=[]
    for x in ks:
        r=norm(x.get("ref") or x.get("url") or "")
        if r.startswith(owner+"/") and r not in refs:refs.append(r)
    sts=[]
    for r in refs:
        try:s=status(account,r)
        except Exception:s="UNKNOWN"
        sts.append({"ref":r,"status":s})
    return {"account":account,"owner":owner,"listing_at_limit":len(ks)>=100,
            "active_or_unknown":sum(x["status"] not in TERMINAL for x in sts),
            "kernels_scanned":len(refs),"statuses":sts}

def publish(title,body):
    gh=os.environ.get("GITHUB_TOKEN")
    if not gh:return
    u=f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/issues/134/comments"
    q=urllib.request.Request(u,data=json.dumps({"body":"## "+title+"\n\n"+body}).encode(),method="POST",
        headers={"Authorization":"Bearer "+gh,"Accept":"application/vnd.github+json",
                 "Content-Type":"application/json","User-Agent":"soilbin-v26"})
    with urllib.request.urlopen(q,timeout=30):pass

RUNNER=(ROOT/"runner.py").read_text(encoding="utf-8")
MODEL=(ROOT/"all_features_51.csv.gz.b64").read_text(encoding="utf-8").strip()
raw_model=gzip.decompress(base64.b64decode(MODEL,validate=True))
if hashlib.sha256(raw_model).hexdigest()!=EXPECTED:raise RuntimeError("MODEL_FINGERPRINT")

def notebook():
    payload={"lane":"V26","campaign":CAMPAIGN,"model_b64":MODEL}
    pp=base64.b64encode(gzip.compress(json.dumps(payload,separators=(",",":")).encode(),mtime=0)).decode()
    rr=base64.b64encode(RUNNER.encode()).decode()
    src=["import base64,gzip,json\n",
         f"PAYLOAD=json.loads(gzip.decompress(base64.b64decode({pp!r})))\n",
         f"exec(compile(base64.b64decode({rr!r}),'soilbin_v26_runner.py','exec'))\n"]
    return json.dumps({"cells":[
      {"cell_type":"markdown","metadata":{},"source":["# SoilBin V26 — LC5 Transition-Aware Residual Challenge\n",
        "Private CPU-only post-lock exploration; full Speed×Load group holdout; LC1 frozen to V23.\n"]},
      {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src}],
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"},"soilbin":{"lane":"V26","campaign":CAMPAIGN,
        "runner_sha256":hashlib.sha256(RUNNER.encode()).hexdigest(),"source_sha256":EXPECTED,
        "lc1_frozen_to_v23":True,"cpu_only":True,"private":True}},
      "nbformat":4,"nbformat_minor":5},ensure_ascii=False)

def output(account,ref):
    res=call({"action":"output_json_files","account_id":account,"kernel_ref":ref,
              "file_names":["RESULTS.json"],"max_bytes_per_file":262144},timeout=180)["result"]
    for f in res.get("files",[]):
        if pathlib.Path(str(f.get("file_name",""))).name=="RESULTS.json" and isinstance(f.get("json"),dict):
            return f["json"]
    raise RuntimeError("V26_RESULTS_MISSING")

def main():
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(POOL)) as ex:
        inv=list(ex.map(inventory,POOL.keys()))
    dump("ACCOUNT_PREFLIGHT.json",inv)
    safe=sorted([x for x in inv if not x["listing_at_limit"] and x["active_or_unknown"]<2],
                key=lambda x:(x["active_or_unknown"],x["account"]))
    if not safe:
        publish("SoilBin V26 blocked","No safe account headroom; no existing kernel was changed.")
        raise RuntimeError("NO_SAFE_ACCOUNT")
    chosen=safe[0];account=chosen["account"];owner=chosen["owner"];ref=owner+"/"+SLUG
    found=raw(account,"ListKernels",{"group":"PROFILE","sortBy":"DATE_RUN","pageSize":100,"search":SLUG}).get("kernels",[])
    recovered=any(norm(x.get("ref") or x.get("url"))==ref for x in found)
    receipt={"account":account,"owner":owner,"kernel_ref":ref,"recovered_existing":recovered,
             "preflight_safe_accounts":[x["account"] for x in safe],
             "runner_sha256":hashlib.sha256(RUNNER.encode()).hexdigest(),"source_sha256":EXPECTED}
    if not recovered:
        text=notebook()
        res=call({"request_id":CAMPAIGN+"-"+account,"provider":"kaggle","operation_class":"compute",
          "account_id":account,"purpose":"User-authorized SoilBin V26 LC5 transition-aware residual research; private CPU kernel; no other run changed.",
          "service":"kernels.KernelsApiService","method":"SaveKernel","body":{"slug":ref,"newTitle":SLUG,"text":text,
          "language":"PYTHON","kernelType":"NOTEBOOK","kernelExecutionType":"SAVE_AND_RUN_ALL","isPrivate":True,
          "enableGpu":False,"enableInternet":False}},kind="write",timeout=240)
        ref=norm(res.get("provider_ref") or (res.get("result") or {}).get("ref") or ref)
        receipt["kernel_ref"]=ref;receipt["submitted"]=True
        receipt["notebook_sha256"]=hashlib.sha256(text.encode()).hexdigest()
    else: receipt["submitted"]=False
    dump("LAUNCH_RECEIPT.json",receipt)
    publish("SoilBin V26 launch",json.dumps(receipt,ensure_ascii=False))
    end=time.time()+75*60
    while time.time()<end:
        st=status(account,ref);print("V26_STATUS",st,flush=True)
        if st in TERMINAL:
            if st not in {"COMPLETE","COMPLETED"}:raise RuntimeError("V26_TERMINAL_"+st)
            break
        time.sleep(20)
    else:raise RuntimeError("V26_POLL_TIMEOUT")
    result=output(account,ref)
    if result.get("schema")!="soilbin.v26.lc5-transition-aware.v1" or result.get("status")!="COMPLETE":
        raise RuntimeError("V26_RESULT_GATE:"+json.dumps(result)[:1800])
    if result.get("source_sha256")!=EXPECTED:raise RuntimeError("V26_SOURCE_GATE")
    guards=result.get("design_guards") or {}
    if guards.get("LC1_frozen_to_V23_for_all_arms") is not True:raise RuntimeError("V26_LC1_GUARD")
    base=result.get("baseline") or {}
    if abs(float(base.get("V23_joint_Group_MAE_N",-1))-23.104189837450967)>1e-6:raise RuntimeError("V26_V23_GATE")
    dump("RESULTS.json",result)
    final={"account":account,"kernel_ref":ref,"status":"COMPLETE","champion":result.get("champion"),
           "baseline":result.get("baseline"),"runner_sha256":receipt["runner_sha256"],"source_sha256":EXPECTED}
    dump("EVIDENCE_SUMMARY.json",final)
    publish("SoilBin V26 COMPLETE",json.dumps(final,ensure_ascii=False))
    print("SOILBIN_V26_COMPLETE",json.dumps(final,ensure_ascii=False),flush=True)

if __name__=="__main__":
    try:main()
    finally:
        if "GITHUB_ENV" in os.environ:
            with open(os.environ["GITHUB_ENV"],"a",encoding="utf-8") as h:
                h.write("SOILBIN_V26_EVIDENCE_DIR="+str(EVID)+"\n")
