from __future__ import annotations
import base64,concurrent.futures,gzip,hashlib,json,os,pathlib,threading,time,urllib.error,urllib.parse,urllib.request

ROOT=pathlib.Path(__file__).resolve().parent
CAMPAIGN="20260929-v31-v50-state-geometry-r1"
READ="https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/read/kaggle"
WRITE="https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
POOL={"kg-02":"radlinaradlina","kg-03":"rezanory","kg-04":"reyhanehazad","kg-06":"msdenis",
      "kg-07":"nisabulutmark","kg-08":"azadkk","kg-09":"mylovevpn1","kg-11":"jobreza1"}
# kg-05 reserved M07; kg-10 reserved M11; kg-01 intentionally kept as master/control.
WAVE1=[f"V{i}" for i in range(32,49)]
WAVE2=["V31","V49","V50"]
TERMINAL={"COMPLETE","COMPLETED","ERROR","FAILED","CANCELLED","CANCELED","CANCEL_ACKNOWLEDGED"}
TOK={}; LOCK=threading.Lock()
EVID=pathlib.Path(os.environ.get("RUNNER_TEMP","."))/"soilbin-v31-v50"
EVID.mkdir(parents=True,exist_ok=True)
STATE={"campaign":CAMPAIGN,"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
       "lanes":{},"policy":{"private":True,"cpu_only":True,"internet":False,"cancel_existing":False,
       "overwrite_existing":False,"reserved_accounts":["kg-05","kg-10"],"master_not_used":"kg-01"}}

def dump(name,obj):
    (EVID/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def checkpoint(): dump("CAMPAIGN_STATE.json",STATE)

def token(kind):
    with LOCK:
        x=TOK.get(kind)
        if x and time.time()-x[0]<180:return x[1]
        aud="cgp-control-plane-v3-action" if kind=="write" else "cgp-control-plane-v3"
        u=os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
        u+=("&" if "?" in u else "?")+"audience="+urllib.parse.quote(aud,safe="")
        req=urllib.request.Request(u,headers={"Authorization":"Bearer "+os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]})
        with urllib.request.urlopen(req,timeout=30) as r:v=json.load(r)["value"]
        if len(v)<100:raise RuntimeError("OIDC_UNAVAILABLE")
        print("::add-mask::"+v,flush=True);TOK[kind]=(time.time(),v);return v

def call(body,kind="read",timeout=120):
    ep=WRITE if kind=="write" else READ
    for a in range(4):
        req=urllib.request.Request(ep,data=json.dumps(body,separators=(",",":")).encode(),method="POST",
          headers={"Authorization":"Bearer "+token(kind),"Content-Type":"application/json","Accept":"application/json",
                   "User-Agent":"soilbin-v31-v50/1.0"})
        try:
            with urllib.request.urlopen(req,timeout=timeout) as r:val=json.load(r)
            if val.get("ok") is not True:raise RuntimeError("BROKER_REJECTED:"+json.dumps(val)[:1500])
            if kind=="read" and val.get("read_only") is not True:raise RuntimeError("READ_ONLY_GUARD")
            return val
        except urllib.error.HTTPError as e:
            d=e.read(2500).decode("utf-8","replace")
            if e.code==403 and "JWT expired" in d and a<3:
                with LOCK:TOK.pop(kind,None)
                continue
            if e.code in (429,502,503,504) and a<3:
                time.sleep(3*(a+1));continue
            raise RuntimeError(f"BROKER_HTTP_{e.code}:{d}") from e
    raise RuntimeError("BROKER_RETRIES_EXHAUSTED")

def raw(account,method,body):
    return call({"action":"raw_read","account_id":account,"service":"kernels.KernelsApiService",
                 "method":method,"body":body})["result"]
def norm(ref):
    ref=str(ref or "").strip()
    if "kaggle.com/" in ref:ref=ref.split("kaggle.com/",1)[1]
    return ref.removeprefix("/code/").removeprefix("code/").lstrip("/")
def status(account,ref):
    o,s=ref.split("/",1)
    x=raw(account,"GetKernelSessionStatus",{"userName":o,"kernelSlug":s})
    return str(x.get("status") or "PENDING").upper()

def inventory(account):
    owner=POOL[account]
    ks=raw(account,"ListKernels",{"group":"PROFILE","sortBy":"DATE_RUN","pageSize":100}).get("kernels",[])
    refs=[]
    for x in ks:
        r=norm(x.get("ref") or x.get("url") or "")
        if r.startswith(owner+"/") and r not in refs:refs.append(r)
    states=[]
    for r in refs:
        try:s=status(account,r)
        except Exception:s="UNKNOWN"
        states.append({"ref":r,"status":s})
    return {"account":account,"owner":owner,"listing_at_limit":len(ks)>=100,
            "active_or_unknown":sum(x["status"] not in TERMINAL for x in states),
            "kernels_scanned":len(refs)}

RUNNER=(ROOT/"campaign_runner.py").read_text(encoding="utf-8")
MODEL=(ROOT.parent/"soilbin-v27-probabilistic"/"all_features_51.csv.gz.b64").read_text().strip()
WAVEFILE=ROOT/"waveform_native_compact.csv.gz.b64"
if WAVEFILE.exists():
    WAVE=WAVEFILE.read_text().strip()
    _raw=gzip.decompress(base64.b64decode(WAVE))
    WAVE_META={"source":"frozen_waveform_native_compact","rows":51,"feature_csv_sha256":hashlib.sha256(_raw).hexdigest(),"b64_file_sha256":hashlib.sha256(WAVEFILE.read_bytes()).hexdigest()}
else:
    V4_WAVE=(ROOT.parent/"soilbin-q1-v4"/"waveform_used_columns.bin.xz.b64").read_text().strip()
    from derive_wave_features import build_feature_csv
    WAVE,WAVE_META=build_feature_csv(MODEL,V4_WAVE)
dump("WAVE_FEATURE_DERIVATION.json",WAVE_META)

def notebook(lane):
    payload={"lane":lane,"campaign":CAMPAIGN,"model_b64":MODEL}
    if lane in WAVE2:
        if not WAVE:raise RuntimeError("WAVE_FEATURE_PACKAGE_MISSING")
        payload["wave_b64"]=WAVE
    pp=base64.b64encode(gzip.compress(json.dumps(payload,separators=(",",":")).encode(),mtime=0)).decode()
    rr=base64.b64encode(RUNNER.encode()).decode()
    src=["import base64,gzip,json\n",
         f"PAYLOAD=json.loads(gzip.decompress(base64.b64decode({pp!r})))\n",
         f"exec(compile(base64.b64decode({rr!r}),'campaign_runner.py','exec'))\n",
         "main(PAYLOAD)\n"]
    meta={"campaign":CAMPAIGN,"lane":lane,"private":True,"cpu_only":True,
          "runner_sha256":hashlib.sha256(RUNNER.encode()).hexdigest(),
          "data_sha256":"dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"}
    return json.dumps({"cells":[
      {"cell_type":"markdown","metadata":{},"source":[f"# SoilBin {lane} — V31–V50 campaign\n",
       "Private CPU-only; whole-Speed×Load grouped evaluation where predictive; no target modification.\n"]},
      {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src}],
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python"},"soilbin":meta},
      "nbformat":4,"nbformat_minor":5},ensure_ascii=False)

def slug(lane):return f"soilbin-{lane.lower()}-state-geometry-20260929-r1"

def launch(lane,account):
    owner=POOL[account];ref=owner+"/"+slug(lane)
    found=raw(account,"ListKernels",{"group":"PROFILE","sortBy":"DATE_RUN","pageSize":100,"search":slug(lane)}).get("kernels",[])
    if any(norm(x.get("ref") or x.get("url"))==ref for x in found):
        return {"account":account,"owner":owner,"ref":ref,"submitted":False,"recovered":True,"state":status(account,ref)}
    text=notebook(lane)
    res=call({"request_id":f"{CAMPAIGN}-{lane.lower()}-{account}","provider":"kaggle","operation_class":"compute",
      "account_id":account,"purpose":f"User-authorized SoilBin {lane} research; private CPU; no existing run changed.",
      "service":"kernels.KernelsApiService","method":"SaveKernel",
      "body":{"slug":ref,"newTitle":slug(lane),"text":text,"language":"PYTHON","kernelType":"NOTEBOOK",
              "kernelExecutionType":"SAVE_AND_RUN_ALL","isPrivate":True,"enableGpu":False,"enableInternet":False}},
      kind="write",timeout=240)
    actual=norm(res.get("provider_ref") or (res.get("result") or {}).get("ref") or ref)
    return {"account":account,"owner":owner,"ref":actual,"submitted":True,"state":"SUBMITTED",
            "notebook_sha256":hashlib.sha256(text.encode()).hexdigest()}

def output(account,ref):
    x=call({"action":"output_json_files","account_id":account,"kernel_ref":ref,
            "file_names":["RESULTS.json"],"max_bytes_per_file":524288},timeout=180)["result"]
    for f in x.get("files",[]):
        if pathlib.Path(str(f.get("file_name",""))).name=="RESULTS.json" and isinstance(f.get("json"),dict):
            return f["json"]
    return {}

def collect(lane,z):
    st=status(z["account"],z["ref"]);z["state"]=st
    if st in TERMINAL:
        r=output(z["account"],z["ref"])
        z["result_status"]=r.get("status","MISSING_RESULT")
        dd=EVID/lane;dd.mkdir(exist_ok=True)
        (dd/"RESULTS.json").write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8")
        return True
    return False

def run_wave(lanes,safe):
    pending=list(lanes);active={}
    while pending or active:
        # Refresh accounts after every completion; one campaign kernel per account at a time.
        used={z["account"] for z in active.values()}
        free=[x["account"] for x in safe if x["account"] not in used]
        while pending and free:
            lane=pending.pop(0);acc=free.pop(0)
            try:z=launch(lane,acc)
            except Exception as e:z={"account":acc,"state":"LAUNCH_FAILED","error":str(e)[:1800]}
            STATE["lanes"][lane]=z;checkpoint()
            print("LAUNCH",lane,json.dumps(z),flush=True)
            if z.get("ref") and z.get("state")!="LAUNCH_FAILED":active[lane]=z
        if not active and pending:raise RuntimeError("NO_ACTIVE_LANES_BUT_PENDING")
        if active:
            time.sleep(12)
            for lane in list(active):
                try:
                    if collect(lane,active[lane]):
                        print("TERMINAL",lane,active[lane]["state"],active[lane].get("result_status"),flush=True)
                        del active[lane]
                except Exception as e:
                    active[lane]["last_error"]=str(e)[:1500]
            checkpoint()
            print("PROGRESS",json.dumps({k:v.get("state") for k,v in STATE["lanes"].items()}),flush=True)

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        inv=list(ex.map(inventory,POOL.keys()))
    dump("ACCOUNT_PREFLIGHT.json",inv);STATE["inventory"]=inv
    safe=sorted([x for x in inv if not x["listing_at_limit"] and x["active_or_unknown"]<2],
                key=lambda x:(x["active_or_unknown"],x["account"]))
    STATE["safe_accounts"]=[x["account"] for x in safe];checkpoint()
    if len(safe)<3:raise RuntimeError("NEED_AT_LEAST_THREE_SAFE_ACCOUNTS")
    print("SAFE_ACCOUNTS",json.dumps(STATE["safe_accounts"]),flush=True)
    run_wave(WAVE1,safe)
    if WAVE:
        run_wave(WAVE2,safe)
        STATE["wave2"]="EXECUTED"
    else:
        STATE["wave2"]="BLOCKED_WAVE_FEATURE_PACKAGE_MISSING"
        for x in WAVE2:STATE["lanes"].setdefault(x,{"state":"READY_NOT_LAUNCHED_WAVE_PACKAGE_MISSING"})
    STATE["ended_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    terminal=[v.get("state") in TERMINAL for v in STATE["lanes"].values() if v.get("ref")]
    STATE["status"]="COMPLETE" if terminal and all(terminal) and WAVE else "PARTIAL_COMPLETE"
    checkpoint();dump("FINAL_SUMMARY.json",STATE)
    print("CAMPAIGN_FINAL",json.dumps({"status":STATE["status"],"wave2":STATE["wave2"],
          "lanes":{k:v.get("state") for k,v in STATE["lanes"].items()}},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
