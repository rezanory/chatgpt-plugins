"""Parallel controller for SoilBin V17-V25.
Maximizes safe concurrency; never cancels/overwrites existing Kaggle work.
"""
from __future__ import annotations
import base64,concurrent.futures,gzip,hashlib,json,os,pathlib,threading,time,urllib.error,urllib.parse,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
CAMPAIGN="20260928-v17-v25-adaptive-state-r1"
READ="https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/read/kaggle"
WRITE="https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
POOL={"kg-02":"radlinaradlina","kg-03":"rezanory","kg-04":"reyhanehazad","kg-06":"msdenis",
      "kg-07":"nisabulutmark","kg-08":"azadkk","kg-09":"mylovevpn1","kg-11":"jobreza1"}
# kg-05 remains reserved for M07; kg-10 remains reserved for M11.
UPSTREAM=["V17E","V17T","V18","V19","V20","V21","V22"]
TERMINAL={"COMPLETE","COMPLETED","ERROR","FAILED","CANCELLED","CANCELED","CANCEL_ACKNOWLEDGED"}
TOK={};LOCK=threading.Lock()
EVID=pathlib.Path(os.environ.get("RUNNER_TEMP","."))/"soilbin-v17-v25"
EVID.mkdir(parents=True,exist_ok=True)
STATE={"campaign":CAMPAIGN,"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
       "lanes":{},"policy":{"cpu_only":True,"private":True,"cancel_existing":False,"overwrite_existing":False}}

def dump(name,obj):
    (EVID/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def checkpoint():dump("CAMPAIGN_STATE.json",STATE)
def token(kind):
    with LOCK:
        x=TOK.get(kind)
        if x and time.time()-x[0]<180:return x[1]
        audience="cgp-control-plane-v3-action" if kind=="write" else "cgp-control-plane-v3"
        u=os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"];u+=("&" if "?" in u else "?")+"audience="+urllib.parse.quote(audience,safe="")
        req=urllib.request.Request(u,headers={"Authorization":"Bearer "+os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]})
        with urllib.request.urlopen(req,timeout=30) as r:v=json.load(r)["value"]
        if len(v)<100:raise RuntimeError("OIDC_UNAVAILABLE")
        print("::add-mask::"+v,flush=True);TOK[kind]=(time.time(),v);return v
def call(body,kind="read",timeout=120):
    endpoint=WRITE if kind=="write" else READ
    for a in range(3):
        req=urllib.request.Request(endpoint,data=json.dumps(body,separators=(",",":")).encode(),method="POST",
            headers={"Authorization":"Bearer "+token(kind),"Content-Type":"application/json","Accept":"application/json",
                     "User-Agent":"soilbin-v17-v25/1.0"})
        try:
            with urllib.request.urlopen(req,timeout=timeout) as r:val=json.load(r)
            if val.get("ok") is not True:raise RuntimeError("BROKER_REJECTED:"+json.dumps(val)[:1000])
            if kind=="read" and val.get("read_only") is not True:raise RuntimeError("READ_ONLY_GUARD")
            return val
        except urllib.error.HTTPError as e:
            detail=e.read(2500).decode("utf-8","replace")
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
    o,s=ref.split("/",1);x=raw(account,"GetKernelSessionStatus",{"userName":o,"kernelSlug":s})
    return str(x.get("status") or "PENDING").upper()
def inventory(account):
    owner=POOL[account];res=raw(account,"ListKernels",{"group":"PROFILE","sortBy":"DATE_RUN","pageSize":100})
    ks=res.get("kernels",[]);refs=[]
    for x in ks:
        r=norm(x.get("ref") or x.get("url") or "")
        if r.startswith(owner+"/") and r not in refs:refs.append(r)
    ss=[]
    for r in refs:
        try:s=status(account,r)
        except Exception:s="UNKNOWN"
        ss.append({"ref":r,"status":s})
    return {"account":account,"owner":owner,"listing_at_limit":len(ks)>=100,
            "active_or_unknown":sum(x["status"] not in TERMINAL for x in ss),"kernels_scanned":len(refs)}
def output(account,ref):
    res=call({"action":"output_json_files","account_id":account,"kernel_ref":ref,
              "file_names":["RESULTS.json"],"max_bytes_per_file":262144},timeout=180)["result"]
    for f in res.get("files",[]):
        if pathlib.Path(str(f.get("file_name",""))).name=="RESULTS.json" and isinstance(f.get("json"),dict):return f["json"]
    return {}
def publish(title,body):
    tok=os.environ.get("GITHUB_TOKEN")
    if not tok:return
    u=f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/issues/134/comments"
    req=urllib.request.Request(u,data=json.dumps({"body":"## "+title+"\n\n"+body}).encode(),method="POST",
        headers={"Authorization":"Bearer "+tok,"Accept":"application/vnd.github+json","Content-Type":"application/json",
                 "User-Agent":"soilbin-v17-v25"})
    with urllib.request.urlopen(req,timeout=30):pass

RUNNER=(ROOT/"runner.py").read_text(encoding="utf-8")
MODEL=(ROOT/"all_features_51.csv.gz.b64").read_text().strip()
ERDC=(ROOT/"external_erdc_crrel_2009.csv.gz.b64").read_text().strip()
CANADA=(ROOT/"external_canada_ayetan_2026.csv.gz.b64").read_text().strip()

def notebook(lane,payload):
    pp=base64.b64encode(gzip.compress(json.dumps(payload,separators=(",",":")).encode(),mtime=0)).decode()
    rr=base64.b64encode(RUNNER.encode()).decode()
    src=["import base64,gzip,json\n",f"PAYLOAD=json.loads(gzip.decompress(base64.b64decode({pp!r})))\n",
         f"exec(compile(base64.b64decode({rr!r}),'soilbin_v17_v25_runner.py','exec'))\n"]
    return json.dumps({"cells":[
      {"cell_type":"markdown","metadata":{},"source":[f"# SoilBin {lane} — adaptive state campaign\n",
        "Private CPU-only post-lock exploration; grouped validation; no target modification.\n"]},
      {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src}],
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"},"soilbin":{"campaign":CAMPAIGN,"lane":lane,"private":True,"cpu_only":True,
        "runner_sha256":hashlib.sha256(RUNNER.encode()).hexdigest()}},
      "nbformat":4,"nbformat_minor":5},ensure_ascii=False)
def slug(lane):
    return f"soilbin-{lane.lower()}-adaptive-state-20260928-r1"
def make_payload(lane,upstream=None,v23=None):
    p={"lane":lane,"campaign":CAMPAIGN,"model_b64":MODEL}
    if lane in ("V23","V24","V25"):p["upstream_results"]=upstream or {}
    if lane in ("V24","V25") and v23 is not None:p["v23_result"]=v23
    if lane=="V25":p.update({"erdc_b64":ERDC,"canada_b64":CANADA})
    return p
def launch(lane,account,payload):
    owner=POOL[account];ref=owner+"/"+slug(lane)
    found=raw(account,"ListKernels",{"group":"PROFILE","sortBy":"DATE_RUN","pageSize":100,"search":slug(lane)}).get("kernels",[])
    if any(norm(x.get("ref"))==ref for x in found):
        return {"account":account,"owner":owner,"ref":ref,"submitted":False,"recovered":True,"state":status(account,ref)}
    text=notebook(lane,payload)
    res=call({"request_id":f"{CAMPAIGN}-{lane.lower()}-{account}","provider":"kaggle","operation_class":"compute",
      "account_id":account,"purpose":f"User-authorized SoilBin {lane} adaptive-state research; private CPU kernel; no other run changed.",
      "service":"kernels.KernelsApiService","method":"SaveKernel","body":{"slug":ref,"newTitle":slug(lane),"text":text,
      "language":"PYTHON","kernelType":"NOTEBOOK","kernelExecutionType":"SAVE_AND_RUN_ALL","isPrivate":True,
      "enableGpu":False,"enableInternet":False}},kind="write",timeout=240)
    actual=norm(res.get("provider_ref") or (res.get("result") or {}).get("ref") or ref)
    return {"account":account,"owner":owner,"ref":actual,"submitted":True,"state":"SUBMITTED",
            "notebook_sha256":hashlib.sha256(text.encode()).hexdigest(),"runner_sha256":hashlib.sha256(RUNNER.encode()).hexdigest()}
def collect_lane(lane,z):
    st=status(z["account"],z["ref"]);z["state"]=st
    if st in TERMINAL:
        r=output(z["account"],z["ref"])
        z["result_status"]=r.get("status","MISSING_RESULT")
        dd=EVID/lane;dd.mkdir(exist_ok=True)
        (dd/"RESULTS.json").write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8")
        return True,r
    return False,None
def wait_lanes(lanes,deadline_minutes):
    pending=set(lanes);results={}
    end=time.time()+deadline_minutes*60
    while pending and time.time()<end:
        for lane in list(pending):
            z=STATE["lanes"][lane]
            try:
                done,r=collect_lane(lane,z)
                if done:results[lane]=r;pending.remove(lane)
            except Exception as e:z["last_error"]=str(e)[:1200]
        checkpoint()
        print("CAMPAIGN_PROGRESS",json.dumps({x:STATE["lanes"][x].get("state") for x in lanes}),flush=True)
        if pending:time.sleep(20)
    for lane in pending:
        STATE["lanes"][lane]["state"]="POLL_TIMEOUT"
        results[lane]={"lane":lane,"status":"POLL_TIMEOUT"}
    checkpoint();return results

def main():
    # Capacity audit
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        inv=list(ex.map(inventory,POOL.keys()))
    dump("ACCOUNT_PREFLIGHT.json",inv)
    safe=sorted([x for x in inv if not x["listing_at_limit"] and x["active_or_unknown"]<2],
                key=lambda x:(x["active_or_unknown"],x["account"]))
    if len(safe)<3:
        STATE["status"]="BLOCKED_INSUFFICIENT_SAFE_ACCOUNTS";STATE["inventory"]=inv;checkpoint()
        raise RuntimeError("NEED_AT_LEAST_THREE_SAFE_ACCOUNTS")
    publish("SoilBin V17–V25 campaign preflight",
            f"Safe accounts available: {len(safe)}/{len(POOL)}. Upstream lanes will be scheduled at maximum safe concurrency; no existing run will be cancelled.")
    # Wave scheduler for 7 independent upstream lanes.
    pending=list(UPSTREAM);active={};results={}
    while pending or active:
        used={z["account"] for z in active.values()}
        free=[x["account"] for x in safe if x["account"] not in used]
        while pending and free:
            lane=pending.pop(0);acc=free.pop(0)
            try:z=launch(lane,acc,make_payload(lane))
            except Exception as e:z={"account":acc,"state":"LAUNCH_FAILED","error":str(e)[:1500]}
            STATE["lanes"][lane]=z;active[lane]=z;checkpoint()
            print("LAUNCH",lane,json.dumps(z),flush=True)
        time.sleep(3)
        for lane in list(active):
            z=active[lane]
            if z.get("state")=="LAUNCH_FAILED":
                results[lane]={"lane":lane,"status":"LAUNCH_FAILED","error":z.get("error")};active.pop(lane);continue
            try:
                done,r=collect_lane(lane,z)
                if done:results[lane]=r;active.pop(lane)
            except Exception as e:z["last_error"]=str(e)[:1200]
        checkpoint()
        if active:time.sleep(17)
    dump("UPSTREAM_RESULTS.json",results)
    publish("SoilBin V17–V22 parallel terminal",
      json.dumps({l:{"account":STATE["lanes"][l]["account"],"state":STATE["lanes"][l].get("state"),
                     "result":results.get(l,{}).get("status")} for l in UPSTREAM},ensure_ascii=False))
    # V23 fan-in on preferred kg-09 if safe, otherwise first safe.
    accounts=[x["account"] for x in safe];a23="kg-09" if "kg-09" in accounts else accounts[0]
    z=launch("V23",a23,make_payload("V23",results));STATE["lanes"]["V23"]=z;checkpoint()
    r23=wait_lanes(["V23"],60)["V23"];results["V23"]=r23;dump("V23_RESULT.json",r23)
    # V24 and V25 can both consume frozen V23 result and run in parallel.
    post_accounts=[a for a in accounts if a!=a23]
    if len(post_accounts)<2:post_accounts=(accounts*2)[:2]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fs={}
        for lane,acc in zip(("V24","V25"),post_accounts[:2]):
            fs[ex.submit(launch,lane,acc,make_payload(lane,results,r23))]=(lane,acc)
        for fut,(lane,acc) in fs.items():
            try:z=fut.result()
            except Exception as e:z={"account":acc,"state":"LAUNCH_FAILED","error":str(e)[:1500]}
            STATE["lanes"][lane]=z
    checkpoint()
    tail=wait_lanes(["V24","V25"],90)
    results.update(tail);dump("RESULTS_ALL_V17_V25.json",results)
    STATE["status"]="EXECUTION_TERMINAL";STATE["finished_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());checkpoint()
    publish("SoilBin V17–V25 campaign terminal",
      json.dumps({l:{"account":STATE["lanes"].get(l,{}).get("account"),"state":STATE["lanes"].get(l,{}).get("state"),
                     "result":results.get(l,{}).get("status")} for l in UPSTREAM+["V23","V24","V25"]},ensure_ascii=False))
    print("SOILBIN_V17_V25_TERMINAL",json.dumps(STATE,ensure_ascii=False),flush=True)

if __name__=="__main__":
    try:main()
    finally:
        if "GITHUB_ENV" in os.environ:
            with open(os.environ["GITHUB_ENV"],"a",encoding="utf-8") as h:h.write("SOILBIN_V17_V25_EVIDENCE_DIR="+str(EVID)+"\n")
