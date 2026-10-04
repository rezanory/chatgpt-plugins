#!/usr/bin/env python3
from __future__ import annotations
import argparse, base64, hashlib, json, os, pathlib, zlib
from collections import Counter, defaultdict
import phase2_final_evidence_extract_v1 as broker

CLOSURE_SHA="83481411e937ba1da52c5e95841e68da22f1522ef70b622d624d6a14d65f87f5"
KERNELS=[
("M01",224,"master","azadka/pneumonia-v1-7-phase2-m01-r224-a18-35698388883"),
("M01",320,"master","azadka/pneumonia-v1-7-phase2-m01-r320-a05-35814596649"),
("M01",384,"master","azadka/pneumonia-v1-7-phase2-m01-r384-a05-36351862662"),
("M02",224,"kg-02","radlinaradlina/pneumonia-v1-7-phase2-m02-r224-a19-35698392981"),
("M02",320,"kg-02","radlinaradlina/pneumonia-v1-7-phase2-m02-r320-a05-35814599480"),
("M02",384,"kg-02","radlinaradlina/pneumonia-v1-7-phase2-m02-r384-a05-35927816711"),
("M03",224,"kg-03","rezanory/pneumonia-v1-7-phase2-m03-r224-a19-35698396139"),
("M03",320,"kg-03","rezanory/pneumonia-v1-7-phase2-m03-r320-a05-35814602879"),
("M03",384,"kg-03","rezanory/pneumonia-v1-7-phase2-m03-r384-a05-36362420209"),
("M04",224,"kg-04","reyhanehazad/pneumonia-v1-7-phase2-m04-r224-a19-35698399363"),
("M04",320,"kg-04","reyhanehazad/pneumonia-v1-7-phase2-m04-r320-a05-35814939405"),
("M04",384,"kg-04","reyhanehazad/pneumonia-v1-7-phase2-m04-r384-a05-36337420014"),
("M05",224,"kg-05","trickermark/pneumonia-v1-7-phase2-m05-r224-a12-36269524803"),
("M05",320,"kg-05","trickermark/pneumonia-v1-7-phase2-m05-r320-a04-36370588557"),
("M05",384,"kg-05","trickermark/pneumonia-v1-7-phase2-m05-r384-a05-36485988648"),
("M06",224,"kg-06","msdenis/pneumonia-v1-7-phase2-m06-r224-a18-35698405831"),
("M06",320,"kg-06","msdenis/pneumonia-v1-7-phase2-m06-r320-a05-35814605997"),
("M06",384,"kg-06","msdenis/pneumonia-v1-7-phase2-m06-r384-a05-36371466038"),
("M07",224,"kg-03","rezanory/m07-final-5fold-fix2-20260914"),
("M07",320,"kg-05","trickermark/m07-runtime-r320-a13-20260917-35265801216"),
("M07",384,"kg-05","trickermark/m07-runtime-r384-a12-20260920-35505225105"),
("M08",224,"kg-07","nisabulutmark/pneumonia-v1-7-phase2-m08-r224-a18-35698409656"),
("M08",320,"kg-07","nisabulutmark/pneumonia-v1-7-phase2-m08-r320-a05-35814941709"),
("M08",384,"kg-07","nisabulutmark/pneumonia-v1-7-phase2-m08-r384-a05-36338577091"),
("M09",224,"kg-08","azadkk/pneumonia-v1-7-phase2-m09-r224-a18-35698412479"),
("M09",320,"kg-08","azadkk/pneumonia-v1-7-phase2-m09-r320-a05-35904860843"),
("M09",384,"kg-08","azadkk/pneumonia-v1-7-phase2-m09-r384-a03-36398484836"),
("M10",224,"kg-09","mylovevpn1/pneumonia-v1-7-phase2-m10-r224-a18-35698415361"),
("M10",320,"kg-09","mylovevpn1/pneumonia-v1-7-phase2-m10-r320-a05-35814608153"),
("M10",384,"kg-09","mylovevpn1/pneumonia-v1-7-phase2-m10-r384-a05-36354744478"),
("M11",224,"kg-10","computstu1/pneumonia-v1-7-phase2-m11-r224-a11-36347200480"),
("M11",320,"kg-10","computstu1/pneumonia-v1-7-phase2-m11-r320-a05-36394459369"),
("M11",384,"kg-10","computstu1/pneumonia-v1-7-phase2-m11-r384-a05-36597162497"),
("M12",224,"kg-11","jobreza1/pneumonia-v1-7-phase2-m12-r224-a18-35698425105"),
("M12",320,"kg-11","jobreza1/pneumonia-v1-7-phase2-m12-r320-a05-35826043198"),
("M12",384,"kg-11","jobreza1/pneumonia-v1-7-phase2-m12-r384-a03-36369326430"),
]

def h(b): return hashlib.sha256(b).hexdigest()
def text(c):
    s=c.get("source","")
    return "".join(s) if isinstance(s,list) else str(s)
def lines(s): return s.splitlines(keepends=True)
def md(s,meta=None): return {"cell_type":"markdown","metadata":meta or {},"source":lines(s)}
def code(s,meta=None): return {"cell_type":"code","execution_count":None,"metadata":meta or {},"outputs":[],"source":lines(s)}
def chash(c): return h((str(c.get("cell_type"))+"\0"+text(c)).encode())
def one(root,name):
    xs=[p for p in root.rglob(name) if p.is_file()]
    if len(xs)!=1: raise RuntimeError(f"COUNT:{name}:{len(xs)}")
    return xs[0]

def status(token,account,ref):
    r=broker.post_json(broker.READ_ENDPOINT,token,{"action":"resolved_kernel_status","account_id":account,"kernel_ref":ref},timeout=180)
    if not r.get("ok"): raise RuntimeError("STATUS:"+ref)
    vals=[]; stack=[r]
    while stack:
        x=stack.pop()
        if not isinstance(x,dict): continue
        vals += [str(x[k]).upper() for k in ("status","state","kernel_status") if isinstance(x.get(k),str)]
        stack += [v for v in x.values() if isinstance(v,dict)]
    if "COMPLETE" not in vals: raise RuntimeError("NOT_COMPLETE:"+ref+":"+repr(vals))

def get(token,account,ref):
    owner,slug=ref.split("/",1)
    r=broker.post_json(broker.READ_ENDPOINT,token,{"action":"raw_read","account_id":account,"service":"kernels.KernelsApiService","method":"GetKernel","body":{"userName":owner,"kernelSlug":slug}},timeout=180)
    if not r.get("ok"): raise RuntimeError("GET:"+ref)
    x=broker._read_payload(r); m=x.get("metadata") or {}; b=x.get("blob") or {}
    if m.get("ref")!=ref: raise RuntimeError("REF:"+ref+":"+str(m.get("ref")))
    s=b.get("source")
    if not isinstance(s,str) or not s.strip(): raise RuntimeError("SOURCE:"+ref)
    return m,s

def parse(s,ref):
    try: x=json.loads(s)
    except json.JSONDecodeError: return [code(s,{"phase2_original_ref":ref})]
    if not isinstance(x,dict) or not isinstance(x.get("cells"),list): return [code(s,{"phase2_original_ref":ref})]
    out=[]
    for i,c in enumerate(x["cells"]):
        if not isinstance(c,dict) or c.get("cell_type") not in ("code","markdown","raw"): continue
        z={"cell_type":c["cell_type"],"metadata":dict(c.get("metadata") or {}),"source":lines(text(c))}
        z["metadata"].update({"phase2_original_ref":ref,"phase2_original_index":i})
        if z["cell_type"]=="code": z.update({"execution_count":None,"outputs":[]})
        out.append(z)
    return out

def metric(u,path):
    for s in u.get("json_sources") or []:
        for b in s.get("metric_blocks") or []:
            if b.get("path")==path: return b.get("values") or {}
    return {}
def pct(x):
    try:return f"{100*float(x):.4f}%"
    except:return "—"

def result_table(model,units):
    a=["### نتایج واقعی "+model,"","| R | OOF Acc | Precision | Recall | AUROC | Locked Acc | Locked Precision | Locked Recall |","|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for u in sorted(units,key=lambda q:q["resolution"]):
        o=metric(u,"oof"); l=((u.get("locked_test_recovery") or {}).get("primary") or {}).get("metrics") or {}
        a.append(f"| {u['resolution']} | {pct(o.get('accuracy'))} | {pct(o.get('macro_precision'))} | {pct(o.get('macro_recall',o.get('balanced_accuracy')))} | {pct(o.get('auroc'))} | {pct(l.get('accuracy'))} | {pct(l.get('macro_precision'))} | {pct(l.get('macro_recall',l.get('balanced_accuracy')))} |")
    return "\n".join(a)+"\n"

def main():
    p=argparse.ArgumentParser()
    for n in ("full_results","closure","dual","paired","output","receipt"): p.add_argument("--"+n.replace("_","-"),dest=n,type=pathlib.Path,required=True)
    a=p.parse_args(); token=os.environ["CGP_READ_OIDC_TOKEN"].strip()
    fullp=one(a.full_results,"PHASE2_FULL_RESULTS_MATRIX_V1.json"); clp=one(a.closure,"PHASE2_FINAL_SCIENTIFIC_CLOSURE_V1.json")
    full=json.loads(fullp.read_text()); cl=json.loads(clp.read_text())
    if full.get("status")!="PASS" or full.get("folds_represented")!=180: raise RuntimeError("FULL")
    if cl.get("status")!="CLOSED_PASS" or cl.get("closure_sha256")!=CLOSURE_SHA: raise RuntimeError("CLOSURE")
    ub={(u["model_id"],int(u["resolution"])):u for u in full["units"]}
    store={}; occ=defaultdict(list); seq={}; kernels=[]
    for i,(m,r,acct,ref) in enumerate(KERNELS,1):
        meta,src=get(token,acct,ref); cells=parse(src,ref); q=[]
        for j,c in enumerate(cells):
            d=chash(c); q.append(d); store.setdefault(d,c); occ[d].append({"ref":ref,"model":m,"resolution":r,"index":j})
        seq[ref]=q
        kernels.append({"model":m,"resolution":r,"account_id":acct,"ref":ref,"title":meta.get("title"),"kernel_id":meta.get("id"),"version":meta.get("currentVersionNumber"),"source_sha256":h(src.encode()),"cell_count":len(cells),"status":"FINAL_EVIDENCE_BOUND_SOURCE_FETCHED"})
        print("HARVEST",i,m,r,ref,len(cells),flush=True)
    cnt=Counter({k:len(v) for k,v in occ.items()}); shared={k for k,v in cnt.items() if v>=2}
    manifest={"schema":"pneumonia.phase2.actual-kaggle-source-manifest.v2","kernels":kernels,"sequences":seq,"unique_cells":len(store),"shared_unique_cells":len(shared),"original_cell_occurrences":sum(map(len,seq.values()))}
    nb={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},"phase2_manifest":manifest},"cells":[]}
    nb["cells"].append(md("# PNEUMONIA Phase-2 Master — سورس واقعی ۳۶ Kernel روی ۱۱ حساب Kaggle\n\nاین فایل مستقیم از GetKernel اجرای واقعی ساخته شده است. سلول‌های کاملاً یکسان فقط یک بار آمده‌اند و ترتیب/hash اصلی همه Kernelها در metadata محفوظ است.\n"))
    mb=base64.b64encode(zlib.compress(json.dumps(manifest,ensure_ascii=False,sort_keys=True).encode(),9)).decode()
    nb["cells"].append(code("import base64,json,zlib\nPHASE2_SOURCE_MANIFEST=json.loads(zlib.decompress(base64.b64decode("+repr(mb)+")).decode())\nprint('kernels',len(PHASE2_SOURCE_MANIFEST['kernels']),'unique cells',PHASE2_SOURCE_MANIFEST['unique_cells'],'original occurrences',PHASE2_SOURCE_MANIFEST['original_cell_occurrences'])\n"))
    nb["cells"].append(md("## Shared Exact Cells\n"))
    for d in sorted(shared):
        c=dict(store[d]); c["metadata"]=dict(c.get("metadata") or {}); c["metadata"].update({"phase2_sha256":d,"phase2_occurrences":cnt[d],"phase2_refs":sorted({x["ref"] for x in occ[d]})}); nb["cells"].append(c)
    for m in [f"M{i:02d}" for i in range(1,13)]:
        nb["cells"].append(md("# "+m+"\n"))
        for e in sorted([x for x in kernels if x["model"]==m],key=lambda z:z["resolution"]):
            nb["cells"].append(md(f"## {m} R{e['resolution']}\n\nAccount: {e['account_id']}  \nKernel: {e['ref']}  \nStatus: COMPLETE  \nSource SHA256: {e['source_sha256']}  \nOriginal cells: {e['cell_count']}\n"))
            for d in seq[e["ref"]]:
                if d in shared: continue
                c=dict(store[d]); c["metadata"]=dict(c.get("metadata") or {}); c["metadata"].update({"phase2_sha256":d,"phase2_ref":e["ref"],"phase2_model":m,"phase2_resolution":e["resolution"]}); nb["cells"].append(c)
        nb["cells"].append(md(result_table(m,[ub[(m,r)] for r in (224,320,384)])))
    fr=fullp.read_bytes(); cr=clp.read_bytes()
    fb=base64.b64encode(zlib.compress(fr,9)).decode(); cb=base64.b64encode(zlib.compress(cr,9)).decode()
    nb["cells"].append(md("# Full Results + Final Scientific Closure\n"))
    nb["cells"].append(code("import base64,json,zlib\nPHASE2_FULL_RESULTS=json.loads(zlib.decompress(base64.b64decode("+repr(fb)+")).decode())\nPHASE2_FINAL_CLOSURE=json.loads(zlib.decompress(base64.b64decode("+repr(cb)+")).decode())\nassert PHASE2_FINAL_CLOSURE['closure_sha256']=="+repr(CLOSURE_SHA)+"\nprint(PHASE2_FINAL_CLOSURE['status'],PHASE2_FINAL_CLOSURE['remaining_gate'],PHASE2_FINAL_CLOSURE['training_folds'])\n"))
    extra={}
    for root,prefix in ((a.dual,"dual"),(a.paired,"paired")):
        for f in root.rglob("*"):
            if f.is_file() and f.suffix.lower() in (".json",".csv",".md"): extra[prefix+"/"+f.name]=base64.b64encode(zlib.compress(f.read_bytes(),9)).decode()
    nb["cells"].append(code("PHASE2_EXTRA_STATS="+repr(json.dumps(extra,sort_keys=True))+"\nprint(sorted(json.loads(PHASE2_EXTRA_STATS)))\n"))
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf-8")
    rec={"schema":"pneumonia.phase2.master.actual-kaggle-sources.build.v2","status":"PASS","output_sha256":h(a.output.read_bytes()),"bytes":a.output.stat().st_size,"cells":len(nb["cells"]),"code_cells":sum(c.get("cell_type")=="code" for c in nb["cells"]),"real_kernel_count":36,"unique_source_cells":len(store),"shared_unique_source_cells":len(shared),"original_cell_occurrences":sum(map(len,seq.values())),"closure_sha256":CLOSURE_SHA,"source_kernels":kernels}
    a.receipt.write_text(json.dumps(rec,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); print("BUILD_PASS "+json.dumps(rec,sort_keys=True))

if __name__=="__main__": main()
