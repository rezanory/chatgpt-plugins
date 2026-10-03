#!/usr/bin/env python3
from __future__ import annotations
import argparse, base64, hashlib, io, json, pathlib, re, textwrap, zipfile

BASE_SHA="219d97dd03c6d8de87b4e40f9d25be618c4d7d3bbda475b2ee3c2e68ad6a3a8d"
CLOSURE_SHA="83481411e937ba1da52c5e95841e68da22f1522ef70b622d624d6a14d65f87f5"
SOURCE_FILES=(
"scripts/pneumonia_phase2_unit.py",
"scripts/phase2_dual_pr_oof_stats_v1.py",
"scripts/phase2_paired_oof_stats_v1.py",
"scripts/phase2_finalist_rsna_external_dispatch_v1.py",
"scripts/m07_rsna_pediatric_external_v1.py",
"scripts/phase2_final_scientific_closure_v1.py",
"scripts/phase2_full_results_reconcile_v1.py",
"scripts/test_pneumonia_phase2_unit.py",
)
FIXES=(
("a6825ac","storage-safe finalization and restore headroom"),
("8b1d590","transient Kaggle state reads"),
("8d2209a","XAI helper topology"),
("c7c749f","decision curve helper"),
("a146ed9","risk coverage helper"),
("e90e329","reliability correctness"),
("06537e","reliability plot helper"),
("8a464c","locked-test logit helper"),
("5ae71d","validation history figures"),
("0852b","confusion helper"),
("4f551b","state snapshot classification"),
("cbb3d7","state lineage"),
("75bb68","token-only owner reconciliation"),
("9a05b8","state restore file semantics"),
("d387fe","A03 runtime recovery"),
("0b060f","frozen M07 recipe binding"),
("d43e2a","legacy M07 gate isolation"),
("f399fa","Phase-2 validation hardening"),
("dcc53ac","distributed comparator campaign"),
("ef9d2bd","external archive materialization"),
("f66a594","cgpzip external restore"),
("8a75191","GPU capacity CPU fallback"),
("d7727fc","corrected final closure"),
("331aa97","verified-output finalist handoff"),
("199bd74","final closure artifact path fix"),
)

def h(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def src(cell):
    v=cell.get("source",[])
    return "".join(v) if isinstance(v,list) else str(v)
def setsrc(cell,s):
    cell["source"]=s.splitlines(keepends=True)
    if cell.get("cell_type")=="code":
        cell["execution_count"]=None;cell["outputs"]=[]
def md(s):return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}
def code(s,tags=()):
    return {"cell_type":"code","execution_count":None,"metadata":{"tags":list(tags)},"outputs":[],"source":s.splitlines(keepends=True)}
def one(root,name):
    rows=[p for p in root.rglob(name) if p.is_file()]
    if len(rows)!=1:raise RuntimeError(f"INPUT_COUNT:{name}:{len(rows)}")
    return rows[0]

def gate(s,i):
    return "# exact corrected runtime cell %d\nif MASTER_RUN_FULL:\n"%i+textwrap.indent(s.rstrip()+"\n","    ")+"else:\n    print('MASTER_REPLAY_MODE: runtime cell %d skipped')\n"%i

def patch_runtime(nb):
    if len(nb.get("cells",[]))!=16:raise RuntimeError("BASE_CELL_TOPOLOGY")
    joined="\n".join(src(c) for c in nb["cells"])
    markers=("PHASE2_UNIT_FROZEN_M07_RECIPE_BOUND","PHASE2_UNIT_LEGACY_M07_PERSISTENCE_SKIPPED","def phase2_restore(","_PHASE2_TRANSIENT_HTTP_STATUSES","_phase2_require_restore_headroom","def run_phase2_model_resolution(","PHASE2_UNIT_NEW_FOLD_BOUND_EXCEEDED",".cgpzip")
    miss=[x for x in markers if x not in joined]
    if miss:raise RuntimeError("RUNTIME_MARKERS_MISSING="+",".join(miss))
    s=src(nb["cells"][1]).replace("PHASE2_RUN_EXTERNAL = True","PHASE2_RUN_EXTERNAL = False")
    pat=re.compile(r"# Exact Phase-2 unit identity injected by pneumonia_phase2_unit\.py\.\nPHASE2_UNIT_TOKEN = .*?\nPHASE2_UNIT_MODEL_ID = .*?\nPHASE2_UNIT_RESOLUTION = .*?\nPHASE2_UNIT_ATTEMPT = .*?\nPHASE2_UNIT_CONTRACT_SHA256 = .*?\nPHASE2_PERSIST_OWNER = .*?\nUNLOCK_REMAINING_MODELS = True\nCGP_PHASE2_MAX_NEW_FOLDS = .*?\nCGP_PHASE2_EXPECTED_RESTORED_FOLDS = .*?\n",re.S)
    repl="""# Master integrated identity.
PHASE2_UNIT_TOKEN = "MASTER_INTEGRATED_V1"
PHASE2_UNIT_MODEL_ID = "MASTER"
PHASE2_UNIT_RESOLUTION = 0
PHASE2_UNIT_ATTEMPT = 1
PHASE2_UNIT_CONTRACT_SHA256 = "MASTER_INTEGRATED_RUNTIME"
PHASE2_PERSIST_OWNER = "azadka"
UNLOCK_REMAINING_MODELS = True
CGP_PHASE2_MAX_NEW_FOLDS = 5
CGP_PHASE2_EXPECTED_RESTORED_FOLDS = []
"""
    s,n=pat.subn(repl,s,count=1)
    if n!=1:raise RuntimeError("IDENTITY_PATCH")
    setsrc(nb["cells"][1],s)
    s=src(nb["cells"][13])
    old='''def phase2_slug(model_id, resolution):
    if str(model_id) == "M07":
        return f"m07-gate-r{int(resolution)}-state-v1-7"
    return f"pneumonia-{model_id.lower()}-r{int(resolution)}-state-v1-7"
'''
    new='''def phase2_slug(model_id, resolution):
    if str(model_id) == "M07":
        return f"m07-gate-r{int(resolution)}-state-v1-7-master-integrated-v1"
    return f"pneumonia-{model_id.lower()}-r{int(resolution)}-state-v1-7-master-integrated-v1"
'''
    if s.count(old)!=1:raise RuntimeError("STATE_NAMESPACE_PATCH")
    setsrc(nb["cells"][13],s.replace(old,new,1))
    s=src(nb["cells"][14])
    old='''    if restored_folds != list(CGP_PHASE2_EXPECTED_RESTORED_FOLDS):
        raise RuntimeError(
            "BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: "
            f"expected restored folds {list(CGP_PHASE2_EXPECTED_RESTORED_FOLDS)}, "
            f"observed {restored_folds}"
        )
'''
    new='''    if restored_folds != list(range(1, len(restored_folds) + 1)):
        raise RuntimeError("MASTER_RESTORE_LINEAGE_NONCONTIGUOUS: " + str(restored_folds))
    globals()["CGP_PHASE2_EXPECTED_RESTORED_FOLDS"] = list(restored_folds)
'''
    if s.count(old)!=1:raise RuntimeError("LINEAGE_PATCH")
    setsrc(nb["cells"][14],s.replace(old,new,1))
    for i in range(1,15):setsrc(nb["cells"][i],gate(src(nb["cells"][i]),i))

def package(args,repo):
    buf=io.BytesIO();manifest={}
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        rows=[
        (one(args.full_results,"PHASE2_FULL_RESULTS_MATRIX_V1.json"),"results/PHASE2_FULL_RESULTS_MATRIX_V1.json"),
        (one(args.closure,"PHASE2_FINAL_SCIENTIFIC_CLOSURE_V1.json"),"closure/PHASE2_FINAL_SCIENTIFIC_CLOSURE_V1.json"),
        (one(args.closure,"PHASE2_FINAL_SCIENTIFIC_CLOSURE_MANIFEST_V1.json"),"closure/PHASE2_FINAL_SCIENTIFIC_CLOSURE_MANIFEST_V1.json"),
        ]
        for root,prefix in ((args.dual,"dual"),(args.paired,"paired")):
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.suffix.lower() in (".json",".csv",".md"):rows.append((p,prefix+"/"+p.name))
        for root,prefix in ((args.external_m09,"external/M09"),(args.external_m10,"external/M10")):
            for p in sorted(root.rglob("*")):
                if p.is_file() and ("_HANDOFF" in str(p) or p.suffix.lower() in (".json",".zip")):rows.append((p,prefix+"/"+p.name))
        rows.append((repo/"evidence/phase2_final_closure_v1/SELECTION_FREEZE.json","closure/SELECTION_FREEZE.json"))
        for rel in SOURCE_FILES:
            p=repo/rel
            if p.is_file():rows.append((p,"source/"+rel))
        seen=set()
        for p,arc in rows:
            if arc in seen:continue
            seen.add(arc);raw=p.read_bytes();z.writestr(arc,raw);manifest[arc]={"sha256":h(raw),"bytes":len(raw)}
        z.writestr("MANIFEST.json",json.dumps(manifest,indent=2,sort_keys=True))
    return buf.getvalue(),manifest

def results_cell(b64,zsha):
    return """# Embedded real evidence and final results.
import base64, hashlib, io, json, pathlib, zipfile
import pandas as pd
from IPython.display import display
MASTER_EVIDENCE_ZIP_SHA256=%r
MASTER_EVIDENCE_B64=%r
raw=base64.b64decode(MASTER_EVIDENCE_B64)
if hashlib.sha256(raw).hexdigest()!=MASTER_EVIDENCE_ZIP_SHA256:raise RuntimeError("EVIDENCE_SHA")
root=(pathlib.Path("/kaggle/working") if pathlib.Path("/kaggle/working").exists() else pathlib.Path.cwd())/"PNEUMONIA_PHASE2_MASTER_EVIDENCE"
root.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as z:z.extractall(root)
matrix=json.loads((root/"results/PHASE2_FULL_RESULTS_MATRIX_V1.json").read_text())
closure=json.loads((root/"closure/PHASE2_FINAL_SCIENTIFIC_CLOSURE_V1.json").read_text())
if closure.get("closure_sha256")!=%r or closure.get("status")!="CLOSED_PASS":raise RuntimeError("CLOSURE_IDENTITY")
def metric(u,path):
    for s in u.get("json_sources") or []:
        for b in s.get("metric_blocks") or []:
            if b.get("path")==path:return b.get("values") or {}
    return {}
units=[];folds=[]
for u in matrix["units"]:
    o=metric(u,"oof");lk=((u.get("locked_test_recovery") or {}).get("primary") or {}).get("metrics") or {}
    for f in u.get("recovered_folds") or []:
        tm=f.get("train_metrics");vm=f.get("validation_metrics")
        folds.append({"model":u["model_id"],"resolution":u["resolution"],"fold":f.get("fold_id"),"train_accuracy":(tm or {}).get("accuracy") if isinstance(tm,dict) else None,"train_macro_precision":(tm or {}).get("macro_precision") if isinstance(tm,dict) else None,"train_macro_recall":(tm or {}).get("balanced_accuracy") if isinstance(tm,dict) else None,"val_accuracy":(vm or {}).get("accuracy") if isinstance(vm,dict) else None,"val_macro_precision":(vm or {}).get("macro_precision") if isinstance(vm,dict) else None,"val_macro_recall":(vm or {}).get("balanced_accuracy") if isinstance(vm,dict) else None})
    units.append({"model":u["model_id"],"resolution":u["resolution"],"oof_accuracy":o.get("accuracy"),"oof_macro_precision":o.get("macro_precision"),"oof_macro_recall":o.get("macro_recall",o.get("balanced_accuracy")),"oof_auroc":o.get("auroc"),"locked_accuracy":lk.get("accuracy"),"locked_macro_precision":lk.get("macro_precision"),"locked_macro_recall":lk.get("macro_recall",lk.get("balanced_accuracy"))})
MASTER_UNIT_RESULTS=pd.DataFrame(units).sort_values(["model","resolution"]).reset_index(drop=True)
MASTER_FOLD_RESULTS=pd.DataFrame(folds).sort_values(["model","resolution","fold"]).reset_index(drop=True)
print("FINAL",closure["status"],"remaining_gate",closure["remaining_gate"],"folds",closure["training_folds"])
display(MASTER_UNIT_RESULTS);display(MASTER_FOLD_RESULTS)
p=root/"dual/DUAL_PR_PAIRED_HOLM_72.csv"
if p.exists():MASTER_DUAL_PR_STATS=pd.read_csv(p);display(MASTER_DUAL_PR_STATS)
final=[]
for x in closure["frozen_candidates"]:
    e=x["external_report_only"]["expanded_pediatric_le18"];l=x["locked_test_report_only"]["metrics"];d=x["development_oof"]
    final.append({"model":x["model_id"],"resolution":x["resolution"],"dev_precision":d["macro_precision"],"dev_recall":d["macro_recall"],"locked_precision":l["macro_precision"],"locked_recall":l["macro_recall"],"external_precision":e["macro_precision"],"external_recall":e["macro_recall"],"external_accuracy":e["accuracy"]})
MASTER_FINALISTS=pd.DataFrame(final);display(MASTER_FINALISTS)
receipt={"schema":"pneumonia.phase2.master.integrated.replay.v1","status":"PASS","closure_sha256":closure["closure_sha256"],"units":36,"folds":180,"mode":"FULL_RERUN" if MASTER_RUN_FULL else "SEALED_RESULTS_REPLAY"}
out=(pathlib.Path("/kaggle/working") if pathlib.Path("/kaggle/working").exists() else pathlib.Path.cwd())/"PNEUMONIA_PHASE2_MASTER_REPLAY_RECEIPT.json"
out.write_text(json.dumps(receipt,indent=2,sort_keys=True))
print("MASTER_REPLAY_RECEIPT="+json.dumps(receipt,sort_keys=True))
"""%(zsha,b64,CLOSURE_SHA)

def driver():
    return '''# Single-account resumable re-execution driver.
MASTER_MODEL_ORDER=tuple("M%02d"%i for i in range(1,13))
MASTER_RESOLUTIONS=(224,320,384)
MASTER_MAX_UNITS_PER_SESSION=int(os.environ.get("PNEUMONIA_MASTER_MAX_UNITS_PER_SESSION","1"))
if MASTER_RUN_FULL:
    if PHASE2_PERSIST_OWNER!="azadka":raise RuntimeError("MASTER_OWNER_DRIFT")
    done=[];count=0
    for model in MASTER_MODEL_ORDER:
        for resolution in MASTER_RESOLUTIONS:
            if count>=MASTER_MAX_UNITS_PER_SESSION:break
            restored=phase2_restore(model,resolution)
            CGP_PHASE2_EXPECTED_RESTORED_FOLDS=list(restored)
            CGP_PHASE2_MAX_NEW_FOLDS=5
            result=run_phase2_model_resolution(model,resolution)
            done.append({"model_id":model,"resolution":resolution,"status":result.get("status")})
            count+=1
            try:tf.keras.backend.clear_session()
            except Exception:pass
            gc.collect()
        if count>=MASTER_MAX_UNITS_PER_SESSION:break
    print("MASTER_FULL_RERUN_SESSION_RESULTS="+json.dumps(done,sort_keys=True))
else:
    print("Replay mode: no training, state mutation, locked-test inference or external inference.")
'''

def main():
    a=argparse.ArgumentParser()
    for x in ("unit_artifact","full_results","closure","dual","paired","external_m09","external_m10","repo_root","output","receipt"):a.add_argument("--"+x.replace("_","-"),dest=x,type=pathlib.Path,required=True)
    args=a.parse_args();repo=args.repo_root
    unit=one(args.unit_artifact,"PNEUMONIA_V17_D260914D_M07_PHASE2_UNLOCK.ipynb");raw=unit.read_bytes()
    if h(raw)!=BASE_SHA:raise RuntimeError("BASE_SHA")
    nb=json.loads(raw.decode());patch_runtime(nb)
    ev,manifest=package(args,repo);zsha=h(ev);b64=base64.b64encode(ev).decode()
    fixes="\n".join("- "+c+" : "+d for c,d in FIXES)
    intro="# PNEUMONIA Phase-2 Master Integrated Final V1\n\nSingle-file M01-M12 / R224-R320-R384 package. It uses the exact corrected runtime notebook from a real successful unit, embeds all final results, and includes current-main source for later external/closure fixes.\n\nDefault is safe sealed-results replay. Set MASTER_RUN_FULL=True to start/resume a fresh campaign under azadka. State names use the dedicated master-integrated-v1 suffix and never overwrite the sealed distributed campaign. Because 180 folds are not practical in one Kaggle session, full rerun is resumable and defaults to one model-resolution unit per session.\n\n## Accepted fix audit\n"+fixes+"\n\n---\n\n"+src(nb["cells"][0])
    nb["cells"][0]=md(intro)
    control='''# MASTER INTEGRATED CONTROL
import os
MASTER_RUN_FULL=os.environ.get("PNEUMONIA_MASTER_RUN_FULL","0").strip().lower() in {"1","true","yes"}
MASTER_RUN_EXTERNAL=os.environ.get("PNEUMONIA_MASTER_RUN_EXTERNAL","0").strip().lower() in {"1","true","yes"}
print("MASTER_RUN_FULL=",MASTER_RUN_FULL,"MASTER_RUN_EXTERNAL=",MASTER_RUN_EXTERNAL)
'''
    nb["cells"].insert(1,code(control,("master-control",)))
    nb["cells"][16]=code(driver(),("master-driver",))
    nb["cells"].append(md("## Sealed final results and embedded evidence"))
    nb["cells"].append(code(results_cell(b64,zsha),("master-results",)))
    nb.setdefault("metadata",{})["pneumonia_master_integrated"]={"schema":"pneumonia.phase2.master.integrated.v1","status":"BUILT_FROM_REAL_EXECUTED_RUNTIME","master_owner":"azadka","kaggle_kernel_ref":"azadka/pneumonia-phase2-master-integrated-v1","base_executed_notebook_sha256":BASE_SHA,"embedded_evidence_zip_sha256":zsha,"final_closure_sha256":CLOSURE_SHA,"source_commit":"199bd7436ff4e4619f424374848c27f6965b3279","default_mode":"SEALED_RESULTS_REPLAY","state_namespace_suffix":"master-integrated-v1"}
    for i,c in enumerate(nb["cells"]):
        if c.get("cell_type")=="code":compile(src(c),"<cell-%d>"%i,"exec")
    joined="\n".join(src(c) for c in nb["cells"])
    checks={"frozen_recipe":"PHASE2_UNIT_FROZEN_M07_RECIPE_BOUND" in joined,"transient_reads":"_PHASE2_TRANSIENT_HTTP_STATUSES" in joined,"headroom":"_phase2_require_restore_headroom" in joined,"cgpzip":".cgpzip" in joined,"five_fold_budget":"CGP_PHASE2_MAX_NEW_FOLDS = 5" in joined,"master_namespace":"master-integrated-v1" in joined,"master_owner":'PHASE2_PERSIST_OWNER = "azadka"' in joined,"closure":CLOSURE_SHA in joined}
    if not all(checks.values()):raise RuntimeError("AUDIT="+json.dumps(checks))
    args.output.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf-8")
    rec={"schema":"pneumonia.phase2.master.integrated.build.v1","status":"PASS","output":args.output.name,"output_sha256":h(args.output.read_bytes()),"bytes":args.output.stat().st_size,"cells":len(nb["cells"]),"base_executed_notebook_sha256":BASE_SHA,"embedded_evidence_zip_sha256":zsha,"embedded_evidence_file_count":len(manifest),"final_closure_sha256":CLOSURE_SHA,"critical_fix_audit":checks,"master_owner":"azadka","kaggle_kernel_ref":"azadka/pneumonia-phase2-master-integrated-v1"}
    args.receipt.write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n");print("PHASE2_MASTER_INTEGRATED_BUILD_PASS "+json.dumps(rec,sort_keys=True))
if __name__=="__main__":main()
