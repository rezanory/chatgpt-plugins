from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/"soilbin-v28-force-reverse-ensemble"
src=(BASE/"outer_worker.py").read_text(encoding="utf-8")
defs=src[:src.index("outer=sys.argv[1]")]
ns={"__name__":"soilbin_v281_defs","__file__":str(BASE/"outer_worker.py")}
exec(compile(defs,str(BASE/"outer_worker.py"),"exec"),ns)
h=ns["h"];N=ns["ns"]

variants=("V23_BASE","STACKED_BOTH","LC5_STACKED","LC5_STACKED_GATED",
          "LC5_T2_STACKED","LC5_T2T6_STACKED","STACKED_GATED_BOTH")
P={v:np.full((len(h),2),np.nan) for v in variants}
audits=[];elapsed={}
for outer in sorted(h.Group.unique()):
    f=ROOT/"workers"/f"{outer}.json"
    if not f.exists():raise RuntimeError("MISSING:"+outer)
    x=json.loads(f.read_text(encoding="utf-8"))
    idx=np.asarray(x["row_indices"],int)
    for v in variants:P[v][idx]=np.asarray(x["predictions"][v],float)
    audits.append(x["audit"]);elapsed[outer]=x["elapsed_seconds"]
for v in variants:
    if not np.isfinite(P[v]).all():raise RuntimeError("INCOMPLETE:"+v)

y=h[N["TARGETS"]].to_numpy(float);g=h.Group.to_numpy()
base=N["force_metrics"](y,P["V23_BASE"],g)
if abs(base["joint_Group_MAE_N"]-N["V23_LOCK"])>1e-6:raise RuntimeError("V23_LOCK_FAIL")
summary=[]
for v in variants:
    m=N["force_metrics"](y,P[v],g)
    row={"variant":v,**m}
    row["joint_improvement_vs_V23_N"]=float(base["joint_Group_MAE_N"]-m["joint_Group_MAE_N"])
    row["joint_improvement_vs_V23_pct"]=float(100*(base["joint_Group_MAE_N"]-m["joint_Group_MAE_N"])/base["joint_Group_MAE_N"])
    row["LC1_improvement_vs_V23_N"]=float(base["LC1_Group_MAE_N"]-m["LC1_Group_MAE_N"])
    row["LC5_improvement_vs_V23_N"]=float(base["LC5_Group_MAE_N"]-m["LC5_Group_MAE_N"])
    row["LC5_improvement_vs_V23_pct"]=float(100*(base["LC5_Group_MAE_N"]-m["LC5_Group_MAE_N"])/base["LC5_Group_MAE_N"])
    row["paired_joint_vs_V23"]=N["paired_group_report"](y,P[v],P["V23_BASE"],g)
    row["paired_LC5_vs_V23"]=N["_lc5_pair_report"](y[:,1],P[v][:,1],P["V23_BASE"][:,1],g)
    row["LC1_pass_MAE_N"]={str(int(t)):float(np.mean(np.abs(y[h.Pass_T.to_numpy()==t,0]-P[v][h.Pass_T.to_numpy()==t,0]))) for t in sorted(h.Pass_T.unique())}
    row["LC5_pass_MAE_N"]={str(int(t)):float(np.mean(np.abs(y[h.Pass_T.to_numpy()==t,1]-P[v][h.Pass_T.to_numpy()==t,1]))) for t in sorted(h.Pass_T.unique())}
    summary.append(row)
champion=min(summary,key=lambda x:x["joint_Group_MAE_N"])
best5=min(summary,key=lambda x:x["LC5_Group_MAE_N"])
result={
 "schema":"soilbin.v28.1.stacked-force.v1","status":"COMPLETE","post_hoc_exploratory":True,
 "question":"Can an OOF-trained stacker of V23 forward and reverse-inverted predictions improve the force endpoint?",
 "source_sha256":N["EXPECTED_DATA"],"arms":list(variants),"summary":summary,
 "champion":champion,"best_LC5":best5,"audit":audits,"worker_elapsed_seconds":elapsed,
 "guards":{"V23_reproduced_exactly":True,"actual_future_target_used_at_inference":False,
   "base_predictions_for_meta_are_outer_training_OOF":True,"meta_second_level_group_OOF_used_for_pass_gate":True,
   "meta_alpha_fixed":10.0,"outer_speed_load_group_held_out":True,"random_split_used":False}
}
(ROOT/"RESULTS.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
print("V281_STACK_FANIN_COMPLETE",json.dumps({"champion":champion["variant"],"joint":champion["joint_Group_MAE_N"],
 "LC1":champion["LC1_Group_MAE_N"],"LC5":champion["LC5_Group_MAE_N"],
 "best_LC5_variant":best5["variant"],"best_LC5":best5["LC5_Group_MAE_N"]},ensure_ascii=False),flush=True)
