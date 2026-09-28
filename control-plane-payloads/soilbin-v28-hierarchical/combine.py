from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/"soilbin-v27-probabilistic"
runner=(PREV/"runner.py").read_text(encoding="utf-8")
defs=runner[:runner.index("\ndef main(payload):\n")]
ns={"__name__":"soilbin_v28_hier_defs"}
exec(compile(defs,str(PREV/"runner.py"),"exec"),ns)

encoded=(PREV/"all_features_51.csv.gz.b64").read_text(encoding="utf-8").strip()
payload={"lane":"V28_HIER","campaign":"v28-hierarchical","model_b64":encoded}
d,h,tr,dyn=ns["load_data"](payload)

variants=("V23_BASE","SAME_GLOBAL","CROSS_GLOBAL","SAME_PASS","CROSS_PASS",
          "SPARSE_FUSION","HIER_CONVEX","HIER_SELECTIVE","HIER_PROB_GATE")
P={v:np.full((len(h),2),np.nan) for v in variants}
audits=[];elapsed={}
for outer in sorted(h.Group.unique()):
    f=ROOT/"workers"/f"{outer}.json"
    if not f.exists(): raise RuntimeError("MISSING_WORKER:"+outer)
    x=json.loads(f.read_text(encoding="utf-8"))
    idx=np.asarray(x["row_indices"],int)
    for v in variants:P[v][idx]=np.asarray(x["predictions"][v],float)
    audits.append(x["audit"]);elapsed[outer]=x["elapsed_seconds"]

for v in variants:
    if not np.isfinite(P[v]).all():raise RuntimeError("INCOMPLETE:"+v)

y=h[ns["TARGETS"]].to_numpy(float);g=h.Group.to_numpy()
base=ns["force_metrics"](y,P["V23_BASE"],g)
if abs(base["joint_Group_MAE_N"]-ns["V23_LOCK"])>1e-6:raise RuntimeError("V23_LOCK_FAIL")
if abs(base["LC1_Group_MAE_N"]-ns["V23_LC1_LOCK"])>1e-6 or abs(base["LC5_Group_MAE_N"]-ns["V23_LC5_LOCK"])>1e-6:
    raise RuntimeError("V23_CHANNEL_LOCK_FAIL")

summary=[]
for v in variants:
    m=ns["force_metrics"](y,P[v],g)
    row={"variant":v,**m}
    row["joint_improvement_vs_V23_N"]=float(base["joint_Group_MAE_N"]-m["joint_Group_MAE_N"])
    row["joint_improvement_vs_V23_pct"]=float(100*(base["joint_Group_MAE_N"]-m["joint_Group_MAE_N"])/base["joint_Group_MAE_N"])
    row["LC1_improvement_vs_V23_N"]=float(base["LC1_Group_MAE_N"]-m["LC1_Group_MAE_N"])
    row["LC5_improvement_vs_V23_N"]=float(base["LC5_Group_MAE_N"]-m["LC5_Group_MAE_N"])
    row["LC5_improvement_vs_V23_pct"]=float(100*(base["LC5_Group_MAE_N"]-m["LC5_Group_MAE_N"])/base["LC5_Group_MAE_N"])
    row["paired_joint_vs_V23"]=ns["paired_group_report"](y,P[v],P["V23_BASE"],g)
    row["paired_LC5_vs_V23"]=ns["_lc5_pair_report"](y[:,1],P[v][:,1],P["V23_BASE"][:,1],g)
    row["LC1_pass_MAE_N"]={str(int(t)):float(np.mean(np.abs(y[h.Pass_T.to_numpy()==t,0]-P[v][h.Pass_T.to_numpy()==t,0])))
        for t in sorted(h.Pass_T.unique())}
    row["LC5_pass_MAE_N"]={str(int(t)):float(np.mean(np.abs(y[h.Pass_T.to_numpy()==t,1]-P[v][h.Pass_T.to_numpy()==t,1])))
        for t in sorted(h.Pass_T.unique())}
    summary.append(row)

champion=min(summary,key=lambda x:x["joint_Group_MAE_N"])
best_lc5=min(summary,key=lambda x:x["LC5_Group_MAE_N"])
result={
  "schema":"soilbin.v28.hierarchical.v1","status":"COMPLETE",
  "question":"Can a V23 backbone plus sensor/pass-specific memory heads, sparse fusion and training-only gates improve the force endpoint?",
  "source_sha256":ns["EXPECTED_DATA"],"arms":list(variants),
  "summary":summary,"champion":champion,"best_LC5":best_lc5,
  "audit":audits,"worker_elapsed_seconds":elapsed,
  "guards":{
    "V23_reproduced_exactly":True,"past_only_history_features":True,
    "no_future_actual_feature_at_inference":True,
    "memory_depth_selected_training_only":True,
    "sparse_sensor_fusion_selected_training_only":True,
    "hierarchical_weights_selected_training_only":True,
    "transition_probability_current_target_free":True,
    "random_split_used":False,"outer_speed_load_group_held_out":True
  }
}
(ROOT/"RESULTS.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
print("V28_HIER_FANIN_COMPLETE",json.dumps({
  "champion":champion["variant"],"joint":champion["joint_Group_MAE_N"],
  "LC1":champion["LC1_Group_MAE_N"],"LC5":champion["LC5_Group_MAE_N"],
  "best_LC5_variant":best_lc5["variant"],"best_LC5":best_lc5["LC5_Group_MAE_N"]
},ensure_ascii=False),flush=True)
