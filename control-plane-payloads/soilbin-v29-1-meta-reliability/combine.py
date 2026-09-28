from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
V27=BASE/"soilbin-v27-probabilistic"
runner=(V27/"runner.py").read_text(encoding="utf-8")
defs=runner[:runner.index("\ndef main(payload):\n")]
N={"__name__":"soilbin_v291_combine","__file__":str(V27/"runner.py")}
exec(compile(defs,str(V27/"runner.py"),"exec"),N)

encoded=(V27/"all_features_51.csv.gz.b64").read_text(encoding="utf-8").strip()
payload={"lane":"V291_META","campaign":"v291-meta","model_b64":encoded}
d,h,tr,dyn=N["load_data"](payload)

variants=("V23_BASE","HIERARCHICAL_BASE","REVERSE_STACK_BASE","ERROR_ROUTER","WINNER_ROUTER","GAP_ROUTER","META_ROUTER")
experts=("V23","HIERARCHICAL","REVERSE_STACK")
P={v:np.full((len(h),2),np.nan) for v in variants}
PE={e:np.full((len(h),2),np.nan) for e in experts}
audits=[];elapsed={}
for outer in sorted(h.Group.unique()):
    f=ROOT/"workers"/f"{outer}.json"
    if not f.exists(): raise RuntimeError("MISSING_WORKER:"+outer)
    x=json.loads(f.read_text(encoding="utf-8"))
    idx=np.asarray(x["row_indices"],int)
    for v in variants:P[v][idx]=np.asarray(x["variant_predictions"][v],float)
    for e in experts:PE[e][idx]=np.asarray(x["expert_predictions"][e],float)
    audits.append(x["audit"]);elapsed[outer]=x["elapsed_seconds"]

for k,p in P.items():
    if not np.isfinite(p).all():raise RuntimeError("INCOMPLETE_VARIANT:"+k)

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

# Descriptive oracle among the 3 experts; never a deployable result.
oracle=np.zeros_like(y)
oracle_sel=np.zeros_like(y,dtype=int)
for j in range(2):
    stack=np.stack([PE[e][:,j] for e in experts],axis=1)
    err=np.abs(y[:,j,None]-stack)
    sel=np.argmin(err,axis=1)
    oracle[:,j]=stack[np.arange(len(h)),sel]
    oracle_sel[:,j]=sel
oracle_metrics=N["force_metrics"](y,oracle,g)

# Aggregate direct quality of each requested meta-learning subtask.
meta_quality={"LC1":{},"LC5":{}}
selection_totals={k:{"LC1":{e:0 for e in experts},"LC5":{e:0 for e in experts}} for k in ("ERROR_ROUTER","WINNER_ROUTER","GAP_ROUTER","META_ROUTER")}
for j,label in enumerate(("LC1","LC5")):
    eq=[];wa=[];gq=[]
    for a in audits:
        d0=a["channels"][str(j)]
        q=d0["outer_test_diagnostics_not_used_for_training"]
        eq.append(q["error_predictor_error_MAE_N"])
        wa.append(q["winner_classifier_accuracy"])
        gq.append(q["gap_expected_error_MAE_N"])
        for router,counts in d0["selection_counts"].items():
            for e,n in counts.items():selection_totals[router][label][e]+=int(n)
    meta_quality[label]={
        "Error_Predictor_outer_error_prediction_MAE_N_mean":float(np.mean(eq)),
        "Error_Predictor_outer_error_prediction_MAE_N_by_fold":[float(x) for x in eq],
        "Winner_Classifier_outer_accuracy_mean":float(np.mean(wa)),
        "Winner_Classifier_outer_accuracy_by_fold":[float(x) for x in wa],
        "Gap_Predictor_outer_expected_error_MAE_N_mean":float(np.mean(gq)),
        "Gap_Predictor_outer_expected_error_MAE_N_by_fold":[float(x) for x in gq],
    }

champ=min(summary,key=lambda x:x["joint_Group_MAE_N"])
out={
    "schema":"soilbin.v29.1.generalization-reliability-meta-learning.v1",
    "status":"COMPLETE",
    "scientific_status":"NESTED_GROUP_OOF_DEVELOPMENT_NOT_INDEPENDENT_VALIDATION",
    "data_sha256":N["EXPECTED_DATA"],
    "requested_modules":{
        "1_Error_Predictor":"COMPLETE",
        "2_Winner_Classifier":"COMPLETE",
        "3_Generalization_Gap_Predictor":"COMPLETE",
        "4_Meta_Router":"COMPLETE"
    },
    "summary":summary,
    "champion":champ,
    "meta_task_quality":meta_quality,
    "selection_totals":selection_totals,
    "oracle_three_expert_descriptive_only":{
        **oracle_metrics,
        "joint_headroom_vs_V23_N":float(base["joint_Group_MAE_N"]-oracle_metrics["joint_Group_MAE_N"]),
        "selection_counts":{
            "LC1":{experts[e]:int(np.sum(oracle_sel[:,0]==e)) for e in range(3)},
            "LC5":{experts[e]:int(np.sum(oracle_sel[:,1]==e)) for e in range(3)}
        }
    },
    "worker_elapsed_seconds":elapsed,
    "guards":{
        "outer_test_labels_used_for_meta_training":False,
        "inner_group_cross_fitting":True,
        "current_or_future_target_used_as_meta_feature":False,
        "random_split_used":False,
        "outer_speed_load_group_held_out":True
    }
}
(ROOT/"RESULTS.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in summary]).to_csv(ROOT/"summary.csv",index=False)
print("V291_META_FANIN_COMPLETE",json.dumps({
    "champion":champ["variant"],"joint":champ["joint_Group_MAE_N"],"LC1":champ["LC1_Group_MAE_N"],"LC5":champ["LC5_Group_MAE_N"],
    "error_predictor_LC5_MAE":meta_quality["LC5"]["Error_Predictor_outer_error_prediction_MAE_N_mean"],
    "winner_LC5_acc":meta_quality["LC5"]["Winner_Classifier_outer_accuracy_mean"],
    "gap_LC5_MAE":meta_quality["LC5"]["Gap_Predictor_outer_expected_error_MAE_N_mean"],
    "oracle_joint":oracle_metrics["joint_Group_MAE_N"]
},ensure_ascii=False),flush=True)
