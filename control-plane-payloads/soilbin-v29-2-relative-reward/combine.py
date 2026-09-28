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
N={"__name__":"soilbin_v292_combine","__file__":str(V27/"runner.py")}
exec(compile(defs,str(V27/"runner.py"),"exec"),N)

encoded=(V27/"all_features_51.csv.gz.b64").read_text(encoding="utf-8").strip()
payload={"lane":"V292_REWARD","campaign":"v292-reward","model_b64":encoded}
d,h,tr,dyn=N["load_data"](payload)

variants=("V23_BASE","STATIC_INV_MAE","TRAIN_PAIRWISE_PRIOR","V18_REFERENCE","RELATIVE_RANK",
          "RELATIVE_PAIRWISE","RELATIVE_HIST_PERCENTILE","RELATIVE_TREND","RELATIVE_RANK_TREND","RELATIVE_PAIRWISE_HIST")
P={v:np.full((len(h),2),np.nan) for v in variants}
audits=[];elapsed={}
for outer in sorted(h.Group.unique()):
    f=ROOT/"workers"/f"{outer}.json"
    if not f.exists():raise RuntimeError("MISSING_WORKER:"+outer)
    x=json.loads(f.read_text(encoding="utf-8"))
    idx=np.asarray(x["row_indices"],int)
    for v in variants:P[v][idx]=np.asarray(x["variant_predictions"][v],float)
    audits.append(x["audit"]);elapsed[outer]=x["elapsed_seconds"]
for v,p in P.items():
    if not np.isfinite(p).all():raise RuntimeError("INCOMPLETE:"+v)

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

champ=min(summary,key=lambda x:x["joint_Group_MAE_N"])
# Compare new relative modes directly against old V18-style reference.
v18=next(r for r in summary if r["variant"]=="V18_REFERENCE")
relative=[r for r in summary if r["variant"].startswith("RELATIVE_")]
best_relative=min(relative,key=lambda x:x["joint_Group_MAE_N"])
out={
    "schema":"soilbin.v29.2.relative-reward-penalty.v1",
    "status":"COMPLETE",
    "scientific_status":"SEQUENTIAL_GROUP_OOF_DEVELOPMENT_NOT_INDEPENDENT_VALIDATION",
    "data_sha256":N["EXPECTED_DATA"],
    "historical_audit":{
        "V18_had_relative_expert_loss_vs_weighted_reference":True,
        "V18_had_rank_or_pairwise_test_result_comparison":False,
        "V18_had_historical_percentile_or_test_trend_comparison":False
    },
    "summary":summary,
    "champion":champ,
    "best_new_relative_mode":best_relative,
    "comparison_best_relative_vs_V18":{
        "Joint_MAE_gain_N":float(v18["joint_Group_MAE_N"]-best_relative["joint_Group_MAE_N"]),
        "LC1_MAE_gain_N":float(v18["LC1_Group_MAE_N"]-best_relative["LC1_Group_MAE_N"]),
        "LC5_MAE_gain_N":float(v18["LC5_Group_MAE_N"]-best_relative["LC5_Group_MAE_N"])
    },
    "worker_elapsed_seconds":elapsed,
    "guards":{
        "current_test_result_affects_current_prediction":False,
        "observed_test_result_may_affect_only_later_passes_same_group":True,
        "outer_test_used_to_tune_eta_or_base_weights":False,
        "outer_training_OOF_used_for_eta_and_priors":True,
        "future_target_leakage":False,
        "random_split_used":False,
        "outer_speed_load_group_held_out":True
    }
}
(ROOT/"RESULTS.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in summary]).to_csv(ROOT/"summary.csv",index=False)
print("V292_REWARD_FANIN_COMPLETE",json.dumps({
    "champion":champ["variant"],"joint":champ["joint_Group_MAE_N"],"LC1":champ["LC1_Group_MAE_N"],"LC5":champ["LC5_Group_MAE_N"],
    "best_relative":best_relative["variant"],"best_relative_joint":best_relative["joint_Group_MAE_N"],
    "gain_vs_V18":out["comparison_best_relative_vs_V18"]["Joint_MAE_gain_N"]
},ensure_ascii=False),flush=True)
