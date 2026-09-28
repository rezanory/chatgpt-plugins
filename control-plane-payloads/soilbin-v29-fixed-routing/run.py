from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
HIER=BASE/"soilbin-v28-hierarchical"
STACK=BASE/"soilbin-v28-stacked-force"
V27=BASE/"soilbin-v27-probabilistic"

runner=(V27/"runner.py").read_text(encoding="utf-8")
defs=runner[:runner.index("\ndef main(payload):\n")]
ns={"__name__":"soilbin_v29_fixed_defs","__file__":str(V27/"runner.py")}
exec(compile(defs,str(V27/"runner.py"),"exec"),ns)

encoded=(V27/"all_features_51.csv.gz.b64").read_text(encoding="utf-8").strip()
payload={"lane":"V29_FIXED_ROUTING","campaign":"v29-fixed-routing","model_b64":encoded}
d,h,tr,dyn=ns["load_data"](payload)

N=len(h)
arms=("V23_BASE","V28_HIER_SELECTIVE","V29_FIXED_ROUTING","V29_FIXED_NO_REVERSE","V29_T2_ONLY_REVERSE","V29_T2_T6_UNCONDITIONAL")
P={a:np.full((N,2),np.nan) for a in arms}
audit=[]

for outer in sorted(h.Group.unique()):
    xh=json.loads((HIER/"workers"/f"{outer}.json").read_text(encoding="utf-8"))
    xs=json.loads((STACK/"workers"/f"{outer}.json").read_text(encoding="utf-8"))
    ih=np.asarray(xh["row_indices"],int); is_=np.asarray(xs["row_indices"],int)
    if not np.array_equal(ih,is_): raise RuntimeError("INDEX_MISMATCH:"+outer)
    idx=ih
    pt=np.asarray(xh["pass_T"],int)
    v23=np.asarray(xh["predictions"]["V23_BASE"],float)
    hier=np.asarray(xh["predictions"]["HIER_SELECTIVE"],float)
    stack_both=np.asarray(xs["predictions"]["STACKED_BOTH"],float)

    P["V23_BASE"][idx]=v23
    P["V28_HIER_SELECTIVE"][idx]=hier

    no_rev=v23.copy()
    no_rev[:,0]=hier[:,0]
    for t in (3,4):
        m=pt==t
        no_rev[m,1]=hier[m,1]
    P["V29_FIXED_NO_REVERSE"][idx]=no_rev

    t2=no_rev.copy()
    m2=pt==2
    t2[m2,1]=stack_both[m2,1]
    P["V29_T2_ONLY_REVERSE"][idx]=t2

    uncond=t2.copy()
    m6=pt==6
    uncond[m6,1]=stack_both[m6,1]
    P["V29_T2_T6_UNCONDITIONAL"][idx]=uncond

    fixed=t2.copy()
    gate=bool(xs["audit"]["channels"]["1"]["pass_gate"]["6"]["use_stacked"])
    if gate:
        fixed[m6,1]=stack_both[m6,1]
    P["V29_FIXED_ROUTING"][idx]=fixed

    audit.append({
        "outer_group":outer,
        "t6_training_only_gate_use_stacked":gate,
        "t6_gate_training_metrics":xs["audit"]["channels"]["1"]["pass_gate"]["6"],
        "routing":{
            "LC1":"HIER_SELECTIVE",
            "LC5_T2":"STACKED",
            "LC5_T3":"HIER_SELECTIVE",
            "LC5_T4":"HIER_SELECTIVE",
            "LC5_T5":"V23",
            "LC5_T6":"STACKED_IF_TRAINING_GATE_ELSE_V23"
        }
    })

for a,p in P.items():
    if not np.isfinite(p).all(): raise RuntimeError("INCOMPLETE:"+a)

y=h[ns["TARGETS"]].to_numpy(float);g=h.Group.to_numpy()
base=ns["force_metrics"](y,P["V23_BASE"],g)
if abs(base["joint_Group_MAE_N"]-ns["V23_LOCK"])>1e-6: raise RuntimeError("V23_LOCK_FAIL")

summary=[]
for a in arms:
    m=ns["force_metrics"](y,P[a],g)
    row={"variant":a,**m}
    row["joint_improvement_vs_V23_N"]=float(base["joint_Group_MAE_N"]-m["joint_Group_MAE_N"])
    row["joint_improvement_vs_V23_pct"]=float(100*(base["joint_Group_MAE_N"]-m["joint_Group_MAE_N"])/base["joint_Group_MAE_N"])
    row["LC1_improvement_vs_V23_N"]=float(base["LC1_Group_MAE_N"]-m["LC1_Group_MAE_N"])
    row["LC5_improvement_vs_V23_N"]=float(base["LC5_Group_MAE_N"]-m["LC5_Group_MAE_N"])
    row["LC5_improvement_vs_V23_pct"]=float(100*(base["LC5_Group_MAE_N"]-m["LC5_Group_MAE_N"])/base["LC5_Group_MAE_N"])
    row["paired_joint_vs_V23"]=ns["paired_group_report"](y,P[a],P["V23_BASE"],g)
    row["paired_LC5_vs_V23"]=ns["_lc5_pair_report"](y[:,1],P[a][:,1],P["V23_BASE"][:,1],g)
    row["LC1_pass_MAE_N"]={str(int(t)):float(np.mean(np.abs(y[h.Pass_T.to_numpy()==t,0]-P[a][h.Pass_T.to_numpy()==t,0]))) for t in sorted(h.Pass_T.unique())}
    row["LC5_pass_MAE_N"]={str(int(t)):float(np.mean(np.abs(y[h.Pass_T.to_numpy()==t,1]-P[a][h.Pass_T.to_numpy()==t,1]))) for t in sorted(h.Pass_T.unique())}
    summary.append(row)

champ=min(summary,key=lambda x:x["joint_Group_MAE_N"])
out={
    "schema":"soilbin.v29.fixed-routing.v1",
    "status":"COMPLETE",
    "scientific_status":"REPEATED_DEVELOPMENT_ON_HISTORICAL_GROUPS_NOT_INDEPENDENT_VALIDATION",
    "data_sha256":ns["EXPECTED_DATA"],
    "protocol_source":"control-plane-payloads/soilbin-v28-synthesis/V29_PREDECLARED_PROTOCOL.md",
    "arms":list(arms),
    "summary":summary,
    "champion":champ,
    "audit":audit,
    "t6_gate_selected_stack_groups":int(sum(x["t6_training_only_gate_use_stacked"] for x in audit)),
    "guards":{
        "outer_speed_load_group_held_out":True,
        "random_split_used":False,
        "outer_test_label_used_for_routing":False,
        "future_actual_target_used_at_inference":False,
        "routing_frozen_before_this_evaluation":True,
        "t6_gate_training_only":True
    }
}
(ROOT/"RESULTS.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in summary]).to_csv(ROOT/"summary.csv",index=False)
print("V29_FIXED_COMPLETE",json.dumps({"champion":champ["variant"],"joint":champ["joint_Group_MAE_N"],"LC1":champ["LC1_Group_MAE_N"],"LC5":champ["LC5_Group_MAE_N"],"t6_gate_stack_groups":out["t6_gate_selected_stack_groups"]},ensure_ascii=False))
