from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
V27=BASE/"soilbin-v27-probabilistic"
runner=(V27/"runner.py").read_text(encoding="utf-8")
defs=runner[:runner.index("\ndef main(payload):\n")]
ns={"__name__":"soilbin_v28_synthesis_defs","__file__":str(V27/"runner.py")}
exec(compile(defs,str(V27/"runner.py"),"exec"),ns)

encoded=(V27/"all_features_51.csv.gz.b64").read_text(encoding="utf-8").strip()
payload={"lane":"V28_SYNTHESIS","campaign":"v28-synthesis","model_b64":encoded}
d,h,tr,dyn=ns["load_data"](payload)
TARGETS=ns["TARGETS"]

HIER=BASE/"soilbin-v28-hierarchical"
STACK=BASE/"soilbin-v28-stacked-force"
EDGE=BASE/"soilbin-v28-forward-reverse-ensemble"
FORCE=BASE/"soilbin-v28-force-reverse-ensemble"
OUT=ROOT/"results"
OUT.mkdir(exist_ok=True)

def gather(folder,variant):
    p=np.full((len(h),2),np.nan)
    for outer in sorted(h.Group.unique()):
        x=json.loads((folder/"workers"/f"{outer}.json").read_text(encoding="utf-8"))
        idx=np.asarray(x["row_indices"],int)
        p[idx]=np.asarray(x["predictions"][variant],float)
    if not np.isfinite(p).all():
        raise RuntimeError("INCOMPLETE:"+variant)
    return p

v23=gather(HIER,"V23_BASE")
hier=gather(HIER,"HIER_SELECTIVE")
stack=gather(STACK,"LC5_T2T6_STACKED")
v23s=gather(STACK,"V23_BASE")
if not np.allclose(v23,v23s,atol=1e-10,rtol=0):
    raise RuntimeError("V23_BASELINE_DISAGREEMENT")

# Post-hoc design headroom only: these are NOT confirmatory candidates.
c1=hier.copy()
hard=np.isin(h.Pass_T.to_numpy(int),[2,6])
c1[hard,1]=stack[hard,1]

c2=v23.copy()
c2[:,0]=hier[:,0]
for t in (3,4):
    mt=h.Pass_T.to_numpy(int)==t
    c2[mt,1]=hier[mt,1]
c2[hard,1]=stack[hard,1]
# T5 stays V23.

candidates={
    "V23_BASE":v23,
    "V28_HIER_SELECTIVE":hier,
    "V28_1_LC5_T2T6_STACKED":stack,
    "POSTHOC_HIER_ALL_PLUS_HARD_STACK":c1,
    "POSTHOC_PASS_SPECIALIZED_HEADROOM":c2,
}

y=h[TARGETS].to_numpy(float)
g=h.Group.to_numpy()
rows=[]
for name,p in candidates.items():
    m=ns["force_metrics"](y,p,g)
    row={"candidate":name,**m}
    row["joint_improvement_vs_V23_N"]=float(ns["V23_LOCK"]-m["joint_Group_MAE_N"])
    row["joint_improvement_vs_V23_pct"]=float(100*(ns["V23_LOCK"]-m["joint_Group_MAE_N"])/ns["V23_LOCK"])
    row["LC1_improvement_vs_V23_N"]=float(ns["V23_LC1_LOCK"]-m["LC1_Group_MAE_N"])
    row["LC5_improvement_vs_V23_N"]=float(ns["V23_LC5_LOCK"]-m["LC5_Group_MAE_N"])
    row["LC5_improvement_vs_V23_pct"]=float(100*(ns["V23_LC5_LOCK"]-m["LC5_Group_MAE_N"])/ns["V23_LC5_LOCK"])
    row["paired_joint"]=ns["paired_group_report"](y,p,v23,g)
    row["paired_LC5"]=ns["_lc5_pair_report"](y[:,1],p[:,1],v23[:,1],g)
    row["LC1_pass_MAE_N"]={str(int(t)):float(np.mean(np.abs(y[h.Pass_T.to_numpy()==t,0]-p[h.Pass_T.to_numpy()==t,0]))) for t in sorted(h.Pass_T.unique())}
    row["LC5_pass_MAE_N"]={str(int(t)):float(np.mean(np.abs(y[h.Pass_T.to_numpy()==t,1]-p[h.Pass_T.to_numpy()==t,1]))) for t in sorted(h.Pass_T.unique())}
    if name in ("V23_BASE","V28_HIER_SELECTIVE"):
        row["status"]="PREDECLARED_GROUP_OOF"
    elif name=="V28_1_LC5_T2T6_STACKED":
        row["status"]="POSTHOC_EXPLORATORY_GROUP_OOF"
    else:
        row["status"]="DESCRIPTIVE_POSTHOC_HEADROOM_ONLY"
    rows.append(row)

scalar_rows=[]
for r in rows:
    scalar_rows.append({
        "candidate":r["candidate"],
        "status":r["status"],
        "joint_Group_MAE_N":r["joint_Group_MAE_N"],
        "LC1_Group_MAE_N":r["LC1_Group_MAE_N"],
        "LC5_Group_MAE_N":r["LC5_Group_MAE_N"],
        "joint_improvement_vs_V23_N":r["joint_improvement_vs_V23_N"],
        "joint_improvement_vs_V23_pct":r["joint_improvement_vs_V23_pct"],
        "LC1_improvement_vs_V23_N":r["LC1_improvement_vs_V23_N"],
        "LC5_improvement_vs_V23_N":r["LC5_improvement_vs_V23_N"],
        "LC5_improvement_vs_V23_pct":r["LC5_improvement_vs_V23_pct"],
        "groups_improved_joint":r["paired_joint"]["groups_improved"],
        "joint_signflip_p":r["paired_joint"]["exact_signflip_one_sided_p"],
        "joint_bootstrap_lo_N":r["paired_joint"]["bootstrap_95CI_N"][0],
        "joint_bootstrap_hi_N":r["paired_joint"]["bootstrap_95CI_N"][1],
        "groups_improved_LC5":r["paired_LC5"]["groups_improved"],
        "LC5_signflip_p":r["paired_LC5"]["exact_signflip_one_sided_p"],
        "LC5_bootstrap_lo_N":r["paired_LC5"]["bootstrap_95CI_N"][0],
        "LC5_bootstrap_hi_N":r["paired_LC5"]["bootstrap_95CI_N"][1],
    })
pd.DataFrame(scalar_rows).to_csv(OUT/"01_candidate_comparison.csv",index=False)

edge=json.loads((EDGE/"results"/"RESULTS.json").read_text(encoding="utf-8"))
force=json.loads((FORCE/"RESULTS.json").read_text(encoding="utf-8"))
hierres=json.loads((HIER/"RESULTS.json").read_text(encoding="utf-8"))
stackres=json.loads((STACK/"RESULTS.json").read_text(encoding="utf-8"))

predeclared_best=min([r for r in rows if r["status"]=="PREDECLARED_GROUP_OOF"],key=lambda x:x["joint_Group_MAE_N"])
strongest_observed=min([r for r in rows if r["status"] in ("PREDECLARED_GROUP_OOF","POSTHOC_EXPLORATORY_GROUP_OOF")],key=lambda x:x["joint_Group_MAE_N"])
headroom=min([r for r in rows if r["status"].startswith("DESCRIPTIVE")],key=lambda x:x["joint_Group_MAE_N"])

summary={
    "schema":"soilbin.v28.synthesis.v1",
    "status":"COMPLETE",
    "data_sha256":ns["EXPECTED_DATA"],
    "predeclared_best":predeclared_best,
    "strongest_observed_group_oof":strongest_observed,
    "posthoc_headroom_best":headroom,
    "edge_stacked_signal":{
        "all60":edge["overall_best"],
        "LC5_T2_T6":edge["hard_LC5_T2_T6_best"],
    },
    "simple_reverse_force_champion":force["champion"],
    "hierarchical_champion":hierres["champion"],
    "stacked_force_champion":stackres["champion"],
    "scientific_status":{
        "V28_HIER_SELECTIVE":"predeclared V28 development arm; promising, not statistically closed",
        "V28_1_LC5_T2T6_STACKED":"post-hoc exploratory after edge-map signal; strongest observed development arm; not independent validation",
        "POSTHOC_PASS_SPECIALIZED_HEADROOM":"descriptive architecture headroom only; invalid as a confirmatory performance claim",
    },
    "next_predeclared_candidate":{
        "name":"V29_HIERARCHICAL_WITH_HARD_ZONE_REVERSE_STACK",
        "LC1":"V28 hierarchical-selective style head",
        "LC5_T2":"OOF stacked Forward plus legal reverse-inverted candidate",
        "LC5_T3_T4":"hierarchical selective memory head",
        "LC5_T5":"V23/backbone unless training-only gate selects otherwise",
        "LC5_T6":"reverse-stacked challenger behind a training-only gate",
        "selection_rule":"all routing and weights learned only inside outer training; no pass choice from outer-test performance",
    },
}
(OUT/"RESULTS.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")

t2=next(r for r in stackres["summary"] if r["variant"]=="LC5_T2_STACKED")
champ=stackres["champion"]
md=f"""# SoilBin V28 Synthesis

## Strongest valid development results

### V28 hierarchical selective
- Joint Group-MAE: {hierres['champion']['joint_Group_MAE_N']:.6f} N
- LC1: {hierres['champion']['LC1_Group_MAE_N']:.6f} N
- LC5: {hierres['champion']['LC5_Group_MAE_N']:.6f} N
- Improvement vs V23 Joint: {hierres['champion']['joint_improvement_vs_V23_N']:.6f} N
- Status: grouped OOF development result, not statistically closed.

### V28.1 hard-zone reverse stack
- Arm: {champ['variant']}
- Joint Group-MAE: {champ['joint_Group_MAE_N']:.6f} N
- LC1: {champ['LC1_Group_MAE_N']:.6f} N
- LC5: {champ['LC5_Group_MAE_N']:.6f} N
- Joint improvement vs V23: {champ['joint_improvement_vs_V23_N']:.6f} N ({champ['joint_improvement_vs_V23_pct']:.3f}%)
- LC5 improvement vs V23: {champ['LC5_improvement_vs_V23_N']:.6f} N ({champ['LC5_improvement_vs_V23_pct']:.3f}%)
- Status: post-hoc exploratory because the stacked hard-zone design followed the edge-map ensemble signal.

## Direct answer to Forward plus Reverse ensemble hypothesis

Simple reverse-only, equal average, global convex, and generic gated ensembles did not improve the endpoint. However, an OOF-trained stacked ensemble showed a positive edge-level signal and a targeted LC5 hard-zone version improved the main endpoint.

Edge map:
- STACKED improved {edge['overall_best']['edges_improved']}/60 true Forward edges.
- Mean edge gain: {edge['overall_best']['mean_gain_N']:.3f} N.
- LC5 T2/T6 subset mean gain: {edge['hard_LC5_T2_T6_best']['mean_gain_N']:.3f} N.

Force endpoint:
- T2-only stack: Joint {t2['joint_Group_MAE_N']:.6f} N.
- T2/T6 stack: Joint {champ['joint_Group_MAE_N']:.6f} N.
- The major gain comes from LC5-T2; T6 adds only a small additional improvement.

## Statistical caution

The strongest V28.1 result improved {champ['paired_joint_vs_V23']['groups_improved']}/9 groups; its exact one-sided sign-flip p-value is {champ['paired_joint_vs_V23']['exact_signflip_one_sided_p']:.6f}. The bootstrap improvement CI crosses zero: {champ['paired_joint_vs_V23']['bootstrap_95CI_N']}. Therefore it is a strong development hypothesis, not statistical closure.

## Post-hoc architecture headroom

The file 01_candidate_comparison.csv includes two explicitly post-hoc composites. They are provided only to estimate architectural headroom and must not be cited as validated performance.

Best descriptive headroom candidate:
- {headroom['candidate']}
- Joint: {headroom['joint_Group_MAE_N']:.6f} N
- LC1: {headroom['LC1_Group_MAE_N']:.6f} N
- LC5: {headroom['LC5_Group_MAE_N']:.6f} N

## Next predeclared experiment

Use V29_HIERARCHICAL_WITH_HARD_ZONE_REVERSE_STACK and freeze all routing before outer-test evaluation:
- LC1: hierarchical selective head.
- LC5-T2: Forward plus legal reverse stack.
- LC5-T3/T4: hierarchical memory head.
- LC5-T5: V23 unless a training-only gate selects a challenger.
- LC5-T6: reverse stack only behind a training-only gate.
"""
(OUT/"02_scientific_synthesis.md").write_text(md,encoding="utf-8")

def sh(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
inventory=[]
for p in sorted(OUT.glob("*")):
    if p.is_file():
        inventory.append({"path":p.name,"bytes":p.stat().st_size,"sha256":sh(p)})
pd.DataFrame(inventory).to_csv(OUT/"03_inventory.csv",index=False)
print(json.dumps({
    "predeclared_best":predeclared_best["candidate"],
    "predeclared_joint":predeclared_best["joint_Group_MAE_N"],
    "predeclared_LC5":predeclared_best["LC5_Group_MAE_N"],
    "strongest_observed":strongest_observed["candidate"],
    "strongest_observed_joint":strongest_observed["joint_Group_MAE_N"],
    "strongest_observed_LC5":strongest_observed["LC5_Group_MAE_N"],
    "headroom":headroom["candidate"],
    "headroom_joint":headroom["joint_Group_MAE_N"],
    "headroom_LC5":headroom["LC5_Group_MAE_N"],
},ensure_ascii=False))
