from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
FIX=BASE/"soilbin-v29-fixed-routing"/"RESULTS.json"
META=BASE/"soilbin-v29-1-meta-reliability"/"RESULTS.json"
REWARD=BASE/"soilbin-v29-2-relative-reward"/"RESULTS.json"
OUT=ROOT/"results";OUT.mkdir(exist_ok=True)

fx=json.loads(FIX.read_text(encoding="utf-8"))
mt=json.loads(META.read_text(encoding="utf-8"))
rw=json.loads(REWARD.read_text(encoding="utf-8"))

rows=[]
for lane,obj,status in [
    ("V29_FIXED_ROUTING",fx,"REPEATED_DEVELOPMENT_OOF"),
    ("V29_1_META_RELIABILITY",mt,"NESTED_GROUP_OOF_DEVELOPMENT"),
    ("V29_2_RELATIVE_REWARD",rw,"SEQUENTIAL_GROUP_OOF_DEVELOPMENT")
]:
    for x in obj["summary"]:
        rows.append({
            "lane":lane,"variant":x["variant"],"scientific_status":status,
            "joint_Group_MAE_N":x["joint_Group_MAE_N"],
            "LC1_Group_MAE_N":x["LC1_Group_MAE_N"],
            "LC5_Group_MAE_N":x["LC5_Group_MAE_N"],
            "joint_improvement_vs_V23_N":x.get("joint_improvement_vs_V23_N",0.0),
            "joint_improvement_vs_V23_pct":x.get("joint_improvement_vs_V23_pct",0.0),
            "LC5_improvement_vs_V23_N":x.get("LC5_improvement_vs_V23_N",0.0),
            "LC5_improvement_vs_V23_pct":x.get("LC5_improvement_vs_V23_pct",0.0),
        })
pd.DataFrame(rows).to_csv(OUT/"01_all_variants.csv",index=False)

fixed_pre=next(x for x in fx["summary"] if x["variant"]=="V29_FIXED_ROUTING")
fixed_best=fx["champion"]
meta_mod={x["variant"]:x for x in mt["summary"]}
best_rel=rw["best_new_relative_mode"]
v18=next(x for x in rw["summary"] if x["variant"]=="V18_REFERENCE")

summary={
    "schema":"soilbin.v29.program-synthesis.v1",
    "status":"COMPLETE",
    "program_progress_pct":100.0,
    "data_sha256":fx["data_sha256"],
    "V29_fixed_routing":{
        "predeclared_fixed":fixed_pre,
        "best_observed_arm":fixed_best,
        "scientific_status":"architecture frozen after V28, re-evaluated on same 9 historical groups; not independent validation"
    },
    "V29_1_generalization_reliability":{
        "modules":{
            "Error_Predictor":meta_mod["ERROR_ROUTER"],
            "Winner_Classifier":meta_mod["WINNER_ROUTER"],
            "Generalization_Gap_Predictor":meta_mod["GAP_ROUTER"],
            "Meta_Router":meta_mod["META_ROUTER"]
        },
        "task_quality":mt["meta_task_quality"],
        "oracle_three_expert_descriptive_only":mt["oracle_three_expert_descriptive_only"],
        "conclusion":"All four modules executed; current learned reliability models do not beat V23. Large oracle headroom shows expert selection remains valuable but is not learnable reliably from the current 9-group sample with these low-capacity meta models."
    },
    "V29_2_relative_reward_penalty":{
        "historical_audit":rw["historical_audit"],
        "old_V18_reference":v18,
        "best_new_relative_mode":best_rel,
        "gain_best_relative_vs_V18":rw["comparison_best_relative_vs_V18"],
        "conclusion":"New relative reward/penalty functions were tested leakage-safely; current-pass outcomes affected only later passes."
    },
    "recommended_next_design":{
        "primary_predictor":"V29 fixed-routing family",
        "reliability_layer":"keep as uncertainty/diagnostic layer, not primary router until more independent groups are available",
        "reward_layer":"retain only if V29.2 shows repeatable improvement over V18/static weighting",
        "data_priority":"additional independent Speed×Load groups / repeated runs are more valuable for reliability learning than adding meta-model complexity"
    }
}
(OUT/"RESULTS.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")

md=f"""# SoilBin V29 — Final Synthesis

## V29 fixed routing
- Frozen fixed-routing Joint Group-MAE: {fixed_pre['joint_Group_MAE_N']:.6f} N
- LC1: {fixed_pre['LC1_Group_MAE_N']:.6f} N
- LC5: {fixed_pre['LC5_Group_MAE_N']:.6f} N
- Improvement vs V23: {fixed_pre['joint_improvement_vs_V23_N']:.6f} N ({fixed_pre['joint_improvement_vs_V23_pct']:.3f}%)
- Best observed V29 arm: {fixed_best['variant']} at {fixed_best['joint_Group_MAE_N']:.6f} N.
- This is repeated development evidence on the same historical groups, not independent validation.

## V29.1 — all four requested meta-learning modules
1. Error Predictor: Joint {meta_mod['ERROR_ROUTER']['joint_Group_MAE_N']:.6f} N
2. Winner Classifier: Joint {meta_mod['WINNER_ROUTER']['joint_Group_MAE_N']:.6f} N
3. Generalization-Gap Predictor: Joint {meta_mod['GAP_ROUTER']['joint_Group_MAE_N']:.6f} N
4. Meta-Router: Joint {meta_mod['META_ROUTER']['joint_Group_MAE_N']:.6f} N

None beats V23 at {meta_mod['V23_BASE']['joint_Group_MAE_N']:.6f} N.
The descriptive three-expert oracle is {mt['oracle_three_expert_descriptive_only']['joint_Group_MAE_N']:.6f} N, showing large selection headroom but poor present reliability identification.

## V29.2 — relative reward/penalty
Historical V18 already used expert loss relative to the current weighted reference, but did not explicitly test rank, pairwise, historical-percentile, or trend-based relative functions.

- V18 reference Joint: {v18['joint_Group_MAE_N']:.6f} N
- Best new relative mode: {best_rel['variant']}
- Best new relative Joint: {best_rel['joint_Group_MAE_N']:.6f} N
- Gain vs V18 reference: {rw['comparison_best_relative_vs_V18']['Joint_MAE_gain_N']:.6f} N

## Scientific interpretation
The strongest evidence continues to favor pass/sensor-specific architecture rather than a universal router or universal reward law. Reliability learning has substantial theoretical headroom, but 9 Speed×Load groups are too small for the current sample-specific meta-router to generalize reliably.

No outer-test label was used to train its own prediction, no random row split was used, and sequential reward updates use an observed pass only for later passes.
"""
(OUT/"02_scientific_synthesis.md").write_text(md,encoding="utf-8")

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inventory=[]
for p in [FIX,META,REWARD,OUT/"01_all_variants.csv",OUT/"RESULTS.json",OUT/"02_scientific_synthesis.md"]:
    inventory.append({"path":str(p.relative_to(BASE)) if p.is_relative_to(BASE) else str(p),"bytes":p.stat().st_size,"sha256":sha(p)})
pd.DataFrame(inventory).to_csv(OUT/"03_inventory.csv",index=False)
print("V29_SYNTHESIS_COMPLETE",json.dumps({
    "fixed":fixed_pre["joint_Group_MAE_N"],
    "fixed_best":fixed_best["joint_Group_MAE_N"],
    "meta_router":meta_mod["META_ROUTER"]["joint_Group_MAE_N"],
    "best_relative":best_rel["joint_Group_MAE_N"]
},ensure_ascii=False),flush=True)
