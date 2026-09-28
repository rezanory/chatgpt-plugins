# SoilBin V16 — functional pass-dynamics challenge
# Post-lock exploratory. No function family is privileged a priori.
from __future__ import annotations

import base64, gzip, hashlib, itertools, json, math, os, random, shutil, sys
from collections import Counter
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

SEED = 20260928
random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
np.random.seed(SEED)
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import binomtest

ROOT = Path(os.environ.get("SOILBIN_OUTPUT_ROOT", "/kaggle/working/SOILBIN_V16"))
TABLE = ROOT / "tables"
STATE = ROOT / "state"
for p in (ROOT, TABLE, STATE):
    p.mkdir(parents=True, exist_ok=True)

EXPECTED_MODEL_SHA = "dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"
EXPECTED_MISSING = {"V1W1T1", "V2W3T2", "V3W2T1"}
TARGETS = ["LC1_peak_magnitude_N_delta", "LC5_peak_magnitude_N_delta"]
TARGET_SHORT = ["LC1", "LC5"]
FROZEN_V5_JOINT_GROUP_MAE_N = 24.28878365273039
EPS = 1e-9

def phase(s):
    print("CGP_PHASE:" + s, flush=True)

def dump(path, obj):
    Path(path).write_text(
        json.dumps(
            obj, ensure_ascii=False, indent=2, allow_nan=False,
            default=lambda x: x.item() if hasattr(x, "item") else str(x),
        ),
        encoding="utf-8",
    )

def decode_model(payload):
    raw = gzip.decompress(base64.b64decode(payload.strip()))
    got = hashlib.sha256(raw).hexdigest()
    if got != EXPECTED_MODEL_SHA:
        raise RuntimeError(f"SOURCE_FINGERPRINT_MISMATCH:{got}")
    return raw.decode("utf-8")

def group_equal_mae(y, p, groups):
    q = pd.DataFrame({"y":np.asarray(y,float), "p":np.asarray(p,float), "g":np.asarray(groups)})
    return float(q.assign(e=lambda d:(d.y-d.p).abs()).groupby("g").e.mean().mean())

phase("DATA_FREEZE")
df = pd.read_csv(StringIO(decode_model(MODEL_CSV_GZ_B64)))
if len(df) != 51 or df.Run_ID.nunique() != 51:
    raise RuntimeError("EXPECTED_51_RUNS")
df["Group"] = df.Run_ID.str.extract(r"^(V\dW\d)")[0]
df["Speed"] = df.Speed_level.astype(float)
df["Load"] = df.Weight_level.astype(float) + 1.0
df["Pass_T"] = df.Pass_T.astype(int)
all_ids = {f"V{v}W{w}T{t}" for v in range(1,4) for w in range(1,4) for t in range(1,7)}
if all_ids - set(df.Run_ID) != EXPECTED_MISSING:
    raise RuntimeError("DESIGN_SLOT_MISMATCH")
for c in TARGETS:
    if c not in df or df[c].isna().any():
        raise RuntimeError("INVALID_TARGET:" + c)

# Consecutive transition table.
rows = []
for g,z in df.groupby("Group"):
    z = z.sort_values("Pass_T")
    by = {int(r.Pass_T):r for _,r in z.iterrows()}
    for t in range(2,7):
        if t not in by or t-1 not in by:
            continue
        a,b = by[t-1],by[t]
        d1 = float(b[TARGETS[0]] - a[TARGETS[0]])
        d5 = float(b[TARGETS[1]] - a[TARGETS[1]])
        rows.append({
            "Group":g, "TargetPass":t,
            "Speed":float(b.Speed), "Load":float(b.Load),
            "signed_LC1":d1, "signed_LC5":d5,
            "D_LC1":abs(d1), "D_LC5":abs(d5),
            "D_joint":0.5*(abs(d1)+abs(d5)),
        })
trans = pd.DataFrame(rows).sort_values(["Group","TargetPass"]).reset_index(drop=True)
if len(trans) != 41 or trans.Group.nunique() != 9:
    raise RuntimeError(f"EXPECTED_41_TRANSITIONS_GOT_{len(trans)}")

# Observed pass dynamics.
pass_summary = trans.groupby("TargetPass").agg(
    n=("D_joint","size"),
    mean_D_joint_N=("D_joint","mean"),
    median_D_joint_N=("D_joint","median"),
    mean_D_LC1_N=("D_LC1","mean"),
    mean_D_LC5_N=("D_LC5","mean"),
    mean_signed_LC1_N=("signed_LC1","mean"),
    mean_signed_LC5_N=("signed_LC5","mean"),
).reset_index()
pass_summary.to_csv(TABLE/"observed_pass_dynamics_v16.csv", index=False)

# T5 -> T6 rebound evidence, paired within the same nine Speed×Load groups.
piv = trans.pivot(index="Group", columns="TargetPass", values="D_joint")
if not ({5,6} <= set(piv.columns)) or piv[[5,6]].isna().any().any():
    raise RuntimeError("T5_T6_PAIRED_DATA_INCOMPLETE")
rebound = pd.DataFrame({
    "Group":piv.index,
    "D5_N":piv[5].to_numpy(float),
    "D6_N":piv[6].to_numpy(float),
})
rebound["difference_N"] = rebound.D6_N - rebound.D5_N
rebound["ratio_D6_over_D5"] = rebound.D6_N / np.maximum(rebound.D5_N, EPS)
rebound.to_csv(TABLE/"t6_rebound_by_group_v16.csv", index=False)
diff = rebound.difference_N.to_numpy(float)
rng = np.random.default_rng(SEED)
boot = diff[rng.integers(0,len(diff),size=(30000,len(diff)))].mean(axis=1)
perm = (np.asarray(list(itertools.product([-1,1], repeat=len(diff))),float) * diff).mean(axis=1)
obs = float(diff.mean())
t6_evidence = {
    "groups_total":int(len(diff)),
    "groups_D6_gt_D5":int(np.sum(diff>0)),
    "groups_D6_gt_2x_D5":int(np.sum(rebound.ratio_D6_over_D5.to_numpy(float)>2.0)),
    "mean_D5_N":float(rebound.D5_N.mean()),
    "mean_D6_N":float(rebound.D6_N.mean()),
    "mean_increase_N":obs,
    "mean_ratio_D6_over_D5":float(rebound.ratio_D6_over_D5.mean()),
    "median_ratio_D6_over_D5":float(rebound.ratio_D6_over_D5.median()),
    "bootstrap_95CI_mean_increase_N":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
    "exact_signflip_one_sided_p":float(np.mean(perm>=obs-1e-12)),
    "exact_directional_binomial_one_sided_p":float(binomtest(int(np.sum(diff>0)), len(diff), .5, alternative="greater").pvalue),
    "interpretation_guard":"Exploratory candidate regime transition only; not proof of structural collapse, fatigue failure, permanent strain, or shakedown class.",
}

# Build two forecast contracts:
# 1) ANCHOR: after the first observed transition in each group, forecast all later D from that first transition.
# 2) ONE_STEP: forecast D_t from the immediately preceding observed transition D_(t-1).
def build_contract(metric, mode):
    out=[]
    for g,z in trans.groupby("Group"):
        z=z.sort_values("TargetPass").reset_index(drop=True)
        if mode=="ANCHOR":
            anchor=z.iloc[0]
            for j in range(1,len(z)):
                r=z.iloc[j]
                out.append({
                    "Group":g, "Speed":float(r.Speed), "Load":float(r.Load),
                    "BasePass":int(anchor.TargetPass), "TargetPass":int(r.TargetPass),
                    "BaseD":float(anchor[metric]), "D":float(r[metric]),
                })
        elif mode=="ONE_STEP":
            for j in range(1,len(z)):
                a,r=z.iloc[j-1],z.iloc[j]
                if int(r.TargetPass)-int(a.TargetPass) != 1:
                    continue
                out.append({
                    "Group":g, "Speed":float(r.Speed), "Load":float(r.Load),
                    "BasePass":int(a.TargetPass), "TargetPass":int(r.TargetPass),
                    "BaseD":float(a[metric]), "D":float(r[metric]),
                })
        else:
            raise KeyError(mode)
    q=pd.DataFrame(out)
    q["ratio"]=q.D/np.maximum(q.BaseD,EPS)
    return q

# Functional families. Raw-ratio linear/log variants are intentionally included beside
# multiplicative log-link families; no family is assumed correct in advance.
FAMILIES = [
    "PERSISTENCE",
    "LINEAR",
    "LOGARITHMIC",
    "POWER_FREE",
    "EXPONENTIAL_FREE",
    "EXPONENTIAL_DECAY",
    "HYPERBOLIC_DECAY",
    "SATURATING_EXP_DECAY",
    "STRETCHED_EXP_DECAY",
    "QUADRATIC_LOG",
    "PIECEWISE_LOG",
    "DECAY_REBOUND",
]
BREAK_FAMILIES={"PIECEWISE_LOG","DECAY_REBOUND"}
MONOTONE_DECAY={"EXPONENTIAL_DECAY","HYPERBOLIC_DECAY","SATURATING_EXP_DECAY","STRETCHED_EXP_DECAY"}

def spec(fam):
    if fam=="PERSISTENCE": return np.array([]),np.array([]),np.array([])
    if fam=="LINEAR": return np.array([-0.1]),np.array([-0.24]),np.array([5.0])
    if fam=="LOGARITHMIC": return np.array([-0.2]),np.array([-0.85]),np.array([5.0])
    if fam=="POWER_FREE": return np.array([-0.5]),np.array([-10.0]),np.array([10.0])
    if fam=="EXPONENTIAL_FREE": return np.array([-0.3]),np.array([-5.0]),np.array([5.0])
    if fam=="EXPONENTIAL_DECAY": return np.array([-0.3]),np.array([-5.0]),np.array([0.0])
    if fam=="HYPERBOLIC_DECAY": return np.array([0.5]),np.array([0.0]),np.array([100.0])
    if fam=="SATURATING_EXP_DECAY": return np.array([0.2,0.5]),np.array([0.001,0.0]),np.array([0.999,20.0])
    if fam=="STRETCHED_EXP_DECAY": return np.array([0.2,0.5,1.0]),np.array([0.001,0.0,0.20]),np.array([0.999,20.0,3.0])
    if fam=="QUADRATIC_LOG": return np.array([-0.2,0.02]),np.array([-5.0,-2.0]),np.array([5.0,2.0])
    if fam=="PIECEWISE_LOG": return np.array([-0.3,0.1]),np.array([-5.0,-5.0]),np.array([5.0,5.0])
    if fam=="DECAY_REBOUND": return np.array([-0.3,0.1]),np.array([-5.0,0.0]),np.array([0.0,5.0])
    raise KeyError(fam)

def ratio_predict(fam, params, t, a, bp=None):
    t=np.asarray(t,float); a=np.asarray(a,float)
    if fam=="PERSISTENCE":
        return np.ones_like(t)
    if fam=="LINEAR":
        return np.maximum(1.0+params[0]*(t-a),1e-6)
    if fam=="LOGARITHMIC":
        return np.maximum(1.0+params[0]*(np.log(t)-np.log(a)),1e-6)
    if fam=="POWER_FREE":
        return np.exp(params[0]*(np.log(t)-np.log(a)))
    if fam in ("EXPONENTIAL_FREE","EXPONENTIAL_DECAY"):
        return np.exp(params[0]*(t-a))
    if fam=="HYPERBOLIC_DECAY":
        k=params[0]
        return (1.0+k*(a-1.0))/np.maximum(1.0+k*(t-1.0),1e-12)
    if fam=="SATURATING_EXP_DECAY":
        q,k=params
        ft=q+(1-q)*np.exp(-k*(t-1))
        fa=q+(1-q)*np.exp(-k*(a-1))
        return ft/np.maximum(fa,1e-12)
    if fam=="STRETCHED_EXP_DECAY":
        q,k,beta=params
        ft=q+(1-q)*np.exp(-np.power(k*np.maximum(t-1,0),beta))
        fa=q+(1-q)*np.exp(-np.power(k*np.maximum(a-1,0),beta))
        return ft/np.maximum(fa,1e-12)
    if fam=="QUADRATIC_LOG":
        b1,b2=params
        return np.exp(b1*(t-a)+b2*(t*t-a*a))
    if fam in BREAK_FAMILIES:
        if bp is None: raise ValueError("BREAKPOINT_REQUIRED")
        b1,b2=params
        before=np.minimum(t,bp)-np.minimum(a,bp)
        after=np.maximum(0,t-bp)-np.maximum(0,a-bp)
        return np.exp(b1*before+b2*after)
    raise KeyError(fam)

def group_weights(groups):
    g=np.asarray(groups)
    vals,counts=np.unique(g,return_counts=True)
    cmap=dict(zip(vals,counts))
    w=np.array([1.0/cmap[x] for x in g],float)
    return w/w.mean()

def fit_family(data, fam, bp=None):
    x0,lo,hi=spec(fam)
    if fam=="PERSISTENCE":
        return x0
    t=data.TargetPass.to_numpy(float); a=data.BasePass.to_numpy(float)
    y=np.maximum(data.ratio.to_numpy(float),1e-8)
    w=np.sqrt(group_weights(data.Group.to_numpy()))
    def residual(p):
        pred=np.maximum(ratio_predict(fam,p,t,a,bp),1e-8)
        return w*(np.log(pred)-np.log(y))
    opt=least_squares(residual,x0,bounds=(lo,hi),loss="soft_l1",f_scale=.35,max_nfev=5000)
    if not np.isfinite(opt.cost):
        raise RuntimeError("NONFINITE_FIT:"+fam)
    return opt.x

def predict_D(data,fam,params,bp=None):
    rr=ratio_predict(fam,params,data.TargetPass.to_numpy(float),data.BasePass.to_numpy(float),bp)
    return data.BaseD.to_numpy(float)*rr

def select_breakpoint(train,fam):
    # Breakpoint is treated as a discrete model parameter and is selected only
    # on the current training partition. The held-out group is never consulted.
    best=None
    for bp in (3,4,5):
        p=fit_family(train,fam,bp)
        ph=predict_D(train,fam,p,bp)
        sc=group_equal_mae(train.D.to_numpy(float),ph,train.Group.to_numpy())
        row=(sc,bp)
        if best is None or row<best: best=row
    if best is None: raise RuntimeError("NO_BREAKPOINT:"+fam)
    return int(best[1])

def outer_family_predictions(data,fam):
    out=[]; bps=[]
    for g in sorted(data.Group.unique()):
        tr=data[data.Group!=g]; te=data[data.Group==g]
        bp=select_breakpoint(tr,fam) if fam in BREAK_FAMILIES else None
        p=fit_family(tr,fam,bp)
        ph=predict_D(te,fam,p,bp)
        for (_,r),pred in zip(te.iterrows(),ph):
            out.append({
                "Group":r.Group,"BasePass":int(r.BasePass),"TargetPass":int(r.TargetPass),
                "y_D_N":float(r.D),"p_D_N":float(pred),"family":fam,
                "breakpoint":bp,
            })
        if bp is not None: bps.append({"outer_group":g,"family":fam,"breakpoint":bp})
    return pd.DataFrame(out),bps

def family_cv_score(train,fam):
    preds=[]
    for vg in sorted(train.Group.unique()):
        tr=train[train.Group!=vg]; va=train[train.Group==vg]
        bp=select_breakpoint(tr,fam) if fam in BREAK_FAMILIES and tr.Group.nunique()>=3 else (4 if fam in BREAK_FAMILIES else None)
        p=fit_family(tr,fam,bp)
        ph=predict_D(va,fam,p,bp)
        preds.append(pd.DataFrame({"y":va.D.to_numpy(float),"p":ph,"g":va.Group.to_numpy()}))
    q=pd.concat(preds,ignore_index=True)
    return group_equal_mae(q.y,q.p,q.g)

def nested_selected_predictions(data):
    out=[]; sel=[]
    for g in sorted(data.Group.unique()):
        tr=data[data.Group!=g]; te=data[data.Group==g]
        scores=[]
        for fam in FAMILIES:
            sc=family_cv_score(tr,fam)
            scores.append((sc,fam))
        scores.sort()
        _,fam=scores[0]
        bp=select_breakpoint(tr,fam) if fam in BREAK_FAMILIES else None
        p=fit_family(tr,fam,bp)
        ph=predict_D(te,fam,p,bp)
        sel.append({"outer_group":g,"selected_family":fam,"selected_breakpoint":bp,"inner_Group_MAE_N":float(scores[0][0])})
        for (_,r),pred in zip(te.iterrows(),ph):
            out.append({"Group":r.Group,"BasePass":int(r.BasePass),"TargetPass":int(r.TargetPass),
                        "y_D_N":float(r.D),"p_D_N":float(pred),"family":"NESTED_SELECTED",
                        "selected_family":fam,"breakpoint":bp})
    return pd.DataFrame(out),pd.DataFrame(sel)

phase("FUNCTIONAL_FAMILY_EVALUATION")
all_summary=[]; all_pred=[]; all_break=[]; selection_tables=[]
# Full nested family selection is reserved for the primary contract only.
# Secondary contracts are fixed-family sensitivity analyses.
contracts=[
    ("ONE_STEP","D_joint"),
    ("ANCHOR","D_joint"),
    ("ONE_STEP","D_LC1"),
    ("ONE_STEP","D_LC5"),
]
for mode,metric in contracts:
    data=build_contract(metric,mode)
    if data.Group.nunique()!=9:
        raise RuntimeError(f"CONTRACT_GROUP_LOSS:{mode}:{metric}:{data.Group.nunique()}")
    for fam in FAMILIES:
        pred,bps=outer_family_predictions(data,fam)
        sc=group_equal_mae(pred.y_D_N,pred.p_D_N,pred.Group)
        all_summary.append({"mode":mode,"metric":metric,"family":fam,"joint_Group_MAE_N":sc,
                            "n":int(len(pred)),"n_groups":int(pred.Group.nunique())})
        pred["mode"]=mode; pred["metric"]=metric
        all_pred.append(pred)
        for x in bps: all_break.append({"mode":mode,"metric":metric,**x})
    if mode=="ONE_STEP" and metric=="D_joint":
        nsp,sel=nested_selected_predictions(data)
        sc=group_equal_mae(nsp.y_D_N,nsp.p_D_N,nsp.Group)
        all_summary.append({"mode":mode,"metric":metric,"family":"NESTED_SELECTED","joint_Group_MAE_N":sc,
                            "n":int(len(nsp)),"n_groups":int(nsp.Group.nunique())})
        nsp["mode"]=mode; nsp["metric"]=metric
        all_pred.append(nsp)
        sel["mode"]=mode; sel["metric"]=metric
        selection_tables.append(sel)
    dump(STATE/"progress_v16.json",{"completed_mode":mode,"completed_metric":metric})

summary=pd.DataFrame(all_summary).sort_values(["mode","metric","joint_Group_MAE_N"]).reset_index(drop=True)
pred=pd.concat(all_pred,ignore_index=True)
breaks=pd.DataFrame(all_break)
selections=pd.concat(selection_tables,ignore_index=True)
summary.to_csv(TABLE/"functional_family_summary_v16.csv",index=False)
pred.to_csv(TABLE/"functional_family_oof_predictions_v16.csv",index=False)
breaks.to_csv(TABLE/"breakpoint_selections_v16.csv",index=False)
selections.to_csv(TABLE/"nested_family_selections_v16.csv",index=False)

# Pass-specific predictive error for the primary ONE_STEP / D_joint contract.
primary_pred=pred[(pred["mode"]=="ONE_STEP")&(pred["metric"]=="D_joint")].copy()
pass_err=[]
for fam,z in primary_pred.groupby("family"):
    for t,q in z.groupby("TargetPass"):
        pass_err.append({"family":fam,"TargetPass":int(t),"n":int(len(q)),
                         "MAE_N":float(np.mean(np.abs(q.y_D_N-q.p_D_N)))})
pass_err=pd.DataFrame(pass_err).sort_values(["family","TargetPass"])
pass_err.to_csv(TABLE/"one_step_pass_error_v16.csv",index=False)

fixed=summary[(summary["mode"]=="ONE_STEP")&(summary["metric"]=="D_joint")&(summary["family"]!="NESTED_SELECTED")].sort_values("joint_Group_MAE_N")
best_fixed=fixed.iloc[0]
nested=summary[(summary["mode"]=="ONE_STEP")&(summary["metric"]=="D_joint")&(summary["family"]=="NESTED_SELECTED")].iloc[0]
mono=fixed[fixed.family.isin(MONOTONE_DECAY)].sort_values("joint_Group_MAE_N").iloc[0]
transition_family=fixed[fixed.family=="DECAY_REBOUND"].iloc[0]
piece_family=fixed[fixed.family=="PIECEWISE_LOG"].iloc[0]

bp_primary=breaks[(breaks["mode"]=="ONE_STEP")&(breaks["metric"]=="D_joint")]
bp_counts={}
for fam,z in bp_primary.groupby("family"):
    bp_counts[fam]={str(int(k)):int(v) for k,v in Counter(z.breakpoint).items()}

sel_primary=selections[(selections["mode"]=="ONE_STEP")&(selections["metric"]=="D_joint")]
selection_counts={k:int(v) for k,v in Counter(sel_primary.selected_family).items()}

# Descriptive winner table by scope; selecting the minimum after seeing all OOF results is descriptive only.
decision={
    "primary_contract":"ONE_STEP conditional transition-magnitude forecast using only the previous observed transition magnitude.",
    "best_fixed_family_descriptive":{
        "family":str(best_fixed.family),
        "Group_MAE_N":float(best_fixed.joint_Group_MAE_N),
    },
    "nested_selected_family":{
        "Group_MAE_N":float(nested.joint_Group_MAE_N),
        "selection_counts":selection_counts,
    },
    "best_monotone_decay_family":{
        "family":str(mono.family),
        "Group_MAE_N":float(mono.joint_Group_MAE_N),
    },
    "decay_rebound":{
        "Group_MAE_N":float(transition_family.joint_Group_MAE_N),
        "delta_vs_best_monotone_N":float(transition_family.joint_Group_MAE_N-mono.joint_Group_MAE_N),
        "outer_breakpoint_counts":bp_counts.get("DECAY_REBOUND",{}),
    },
    "piecewise_log":{
        "Group_MAE_N":float(piece_family.joint_Group_MAE_N),
        "outer_breakpoint_counts":bp_counts.get("PIECEWISE_LOG",{}),
    },
    "t6_rebound":t6_evidence,
}

results={
    "schema":"soilbin.q1.v16.functional-pass-dynamics",
    "created_utc":datetime.now(timezone.utc).isoformat(),
    "post_lock_exploratory":True,
    "does_not_supersede_v5":True,
    "n_valid_runs":51,
    "n_groups":9,
    "n_consecutive_transitions":41,
    "missing_design_slots":sorted(EXPECTED_MISSING),
    "families":FAMILIES,
    "hypothesis":"Repeated-pass change is not assumed exponential. Linear, logarithmic, power, exponential, hyperbolic, saturating, stretched-exponential, quadratic-log, piecewise and decay-to-rebound laws compete under group-held-out validation.",
    "observed_pass_dynamics":pass_summary.to_dict(orient="records"),
    "t6_candidate_transition_evidence":t6_evidence,
    "primary_decision":decision,
    "functional_family_summary":summary.to_dict(orient="records"),
    "guards":{
        "random_split_used":False,
        "outer_group_held_out":True,
        "family_selection_for_nested_selected_uses_outer_training_only":True,
        "breakpoint_selection_uses_outer_training_only":True,
        "current_transition_target_used_to_predict_itself":False,
        "external_labels_used_for_selection":False,
        "collapse_confirmed":False,
        "fatigue_failure_confirmed":False,
        "shakedown_class_confirmed":False,
        "interpretation":"T6 may be called a candidate regime transition only unless independent deformation/strain or extended-pass evidence supports a mechanical failure classification.",
    },
    "v5_reference":{"joint_Group_MAE_N":FROZEN_V5_JOINT_GROUP_MAE_N,
                    "comparison_scope":"Historical force-prediction reference only; V16 primary metric predicts transition magnitude and is not numerically interchangeable with V5 force MAE."},
}
dump(ROOT/"RESULTS_V16.json",results)
manifest={
    "schema":"soilbin.q1.v16.run-manifest",
    "finished_utc":datetime.now(timezone.utc).isoformat(),
    "seed":SEED,
    "source_sha256":EXPECTED_MODEL_SHA,
    "families":FAMILIES,
    "primary_contract":"ONE_STEP_D_joint",
    "secondary_contract":"ANCHOR_D_joint plus channel sensitivities",
    "output_files":sorted([p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()]),
}
dump(ROOT/"RUN_MANIFEST_V16.json",manifest)
shutil.make_archive(str(ROOT.parent/"SoilBin_Q1_Q2_V16_FUNCTIONAL_PASS_DYNAMICS_20260928"),"zip",ROOT)

phase("COMPLETE")
print("SOILBIN_V16_COMPLETE",json.dumps({
    "best_fixed_family":str(best_fixed.family),
    "best_fixed_Group_MAE_N":float(best_fixed.joint_Group_MAE_N),
    "nested_selected_Group_MAE_N":float(nested.joint_Group_MAE_N),
    "best_monotone_family":str(mono.family),
    "best_monotone_Group_MAE_N":float(mono.joint_Group_MAE_N),
    "decay_rebound_Group_MAE_N":float(transition_family.joint_Group_MAE_N),
    "t6_groups_increased":int(t6_evidence["groups_D6_gt_D5"]),
    "t6_groups_over_2x":int(t6_evidence["groups_D6_gt_2x_D5"]),
    "t6_directional_p":float(t6_evidence["exact_directional_binomial_one_sided_p"]),
},sort_keys=True),flush=True)
