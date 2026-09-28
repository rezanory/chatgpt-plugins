# SoilBin Q1/Q2 V8 — latent soil-state memory + hard-group error diagnostics
# Post-lock exploratory challenge. Frozen V5 remains the scientific reference.
from __future__ import annotations
import os, sys, json, math, hashlib, random, gzip, base64, itertools, shutil, warnings
from pathlib import Path
from io import StringIO
from datetime import datetime, timezone
from collections import Counter

SEED = 20260914
random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
np.random.seed(SEED)
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path("/kaggle/working/SOILBIN_V8")
TABLE = ROOT / "tables"
FIG = ROOT / "figures"
STATE = ROOT / "state"
for p in (ROOT, TABLE, FIG, STATE):
    p.mkdir(parents=True, exist_ok=True)

EXPECTED_MODEL_SHA = "dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"
EXPECTED_MISSING = {"V1W1T1", "V2W3T2", "V3W2T1"}
TARGET_COLS = ["LC1_peak_magnitude_N_delta", "LC5_peak_magnitude_N_delta"]
TARGET_NAMES = ["LC1_proxy_N", "LC5_proxy_N"]
FROZEN_V5_JOINT = 24.28878365273039
FROZEN_V5_LC1 = 12.785420282422
FROZEN_V5_LC5 = 35.79214702303878
RESPONSE_STATUS = "CALIBRATED_VERTICAL_FORCE_PROXY_N_PENDING_SENSOR_AREA_OR_STRESS_CALIBRATION"
DEPTH_STATUS = "PROVISIONAL_LC5_5CM_LC1_15CM_PENDING_LAYOUT_CONFIRMATION"

def phase(x):
    print("CGP_PHASE:"+x, flush=True)

def dump(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                   default=lambda v: v.item() if hasattr(v,"item") else str(v)),
        encoding="utf-8"
    )

def decode_model(payload):
    raw=gzip.decompress(base64.b64decode(payload.strip()))
    got=hashlib.sha256(raw).hexdigest()
    if got != EXPECTED_MODEL_SHA:
        raise RuntimeError(f"SOURCE_FINGERPRINT_MISMATCH:{got}")
    return raw.decode("utf-8")

def group_equal_mae(y,p,groups):
    q=pd.DataFrame({"y":np.asarray(y,float),"p":np.asarray(p,float),"g":np.asarray(groups)})
    return float(q.assign(e=lambda d:(d.y-d.p).abs()).groupby("g").e.mean().mean())

def metrics_block(pred):
    out={"n":int(len(pred)),"n_groups":int(pred.Group_VW.nunique())}
    gm=[]
    for t in TARGET_NAMES:
        y=pred[f"y_{t}"].to_numpy(float); p=pred[f"p_{t}"].to_numpy(float)
        out[f"{t}_Group_MAE"]=group_equal_mae(y,p,pred.Group_VW)
        out[f"{t}_MAE"]=float(mean_absolute_error(y,p))
        out[f"{t}_RMSE"]=float(mean_squared_error(y,p)**0.5)
        out[f"{t}_Bias"]=float(np.mean(p-y))
        out[f"{t}_R2"]=float(r2_score(y,p))
        gm.append(out[f"{t}_Group_MAE"])
    out["joint_Group_MAE"]=float(np.mean(gm))
    return out

def make_tree(name,params):
    if name=="RandomForest":
        return Pipeline([
            ("impute",SimpleImputer(strategy="median")),
            ("m",RandomForestRegressor(
                n_estimators=250,max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=SEED,n_jobs=1,max_features=1.0))
        ])
    if name=="ExtraTrees":
        return Pipeline([
            ("impute",SimpleImputer(strategy="median")),
            ("m",ExtraTreesRegressor(
                n_estimators=250,max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=SEED,n_jobs=1,max_features=1.0))
        ])
    raise KeyError(name)

TREE_CANDIDATES=[
    ("RandomForest",{"max_depth":None,"min_samples_leaf":1}),
    ("RandomForest",{"max_depth":4,"min_samples_leaf":2}),
    ("ExtraTrees",{"max_depth":None,"min_samples_leaf":1}),
    ("ExtraTrees",{"max_depth":4,"min_samples_leaf":2}),
]

phase("DATA_FREEZE")
df=pd.read_csv(StringIO(decode_model(MODEL_CSV_GZ_B64)))
if len(df)!=51 or df.Run_ID.nunique()!=51:
    raise RuntimeError("EXPECTED_51_RUNS")
df["Group_VW"]=df.Run_ID.str.extract(r"^(V\dW\d)")[0]
df["Speed_kmh"]=df.Speed_level.astype(float)
df["Load_kN"]=df.Weight_level.astype(float)+1.0
df["Pass_T"]=df.Pass_T.astype(int)
all_ids={f"V{v}W{w}T{t}" for v in range(1,4) for w in range(1,4) for t in range(1,7)}
if all_ids-set(df.Run_ID)!=EXPECTED_MISSING:
    raise RuntimeError("DESIGN_SLOT_MISMATCH")

# Exact V5 history table.
prev=df.copy(); prev["Pass_T"]=prev["Pass_T"]+1
prev=prev[["Group_VW","Pass_T"]+TARGET_COLS+["QC_status"]].rename(columns={
    TARGET_COLS[0]:"prev_LC1", TARGET_COLS[1]:"prev_LC5", "QC_status":"prev_QC_status"
})
hist=df.merge(prev,on=["Group_VW","Pass_T"],how="left",validate="one_to_one")
hist=hist[hist.prev_LC1.notna() & hist.prev_LC5.notna()].copy().reset_index(drop=True)
if len(hist)!=41:
    raise RuntimeError(f"EXPECTED_41_HISTORY_PAIRS:{len(hist)}")
hist["row_id"]=np.arange(len(hist),dtype=int)
hist["pair_qc_ok"]=(hist.QC_status.fillna("")=="OK")&(hist.prev_QC_status.fillna("")=="OK")
prev_peaks=["prev_LC1","prev_LC5"]

# Latent state descriptors: all use only observed passes before target pass.
seq={g:z.sort_values("Pass_T").copy() for g,z in df.groupby("Group_VW")}
state_rows=[]
for _,r in hist.iterrows():
    past=seq[r.Group_VW]
    past=past[past.Pass_T < int(r.Pass_T)].sort_values("Pass_T")
    V=past[TARGET_COLS].to_numpy(float)
    P=past.Pass_T.to_numpy(float)
    prevv=V[-1]
    first=V[0]
    def ewma(alpha):
        s=V[0].copy()
        for v in V[1:]:
            s=alpha*v+(1-alpha)*s
        return s
    e25,e50,e75=ewma(.25),ewma(.50),ewma(.75)
    if len(V)>=2:
        vel=V[-1]-V[-2]
        path=float(np.sqrt(((V[1:]-V[:-1])**2).sum(axis=1)).sum())
        signed_path=np.abs(V[1:]-V[:-1]).sum(axis=0)
        last_step_gap=float(P[-1]-P[-2])
    else:
        vel=np.zeros(2); path=0.0; signed_path=np.zeros(2); last_step_gap=0.0
    if len(V)>=3:
        vel_prev=V[-2]-V[-3]
        accel=vel-vel_prev
    else:
        accel=np.zeros(2)
    mins=V.min(axis=0); maxs=V.max(axis=0); ranges=np.maximum(maxs-mins,1e-9)
    pos=(prevv-mins)/ranges
    state_rows.append({
        "row_id":int(r.row_id),
        "history_count":float(len(V)),
        "history_span":float(P[-1]-P[0]+1),
        "observed_fraction":float(len(V)/max(P[-1]-P[0]+1,1)),
        "ew25_LC1":float(e25[0]),"ew25_LC5":float(e25[1]),
        "ew50_LC1":float(e50[0]),"ew50_LC5":float(e50[1]),
        "ew75_LC1":float(e75[0]),"ew75_LC5":float(e75[1]),
        "gap25_LC1":float(prevv[0]-e25[0]),"gap25_LC5":float(prevv[1]-e25[1]),
        "gap50_LC1":float(prevv[0]-e50[0]),"gap50_LC5":float(prevv[1]-e50[1]),
        "gap75_LC1":float(prevv[0]-e75[0]),"gap75_LC5":float(prevv[1]-e75[1]),
        "ew_short_long_LC1":float(e75[0]-e25[0]),"ew_short_long_LC5":float(e75[1]-e25[1]),
        "disp_LC1":float(prevv[0]-first[0]),"disp_LC5":float(prevv[1]-first[1]),
        "vel_LC1":float(vel[0]),"vel_LC5":float(vel[1]),
        "accel_LC1":float(accel[0]),"accel_LC5":float(accel[1]),
        "path_euclid":path,
        "path_abs_LC1":float(signed_path[0]),"path_abs_LC5":float(signed_path[1]),
        "position_LC1":float(pos[0]),"position_LC5":float(pos[1]),
        "prev_norm":float(np.sqrt((prevv**2).sum())),
        "prev_angle":float(np.arctan2(prevv[1],prevv[0])),
        "prev_ratio":float(prevv[1]/max(prevv[0],1e-9)),
        "prev_diff":float(prevv[1]-prevv[0]),
        "recent_speed_norm":float(np.sqrt((vel**2).sum())),
        "curvature_norm":float(np.sqrt((accel**2).sum())),
        "last_step_gap":last_step_gap,
    })
hist=hist.merge(pd.DataFrame(state_rows),on="row_id",validate="one_to_one")

BASE=["Load_kN","Speed_kmh","Pass_T","prev_LC1","prev_LC5"]
EWMA50=BASE+["ew50_LC1","ew50_LC5","gap50_LC1","gap50_LC5","history_count","observed_fraction"]
MULTISCALE=BASE+[
    "ew25_LC1","ew25_LC5","ew50_LC1","ew50_LC5","ew75_LC1","ew75_LC5",
    "gap25_LC1","gap25_LC5","gap50_LC1","gap50_LC5","gap75_LC1","gap75_LC5",
    "ew_short_long_LC1","ew_short_long_LC5","history_count","observed_fraction"
]
PATH_STATE=BASE+[
    "disp_LC1","disp_LC5","vel_LC1","vel_LC5","accel_LC1","accel_LC5",
    "path_euclid","path_abs_LC1","path_abs_LC5","position_LC1","position_LC5",
    "recent_speed_norm","curvature_norm","history_count","observed_fraction"
]
COMPACT_STATE=BASE+[
    "ew50_LC1","ew50_LC5","gap50_LC1","gap50_LC5",
    "disp_LC1","disp_LC5","vel_LC1","vel_LC5",
    "path_euclid","prev_norm","prev_angle","prev_ratio","history_count","observed_fraction"
]
FULL_STATE=list(dict.fromkeys(MULTISCALE+PATH_STATE+["prev_norm","prev_angle","prev_ratio","prev_diff"]))

TASKS={
    "D_V5_BASELINE":BASE,
    "STATE_EWMA50":EWMA50,
    "STATE_MULTISCALE":MULTISCALE,
    "STATE_PATH":PATH_STATE,
    "STATE_COMPACT":COMPACT_STATE,
    "STATE_FULL":FULL_STATE,
}

def ydelta(data):
    return data[TARGET_COLS].to_numpy(float)-data[prev_peaks].to_numpy(float)

def inner_select(train, feats, search_rows, outer_label, task):
    groups=train.Group_VW.to_numpy()
    gkf=GroupKFold(n_splits=min(4,len(np.unique(groups))))
    X=train[feats]; yd=ydelta(train)
    best=None
    for ci,(name,params) in enumerate(TREE_CANDIDATES):
        pdlt=np.full((len(train),2),np.nan)
        failed=None
        for tr,va in gkf.split(X,groups=groups):
            try:
                for ti in range(2):
                    est=make_tree(name,params); est.fit(X.iloc[tr],yd[tr,ti])
                    pdlt[va,ti]=est.predict(X.iloc[va])
            except Exception as exc:
                failed=f"{type(exc).__name__}:{str(exc)[:180]}"; break
        if failed or np.isnan(pdlt).any():
            score=float("inf")
        else:
            final=pdlt+train[prev_peaks].to_numpy(float)
            score=float(np.mean([group_equal_mae(train[TARGET_COLS[ti]],final[:,ti],groups) for ti in range(2)]))
        search_rows.append({"outer":str(outer_label),"task":task,"candidate_index":ci,"model":name,
                            "params":json.dumps(params,sort_keys=True),
                            "inner_joint_Group_MAE":score if math.isfinite(score) else None,"failed":failed})
        if math.isfinite(score) and (best is None or score<best[0]-1e-12):
            best=(score,name,params)
    if best is None: raise RuntimeError("NO_VALID_TREE")
    return best

def fit_outer_task(task,feats):
    preds=[]; searches=[]; sels=[]
    for g in pd.unique(hist.Group_VW):
        te=hist[hist.Group_VW==g].copy(); tr=hist[hist.Group_VW!=g].copy()
        sc,name,params=inner_select(tr,feats,searches,g,task)
        yd=ydelta(tr); pdelta=[]
        for ti in range(2):
            est=make_tree(name,params); est.fit(tr[feats],yd[:,ti]); pdelta.append(est.predict(te[feats]))
        final=np.column_stack(pdelta)+te[prev_peaks].to_numpy(float)
        sels.append({"task":task,"outer_group":g,"model":name,"params":json.dumps(params,sort_keys=True),"inner_score":sc})
        for j,(_,r) in enumerate(te.iterrows()):
            row={"row_id":int(r.row_id),"Run_ID":r.Run_ID,"Group_VW":r.Group_VW,
                 "Speed_kmh":float(r.Speed_kmh),"Load_kN":float(r.Load_kN),"Pass_T":int(r.Pass_T),
                 "task":task}
            for ti,t in enumerate(TARGET_NAMES):
                row[f"y_{t}"]=float(r[TARGET_COLS[ti]]); row[f"p_{t}"]=float(final[j,ti])
            preds.append(row)
    p=pd.DataFrame(preds)
    if p.row_id.nunique()!=41: raise RuntimeError(f"INCOMPLETE_OOF:{task}")
    return p,pd.DataFrame(searches),pd.DataFrame(sels)

phase("PRIMARY_STATE_MODELS")
preds=[]; searches=[]; sels=[]; summaries=[]
for i,(task,feats) in enumerate(TASKS.items(),1):
    phase(f"TASK_{i}_{task}")
    p,s,z=fit_outer_task(task,feats)
    preds.append(p); searches.append(s); sels.append(z)
    m=metrics_block(p); m["task"]=task; summaries.append(m)
    dump(STATE/"progress_v8.json",{"completed":list(TASKS)[:i],"total":len(TASKS)})

pred=pd.concat(preds,ignore_index=True)
search=pd.concat(searches,ignore_index=True)
sel=pd.concat(sels,ignore_index=True)
summary=pd.DataFrame(summaries).sort_values("joint_Group_MAE").reset_index(drop=True)
base=summary[summary.task=="D_V5_BASELINE"].iloc[0]
if abs(float(base.joint_Group_MAE)-FROZEN_V5_JOINT)>1e-9:
    raise RuntimeError(f"V5_EXACT_REPRODUCTION_FAILED:{float(base.joint_Group_MAE)}")
print("V5_EXACT_REPRODUCTION_PASS",float(base.joint_Group_MAE),flush=True)

# Training-only residual correction on top of exact V5 representation.
# Inner OOF residuals are created inside each outer-training set, then a fixed Ridge(alpha=10) corrects outer-test predictions.
phase("RESIDUAL_CORRECTION")
RESIDUAL_VARIANTS={
    "RESIDUAL_RIDGE_COMPACT":COMPACT_STATE,
    "RESIDUAL_RIDGE_FULL":FULL_STATE,
}
res_pred=[]; res_rows=[]
for task,feats_corr in RESIDUAL_VARIANTS.items():
    allrows=[]
    for outer in pd.unique(hist.Group_VW):
        te=hist[hist.Group_VW==outer].copy(); tr=hist[hist.Group_VW!=outer].copy()
        # Select exact V5 base family on outer training only.
        dummy=[]
        _,name,params=inner_select(tr,BASE,dummy,outer,"D_V5_BASELINE")
        yd=ydelta(tr)
        # Inner OOF base predictions on outer training.
        groups=tr.Group_VW.to_numpy(); gkf=GroupKFold(n_splits=min(4,tr.Group_VW.nunique()))
        inner_final=np.full((len(tr),2),np.nan)
        for itr,iva in gkf.split(tr[BASE],groups=groups):
            for ti in range(2):
                est=make_tree(name,params); est.fit(tr.iloc[itr][BASE],yd[itr,ti])
                inner_final[iva,ti]=est.predict(tr.iloc[iva][BASE])+tr.iloc[iva][prev_peaks[ti]].to_numpy(float)
        residual=tr[TARGET_COLS].to_numpy(float)-inner_final
        # Fit base on full outer-training and predict outer test.
        base_delta=[]
        for ti in range(2):
            est=make_tree(name,params); est.fit(tr[BASE],yd[:,ti]); base_delta.append(est.predict(te[BASE]))
        base_outer=np.column_stack(base_delta)+te[prev_peaks].to_numpy(float)
        # Fixed linear residual correction, no extra hyperparameter search.
        corr=[]
        for ti in range(2):
            model=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("ridge",Ridge(alpha=10.0))])
            model.fit(tr[feats_corr],residual[:,ti])
            corr.append(model.predict(te[feats_corr]))
        final=base_outer+np.column_stack(corr)
        for j,(_,r) in enumerate(te.iterrows()):
            row={"row_id":int(r.row_id),"Run_ID":r.Run_ID,"Group_VW":r.Group_VW,
                 "Speed_kmh":float(r.Speed_kmh),"Load_kN":float(r.Load_kN),"Pass_T":int(r.Pass_T),"task":task}
            for ti,t in enumerate(TARGET_NAMES):
                row[f"y_{t}"]=float(r[TARGET_COLS[ti]]); row[f"p_{t}"]=float(final[j,ti])
            allrows.append(row)
    rp=pd.DataFrame(allrows)
    if rp.row_id.nunique()!=41: raise RuntimeError(f"INCOMPLETE_RESIDUAL_OOF:{task}")
    res_pred.append(rp)
    m=metrics_block(rp); m["task"]=task; res_rows.append(m)

if res_pred:
    pred=pd.concat([pred]+res_pred,ignore_index=True)
    summary=pd.concat([summary,pd.DataFrame(res_rows)],ignore_index=True).sort_values("joint_Group_MAE").reset_index(drop=True)

# Paired group inference for all challengers.
phase("PAIRED_INFERENCE")
basep=pred[pred.task=="D_V5_BASELINE"].copy()
paired=[]; group_pairs=[]
rng=np.random.default_rng(20260928)
for task in [x for x in summary.task if x!="D_V5_BASELINE"]:
    cp=pred[pred.task==task].copy()
    diffs=[]
    for g in sorted(hist.Group_VW.unique()):
        b=basep[basep.Group_VW==g]; c=cp[cp.Group_VW==g]
        be=float(np.mean([mean_absolute_error(b[f"y_{t}"],b[f"p_{t}"]) for t in TARGET_NAMES]))
        ce=float(np.mean([mean_absolute_error(c[f"y_{t}"],c[f"p_{t}"]) for t in TARGET_NAMES]))
        diffs.append(be-ce)
        group_pairs.append({"task":task,"Group_VW":g,"baseline_MAE":be,"challenger_MAE":ce,"improvement_N":be-ce})
    diffs=np.asarray(diffs,float); obs=float(diffs.mean())
    perm=np.array([np.mean(diffs*np.array(signs)) for signs in itertools.product([-1,1],repeat=9)])
    boot=np.array([np.mean(rng.choice(diffs,size=9,replace=True)) for _ in range(20000)])
    m=summary[summary.task==task].iloc[0]
    paired.append({
        "task":task,"joint_Group_MAE_N":float(m.joint_Group_MAE),
        "relative_improvement_vs_V5_pct":float(100*(FROZEN_V5_JOINT-float(m.joint_Group_MAE))/FROZEN_V5_JOINT),
        "mean_group_paired_improvement_N":obs,
        "bootstrap_95CI_low_N":float(np.quantile(boot,.025)),
        "bootstrap_95CI_high_N":float(np.quantile(boot,.975)),
        "exact_signflip_one_sided_p":float(np.mean(perm>=obs-1e-12)),
        "groups_improved":int(np.sum(diffs>0)),
    })
paired_df=pd.DataFrame(paired).sort_values("joint_Group_MAE_N").reset_index(drop=True)
group_pairs_df=pd.DataFrame(group_pairs)

# V5 hard-group error atlas.
phase("HARD_GROUP_DIAGNOSTICS")
row_base=basep.copy()
for t in TARGET_NAMES:
    row_base[f"ae_{t}"]=(row_base[f"y_{t}"]-row_base[f"p_{t}"]).abs()
    row_base[f"err_{t}"]=row_base[f"p_{t}"]-row_base[f"y_{t}"]
row_base["joint_abs_error"]=row_base[[f"ae_{t}" for t in TARGET_NAMES]].mean(axis=1)
# Attach true transition magnitude and state descriptors.
diag=row_base.merge(
    hist[["row_id","prev_LC1","prev_LC5"]+[c for c in hist.columns if c.startswith(("ew","gap","disp","vel","accel","path_","position_","prev_norm","prev_angle","prev_ratio","recent_speed_norm","curvature_norm","history_count","observed_fraction"))]],
    on="row_id",how="left",validate="one_to_one"
)
diag["true_delta_LC1"]=(diag["y_LC1_proxy_N"]-diag["prev_LC1"]).abs()
diag["true_delta_LC5"]=(diag["y_LC5_proxy_N"]-diag["prev_LC5"]).abs()
diag["true_delta_joint"]=diag[["true_delta_LC1","true_delta_LC5"]].mean(axis=1)

# Training-distance: standardized V5 feature distance from each outer group's training cloud.
dist=[]
for g in pd.unique(hist.Group_VW):
    te=hist[hist.Group_VW==g]; tr=hist[hist.Group_VW!=g]
    mu=tr[BASE].mean().to_numpy(float); sd=tr[BASE].std(ddof=0).replace(0,1).to_numpy(float)
    X=(te[BASE].to_numpy(float)-mu)/sd
    for rid,d in zip(te.row_id,np.sqrt((X**2).sum(axis=1))):
        dist.append({"row_id":int(rid),"train_state_distance":float(d)})
diag=diag.merge(pd.DataFrame(dist),on="row_id",validate="one_to_one")

# Missing-design-slot relation per group.
missing_by_group={g:0 for g in sorted(hist.Group_VW.unique())}
for rid in EXPECTED_MISSING:
    missing_by_group[rid[:4]]=missing_by_group.get(rid[:4],0)+1

groups=[]
for g,z in diag.groupby("Group_VW"):
    cond=hist[hist.Group_VW==g].iloc[0]
    groups.append({
        "Group_VW":g,"Speed_kmh":float(cond.Speed_kmh),"Load_kN":float(cond.Load_kN),
        "n_transitions":int(len(z)),"missing_design_slots":int(missing_by_group.get(g,0)),
        "joint_Group_MAE_N":float(z.joint_abs_error.mean()),
        "LC1_MAE_N":float(z.ae_LC1_proxy_N.mean()),"LC5_MAE_N":float(z.ae_LC5_proxy_N.mean()),
        "bias_LC1_N":float(z.err_LC1_proxy_N.mean()),"bias_LC5_N":float(z.err_LC5_proxy_N.mean()),
        "mean_true_delta_N":float(z.true_delta_joint.mean()),
        "max_true_delta_N":float(z.true_delta_joint.max()),
        "mean_train_state_distance":float(z.train_state_distance.mean()),
        "mean_history_count":float(z.history_count.mean()),
        "pass6_joint_error_N":float(z[z.Pass_T==6].joint_abs_error.mean()) if (z.Pass_T==6).any() else None,
    })
group_diag=pd.DataFrame(groups).sort_values("joint_Group_MAE_N",ascending=False).reset_index(drop=True)
group_diag["rank_hardest"]=np.arange(1,len(group_diag)+1)
top3=group_diag.head(3).copy()

# Error concentration and associations.
total_group_error=float(group_diag.joint_Group_MAE_N.sum())
top3_share=float(100*top3.joint_Group_MAE_N.sum()/total_group_error)
assoc={}
for col in ["mean_true_delta_N","max_true_delta_N","mean_train_state_distance","Speed_kmh","Load_kN","missing_design_slots","mean_history_count"]:
    rho,pv=spearmanr(group_diag[col],group_diag.joint_Group_MAE_N)
    assoc[col]={"spearman_rho":float(rho) if np.isfinite(rho) else None,
                "p_value":float(pv) if np.isfinite(pv) else None}

pass_diag=[]
for p,z in diag.groupby("Pass_T"):
    pass_diag.append({"Pass_T":int(p),"n":int(len(z)),"joint_MAE_N":float(z.joint_abs_error.mean()),
                      "LC1_MAE_N":float(z.ae_LC1_proxy_N.mean()),"LC5_MAE_N":float(z.ae_LC5_proxy_N.mean()),
                      "mean_true_delta_N":float(z.true_delta_joint.mean())})
pass_diag=pd.DataFrame(pass_diag).sort_values("Pass_T")

# Compare V5 to persistence by group.
persist=[]
for g,z in hist.groupby("Group_VW"):
    vals=[]
    for ti,c in enumerate(TARGET_COLS):
        vals.append(float(mean_absolute_error(z[c],z[prev_peaks[ti]])))
    persist.append({"Group_VW":g,"persistence_joint_MAE_N":float(np.mean(vals))})
persist=pd.DataFrame(persist)
group_diag=group_diag.merge(persist,on="Group_VW",validate="one_to_one")
group_diag["V5_gain_vs_persistence_N"]=group_diag.persistence_joint_MAE_N-group_diag.joint_Group_MAE_N

# Model-selection instability by group.
base_sel=sel[sel.task=="D_V5_BASELINE"].copy()
group_diag=group_diag.merge(base_sel[["outer_group","model","inner_score"]].rename(columns={"outer_group":"Group_VW","model":"V5_selected_family","inner_score":"V5_inner_score"}),on="Group_VW",how="left")

# Save.
pred.to_csv(TABLE/"oof_predictions_v8.csv",index=False)
summary.to_csv(TABLE/"summary_v8.csv",index=False)
search.to_csv(TABLE/"hyperparameter_search_v8.csv",index=False)
sel.to_csv(TABLE/"selection_v8.csv",index=False)
paired_df.to_csv(TABLE/"paired_vs_v5_v8.csv",index=False)
group_pairs_df.to_csv(TABLE/"paired_groups_v8.csv",index=False)
diag.to_csv(TABLE/"v5_row_error_atlas_v8.csv",index=False)
group_diag.to_csv(TABLE/"v5_group_error_atlas_v8.csv",index=False)
pass_diag.to_csv(TABLE/"v5_pass_error_atlas_v8.csv",index=False)

best=summary[summary.task!="D_V5_BASELINE"].sort_values("joint_Group_MAE").iloc[0]
best_pair=paired_df[paired_df.task==best.task].iloc[0]
hard_groups=top3.Group_VW.tolist()

results={
    "schema":"soilbin.q1.v8.state-error",
    "created_utc":datetime.now(timezone.utc).isoformat(),
    "post_lock_exploratory":True,
    "does_not_supersede_v5":True,
    "baseline_reproduction":{
        "joint_Group_MAE_N":float(base.joint_Group_MAE),
        "LC1_Group_MAE_N":float(base.LC1_proxy_N_Group_MAE),
        "LC5_Group_MAE_N":float(base.LC5_proxy_N_Group_MAE),
        "exact_match_to_frozen_v5":True,
    },
    "primary_summary":summary.to_dict(orient="records"),
    "paired_vs_v5":paired_df.to_dict(orient="records"),
    "best_state_candidate":{
        "task":str(best.task),"joint_Group_MAE_N":float(best.joint_Group_MAE),
        "relative_improvement_vs_V5_pct":float(100*(FROZEN_V5_JOINT-float(best.joint_Group_MAE))/FROZEN_V5_JOINT),
        "paired":best_pair.to_dict(),
    },
    "hard_group_diagnostics":{
        "top3_groups":hard_groups,
        "top3_error_share_pct":top3_share,
        "top3_rows":top3.to_dict(orient="records"),
        "associations":assoc,
        "pass_errors":pass_diag.to_dict(orient="records"),
    },
    "guards":{
        "random_split_used":False,
        "external_labels_used_for_selection":False,
        "current_pass_waveform_used":False,
        "only_past_observed_state_used":True,
        "residual_correction_outer_test_leakage":False,
        "response_status":RESPONSE_STATUS,
        "depth_mapping_status":DEPTH_STATUS,
    },
    "interpretation":"V8 tests compact dynamical soil-state summaries and training-only residual correction, while separately diagnosing where the exact frozen V5 protocol fails. Results are exploratory and do not overwrite V5.",
}
dump(ROOT/"RESULTS_V8.json",results)

# Figures.
plt.figure(figsize=(12,6))
q=summary.sort_values("joint_Group_MAE")
plt.bar(np.arange(len(q)),q.joint_Group_MAE)
plt.axhline(FROZEN_V5_JOINT,linestyle="--",linewidth=1)
plt.xticks(np.arange(len(q)),q.task,rotation=55,ha="right",fontsize=8)
plt.ylabel("Joint group-equal MAE (N)")
plt.tight_layout(); plt.savefig(FIG/"01_state_candidates.png",dpi=180); plt.close()

plt.figure(figsize=(10,5))
q=group_diag.sort_values("joint_Group_MAE_N",ascending=False)
plt.bar(np.arange(len(q)),q.joint_Group_MAE_N)
plt.xticks(np.arange(len(q)),q.Group_VW,rotation=45)
plt.ylabel("V5 group MAE (N)")
plt.tight_layout(); plt.savefig(FIG/"02_v5_hard_groups.png",dpi=180); plt.close()

report=[
    "# SoilBin V8 — latent state + hard-group diagnostics",
    "",
    "**Post-lock exploratory analysis; V5 remains frozen.**",
    "",
    f"- Exact V5 reproduction: {float(base.joint_Group_MAE):.6f} N.",
    f"- Best V8 challenger: {best.task} = {float(best.joint_Group_MAE):.6f} N.",
    f"- Top-3 hard groups: {', '.join(hard_groups)}.",
    f"- Top-3 share of summed group error: {top3_share:.2f}%.",
    "",
    "## Candidate models",
]
for _,r in summary.iterrows():
    report.append(f"- {r.task}: {r.joint_Group_MAE:.6f} N (LC1={r.LC1_proxy_N_Group_MAE:.6f}; LC5={r.LC5_proxy_N_Group_MAE:.6f})")
report += ["","## Hard groups"]
for _,r in top3.iterrows():
    report.append(f"- #{int(r.rank_hardest)} {r.Group_VW}: {r.joint_Group_MAE_N:.3f} N; delta={r.mean_true_delta_N:.3f} N; train-distance={r.mean_train_state_distance:.3f}; missing slots={int(r.missing_design_slots)}")
(ROOT/"FINAL_REPORT_V8.md").write_text("\n".join(report),encoding="utf-8")

manifest={
    "schema":"soilbin.q1.v8.run-manifest",
    "finished_utc":datetime.now(timezone.utc).isoformat(),
    "seed":SEED,"python":sys.version.split()[0],
    "source_sha256":EXPECTED_MODEL_SHA,
    "baseline_exact":True,
    "tree_candidates":[{"model":n,"params":p} for n,p in TREE_CANDIDATES],
    "state_tasks":TASKS,
    "residual_variants":RESIDUAL_VARIANTS,
    "files":sorted([p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()]),
}
dump(ROOT/"RUN_MANIFEST_V8.json",manifest)
shutil.make_archive("/kaggle/working/SoilBin_Q1_Q2_V8_STATE_ERROR_20260928","zip",ROOT)

phase("COMPLETE")
print("SOILBIN_V8_COMPLETE",json.dumps({
    "v5_mae_N":float(base.joint_Group_MAE),
    "best_task":str(best.task),
    "best_mae_N":float(best.joint_Group_MAE),
    "best_improvement_pct":float(100*(FROZEN_V5_JOINT-float(best.joint_Group_MAE))/FROZEN_V5_JOINT),
    "hard_groups":hard_groups,
    "top3_error_share_pct":top3_share,
},sort_keys=True),flush=True)
