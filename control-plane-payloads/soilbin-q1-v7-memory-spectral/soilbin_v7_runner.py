# SoilBin Q1/Q2 V7 — V5-family multi-pass + fusion + spectral integration
# Post-lock exploratory model challenge. Does NOT supersede frozen V5 claims.
from __future__ import annotations
import os, sys, json, math, hashlib, random, gzip, base64, itertools, shutil, warnings
from pathlib import Path
from io import StringIO
from datetime import datetime, timezone
from collections import Counter

SEED = 20260914  # exact V3/V5 seed for reproducibility
random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
np.random.seed(SEED)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

ROOT = Path("/kaggle/working/SOILBIN_V7")
TABLE = ROOT / "tables"
FIG = ROOT / "figures"
STATE = ROOT / "state"
for p in (ROOT, TABLE, FIG, STATE):
    p.mkdir(parents=True, exist_ok=True)

EXPECTED_MODEL_SHA = "dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"
EXPECTED_MISSING = {"V1W1T1", "V2W3T2", "V3W2T1"}
TARGET_COLS = ["LC1_peak_magnitude_N_delta", "LC5_peak_magnitude_N_delta"]
TARGET_NAMES = ["LC1_proxy_N", "LC5_proxy_N"]
FROZEN_V5_JOINT_GROUP_MAE = 24.28878365273039
FROZEN_V5_LC1_GROUP_MAE = 12.785420282422
FROZEN_V5_LC5_GROUP_MAE = 35.79214702303878
FROZEN_V5_TASK = "D_delta_prev_peak"
RESPONSE_STATUS = "CALIBRATED_VERTICAL_FORCE_PROXY_N_PENDING_SENSOR_AREA_OR_STRESS_CALIBRATION"
DEPTH_STATUS = "PROVISIONAL_LC5_5CM_LC1_15CM_PENDING_LAYOUT_CONFIRMATION"

def phase(x):
    print(f"CGP_PHASE:{x}", flush=True)

def dump(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                   default=lambda v: v.item() if hasattr(v, "item") else str(v)),
        encoding="utf-8"
    )

def decode_gz_b64(payload, expected_sha):
    raw = gzip.decompress(base64.b64decode(payload.strip()))
    got = hashlib.sha256(raw).hexdigest()
    if got != expected_sha:
        raise RuntimeError(f"SOURCE_FINGERPRINT_MISMATCH:{got}:{expected_sha}")
    return raw.decode("utf-8")

def group_equal_mae(y, p, groups):
    tmp = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float), "g": np.asarray(groups)})
    return float(tmp.assign(e=lambda d: (d.y-d.p).abs()).groupby("g").e.mean().mean())

def metrics_block(pred):
    out = {"n": int(len(pred)), "n_groups": int(pred.Group_VW.nunique())}
    vals = []
    for t in TARGET_NAMES:
        y = pred[f"y_{t}"].to_numpy(float)
        p = pred[f"p_{t}"].to_numpy(float)
        out[f"{t}_Group_MAE"] = group_equal_mae(y, p, pred.Group_VW)
        out[f"{t}_MAE"] = float(mean_absolute_error(y,p))
        out[f"{t}_RMSE"] = float(mean_squared_error(y,p)**0.5)
        out[f"{t}_Bias"] = float(np.mean(p-y))
        out[f"{t}_R2"] = float(r2_score(y,p))
        vals.append(out[f"{t}_Group_MAE"])
    out["joint_Group_MAE"] = float(np.mean(vals))
    return out

def make_estimator(name, params):
    if name == "RandomForest":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("m", RandomForestRegressor(
                n_estimators=250,
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=SEED, n_jobs=1, max_features=1.0
            ))
        ])
    if name == "ExtraTrees":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("m", ExtraTreesRegressor(
                n_estimators=250,
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=SEED, n_jobs=1, max_features=1.0
            ))
        ])
    raise KeyError(name)

# Exact winner-family configurations from the V3 candidate catalog.
CANDIDATES = [
    ("RandomForest", {"max_depth":None, "min_samples_leaf":1}),
    ("RandomForest", {"max_depth":4, "min_samples_leaf":2}),
    ("ExtraTrees", {"max_depth":None, "min_samples_leaf":1}),
    ("ExtraTrees", {"max_depth":4, "min_samples_leaf":2}),
]

phase("DATA_FREEZE")
model_raw = decode_gz_b64(MODEL_CSV_GZ_B64, EXPECTED_MODEL_SHA)
df = pd.read_csv(StringIO(model_raw))
if len(df) != 51 or df.Run_ID.nunique() != 51:
    raise RuntimeError("EXPECTED_51_RUNS")
df["Group_VW"] = df.Run_ID.str.extract(r"^(V\dW\d)")[0]
df["Speed_kmh"] = df["Speed_level"].astype(float)
df["Load_kN"] = df["Weight_level"].astype(float) + 1.0
df["Pass_T"] = df["Pass_T"].astype(int)
all_ids = {f"V{v}W{w}T{t}" for v in range(1,4) for w in range(1,4) for t in range(1,7)}
if all_ids - set(df.Run_ID) != EXPECTED_MISSING:
    raise RuntimeError("DESIGN_SLOT_MISMATCH")
for c in TARGET_COLS:
    if c not in df or df[c].isna().any():
        raise RuntimeError(f"TARGET_INVALID:{c}")

# Build exact 41 V3/V5 consecutive history pairs first.
prev = df.copy()
prev["Pass_T"] = prev["Pass_T"] + 1
rename = {c:f"prev_{c}" for c in TARGET_COLS + ["QC_status"]}
prev_small = prev[["Group_VW","Pass_T"] + TARGET_COLS + ["QC_status"]].rename(columns=rename)
hist = df.merge(prev_small, on=["Group_VW","Pass_T"], how="left", validate="one_to_one")
prev_peaks = [f"prev_{c}" for c in TARGET_COLS]
hist = hist[hist[prev_peaks[0]].notna() & hist[prev_peaks[1]].notna()].copy().reset_index(drop=True)
if len(hist) != 41:
    raise RuntimeError(f"EXPECTED_41_HISTORY_PAIRS_GOT_{len(hist)}")
hist["row_id"] = np.arange(len(hist),dtype=int)
hist["pair_qc_ok"] = (hist["QC_status"].fillna("")=="OK") & (hist["prev_QC_status"].fillna("")=="OK")

# Add lag-2/lag-3 and history summaries using only passes strictly earlier than current pass.
for lag in (2,3):
    src = df[["Group_VW","Pass_T"] + TARGET_COLS].copy()
    src["Pass_T"] = src["Pass_T"] + lag
    src = src.rename(columns={TARGET_COLS[0]:f"lag{lag}_LC1", TARGET_COLS[1]:f"lag{lag}_LC5"})
    hist = hist.merge(src, on=["Group_VW","Pass_T"], how="left", validate="one_to_one")
    hist[f"lag{lag}_available"] = hist[f"lag{lag}_LC1"].notna().astype(float)

seq = {g:z.sort_values("Pass_T").copy() for g,z in df.groupby("Group_VW")}
summary_rows = []
for _, r in hist.iterrows():
    z = seq[r.Group_VW]
    past = z[z.Pass_T < int(r.Pass_T)].sort_values("Pass_T")
    out = {"row_id":int(r.row_id), "history_count":float(len(past))}
    for ch,col in zip(("LC1","LC5"),TARGET_COLS):
        vals = past[col].to_numpy(float)
        x = past.Pass_T.to_numpy(float)
        out[f"hist_{ch}_mean"] = float(np.mean(vals))
        out[f"hist_{ch}_std"] = float(np.std(vals,ddof=0))
        out[f"hist_{ch}_min"] = float(np.min(vals))
        out[f"hist_{ch}_max"] = float(np.max(vals))
        out[f"hist_{ch}_range"] = float(np.ptp(vals))
        out[f"hist_{ch}_weighted"] = float(np.average(vals, weights=np.arange(1,len(vals)+1,dtype=float)))
        out[f"hist_{ch}_cumdelta"] = float(vals[-1]-vals[0])
        out[f"hist_{ch}_lastdelta"] = float(vals[-1]-vals[-2]) if len(vals)>=2 else 0.0
        out[f"hist_{ch}_slope"] = float(np.polyfit(x,vals,1)[0]) if len(vals)>=2 else 0.0
    summary_rows.append(out)
hist = hist.merge(pd.DataFrame(summary_rows), on="row_id", how="left", validate="one_to_one")

# Explicit cross-channel fusion of previous-pass state.
hist["prev_sum"] = hist[prev_peaks[0]] + hist[prev_peaks[1]]
hist["prev_diff_LC5_minus_LC1"] = hist[prev_peaks[1]] - hist[prev_peaks[0]]
hist["prev_ratio_LC5_to_LC1"] = hist[prev_peaks[1]] / np.maximum(hist[prev_peaks[0]], 1e-9)
hist["prev_log_ratio_LC5_to_LC1"] = np.log(np.maximum(hist[prev_peaks[1]],1e-9) / np.maximum(hist[prev_peaks[0]],1e-9))

# Fixed six-pass path-graph spectral coordinates. These use only the predeclared pass topology.
W = np.zeros((6,6),float)
for i in range(5):
    W[i,i+1] = W[i+1,i] = 1.0
L = np.diag(W.sum(axis=1)) - W
evals,evecs = np.linalg.eigh(L)
hist["pass_spec1"] = [float(evecs[int(t)-1,1]) for t in hist.Pass_T]
hist["pass_spec2"] = [float(evecs[int(t)-1,2]) for t in hist.Pass_T]

conditions = ["Load_kN","Speed_kmh","Pass_T"]
baseline = conditions + prev_peaks
lag2 = baseline + ["lag2_LC1","lag2_LC5","lag2_available"]
lag3 = lag2 + ["lag3_LC1","lag3_LC5","lag3_available"]
summary_feats = ["history_count"] + [
    f"hist_{ch}_{s}" for ch in ("LC1","LC5")
    for s in ("mean","std","range","weighted","cumdelta","lastdelta","slope")
]
fusion_feats = ["prev_sum","prev_diff_LC5_minus_LC1","prev_ratio_LC5_to_LC1","prev_log_ratio_LC5_to_LC1"]
spectral_aug = baseline + ["pass_spec1","pass_spec2"]
spectral_replace = ["Load_kN","Speed_kmh"] + prev_peaks + ["pass_spec1","pass_spec2"]

TASKS = {
    "D_V5_BASELINE": {"features":baseline},
    "M2_LAG2": {"features":lag2},
    "M3_LAG3": {"features":lag3},
    "SUM_HISTORY": {"features":baseline + summary_feats},
    "FUSION_EXPLICIT": {"features":baseline + fusion_feats},
    "SPECTRAL_REPLACE_PASS": {"features":spectral_replace},
    "SPECTRAL_AUGMENT_PASS": {"features":spectral_aug},
    "FULL_M3_SUM_FUSION_SPECTRAL": {
        "features":lag3 + summary_feats + fusion_feats + ["pass_spec1","pass_spec2"]
    },
}

def y_delta(data):
    return data[TARGET_COLS].to_numpy(float) - data[prev_peaks].to_numpy(float)

def inner_select(train, task_name, search_rows, outer_label):
    feats = TASKS[task_name]["features"]
    groups = train.Group_VW.to_numpy()
    n_splits = min(4, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("INNER_GROUPS_LT_2")
    gkf = GroupKFold(n_splits=n_splits)
    X = train[feats]
    yd = y_delta(train)
    best = None
    for ci,(name,params) in enumerate(CANDIDATES):
        p_delta = np.full((len(train),2),np.nan,float)
        failed = None
        for tr,va in gkf.split(X, groups=groups):
            try:
                for ti in range(2):
                    est = make_estimator(name,params)
                    est.fit(X.iloc[tr],yd[tr,ti])
                    p_delta[va,ti] = est.predict(X.iloc[va])
            except Exception as exc:
                failed = f"{type(exc).__name__}:{str(exc)[:160]}"
                break
        if failed or np.isnan(p_delta).any():
            score = float("inf")
        else:
            final = p_delta + train[prev_peaks].to_numpy(float)
            score = float(np.mean([
                group_equal_mae(train[TARGET_COLS[ti]], final[:,ti], groups)
                for ti in range(2)
            ]))
        search_rows.append({
            "outer":str(outer_label),"task":task_name,"candidate_index":ci,
            "model":name,"params":json.dumps(params,sort_keys=True),
            "inner_joint_Group_MAE":score if math.isfinite(score) else None,
            "failed":failed
        })
        if math.isfinite(score) and (best is None or score < best[0]-1e-12):
            best = (score,name,params)
    if best is None:
        raise RuntimeError(f"NO_VALID_CANDIDATE:{task_name}:{outer_label}")
    return best

def outer_labels(data, scheme):
    if scheme=="VW": return list(pd.unique(data.Group_VW)), "Group_VW"
    if scheme=="speed": return sorted(pd.unique(data.Speed_kmh)), "Speed_kmh"
    if scheme=="load": return sorted(pd.unique(data.Load_kN)), "Load_kN"
    raise KeyError(scheme)

def run_task(data, task_name, scheme="VW", qc_only=False):
    d = data[data.pair_qc_ok].copy() if qc_only else data.copy()
    labels,col = outer_labels(d,scheme)
    preds=[]; searches=[]; sels=[]
    for label in labels:
        te=d[d[col]==label].copy()
        tr=d[d[col]!=label].copy()
        if len(te)==0 or tr.Group_VW.nunique()<2:
            continue
        inner_score,name,params = inner_select(tr,task_name,searches,f"{scheme}:{label}")
        feats = TASKS[task_name]["features"]
        Xtr=tr[feats]; Xte=te[feats]; yd=y_delta(tr)
        pdelta=[]
        for ti in range(2):
            est=make_estimator(name,params)
            est.fit(Xtr,yd[:,ti])
            pdelta.append(est.predict(Xte))
        final=np.column_stack(pdelta)+te[prev_peaks].to_numpy(float)
        sels.append({
            "task":task_name,"scheme":scheme,"qc_only":bool(qc_only),
            "outer_label":str(label),"model":name,
            "params":json.dumps(params,sort_keys=True),
            "inner_joint_Group_MAE":float(inner_score)
        })
        for j,(_,r) in enumerate(te.iterrows()):
            row={
                "row_id":int(r.row_id),"Run_ID":r.Run_ID,"Group_VW":r.Group_VW,
                "Speed_kmh":float(r.Speed_kmh),"Load_kN":float(r.Load_kN),"Pass_T":int(r.Pass_T),
                "task":task_name,"scheme":scheme,"qc_only":bool(qc_only),
                "outer_label":str(label),"model":"Selected_inner_tree_family"
            }
            for ti,t in enumerate(TARGET_NAMES):
                row[f"y_{t}"]=float(r[TARGET_COLS[ti]])
                row[f"p_{t}"]=float(final[j,ti])
            preds.append(row)
    pred=pd.DataFrame(preds)
    if pred.row_id.nunique()!=len(d):
        raise RuntimeError(f"INCOMPLETE_OOF:{task_name}:{scheme}:{qc_only}:{pred.row_id.nunique()}:{len(d)}")
    return pred,pd.DataFrame(searches),pd.DataFrame(sels)

phase("PRIMARY_NESTED_GROUPED_CV")
preds=[]; searches=[]; sels=[]; summary=[]
ACTIVE_TASKS = {"D_V5_BASELINE":TASKS["D_V5_BASELINE"]} if os.environ.get("SOILBIN_V7_BASELINE_ONLY","")=="1" else TASKS
for i,task in enumerate(ACTIVE_TASKS,1):
    phase(f"TASK_{i}_{task}")
    p,s,z = run_task(hist,task,"VW",False)
    preds.append(p); searches.append(s); sels.append(z)
    m=metrics_block(p); m.update({"task":task,"scheme":"VW","qc_only":False})
    summary.append(m)
    dump(STATE/"progress_v7.json",{"completed":list(ACTIVE_TASKS)[:i],"total":len(ACTIVE_TASKS)})

pred=pd.concat(preds,ignore_index=True)
search=pd.concat(searches,ignore_index=True)
sel=pd.concat(sels,ignore_index=True)
summary_df=pd.DataFrame(summary).sort_values("joint_Group_MAE").reset_index(drop=True)

# V5-family representation reproduction gate. Runtime/library versions may cause small deterministic drift.
base_metrics=summary_df[summary_df.task=="D_V5_BASELINE"].iloc[0]
baseline_drift_pct=float(100*(float(base_metrics.joint_Group_MAE)-FROZEN_V5_JOINT_GROUP_MAE)/FROZEN_V5_JOINT_GROUP_MAE)
baseline_exact_match=bool(abs(float(base_metrics.joint_Group_MAE)-FROZEN_V5_JOINT_GROUP_MAE) <= 1e-9)
baseline_within_2pct=bool(abs(baseline_drift_pct) <= 2.0)
if not baseline_within_2pct:
    raise RuntimeError(f"V5_BASELINE_REPRODUCTION_DRIFT_GT_2PCT:{float(base_metrics.joint_Group_MAE)}:{FROZEN_V5_JOINT_GROUP_MAE}:{baseline_drift_pct}")
print("V5_BASELINE_REPRODUCED_WITHIN_2PCT", float(base_metrics.joint_Group_MAE), baseline_drift_pct, flush=True)

if os.environ.get("SOILBIN_V7_BASELINE_ONLY","")=="1":
    print("SOILBIN_V7_BASELINE_ONLY_COMPLETE",flush=True)
    raise SystemExit(0)

# Paired group-level inference for every extension versus exact V5 baseline.
phase("PAIRED_INFERENCE")
baseline_pred=pred[(pred.task=="D_V5_BASELINE")&(pred.scheme=="VW")&(~pred.qc_only)].copy()
paired_rows=[]
group_rows=[]
rng=np.random.default_rng(20260928)
for task in TASKS:
    if task=="D_V5_BASELINE": continue
    challenger=pred[(pred.task==task)&(pred.scheme=="VW")&(~pred.qc_only)].copy()
    diffs=[]
    improved=0
    for g in sorted(hist.Group_VW.unique()):
        b=baseline_pred[baseline_pred.Group_VW==g]
        c=challenger[challenger.Group_VW==g]
        bmae=float(np.mean([mean_absolute_error(b[f"y_{t}"],b[f"p_{t}"]) for t in TARGET_NAMES]))
        cmae=float(np.mean([mean_absolute_error(c[f"y_{t}"],c[f"p_{t}"]) for t in TARGET_NAMES]))
        diff=bmae-cmae
        diffs.append(diff); improved += int(diff>0)
        group_rows.append({"task":task,"Group_VW":g,"baseline_joint_MAE":bmae,"challenger_joint_MAE":cmae,"improvement_N":diff})
    diffs=np.asarray(diffs,float)
    obs=float(np.mean(diffs))
    perm=np.array([np.mean(diffs*np.array(signs)) for signs in itertools.product([-1,1],repeat=len(diffs))])
    pone=float(np.sum(perm>=obs-1e-12)/len(perm))
    boot=np.array([np.mean(rng.choice(diffs,size=len(diffs),replace=True)) for _ in range(20000)])
    met=summary_df[summary_df.task==task].iloc[0]
    rel_runtime=float(100*(float(base_metrics.joint_Group_MAE)-float(met.joint_Group_MAE))/float(base_metrics.joint_Group_MAE))
    rel_frozen=float(100*(FROZEN_V5_JOINT_GROUP_MAE-float(met.joint_Group_MAE))/FROZEN_V5_JOINT_GROUP_MAE)
    lc1_worse=float(100*(float(met.LC1_proxy_N_Group_MAE)-float(base_metrics.LC1_proxy_N_Group_MAE))/float(base_metrics.LC1_proxy_N_Group_MAE))
    lc5_worse=float(100*(float(met.LC5_proxy_N_Group_MAE)-float(base_metrics.LC5_proxy_N_Group_MAE))/float(base_metrics.LC5_proxy_N_Group_MAE))
    guard=bool(lc1_worse<=5.0 and lc5_worse<=5.0)
    promising=bool((rel_runtime>=5.0 or (obs>0 and pone<=0.05)) and guard)
    paired_rows.append({
        "task":task,
        "joint_Group_MAE_N":float(met.joint_Group_MAE),
        "relative_improvement_vs_runtime_baseline_pct":rel_runtime,
        "relative_improvement_vs_frozen_V5_reference_pct":rel_frozen,
        "mean_group_paired_improvement_N":obs,
        "group_bootstrap_95CI_low_N":float(np.quantile(boot,.025)),
        "group_bootstrap_95CI_high_N":float(np.quantile(boot,.975)),
        "exact_signflip_one_sided_p":pone,
        "groups_improved":int(improved),
        "groups_total":int(len(diffs)),
        "LC1_worsening_vs_runtime_baseline_pct":lc1_worse,
        "LC5_worsening_vs_runtime_baseline_pct":lc5_worse,
        "per_target_no_worse_than_5pct":guard,
        "promising_exploratory_candidate":promising,
    })
paired_df=pd.DataFrame(paired_rows).sort_values("joint_Group_MAE_N").reset_index(drop=True)
group_df=pd.DataFrame(group_rows)

# Pre-registered FULL task robustness; baseline also repeated for direct comparison.
phase("ROBUSTNESS")
rob_rows=[]
rob_pred=[]
rob_sel=[]
rob_search=[]
for task in ["D_V5_BASELINE","FULL_M3_SUM_FUSION_SPECTRAL"]:
    for scheme,qc in [("speed",False),("load",False),("VW",True)]:
        p,s,z=run_task(hist,task,scheme,qc)
        rob_pred.append(p); rob_sel.append(z); rob_search.append(s)
        m=metrics_block(p); m.update({"task":task,"scheme":scheme,"qc_only":bool(qc)})
        rob_rows.append(m)
rob_df=pd.DataFrame(rob_rows)

# Selection counts.
selection_counts={}
for task,g in sel.groupby("task"):
    selection_counts[task]=dict(Counter(g.model))

# Best extension is descriptive only; no overwriting V5.
ext_summary=summary_df[summary_df.task!="D_V5_BASELINE"].copy()
best=ext_summary.iloc[0].to_dict()
best_pair=paired_df.iloc[0].to_dict()

# Persistence reference on same 41 pairs.
pers=[]
for _,r in hist.iterrows():
    row={"Group_VW":r.Group_VW}
    for ti,t in enumerate(TARGET_NAMES):
        row[f"y_{t}"]=float(r[TARGET_COLS[ti]])
        row[f"p_{t}"]=float(r[prev_peaks[ti]])
    pers.append(row)
pers=pd.DataFrame(pers)
pvals=[group_equal_mae(pers[f"y_{t}"],pers[f"p_{t}"],pers.Group_VW) for t in TARGET_NAMES]
persistence_joint=float(np.mean(pvals))

# Save evidence.
pred_all=pd.concat([pred]+rob_pred,ignore_index=True)
sel_all=pd.concat([sel]+rob_sel,ignore_index=True)
search_all=pd.concat([search]+rob_search,ignore_index=True)
pred_all.to_csv(TABLE/"oof_predictions_v7.csv",index=False)
sel_all.to_csv(TABLE/"selection_v7.csv",index=False)
search_all.to_csv(TABLE/"hyperparameter_search_v7.csv",index=False)
summary_df.to_csv(TABLE/"primary_summary_v7.csv",index=False)
paired_df.to_csv(TABLE/"paired_vs_v5_v7.csv",index=False)
group_df.to_csv(TABLE/"paired_groups_v7.csv",index=False)
rob_df.to_csv(TABLE/"robustness_v7.csv",index=False)

# Compact results.
results={
    "schema":"soilbin.q1.v7.memory-spectral",
    "created_utc":datetime.now(timezone.utc).isoformat(),
    "post_lock_exploratory":True,
    "does_not_supersede_v5":True,
    "source_model_sha256":EXPECTED_MODEL_SHA,
    "n_runs":51,"n_history_pairs":41,"n_groups":9,
    "candidate_family":["RandomForest","ExtraTrees"],
    "candidate_config_count":len(CANDIDATES),
    "baseline_reproduction":{
        "task":"D_V5_BASELINE",
        "joint_Group_MAE_N":float(base_metrics.joint_Group_MAE),
        "LC1_Group_MAE_N":float(base_metrics.LC1_proxy_N_Group_MAE),
        "LC5_Group_MAE_N":float(base_metrics.LC5_proxy_N_Group_MAE),
        "frozen_v5_joint_Group_MAE_N":FROZEN_V5_JOINT_GROUP_MAE,
        "runtime_drift_pct":baseline_drift_pct,
        "exact_match_to_frozen_v5":baseline_exact_match,
        "within_2pct_of_frozen_v5":baseline_within_2pct,
    },
    "persistence_joint_Group_MAE_N":persistence_joint,
    "primary_summary":summary_df.to_dict(orient="records"),
    "paired_vs_v5":paired_df.to_dict(orient="records"),
    "robustness":rob_df.to_dict(orient="records"),
    "selection_counts":selection_counts,
    "best_extension_descriptive":{
        "task":best["task"],
        "joint_Group_MAE_N":float(best["joint_Group_MAE"]),
        "relative_improvement_vs_runtime_baseline_pct":float(100*(float(base_metrics.joint_Group_MAE)-float(best["joint_Group_MAE"]))/float(base_metrics.joint_Group_MAE)),
        "relative_improvement_vs_frozen_v5_reference_pct":float(100*(FROZEN_V5_JOINT_GROUP_MAE-float(best["joint_Group_MAE"]))/FROZEN_V5_JOINT_GROUP_MAE),
        "paired":best_pair,
    },
    "full_pre_registered_challenger":paired_df[paired_df.task=="FULL_M3_SUM_FUSION_SPECTRAL"].iloc[0].to_dict(),
    "spectral_eigenvalues":evals.tolist(),
    "guards":{
        "random_split_used":False,
        "external_labels_used_for_selection":False,
        "current_pass_waveform_used":False,
        "only_past_state_features":True,
        "response_status":RESPONSE_STATUS,
        "depth_mapping_status":DEPTH_STATUS,
    },
    "interpretation":"Exploratory post-lock challenge. V5 remains frozen regardless of V7 outcome until independent confirmation and an explicit new selection policy.",
}
dump(ROOT/"RESULTS_V7.json",results)

# Figure.
plot=summary_df.sort_values("joint_Group_MAE")
plt.figure(figsize=(12,6))
plt.bar(np.arange(len(plot)),plot.joint_Group_MAE)
plt.axhline(FROZEN_V5_JOINT_GROUP_MAE,linestyle="--",linewidth=1)
plt.xticks(np.arange(len(plot)),plot.task,rotation=55,ha="right",fontsize=8)
plt.ylabel("Joint group-equal MAE (N)")
plt.tight_layout()
plt.savefig(FIG/"01_v7_primary_comparison.png",dpi=180)
plt.close()

report=[
    "# SoilBin V7 — V5-family multi-pass + fusion + spectral integration",
    "",
    "**Post-lock exploratory analysis; V5 remains frozen.**",
    "",
    f"- V5 representation baseline in current runtime: {float(base_metrics.joint_Group_MAE):.6f} N (drift vs frozen V5={baseline_drift_pct:.3f}%).",
    f"- Persistence reference: {persistence_joint:.6f} N.",
    f"- Best extension (descriptive): {best['task']} = {float(best['joint_Group_MAE']):.6f} N.",
    "",
    "## Primary tasks"
]
for _,r in summary_df.iterrows():
    report.append(f"- {r.task}: joint Group-MAE={r.joint_Group_MAE:.6f} N; LC1={r.LC1_proxy_N_Group_MAE:.6f}; LC5={r.LC5_proxy_N_Group_MAE:.6f}")
report += ["","## Paired versus V5"]
for _,r in paired_df.iterrows():
    report.append(
        f"- {r.task}: improvement_vs_runtime_baseline={r.relative_improvement_vs_runtime_baseline_pct:.3f}%; "
        f"improvement_vs_frozen_V5={r.relative_improvement_vs_frozen_V5_reference_pct:.3f}%; "
        f"mean paired={r.mean_group_paired_improvement_N:.3f} N; "
        f"p={r.exact_signflip_one_sided_p:.6f}; groups={int(r.groups_improved)}/9; "
        f"promising={bool(r.promising_exploratory_candidate)}"
    )
(ROOT/"FINAL_REPORT_V7.md").write_text("\n".join(report),encoding="utf-8")

manifest={
    "schema":"soilbin.q1.v7.run-manifest",
    "finished_utc":datetime.now(timezone.utc).isoformat(),
    "seed":SEED,
    "python":sys.version.split()[0],
    "source_sha256":EXPECTED_MODEL_SHA,
    "baseline_reproduction_exact":baseline_exact_match,
    "baseline_reproduction_within_2pct":baseline_within_2pct,
    "tasks":{k:v["features"] for k,v in TASKS.items()},
    "candidate_catalog":[{"model":n,"params":p} for n,p in CANDIDATES],
    "files":sorted([p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()]),
}
dump(ROOT/"RUN_MANIFEST_V7.json",manifest)

shutil.make_archive("/kaggle/working/SoilBin_Q1_Q2_V7_MEMORY_SPECTRAL_20260928","zip",ROOT)
phase("COMPLETE")
print("SOILBIN_V7_COMPLETE",json.dumps({
    "v5_reproduced":float(base_metrics.joint_Group_MAE),
    "best_extension":best["task"],
    "best_extension_mae_N":float(best["joint_Group_MAE"]),
    "best_extension_improvement_vs_runtime_baseline_pct":float(100*(float(base_metrics.joint_Group_MAE)-float(best["joint_Group_MAE"]))/float(base_metrics.joint_Group_MAE)),
    "best_extension_improvement_vs_frozen_v5_pct":float(100*(FROZEN_V5_JOINT_GROUP_MAE-float(best["joint_Group_MAE"]))/FROZEN_V5_JOINT_GROUP_MAE),
    "full_challenger_mae_N":float(summary_df[summary_df.task=="FULL_M3_SUM_FUSION_SPECTRAL"].iloc[0].joint_Group_MAE),
},sort_keys=True),flush=True)
