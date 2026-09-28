# SoilBin Q1/Q2 V6 — 23-item exploratory extension
# Post-lock exploratory compute. Does NOT supersede V5 frozen manuscript/model claims.
from __future__ import annotations
import os, sys, json, math, hashlib, random, gzip, base64, itertools, warnings, shutil
from pathlib import Path
from io import StringIO
from datetime import datetime, timezone
from collections import defaultdict

SEED = 20260928
random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
np.random.seed(SEED)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

ROOT = Path("/kaggle/working/SOILBIN_V6_23")
TABLE = ROOT / "tables"
FIG = ROOT / "figures"
ITEM = ROOT / "items"
for p in (ROOT, TABLE, FIG, ITEM):
    p.mkdir(parents=True, exist_ok=True)

EXPECTED_MODEL_SHA = "dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"
EXPECTED_CORROBORATION_SHA = "6d28c811f1ae1dfcf868ec4cea110608178986f6133aaf6220f70e41926a9f40"
EXPECTED_REGISTRY_SHA = "d4c54e2fee3df55e0155c6ed9c5ca82296ff0653bd53cabe6fb2da3abdddf5d1"
EXPECTED_MISSING = {"V1W1T1", "V2W3T2", "V3W2T1"}
TARGETS = ["LC1_peak_magnitude_N_delta", "LC5_peak_magnitude_N_delta"]
TARGET_SHORT = ["LC1", "LC5"]
LOCKED_V5_CHAMPION = "V3_D_DELTA_PREV_PEAK_FROZEN"
LOCKED_V5_JOINT_GROUP_MAE_N = 24.28878365273039
DEPTH_STATUS = "PROVISIONAL_LC5_5CM_LC1_15CM_PENDING_LAYOUT_CONFIRMATION"
RESPONSE_STATUS = "CALIBRATED_VERTICAL_FORCE_PROXY_N_PENDING_SENSOR_AREA_OR_STRESS_CALIBRATION"

def phase(name):
    print("CGP_PHASE:" + name, flush=True)

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                                     default=lambda x: x.item() if hasattr(x, "item") else str(x)),
                          encoding="utf-8")

def decode_gz_b64(payload, expected_sha):
    raw = gzip.decompress(base64.b64decode(payload.strip()))
    got = hashlib.sha256(raw).hexdigest()
    if got != expected_sha:
        raise RuntimeError(f"SOURCE_FINGERPRINT_MISMATCH:{got}:{expected_sha}")
    return raw.decode("utf-8")

def decode_b64(payload, expected_sha):
    raw = base64.b64decode(payload.strip())
    got = hashlib.sha256(raw).hexdigest()
    if got != expected_sha:
        raise RuntimeError(f"SOURCE_FINGERPRINT_MISMATCH:{got}:{expected_sha}")
    return raw.decode("utf-8")

def group_equal_mae(y, p, groups):
    y = np.asarray(y, float); p = np.asarray(p, float); groups = np.asarray(groups)
    vals = []
    for g in pd.unique(groups):
        m = groups == g
        vals.append(np.mean(np.abs(y[m] - p[m]), axis=0))
    return np.mean(np.vstack(vals), axis=0)

def joint_group_mae(y, p, groups):
    return float(np.mean(group_equal_mae(y, p, groups)))

def fixed_ridge(alpha=10.0):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha))
    ])

def grouped_oof_fixed(X, y, groups, alpha=10.0):
    X = np.asarray(X, float); y = np.asarray(y, float); groups = np.asarray(groups)
    pred = np.full_like(y, np.nan, dtype=float)
    for g in pd.unique(groups):
        tr = groups != g; te = groups == g
        m = fixed_ridge(alpha)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred, joint_group_mae(y, pred, groups)

def select_alpha_train(X, y, groups, alphas=(0.1, 1.0, 10.0, 100.0)):
    ug = np.unique(groups)
    if len(ug) < 3:
        return 10.0
    n = min(4, len(ug))
    splitter = GroupKFold(n_splits=n)
    best = None
    for alpha in alphas:
        ps = np.full_like(y, np.nan, dtype=float)
        for tr, va in splitter.split(X, y, groups):
            m = fixed_ridge(alpha)
            m.fit(X[tr], y[tr]); ps[va] = m.predict(X[va])
        score = joint_group_mae(y, ps, groups)
        row = (score, alpha)
        if best is None or row < best:
            best = row
    return float(best[1])

def grouped_oof_nested_ridge(X, y, groups):
    X = np.asarray(X, float); y = np.asarray(y, float); groups = np.asarray(groups)
    pred = np.full_like(y, np.nan, dtype=float)
    chosen = {}
    for g in pd.unique(groups):
        tr = groups != g; te = groups == g
        a = select_alpha_train(X[tr], y[tr], groups[tr])
        m = fixed_ridge(a)
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
        chosen[str(g)] = a
    return pred, joint_group_mae(y, pred, groups), chosen

def linear_slope(vals):
    z = np.asarray(vals, float)
    ok = np.isfinite(z)
    if ok.sum() < 2:
        return np.nan
    x = np.arange(ok.sum(), dtype=float)
    return float(np.polyfit(x, z[ok], 1)[0])

phase("DATA_FREEZE")
model_raw = decode_gz_b64(MODEL_CSV_GZ_B64, EXPECTED_MODEL_SHA)
corroboration_raw = decode_gz_b64(CORROBORATION_CSV_GZ_B64, EXPECTED_CORROBORATION_SHA)
registry_raw = decode_b64(EXTERNAL_REGISTRY_B64, EXPECTED_REGISTRY_SHA)
df = pd.read_csv(StringIO(model_raw))
corroboration = pd.read_csv(StringIO(corroboration_raw))
registry = json.loads(registry_raw)

if len(df) != 51 or df.Run_ID.nunique() != 51:
    raise RuntimeError("EXPECTED_51_RUNS")
df["Group_VW"] = df.Run_ID.str.extract(r"^(V\dW\d)")[0]
df["Speed"] = df["Speed_level"].astype(float)
df["Load"] = df["Weight_level"].astype(float) + 1.0
df["Pass"] = df["Pass_T"].astype(int)
all_ids = {f"V{v}W{w}T{t}" for v in range(1,4) for w in range(1,4) for t in range(1,7)}
if all_ids - set(df.Run_ID) != EXPECTED_MISSING:
    raise RuntimeError("DESIGN_SLOT_MISMATCH")
for c in TARGETS:
    if c not in df.columns or df[c].isna().any():
        raise RuntimeError("INVALID_TARGET:" + c)

seq = {g: z.sort_values("Pass").copy() for g, z in df.groupby("Group_VW")}
group_conditions = df.groupby("Group_VW")[["Speed", "Load"]].first().sort_index()

# Consecutive transition table with history summaries.
records = []
for g, z in seq.items():
    by = {int(r.Pass): r for _, r in z.iterrows()}
    for t in range(2, 7):
        if t not in by or (t-1) not in by:
            continue
        cur, prev = by[t], by[t-1]
        hist_passes = sorted([k for k in by if k < t])
        row = {
            "Group_VW": g, "TargetPass": t, "Speed": float(cur.Speed), "Load": float(cur.Load),
            "prev_LC1": float(prev[TARGETS[0]]), "prev_LC5": float(prev[TARGETS[1]]),
            "y_LC1": float(cur[TARGETS[0]]), "y_LC5": float(cur[TARGETS[1]]),
        }
        for lag in (1,2,3,4,5):
            q = t-lag
            row[f"lag{lag}_LC1"] = float(by[q][TARGETS[0]]) if q in by else np.nan
            row[f"lag{lag}_LC5"] = float(by[q][TARGETS[1]]) if q in by else np.nan
        for ch, col in zip(TARGET_SHORT, TARGETS):
            vals = [float(by[k][col]) for k in hist_passes]
            row[f"hist_{ch}_mean"] = float(np.mean(vals))
            row[f"hist_{ch}_std"] = float(np.std(vals, ddof=0))
            row[f"hist_{ch}_min"] = float(np.min(vals))
            row[f"hist_{ch}_max"] = float(np.max(vals))
            row[f"hist_{ch}_slope"] = linear_slope(vals)
            row[f"hist_{ch}_cumdelta"] = float(vals[-1] - vals[0])
            row[f"hist_{ch}_lastdelta"] = float(vals[-1] - vals[-2]) if len(vals)>=2 else 0.0
            row[f"hist_{ch}_curvature"] = float((vals[-1]-vals[-2])-(vals[-2]-vals[-3])) if len(vals)>=3 else 0.0
            weights = np.arange(1, len(vals)+1, dtype=float)
            row[f"hist_{ch}_weighted"] = float(np.average(vals, weights=weights))
        records.append(row)
trans = pd.DataFrame(records)
if len(trans) != 41:
    raise RuntimeError(f"EXPECTED_41_TRANSITIONS_GOT_{len(trans)}")
Ytrans = trans[["y_LC1","y_LC5"]].to_numpy(float)
Gtrans = trans.Group_VW.to_numpy()

item_rows = []
def item_result(item_id, title, status, primary_metric=None, note=None, files=None):
    row = {"item_id": item_id, "title": title, "status": status,
           "primary_metric": primary_metric, "note": note, "files": files or []}
    item_rows.append(row)
    dump(ITEM / f"{item_id.replace('.','_')}.json", row | {"created_utc": datetime.now(timezone.utc).isoformat()})
    return row

# -------------------------------------------------------------------------
phase("ITEM_1_CALIBRATION")
cal_rows = []
for ch in TARGET_SHORT:
    dcol = f"{ch}_peak_delta_raw_kg"
    ncol = f"{ch}_peak_magnitude_N_delta"
    ratios = df[ncol].to_numpy(float) / np.maximum(np.abs(df[dcol].to_numpy(float)), 1e-12)
    cal_rows.append({
        "channel": ch, "n": len(ratios), "median_N_per_rawkg": float(np.median(ratios)),
        "mean_N_per_rawkg": float(np.mean(ratios)), "sd_N_per_rawkg": float(np.std(ratios)),
        "cv_pct": float(100*np.std(ratios)/np.mean(ratios)),
        "min_N_per_rawkg": float(np.min(ratios)), "max_N_per_rawkg": float(np.max(ratios)),
        "sample_dt_ms_unique": sorted(map(float, df[f"{ch}_sample_dt_ms"].dropna().unique())),
        "n_samples_range": [int(df[f"{ch}_n_samples"].min()), int(df[f"{ch}_n_samples"].max())],
    })
cal = pd.DataFrame(cal_rows)
cal.to_csv(TABLE/"01_calibration_audit.csv", index=False)
calibration_consistent = bool(cal.cv_pct.max() < 0.1)
cal_meta = {
    "response_status": RESPONSE_STATUS, "depth_status": DEPTH_STATUS,
    "unknown_effective_sensor_area": True, "unknown_confirmed_depth_mapping": True,
    "numeric_rescaling_applied": False,
    "reason": "Existing N/raw conversion is internally audited. No force-to-stress conversion or depth reassignment is applied without physical calibration evidence.",
    "scale_consistent_cv_lt_0_1pct": calibration_consistent,
}
dump(ITEM/"01_calibration_details.json", cal_meta)
item_result("1", "Experimental conditions and calibration audit",
            "COMPUTED_WITH_PHYSICAL_PLACEHOLDERS",
            float(cal.cv_pct.max()), "Max conversion CV%; no invented sensor area/depth.", ["tables/01_calibration_audit.csv"])

# -------------------------------------------------------------------------
phase("ITEM_2_MULTI_CHANNEL_MEMORY")
cond_cols = ["Speed","Load","TargetPass"]
lag1_cols = cond_cols + ["lag1_LC1","lag1_LC5"]
multi_cols = lag1_cols + ["lag2_LC1","lag2_LC5","lag3_LC1","lag3_LC5",
                          "hist_LC1_mean","hist_LC5_mean","hist_LC1_slope","hist_LC5_slope",
                          "hist_LC1_weighted","hist_LC5_weighted"]
rows = []
for name, cols in [("conditions_only",cond_cols),("lag1_both_channels",lag1_cols),("cumulative_multichannel_history",multi_cols)]:
    pred, mae, chosen = grouped_oof_nested_ridge(trans[cols].to_numpy(float), Ytrans, Gtrans)
    rows.append({"representation":name,"joint_group_mae_N":mae,"chosen_alpha_by_outer_group":json.dumps(chosen,sort_keys=True)})
mem = pd.DataFrame(rows)
mem.to_csv(TABLE/"02_multichannel_memory.csv",index=False)
item_result("2","Cumulative multi-channel (provisional multi-depth) soil memory","COMPLETE_EXPLORATORY",
            float(mem.joint_group_mae_N.min()), "Channels analyzed jointly; physical depth labels remain provisional.", ["tables/02_multichannel_memory.csv"])

# -------------------------------------------------------------------------
phase("ITEM_3_PRIOR_ART")
gap_tags = {
    "EV_CANADA_AYETAN_2026": ["depth","load/tire treatment","external physics"],
    "EV_ERDC_CRREL_2009": ["repeated pass","depth","external physics"],
    "EV_KELLER_2014": ["soil stress","depth","physics"],
    "EV_ARVIDSSON_KELLER_2007": ["soil stress","wheel load","physics"],
    "EV_KELLER_2016": ["soil stress","depth","physics"],
    "EV_BAHRAMI_2023": ["soil-tool","mechanistic corroboration"],
    "REP_GHESHLAGHI_MARDANI_2021": ["near-domain ML","replication"],
    "REP_FARHADI_2025": ["near-domain ML","replication"],
}
prior_rows = []
for r in registry:
    prior_rows.append({"id":r.get("id"),"role":r.get("role"),"doi":r.get("doi"),"url":r.get("url"),
                       "mapped_topics":";".join(gap_tags.get(r.get("id"),[]))})
prior = pd.DataFrame(prior_rows)
prior.to_csv(TABLE/"03_prior_art_gap_matrix.csv",index=False)
prior_meta = {
    "frozen_registry_sources": int(len(prior)), "structured_corroboration_rows": int(len(corroboration)),
    "live_search_performed_inside_compute": False,
    "novelty_questions_to_test": [
        "Does cumulative pass history improve grouped generalization beyond previous-pass state?",
        "Can pass/channel graph structure reveal stable low-dimensional soil-state organization?",
        "Can information-optimal design reduce experimental burden without losing predictive information?"
    ]
}
dump(ITEM/"03_prior_art_meta.json",prior_meta)
item_result("3","Dedicated prior-art gap analysis","COMPLETE_ON_FROZEN_REGISTRY",
            int(len(prior)), "Registry/corroboration evidence only; no external labels tune models.", ["tables/03_prior_art_gap_matrix.csv"])

# -------------------------------------------------------------------------
phase("ITEM_4_1_TRAJECTORY")
def fit_family(x,y,fam):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if fam=="linear":
        c=np.polyfit(x,y,1); p=np.polyval(c,x); k=2
    elif fam=="quadratic":
        c=np.polyfit(x,y,2); p=np.polyval(c,x); k=3
    elif fam=="log":
        A=np.c_[np.ones(len(x)),np.log(x)]; c=np.linalg.lstsq(A,y,rcond=None)[0]; p=A@c; k=2
    elif fam=="exponential":
        ly=np.log(np.maximum(y,1e-9)); c=np.polyfit(x,ly,1); p=np.exp(np.polyval(c,x)); k=2
    elif fam=="power":
        A=np.c_[np.ones(len(x)),np.log(x)]; c=np.linalg.lstsq(A,np.log(np.maximum(y,1e-9)),rcond=None)[0]; p=np.exp(A@c); k=2
    elif fam=="saturating":
        def f(t,c,a,b): return c+a*np.exp(-b*t)
        popt,_=curve_fit(f,x,y,p0=[float(y[-1]),float(y[0]-y[-1]),0.5],maxfev=10000,
                         bounds=([-1e4,-1e4,1e-5],[1e4,1e4,10]))
        p=f(x,*popt); k=3
    elif fam=="piecewise":
        best=None
        for bp in range(2,6):
            left=x<=bp; right=x>bp
            if left.sum()<2 or right.sum()<2: continue
            c1=np.polyfit(x[left],y[left],1); c2=np.polyfit(x[right],y[right],1)
            pp=np.empty_like(y); pp[left]=np.polyval(c1,x[left]); pp[right]=np.polyval(c2,x[right])
            sse=float(np.sum((y-pp)**2))
            if best is None or sse<best[0]: best=(sse,pp,bp)
        if best is None: raise RuntimeError("NO_PIECEWISE")
        p=best[1]; k=4
    else: raise KeyError(fam)
    resid=y-p; sse=max(float(np.sum(resid**2)),1e-12); n=len(y)
    rmse=float(np.sqrt(sse/n))
    aic=float(n*np.log(sse/n)+2*k)
    aicc=float(aic + (2*k*(k+1)/(n-k-1))) if n>k+1 else float("inf")
    return rmse,aicc

traj=[]
families=["linear","quadratic","log","exponential","power","saturating","piecewise"]
for g,z in seq.items():
    x=z.Pass.to_numpy(float)
    for ch,col in zip(TARGET_SHORT,TARGETS):
        y=z[col].to_numpy(float)
        for fam in families:
            try:
                rmse,aicc=fit_family(x,y,fam)
                traj.append({"Group_VW":g,"channel":ch,"family":fam,"n":len(x),"rmse_N":rmse,"AICc":aicc})
            except Exception as e:
                traj.append({"Group_VW":g,"channel":ch,"family":fam,"n":len(x),"rmse_N":np.nan,"AICc":np.inf})
trajdf=pd.DataFrame(traj)
trajdf["AICc_rank"]=trajdf.groupby(["Group_VW","channel"]).AICc.rank(method="min")
trajdf.to_csv(TABLE/"04_1_trajectory_families.csv",index=False)
wins=trajdf[trajdf.AICc_rank==1].groupby("family").size().sort_values(ascending=False)
item_result("4.1","T1-T6 trajectory family analysis","COMPLETE_EXPLORATORY",
            int(wins.iloc[0]) if len(wins) else 0, "Primary metric is number of group×channel AICc wins.", ["tables/04_1_trajectory_families.csv"])

# -------------------------------------------------------------------------
phase("ITEM_4_2_MEMORY_DEPTH")
common=trans[(trans.TargetPass>=4) & trans[["lag1_LC1","lag1_LC5","lag2_LC1","lag2_LC5","lag3_LC1","lag3_LC5"]].notna().all(axis=1)].copy()
md=[]
sets=[
 ("lag1",cond_cols+["lag1_LC1","lag1_LC5"]),
 ("lag1_2",cond_cols+["lag1_LC1","lag1_LC5","lag2_LC1","lag2_LC5"]),
 ("lag1_2_3",cond_cols+["lag1_LC1","lag1_LC5","lag2_LC1","lag2_LC5","lag3_LC1","lag3_LC5"]),
]
yc=common[["y_LC1","y_LC5"]].to_numpy(float); gc=common.Group_VW.to_numpy()
for name,cols in sets:
    p,m,a=grouped_oof_nested_ridge(common[cols].to_numpy(float),yc,gc)
    md.append({"memory":name,"n":len(common),"joint_group_mae_N":m})
mddf=pd.DataFrame(md); mddf.to_csv(TABLE/"04_2_memory_depth.csv",index=False)
item_result("4.2","Memory depth","COMPLETE_EXPLORATORY",float(mddf.joint_group_mae_N.min()),"Common complete-lag subset used for fair comparison.",["tables/04_2_memory_depth.csv"])

# -------------------------------------------------------------------------
phase("ITEM_4_3_LEAVE_ONE_PASS_OUT")
lopo=[]
X=trans[lag1_cols].to_numpy(float); Y=Ytrans
for tp in sorted(trans.TargetPass.unique()):
    te=(trans.TargetPass.to_numpy()==tp); tr=~te
    m=fixed_ridge(10.0); m.fit(X[tr],Y[tr]); pp=m.predict(X[te])
    lopo.append({"held_out_pass":int(tp),"n_test":int(te.sum()),"joint_MAE_N":float(mean_absolute_error(Y[te],pp))})
lopodf=pd.DataFrame(lopo); lopodf.to_csv(TABLE/"04_3_leave_one_pass_out.csv",index=False)
item_result("4.3","Leave-One-Pass-Out","COMPLETE_ROBUSTNESS",float(lopodf.joint_MAE_N.mean()),"Pass-generalization stress test; not the primary grouped generalization estimate.",["tables/04_3_leave_one_pass_out.csv"])

# -------------------------------------------------------------------------
phase("ITEM_4_4_ALL_PAIRWISE")
pair_rows=[]
for s in range(1,6):
    for t in range(s+1,7):
        rr=[]
        for g,z in seq.items():
            by={int(r.Pass):r for _,r in z.iterrows()}
            if s in by and t in by:
                a,b=by[s],by[t]
                rr.append({"g":g,"Speed":float(a.Speed),"Load":float(a.Load),"source_pass":s,"target_pass":t,
                           "src1":float(a[TARGETS[0]]),"src5":float(a[TARGETS[1]]),
                           "y1":float(b[TARGETS[0]]),"y5":float(b[TARGETS[1]])})
        q=pd.DataFrame(rr)
        if len(q)>=5 and q.g.nunique()>=5:
            xx=q[["Speed","Load","source_pass","target_pass","src1","src5"]].to_numpy(float)
            yy=q[["y1","y5"]].to_numpy(float); gg=q.g.to_numpy()
            pp,mae=grouped_oof_fixed(xx,yy,gg,10.0)
            xb=q[["Speed","Load","source_pass","target_pass"]].to_numpy(float)
            _,base=grouped_oof_fixed(xb,yy,gg,10.0)
            pair_rows.append({"source_pass":s,"target_pass":t,"horizon":t-s,"n":len(q),
                              "condition_only_mae_N":base,"source_state_mae_N":mae,
                              "predictive_gain_N":base-mae})
pairdf=pd.DataFrame(pair_rows); pairdf.to_csv(TABLE/"04_4_all_15_pass_pairs.csv",index=False)
item_result("4.4","All pairwise pass relations","COMPLETE_EXPLORATORY",float(pairdf.predictive_gain_N.mean()),"Mean MAE gain from source pass state across valid pairwise relations.",["tables/04_4_all_15_pass_pairs.csv"])

# -------------------------------------------------------------------------
phase("ITEM_4_5_RAW_VS_SUMMARY")
raw_cols=cond_cols+[f"lag{k}_{ch}" for k in range(1,6) for ch in TARGET_SHORT]
summary_cols=cond_cols+[f"hist_{ch}_{s}" for ch in TARGET_SHORT for s in ["mean","std","min","max","slope","cumdelta","lastdelta","curvature","weighted"]]
rvs=[]
for name,cols in [("raw_lag_slots",raw_cols),("summary_history",summary_cols),("raw_plus_summary",raw_cols+summary_cols[3:])]:
    p,m,a=grouped_oof_nested_ridge(trans[cols].to_numpy(float),Ytrans,Gtrans)
    rvs.append({"representation":name,"joint_group_mae_N":m})
rvsdf=pd.DataFrame(rvs); rvsdf.to_csv(TABLE/"04_5_raw_vs_summary_history.csv",index=False)
item_result("4.5","Raw history versus summarized history","COMPLETE_EXPLORATORY",float(rvsdf.joint_group_mae_N.min()),None,["tables/04_5_raw_vs_summary_history.csv"])

# -------------------------------------------------------------------------
phase("ITEM_4_6_SHARED_HEAD")
shared_pred,shared_mae,_=grouped_oof_nested_ridge(trans[lag1_cols].to_numpy(float),Ytrans,Gtrans)
sep_pred=np.full_like(Ytrans,np.nan)
for g in np.unique(Gtrans):
    te=Gtrans==g; tr=~te
    for tp in sorted(trans.TargetPass.unique()):
        tep=te & (trans.TargetPass.to_numpy()==tp)
        if not tep.any(): continue
        trp=tr & (trans.TargetPass.to_numpy()==tp)
        if trp.sum()<3: trp=tr
        m=fixed_ridge(10.0); m.fit(trans.loc[trp,lag1_cols].to_numpy(float),Ytrans[trp])
        sep_pred[tep]=m.predict(trans.loc[tep,lag1_cols].to_numpy(float))
sep_mae=joint_group_mae(Ytrans,sep_pred,Gtrans)
head=pd.DataFrame([{"architecture":"shared_model_with_pass_feature","joint_group_mae_N":shared_mae},
                   {"architecture":"pass_specific_heads","joint_group_mae_N":sep_mae}])
head.to_csv(TABLE/"04_6_shared_vs_pass_heads.csv",index=False)
item_result("4.6","Shared backbone versus pass-specific heads","COMPLETE_EXPLORATORY",float(head.joint_group_mae_N.min()),None,["tables/04_6_shared_vs_pass_heads.csv"])

# -------------------------------------------------------------------------
phase("ITEM_4_7_RECURSIVE")
rec_rows=[]
for held in sorted(seq):
    train=trans[trans.Group_VW!=held].copy()
    Xtr=train[lag1_cols].to_numpy(float); Ydelta=train[["y_LC1","y_LC5"]].to_numpy(float)-train[["prev_LC1","prev_LC5"]].to_numpy(float)
    a=select_alpha_train(Xtr,Ydelta,train.Group_VW.to_numpy())
    m=fixed_ridge(a); m.fit(Xtr,Ydelta)
    z=seq[held]; by={int(r.Pass):r for _,r in z.iterrows()}
    start=min(by)
    state=np.array([float(by[start][TARGETS[0]]),float(by[start][TARGETS[1]])])
    for t in range(start+1,7):
        x=np.array([[float(z.Speed.iloc[0]),float(z.Load.iloc[0]),float(t),state[0],state[1]]])
        state=state+m.predict(x)[0]
        if t in by:
            truth=np.array([float(by[t][TARGETS[0]]),float(by[t][TARGETS[1]])])
            rec_rows.append({"Group_VW":held,"pass":t,"horizon":t-start,
                             "LC1_true":truth[0],"LC1_pred":state[0],"LC5_true":truth[1],"LC5_pred":state[1],
                             "joint_abs_error_N":float(np.mean(np.abs(truth-state)))})
recdf=pd.DataFrame(rec_rows); recdf.to_csv(TABLE/"04_7_recursive_forecasting.csv",index=False)
item_result("4.7","Recursive chained forecasting","COMPLETE_EXPLORATORY",float(recdf.joint_abs_error_N.mean()),"Mean recursive absolute error across observed future passes.",["tables/04_7_recursive_forecasting.csv"])

# -------------------------------------------------------------------------
phase("ITEM_4_8_MISSING_PASS")
recon_train=[]
for g,z in seq.items():
    by={int(r.Pass):r for _,r in z.iterrows()}
    for t in range(2,6):
        if (t-1) in by and t in by and (t+1) in by:
            a,b,c=by[t-1],by[t],by[t+1]
            recon_train.append({"g":g,"t":t,"Speed":float(b.Speed),"Load":float(b.Load),
                                "p1":float(a[TARGETS[0]]),"p5":float(a[TARGETS[1]]),
                                "n1":float(c[TARGETS[0]]),"n5":float(c[TARGETS[1]]),
                                "y1":float(b[TARGETS[0]]),"y5":float(b[TARGETS[1]])})
rtdf=pd.DataFrame(recon_train)
xr=rtdf[["Speed","Load","t","p1","p5","n1","n5"]].to_numpy(float); yr=rtdf[["y1","y5"]].to_numpy(float); gr=rtdf.g.to_numpy()
pr,mr=grouped_oof_fixed(xr,yr,gr,10.0)
# Fit all for V2W3T2 estimate.
rm=fixed_ridge(10.0); rm.fit(xr,yr)
v2t1=df[df.Run_ID=="V2W3T1"].iloc[0]; v2t3=df[df.Run_ID=="V2W3T3"].iloc[0]
xmiss=np.array([[float(v2t1.Speed),float(v2t1.Load),2,float(v2t1[TARGETS[0]]),float(v2t1[TARGETS[1]]),float(v2t3[TARGETS[0]]),float(v2t3[TARGETS[1]])]])
est_t2=rm.predict(xmiss)[0]
# Reverse T1 reconstruction from T2 across groups with both.
rev=[]
for g,z in seq.items():
    by={int(r.Pass):r for _,r in z.iterrows()}
    if 1 in by and 2 in by:
        a,b=by[1],by[2]
        rev.append({"g":g,"Speed":float(a.Speed),"Load":float(a.Load),"n1":float(b[TARGETS[0]]),"n5":float(b[TARGETS[1]]),"y1":float(a[TARGETS[0]]),"y5":float(a[TARGETS[1]])})
revdf=pd.DataFrame(rev); xm=revdf[["Speed","Load","n1","n5"]].to_numpy(float); ym=revdf[["y1","y5"]].to_numpy(float)
revmodel=fixed_ridge(10.0); revmodel.fit(xm,ym)
missing_est=[{"Run_ID":"V2W3T2","LC1_est_N":float(est_t2[0]),"LC5_est_N":float(est_t2[1]),"method":"bidirectional interior reconstruction"}]
for rid in ["V1W1T1","V3W2T1"]:
    g=rid[:4]; t2=df[df.Run_ID==g+"T2"].iloc[0]
    xx=np.array([[float(t2.Speed),float(t2.Load),float(t2[TARGETS[0]]),float(t2[TARGETS[1]])]])
    ee=revmodel.predict(xx)[0]
    missing_est.append({"Run_ID":rid,"LC1_est_N":float(ee[0]),"LC5_est_N":float(ee[1]),"method":"reverse T1 from observed T2"})
pd.DataFrame(missing_est).to_csv(TABLE/"04_8_missing_slot_estimates.csv",index=False)
pd.DataFrame({"y1":yr[:,0],"p1":pr[:,0],"y5":yr[:,1],"p5":pr[:,1],"g":gr}).to_csv(TABLE/"04_8_reconstruction_oof.csv",index=False)
item_result("4.8","Missing-pass reconstruction","COMPLETE_EXPLORATORY",float(mr),"Missing design-slot values are estimates only, never treated as observed data.",["tables/04_8_missing_slot_estimates.csv","tables/04_8_reconstruction_oof.csv"])

# -------------------------------------------------------------------------
phase("ITEM_4_9_LC_ABLATION")
abl=[]
for name,cols in [
    ("conditions_only",cond_cols),
    ("prev_LC1_only",cond_cols+["prev_LC1"]),
    ("prev_LC5_only",cond_cols+["prev_LC5"]),
    ("both_channels",lag1_cols),
]:
    pp,mm,aa=grouped_oof_nested_ridge(trans[cols].to_numpy(float),Ytrans,Gtrans)
    abl.append({"history_input":name,"joint_group_mae_N":mm})
abldf=pd.DataFrame(abl); abldf.to_csv(TABLE/"04_9_lc1_lc5_ablation.csv",index=False)
item_result("4.9","LC1/LC5 ablation and fusion","COMPLETE_EXPLORATORY",float(abldf.joint_group_mae_N.min()),None,["tables/04_9_lc1_lc5_ablation.csv"])

# -------------------------------------------------------------------------
phase("ITEM_4_10_HISTORY_SHUFFLE")
Xbase=trans[lag1_cols].to_numpy(float)
truep,truemae=grouped_oof_fixed(Xbase,Ytrans,Gtrans,10.0)
rng=np.random.default_rng(SEED)
shuffle_scores=[]
for rep in range(100):
    pp=np.full_like(Ytrans,np.nan)
    for g in np.unique(Gtrans):
        tr=Gtrans!=g; te=Gtrans==g
        xt=Xbase[tr].copy()
        perm=rng.permutation(xt.shape[0])
        xt[:,3:5]=xt[perm,3:5]
        m=fixed_ridge(10.0); m.fit(xt,Ytrans[tr]); pp[te]=m.predict(Xbase[te])
    shuffle_scores.append(joint_group_mae(Ytrans,pp,Gtrans))
sh=pd.DataFrame({"shuffle_rep":np.arange(100),"joint_group_mae_N":shuffle_scores})
sh.to_csv(TABLE/"04_10_history_shuffle.csv",index=False)
shmeta={"true_history_mae_N":truemae,"shuffle_mean_mae_N":float(np.mean(shuffle_scores)),
        "shuffle_p05":float(np.quantile(shuffle_scores,.05)),"shuffle_p95":float(np.quantile(shuffle_scores,.95)),
        "fraction_shuffle_better_or_equal":float(np.mean(np.asarray(shuffle_scores)<=truemae))}
dump(ITEM/"04_10_history_shuffle_meta.json",shmeta)
item_result("4.10","History-shuffle negative control","COMPLETE_NEGATIVE_CONTROL",float(np.mean(shuffle_scores)-truemae),"Positive = degradation after destroying history alignment.",["tables/04_10_history_shuffle.csv"])

# -------------------------------------------------------------------------
phase("ITEM_4_11_MISSING_HISTORY")
miss_rows=[]
for scenario in ["normal","drop_LC1","drop_LC5","drop_both"]:
    pp=np.full_like(Ytrans,np.nan)
    for g in np.unique(Gtrans):
        tr=Gtrans!=g; te=Gtrans==g
        m=fixed_ridge(10.0); m.fit(Xbase[tr],Ytrans[tr])
        xx=Xbase[te].copy()
        if scenario in ("drop_LC1","drop_both"): xx[:,3]=np.nan
        if scenario in ("drop_LC5","drop_both"): xx[:,4]=np.nan
        pp[te]=m.predict(xx)
    miss_rows.append({"scenario":scenario,"joint_group_mae_N":joint_group_mae(Ytrans,pp,Gtrans)})
missdf=pd.DataFrame(miss_rows); missdf.to_csv(TABLE/"04_11_missing_history_robustness.csv",index=False)
item_result("4.11","Robustness to missing history","COMPLETE_ROBUSTNESS",float(missdf.joint_group_mae_N.max()),"Worst-case joint grouped MAE among missing-history scenarios.",["tables/04_11_missing_history_robustness.csv"])

# -------------------------------------------------------------------------
phase("ITEM_4_12_MULTI_HORIZON")
mh=[]
for h in range(1,6):
    rr=[]
    for g,z in seq.items():
        by={int(r.Pass):r for _,r in z.iterrows()}
        for s in range(1,7-h):
            t=s+h
            if s in by and t in by:
                a,b=by[s],by[t]
                rr.append({"g":g,"Speed":float(a.Speed),"Load":float(a.Load),"source_pass":s,"target_pass":t,"h":h,
                           "src1":float(a[TARGETS[0]]),"src5":float(a[TARGETS[1]]),"y1":float(b[TARGETS[0]]),"y5":float(b[TARGETS[1]])})
    q=pd.DataFrame(rr)
    if len(q)>=5:
        xx=q[["Speed","Load","source_pass","target_pass","h","src1","src5"]].to_numpy(float); yy=q[["y1","y5"]].to_numpy(float); gg=q.g.to_numpy()
        pp,mm=grouped_oof_fixed(xx,yy,gg,10.0)
        mh.append({"horizon":h,"n":len(q),"joint_group_mae_N":mm})
mhdf=pd.DataFrame(mh); mhdf.to_csv(TABLE/"04_12_multi_horizon.csv",index=False)
item_result("4.12","Multi-horizon / early-history forecasting","COMPLETE_EXPLORATORY",float(mhdf.joint_group_mae_N.mean()),None,["tables/04_12_multi_horizon.csv"])

# -------------------------------------------------------------------------
phase("ITEM_5_1_PASS_GRAPH")
pass_edges=pairdf.copy()
pass_edges["edge_weight_gain_N"]=pass_edges.predictive_gain_N.clip(lower=0.0)
pass_edges.to_csv(TABLE/"05_1_pass_graph_edges.csv",index=False)
item_result("5.1","Pass graph","COMPLETE_EXPLORATORY",float(pass_edges.edge_weight_gain_N.sum()),"Edge weights = non-negative grouped predictive gain over conditions-only.",["tables/05_1_pass_graph_edges.csv"])

# -------------------------------------------------------------------------
phase("ITEM_5_2_LC_GRAPH")
nodes=[(t,ch) for t in range(1,7) for ch in TARGET_SHORT]
node_vectors={}
for t,ch in nodes:
    col=TARGETS[0] if ch=="LC1" else TARGETS[1]
    vals={}
    for g,z in seq.items():
        q=z[z.Pass==t]
        if len(q): vals[g]=float(q.iloc[0][col])
    node_vectors[(t,ch)]=vals

lc_edges=[]
for i,a in enumerate(nodes):
    for b in nodes[i+1:]:
        common_groups=sorted(set(node_vectors[a]) & set(node_vectors[b]))
        if len(common_groups)<5: continue
        va=np.array([node_vectors[a][g] for g in common_groups],float)
        vb=np.array([node_vectors[b][g] for g in common_groups],float)
        C=np.array([[1.0, float(group_conditions.loc[g,"Speed"]), float(group_conditions.loc[g,"Load"])] for g in common_groups])
        ra=va-C@np.linalg.lstsq(C,va,rcond=None)[0]
        rb=vb-C@np.linalg.lstsq(C,vb,rcond=None)[0]
        if np.std(ra)<1e-12 or np.std(rb)<1e-12: r=0.0
        else: r=float(np.corrcoef(ra,rb)[0,1])
        lc_edges.append({"node_a":f"T{a[0]}-{a[1]}","node_b":f"T{b[0]}-{b[1]}","n":len(common_groups),
                         "partial_corr_speed_load":r,"weight_abs_partial_corr":abs(r)})
lcedf=pd.DataFrame(lc_edges); lcedf.to_csv(TABLE/"05_2_lc_graph_edges.csv",index=False)
item_result("5.2","LC1/LC5 12-node graph","COMPLETE_EXPLORATORY",float(lcedf.weight_abs_partial_corr.mean()),"Weights are absolute residual correlations after linear Speed/Load adjustment.",["tables/05_2_lc_graph_edges.csv"])

# -------------------------------------------------------------------------
phase("ITEM_5_3_SPECTRAL")
node_names=[f"T{t}-{ch}" for t,ch in nodes]; idx={n:i for i,n in enumerate(node_names)}
W=np.zeros((len(nodes),len(nodes)),float)
for _,r in lcedf.iterrows():
    i=idx[r.node_a]; j=idx[r.node_b]; W[i,j]=W[j,i]=float(r.weight_abs_partial_corr)
D=np.diag(W.sum(axis=1)); L=D-W
evals,evecs=np.linalg.eigh(L)
spec=pd.DataFrame({"eigen_index":np.arange(len(evals)),"eigenvalue":evals})
spec.to_csv(TABLE/"05_3_laplacian_spectrum.csv",index=False)
emb=pd.DataFrame({"node":node_names,"eigvec_2":evecs[:,1] if len(evals)>1 else 0.0,
                  "eigvec_3":evecs[:,2] if len(evals)>2 else 0.0})
emb.to_csv(TABLE/"05_3_spectral_embedding.csv",index=False)
gap=float(evals[1]) if len(evals)>1 else 0.0
item_result("5.3","Graph spectral analysis","COMPLETE_EXPLORATORY",gap,"Algebraic connectivity / first non-zero Laplacian eigenvalue.",["tables/05_3_laplacian_spectrum.csv","tables/05_3_spectral_embedding.csv"])

# -------------------------------------------------------------------------
phase("ITEM_5_4_EFFECTIVE_RESISTANCE")
connected=bool(gap>1e-8)
if connected:
    Lp=np.linalg.pinv(L)
    R=np.zeros_like(L)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            R[i,j]=Lp[i,i]+Lp[j,j]-2*Lp[i,j]
    rdf=pd.DataFrame(R,index=node_names,columns=node_names)
    rdf.to_csv(TABLE/"05_4_effective_resistance.csv")
    metric=float(np.mean(R[np.triu_indices_from(R,1)]))
    status="COMPLETE_DESCRIPTIVE"
    note="Graph connected; effective resistance reported as descriptive network distance, not physical electrical resistance."
else:
    pd.DataFrame().to_csv(TABLE/"05_4_effective_resistance.csv",index=False)
    metric=None; status="NOT_INTERPRETABLE_DISCONNECTED_GRAPH"
    note="Graph disconnected under available-data weights; no effective-resistance interpretation."
item_result("5.4","Conditional/effective resistance","%s"%status,metric,note,["tables/05_4_effective_resistance.csv"])

# -------------------------------------------------------------------------
phase("ITEM_5_5_INFORMATION_DESIGN")
groups=group_conditions.index.tolist()
Xg=np.array([[1.0,float(group_conditions.loc[g,"Speed"]),float(group_conditions.loc[g,"Load"]),
              float(group_conditions.loc[g,"Speed"])*float(group_conditions.loc[g,"Load"]),
              float(group_conditions.loc[g,"Speed"])**2,float(group_conditions.loc[g,"Load"])**2] for g in groups])
selected=[]; design_rows=[]
remaining=list(range(len(groups)))
for step in range(len(groups)):
    best=None
    for j in remaining:
        cand=selected+[j]
        M=Xg[cand].T@Xg[cand]+1e-6*np.eye(Xg.shape[1])
        val=float(np.linalg.slogdet(M)[1])
        if best is None or val>best[0]: best=(val,j)
    selected.append(best[1]); remaining.remove(best[1])
    design_rows.append({"budget":step+1,"added_group":groups[best[1]],"selected_groups":";".join(groups[k] for k in selected),"logdet":best[0]})
designdf=pd.DataFrame(design_rows); designdf.to_csv(TABLE/"05_5_d_optimal_group_order.csv",index=False)
item_result("5.5","Information-optimal experimental subset selection","COMPLETE_EXPLORATORY",float(designdf.logdet.iloc[-1]),"Greedy D-optimal ordering over the nine existing Speed×Load groups.",["tables/05_5_d_optimal_group_order.csv"])

# -------------------------------------------------------------------------
phase("ITEM_5_6_SPECTRAL_REGULARIZATION")
# Run-level model: global Speed/Load effects + pass-specific intercepts with chain-Laplacian penalty.
run_groups=df.Group_VW.to_numpy(); yrun=df[TARGETS].to_numpy(float)
pass_onehot=np.eye(6)[df.Pass.to_numpy(int)-1]
Xrun=np.c_[np.ones(len(df)),df.Speed.to_numpy(float),df.Load.to_numpy(float),pass_onehot]
chainW=np.zeros((6,6))
for i in range(5): chainW[i,i+1]=chainW[i+1,i]=1.0
chainL=np.diag(chainW.sum(axis=1))-chainW
P0=np.zeros((Xrun.shape[1],Xrun.shape[1])); P0[3:,3:]=chainL
def lap_fit_predict(Xtr,Ytr,Xte,lam):
    A=Xtr.T@Xtr + lam*P0 + 1e-6*np.eye(Xtr.shape[1])
    B=np.linalg.solve(A,Xtr.T@Ytr)
    return Xte@B
lap_rows=[]; pred_lap=np.full_like(yrun,np.nan); chosen_lam={}
for g in np.unique(run_groups):
    tr=run_groups!=g; te=run_groups==g
    ug=np.unique(run_groups[tr]); best=None
    for lam in [0.0,0.1,1.0,10.0,100.0]:
        pp=np.full_like(yrun[tr],np.nan); gtr=run_groups[tr]; Xtr=Xrun[tr]; Ytr=yrun[tr]
        for vg in ug:
            a=gtr!=vg; b=gtr==vg
            pp[b]=lap_fit_predict(Xtr[a],Ytr[a],Xtr[b],lam)
        sc=joint_group_mae(Ytr,pp,gtr)
        if best is None or (sc,lam)<best: best=(sc,lam)
    lam=best[1]; chosen_lam[str(g)]=lam
    pred_lap[te]=lap_fit_predict(Xrun[tr],yrun[tr],Xrun[te],lam)
lap_mae=joint_group_mae(yrun,pred_lap,run_groups)
# unpenalized equivalent
pred_un=np.full_like(yrun,np.nan)
for g in np.unique(run_groups):
    tr=run_groups!=g;te=run_groups==g
    pred_un[te]=lap_fit_predict(Xrun[tr],yrun[tr],Xrun[te],0.0)
un_mae=joint_group_mae(yrun,pred_un,run_groups)
lapdf=pd.DataFrame([{"method":"unpenalized_pass_effect","joint_group_mae_N":un_mae},
                    {"method":"chain_laplacian_regularized","joint_group_mae_N":lap_mae}])
lapdf.to_csv(TABLE/"05_6_spectral_regularization.csv",index=False)
dump(ITEM/"05_6_chosen_lambda.json",chosen_lam)
item_result("5.6","Spectral/Laplacian regularization","COMPLETE_EXPLORATORY",float(lap_mae-un_mae),"Negative = regularization improvement on run-level grouped CV.",["tables/05_6_spectral_regularization.csv"])

# -------------------------------------------------------------------------
phase("ITEM_5_7_SPECTRAL_VS_CURRENT")
# Pass chain eigenvectors as leakage-free structural encoding in next-pass model.
cevals,cevecs=np.linalg.eigh(chainL)
def spectral_pass_feats(pass_values):
    arr=[]
    for t in pass_values.astype(int):
        arr.append([cevecs[t-1,1],cevecs[t-1,2]])
    return np.asarray(arr,float)
Xs=np.c_[trans[["Speed","Load","prev_LC1","prev_LC5"]].to_numpy(float), spectral_pass_feats(trans.TargetPass.to_numpy())]
Xn=trans[lag1_cols].to_numpy(float)
_,spectral_mae,_=grouped_oof_nested_ridge(Xs,Ytrans,Gtrans)
_,numeric_mae,_=grouped_oof_nested_ridge(Xn,Ytrans,Gtrans)
cmp=pd.DataFrame([
    {"method":"ridge_numeric_pass","joint_group_mae_N":numeric_mae,"comparison_scope":"same V6 Ridge protocol"},
    {"method":"ridge_spectral_pass_embedding","joint_group_mae_N":spectral_mae,"comparison_scope":"same V6 Ridge protocol"},
    {"method":"V5_locked_champion_reference","joint_group_mae_N":LOCKED_V5_JOINT_GROUP_MAE_N,"comparison_scope":"reference only; frozen prior protocol"},
])
cmp.to_csv(TABLE/"05_7_spectral_vs_current.csv",index=False)
item_result("5.7","Spectral methods versus current model","COMPLETE_EXPLORATORY",float(spectral_mae-numeric_mae),"Negative = spectral pass encoding improves same-protocol Ridge; V5 is reference only.",["tables/05_7_spectral_vs_current.csv"])

# -------------------------------------------------------------------------
phase("ITEM_5_8_TSP")
coords=np.array([[float(group_conditions.loc[g,"Speed"]),float(group_conditions.loc[g,"Load"])] for g in groups])
coords=(coords-coords.mean(axis=0))/np.maximum(coords.std(axis=0),1e-12)
dist=np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(axis=2))
# Exact minimum Hamiltonian path by subset DP, free start/end.
n=len(groups); dp={}
for i in range(n): dp[(1<<i,i)]=(0.0,[i])
for mask in range(1,1<<n):
    for j in range(n):
        key=(mask,j)
        if key not in dp: continue
        cost,path=dp[key]
        for k in range(n):
            if mask&(1<<k): continue
            nk=(mask|1<<k,k); cand=(cost+dist[j,k],path+[k])
            if nk not in dp or cand[0]<dp[nk][0]: dp[nk]=cand
full=(1<<n)-1
best=min((v for (m,j),v in dp.items() if m==full),key=lambda x:x[0])
order=[groups[i] for i in best[1]]
naive=groups
naive_cost=sum(dist[groups.index(naive[i]),groups.index(naive[i+1])] for i in range(n-1))
tsp={"optimal_open_path":order,"standardized_setup_cost":float(best[0]),"naive_group_order":naive,
     "naive_cost":float(naive_cost),"relative_cost_reduction_pct":float(100*(naive_cost-best[0])/naive_cost) if naive_cost else 0.0,
     "scope":"logistics/setup order only; never a force-prediction model"}
dump(TABLE/"05_8_tsp_setup_order.json",tsp)
item_result("5.8","TSP for experimental execution order","COMPLETE_LOGISTICS_ONLY",float(tsp["relative_cost_reduction_pct"]),tsp["scope"],["tables/05_8_tsp_setup_order.json"])

# -------------------------------------------------------------------------
phase("MASTER_SUMMARY")
summary=pd.DataFrame(item_rows)
if len(summary)!=23:
    raise RuntimeError(f"EXPECTED_23_ITEMS_GOT_{len(summary)}")
summary.to_csv(ROOT/"23_ITEM_SUMMARY.csv",index=False)
master={
    "schema":"soilbin.q1.v6.23-extension",
    "created_utc":datetime.now(timezone.utc).isoformat(),
    "post_lock_exploratory":True,
    "does_not_supersede_v5":True,
    "v5_locked_champion":LOCKED_V5_CHAMPION,
    "v5_locked_joint_group_mae_N":LOCKED_V5_JOINT_GROUP_MAE_N,
    "n_valid_runs":51,"n_groups":9,"n_consecutive_transitions":41,
    "missing_design_slots":sorted(EXPECTED_MISSING),
    "response_status":RESPONSE_STATUS,"depth_mapping_status":DEPTH_STATUS,
    "items_total":23,"items_emitted":len(item_rows),
    "external_labels_used_for_model_selection":False,
    "random_split_used":False,
    "primary_validation_principle":"group-held-out whenever predictive generalization is estimated",
    "limitations":[
        "Small sample: 51 runs and 41 consecutive transitions.",
        "LC1/LC5 physical depth assignment remains provisional.",
        "No effective sensor area/direct stress calibration; force-to-stress conversion is not attempted.",
        "Item 3 uses the frozen V3 prior-art registry/corroboration snapshot; live literature search is outside this Kaggle compute.",
        "All new analyses are post-lock exploratory and require independent confirmation before changing V5 claims."
    ],
    "files":sorted([p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()])
}
dump(ROOT/"MASTER_MANIFEST_V6_23.json",master)

# A compact figure comparing key predictive experiments.
plot_parts=[]
for label,table_path,namecol,valcol in [
    ("memory",TABLE/"02_multichannel_memory.csv","representation","joint_group_mae_N"),
    ("lc_ablation",TABLE/"04_9_lc1_lc5_ablation.csv","history_input","joint_group_mae_N"),
    ("spectral",TABLE/"05_7_spectral_vs_current.csv","method","joint_group_mae_N"),
]:
    q=pd.read_csv(table_path)
    for _,r in q.iterrows():
        plot_parts.append({"label":f"{label}:{r[namecol]}","mae":float(r[valcol])})
pdf=pd.DataFrame(plot_parts)
plt.figure(figsize=(12,6))
plt.bar(np.arange(len(pdf)),pdf.mae)
plt.xticks(np.arange(len(pdf)),pdf.label,rotation=65,ha="right",fontsize=8)
plt.ylabel("Joint group-equal MAE (N)")
plt.tight_layout()
plt.savefig(FIG/"v6_23_key_model_comparisons.png",dpi=180)
plt.close()

report = [
"# SoilBin V6 — 23-item exploratory extension",
"",
"**This is a post-lock exploratory analysis. It does not supersede V5.**",
"",
f"- Valid runs: 51; V×W groups: 9; consecutive transitions: 41.",
f"- Locked V5 reference: {LOCKED_V5_CHAMPION}, joint group-equal MAE = {LOCKED_V5_JOINT_GROUP_MAE_N:.3f} N.",
"- Random splits: not used for predictive generalization estimates.",
"- External labels: not used for model selection.",
"- Force-to-stress conversion: not attempted; effective sensor area remains unknown.",
"- LC depth mapping: provisional only.",
"",
"## 23-item status",
]
for r in item_rows:
    report.append(f"- {r['item_id']} — {r['title']}: **{r['status']}**; primary metric = {r['primary_metric']}")
(ROOT/"FINAL_REPORT_V6_23.md").write_text("\n".join(report),encoding="utf-8")

results_compact = {
    "schema":"soilbin.q1.v6.23-extension.results",
    "items":item_rows,
    "calibration":cal.to_dict(orient="records"),
    "multichannel_memory":mem.to_dict(orient="records"),
    "memory_depth":mddf.to_dict(orient="records"),
    "leave_one_pass_out":lopodf.to_dict(orient="records"),
    "pairwise_pass_relations":pairdf.to_dict(orient="records"),
    "raw_vs_summary":rvsdf.to_dict(orient="records"),
    "shared_vs_pass_heads":head.to_dict(orient="records"),
    "recursive_mean_abs_error_N":float(recdf.joint_abs_error_N.mean()),
    "missing_pass_reconstruction_group_mae_N":float(mr),
    "missing_slot_estimates":missing_est,
    "lc_ablation":abldf.to_dict(orient="records"),
    "history_shuffle":shmeta,
    "missing_history_robustness":missdf.to_dict(orient="records"),
    "multi_horizon":mhdf.to_dict(orient="records"),
    "pass_graph_total_positive_gain_N":float(pass_edges.edge_weight_gain_N.sum()),
    "lc_graph_mean_abs_partial_corr":float(lcedf.weight_abs_partial_corr.mean()),
    "spectral_gap":gap,
    "effective_resistance_interpretable":connected,
    "d_optimal_order":designdf.to_dict(orient="records"),
    "spectral_regularization":lapdf.to_dict(orient="records"),
    "spectral_vs_current":cmp.to_dict(orient="records"),
    "tsp":tsp,
    "v5_locked_reference":{"champion":LOCKED_V5_CHAMPION,"joint_group_mae_N":LOCKED_V5_JOINT_GROUP_MAE_N},
}
dump(ROOT/"RESULTS_V6_23.json", results_compact)

zip_path = shutil.make_archive("/kaggle/working/SoilBin_Q1_Q2_V6_23_EXTENSION_20260928","zip",ROOT)
phase("COMPLETE")
print("SOILBIN_V6_23_COMPLETE", json.dumps({
    "items":23,
    "zip":zip_path,
    "locked_v5_reference_mae_N":LOCKED_V5_JOINT_GROUP_MAE_N,
    "best_multichannel_memory_mae_N":float(mem.joint_group_mae_N.min()),
    "best_history_representation_mae_N":float(rvsdf.joint_group_mae_N.min()),
    "best_lc_ablation_mae_N":float(abldf.joint_group_mae_N.min()),
    "spectral_minus_numeric_ridge_mae_N":float(spectral_mae-numeric_mae),
    "history_shuffle_degradation_N":float(np.mean(shuffle_scores)-truemae),
}, sort_keys=True), flush=True)
