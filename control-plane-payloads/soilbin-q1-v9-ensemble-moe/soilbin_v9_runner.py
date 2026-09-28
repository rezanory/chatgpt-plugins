# SoilBin Q1/Q2 V9 — leakage-safe ensemble and mixture-of-experts ablation
# Post-lock exploratory. Frozen V5 is never overwritten.
from __future__ import annotations
import os, sys, json, math, hashlib, random, gzip, base64, itertools, shutil, warnings
from pathlib import Path
from io import StringIO
from datetime import datetime, timezone
from collections import Counter, defaultdict

SEED=20260914
random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED",str(SEED))
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np
np.random.seed(SEED)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

ROOT=Path("/kaggle/working/SOILBIN_V9")
TABLE=ROOT/"tables"; FIG=ROOT/"figures"; STATE=ROOT/"state"
for p in (ROOT,TABLE,FIG,STATE): p.mkdir(parents=True,exist_ok=True)

EXPECTED_MODEL_SHA="dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"
EXPECTED_MISSING={"V1W1T1","V2W3T2","V3W2T1"}
TARGET_COLS=["LC1_peak_magnitude_N_delta","LC5_peak_magnitude_N_delta"]
TARGET_NAMES=["LC1_proxy_N","LC5_proxy_N"]
FROZEN_V5=24.28878365273039
FROZEN_V5_LC1=12.785420282422
FROZEN_V5_LC5=35.79214702303878

def phase(s): print("CGP_PHASE:"+s,flush=True)
def dump(p,o): Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if hasattr(v,"item") else str(v)),encoding="utf-8")
def decode(payload):
    raw=gzip.decompress(base64.b64decode(payload.strip()))
    got=hashlib.sha256(raw).hexdigest()
    if got!=EXPECTED_MODEL_SHA: raise RuntimeError("SOURCE_FINGERPRINT_MISMATCH:"+got)
    return raw.decode("utf-8")
def group_mae(y,p,g):
    q=pd.DataFrame({"y":np.asarray(y,float),"p":np.asarray(p,float),"g":np.asarray(g)})
    return float(q.assign(e=lambda d:(d.y-d.p).abs()).groupby("g").e.mean().mean())
def joint_mae(y,p,g):
    return float(np.mean([group_mae(y[:,i],p[:,i],g) for i in range(2)]))
def metrics(y,p,g):
    out={}
    for i,t in enumerate(TARGET_NAMES):
        out[f"{t}_Group_MAE"]=group_mae(y[:,i],p[:,i],g)
        out[f"{t}_MAE"]=float(mean_absolute_error(y[:,i],p[:,i]))
        out[f"{t}_RMSE"]=float(mean_squared_error(y[:,i],p[:,i])**0.5)
        out[f"{t}_Bias"]=float(np.mean(p[:,i]-y[:,i]))
        out[f"{t}_R2"]=float(r2_score(y[:,i],p[:,i]))
    out["joint_Group_MAE"]=float(np.mean([out[f"{t}_Group_MAE"] for t in TARGET_NAMES]))
    return out
def make_tree(name,params):
    if name=="RandomForest":
        return Pipeline([("impute",SimpleImputer(strategy="median")),("m",RandomForestRegressor(
            n_estimators=250,max_depth=params["max_depth"],min_samples_leaf=params["min_samples_leaf"],
            random_state=SEED,n_jobs=1,max_features=1.0))])
    if name=="ExtraTrees":
        return Pipeline([("impute",SimpleImputer(strategy="median")),("m",ExtraTreesRegressor(
            n_estimators=250,max_depth=params["max_depth"],min_samples_leaf=params["min_samples_leaf"],
            random_state=SEED,n_jobs=1,max_features=1.0))])
    raise KeyError(name)

CANDIDATES=[
 ("RandomForest",{"max_depth":None,"min_samples_leaf":1}),
 ("RandomForest",{"max_depth":4,"min_samples_leaf":2}),
 ("ExtraTrees",{"max_depth":None,"min_samples_leaf":1}),
 ("ExtraTrees",{"max_depth":4,"min_samples_leaf":2}),
]

phase("DATA_FREEZE")
df=pd.read_csv(StringIO(decode(MODEL_CSV_GZ_B64)))
if len(df)!=51 or df.Run_ID.nunique()!=51: raise RuntimeError("EXPECTED_51_RUNS")
df["Group_VW"]=df.Run_ID.str.extract(r"^(V\dW\d)")[0]
df["Speed_kmh"]=df.Speed_level.astype(float)
df["Load_kN"]=df.Weight_level.astype(float)+1.0
df["Pass_T"]=df.Pass_T.astype(int)
all_ids={f"V{v}W{w}T{t}" for v in range(1,4) for w in range(1,4) for t in range(1,7)}
if all_ids-set(df.Run_ID)!=EXPECTED_MISSING: raise RuntimeError("DESIGN_SLOT_MISMATCH")

# Exact V5 41-pair table.
prev=df.copy(); prev["Pass_T"]+=1
prev=prev[["Group_VW","Pass_T"]+TARGET_COLS].rename(columns={TARGET_COLS[0]:"prev_LC1",TARGET_COLS[1]:"prev_LC5"})
hist=df.merge(prev,on=["Group_VW","Pass_T"],how="left",validate="one_to_one")
hist=hist[hist.prev_LC1.notna()&hist.prev_LC5.notna()].copy().reset_index(drop=True)
if len(hist)!=41: raise RuntimeError("EXPECTED_41_HISTORY_PAIRS")
hist["row_id"]=np.arange(len(hist),dtype=int)
prev_peaks=["prev_LC1","prev_LC5"]

# Lag features.
for lag in (2,3):
    s=df[["Group_VW","Pass_T"]+TARGET_COLS].copy(); s["Pass_T"]+=lag
    s=s.rename(columns={TARGET_COLS[0]:f"lag{lag}_LC1",TARGET_COLS[1]:f"lag{lag}_LC5"})
    hist=hist.merge(s,on=["Group_VW","Pass_T"],how="left",validate="one_to_one")
    hist[f"lag{lag}_available"]=hist[f"lag{lag}_LC1"].notna().astype(float)

# Historical summaries/state.
seq={g:z.sort_values("Pass_T") for g,z in df.groupby("Group_VW")}
rows=[]
for _,r in hist.iterrows():
    past=seq[r.Group_VW]; past=past[past.Pass_T<int(r.Pass_T)].sort_values("Pass_T")
    V=past[TARGET_COLS].to_numpy(float); P=past.Pass_T.to_numpy(float); last=V[-1]; first=V[0]
    def ew(a):
        z=V[0].copy()
        for v in V[1:]: z=a*v+(1-a)*z
        return z
    e25,e50,e75=ew(.25),ew(.5),ew(.75)
    vel=V[-1]-V[-2] if len(V)>=2 else np.zeros(2)
    accel=(V[-1]-V[-2])-(V[-2]-V[-3]) if len(V)>=3 else np.zeros(2)
    mins,maxs=V.min(0),V.max(0); rng=np.maximum(maxs-mins,1e-9)
    abs_path=np.abs(np.diff(V,axis=0)).sum(0) if len(V)>=2 else np.zeros(2)
    path=float(np.sqrt((np.diff(V,axis=0)**2).sum(1)).sum()) if len(V)>=2 else 0.0
    out={"row_id":int(r.row_id),"history_count":float(len(V)),"observed_fraction":float(len(V)/max(P[-1]-P[0]+1,1))}
    for ch,j in [("LC1",0),("LC5",1)]:
        vals=V[:,j]
        out.update({
            f"hist_{ch}_mean":float(vals.mean()),f"hist_{ch}_std":float(vals.std(ddof=0)),
            f"hist_{ch}_range":float(vals.max()-vals.min()),
            f"hist_{ch}_weighted":float(np.average(vals,weights=np.arange(1,len(vals)+1))),
            f"hist_{ch}_cumdelta":float(vals[-1]-vals[0]),
            f"hist_{ch}_lastdelta":float(vals[-1]-vals[-2]) if len(vals)>=2 else 0.0,
            f"hist_{ch}_slope":float(np.polyfit(P,vals,1)[0]) if len(vals)>=2 else 0.0,
            f"ew25_{ch}":float(e25[j]),f"ew50_{ch}":float(e50[j]),f"ew75_{ch}":float(e75[j]),
            f"gap50_{ch}":float(last[j]-e50[j]),f"disp_{ch}":float(last[j]-first[j]),
            f"vel_{ch}":float(vel[j]),f"accel_{ch}":float(accel[j]),
            f"path_abs_{ch}":float(abs_path[j]),f"position_{ch}":float((last[j]-mins[j])/rng[j]),
        })
    out.update({
        "path_euclid":path,"prev_norm":float(np.linalg.norm(last)),
        "prev_angle":float(np.arctan2(last[1],last[0])),
        "prev_ratio":float(last[1]/max(last[0],1e-9)),
        "prev_diff":float(last[1]-last[0]),
        "recent_speed_norm":float(np.linalg.norm(vel)),"curvature_norm":float(np.linalg.norm(accel)),
    })
    rows.append(out)
hist=hist.merge(pd.DataFrame(rows),on="row_id",validate="one_to_one")
hist["prev_sum"]=hist.prev_LC1+hist.prev_LC5
hist["prev_diff_fusion"]=hist.prev_LC5-hist.prev_LC1
hist["prev_ratio_fusion"]=hist.prev_LC5/np.maximum(hist.prev_LC1,1e-9)
hist["prev_log_ratio_fusion"]=np.log(np.maximum(hist.prev_LC5,1e-9)/np.maximum(hist.prev_LC1,1e-9))

# Fixed pass-graph spectral coordinates.
W=np.zeros((6,6))
for i in range(5): W[i,i+1]=W[i+1,i]=1
L=np.diag(W.sum(1))-W
_,EV=np.linalg.eigh(L)
hist["pass_spec1"]=[float(EV[int(t)-1,1]) for t in hist.Pass_T]
hist["pass_spec2"]=[float(EV[int(t)-1,2]) for t in hist.Pass_T]

BASE=["Load_kN","Speed_kmh","Pass_T","prev_LC1","prev_LC5"]
SUMFEATS=["history_count"]+[f"hist_{ch}_{s}" for ch in ("LC1","LC5") for s in ("mean","std","range","weighted","cumdelta","lastdelta","slope")]
V7FULL=list(dict.fromkeys(BASE+["lag2_LC1","lag2_LC5","lag2_available","lag3_LC1","lag3_LC5","lag3_available"]+SUMFEATS+
    ["prev_sum","prev_diff_fusion","prev_ratio_fusion","prev_log_ratio_fusion","pass_spec1","pass_spec2"]))
EWMA=BASE+["ew50_LC1","ew50_LC5","gap50_LC1","gap50_LC5","history_count","observed_fraction"]
MULTI=BASE+["ew25_LC1","ew25_LC5","ew50_LC1","ew50_LC5","ew75_LC1","ew75_LC5","gap50_LC1","gap50_LC5","history_count","observed_fraction"]
COMPACT=BASE+["ew50_LC1","ew50_LC5","gap50_LC1","gap50_LC5","disp_LC1","disp_LC5","vel_LC1","vel_LC5","path_euclid","prev_norm","prev_angle","prev_ratio","history_count","observed_fraction"]
PATH=BASE+["disp_LC1","disp_LC5","vel_LC1","vel_LC5","accel_LC1","accel_LC5","path_euclid","path_abs_LC1","path_abs_LC5","position_LC1","position_LC5","recent_speed_norm","curvature_norm","history_count","observed_fraction"]
FULLSTATE=list(dict.fromkeys(MULTI+PATH+["prev_norm","prev_angle","prev_ratio","prev_diff"]))

TASKS={
 "V5":BASE,
 "FUSION":BASE+["prev_sum","prev_diff_fusion","prev_ratio_fusion","prev_log_ratio_fusion"],
 "LAG2":BASE+["lag2_LC1","lag2_LC5","lag2_available"],
 "LAG3":BASE+["lag2_LC1","lag2_LC5","lag2_available","lag3_LC1","lag3_LC5","lag3_available"],
 "SUMMARY":BASE+SUMFEATS,
 "SPECTRAL_REPLACE":["Load_kN","Speed_kmh","prev_LC1","prev_LC5","pass_spec1","pass_spec2"],
 "SPECTRAL_AUGMENT":BASE+["pass_spec1","pass_spec2"],
 "V7_FULL":V7FULL,
 "STATE_EWMA":EWMA,
 "STATE_MULTISCALE":MULTI,
 "STATE_COMPACT":COMPACT,
 "STATE_PATH":PATH,
 "STATE_FULL":FULLSTATE,
}
EXPERTS=list(TASKS)+["PERSISTENCE"]

Y=hist[TARGET_COLS].to_numpy(float)
G=hist.Group_VW.to_numpy()
PASSES=hist.Pass_T.to_numpy(int)

def ydelta(d): return d[TARGET_COLS].to_numpy(float)-d[prev_peaks].to_numpy(float)

# Select a representation's tree config on outer-training only, returning selected-config OOF and outer-test predictions.
def fit_expert(tr,te,task,trace):
    feats=TASKS[task]; groups=tr.Group_VW.to_numpy()
    gkf=GroupKFold(n_splits=min(4,len(np.unique(groups))))
    yd=ydelta(tr); X=tr[feats]
    best=None
    for ci,(name,params) in enumerate(CANDIDATES):
        po=np.full((len(tr),2),np.nan)
        for itr,iva in gkf.split(X,groups=groups):
            for ti in range(2):
                est=make_tree(name,params); est.fit(X.iloc[itr],yd[itr,ti])
                po[iva,ti]=est.predict(X.iloc[iva])+tr.iloc[iva][prev_peaks[ti]].to_numpy(float)
        sc=joint_mae(tr[TARGET_COLS].to_numpy(float),po,groups)
        if best is None or sc<best[0]-1e-12: best=(sc,name,params,po)
    sc,name,params,train_oof=best
    test_delta=[]
    for ti in range(2):
        est=make_tree(name,params); est.fit(tr[feats],yd[:,ti]); test_delta.append(est.predict(te[feats]))
    test_pred=np.column_stack(test_delta)+te[prev_peaks].to_numpy(float)
    trace.append({"outer_group":str(te.Group_VW.iloc[0]),"task":task,"selected_model":name,"params":json.dumps(params,sort_keys=True),"inner_group_MAE":float(sc)})
    return train_oof,test_pred

def pass_regime(p):
    return "T2" if int(p)==2 else ("T6" if int(p)==6 else "T3_T5")

def choose_best_expert(train_y,pmap,groups,mask=None,target=None):
    if mask is None: mask=np.ones(len(train_y),dtype=bool)
    best=None
    for e,p in pmap.items():
        if target is None: sc=joint_mae(train_y[mask],p[mask],groups[mask])
        else: sc=group_mae(train_y[mask,target],p[mask,target],groups[mask])
        if best is None or sc<best[0]-1e-12: best=(sc,e)
    return best

def best_pair_blend(train_y,pmap,groups,target=None,mask=None):
    if mask is None: mask=np.ones(len(train_y),dtype=bool)
    names=list(pmap)
    best=None
    weights=np.linspace(0,1,21)
    for i,a in enumerate(names):
        for b in names[i:]:
            for w in weights:
                pp=(1-w)*pmap[a]+w*pmap[b]
                if target is None: sc=joint_mae(train_y[mask],pp[mask],groups[mask])
                else: sc=group_mae(train_y[mask,target],pp[mask,target],groups[mask])
                row=(sc,a,b,float(w))
                if best is None or row[0]<best[0]-1e-12: best=row
    return best

phase("NESTED_ENSEMBLE")
method_rows=[]; fold_trace=[]; base_oof_rows=[]; pair_fold_rows=[]; greedy_fold_rows=[]; gate_fold_rows=[]
global_pred={m:np.full((len(hist),2),np.nan) for m in [
    "V5","EQUAL_ALL","STACK_RIDGE10","STACK_RIDGE100","CHANNEL_SELECT","REGIME_SELECT","CHANNEL_REGIME_SELECT","CHANNEL_REGIME_BLEND2","GREEDY_BLEND"
]}
pair_global={e:np.full((len(hist),2),np.nan) for e in EXPERTS if e!="V5"}
expert_global={e:np.full((len(hist),2),np.nan) for e in EXPERTS}

for oi,outer in enumerate(pd.unique(hist.Group_VW),1):
    phase(f"OUTER_{oi}_{outer}")
    te_mask=hist.Group_VW.to_numpy()==outer; tr_mask=~te_mask
    tr=hist[tr_mask].copy(); te=hist[te_mask].copy()
    ytr=tr[TARGET_COLS].to_numpy(float); gtr=tr.Group_VW.to_numpy(); ptr=tr.Pass_T.to_numpy(int)
    yte=te[TARGET_COLS].to_numpy(float)
    ptrain={}; ptest={}
    trace=[]
    for task in TASKS:
        a,b=fit_expert(tr,te,task,trace); ptrain[task]=a; ptest[task]=b
    ptrain["PERSISTENCE"]=tr[prev_peaks].to_numpy(float)
    ptest["PERSISTENCE"]=te[prev_peaks].to_numpy(float)
    for e in EXPERTS: expert_global[e][te_mask]=ptest[e]
    global_pred["V5"][te_mask]=ptest["V5"]

    # Equal average all experts.
    global_pred["EQUAL_ALL"][te_mask]=np.mean(np.stack([ptest[e] for e in EXPERTS]),axis=0)

    # V5 + each candidate pairwise convex blend, weight chosen inside outer-training OOF only.
    grid=np.linspace(0,1,21)
    for e in EXPERTS:
        if e=="V5": continue
        best=None
        for w in grid:
            pp=(1-w)*ptrain["V5"]+w*ptrain[e]
            sc=joint_mae(ytr,pp,gtr)
            if best is None or sc<best[0]-1e-12: best=(sc,float(w))
        sc,w=best
        pair_global[e][te_mask]=(1-w)*ptest["V5"]+w*ptest[e]
        pair_fold_rows.append({"outer_group":outer,"candidate":e,"weight_candidate":w,"inner_joint_MAE":sc})

    # Fixed ridge stacking from cross-fitted outer-training expert predictions.
    Xtr=np.column_stack([ptrain[e][:,ti] for e in EXPERTS for ti in []]) if False else None
    for alpha,name in [(10.0,"STACK_RIDGE10"),(100.0,"STACK_RIDGE100")]:
        out=np.zeros((len(te),2))
        for ti in range(2):
            xm=np.column_stack([ptrain[e][:,ti] for e in EXPERTS])
            xt=np.column_stack([ptest[e][:,ti] for e in EXPERTS])
            model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=alpha))])
            model.fit(xm,ytr[:,ti]); out[:,ti]=model.predict(xt)
        global_pred[name][te_mask]=out

    # Channel-specific best expert.
    out=np.zeros((len(te),2)); selections={}
    for ti,t in enumerate(TARGET_NAMES):
        sc,e=choose_best_expert(ytr,ptrain,gtr,target=ti); out[:,ti]=ptest[e][:,ti]; selections[t]=e
    global_pred["CHANNEL_SELECT"][te_mask]=out
    gate_fold_rows.append({"outer_group":outer,"method":"CHANNEL_SELECT","selections":json.dumps(selections,sort_keys=True)})

    # Regime-specific joint expert.
    out=np.zeros((len(te),2)); selections={}
    for reg in ("T2","T3_T5","T6"):
        m=np.array([pass_regime(p)==reg for p in ptr])
        sc,e=choose_best_expert(ytr,ptrain,gtr,mask=m)
        tm=np.array([pass_regime(p)==reg for p in te.Pass_T.to_numpy(int)])
        out[tm]=ptest[e][tm]; selections[reg]=e
    global_pred["REGIME_SELECT"][te_mask]=out
    gate_fold_rows.append({"outer_group":outer,"method":"REGIME_SELECT","selections":json.dumps(selections,sort_keys=True)})

    # Channel x regime hard gate.
    out=np.zeros((len(te),2)); selections={}
    for ti,t in enumerate(TARGET_NAMES):
        for reg in ("T2","T3_T5","T6"):
            m=np.array([pass_regime(p)==reg for p in ptr])
            sc,e=choose_best_expert(ytr,ptrain,gtr,mask=m,target=ti)
            tm=np.array([pass_regime(p)==reg for p in te.Pass_T.to_numpy(int)])
            out[tm,ti]=ptest[e][tm,ti]; selections[f"{t}:{reg}"]=e
    global_pred["CHANNEL_REGIME_SELECT"][te_mask]=out
    gate_fold_rows.append({"outer_group":outer,"method":"CHANNEL_REGIME_SELECT","selections":json.dumps(selections,sort_keys=True)})

    # Channel x regime best two-expert convex blend.
    out=np.zeros((len(te),2)); selections={}
    for ti,t in enumerate(TARGET_NAMES):
        for reg in ("T2","T3_T5","T6"):
            m=np.array([pass_regime(p)==reg for p in ptr])
            sc,a,b,w=best_pair_blend(ytr,ptrain,gtr,target=ti,mask=m)
            tm=np.array([pass_regime(p)==reg for p in te.Pass_T.to_numpy(int)])
            out[tm,ti]=(1-w)*ptest[a][tm,ti]+w*ptest[b][tm,ti]
            selections[f"{t}:{reg}"]={"a":a,"b":b,"w_b":w}
    global_pred["CHANNEL_REGIME_BLEND2"][te_mask]=out
    gate_fold_rows.append({"outer_group":outer,"method":"CHANNEL_REGIME_BLEND2","selections":json.dumps(selections,sort_keys=True)})

    # Greedy forward convex blend from V5, up to four useful additions; all decisions use outer-training OOF only.
    cur_tr=ptrain["V5"].copy(); cur_te=ptest["V5"].copy(); selected=["V5"]; unused=[e for e in EXPERTS if e!="V5"]
    current=joint_mae(ytr,cur_tr,gtr); steps=[]
    for step in range(1,5):
        best=None
        for e in unused:
            for w in np.linspace(0.05,0.75,15):
                pp=(1-w)*cur_tr+w*ptrain[e]
                sc=joint_mae(ytr,pp,gtr)
                if best is None or sc<best[0]-1e-12: best=(sc,e,float(w),pp)
        if best is None or best[0]>=current-1e-6: break
        sc,e,w,newtr=best
        cur_tr=newtr; cur_te=(1-w)*cur_te+w*ptest[e]; current=sc
        selected.append(e); unused.remove(e); steps.append({"step":step,"expert":e,"weight_new":w,"inner_joint_MAE":sc})
    global_pred["GREEDY_BLEND"][te_mask]=cur_te
    greedy_fold_rows.append({"outer_group":outer,"selected":";".join(selected),"steps":json.dumps(steps,sort_keys=True),"final_inner_joint_MAE":current})

    fold_trace.extend(trace)
    dump(STATE/"progress_v9.json",{"outer_completed":oi,"outer_total":9,"outer_group":outer})

# Evaluate base experts and all ensemble methods globally.
phase("EVALUATION")
summary=[]
for e,p in expert_global.items():
    m=metrics(Y,p,G); m.update({"method":e,"kind":"base_expert"}); summary.append(m)
for name,p in global_pred.items():
    m=metrics(Y,p,G); m.update({"method":name,"kind":"ensemble" if name!="V5" else "baseline"}); summary.append(m)
for e,p in pair_global.items():
    name="V5_PAIR_"+e
    m=metrics(Y,p,G); m.update({"method":name,"kind":"pairwise_blend"}); summary.append(m)
summary_df=pd.DataFrame(summary).sort_values("joint_Group_MAE").reset_index(drop=True)

# Exact frozen V5 gate.
v5=summary_df[summary_df.method=="V5"].iloc[0]
if abs(float(v5.joint_Group_MAE)-FROZEN_V5)>1e-9 or abs(float(v5.LC1_proxy_N_Group_MAE)-FROZEN_V5_LC1)>1e-9 or abs(float(v5.LC5_proxy_N_Group_MAE)-FROZEN_V5_LC5)>1e-9:
    raise RuntimeError(f"V5_EXACT_REPRODUCTION_FAILED:{float(v5.joint_Group_MAE)}")
print("V5_EXACT_REPRODUCTION_PASS",float(v5.joint_Group_MAE),flush=True)

# Paired group-level tests vs V5 for ensemble methods.
paired=[]
rng=np.random.default_rng(20260928)
v5p=global_pred["V5"]
test_methods={k:v for k,v in global_pred.items() if k!="V5"}
test_methods.update({"V5_PAIR_"+e:p for e,p in pair_global.items()})
for name,p in test_methods.items():
    diffs=[]; group_rows=[]
    for g in pd.unique(G):
        m=G==g
        be=float(np.mean(np.abs(Y[m]-v5p[m])))
        ce=float(np.mean(np.abs(Y[m]-p[m])))
        diffs.append(be-ce); group_rows.append((g,be,ce,be-ce))
    d=np.asarray(diffs); obs=float(d.mean())
    perm=np.array([np.mean(d*np.array(s)) for s in itertools.product([-1,1],repeat=9)])
    boot=np.array([np.mean(rng.choice(d,size=9,replace=True)) for _ in range(20000)])
    met=summary_df[summary_df.method==name].iloc[0]
    paired.append({
        "method":name,"joint_Group_MAE_N":float(met.joint_Group_MAE),
        "relative_improvement_vs_V5_pct":float(100*(FROZEN_V5-float(met.joint_Group_MAE))/FROZEN_V5),
        "mean_group_paired_improvement_N":obs,
        "bootstrap_95CI_low_N":float(np.quantile(boot,.025)),
        "bootstrap_95CI_high_N":float(np.quantile(boot,.975)),
        "exact_signflip_one_sided_p":float(np.mean(perm>=obs-1e-12)),
        "groups_improved":int(np.sum(d>0))
    })
paired_df=pd.DataFrame(paired).sort_values("joint_Group_MAE_N").reset_index(drop=True)

# Error complementarity diagnostics across base experts.
phase("COMPLEMENTARITY")
corr_rows=[]
for ti,t in enumerate(TARGET_NAMES):
    E=pd.DataFrame({e:expert_global[e][:,ti]-Y[:,ti] for e in EXPERTS})
    C=E.corr()
    for i,a in enumerate(EXPERTS):
        for b in EXPERTS[i+1:]:
            corr_rows.append({"target":t,"expert_a":a,"expert_b":b,"signed_error_corr":float(C.loc[a,b])})
corr_df=pd.DataFrame(corr_rows).sort_values("signed_error_corr")
most_diverse=corr_df.head(10).to_dict(orient="records")

# Oracle headroom: descriptive only, never a deployable estimate.
abs_err=np.stack([np.abs(expert_global[e]-Y) for e in EXPERTS],axis=2) # n,2,E
oracle_target=np.take_along_axis(np.stack([expert_global[e] for e in EXPERTS],axis=2),np.argmin(abs_err,axis=2)[:,:,None],axis=2)[:,:,0]
oracle_target_mae=joint_mae(Y,oracle_target,G)
# Same expert per row minimizes joint target error.
row_err=abs_err.mean(axis=1)
best_idx=np.argmin(row_err,axis=1)
Pstack=np.stack([expert_global[e] for e in EXPERTS],axis=2)
oracle_row=np.zeros_like(Y)
for i,j in enumerate(best_idx): oracle_row[i]=Pstack[i,:,j]
oracle_row_mae=joint_mae(Y,oracle_row,G)

# Where candidate experts complement hard V5 groups.
group_comp=[]
for e in EXPERTS:
    if e=="V5": continue
    for g in pd.unique(G):
        m=G==g
        be=float(np.mean(np.abs(Y[m]-v5p[m])))
        ce=float(np.mean(np.abs(Y[m]-expert_global[e][m])))
        group_comp.append({"expert":e,"Group_VW":g,"V5_MAE_N":be,"expert_MAE_N":ce,"improvement_N":be-ce})
group_comp_df=pd.DataFrame(group_comp)

# Selection counts.
gate_df=pd.DataFrame(gate_fold_rows)
pair_fold_df=pd.DataFrame(pair_fold_rows)
greedy_df=pd.DataFrame(greedy_fold_rows)
trace_df=pd.DataFrame(fold_trace)
selection_counts={task:dict(Counter(z.selected_model)) for task,z in trace_df.groupby("task")}
gate_counts={}
for method,z in gate_df.groupby("method"):
    c=Counter()
    for s in z.selections:
        obj=json.loads(s)
        for k,v in obj.items():
            if isinstance(v,str): c[f"{k}->{v}"]+=1
            else: c[f"{k}->{v.get('a')}+{v.get('b')}"]+=1
    gate_counts[method]=dict(c)

# Save evidence.
summary_df.to_csv(TABLE/"summary_v9.csv",index=False)
paired_df.to_csv(TABLE/"paired_vs_v5_v9.csv",index=False)
pair_fold_df.to_csv(TABLE/"pairwise_blend_outer_weights_v9.csv",index=False)
greedy_df.to_csv(TABLE/"greedy_outer_selections_v9.csv",index=False)
gate_df.to_csv(TABLE/"moe_outer_selections_v9.csv",index=False)
trace_df.to_csv(TABLE/"base_expert_selection_v9.csv",index=False)
corr_df.to_csv(TABLE/"base_error_correlations_v9.csv",index=False)
group_comp_df.to_csv(TABLE/"base_group_complementarity_v9.csv",index=False)

best=summary_df[(summary_df.kind!="base_expert")&(summary_df.method!="V5")].iloc[0]
best_pair=paired_df[paired_df.method==best.method].iloc[0]
best_pairwise=summary_df[summary_df.kind=="pairwise_blend"].iloc[0]
results={
 "schema":"soilbin.q1.v9.ensemble-moe",
 "created_utc":datetime.now(timezone.utc).isoformat(),
 "post_lock_exploratory":True,"does_not_supersede_v5":True,
 "baseline":{"joint_Group_MAE_N":float(v5.joint_Group_MAE),"LC1_Group_MAE_N":float(v5.LC1_proxy_N_Group_MAE),"LC5_Group_MAE_N":float(v5.LC5_proxy_N_Group_MAE),"exact_match":True},
 "best_ensemble":{"method":str(best.method),"joint_Group_MAE_N":float(best.joint_Group_MAE),
                  "relative_improvement_vs_V5_pct":float(100*(FROZEN_V5-float(best.joint_Group_MAE))/FROZEN_V5),
                  "paired":best_pair.to_dict()},
 "best_pairwise_blend":{"method":str(best_pairwise.method),"joint_Group_MAE_N":float(best_pairwise.joint_Group_MAE),
                         "relative_improvement_vs_V5_pct":float(100*(FROZEN_V5-float(best_pairwise.joint_Group_MAE))/FROZEN_V5)},
 "summary":summary_df.to_dict(orient="records"),
 "paired_vs_v5":paired_df.to_dict(orient="records"),
 "oracle_headroom":{"per_target_oracle_joint_Group_MAE_N":oracle_target_mae,"per_row_single_expert_oracle_joint_Group_MAE_N":oracle_row_mae,
                    "guard":"Descriptive unattainable lower bounds using test truth; never model performance."},
 "most_diverse_error_pairs":most_diverse,
 "base_selection_counts":selection_counts,
 "gate_selection_counts":gate_counts,
 "guards":{"random_split_used":False,"outer_test_used_for_weight_selection":False,"weights_and_gates_learned_on_outer_training_only":True,
           "external_labels_used_for_selection":False,"current_pass_waveform_used":False},
 "interpretation":"V9 tests whether individually weaker representations add complementary error information through leakage-safe pairwise blending, stacking, greedy forward blending, and channel/pass-regime mixture-of-experts. Frozen V5 remains reference."
}
dump(ROOT/"RESULTS_V9.json",results)
manifest={"schema":"soilbin.q1.v9.run-manifest","finished_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,
          "source_sha256":EXPECTED_MODEL_SHA,"experts":EXPERTS,"tasks":TASKS,"candidate_catalog":[{"model":n,"params":p} for n,p in CANDIDATES],
          "files":sorted([p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()])}
dump(ROOT/"RUN_MANIFEST_V9.json",manifest)

# Plot.
q=summary_df[summary_df.kind!="base_expert"].head(18)
plt.figure(figsize=(13,6)); plt.bar(np.arange(len(q)),q.joint_Group_MAE); plt.axhline(FROZEN_V5,linestyle="--",linewidth=1)
plt.xticks(np.arange(len(q)),q.method,rotation=65,ha="right",fontsize=8); plt.ylabel("Joint group-equal MAE (N)")
plt.tight_layout(); plt.savefig(FIG/"01_ensemble_comparison.png",dpi=180); plt.close()

report=["# SoilBin V9 — Ensemble & Mixture-of-Experts Ablation","","**Post-lock exploratory; V5 frozen.**","",
        f"- Exact V5: {FROZEN_V5:.6f} N.",
        f"- Best ensemble: {best.method} = {float(best.joint_Group_MAE):.6f} N.",
        f"- Best pairwise blend: {best_pairwise.method} = {float(best_pairwise.joint_Group_MAE):.6f} N.",
        f"- Oracle per-row single-expert lower bound: {oracle_row_mae:.6f} N.",
        f"- Oracle per-target lower bound: {oracle_target_mae:.6f} N.","","## Ensemble summary"]
for _,r in summary_df[summary_df.kind!="base_expert"].iterrows():
    report.append(f"- {r.method}: {r.joint_Group_MAE:.6f} N")
(ROOT/"FINAL_REPORT_V9.md").write_text("\n".join(report),encoding="utf-8")
shutil.make_archive("/kaggle/working/SoilBin_Q1_Q2_V9_ENSEMBLE_MOE_20260928","zip",ROOT)
phase("COMPLETE")
print("SOILBIN_V9_COMPLETE",json.dumps({
 "v5_mae_N":FROZEN_V5,"best_ensemble":str(best.method),"best_ensemble_mae_N":float(best.joint_Group_MAE),
 "best_ensemble_improvement_pct":float(100*(FROZEN_V5-float(best.joint_Group_MAE))/FROZEN_V5),
 "best_pairwise":str(best_pairwise.method),"best_pairwise_mae_N":float(best_pairwise.joint_Group_MAE),
 "oracle_row_mae_N":oracle_row_mae,"oracle_target_mae_N":oracle_target_mae
},sort_keys=True),flush=True)
