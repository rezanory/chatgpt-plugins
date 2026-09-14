# SoilBin Q1/Q2 waveform-derived history benchmark v2.0
from __future__ import annotations
import os, sys, json, math, hashlib, random, warnings, gzip, base64
from pathlib import Path
from io import StringIO
from datetime import datetime, timezone
from collections import Counter

SEED=20260914
random.seed(SEED); os.environ.setdefault("PYTHONHASHSEED",str(SEED))
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np
np.random.seed(SEED)
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
try:
    from catboost import CatBoostRegressor
except Exception:
    import subprocess
    subprocess.check_call([sys.executable,"-m","pip","install","-q","catboost==1.2.8"])
    from catboost import CatBoostRegressor

ROOT=Path("/kaggle/working/SOILBIN_WAVEFORM_V2")
ROOT.mkdir(parents=True,exist_ok=True); TABLE=ROOT/"tables"; FIG=ROOT/"figures"; STATE=ROOT/"state"
TABLE.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
EXPECTED_CSV_SHA="dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"
EXPECTED_MISSING={"V1W1T1","V2W3T2","V3W2T1"}
RESPONSE_STATUS="CALIBRATED_VERTICAL_FORCE_PROXY_N_PENDING_SENSOR_AREA_OR_STRESS_CALIBRATION"
DEPTH_STATUS="PROVISIONAL_LC5_5CM_LC1_15CM_PENDING_LAYOUT_CONFIRMATION"
TARGET_COLS=["LC1_peak_magnitude_N_delta","LC5_peak_magnitude_N_delta"]
TARGET_NAMES=["LC1_proxy_N","LC5_proxy_N"]
N_JOBS=max(1,min(4,(os.cpu_count() or 2)))

def phase(x): print(f"CGP_PHASE:{x}",flush=True)
def jsonable(v):
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,)): return float(v)
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,(pd.Timestamp,datetime)): return str(v)
    return v
def jdump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=jsonable),encoding="utf-8")

phase("DATA_FREEZE")
raw=gzip.decompress(base64.b64decode(MODEL_CSV_GZ_B64.strip())).decode("utf-8")
if hashlib.sha256(raw.encode()).hexdigest()!=EXPECTED_CSV_SHA: raise RuntimeError("V2_SOURCE_FINGERPRINT_MISMATCH")
df=pd.read_csv(StringIO(raw))
if len(df)!=51 or df.Run_ID.nunique()!=51: raise RuntimeError("EXPECTED_51_RUNS")
df["Group_VW"]=df.Run_ID.str.extract(r"^(V\dW\d)")[0]
df["Speed_kmh"]=df["Speed_level"].astype(float)
df["Load_kN"]=df["Weight_level"].astype(float)+1.0
all_ids={f"V{v}W{w}T{t}" for v in range(1,4) for w in range(1,4) for t in range(1,7)}
if all_ids-set(df.Run_ID)!=EXPECTED_MISSING: raise RuntimeError("DESIGN_SLOT_MISMATCH")
for c in TARGET_COLS:
    if c not in df or df[c].isna().any(): raise RuntimeError(f"TARGET_INVALID:{c}")

prev=df.copy(); prev["Pass_T"]=prev["Pass_T"].astype(int)+1
# Morphology is deliberately restricted to non-target descriptors from the PREVIOUS pass.
# Peak signed/magnitude/full-calibration fields are excluded here so C tests information beyond B_prev_peak.
morph_suffixes=("baseline_noise_sigma_raw_kg","event_duration_ms","fwhm_ms","rise_time_ms","fall_time_ms","impulse_abs_Ns","impulse_signed_Ns","rms_event_N","mean_abs_event_N")
morph=[c for c in df.columns if c.startswith(("LC1_","LC5_")) and c.endswith(morph_suffixes)]
morph += [c for c in ["Peak_delay_LC5_minus_LC1_ms","Impulse_ratio_LC5_to_LC1"] if c in df.columns]
prev_peak_cols=TARGET_COLS[:]
rename={c:f"prev_{c}" for c in set(morph+prev_peak_cols+["QC_status"])}
prev_small=prev[["Group_VW","Pass_T","QC_status"]+list(dict.fromkeys(morph+prev_peak_cols))].rename(columns=rename)
hist=df.merge(prev_small,on=["Group_VW","Pass_T"],how="left",validate="one_to_one")
hist=hist[hist[f"prev_{TARGET_COLS[0]}"].notna() & hist[f"prev_{TARGET_COLS[1]}"].notna()].copy().reset_index(drop=True)
if len(hist)!=41: raise RuntimeError(f"EXPECTED_41_CONSECUTIVE_RUNS_GOT_{len(hist)}")
hist["pair_qc_ok"]=(hist["QC_status"].fillna("")=="OK") & (hist["prev_QC_status"].fillna("")=="OK")
conditions=["Load_kN","Speed_kmh","Pass_T"]
prev_peaks=[f"prev_{c}" for c in prev_peak_cols]
prev_morph=[f"prev_{c}" for c in morph if f"prev_{c}" not in prev_peaks]
feature_sets={
 "A_conditions":conditions,
 "B_prev_peak":conditions+prev_peaks,
 "C_prev_waveform_coupled":conditions+prev_peaks+prev_morph,
 "D_delta_waveform_coupled":conditions+prev_peaks+prev_morph,
}
for k,cols in feature_sets.items():
    bad=[c for c in cols if c not in hist.columns]
    if bad: raise RuntimeError(f"MISSING_FEATURES:{k}:{bad}")
qc={"schema":"soilbin.q1.waveform.v2","created_utc":datetime.now(timezone.utc).isoformat(),"source_csv_sha256":EXPECTED_CSV_SHA,"valid_runs":51,"consecutive_pairs":41,"groups":int(hist.Group_VW.nunique()),"missing_design_slots":sorted(EXPECTED_MISSING),"review_runs":df.loc[df.QC_status!="OK","Run_ID"].tolist(),"pair_qc_ok_count":int(hist.pair_qc_ok.sum()),"pair_qc_review_count":int((~hist.pair_qc_ok).sum()),"response_status":RESPONSE_STATUS,"depth_mapping_status":DEPTH_STATUS,"leakage_guard":"Current-run waveform descriptors are forbidden as predictors. Only current operating conditions and previous-pass descriptors are used.","external_validation":False}
jdump(ROOT/"qc_v2.json",qc)

MODEL_FAMILIES=["Linear","PolynomialRidge","SVR","GPR","RandomForest","ExtraTrees","CatBoost","MLP"]
def configs(name):
    if name=="Linear": return [{}]
    if name=="PolynomialRidge": return [{"alpha":1.0},{"alpha":10.0}]
    if name=="SVR": return [{"C":1.0,"epsilon":0.1},{"C":10.0,"epsilon":0.1}]
    if name=="GPR": return [{"nu":1.5,"noise":0.05},{"nu":2.5,"noise":0.05}]
    if name in ("RandomForest","ExtraTrees"): return [{"max_depth":d,"min_samples_leaf":m} for d,m in [(2,1),(None,2),(None,1)]]
    if name=="CatBoost": return [{"depth":2,"learning_rate":0.05},{"depth":3,"learning_rate":0.05}]
    if name=="MLP": return [{"hidden_layer_sizes":(8,),"alpha":0.05},{"hidden_layer_sizes":(16,),"alpha":0.1}]
    raise KeyError(name)
def estimator(name,p,nf):
    if name=="Linear": return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",LinearRegression())])
    if name=="PolynomialRidge": return Pipeline([("imp",SimpleImputer(strategy="median")),("sc1",StandardScaler()),("poly",PolynomialFeatures(2,include_bias=False)),("sc2",StandardScaler()),("m",Ridge(alpha=p["alpha"]))])
    if name=="SVR": return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",MultiOutputRegressor(SVR(kernel="rbf",C=p["C"],epsilon=p["epsilon"],gamma="scale")))])
    if name=="GPR":
        ker=ConstantKernel(1.0,(1e-3,1e3))*Matern(length_scale=np.ones(nf),length_scale_bounds=(1e-2,100),nu=p["nu"])+WhiteKernel(noise_level=p["noise"],noise_level_bounds=(1e-5,1.0))
        return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",GaussianProcessRegressor(kernel=ker,alpha=1e-6,normalize_y=True,n_restarts_optimizer=0,random_state=SEED))])
    if name=="RandomForest": return Pipeline([("imp",SimpleImputer(strategy="median")),("m",RandomForestRegressor(n_estimators=300,random_state=SEED,n_jobs=1,max_features=1.0,**p))])
    if name=="ExtraTrees": return Pipeline([("imp",SimpleImputer(strategy="median")),("m",ExtraTreesRegressor(n_estimators=300,random_state=SEED,n_jobs=1,max_features=1.0,**p))])
    if name=="CatBoost": return Pipeline([("imp",SimpleImputer(strategy="median")),("m",CatBoostRegressor(loss_function="MultiRMSE",iterations=300,random_seed=SEED,thread_count=1,verbose=False,allow_writing_files=False,**p))])
    if name=="MLP": return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",MLPRegressor(solver="lbfgs",activation="tanh",random_state=SEED,max_iter=4000,**p))])
    raise KeyError(name)

def group_mae(y,p,g):
    y=np.asarray(y,float); p=np.asarray(p,float); g=np.asarray(g); per=[]
    for gg in np.unique(g):
        ix=g==gg; per.append(np.mean(np.abs(y[ix]-p[ix]),axis=0))
    return np.mean(np.stack(per),axis=0)
def joint_score(y,p,g): return float(np.mean(group_mae(y,p,g)))
def metrics(y,p,g):
    y=np.asarray(y,float); p=np.asarray(p,float); gm=group_mae(y,p,g)
    out={"n":int(len(y)),"n_groups":int(len(np.unique(g))),"joint_Group_MAE":float(np.mean(gm))}
    for j,name in enumerate(TARGET_NAMES):
        e=p[:,j]-y[:,j]; ss=float(np.sum((y[:,j]-y[:,j].mean())**2))
        out[f"{name}_Group_MAE"]=float(gm[j]); out[f"{name}_MAE"]=float(np.mean(np.abs(e))); out[f"{name}_RMSE"]=float(np.sqrt(np.mean(e*e))); out[f"{name}_Bias"]=float(np.mean(e)); out[f"{name}_R2"]=None if ss<=1e-15 else float(1-np.sum(e*e)/ss)
    return out
def inner_split_indices(d,train):
    groups=d.loc[train,"Group_VW"].to_numpy(); n=min(4,len(np.unique(groups)))
    if n<2: raise RuntimeError("INSUFFICIENT_INNER_GROUPS")
    sp=GroupKFold(n_splits=n); z=np.zeros((len(train),1)); return [(np.asarray(train)[a],np.asarray(train)[b]) for a,b in sp.split(z,groups=groups)]
def fit_predict(name,p,Xtr,ytr,Xte):
    m=estimator(name,p,Xtr.shape[1])
    with threadpool_limits(1): m.fit(Xtr,ytr); pred=np.asarray(m.predict(Xte),float)
    if pred.ndim==1: pred=pred[:,None]
    return m,pred
def tune_family(d,features,mode,train,name):
    rows=[]; best=None; best_score=math.inf
    for ci,p in enumerate(configs(name)):
        yy=[]; pp=[]; gg=[]; err=None
        for a,b in inner_split_indices(d,train):
            ytr=d.loc[a,TARGET_COLS].to_numpy(float)
            if mode=="delta": ytr=ytr-d.loc[a,prev_peaks].to_numpy(float)
            try:
                _,rawp=fit_predict(name,p,d.loc[a,features].to_numpy(float),ytr,d.loc[b,features].to_numpy(float)); pred=rawp+d.loc[b,prev_peaks].to_numpy(float) if mode=="delta" else rawp
                yy.append(d.loc[b,TARGET_COLS].to_numpy(float)); pp.append(pred); gg.append(d.loc[b,"Group_VW"].to_numpy())
            except Exception as e: err=f"{type(e).__name__}:{e}"; break
        score=None
        if err is None:
            Y=np.vstack(yy); P=np.vstack(pp); G=np.concatenate(gg); score=joint_score(Y,P,G)
            if score<best_score: best_score=score; best=p.copy()
        rows.append({"model":name,"candidate":ci,"params":p,"inner_joint_Group_MAE":score,"error":err})
    return best,best_score,rows
def inner_oof_for_selected(d,features,mode,train,name,p):
    Y=[]; P=[]; G=[]
    for a,b in inner_split_indices(d,train):
        ytr=d.loc[a,TARGET_COLS].to_numpy(float)
        if mode=="delta": ytr=ytr-d.loc[a,prev_peaks].to_numpy(float)
        _,rawp=fit_predict(name,p,d.loc[a,features].to_numpy(float),ytr,d.loc[b,features].to_numpy(float)); pred=rawp+d.loc[b,prev_peaks].to_numpy(float) if mode=="delta" else rawp
        Y.append(d.loc[b,TARGET_COLS].to_numpy(float)); P.append(pred); G.append(d.loc[b,"Group_VW"].to_numpy())
    return np.vstack(Y),np.vstack(P),np.concatenate(G)

def outer_one(task,scheme,label,qc_only=False):
    features=feature_sets[task]; mode="delta" if task.startswith("D_") else "direct"; d=hist[hist.pair_qc_ok].copy().reset_index(drop=True) if qc_only else hist.copy()
    split=d.Group_VW.astype(str) if scheme=="VW" else (d.Speed_kmh.astype(str) if scheme=="speed" else d.Load_kN.astype(str)); test=np.flatnonzero(split.to_numpy()==str(label)); train=np.flatnonzero(split.to_numpy()!=str(label))
    if not len(test) or not len(train): return [],[],[],[]
    if scheme=="VW" and set(d.loc[train,"Group_VW"]) & set(d.loc[test,"Group_VW"]): raise RuntimeError("GROUP_LEAKAGE")
    search=[]; preds=[]; intervals=[]; selection={}; fitted={}
    for name in MODEL_FAMILIES:
        bp,score,ledger=tune_family(d,features,mode,train,name)
        for r in ledger: search.append({"task":task,"scheme":scheme,"outer":str(label),"qc_only":qc_only,**r})
        if bp is None: continue
        ytr=d.loc[train,TARGET_COLS].to_numpy(float); ytr=ytr-d.loc[train,prev_peaks].to_numpy(float) if mode=="delta" else ytr
        try:
            _,rawp=fit_predict(name,bp,d.loc[train,features].to_numpy(float),ytr,d.loc[test,features].to_numpy(float)); pred=rawp+d.loc[test,prev_peaks].to_numpy(float) if mode=="delta" else rawp
            selection[name]=score; fitted[name]=(bp,pred)
            for k,ix in enumerate(test):
                row={"task":task,"scheme":scheme,"outer":str(label),"qc_only":qc_only,"model":name,"selected_family":"","Run_ID":d.loc[ix,"Run_ID"],"Group_VW":d.loc[ix,"Group_VW"],"Speed_kmh":float(d.loc[ix,"Speed_kmh"]),"Load_kN":float(d.loc[ix,"Load_kN"]),"Pass_T":int(d.loc[ix,"Pass_T"])}
                for j,t in enumerate(TARGET_NAMES): row[f"{t}_true"]=float(d.loc[ix,TARGET_COLS[j]]); row[f"{t}_pred"]=float(pred[k,j])
                preds.append(row)
        except Exception: pass
    ytest=d.loc[test,TARGET_COLS].to_numpy(float)
    bpreds=[("Persistence",d.loc[test,prev_peaks].to_numpy(float)),("TrainMedian",np.tile(np.median(d.loc[train,TARGET_COLS].to_numpy(float),axis=0),(len(test),1)))]
    for bname,bpred in bpreds:
        for k,ix in enumerate(test):
            row={"task":task,"scheme":scheme,"outer":str(label),"qc_only":qc_only,"model":bname,"selected_family":"","Run_ID":d.loc[ix,"Run_ID"],"Group_VW":d.loc[ix,"Group_VW"],"Speed_kmh":float(d.loc[ix,"Speed_kmh"]),"Load_kN":float(d.loc[ix,"Load_kN"]),"Pass_T":int(d.loc[ix,"Pass_T"])}
            for j,t in enumerate(TARGET_NAMES): row[f"{t}_true"]=float(ytest[k,j]); row[f"{t}_pred"]=float(bpred[k,j])
            preds.append(row)
    if selection:
        winner=min(selection,key=lambda x:(selection[x],x)); bp,pred=fitted[winner]; yin,pin,gin=inner_oof_for_selected(d,features,mode,train,winner,bp); absr=np.abs(yin-pin); q80=np.quantile(absr,0.80,axis=0,method="higher"); q95=np.quantile(absr,0.95,axis=0,method="higher")
        for k,ix in enumerate(test):
            row={"task":task,"scheme":scheme,"outer":str(label),"qc_only":qc_only,"model":"Selected_inner","selected_family":winner,"Run_ID":d.loc[ix,"Run_ID"],"Group_VW":d.loc[ix,"Group_VW"],"Speed_kmh":float(d.loc[ix,"Speed_kmh"]),"Load_kN":float(d.loc[ix,"Load_kN"]),"Pass_T":int(d.loc[ix,"Pass_T"])}
            for j,t in enumerate(TARGET_NAMES): row[f"{t}_true"]=float(ytest[k,j]); row[f"{t}_pred"]=float(pred[k,j])
            preds.append(row)
            for nominal,q in [(0.80,q80),(0.95,q95)]:
                for j,t in enumerate(TARGET_NAMES): intervals.append({"task":task,"scheme":scheme,"outer":str(label),"qc_only":qc_only,"selected_family":winner,"Run_ID":d.loc[ix,"Run_ID"],"target":t,"nominal":nominal,"half_width_N":float(q[j]),"covered":bool(abs(ytest[k,j]-pred[k,j])<=q[j])})
    return preds,search,intervals,[{"task":task,"scheme":scheme,"outer":str(label),"qc_only":qc_only,"selected_family":min(selection,key=lambda x:(selection[x],x)) if selection else None}]
def run_task(task,scheme="VW",qc_only=False):
    d=hist[hist.pair_qc_ok].copy() if qc_only else hist; labels=np.unique(d.Group_VW.astype(str) if scheme=="VW" else (d.Speed_kmh.astype(str) if scheme=="speed" else d.Load_kN.astype(str)))
    parts=Parallel(n_jobs=N_JOBS,backend="loky")(delayed(outer_one)(task,scheme,l,qc_only) for l in labels); out=[[],[],[],[]]
    for part in parts:
        for i,x in enumerate(part): out[i]+=x
    return out

phase("NESTED_GROUPED_CV_START")
pred_rows=[]; search_rows=[]; interval_rows=[]; selection_rows=[]; tasks=["A_conditions","B_prev_peak","C_prev_waveform_coupled","D_delta_waveform_coupled"]
for i,task in enumerate(tasks,1):
    phase(f"CORE_{i}_{task}"); p,s,u,z=run_task(task,"VW",False); pred_rows+=p; search_rows+=s; interval_rows+=u; selection_rows+=z; jdump(STATE/"progress_v2.json",{"completed_tasks":tasks[:i],"pred_rows":len(pred_rows),"search_rows":len(search_rows)})
for scheme in ["speed","load"]:
    phase(f"ROBUSTNESS_C_{scheme}"); p,s,u,z=run_task("C_prev_waveform_coupled",scheme,False); pred_rows+=p; search_rows+=s; interval_rows+=u; selection_rows+=z
phase("QC_SENSITIVITY"); p,s,u,z=run_task("C_prev_waveform_coupled","VW",True); pred_rows+=p; search_rows+=s; interval_rows+=u; selection_rows+=z
pred=pd.DataFrame(pred_rows); search=pd.DataFrame(search_rows); ints=pd.DataFrame(interval_rows); sels=pd.DataFrame(selection_rows)
pred.to_csv(TABLE/"oof_predictions_v2.csv",index=False); search.to_csv(TABLE/"hyperparameter_search_v2.csv",index=False); ints.to_csv(TABLE/"uncertainty_intervals_v2.csv",index=False); sels.to_csv(TABLE/"selection_v2.csv",index=False)
summary=[]
for (task,scheme,qc_only,model),z in pred.groupby(["task","scheme","qc_only","model"],sort=False):
    d=hist[hist.pair_qc_ok] if qc_only else hist; expected=len(d); complete=len(z)==expected and z.Run_ID.nunique()==expected; row={"task":task,"scheme":scheme,"qc_only":bool(qc_only),"model":model,"complete_oof":bool(complete),"expected_n":int(expected),"n":int(len(z))}
    if complete:
        Y=np.column_stack([z[f"{t}_true"] for t in TARGET_NAMES]); P=np.column_stack([z[f"{t}_pred"] for t in TARGET_NAMES]); row.update(metrics(Y,P,z.Group_VW.to_numpy()))
    summary.append(row)
summary=pd.DataFrame(summary); summary.to_csv(TABLE/"summary_v2.csv",index=False)
coverage=[]
if not ints.empty:
    for (task,scheme,qc_only,target,nominal),z in ints.groupby(["task","scheme","qc_only","target","nominal"]): coverage.append({"task":task,"scheme":scheme,"qc_only":bool(qc_only),"target":target,"nominal":float(nominal),"n":int(len(z)),"empirical_coverage":float(z.covered.mean()),"mean_half_width_N":float(z.half_width_N.mean())})
coverage=pd.DataFrame(coverage); coverage.to_csv(TABLE/"uncertainty_calibration_v2.csv",index=False); jdump(ROOT/"uncertainty_calibration_v2.json",{"method":"Inner GroupKFold absolute-residual quantile calibrated only on outer-training groups; development coverage, not external guarantee.","rows":coverage.to_dict("records")})
def get_metric(task,model="Selected_inner",qc_only=False,scheme="VW"):
    q=summary[(summary.task==task)&(summary.model==model)&(summary.qc_only==qc_only)&(summary.scheme==scheme)&(summary.complete_oof==True)]; return None if q.empty else q.iloc[0].to_dict()
ablation=[]
for task in tasks:
    r=get_metric(task)
    if r: ablation.append(r)
jdump(ROOT/"waveform_ablation_v2.json",{"question":"Does previous-pass dual-sensor waveform morphology improve next-pass force-proxy prediction beyond current conditions and previous peak alone?","rows":ablation})
sel_counts={}
for task in tasks:
    q=sels[(sels.task==task)&(sels.scheme=="VW")&(sels.qc_only==False)]; sel_counts[task]=dict(Counter(q.selected_family.dropna()))
jdump(ROOT/"coupled_sensor_v2.json",{"targets":TARGET_NAMES,"joint_metric":"mean of target-wise group-equal MAE","selection_counts":sel_counts,"previous_pass_feature_count":len(prev_morph),"previous_pass_features":prev_morph})
result={"schema":"soilbin.q1.waveform.v2","response_status":RESPONSE_STATUS,"depth_mapping_status":DEPTH_STATUS,"validation":"Nested grouped CV; outer leave-one-VxW-sequence-out; inner GroupKFold; no random split","n_runs":51,"n_history_pairs":41,"n_groups":9,"models":MODEL_FAMILIES,"primary_coupled":get_metric("C_prev_waveform_coupled"),"delta_coupled":get_metric("D_delta_waveform_coupled"),"previous_peak_only":get_metric("B_prev_peak"),"conditions_only":get_metric("A_conditions"),"persistence":get_metric("C_prev_waveform_coupled","Persistence"),"qc_ok_sensitivity":get_metric("C_prev_waveform_coupled","Selected_inner",True),"external_validation":False}
jdump(ROOT/"summary_v2.json",result)
plt.rcParams.update({"figure.dpi":120,"savefig.dpi":180}); q=summary[(summary.scheme=="VW")&(summary.qc_only==False)&(summary.model=="Selected_inner")&(summary.complete_oof==True)].copy(); plt.figure(figsize=(8,4.8)); plt.barh(q.task,q.joint_Group_MAE); plt.xlabel("Joint group-equal MAE (N)"); plt.title("Leakage-safe history / waveform-derived ablation"); plt.tight_layout(); plt.savefig(FIG/"01_waveform_ablation.png"); plt.close()
for ti,t in enumerate(TARGET_NAMES,2):
    z=pred[(pred.task=="C_prev_waveform_coupled")&(pred.scheme=="VW")&(pred.qc_only==False)&(pred.model=="Selected_inner")]; plt.figure(figsize=(5,5)); plt.scatter(z[f"{t}_true"],z[f"{t}_pred"],alpha=.85); lo=min(z[f"{t}_true"].min(),z[f"{t}_pred"].min()); hi=max(z[f"{t}_true"].max(),z[f"{t}_pred"].max()); plt.plot([lo,hi],[lo,hi],"--"); plt.xlabel(f"Observed {t}"); plt.ylabel(f"OOF predicted {t}"); plt.tight_layout(); plt.savefig(FIG/f"0{ti}_{t}_observed_predicted.png"); plt.close()
pub={"checklist":{"data_qc":True,"leakage_guard_current_waveform_forbidden":True,"nested_grouped_cv":True,"random_split_used":False,"previous_peak_ablation":True,"previous_waveform_morphology_ablation":True,"coupled_LC1_LC5":True,"delta_model":True,"leave_speed_out":True,"leave_load_out":True,"group_calibrated_uncertainty":True,"qc_sensitivity":True,"external_validation":False,"depth_mapping_confirmed":False,"force_to_stress_conversion_confirmed":False},"limitations":["Previous-pass waveform morphology is derived from recorded load-cell traces; current-run waveform descriptors are not used as predictors.","Only 41 consecutive next-pass pairs across nine VxW groups are available.","One run has a QC REVIEW flag; primary all-data and QC-OK sensitivity are both reported.","Uncertainty intervals are calibrated within outer-training groups and are not external coverage guarantees.","Depth mapping remains provisional.","Response is calibrated vertical force proxy in N, not verified soil stress/kPa.","Independent external validation remains pending."],"figure_count":len(list(FIG.glob("*.png"))),"table_count":len(list(TABLE.glob("*.csv")))}
jdump(ROOT/"publication_manifest_v2.json",pub)
manifest={"schema":"soilbin.q1.waveform.v2","finished_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,"python":sys.version.split()[0],"cpu_count":os.cpu_count(),"n_jobs":N_JOBS,"source_csv_sha256":EXPECTED_CSV_SHA,"tasks":tasks,"models":MODEL_FAMILIES,"feature_sets":feature_sets,"response_status":RESPONSE_STATUS,"depth_mapping_status":DEPTH_STATUS,"external_validation":False,"evidence_files":[str(p.relative_to(ROOT)) for p in sorted(ROOT.rglob("*")) if p.is_file()]}
jdump(ROOT/"run_manifest_v2.json",manifest)
(ROOT/"FINAL_REPORT_V2.md").write_text("# SoilBin Q1/Q2 waveform-derived history benchmark V2\n\nPrimary scientific guard: calibrated vertical force proxy (N), not verified soil stress/kPa.\n\n"+f"History pairs: {len(hist)} across {hist.Group_VW.nunique()} V×W groups. Current-run waveform descriptors are forbidden as predictors.\n",encoding="utf-8")
phase("SOILBIN_WAVEFORM_V2_COMPLETE")
