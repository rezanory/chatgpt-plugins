# SoilBin Q1/Q2 V3 — external validation + complete delta/waveform ablation
from __future__ import annotations
import os, sys, json, math, hashlib, random, warnings, gzip, base64, itertools
from pathlib import Path
from io import StringIO
from datetime import datetime, timezone
from collections import Counter, defaultdict

SEED=20260914
random.seed(SEED); os.environ.setdefault("PYTHONHASHSEED",str(SEED))
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np
np.random.seed(SEED)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, SplineTransformer
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
try:
    from catboost import CatBoostRegressor
    HAVE_CAT=True
except Exception:
    HAVE_CAT=False
try:
    from xgboost import XGBRegressor
    HAVE_XGB=True
except Exception:
    HAVE_XGB=False
try:
    from lightgbm import LGBMRegressor
    HAVE_LGBM=True
except Exception:
    HAVE_LGBM=False

ROOT=Path("/kaggle/working/SOILBIN_V3")
TABLE=ROOT/"tables"; FIG=ROOT/"figures"; STATE=ROOT/"state"; EXTD=ROOT/"external_data"
for p in [ROOT,TABLE,FIG,STATE,EXTD]: p.mkdir(parents=True,exist_ok=True)

EXPECTED_MODEL_SHA="dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"
EXPECTED_CANADA_SHA="20ed0fd6b51c0fca7bb5e78e50cd7901e13d1ded3b03b3e13bff5df829592407"
EXPECTED_ERDC_SHA="9a445ba5080fa637eb89b4992e4c3c0b7eaa789ada0d1dfca2f2516f7fc316a6"
EXPECTED_CORROBORATION_SHA="6d28c811f1ae1dfcf868ec4cea110608178986f6133aaf6220f70e41926a9f40"
EXPECTED_REGISTRY_SHA="d4c54e2fee3df55e0155c6ed9c5ca82296ff0653bd53cabe6fb2da3abdddf5d1"
EXPECTED_MISSING={"V1W1T1","V2W3T2","V3W2T1"}
RESPONSE_STATUS="CALIBRATED_VERTICAL_FORCE_PROXY_N_PENDING_SENSOR_AREA_OR_STRESS_CALIBRATION"
DEPTH_STATUS="PROVISIONAL_LC5_5CM_LC1_15CM_PENDING_LAYOUT_CONFIRMATION"
TARGET_COLS=["LC1_peak_magnitude_N_delta","LC5_peak_magnitude_N_delta"]
TARGET_NAMES=["LC1_proxy_N","LC5_proxy_N"]
N_JOBS=max(1,min(4,(os.cpu_count() or 2)))

def phase(x): print(f"CGP_PHASE:{x}",flush=True)
def jdump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v: v.item() if hasattr(v,"item") else str(v)),encoding="utf-8")
def decode_gz_b64(payload, expected_sha):
    raw=gzip.decompress(base64.b64decode(payload.strip()))
    got=hashlib.sha256(raw).hexdigest()
    if got!=expected_sha: raise RuntimeError(f"SOURCE_FINGERPRINT_MISMATCH:{got}:{expected_sha}")
    return raw.decode("utf-8")
def group_equal_mae(y,p,groups):
    tmp=pd.DataFrame({"y":np.asarray(y,dtype=float),"p":np.asarray(p,dtype=float),"g":np.asarray(groups)})
    return float(tmp.assign(e=lambda d:(d.y-d.p).abs()).groupby("g").e.mean().mean())
def metrics_block(df_pred, expected_n):
    out={"complete_oof":bool(len(df_pred)==expected_n and df_pred.row_id.nunique()==expected_n),
         "expected_n":int(expected_n),"n":int(len(df_pred)),"n_groups":int(df_pred.Group_VW.nunique())}
    vals=[]
    for t in TARGET_NAMES:
        y=df_pred[f"y_{t}"].to_numpy(float); p=df_pred[f"p_{t}"].to_numpy(float)
        out[f"{t}_Group_MAE"]=group_equal_mae(y,p,df_pred.Group_VW)
        out[f"{t}_MAE"]=float(mean_absolute_error(y,p))
        out[f"{t}_RMSE"]=float(mean_squared_error(y,p)**0.5)
        out[f"{t}_Bias"]=float(np.mean(p-y))
        out[f"{t}_R2"]=float(r2_score(y,p))
        vals.append(out[f"{t}_Group_MAE"])
    out["joint_Group_MAE"]=float(np.mean(vals))
    return out
def qhigher(a,q):
    a=np.sort(np.asarray(a,float))
    if len(a)==0: return float("nan")
    return float(a[min(len(a)-1,max(0,int(math.ceil(q*len(a))-1)))])

phase("DATA_FREEZE")
model_raw=decode_gz_b64(MODEL_CSV_GZ_B64,EXPECTED_MODEL_SHA)
canada_raw=decode_gz_b64(CANADA_CSV_GZ_B64,EXPECTED_CANADA_SHA)
erdc_raw=decode_gz_b64(ERDC_CSV_GZ_B64,EXPECTED_ERDC_SHA)
corroboration_raw=decode_gz_b64(CORROBORATION_CSV_GZ_B64,EXPECTED_CORROBORATION_SHA)
reg_raw=base64.b64decode(EXTERNAL_REGISTRY_B64.strip())
if hashlib.sha256(reg_raw).hexdigest()!=EXPECTED_REGISTRY_SHA: raise RuntimeError("REGISTRY_FINGERPRINT_MISMATCH")
registry=json.loads(reg_raw.decode("utf-8"))
df=pd.read_csv(StringIO(model_raw)); canada=pd.read_csv(StringIO(canada_raw)); erdc=pd.read_csv(StringIO(erdc_raw)); corroboration=pd.read_csv(StringIO(corroboration_raw))
(EXTD/"external_canada_ayetan_2026.csv").write_text(canada_raw,encoding="utf-8")
(EXTD/"external_erdc_crrel_2009.csv").write_text(erdc_raw,encoding="utf-8")
(EXTD/"external_literature_corroboration_v3.csv").write_text(corroboration_raw,encoding="utf-8")
jdump(EXTD/"external_validation_registry.json",registry)

if len(df)!=51 or df.Run_ID.nunique()!=51: raise RuntimeError("EXPECTED_51_RUNS")
df["Group_VW"]=df.Run_ID.str.extract(r"^(V\dW\d)")[0]
df["Speed_kmh"]=df["Speed_level"].astype(float)
df["Load_kN"]=df["Weight_level"].astype(float)+1.0
df["Pass_T"]=df["Pass_T"].astype(int)
all_ids={f"V{v}W{w}T{t}" for v in range(1,4) for w in range(1,4) for t in range(1,7)}
if all_ids-set(df.Run_ID)!=EXPECTED_MISSING: raise RuntimeError("DESIGN_SLOT_MISMATCH")
for c in TARGET_COLS:
    if c not in df or df[c].isna().any(): raise RuntimeError(f"TARGET_INVALID:{c}")

prev=df.copy(); prev["Pass_T"]=prev["Pass_T"]+1
morph_suffixes=("baseline_noise_sigma_raw_kg","event_duration_ms","fwhm_ms","rise_time_ms","fall_time_ms","impulse_abs_Ns","impulse_signed_Ns","rms_event_N","mean_abs_event_N")
morph=[c for c in df.columns if c.startswith(("LC1_","LC5_")) and c.endswith(morph_suffixes)]
morph += [c for c in ["Peak_delay_LC5_minus_LC1_ms","Impulse_ratio_LC5_to_LC1"] if c in df.columns]
rename={c:f"prev_{c}" for c in set(morph+TARGET_COLS+["QC_status"])}
prev_small=prev[["Group_VW","Pass_T","QC_status"]+list(dict.fromkeys(morph+TARGET_COLS))].rename(columns=rename)
hist=df.merge(prev_small,on=["Group_VW","Pass_T"],how="left",validate="one_to_one")
hist=hist[hist[f"prev_{TARGET_COLS[0]}"].notna() & hist[f"prev_{TARGET_COLS[1]}"].notna()].copy().reset_index(drop=True)
if len(hist)!=41: raise RuntimeError(f"EXPECTED_41_HISTORY_PAIRS_GOT_{len(hist)}")
hist["row_id"]=np.arange(len(hist),dtype=int)
hist["pair_qc_ok"]=(hist["QC_status"].fillna("")=="OK") & (hist["prev_QC_status"].fillna("")=="OK")
conditions=["Load_kN","Speed_kmh","Pass_T"]
prev_peaks=[f"prev_{c}" for c in TARGET_COLS]
prev_morph=[f"prev_{c}" for c in morph if f"prev_{c}" not in prev_peaks]
TASKS={
 "A_conditions_abs":{"features":conditions,"delta":False},
 "B_prev_peak_abs":{"features":conditions+prev_peaks,"delta":False},
 "C_prev_waveform_abs":{"features":conditions+prev_peaks+prev_morph,"delta":False},
 "D_delta_prev_peak":{"features":conditions+prev_peaks,"delta":True},
 "E_delta_prev_peak_waveform":{"features":conditions+prev_peaks+prev_morph,"delta":True},
}
qc={
 "schema":"soilbin.q1.v3","created_utc":datetime.now(timezone.utc).isoformat(),
 "source_model_csv_sha256":EXPECTED_MODEL_SHA,"canada_csv_sha256":EXPECTED_CANADA_SHA,"erdc_csv_sha256":EXPECTED_ERDC_SHA,"corroboration_csv_sha256":EXPECTED_CORROBORATION_SHA,
 "external_registry_sha256":EXPECTED_REGISTRY_SHA,"valid_runs":51,"history_pairs":41,"groups":9,
 "pair_qc_ok_count":int(hist.pair_qc_ok.sum()),"pair_qc_review_count":int((~hist.pair_qc_ok).sum()),
 "missing_design_slots":sorted(EXPECTED_MISSING),
 "response_status":RESPONSE_STATUS,"depth_mapping_status":DEPTH_STATUS,
 "leakage_guard":"No current-run waveform or current-run peak feature is a predictor; only current operating conditions and previous-pass features are allowed.",
 "external_labels_used_for_tuning":False,"absolute_external_prediction_validation":False,
 "absolute_external_blocker":"External sources report stress in kPa while SoilBin response remains calibrated vertical force proxy in N; effective sensor area/direct stress calibration is required before absolute transfer."
}
jdump(ROOT/"qc_v3.json",qc)

def make_estimator(name, params):
    if name=="Linear":
        return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",LinearRegression())])
    if name=="Ridge":
        return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",Ridge(alpha=params["alpha"]))])
    if name=="PolynomialRidge":
        return Pipeline([("impute",SimpleImputer(strategy="median")),("poly",PolynomialFeatures(degree=2,include_bias=False)),("scale",StandardScaler()),("m",Ridge(alpha=params["alpha"]))])
    if name=="SplineRidgeGAM":
        return Pipeline([("impute",SimpleImputer(strategy="median")),("spline",SplineTransformer(n_knots=params["n_knots"],degree=3,include_bias=False)),("scale",StandardScaler()),("m",Ridge(alpha=params["alpha"]))])
    if name=="SVR":
        return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",SVR(C=params["C"],epsilon=params["epsilon"],gamma="scale"))])
    if name=="GPR":
        ker=ConstantKernel(1.0,(0.01,100.0))*Matern(length_scale=1.0,length_scale_bounds=(0.01,100.0),nu=1.5)+WhiteKernel(noise_level=1.0,noise_level_bounds=(1e-5,100.0))
        return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",GaussianProcessRegressor(kernel=ker,alpha=1e-6,normalize_y=True,random_state=SEED,n_restarts_optimizer=0))])
    if name=="RandomForest":
        return Pipeline([("impute",SimpleImputer(strategy="median")),("m",RandomForestRegressor(n_estimators=250,max_depth=params["max_depth"],min_samples_leaf=params["min_samples_leaf"],random_state=SEED,n_jobs=1,max_features=1.0))])
    if name=="ExtraTrees":
        return Pipeline([("impute",SimpleImputer(strategy="median")),("m",ExtraTreesRegressor(n_estimators=250,max_depth=params["max_depth"],min_samples_leaf=params["min_samples_leaf"],random_state=SEED,n_jobs=1,max_features=1.0))])
    if name=="HistGradientBoosting":
        return Pipeline([("impute",SimpleImputer(strategy="median")),("m",HistGradientBoostingRegressor(max_iter=200,learning_rate=params["learning_rate"],max_leaf_nodes=params["max_leaf_nodes"],l2_regularization=params["l2"],random_state=SEED))])
    if name=="CatBoost":
        return Pipeline([("impute",SimpleImputer(strategy="median")),("m",CatBoostRegressor(iterations=300,depth=params["depth"],learning_rate=params["learning_rate"],loss_function="MAE",verbose=False,random_seed=SEED,allow_writing_files=False,thread_count=1))])
    if name=="XGBoost":
        return Pipeline([("impute",SimpleImputer(strategy="median")),("m",XGBRegressor(n_estimators=300,max_depth=params["max_depth"],learning_rate=params["learning_rate"],subsample=0.9,colsample_bytree=0.9,objective="reg:squarederror",reg_lambda=1.0,random_state=SEED,n_jobs=1,tree_method="hist"))])
    if name=="LightGBM":
        return Pipeline([("impute",SimpleImputer(strategy="median")),("m",LGBMRegressor(n_estimators=300,num_leaves=params["num_leaves"],learning_rate=params["learning_rate"],min_child_samples=5,subsample=0.9,colsample_bytree=0.9,reg_lambda=1.0,random_state=SEED,n_jobs=1,verbose=-1))])
    if name=="MLP":
        return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",MLPRegressor(hidden_layer_sizes=params["hidden"],alpha=params["alpha"],learning_rate_init=0.01,max_iter=1200,early_stopping=False,random_state=SEED))])
    raise KeyError(name)

CANDIDATES=[
 ("Linear",{}),
 ("Ridge",{"alpha":1.0}),("Ridge",{"alpha":10.0}),
 ("PolynomialRidge",{"alpha":1.0}),("PolynomialRidge",{"alpha":10.0}),
 ("SplineRidgeGAM",{"n_knots":3,"alpha":1.0}),("SplineRidgeGAM",{"n_knots":4,"alpha":10.0}),
 ("SVR",{"C":10.0,"epsilon":0.1}),("SVR",{"C":100.0,"epsilon":0.1}),
 ("GPR",{}),
 ("RandomForest",{"max_depth":None,"min_samples_leaf":1}),("RandomForest",{"max_depth":4,"min_samples_leaf":2}),
 ("ExtraTrees",{"max_depth":None,"min_samples_leaf":1}),("ExtraTrees",{"max_depth":4,"min_samples_leaf":2}),
 ("HistGradientBoosting",{"learning_rate":0.05,"max_leaf_nodes":7,"l2":1.0}),
 ("MLP",{"hidden":(16,),"alpha":0.01}),("MLP",{"hidden":(32,16),"alpha":0.01}),
]
if HAVE_CAT:
    CANDIDATES += [("CatBoost",{"depth":3,"learning_rate":0.03}),("CatBoost",{"depth":4,"learning_rate":0.05})]
if HAVE_XGB:
    CANDIDATES += [("XGBoost",{"max_depth":2,"learning_rate":0.03}),("XGBoost",{"max_depth":3,"learning_rate":0.05})]
if HAVE_LGBM:
    CANDIDATES += [("LightGBM",{"num_leaves":7,"learning_rate":0.03}),("LightGBM",{"num_leaves":15,"learning_rate":0.05})]

def y_train_matrix(data, delta):
    y=data[TARGET_COLS].to_numpy(float)
    if delta:
        prev_y=data[prev_peaks].to_numpy(float)
        return y-prev_y
    return y
def final_from_model(pred_model, data, delta):
    p=np.column_stack(pred_model)
    if delta: p=p+data[prev_peaks].to_numpy(float)
    return p

def inner_select(train, task_name, search_rows, outer_label):
    spec=TASKS[task_name]; feats=spec["features"]; delta=spec["delta"]
    groups=train.Group_VW.to_numpy()
    unique=np.unique(groups)
    n_splits=min(4,len(unique))
    if n_splits<2: raise RuntimeError("INNER_GROUPS_LT_2")
    gkf=GroupKFold(n_splits=n_splits)
    X=train[feats]
    y_model=y_train_matrix(train,delta)
    best=None
    for ci,(name,params) in enumerate(CANDIDATES):
        inner_pred=np.full((len(train),2),np.nan,float)
        failed=None
        for tr,va in gkf.split(X,groups=groups):
            try:
                for ti in range(2):
                    est=make_estimator(name,params)
                    est.fit(X.iloc[tr],y_model[tr,ti])
                    inner_pred[va,ti]=est.predict(X.iloc[va])
            except Exception as exc:
                failed=f"{type(exc).__name__}:{str(exc)[:160]}"
                break
        if failed or np.isnan(inner_pred).any():
            score=float("inf")
        else:
            final=inner_pred + (train[prev_peaks].to_numpy(float) if delta else 0.0)
            s=[group_equal_mae(train[TARGET_COLS[ti]],final[:,ti],groups) for ti in range(2)]
            score=float(np.mean(s))
        row={"outer":str(outer_label),"task":task_name,"candidate_index":ci,"model":name,"params":json.dumps(params,sort_keys=True,default=list),"inner_joint_Group_MAE":score if math.isfinite(score) else None,"failed":failed}
        search_rows.append(row)
        if math.isfinite(score) and (best is None or score<best[0]-1e-12):
            best=(score,name,params)
    if best is None: raise RuntimeError(f"NO_VALID_CANDIDATE:{task_name}:{outer_label}")
    return best

def outer_labels(data, scheme):
    if scheme=="VW": return list(pd.unique(data.Group_VW)), "Group_VW"
    if scheme=="speed": return sorted(pd.unique(data.Speed_kmh)), "Speed_kmh"
    if scheme=="load": return sorted(pd.unique(data.Load_kN)), "Load_kN"
    raise KeyError(scheme)

def run_task(data, task_name, scheme="VW", qc_only=False):
    d=data[data.pair_qc_ok].copy() if qc_only else data.copy()
    labels,col=outer_labels(d,scheme)
    pred_rows=[]; search_rows=[]; selection_rows=[]; interval_rows=[]
    expected_n=len(d)
    for fold,label in enumerate(labels):
        te=d[d[col]==label].copy(); tr=d[d[col]!=label].copy()
        if len(te)==0 or tr.Group_VW.nunique()<2: continue
        best_score,name,params=inner_select(tr,task_name,search_rows,f"{scheme}:{label}")
        spec=TASKS[task_name]; feats=spec["features"]; delta=spec["delta"]
        Xtr=tr[feats]; Xte=te[feats]; ym=y_train_matrix(tr,delta)
        pred_model=[]; calib_abs=[[],[]]
        # Calibration residuals: inner grouped OOF using selected family/params only.
        ig=tr.Group_VW.to_numpy(); gkf=GroupKFold(n_splits=min(4,tr.Group_VW.nunique()))
        inner_selected=np.full((len(tr),2),np.nan)
        for itr,iva in gkf.split(Xtr,groups=ig):
            for ti in range(2):
                est=make_estimator(name,params); est.fit(Xtr.iloc[itr],ym[itr,ti]); inner_selected[iva,ti]=est.predict(Xtr.iloc[iva])
        inner_final=inner_selected + (tr[prev_peaks].to_numpy(float) if delta else 0.0)
        for ti in range(2): calib_abs[ti]=np.abs(tr[TARGET_COLS[ti]].to_numpy(float)-inner_final[:,ti])
        for ti in range(2):
            est=make_estimator(name,params); est.fit(Xtr,ym[:,ti]); pred_model.append(est.predict(Xte))
        final=final_from_model(pred_model,te,delta)
        selection_rows.append({"task":task_name,"scheme":scheme,"qc_only":bool(qc_only),"outer_label":str(label),"model":name,"params":json.dumps(params,sort_keys=True,default=list),"inner_joint_Group_MAE":best_score})
        for j,(_,r) in enumerate(te.iterrows()):
            row={"row_id":int(r.row_id),"Run_ID":r.Run_ID,"Group_VW":r.Group_VW,"Speed_kmh":float(r.Speed_kmh),"Load_kN":float(r.Load_kN),"Pass_T":int(r.Pass_T),
                 "task":task_name,"scheme":scheme,"qc_only":bool(qc_only),"model":"Selected_inner","outer_label":str(label)}
            for ti,t in enumerate(TARGET_NAMES):
                row[f"y_{t}"]=float(r[TARGET_COLS[ti]]); row[f"p_{t}"]=float(final[j,ti])
                for nominal in (0.80,0.95):
                    hw=qhigher(calib_abs[ti],nominal)
                    interval_rows.append({"row_id":int(r.row_id),"Run_ID":r.Run_ID,"Group_VW":r.Group_VW,"task":task_name,"scheme":scheme,"qc_only":bool(qc_only),"target":t,"nominal":nominal,
                                          "y":float(r[TARGET_COLS[ti]]),"pred":float(final[j,ti]),"half_width_N":hw,
                                          "covered":bool(abs(float(r[TARGET_COLS[ti]])-float(final[j,ti]))<=hw)})
            pred_rows.append(row)
    pred=pd.DataFrame(pred_rows)
    if pred.row_id.nunique()!=expected_n: raise RuntimeError(f"INCOMPLETE_OOF:{task_name}:{scheme}:{qc_only}:{pred.row_id.nunique()}:{expected_n}")
    return pred,pd.DataFrame(search_rows),pd.DataFrame(selection_rows),pd.DataFrame(interval_rows)

phase("NESTED_GROUPED_CV_START")
pred_all=[]; search_all=[]; sel_all=[]; int_all=[]
for i,task in enumerate(TASKS,1):
    phase(f"CORE_{i}_{task}")
    p,s,z,u=run_task(hist,task,"VW",False)
    pred_all.append(p); search_all.append(s); sel_all.append(z); int_all.append(u)
    jdump(STATE/"progress_v3.json",{"completed_tasks":list(TASKS)[:i],"pred_rows":int(sum(len(x) for x in pred_all)),"search_rows":int(sum(len(x) for x in search_all))})

phase("ROBUSTNESS")
for task in ["D_delta_prev_peak","E_delta_prev_peak_waveform"]:
    for scheme in ["speed","load"]:
        p,s,z,u=run_task(hist,task,scheme,False); pred_all.append(p); search_all.append(s); sel_all.append(z); int_all.append(u)
    p,s,z,u=run_task(hist,task,"VW",True); pred_all.append(p); search_all.append(s); sel_all.append(z); int_all.append(u)

pred=pd.concat(pred_all,ignore_index=True); search=pd.concat(search_all,ignore_index=True); sels=pd.concat(sel_all,ignore_index=True); ints=pd.concat(int_all,ignore_index=True)
pred.to_csv(TABLE/"oof_predictions_v3.csv",index=False)
search.to_csv(TABLE/"hyperparameter_search_v3.csv",index=False)
sels.to_csv(TABLE/"selection_v3.csv",index=False)
ints.to_csv(TABLE/"uncertainty_intervals_v3.csv",index=False)

# Baselines on primary 41 history pairs
base_rows=[]
for model in ["Persistence","TrainMedian"]:
    for _,r in hist.iterrows():
        row={"row_id":int(r.row_id),"Run_ID":r.Run_ID,"Group_VW":r.Group_VW,"task":"BASELINE","scheme":"VW","qc_only":False,"model":model}
        for ti,t in enumerate(TARGET_NAMES):
            if model=="Persistence": val=float(r[prev_peaks[ti]])
            else:
                tr=hist[hist.Group_VW!=r.Group_VW]
                val=float(tr[TARGET_COLS[ti]].median())
            row[f"y_{t}"]=float(r[TARGET_COLS[ti]]); row[f"p_{t}"]=val
        base_rows.append(row)
base=pd.DataFrame(base_rows); base.to_csv(TABLE/"baseline_predictions_v3.csv",index=False)

summary_rows=[]
for (task,scheme,qc),g in pred.groupby(["task","scheme","qc_only"],sort=False):
    m=metrics_block(g,len(hist[hist.pair_qc_ok]) if qc else len(hist))
    m.update({"task":task,"scheme":scheme,"qc_only":bool(qc),"model":"Selected_inner"}); summary_rows.append(m)
for model,g in base.groupby("model"):
    m=metrics_block(g,len(hist)); m.update({"task":"BASELINE","scheme":"VW","qc_only":False,"model":model}); summary_rows.append(m)
summary=pd.DataFrame(summary_rows); summary.to_csv(TABLE/"summary_v3.csv",index=False)

# Primary internal champion from five primary VW tasks only, selected without external labels.
primary=summary[(summary.scheme=="VW") & (~summary.qc_only) & (summary.model=="Selected_inner")].sort_values("joint_Group_MAE")
champ=primary.iloc[0].to_dict()

# Direct requested ablation: Delta + previous peak only vs Delta + previous peak + waveform.
d=pred[(pred.task=="D_delta_prev_peak")&(pred.scheme=="VW")&(~pred.qc_only)].copy()
e=pred[(pred.task=="E_delta_prev_peak_waveform")&(pred.scheme=="VW")&(~pred.qc_only)].copy()
group_cmp=[]
for g in sorted(hist.Group_VW.unique()):
    dg=d[d.Group_VW==g]; eg=e[e.Group_VW==g]
    dmae=np.mean([mean_absolute_error(dg[f"y_{t}"],dg[f"p_{t}"]) for t in TARGET_NAMES])
    emae=np.mean([mean_absolute_error(eg[f"y_{t}"],eg[f"p_{t}"]) for t in TARGET_NAMES])
    group_cmp.append({"Group_VW":g,"D_delta_prev_peak_joint_MAE":float(dmae),"E_delta_peak_waveform_joint_MAE":float(emae),"improvement_D_minus_E":float(dmae-emae)})
cmpdf=pd.DataFrame(group_cmp); cmpdf.to_csv(TABLE/"delta_waveform_paired_groups_v3.csv",index=False)
obs=float(cmpdf.improvement_D_minus_E.mean())
diff=cmpdf.improvement_D_minus_E.to_numpy(float)
perm=np.array([np.mean(diff*np.array(signs)) for signs in itertools.product([-1,1],repeat=len(diff))])
p_one=float(np.sum(perm>=obs-1e-12)/len(perm))
rng=np.random.default_rng(SEED)
boot=np.array([np.mean(rng.choice(diff,size=len(diff),replace=True)) for _ in range(20000)])
ci=[float(np.quantile(boot,0.025)),float(np.quantile(boot,0.975))]
dmet=metrics_block(d,len(hist)); emet=metrics_block(e,len(hist))
paired={
 "comparison":"D_delta_prev_peak vs E_delta_prev_peak_waveform",
 "D_joint_Group_MAE":dmet["joint_Group_MAE"],"E_joint_Group_MAE":emet["joint_Group_MAE"],
 "relative_improvement_pct":float(100*(dmet["joint_Group_MAE"]-emet["joint_Group_MAE"])/dmet["joint_Group_MAE"]),
 "mean_group_paired_improvement_N":obs,"group_bootstrap_95CI_N":ci,
 "exact_group_signflip_one_sided_p":p_one,"n_groups":int(len(diff)),
 "interpretation_guard":"Group-level paired evidence only; nine VxW groups limit inferential precision."
}
jdump(ROOT/"delta_peak_vs_waveform_v3.json",paired)

# Uncertainty calibration
cov=[]
for (task,scheme,qc,target,nom),g in ints.groupby(["task","scheme","qc_only","target","nominal"]):
    cov.append({"task":task,"scheme":scheme,"qc_only":bool(qc),"target":target,"nominal":float(nom),"n":int(len(g)),
                "empirical_coverage":float(g.covered.mean()),"mean_half_width_N":float(g.half_width_N.mean())})
coverage=pd.DataFrame(cov); coverage.to_csv(TABLE/"uncertainty_calibration_v3.csv",index=False)
jdump(ROOT/"uncertainty_calibration_v3.json",{"method":"Inner GroupKFold absolute-residual higher-quantile calibration using outer-training groups only; development coverage, not external guarantee.","rows":cov})

# Internal condition effects, for external directional corroboration (exploratory, 9 clusters).
phase("INTERNAL_EFFECTS")
effect_rows=[]
try:
    import statsmodels.formula.api as smf
    long=[]
    for _,r in df.iterrows():
        for ti,t in enumerate(TARGET_NAMES):
            long.append({"Group_VW":r.Group_VW,"Load_kN":r.Load_kN,"Speed_kmh":r.Speed_kmh,"Pass_T":r.Pass_T,"target":t,"proxy_N":float(r[TARGET_COLS[ti]])})
    ldf=pd.DataFrame(long)
    for target in TARGET_NAMES:
        sub=ldf[ldf.target==target]
        fit=smf.ols("proxy_N ~ Load_kN + Speed_kmh + Pass_T + Load_kN:Pass_T + Load_kN:Speed_kmh",data=sub).fit(cov_type="cluster",cov_kwds={"groups":sub.Group_VW})
        for term,val in fit.params.items():
            effect_rows.append({"target":target,"term":term,"coef":float(val),"se":float(fit.bse[term]),"p":float(fit.pvalues[term]),"clusters":9})
except Exception as exc:
    effect_rows=[{"error":f"{type(exc).__name__}:{str(exc)[:300]}"}]
pd.DataFrame(effect_rows).to_csv(TABLE/"internal_effects_v3.csv",index=False)

phase("EXTERNAL_VALIDATION")
# Canada independent numeric external effects.
can_effects=[]
for treatment in ["inflated","deflated"]:
    sub=canada[canada.treatment==treatment]
    for site in sorted(sub.site.unique()):
        for depth in sorted(sub.depth_cm.unique()):
            z=sub[(sub.site==site)&(sub.depth_cm==depth)].set_index("wheel_position")
            if {"front","rear"}<=set(z.index):
                can_effects.append({"contrast":"rear_minus_front","treatment":treatment,"site":site,"depth_cm":float(depth),
                                    "difference_kpa":float(z.loc["rear","peak_mean_stress_kpa"]-z.loc["front","peak_mean_stress_kpa"]),
                                    "ratio":float(z.loc["rear","peak_mean_stress_kpa"]/z.loc["front","peak_mean_stress_kpa"]) if z.loc["front","peak_mean_stress_kpa"]!=0 else None})
for site in sorted(canada.site.unique()):
    for wheelpos in ["front","rear"]:
        for depth in sorted(canada.depth_cm.unique()):
            z=canada[(canada.site==site)&(canada.wheel_position==wheelpos)&(canada.depth_cm==depth)].set_index("treatment")
            if {"inflated","deflated"}<=set(z.index):
                a=float(z.loc["inflated","peak_mean_stress_kpa"]); b=float(z.loc["deflated","peak_mean_stress_kpa"])
                can_effects.append({"contrast":"inflated_minus_deflated","wheel_position":wheelpos,"site":site,"depth_cm":float(depth),
                                    "difference_kpa":a-b,"relative_reduction_deflated_pct":float(100*(a-b)/a) if a else None})
for treatment in ["inflated","deflated"]:
    for site in sorted(canada.site.unique()):
        for wheelpos in ["front","rear"]:
            z=canada[(canada.treatment==treatment)&(canada.site==site)&(canada.wheel_position==wheelpos)].set_index("depth_cm")
            if {15,50}<=set(z.index):
                a=float(z.loc[15,"peak_mean_stress_kpa"]); b=float(z.loc[50,"peak_mean_stress_kpa"])
                can_effects.append({"contrast":"depth_15_minus_50","treatment":treatment,"wheel_position":wheelpos,"site":site,
                                    "difference_kpa":a-b,"ratio_15_to_50":float(a/b) if b else None})
can_eff=pd.DataFrame(can_effects); can_eff.to_csv(TABLE/"external_canada_effects_v3.csv",index=False)

can_summary={
 "n_rows":int(len(canada)),
 "rear_gt_front_fraction":float((can_eff[can_eff.contrast=="rear_minus_front"].difference_kpa>0).mean()),
 "deflated_lower_than_inflated_fraction":float((can_eff[can_eff.contrast=="inflated_minus_deflated"].difference_kpa>0).mean()),
 "depth_15_gt_50_fraction":float((can_eff[can_eff.contrast=="depth_15_minus_50"].difference_kpa>0).mean()),
 "role":"independent numeric external physics/trend validation",
 "absolute_prediction_test":False,
 "reason_not_absolute":"kPa external stress cannot be directly compared with SoilBin force proxy N until sensor effective area/direct stress calibration is established."
}

# ERDC independent evidence. Only MDT rows are true sequential repeated passes; CIV rows are separate trials.
erdc_effects=[]
mdt=erdc[erdc.vehicle=="MDT"].copy()
for keys,g in mdt.groupby(["vehicle","mode","soil","depth_cm"],dropna=False):
    g=g.sort_values("trial_pass")
    x=g.trial_pass.to_numpy(float); y=g.measured_center_pressure_kpa.to_numpy(float)
    if len(g)>=2:
        slope=float(np.polyfit(x,y,1)[0])
        erdc_effects.append({"evidence_type":"sequential_pass","vehicle":keys[0],"mode":keys[1],"soil":keys[2],"depth_cm":float(keys[3]),"n":int(len(g)),
                             "pass_slope_kpa_per_pass":slope,"first_to_last_pct":float(100*(y[-1]-y[0])/y[0]) if y[0] else None,
                             "monotonic_increase":bool(np.all(np.diff(y)>=0)),"monotonic_decrease":bool(np.all(np.diff(y)<=0))})
civ=erdc[erdc.vehicle=="CIV"].copy()
for keys,g in civ.groupby(["vehicle","mode","soil","depth_cm"],dropna=False):
    x=g.vertical_force_lb.to_numpy(float); y=g.measured_center_pressure_kpa.to_numpy(float)
    slope=float(np.polyfit(x,y,1)[0]) if len(g)>=2 and np.ptp(x)>0 else None
    erdc_effects.append({"evidence_type":"independent_trial_force_pressure","vehicle":keys[0],"mode":keys[1],"soil":keys[2],"depth_cm":float(keys[3]),"n":int(len(g)),
                         "pressure_vs_force_slope_kpa_per_lb":slope,"mean_pressure_kpa":float(np.mean(y))})
erdc_eff=pd.DataFrame(erdc_effects); erdc_eff.to_csv(TABLE/"external_erdc_pass_effects_v3.csv",index=False)
mdt_eff=erdc_eff[erdc_eff.evidence_type=="sequential_pass"]
erdc_summary={"n_rows":int(len(erdc)),"sequential_mdt_rows":int(len(mdt)),"civ_trial_rows":int(len(civ)),"groups":erdc_effects,
              "mdt_positive_pass_slope_fraction":float((mdt_eff.pass_slope_kpa_per_pass>0).mean()) if len(mdt_eff) else None,
              "role":"independent numeric external evidence: MDT sequential-pass validation plus CIV force/pressure trial corroboration",
              "device_limitation":"Original ERDC report flags pressure-pad reliability and wheel-load synchronization limitations; retained in interpretation."}

# Corroboration registry is never used for fitting or selection.
ext_summary={
 "schema":"soilbin.q1.v3.external",
 "independent_numeric_rows":int(len(canada)+len(erdc)),
 "independent_numeric_datasets":["EV_CANADA_AYETAN_2026","EV_ERDC_CRREL_2009"],
 "corroboration_datasets":[r["id"] for r in registry if "corroboration" in r["role"]],
 "near_domain_replication":[r["id"] for r in registry if r["role"]=="near_domain_replication_only"],
 "structured_corroboration_rows":int(len(corroboration)),
 "canada":can_summary,"erdc":erdc_summary,
 "external_labels_used_for_tuning":False,
 "external_validation_level":"independent numeric trend/physics validation plus literature corroboration; not absolute cross-unit prediction",
 "absolute_external_validation":False,
 "absolute_external_blocker":"Sensor effective area/direct N-to-kPa calibration and confirmed depth mapping are required."
}
jdump(ROOT/"external_validation_v3.json",ext_summary)

# Selection/model availability
avail={"CatBoost":HAVE_CAT,"XGBoost":HAVE_XGB,"LightGBM":HAVE_LGBM,"candidate_count":len(CANDIDATES)}
jdump(ROOT/"model_availability_v3.json",avail)
sel_counts={k:dict(Counter(sels[(sels.task==k)&(sels.scheme=="VW")&(~sels.qc_only)].model)) for k in TASKS}
jdump(ROOT/"model_selection_v3.json",{"primary_champion_internal_only":champ,"selection_counts":sel_counts,"candidate_catalog":[{"model":n,"params":p} for n,p in CANDIDATES]})

# Figures
phase("FIGURES")
fig_primary=primary.sort_values("joint_Group_MAE")
plt.figure(figsize=(10,5)); plt.bar(fig_primary.task,fig_primary.joint_Group_MAE); plt.ylabel("Joint group-equal MAE (N)"); plt.xticks(rotation=30,ha="right"); plt.tight_layout(); plt.savefig(FIG/"01_internal_ablation_v3.png",dpi=180); plt.close()
plt.figure(figsize=(8,4)); x=np.arange(len(cmpdf)); plt.bar(x-0.18,cmpdf.D_delta_prev_peak_joint_MAE,width=0.36,label="Delta + previous peak"); plt.bar(x+0.18,cmpdf.E_delta_peak_waveform_joint_MAE,width=0.36,label="+ previous waveform"); plt.xticks(x,cmpdf.Group_VW); plt.ylabel("Joint MAE (N)"); plt.legend(); plt.tight_layout(); plt.savefig(FIG/"02_delta_peak_vs_waveform_groups_v3.png",dpi=180); plt.close()
cplot=canada.groupby(["depth_cm","treatment"]).peak_mean_stress_kpa.mean().reset_index()
plt.figure(figsize=(7,4))
for tr,g in cplot.groupby("treatment"):
    plt.plot(g.depth_cm,g.peak_mean_stress_kpa,marker="o",label=tr)
plt.xlabel("Depth (cm)"); plt.ylabel("Mean external peak stress (kPa)"); plt.legend(); plt.tight_layout(); plt.savefig(FIG/"03_external_canada_depth_v3.png",dpi=180); plt.close()
plt.figure(figsize=(8,4))
for keys,g in erdc.groupby(["vehicle","mode","soil"]):
    plt.plot(g.trial_pass,g.measured_center_pressure_kpa,marker="o",label="/".join(keys))
plt.xlabel("Trial/pass"); plt.ylabel("External center pressure (kPa)"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG/"04_external_erdc_pass_v3.png",dpi=180); plt.close()

# Publication manifest and summary
result={
 "schema":"soilbin.q1.v3",
 "response_status":RESPONSE_STATUS,"depth_mapping_status":DEPTH_STATUS,
 "validation":"Nested grouped CV; outer leave-one-VxW-sequence-out; inner GroupKFold; no random split; external labels never used for tuning.",
 "n_runs":51,"n_history_pairs":41,"n_groups":9,
 "tasks":list(TASKS),"primary_champion_internal_only":champ,
 "delta_peak_vs_waveform":paired,
 "external_validation":ext_summary,
 "external_validation_absolute":False,
 "external_validation_trend_physics":True,
 "external_validation_independent_numeric_rows":int(len(canada)+len(erdc)),
 "external_validation_pending_for_absolute_transfer":"effective sensor area/direct stress calibration",
}
jdump(ROOT/"summary_v3.json",result)
pub={
 "checklist":{
  "data_qc":True,"current_waveform_leakage_guard":True,"nested_grouped_cv":True,"random_split_used":False,
  "delta_previous_peak_ablation":True,"delta_previous_peak_plus_waveform_ablation":True,
  "paired_group_inference":True,"xgboost_benchmarked_if_available":HAVE_XGB,"lightgbm_benchmarked_if_available":HAVE_LGBM,
  "catboost_benchmarked_if_available":HAVE_CAT,"leave_speed_out":True,"leave_load_out":True,
  "qc_sensitivity":True,"group_calibrated_uncertainty":True,
  "independent_external_numeric_datasets":True,"external_labels_used_for_tuning":False,
  "absolute_external_validation":False,"depth_mapping_confirmed":False,"force_to_stress_conversion_confirmed":False
 },
 "limitations":[
  "Only 41 consecutive next-pass pairs across nine VxW groups are available.",
  "One run has a QC REVIEW flag; all-data and QC-OK sensitivity are both reported.",
  "External numeric data are independent but reported in kPa, whereas SoilBin response is a calibrated vertical force proxy in N.",
  "Therefore V3 external validation is a trend/physics validation, not an absolute cross-unit prediction test.",
  "ERDC pressure-pad study reports device/synchronization limitations; those rows are retained with caution.",
  "Depth mapping remains provisional and independent external absolute validation awaits effective sensor area/direct stress calibration."
 ],
 "external_sources":[{"id":r["id"],"role":r["role"],"doi":r.get("doi"),"url":r["url"]} for r in registry],
 "figure_count":4
}
jdump(ROOT/"publication_manifest_v3.json",pub)
manifest={
 "schema":"soilbin.q1.v3","finished_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,"python":sys.version.split()[0],
 "cpu_count":os.cpu_count(),"n_jobs":N_JOBS,
 "source_fingerprints":{"model_csv":EXPECTED_MODEL_SHA,"canada_csv":EXPECTED_CANADA_SHA,"erdc_csv":EXPECTED_ERDC_SHA,"corroboration_csv":EXPECTED_CORROBORATION_SHA,"external_registry":EXPECTED_REGISTRY_SHA},
 "models_available":avail,"tasks":list(TASKS),"feature_sets":{k:v["features"] for k,v in TASKS.items()},
 "response_status":RESPONSE_STATUS,"depth_mapping_status":DEPTH_STATUS,
 "external_labels_used_for_tuning":False,"absolute_external_validation":False,
 "evidence_files":sorted([p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()])
}
jdump(ROOT/"run_manifest_v3.json",manifest)
(ROOT/"FINAL_REPORT_V3.md").write_text(
    "# SoilBin Q1/Q2 V3 — External Validation + Complete Delta/Waveform Ablation\n\n"
    f"Internal champion (selected without external labels): **{champ['task']}**, joint group-equal MAE = **{champ['joint_Group_MAE']:.3f} N**.\n\n"
    f"Requested paired ablation: Delta+previous-peak = **{paired['D_joint_Group_MAE']:.3f} N** vs Delta+previous-peak+previous-waveform = **{paired['E_joint_Group_MAE']:.3f} N**; relative improvement = **{paired['relative_improvement_pct']:.2f}%**.\n\n"
    f"Independent numeric external evidence: **{len(canada)+len(erdc)} rows** from Canada 2026 and ERDC/CRREL 2009. External labels were not used in tuning.\n\n"
    "Important guard: external sources report stress in kPa while the SoilBin response remains a calibrated vertical force proxy in N. V3 therefore performs independent trend/physics validation, not an absolute N-to-kPa prediction claim. Absolute external validation requires effective sensor area/direct stress calibration and confirmed depth mapping.\n",
    encoding="utf-8")
phase("COMPLETE")
print("SOILBIN_V3_COMPLETE",json.dumps({"champion":champ["task"],"joint_Group_MAE":champ["joint_Group_MAE"],"delta_waveform_improvement_pct":paired["relative_improvement_pct"],"external_numeric_rows":len(canada)+len(erdc),"absolute_external_validation":False},sort_keys=True),flush=True)
