# SoilBin Q1/Q2 V4 — previous-pass raw waveform representation learning
from __future__ import annotations
import os, sys, json, math, hashlib, random, warnings, gzip, base64, itertools, lzma, struct
from pathlib import Path
from io import StringIO
from datetime import datetime, timezone
from collections import Counter

SEED=20260914
random.seed(SEED); os.environ.setdefault('PYTHONHASHSEED',str(SEED))
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
np.random.seed(SEED)
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import dct
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings('ignore')
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
try:
    import torch
    import torch.nn as nn
    HAVE_TORCH=True
except Exception:
    HAVE_TORCH=False

ROOT=Path('/kaggle/working/SOILBIN_V4')
TABLE=ROOT/'tables'; FIG=ROOT/'figures'; STATE=ROOT/'state'
for p in (ROOT,TABLE,FIG,STATE): p.mkdir(parents=True,exist_ok=True)
EXPECTED_MODEL_SHA='dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059'
EXPECTED_RAW_SHA='ea227f412827a962a1876ceb9aeb9946034b435052e9e74115e32bd7dca73cb0'
EXPECTED_WAVE_COMPACT_BLOB_SHA='786cd72be8a6e37aa0e34cee044f70b047c9a84f5a9230abc9028117db765677'
EXPECTED_WAVE_COMPACT_XZ_SHA='6ccca93e5e146e67908e37cf07575bdf1ed44b0f3b84fcf0ee3198cd2ffa02c9'
EXPECTED_MISSING={'V1W1T1','V2W3T2','V3W2T1'}
RESPONSE_STATUS='CALIBRATED_VERTICAL_FORCE_PROXY_N_PENDING_SENSOR_AREA_OR_STRESS_CALIBRATION'
DEPTH_STATUS='PROVISIONAL_LC5_5CM_LC1_15CM_PENDING_LAYOUT_CONFIRMATION'
TARGET_COLS=['LC1_peak_magnitude_N_delta','LC5_peak_magnitude_N_delta']
TARGET_NAMES=['LC1_proxy_N','LC5_proxy_N']
RESAMPLE_N=256
N_JOBS=max(1,min(4,(os.cpu_count() or 2)))
V3_BASELINE_REP='V3_D_DELTA_PREV_PEAK_FROZEN'
V3_BASELINE_EXPECTED_JOINT_GROUP_MAE=24.28878365273039
V3_BASELINE_REPRO_TOL_N=1e-6
V5_POLICY_SHA256='c3c802b74ee0426e23d6c8ea6e4fd9ca26cf1128f5da3a32ca2b0b20aa6c7c02'

def phase(x): print(f'CGP_PHASE:{x}',flush=True)
def jdump(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def decode_gz_b64(payload, expected_sha):
    raw=gzip.decompress(base64.b64decode(payload.strip()))
    got=hashlib.sha256(raw).hexdigest()
    if got!=expected_sha: raise RuntimeError(f'SOURCE_FINGERPRINT_MISMATCH:{got}:{expected_sha}')
    return raw.decode('utf-8')
def decode_wave_compact_b64(payload):
    xz=base64.b64decode(payload.strip())
    got_xz=hashlib.sha256(xz).hexdigest()
    if got_xz!=EXPECTED_WAVE_COMPACT_XZ_SHA:
        raise RuntimeError(f'WAVE_COMPACT_XZ_FINGERPRINT_MISMATCH:{got_xz}')
    blob=lzma.decompress(xz)
    got_blob=hashlib.sha256(blob).hexdigest()
    if got_blob!=EXPECTED_WAVE_COMPACT_BLOB_SHA:
        raise RuntimeError(f'WAVE_COMPACT_BLOB_FINGERPRINT_MISMATCH:{got_blob}')
    hlen=struct.unpack('<I',blob[:4])[0]
    meta=json.loads(blob[4:4+hlen].decode('utf-8'))
    n=int(meta['n']); off=4+hlen
    run=np.frombuffer(blob,dtype=np.uint8,count=n,offset=off).copy(); off+=n
    time=np.frombuffer(blob,dtype='<u4',count=n,offset=off).copy(); off+=4*n
    f1=np.frombuffer(blob,dtype='<f8',count=n,offset=off).copy(); off+=8*n
    f5=np.frombuffer(blob,dtype='<f8',count=n,offset=off).copy(); off+=8*n
    if off!=len(blob): raise RuntimeError('WAVE_COMPACT_TRAILING_BYTES')
    runs=meta['runs']
    frame=pd.DataFrame({'Run_ID':[runs[int(i)] for i in run],
                        'time_ms':time.astype(float),
                        'LC1_force_delta_N':f1,
                        'LC5_force_delta_N':f5})
    if len(frame)!=80958 or frame.Run_ID.nunique()!=51:
        raise RuntimeError('WAVE_COMPACT_SHAPE_MISMATCH')
    return frame
def group_equal_mae(y,p,g):
    t=pd.DataFrame({'y':np.asarray(y,float),'p':np.asarray(p,float),'g':np.asarray(g)})
    return float(t.assign(e=lambda d:(d.y-d.p).abs()).groupby('g').e.mean().mean())
def metrics(y,p,groups):
    out={}
    for i,name in enumerate(TARGET_NAMES):
        yy=np.asarray(y[:,i],float); pp=np.asarray(p[:,i],float)
        out[f'{name}_Group_MAE']=group_equal_mae(yy,pp,groups)
        out[f'{name}_MAE']=float(mean_absolute_error(yy,pp))
        out[f'{name}_RMSE']=float(mean_squared_error(yy,pp)**0.5)
        out[f'{name}_Bias']=float(np.mean(pp-yy))
        out[f'{name}_R2']=float(r2_score(yy,pp))
    out['joint_Group_MAE']=float(np.mean([out[f'{n}_Group_MAE'] for n in TARGET_NAMES]))
    return out

def event_resample(g):
    g=g.sort_values('time_ms')
    t=g.time_ms.to_numpy(float)
    x=np.column_stack([g.LC1_force_delta_N.to_numpy(float),g.LC5_force_delta_N.to_numpy(float)])
    # Previous-pass waveform only. Event bounds are derived from that same previous trace, never current-pass data.
    mag=np.sqrt(np.sum(x*x,axis=1))
    n0=max(10,min(len(mag)//8,200))
    base=np.r_[mag[:n0],mag[-n0:]] if len(mag)>=2*n0 else mag
    med=float(np.median(base)); mad=float(np.median(np.abs(base-med)))
    sigma=max(1e-9,1.4826*mad)
    peak=float(np.max(mag)); thr=max(med+5*sigma,0.10*peak)
    idx=np.flatnonzero(mag>=thr)
    if len(idx)<3:
        c=int(np.argmax(mag)); half=min(max(20,len(mag)//10),len(mag)//2)
        a=max(0,c-half); b=min(len(mag)-1,c+half)
    else:
        a=int(idx[0]); b=int(idx[-1]); pad=max(5,int(round(0.10*(b-a+1))))
        a=max(0,a-pad); b=min(len(mag)-1,b+pad)
    if b<=a: raise RuntimeError('INVALID_EVENT_WINDOW')
    tt=t[a:b+1]; xx=x[a:b+1]
    u=np.linspace(tt[0],tt[-1],RESAMPLE_N)
    y=np.column_stack([np.interp(u,tt,xx[:,0]),np.interp(u,tt,xx[:,1])])
    return y.astype(np.float32), {'event_start_ms':float(tt[0]),'event_end_ms':float(tt[-1]),'event_duration_ms':float(tt[-1]-tt[0]),'peak_combined_N':peak,'threshold_N':thr,'raw_n':int(len(g))}

def dct_features(w,k=32):
    feats=[]
    for ch in range(2):
        c=dct(w[:,ch],type=2,norm='ortho')[:k]
        feats.extend(c.tolist())
    return np.asarray(feats,float)
def spectral_features(w,k=24):
    feats=[]
    for ch in range(2):
        f=np.abs(np.fft.rfft(w[:,ch]))
        feats.extend(np.log1p(f[:k]).tolist())
    return np.asarray(feats,float)

phase('DATA_FREEZE')
model_raw=decode_gz_b64(MODEL_CSV_GZ_B64,EXPECTED_MODEL_SHA)
ts=decode_wave_compact_b64(RAW_WAVE_MIN_XZ_B64)
feat=pd.read_csv(StringIO(model_raw))
if len(feat)!=51 or feat.Run_ID.nunique()!=51 or ts.Run_ID.nunique()!=51: raise RuntimeError('EXPECTED_51_RUNS')
feat['Pass_T']=feat.Pass_T.astype(int); feat['Group_VW']=feat.Run_ID.str.extract(r'^(V\dW\d)')[0]
feat['Speed_kmh']=feat.Speed_level.astype(float); feat['Load_kN']=feat.Weight_level.astype(float)+1.0
all_ids={f'V{v}W{w}T{t}' for v in range(1,4) for w in range(1,4) for t in range(1,7)}
if all_ids-set(feat.Run_ID)!=EXPECTED_MISSING: raise RuntimeError('DESIGN_SLOT_MISMATCH')
for c in TARGET_COLS:
    if c not in feat or feat[c].isna().any(): raise RuntimeError(f'TARGET_INVALID:{c}')
# Build fixed previous-pass raw representations for all available runs.
wave_map={}; meta_rows=[]
for run_id,g in ts.groupby('Run_ID',sort=False):
    w,m=event_resample(g); wave_map[run_id]=w; m['Run_ID']=run_id; meta_rows.append(m)
pd.DataFrame(meta_rows).to_csv(TABLE/'previous_waveform_event_windows_v4.csv',index=False)
# Pair current run with exact consecutive previous pass within VxW.
lookup={(r.Group_VW,int(r.Pass_T)):r.Run_ID for _,r in feat.iterrows()}
rows=[]; waves=[]
for _,r in feat.iterrows():
    prev_id=lookup.get((r.Group_VW,int(r.Pass_T)-1))
    if not prev_id or prev_id not in wave_map: continue
    rr=r.to_dict(); rr['prev_Run_ID']=prev_id; rr['row_id']=len(rows); rows.append(rr); waves.append(wave_map[prev_id])
hist=pd.DataFrame(rows).reset_index(drop=True); W=np.stack(waves,axis=0)
if len(hist)!=41 or W.shape!=(41,RESAMPLE_N,2): raise RuntimeError(f'EXPECTED_41_PAIRS_GOT:{len(hist)}:{W.shape}')
# Previous peaks are from previous run feature table and are safe history predictors.
prev_targets=feat[['Run_ID']+TARGET_COLS].rename(columns={'Run_ID':'prev_Run_ID',**{c:'prev_'+c for c in TARGET_COLS}})
hist=hist.merge(prev_targets,on='prev_Run_ID',how='left',validate='many_to_one')
prev_peaks=['prev_'+c for c in TARGET_COLS]
if hist[prev_peaks].isna().any().any(): raise RuntimeError('PREVIOUS_PEAK_MISSING')
conditions=['Load_kN','Speed_kmh','Pass_T']
# Precompute deterministic representations (no labels, no cross-row fitting).
DCT=np.stack([dct_features(w,32) for w in W])
SPEC=np.stack([spectral_features(w,24) for w in W])
SCALAR=hist[conditions+prev_peaks].to_numpy(float)
X_DCT=np.column_stack([SCALAR,DCT])
X_SPEC=np.column_stack([SCALAR,SPEC])
X_RAW=np.column_stack([SCALAR,W.reshape(len(W),-1)])
y_abs=hist[TARGET_COLS].to_numpy(float); y_prev=hist[prev_peaks].to_numpy(float); y_delta=y_abs-y_prev
groups=hist.Group_VW.to_numpy()

qc={'schema':'soilbin.q1.v4','created_utc':datetime.now(timezone.utc).isoformat(),'source_model_csv_sha256':EXPECTED_MODEL_SHA,'source_raw_timeseries_sha256':EXPECTED_RAW_SHA,'waveform_used_columns_compact_blob_sha256':EXPECTED_WAVE_COMPACT_BLOB_SHA,'valid_runs':51,'history_pairs':41,'groups':9,'resample_n':RESAMPLE_N,'wave_shape':list(W.shape),'response_status':RESPONSE_STATUS,'depth_mapping_status':DEPTH_STATUS,'leakage_guard':'Only previous-pass raw LC1/LC5 waveforms, previous peaks, and current operating conditions are predictors. Current-pass waveform is never used.','external_labels_used':False,'scientific_compute_location':'Kaggle','v3_frozen_baseline_expected_joint_Group_MAE':V3_BASELINE_EXPECTED_JOINT_GROUP_MAE,'v5_policy_sha256':V5_POLICY_SHA256}
jdump(ROOT/'qc_v4.json',qc)

# Frozen V3 champion reproduction. This is the exact V3 D_delta_prev_peak feature set,
# candidate catalog, seed, and nested leave-one-VxW-group-out selection. It is included
# in V4 before raw-waveform selection so V4 can make a direct paired complexity-aware comparison.
def v3_est(name,p):
    if name=='Linear': return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',LinearRegression())])
    if name=='Ridge': return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',Ridge(alpha=p['alpha']))])
    if name=='PolynomialRidge': return Pipeline([('imp',SimpleImputer(strategy='median')),('poly',PolynomialFeatures(degree=2,include_bias=False)),('sc',StandardScaler()),('m',Ridge(alpha=p['alpha']))])
    if name=='SplineRidgeGAM': return Pipeline([('imp',SimpleImputer(strategy='median')),('spline',SplineTransformer(n_knots=p['n_knots'],degree=3,include_bias=False)),('sc',StandardScaler()),('m',Ridge(alpha=p['alpha']))])
    if name=='SVR': return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',SVR(C=p['C'],epsilon=p['epsilon'],gamma='scale'))])
    if name=='GPR':
        ker=ConstantKernel(1.0,(0.01,100.0))*Matern(length_scale=1.0,length_scale_bounds=(0.01,100.0),nu=1.5)+WhiteKernel(noise_level=1.0,noise_level_bounds=(1e-5,100.0))
        return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',GaussianProcessRegressor(kernel=ker,alpha=1e-6,normalize_y=True,random_state=SEED,n_restarts_optimizer=0))])
    if name=='RandomForest': return Pipeline([('imp',SimpleImputer(strategy='median')),('m',RandomForestRegressor(n_estimators=250,max_depth=p['max_depth'],min_samples_leaf=p['min_samples_leaf'],random_state=SEED,n_jobs=1,max_features=1.0))])
    if name=='ExtraTrees': return Pipeline([('imp',SimpleImputer(strategy='median')),('m',ExtraTreesRegressor(n_estimators=250,max_depth=p['max_depth'],min_samples_leaf=p['min_samples_leaf'],random_state=SEED,n_jobs=1,max_features=1.0))])
    if name=='HistGradientBoosting': return Pipeline([('imp',SimpleImputer(strategy='median')),('m',HistGradientBoostingRegressor(max_iter=200,learning_rate=p['learning_rate'],max_leaf_nodes=p['max_leaf_nodes'],l2_regularization=p['l2'],random_state=SEED))])
    if name=='CatBoost': return Pipeline([('imp',SimpleImputer(strategy='median')),('m',CatBoostRegressor(iterations=300,depth=p['depth'],learning_rate=p['learning_rate'],loss_function='MAE',verbose=False,random_seed=SEED,allow_writing_files=False,thread_count=1))])
    if name=='XGBoost': return Pipeline([('imp',SimpleImputer(strategy='median')),('m',XGBRegressor(n_estimators=300,max_depth=p['max_depth'],learning_rate=p['learning_rate'],subsample=.9,colsample_bytree=.9,objective='reg:squarederror',reg_lambda=1.0,random_state=SEED,n_jobs=1,tree_method='hist'))])
    if name=='LightGBM': return Pipeline([('imp',SimpleImputer(strategy='median')),('m',LGBMRegressor(n_estimators=300,num_leaves=p['num_leaves'],learning_rate=p['learning_rate'],min_child_samples=5,subsample=.9,colsample_bytree=.9,reg_lambda=1.0,random_state=SEED,n_jobs=1,verbose=-1))])
    if name=='MLP': return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',MLPRegressor(hidden_layer_sizes=p['hidden'],alpha=p['alpha'],learning_rate_init=0.01,max_iter=1200,early_stopping=False,random_state=SEED))])
    raise KeyError(name)

V3_CANDIDATES=[
 ('Linear',{}),('Ridge',{'alpha':1.0}),('Ridge',{'alpha':10.0}),
 ('PolynomialRidge',{'alpha':1.0}),('PolynomialRidge',{'alpha':10.0}),
 ('SplineRidgeGAM',{'n_knots':3,'alpha':1.0}),('SplineRidgeGAM',{'n_knots':4,'alpha':10.0}),
 ('SVR',{'C':10.0,'epsilon':0.1}),('SVR',{'C':100.0,'epsilon':0.1}),('GPR',{}),
 ('RandomForest',{'max_depth':None,'min_samples_leaf':1}),('RandomForest',{'max_depth':4,'min_samples_leaf':2}),
 ('ExtraTrees',{'max_depth':None,'min_samples_leaf':1}),('ExtraTrees',{'max_depth':4,'min_samples_leaf':2}),
 ('HistGradientBoosting',{'learning_rate':.05,'max_leaf_nodes':7,'l2':1.0}),
 ('MLP',{'hidden':(16,),'alpha':.01}),('MLP',{'hidden':(32,16),'alpha':.01}),
]
if HAVE_CAT: V3_CANDIDATES += [('CatBoost',{'depth':3,'learning_rate':.03}),('CatBoost',{'depth':4,'learning_rate':.05})]
if HAVE_XGB: V3_CANDIDATES += [('XGBoost',{'max_depth':2,'learning_rate':.03}),('XGBoost',{'max_depth':3,'learning_rate':.05})]
if HAVE_LGBM: V3_CANDIDATES += [('LightGBM',{'num_leaves':7,'learning_rate':.03}),('LightGBM',{'num_leaves':15,'learning_rate':.05})]

def v3_inner_select(tr_idx,outer_group,search_rows):
    g=groups[tr_idx]; cv=GroupKFold(n_splits=min(4,len(np.unique(g)))); best=None
    for ci,(name,p) in enumerate(V3_CANDIDATES):
        pp=np.full((len(tr_idx),2),np.nan); fail=None
        for a,b in cv.split(SCALAR[tr_idx],groups=g):
            try:
                for t in range(2):
                    m=v3_est(name,p); m.fit(SCALAR[tr_idx][a],y_delta[tr_idx][a,t]); pp[b,t]=m.predict(SCALAR[tr_idx][b])
            except Exception as exc:
                fail=f'{type(exc).__name__}:{str(exc)[:140]}'; break
        score=None
        if fail is None and np.isfinite(pp).all():
            final=pp+y_prev[tr_idx]; score=float(np.mean([group_equal_mae(y_abs[tr_idx,t],final[:,t],g) for t in range(2)]))
        search_rows.append({'outer_group':outer_group,'representation':V3_BASELINE_REP,'candidate_index':ci,'model':name,'params':json.dumps(p,sort_keys=True,default=list),'inner_joint_Group_MAE':score,'failed':fail})
        if score is not None and (best is None or score<best[0]-1e-12): best=(score,name,p)
    if best is None: raise RuntimeError(f'NO_V3_BASELINE_MODEL:{outer_group}')
    return best

def run_v3_baseline():
    pred=[]; search=[]; sels=[]
    for og in sorted(np.unique(groups)):
        te=np.flatnonzero(groups==og); tr=np.flatnonzero(groups!=og)
        score,name,p=v3_inner_select(tr,og,search)
        pdlt=np.zeros((len(te),2))
        for t in range(2):
            m=v3_est(name,p); m.fit(SCALAR[tr],y_delta[tr,t]); pdlt[:,t]=m.predict(SCALAR[te])
        final=pdlt+y_prev[te]
        sels.append({'outer_group':og,'representation':V3_BASELINE_REP,'model':name,'params':json.dumps(p,sort_keys=True,default=list),'inner_joint_Group_MAE':score})
        for j,idx in enumerate(te):
            row={'row_id':int(idx),'Group_VW':groups[idx],'representation':V3_BASELINE_REP,'model':'Selected_inner'}
            for t,nm in enumerate(TARGET_NAMES): row[f'y_{nm}']=float(y_abs[idx,t]); row[f'p_{nm}']=float(final[j,t])
            pred.append(row)
    g=pd.DataFrame(pred); yy=np.column_stack([g[f'y_{n}'] for n in TARGET_NAMES]); pp=np.column_stack([g[f'p_{n}'] for n in TARGET_NAMES]); m=metrics(yy,pp,g.Group_VW.to_numpy())
    delta=abs(float(m['joint_Group_MAE'])-V3_BASELINE_EXPECTED_JOINT_GROUP_MAE)
    jdump(ROOT/'v3_champion_reproduction_v4.json',{'representation':V3_BASELINE_REP,'expected_joint_Group_MAE':V3_BASELINE_EXPECTED_JOINT_GROUP_MAE,'observed_joint_Group_MAE':m['joint_Group_MAE'],'absolute_delta_N':delta,'tolerance_N':V3_BASELINE_REPRO_TOL_N,'reproduced':bool(delta<=V3_BASELINE_REPRO_TOL_N),'candidate_count':len(V3_CANDIDATES)})
    if delta>V3_BASELINE_REPRO_TOL_N: raise RuntimeError(f'V3_BASELINE_REPRODUCTION_DRIFT:{m["joint_Group_MAE"]}:{V3_BASELINE_EXPECTED_JOINT_GROUP_MAE}')
    return pred,search,sels

# Classical raw-representation candidate grid. All selection is nested within training VxW groups.
def est(name,p):
    if name=='Ridge': return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',Ridge(alpha=p['alpha']))])
    if name=='ExtraTrees': return Pipeline([('imp',SimpleImputer(strategy='median')),('m',ExtraTreesRegressor(n_estimators=300,max_depth=p['max_depth'],min_samples_leaf=p['leaf'],random_state=SEED,n_jobs=1,max_features=1.0))])
    if name=='HistGB': return Pipeline([('imp',SimpleImputer(strategy='median')),('m',HistGradientBoostingRegressor(max_iter=250,learning_rate=p['lr'],max_leaf_nodes=p['leaves'],l2_regularization=p['l2'],random_state=SEED))])
    if name=='CatBoost': return Pipeline([('imp',SimpleImputer(strategy='median')),('m',CatBoostRegressor(iterations=350,depth=p['depth'],learning_rate=p['lr'],loss_function='MAE',verbose=False,random_seed=SEED,allow_writing_files=False,thread_count=1))])
    if name=='XGBoost': return Pipeline([('imp',SimpleImputer(strategy='median')),('m',XGBRegressor(n_estimators=350,max_depth=p['depth'],learning_rate=p['lr'],subsample=.9,colsample_bytree=.9,objective='reg:squarederror',random_state=SEED,n_jobs=1,tree_method='hist'))])
    raise KeyError(name)
GRID=[('Ridge',{'alpha':1.0}),('Ridge',{'alpha':10.0}),('Ridge',{'alpha':100.0}),('ExtraTrees',{'max_depth':None,'leaf':1}),('ExtraTrees',{'max_depth':4,'leaf':2}),('HistGB',{'lr':.05,'leaves':7,'l2':1.0})]
if HAVE_CAT: GRID += [('CatBoost',{'depth':3,'lr':.03}),('CatBoost',{'depth':4,'lr':.05})]
if HAVE_XGB: GRID += [('XGBoost',{'depth':2,'lr':.03}),('XGBoost',{'depth':3,'lr':.05})]
REP={'DCT':X_DCT,'SPECTRAL':X_SPEC,'RAW_FLAT':X_RAW}

def inner_select(X,y,tr_idx,outer_group,rep,search_rows):
    g=groups[tr_idx]; n=min(4,len(np.unique(g))); cv=GroupKFold(n_splits=n)
    best=None
    for ci,(name,p) in enumerate(GRID):
        pp=np.full((len(tr_idx),2),np.nan)
        fail=None
        for a,b in cv.split(X[tr_idx],groups=g):
            try:
                for t in range(2):
                    m=est(name,p); m.fit(X[tr_idx][a],y[tr_idx][a,t]); pp[b,t]=m.predict(X[tr_idx][b])
            except Exception as exc:
                fail=f'{type(exc).__name__}:{str(exc)[:120]}'; break
        score=None
        if fail is None and np.isfinite(pp).all():
            final=pp+y_prev[tr_idx]
            score=float(np.mean([group_equal_mae(y_abs[tr_idx,t],final[:,t],g) for t in range(2)]))
        search_rows.append({'outer_group':outer_group,'representation':rep,'candidate_index':ci,'model':name,'params':json.dumps(p,sort_keys=True),'inner_joint_Group_MAE':score,'failed':fail})
        if score is not None and (best is None or score<best[0]-1e-12): best=(score,name,p)
    if best is None: raise RuntimeError(f'NO_CLASSICAL_MODEL:{rep}:{outer_group}')
    return best

def run_classical(rep):
    X=REP[rep]; pred=[]; search=[]; sels=[]
    for og in sorted(np.unique(groups)):
        te=np.flatnonzero(groups==og); tr=np.flatnonzero(groups!=og)
        score,name,p=inner_select(X,y_delta,tr,og,rep,search)
        pdlt=np.zeros((len(te),2))
        for t in range(2):
            m=est(name,p); m.fit(X[tr],y_delta[tr,t]); pdlt[:,t]=m.predict(X[te])
        final=pdlt+y_prev[te]
        sels.append({'outer_group':og,'representation':rep,'model':name,'params':json.dumps(p,sort_keys=True),'inner_joint_Group_MAE':score})
        for j,idx in enumerate(te):
            row={'row_id':int(idx),'Group_VW':groups[idx],'representation':rep,'model':'Selected_inner'}
            for t,nm in enumerate(TARGET_NAMES): row[f'y_{nm}']=float(y_abs[idx,t]); row[f'p_{nm}']=float(final[j,t])
            pred.append(row)
    return pred,search,sels

phase('V3_FROZEN_BASELINE_REPRODUCTION')
pred_rows,search_rows,sel_rows=run_v3_baseline()
phase('CLASSICAL_RAW_REPRESENTATIONS')
for rep in ['DCT','SPECTRAL','RAW_FLAT']:
    phase('REP_'+rep)
    p,s,z=run_classical(rep); pred_rows+=p; search_rows+=s; sel_rows+=z
    jdump(STATE/'progress_v4.json',{'completed_representations':[r for r in ['DCT','SPECTRAL','RAW_FLAT'] if any(x['representation']==r for x in pred_rows)],'pred_rows':len(pred_rows),'search_rows':len(search_rows)})

# Compact CNN: fixed architectures; inner group CV selects architecture and training epoch budget only from internal groups.
cnn_rows=[]; cnn_search=[]; cnn_sel=[]
if HAVE_TORCH:
    torch.set_num_threads(1)
    class Net(nn.Module):
        def __init__(self,width=8,drop=.25):
            super().__init__()
            self.conv=nn.Sequential(nn.Conv1d(2,width,7,padding=3),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(width,width*2,5,padding=2),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(width*2,width*2,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(8))
            self.head=nn.Sequential(nn.Linear(width*2*8+5,32),nn.ReLU(),nn.Dropout(drop),nn.Linear(32,2))
        def forward(self,w,s): return self.head(torch.cat([self.conv(w).flatten(1),s],dim=1))
    CNN_CFG=[{'width':6,'drop':.25,'wd':1e-3,'epochs':180},{'width':10,'drop':.35,'wd':3e-3,'epochs':240}]
    def fit_cnn(train_idx,cfg,seed,epochs=None):
        torch.manual_seed(seed); np.random.seed(seed)
        tr=np.asarray(train_idx,int)
        # Scale each channel and scalar predictor using training rows only.
        wm=W[tr].astype(np.float32); sm=SCALAR[tr].astype(np.float32); yy=y_delta[tr].astype(np.float32)
        wmean=wm.mean(axis=(0,1),keepdims=True); wstd=wm.std(axis=(0,1),keepdims=True)+1e-6
        smean=sm.mean(axis=0,keepdims=True); sstd=sm.std(axis=0,keepdims=True)+1e-6
        ymean=yy.mean(axis=0,keepdims=True); ystd=yy.std(axis=0,keepdims=True)+1e-6
        wt=torch.tensor(((wm-wmean)/wstd).transpose(0,2,1)); st=torch.tensor((sm-smean)/sstd); yt=torch.tensor((yy-ymean)/ystd)
        model=Net(cfg['width'],cfg['drop']); opt=torch.optim.AdamW(model.parameters(),lr=0.005,weight_decay=cfg['wd']); lossfn=nn.L1Loss()
        model.train()
        n_ep=int(epochs or cfg['epochs'])
        for ep in range(n_ep):
            opt.zero_grad(); loss=lossfn(model(wt,st),yt); loss.backward(); opt.step()
        return model,(wmean,wstd,smean,sstd,ymean,ystd)
    def predict_cnn(model,norm,idx):
        wmean,wstd,smean,sstd,ymean,ystd=norm; ii=np.asarray(idx,int)
        wt=torch.tensor(((W[ii].astype(np.float32)-wmean)/wstd).transpose(0,2,1)); st=torch.tensor((SCALAR[ii].astype(np.float32)-smean)/sstd)
        model.eval()
        with torch.no_grad(): z=model(wt,st).cpu().numpy()
        return z*ystd+ ymean
    phase('CNN_NESTED_GROUPED')
    for og in sorted(np.unique(groups)):
        te=np.flatnonzero(groups==og); tr=np.flatnonzero(groups!=og); gtr=groups[tr]
        cv=GroupKFold(n_splits=min(4,len(np.unique(gtr))))
        best=None
        for ci,cfg in enumerate(CNN_CFG):
            pp=np.full((len(tr),2),np.nan); fail=None
            for fold,(a,b) in enumerate(cv.split(tr,groups=gtr)):
                try:
                    m,norm=fit_cnn(tr[a],cfg,SEED+ci*100+fold)
                    pp[b]=predict_cnn(m,norm,tr[b])
                except Exception as exc:
                    fail=f'{type(exc).__name__}:{str(exc)[:140]}'; break
            score=None
            if fail is None and np.isfinite(pp).all():
                final=pp+y_prev[tr]; score=float(np.mean([group_equal_mae(y_abs[tr,t],final[:,t],gtr) for t in range(2)]))
            cnn_search.append({'outer_group':og,'candidate_index':ci,'config':json.dumps(cfg,sort_keys=True),'inner_joint_Group_MAE':score,'failed':fail})
            if score is not None and (best is None or score<best[0]-1e-12): best=(score,ci,cfg)
        if best is None: raise RuntimeError(f'NO_CNN_MODEL:{og}')
        score,ci,cfg=best
        # Ensemble three seeds, trained only on outer-training groups.
        ens=[]
        for k in range(3):
            m,norm=fit_cnn(tr,cfg,SEED+1000+ci*100+k); ens.append(predict_cnn(m,norm,te))
        pdlt=np.mean(np.stack(ens),axis=0); final=pdlt+y_prev[te]
        cnn_sel.append({'outer_group':og,'representation':'COMPACT_CNN','config':json.dumps(cfg,sort_keys=True),'inner_joint_Group_MAE':score,'ensemble_seeds':3})
        for j,idx in enumerate(te):
            row={'row_id':int(idx),'Group_VW':groups[idx],'representation':'COMPACT_CNN','model':'Selected_inner'}
            for t,nm in enumerate(TARGET_NAMES): row[f'y_{nm}']=float(y_abs[idx,t]); row[f'p_{nm}']=float(final[j,t])
            cnn_rows.append(row)
else:
    jdump(ROOT/'cnn_unavailable_v4.json',{'torch_available':False})

pred_rows+=cnn_rows; search_rows+=cnn_search; sel_rows+=cnn_sel
pred=pd.DataFrame(pred_rows); search=pd.DataFrame(search_rows); sels=pd.DataFrame(sel_rows)
pred.to_csv(TABLE/'oof_predictions_v4.csv',index=False); search.to_csv(TABLE/'hyperparameter_search_v4.csv',index=False); sels.to_csv(TABLE/'selection_v4.csv',index=False)
# Summaries per representation, including the frozen V3 champion reproduced in the same run.
summ=[]
for rep,g in pred.groupby('representation'):
    yy=np.column_stack([g[f'y_{n}'] for n in TARGET_NAMES]); pp=np.column_stack([g[f'p_{n}'] for n in TARGET_NAMES]); m=metrics(yy,pp,g.Group_VW.to_numpy()); m.update({'representation':rep,'n':int(len(g)),'complete_oof':bool(len(g)==41)}); summ.append(m)
summary=pd.DataFrame(summ).sort_values('joint_Group_MAE'); summary.to_csv(TABLE/'summary_v4.csv',index=False)
overall_champ=summary.iloc[0].to_dict()
baseline=summary[summary.representation==V3_BASELINE_REP].iloc[0].to_dict()
raw_summary=summary[summary.representation!=V3_BASELINE_REP].sort_values('joint_Group_MAE')
if len(raw_summary)==0: raise RuntimeError('NO_RAW_WAVEFORM_REPRESENTATION')
raw_champ=raw_summary.iloc[0].to_dict(); raw_rep=raw_champ['representation']

# Paired group-level comparisons against every representation and a dedicated frozen-V3-vs-best-raw test.
pairs=[]
for rep in sorted(pred.representation.unique()):
    if rep==overall_champ['representation']: continue
    cg=pred[pred.representation==overall_champ['representation']]; rg=pred[pred.representation==rep]; diffs=[]
    for gv in sorted(np.unique(groups)):
        a=cg[cg.Group_VW==gv]; b=rg[rg.Group_VW==gv]
        ca=np.mean([mean_absolute_error(a[f'y_{n}'],a[f'p_{n}']) for n in TARGET_NAMES]); rb=np.mean([mean_absolute_error(b[f'y_{n}'],b[f'p_{n}']) for n in TARGET_NAMES]); diffs.append(rb-ca)
    diffs=np.asarray(diffs,float); obs=float(diffs.mean()); null=np.array([np.mean(diffs*np.asarray(z)) for z in itertools.product([-1,1],repeat=len(diffs))]); p=float(np.mean(null>=obs-1e-12)); pairs.append({'champion':overall_champ['representation'],'comparator':rep,'mean_group_improvement_N':obs,'exact_signflip_one_sided_p':p,'n_groups':len(diffs)})

bdf=pred[pred.representation==V3_BASELINE_REP]; rdf=pred[pred.representation==raw_rep]
raw_improvements=[]
for gv in sorted(np.unique(groups)):
    b=bdf[bdf.Group_VW==gv]; r=rdf[rdf.Group_VW==gv]
    be=np.mean([mean_absolute_error(b[f'y_{n}'],b[f'p_{n}']) for n in TARGET_NAMES]); re=np.mean([mean_absolute_error(r[f'y_{n}'],r[f'p_{n}']) for n in TARGET_NAMES]); raw_improvements.append(be-re)
raw_improvements=np.asarray(raw_improvements,float); raw_obs=float(raw_improvements.mean()); raw_null=np.array([np.mean(raw_improvements*np.asarray(z)) for z in itertools.product([-1,1],repeat=len(raw_improvements))]); raw_p=float(np.mean(raw_null>=raw_obs-1e-12))
relative_raw_improvement_pct=float((baseline['joint_Group_MAE']-raw_champ['joint_Group_MAE'])/baseline['joint_Group_MAE']*100.0)
target_guard=bool(raw_champ['LC1_proxy_N_Group_MAE']<=1.05*baseline['LC1_proxy_N_Group_MAE'] and raw_champ['LC5_proxy_N_Group_MAE']<=1.05*baseline['LC5_proxy_N_Group_MAE'])
material_pair=bool(raw_obs>0 and raw_p<=0.05)
if relative_raw_improvement_pct < 2.0:
    v5_decision='RETAIN_V3_BASELINE_COMPLEXITY_TIE_RULE'
elif target_guard and (relative_raw_improvement_pct>=5.0 or material_pair):
    v5_decision='ADMIT_RAW_WAVEFORM_CHAMPION_TO_V5_FINAL_GATE'
else:
    v5_decision='RETAIN_V3_BASELINE_V4_RAW_GAIN_INSUFFICIENT'
if raw_rep=='COMPACT_CNN' and v5_decision.startswith('ADMIT_'):
    v5_decision='ADMIT_COMPACT_CNN_TO_V5_PENDING_STABILITY_GATE'
comparison={'baseline_representation':V3_BASELINE_REP,'baseline_joint_Group_MAE':float(baseline['joint_Group_MAE']),'raw_waveform_champion':raw_rep,'raw_joint_Group_MAE':float(raw_champ['joint_Group_MAE']),'relative_raw_improvement_pct':relative_raw_improvement_pct,'mean_paired_group_raw_improvement_N':raw_obs,'exact_signflip_one_sided_p':raw_p,'n_groups':9,'LC1_baseline_Group_MAE':float(baseline['LC1_proxy_N_Group_MAE']),'LC1_raw_Group_MAE':float(raw_champ['LC1_proxy_N_Group_MAE']),'LC5_baseline_Group_MAE':float(baseline['LC5_proxy_N_Group_MAE']),'LC5_raw_Group_MAE':float(raw_champ['LC5_proxy_N_Group_MAE']),'target_guard_no_worse_than_5pct':target_guard,'v5_policy_sha256':V5_POLICY_SHA256,'v5_admission_decision':v5_decision,'policy_note':'Frozen before V3/V4 results: <2% chooses simpler; raw/deep replacement requires >=5% gain OR materially stronger exact paired evidence, plus <=5% worsening per target. Material paired evidence operationalized here as positive mean paired improvement and exact one-sided sign-flip p<=0.05.'}
pd.DataFrame(pairs).to_csv(TABLE/'paired_representation_tests_v4.csv',index=False)
jdump(ROOT/'paired_representation_tests_v4.json',{'rows':pairs,'v3_vs_best_raw':comparison,'guard':'Nine VxW groups; inferential precision is limited.'})
jdump(ROOT/'v3_vs_raw_waveform_v4.json',comparison)

# Figures
plt.figure(figsize=(9,4)); plt.bar(summary.representation,summary.joint_Group_MAE); plt.ylabel('Joint group-equal MAE (N)'); plt.xticks(rotation=25,ha='right'); plt.tight_layout(); plt.savefig(FIG/'01_raw_representation_ranking_v4.png',dpi=180); plt.close()
for ti,nm in enumerate(TARGET_NAMES):
    g=pred[pred.representation==overall_champ['representation']]
    plt.figure(figsize=(5,5)); plt.scatter(g[f'y_{nm}'],g[f'p_{nm}']); lo=min(g[f'y_{nm}'].min(),g[f'p_{nm}'].min()); hi=max(g[f'y_{nm}'].max(),g[f'p_{nm}'].max()); plt.plot([lo,hi],[lo,hi],'--'); plt.xlabel('Observed N'); plt.ylabel('Predicted N'); plt.title(f"{overall_champ['representation']} — {nm}"); plt.tight_layout(); plt.savefig(FIG/f'0{ti+2}_{nm}_observed_predicted_v4.png',dpi=180); plt.close()

result={'schema':'soilbin.q1.v4','response_status':RESPONSE_STATUS,'depth_mapping_status':DEPTH_STATUS,'n_runs':51,'n_history_pairs':41,'n_groups':9,'raw_timeseries_rows':int(len(ts)),'resample_n':RESAMPLE_N,'representations':sorted(pred.representation.unique().tolist()),'torch_available':HAVE_TORCH,'v3_frozen_baseline':baseline,'raw_waveform_champion':raw_champ,'overall_internal_champion':overall_champ,'v3_vs_raw_waveform':comparison,'paired_tests':pairs,'external_validation_reused_for_selection':False,'leakage_guard':qc['leakage_guard']}
jdump(ROOT/'summary_v4.json',result)
pub={'checklist':{'raw_previous_pass_waveform':True,'current_pass_waveform_forbidden':True,'event_resampling_deterministic':True,'nested_grouped_cv':True,'random_split_used':False,'v3_champion_reproduced_same_run':True,'v3_champion_expected_joint_Group_MAE':V3_BASELINE_EXPECTED_JOINT_GROUP_MAE,'dct_representation':True,'spectral_representation':True,'raw_flat_ridge_tree_benchmark':True,'compact_cnn_if_available':HAVE_TORCH,'paired_group_tests':True,'pre_registered_v5_complexity_policy_applied':True,'external_labels_used':False},'limitations':['Only 41 history pairs across nine VxW groups; deep learning is exploratory and must beat the frozen V3 baseline under grouped OOF evaluation.','Event-centered resampling is derived only from previous-pass traces.','Response remains calibrated vertical force proxy N; depth mapping remains provisional.','V4 does not re-tune against any external dataset.','The V3 baseline is reproduced in-run solely to permit paired internal comparison; V3 external labels remain absent.'],'figure_count':3}
jdump(ROOT/'publication_manifest_v4.json',pub)
manifest={'schema':'soilbin.q1.v4','finished_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,'python':sys.version.split()[0],'cpu_count':os.cpu_count(),'source_fingerprints':{'model_csv':EXPECTED_MODEL_SHA,'raw_timeseries_source':EXPECTED_RAW_SHA,'waveform_used_columns_compact_blob':EXPECTED_WAVE_COMPACT_BLOB_SHA,'waveform_used_columns_compact_xz':EXPECTED_WAVE_COMPACT_XZ_SHA},'models_available':{'CatBoost':HAVE_CAT,'XGBoost':HAVE_XGB,'LightGBM':HAVE_LGBM,'PyTorch':HAVE_TORCH},'v3_baseline_expected_joint_Group_MAE':V3_BASELINE_EXPECTED_JOINT_GROUP_MAE,'v5_policy_sha256':V5_POLICY_SHA256,'response_status':RESPONSE_STATUS,'depth_mapping_status':DEPTH_STATUS,'external_labels_used':False,'evidence_files':sorted([p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()])}
jdump(ROOT/'run_manifest_v4.json',manifest)
(ROOT/'FINAL_REPORT_V4.md').write_text('# SoilBin Q1/Q2 V4 — Previous-pass Raw Waveform Representation Learning\n\n'+f"Frozen V3 baseline: **{baseline['joint_Group_MAE']:.3f} N** joint group-equal MAE.\n\nBest raw-waveform representation: **{raw_rep}**, joint group-equal MAE = **{raw_champ['joint_Group_MAE']:.3f} N** ({relative_raw_improvement_pct:+.2f}% improvement vs V3 baseline).\n\nV5 admission decision: **{v5_decision}**.\n\nCurrent-pass waveform is never used; only previous-pass LC1/LC5 raw traces, previous peaks, and current operating conditions are predictors. Deep models are exploratory because only 41 history pairs / 9 groups are available.\n",encoding='utf-8')
phase('COMPLETE')
print('SOILBIN_V4_COMPLETE',json.dumps({'overall_champion':overall_champ['representation'],'overall_joint_Group_MAE':overall_champ['joint_Group_MAE'],'raw_champion':raw_rep,'raw_joint_Group_MAE':raw_champ['joint_Group_MAE'],'relative_raw_improvement_pct':relative_raw_improvement_pct,'v5_decision':v5_decision,'torch_available':HAVE_TORCH},sort_keys=True),flush=True)
