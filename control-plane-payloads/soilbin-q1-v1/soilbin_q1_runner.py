# SoilBin Q1/Q2 evidence benchmark v1.0
from __future__ import annotations
import os, sys, json, math, hashlib, random, warnings, zipfile, subprocess, importlib.util
from pathlib import Path
from io import StringIO
from collections import defaultdict, Counter
from datetime import datetime, timezone

SEED = 20260914
random.seed(SEED)
os.environ.setdefault('PYTHONHASHSEED', str(SEED))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
np.random.seed(SEED)
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

try:
    from catboost import CatBoostRegressor
except Exception:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'catboost==1.2.8'])
    from catboost import CatBoostRegressor

try:
    import statsmodels.formula.api as smf
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False

ROOT = Path('/kaggle/working/SOILBIN_EVIDENCE') if Path('/kaggle/working').exists() else Path('SOILBIN_EVIDENCE')
ROOT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
TABLE = ROOT / 'tables'; TABLE.mkdir(exist_ok=True)
STATE = ROOT / 'state'; STATE.mkdir(exist_ok=True)

MODELS = ['Linear','PolynomialRidge','GAM','SVR','GPR','RandomForest','ExtraTrees','CatBoost','MLP']
BASELINES = ['Median','Persistence']
DEPTH_MAP = {'LC5_peak_magnitude_N': 5, 'LC1_peak_magnitude_N': 15}
DEPTH_MAP_STATUS = 'PROVISIONAL_INFERRED_FROM_RESPONSE_MAGNITUDE_PENDING_EXPERIMENTAL_LAYOUT_CONFIRMATION'
RESPONSE_STATUS = 'CALIBRATED_VERTICAL_FORCE_PROXY_N_PENDING_SENSOR_AREA_OR_STRESS_CALIBRATION'
EXPECTED_MISSING = {'V1W1T1','V2W3T2','V3W2T1'}
MAX_CANDIDATES = 20
INNER_FOLDS = 4
N_JOBS = max(1, min(4, (os.cpu_count() or 2)))


def phase(name):
    print(f'CGP_PHASE:{name}', flush=True)


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def jsonable(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (pd.Timestamp, datetime)): return str(v)
    return v


def df_records(df):
    return [{str(k): jsonable(v) for k,v in r.items()} for r in df.to_dict('records')]


def group_mae(y, p, groups):
    return float(np.mean([np.mean(np.abs(y[groups == g] - p[groups == g])) for g in np.unique(groups)]))


def metrics(y, p, groups):
    y=np.asarray(y,float); p=np.asarray(p,float); groups=np.asarray(groups)
    e=p-y; ss=float(np.sum((y-y.mean())**2))
    return {'n':int(len(y)),'n_groups':int(len(np.unique(groups))),
            'Group_MAE':group_mae(y,p,groups),'MAE':float(np.mean(np.abs(e))),
            'RMSE':float(np.sqrt(np.mean(e*e))),'Bias':float(np.mean(e)),
            'R2_pooled':None if ss <= 1e-15 else float(1-np.sum(e*e)/ss),
            'negative_prediction_count':int(np.sum(p<0))}


def build_data():
    c=pd.read_csv(StringIO(CONDITIONS_CSV.strip()), encoding='utf-8-sig')
    y=pd.read_csv(StringIO(TARGETS_CSV.strip()), encoding='utf-8-sig')
    assert len(c)==51 and len(y)==51 and c.Run_ID.nunique()==51
    all_ids={f'V{v}W{w}T{t}' for v in range(1,4) for w in range(1,4) for t in range(1,7)}
    assert all_ids-set(c.Run_ID)==EXPECTED_MISSING
    c=c.reset_index(drop=True); y=y.reset_index(drop=True)
    c['Speed_kmh']=c['Speed_level'].astype(float)
    c['Load_kN']=c['Weight_level'].astype(float)+1.0
    rows=[]
    for i,r in c.iterrows():
        for sensor,col in [('LC5','LC5_peak_magnitude_N'),('LC1','LC1_peak_magnitude_N')]:
            rows.append({'Run_ID':r.Run_ID,'Group_VW':r.Group_VW,'Speed_kmh':float(r.Speed_kmh),
                         'Load_kN':float(r.Load_kN),'Pass_T':int(r.Pass_T),'Sensor':sensor,
                         'Depth_cm':int(DEPTH_MAP[col]),'Response_N':float(y.loc[i,col])})
    long=pd.DataFrame(rows).sort_values(['Group_VW','Pass_T','Depth_cm']).reset_index(drop=True)
    assert len(long)==102 and long[['Run_ID','Depth_cm']].drop_duplicates().shape[0]==102
    assert set(long.Depth_cm)=={5,15} and long.Group_VW.nunique()==9
    lookup={(r.Group_VW,int(r.Pass_T),int(r.Depth_cm)):float(r.Response_N) for r in long.itertuples()}
    prev=[]
    for r in long.itertuples(): prev.append(lookup.get((r.Group_VW,int(r.Pass_T)-1,int(r.Depth_cm)), np.nan))
    long['Previous_Response_N']=prev
    paired=long[long.Previous_Response_N.notna()].copy().reset_index(drop=True)
    assert len(paired)==82
    assert not ((paired.Group_VW=='V2W3') & (paired.Pass_T==3)).any()
    long.to_csv(TABLE/'soilstress_long_proxy_N.csv',index=False)
    paired.to_csv(TABLE/'history_pairs_proxy_N.csv',index=False)
    return c,y,long,paired

phase('BOOTSTRAP')
conditions,targets,long_df,paired_df=build_data()

qc={
 'schema':'soilbin.q1.v1','created_utc':datetime.now(timezone.utc).isoformat(),
 'conditions_sha256':sha_text(CONDITIONS_CSV.strip()),'targets_sha256':sha_text(TARGETS_CSV.strip()),
 'valid_runs':51,'long_rows':102,'consecutive_history_rows':82,'validation_groups':9,
 'missing_design_slots':sorted(EXPECTED_MISSING),'synthetic_rows':0,'automatic_outlier_removals':0,
 'speed_kmh_levels':[1,2,3],'vertical_load_kN_levels':[2,3,4],'pass_levels':[1,2,3,4,5,6],
 'depth_cm_levels':[5,15],'depth_mapping':{'LC5':5,'LC1':15},'depth_mapping_status':DEPTH_MAP_STATUS,
 'response_column':'Response_N','response_status':RESPONSE_STATUS,
 'independent_external_validation':False,
 'independence_warning':'102 depth observations arise from 51 runs; depths within a run and sequential passes are not independent replicates.',
 'stress_claim_block':'Do not label Response_N as kPa stress until sensor effective area or direct stress calibration is confirmed.'
}
jdump(ROOT/'qc.json',qc)
phase('DATA_QC_COMPLETE')

# Descriptive evidence
desc=[]
for by in [['Depth_cm'],['Load_kN','Depth_cm'],['Speed_kmh','Depth_cm'],['Pass_T','Depth_cm'],['Group_VW','Depth_cm']]:
    z=long_df.groupby(by).Response_N.agg(['count','mean','std','median','min','max']).reset_index()
    z['grouping']='+'.join(by); desc.append(z)
descriptive=pd.concat(desc,ignore_index=True,sort=False)
descriptive.to_csv(TABLE/'descriptive_statistics.csv',index=False)

stats_rows=[]; stats_note='statsmodels unavailable'
if STATSMODELS_OK:
    formula='Response_N ~ Load_kN + Speed_kmh + Pass_T + C(Depth_cm) + Load_kN:C(Depth_cm) + Speed_kmh:C(Depth_cm) + Pass_T:C(Depth_cm) + Load_kN:Pass_T + Speed_kmh:Pass_T + Load_kN:Speed_kmh'
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        fit=smf.ols(formula,long_df).fit(cov_type='cluster',cov_kwds={'groups':long_df['Group_VW']})
    ci=fit.conf_int()
    for term in fit.params.index:
        stats_rows.append({'term':term,'coef':float(fit.params[term]),'SE_cluster':float(fit.bse[term]),
                           'CI95_low':float(ci.loc[term,0]),'CI95_high':float(ci.loc[term,1]),
                           'p_cluster_exploratory':float(fit.pvalues[term])})
    stats_note='Cluster-robust OLS by 9 VxW groups; p-values are exploratory because only nine clusters exist.'
    pd.DataFrame(stats_rows).to_csv(TABLE/'statistical_effects.csv',index=False)
jdump(ROOT/'statistical_effects.json',{'note':stats_note,'rows':stats_rows})
phase('STATISTICS_COMPLETE')


def candidate_grid(name):
    grids={
      'Linear':{},
      'PolynomialRidge':{'alpha':[0.1,1.0,10.0,100.0]},
      'GAM':{'alpha':[0.1,1.0,10.0],'n_knots':[3]},
      'SVR':{'C':[1.0,10.0,100.0],'gamma':[0.1,0.5],'epsilon':[0.05,0.1]},
      'GPR':{'nu':[1.5,2.5],'noise_level':[0.01,0.1]},
      'RandomForest':{'max_depth':[2,3,4],'min_samples_leaf':[2,4]},
      'ExtraTrees':{'max_depth':[2,3,4],'min_samples_leaf':[2,4]},
      'CatBoost':{'depth':[2,3],'iterations':[150,300],'learning_rate':[0.03,0.08]},
      'MLP':{'hidden_layer_sizes':[(4,),(8,)],'alpha':[0.1,1.0,10.0]},
    }
    vals=list(ParameterGrid(grids[name]))
    return vals[:MAX_CANDIDATES]


def make_est(name,p,nf):
    if name=='Linear': return StandardScaler(), LinearRegression()
    if name=='PolynomialRidge':
        return Pipeline([('s1',StandardScaler()),('poly',PolynomialFeatures(2,include_bias=False)),('s2',StandardScaler())]), Ridge(alpha=p['alpha'])
    if name=='GAM':
        ct=ColumnTransformer([('linear','passthrough',[i for i in range(nf) if i!=2]),
                              ('pass_spline',SplineTransformer(n_knots=p['n_knots'],degree=2,include_bias=False,extrapolation='linear'),[2])])
        return Pipeline([('add',ct),('scale',StandardScaler())]), Ridge(alpha=p['alpha'])
    if name=='SVR': return StandardScaler(), SVR(kernel='rbf',**p)
    if name=='GPR':
        ker=ConstantKernel(1.0,(1e-3,1e3))*Matern(length_scale=np.ones(nf),length_scale_bounds=(1e-2,100),nu=p['nu'])+WhiteKernel(noise_level=p['noise_level'],noise_level_bounds=(1e-5,1.0))
        return StandardScaler(), GaussianProcessRegressor(kernel=ker,alpha=1e-8,normalize_y=False,n_restarts_optimizer=0,random_state=SEED)
    if name=='RandomForest': return None, RandomForestRegressor(n_estimators=300,random_state=SEED,n_jobs=1,max_features=1.0,**p)
    if name=='ExtraTrees': return None, ExtraTreesRegressor(n_estimators=300,random_state=SEED,n_jobs=1,max_features=1.0,**p)
    if name=='CatBoost': return None, CatBoostRegressor(loss_function='RMSE',random_seed=SEED,thread_count=1,verbose=False,allow_writing_files=False,use_best_model=False,l2_leaf_reg=3,**p)
    if name=='MLP': return StandardScaler(), MLPRegressor(solver='lbfgs',activation='tanh',random_state=SEED,max_iter=3000,early_stopping=False,**p)
    raise KeyError(name)


def fit_bundle(name,p,X,y):
    ym=float(np.mean(y)); ys=float(np.std(y)); ys=ys if ys>1e-12 else 1.0
    pre,est=make_est(name,p,X.shape[1]); Xt=pre.fit_transform(X) if pre is not None else X
    with threadpool_limits(1): est.fit(Xt,(y-ym)/ys)
    return (pre,est,ym,ys,name)


def pred_bundle(bundle,X,need_sd=False):
    pre,est,ym,ys,name=bundle; Xt=pre.transform(X) if pre is not None else X
    with threadpool_limits(1):
        if name=='GPR' and need_sd:
            z,sd=est.predict(Xt,return_std=True); return np.asarray(z)*ys+ym,np.asarray(sd)*ys
        z=est.predict(Xt)
    return np.asarray(z)*ys+ym,None


def task_frame(name):
    if name=='A_all': d=long_df.copy(); feats=['Load_kN','Speed_kmh','Pass_T','Depth_cm']; mode='direct'
    elif name=='A_paired': d=paired_df.copy(); feats=['Load_kN','Speed_kmh','Pass_T','Depth_cm']; mode='direct'
    elif name=='B_history': d=paired_df.copy(); feats=['Load_kN','Speed_kmh','Pass_T','Depth_cm','Previous_Response_N']; mode='direct'
    elif name=='C_delta': d=paired_df.copy(); feats=['Load_kN','Speed_kmh','Pass_T','Depth_cm','Previous_Response_N']; mode='delta'
    elif name=='A_depth5': d=long_df[long_df.Depth_cm==5].copy(); feats=['Load_kN','Speed_kmh','Pass_T']; mode='direct'
    elif name=='A_depth15': d=long_df[long_df.Depth_cm==15].copy(); feats=['Load_kN','Speed_kmh','Pass_T']; mode='direct'
    elif name=='B_depth5': d=paired_df[paired_df.Depth_cm==5].copy(); feats=['Load_kN','Speed_kmh','Pass_T','Previous_Response_N']; mode='direct'
    elif name=='B_depth15': d=paired_df[paired_df.Depth_cm==15].copy(); feats=['Load_kN','Speed_kmh','Pass_T','Previous_Response_N']; mode='direct'
    else: raise KeyError(name)
    d=d.reset_index(drop=True); return d,feats,mode


def inner_splits(d,train_idx):
    groups=d.loc[train_idx,'Group_VW'].to_numpy(); n=min(INNER_FOLDS,len(np.unique(groups)))
    if n<2: raise RuntimeError('insufficient groups for inner CV')
    sp=GroupKFold(n_splits=n); X=np.zeros((len(train_idx),1))
    return [(np.asarray(train_idx)[a],np.asarray(train_idx)[b]) for a,b in sp.split(X,groups=groups)]


def tune(d,feats,mode,train_idx,model):
    best=None; bestscore=math.inf; ledger=[]
    for ci,p in enumerate(candidate_grid(model)):
        vals=[]; failed=None
        for a,b in inner_splits(d,train_idx):
            Xtr=d.loc[a,feats].to_numpy(float); Xv=d.loc[b,feats].to_numpy(float)
            ytr=d.loc[a,'Response_N'].to_numpy(float); yv=d.loc[b,'Response_N'].to_numpy(float)
            target=ytr-d.loc[a,'Previous_Response_N'].to_numpy(float) if mode=='delta' else ytr
            try:
                fb=fit_bundle(model,p,Xtr,target); raw,_=pred_bundle(fb,Xv)
                pv=raw+d.loc[b,'Previous_Response_N'].to_numpy(float) if mode=='delta' else raw
                vals.append((yv,pv,d.loc[b,'Group_VW'].to_numpy()))
            except Exception as e:
                failed=f'{type(e).__name__}:{e}'; break
        score=None
        if failed is None:
            yy=np.concatenate([x[0] for x in vals]); pp=np.concatenate([x[1] for x in vals]); gg=np.concatenate([x[2] for x in vals])
            score=group_mae(yy,pp,gg)
            if score<bestscore: bestscore=score; best=p.copy()
        ledger.append({'candidate':ci,'params':p,'inner_Group_MAE':score,'error':failed})
    return best,bestscore,ledger


def baseline_predict(d,mode,train,test,kind):
    if kind=='Persistence': return d.loc[test,'Previous_Response_N'].to_numpy(float)
    train_target=d.loc[train,'Response_N'].to_numpy(float)
    if mode=='delta':
        delta=train_target-d.loc[train,'Previous_Response_N'].to_numpy(float)
        return d.loc[test,'Previous_Response_N'].to_numpy(float)+float(np.median(delta))
    return np.full(len(test),float(np.median(train_target)))


def baseline_inner(d,mode,train,kind):
    out=[]
    for a,b in inner_splits(d,train):
        p=baseline_predict(d,mode,a,b,kind)
        out.append((d.loc[b,'Response_N'].to_numpy(float),p,d.loc[b,'Group_VW'].to_numpy()))
    yy=np.concatenate([x[0] for x in out]); pp=np.concatenate([x[1] for x in out]); gg=np.concatenate([x[2] for x in out])
    return group_mae(yy,pp,gg)


def one_outer(task,scheme,label):
    d,feats,mode=task_frame(task)
    splitvals=d.Group_VW.astype(str) if scheme=='VW' else (d.Speed_kmh.astype(str) if scheme=='speed' else d.Load_kN.astype(str))
    test=np.flatnonzero(splitvals.to_numpy()==str(label)); train=np.flatnonzero(splitvals.to_numpy()!=str(label))
    if not len(test) or not len(train): raise RuntimeError('empty outer split')
    assert not (set(d.loc[train,'Group_VW']) & set(d.loc[test,'Group_VW']))
    foldrows=[]; preds=[]; searchrows=[]; selection={}; trained={}
    for model in MODELS:
        try:
            bp,ins,ledger=tune(d,feats,mode,train,model)
            for q in ledger: searchrows.append({'task':task,'scheme':scheme,'outer':str(label),'model':model,**q})
            if bp is None: raise RuntimeError('no valid candidate')
            Xtr=d.loc[train,feats].to_numpy(float); Xt=d.loc[test,feats].to_numpy(float)
            ytr=d.loc[train,'Response_N'].to_numpy(float); target=ytr-d.loc[train,'Previous_Response_N'].to_numpy(float) if mode=='delta' else ytr
            fb=fit_bundle(model,bp,Xtr,target); raw,sd=pred_bundle(fb,Xt,need_sd=(model=='GPR'))
            pred=raw+d.loc[test,'Previous_Response_N'].to_numpy(float) if mode=='delta' else raw
            ytrue=d.loc[test,'Response_N'].to_numpy(float); grp=d.loc[test,'Group_VW'].to_numpy()
            selection[model]=ins; trained[model]=(pred,sd)
            foldrows.append({'task':task,'scheme':scheme,'outer':str(label),'model':model,'status':'OK','inner_Group_MAE':ins,'test_Group_MAE':group_mae(ytrue,pred,grp),'params':json.dumps(bp,default=jsonable)})
            for k,ix in enumerate(test):
                row={'task':task,'scheme':scheme,'outer':str(label),'model':model,'selected_family':'','Run_ID':d.loc[ix,'Run_ID'],'Group_VW':d.loc[ix,'Group_VW'],'Speed_kmh':float(d.loc[ix,'Speed_kmh']),'Load_kN':float(d.loc[ix,'Load_kN']),'Pass_T':int(d.loc[ix,'Pass_T']),'Depth_cm':int(d.loc[ix,'Depth_cm']),'y_true':float(ytrue[k]),'y_pred':float(pred[k]),'std':None}
                if sd is not None: row['std']=float(sd[k])
                preds.append(row)
        except Exception as e:
            foldrows.append({'task':task,'scheme':scheme,'outer':str(label),'model':model,'status':'FAILED','inner_Group_MAE':None,'test_Group_MAE':None,'params':'','error':f'{type(e).__name__}:{e}'})
    bnames=['Median']+([] if task in ['A_all','A_depth5','A_depth15'] else ['Persistence'])
    for bname in bnames:
        p=baseline_predict(d,mode,train,test,bname); ins=baseline_inner(d,mode,train,bname); selection[bname]=ins; trained[bname]=(p,None)
        ytrue=d.loc[test,'Response_N'].to_numpy(float)
        for k,ix in enumerate(test): preds.append({'task':task,'scheme':scheme,'outer':str(label),'model':bname,'selected_family':'','Run_ID':d.loc[ix,'Run_ID'],'Group_VW':d.loc[ix,'Group_VW'],'Speed_kmh':float(d.loc[ix,'Speed_kmh']),'Load_kN':float(d.loc[ix,'Load_kN']),'Pass_T':int(d.loc[ix,'Pass_T']),'Depth_cm':int(d.loc[ix,'Depth_cm']),'y_true':float(ytrue[k]),'y_pred':float(p[k]),'std':None})
    winner=min(selection,key=lambda x:(selection[x],x))
    if winner in trained:
        p,sd=trained[winner]; ytrue=d.loc[test,'Response_N'].to_numpy(float)
        for k,ix in enumerate(test): preds.append({'task':task,'scheme':scheme,'outer':str(label),'model':'Selected_inner','selected_family':winner,'Run_ID':d.loc[ix,'Run_ID'],'Group_VW':d.loc[ix,'Group_VW'],'Speed_kmh':float(d.loc[ix,'Speed_kmh']),'Load_kN':float(d.loc[ix,'Load_kN']),'Pass_T':int(d.loc[ix,'Pass_T']),'Depth_cm':int(d.loc[ix,'Depth_cm']),'y_true':float(ytrue[k]),'y_pred':float(p[k]),'std':None if sd is None else float(sd[k])})
    return foldrows,preds,searchrows


def run_scheme(task,scheme):
    d,_,_=task_frame(task)
    labels=np.unique(d.Group_VW.astype(str) if scheme=='VW' else (d.Speed_kmh.astype(str) if scheme=='speed' else d.Load_kN.astype(str)))
    results=Parallel(n_jobs=N_JOBS,backend='loky',verbose=0)(delayed(one_outer)(task,scheme,l) for l in labels)
    fr=[]; pr=[]; sr=[]
    for a,b,c in results: fr+=a; pr+=b; sr+=c
    return fr,pr,sr

phase('BENCHMARK_CORE_START')
core_tasks=['A_all','A_paired','B_history','C_delta','A_depth5','A_depth15','B_depth5','B_depth15']
fold_rows=[]; pred_rows=[]; search_rows=[]
for ti,task in enumerate(core_tasks,1):
    phase(f'CORE_{ti:02d}_{task}')
    a,b,c=run_scheme(task,'VW'); fold_rows+=a; pred_rows+=b; search_rows+=c
    jdump(STATE/'progress.json',{'completed_core_tasks':core_tasks[:ti],'fold_rows':len(fold_rows),'prediction_rows':len(pred_rows)})
phase('BENCHMARK_CORE_COMPLETE')

# Stress tests only on unified memoryless/history to limit multiplicity and preserve interpretability.
for task in ['A_all','B_history']:
    for scheme in ['speed','load']:
        phase(f'STRESS_{task}_{scheme}')
        a,b,c=run_scheme(task,scheme); fold_rows+=a; pred_rows+=b; search_rows+=c
phase('ROBUSTNESS_COMPLETE')

fold_df=pd.DataFrame(fold_rows); pred_df=pd.DataFrame(pred_rows); search_df=pd.DataFrame(search_rows)
fold_df.to_csv(TABLE/'fold_results.csv',index=False); pred_df.to_csv(TABLE/'oof_predictions.csv',index=False); search_df.to_csv(TABLE/'hyperparameter_search.csv',index=False)

summary=[]; bygroup=[]; bypass=[]; intervals=[]
for (task,scheme,model),z in pred_df.groupby(['task','scheme','model'],sort=False):
    d,_,_=task_frame(task); expected=len(d)
    complete=(len(z)==expected and z[['Run_ID','Depth_cm']].drop_duplicates().shape[0]==expected)
    base={'task':task,'scheme':scheme,'model':model,'complete_oof':bool(complete),'expected_n':int(expected)}
    if not complete:
        summary.append({**base,'n':int(len(z))}); continue
    y=z.y_true.to_numpy(float); p=z.y_pred.to_numpy(float); g=z.Group_VW.to_numpy()
    summary.append({**base,**metrics(y,p,g)})
    for gv,zz in z.groupby('Group_VW'):
        bygroup.append({**base,'Group_VW':gv,**metrics(zz.y_true.to_numpy(float),zz.y_pred.to_numpy(float),zz.Group_VW.to_numpy())})
    for ps,zz in z.groupby('Pass_T'):
        bypass.append({**base,'Pass_T':int(ps),**metrics(zz.y_true.to_numpy(float),zz.y_pred.to_numpy(float),zz.Group_VW.to_numpy())})
    if model=='GPR' and z['std'].notna().all():
        for nominal,q in [(0.80,1.2815515655446004),(0.95,1.959963984540054)]:
            sd=z['std'].to_numpy(float); lo=p-q*sd; hi=p+q*sd; cov=((y>=lo)&(y<=hi)).astype(float)
            intervals.append({**base,'nominal':nominal,'empirical_coverage':float(cov.mean()),'mean_width_N':float(np.mean(hi-lo))})
summary_df=pd.DataFrame(summary); bygroup_df=pd.DataFrame(bygroup); bypass_df=pd.DataFrame(bypass); interval_df=pd.DataFrame(intervals)
summary_df.to_csv(TABLE/'summary.csv',index=False); bygroup_df.to_csv(TABLE/'by_group.csv',index=False); bypass_df.to_csv(TABLE/'by_pass.csv',index=False); interval_df.to_csv(TABLE/'gpr_intervals.csv',index=False)

# History ablation on identical 82 rows.
def sm(task,model,scheme='VW'):
    q=summary_df[(summary_df.task==task)&(summary_df.model==model)&(summary_df.scheme==scheme)&(summary_df.complete_oof==True)]
    return None if q.empty else q.iloc[0].to_dict()
hist=[]
for model in MODELS+['Median','Persistence','Selected_inner']:
    a=sm('A_paired',model); b=sm('B_history',model); c=sm('C_delta',model)
    if b:
        hist.append({'model':model,'A_paired_Group_MAE':None if not a else a.get('Group_MAE'),'B_history_Group_MAE':b.get('Group_MAE'),'C_delta_Group_MAE':None if not c else c.get('Group_MAE'),'history_improvement_vs_A':None if not a else float(a.get('Group_MAE')-b.get('Group_MAE'))})
hist_df=pd.DataFrame(hist); hist_df.to_csv(TABLE/'history_ablation.csv',index=False)
jdump(ROOT/'history_ablation.json',{'rows':df_records(hist_df)})
jdump(ROOT/'uncertainty.json',{'note':'GPR model-based conditional intervals; not distribution-free external coverage.','rows':df_records(interval_df)})

# Development-only full-data permutation importance for explainability on A_all.
phase('EXPLAINABILITY_START')
imp=[]; d,feats,mode=task_frame('A_all'); allix=np.arange(len(d)); groups=d.Group_VW.to_numpy(); sp=GroupKFold(n_splits=4)
for model in MODELS:
    # choose parameter set by grouped CV across all development data
    bp,score,_=tune(d,feats,mode,allix,model)
    if bp is None: continue
    fb=fit_bundle(model,bp,d[feats].to_numpy(float),d.Response_N.to_numpy(float)); basep,_=pred_bundle(fb,d[feats].to_numpy(float)); base=float(np.mean(np.abs(d.Response_N.to_numpy(float)-basep)))
    rng=np.random.default_rng(SEED)
    for fi,f in enumerate(feats):
        vals=[]
        for _ in range(20):
            X=d[feats].to_numpy(float).copy(); rng.shuffle(X[:,fi]); pp,_=pred_bundle(fb,X); vals.append(float(np.mean(np.abs(d.Response_N.to_numpy(float)-pp))-base))
        imp.append({'model':model,'feature':f,'MAE_increase_mean':float(np.mean(vals)),'MAE_increase_sd':float(np.std(vals)),'interpretation':'development-only descriptive permutation importance; not external validation'})
imp_df=pd.DataFrame(imp); imp_df.to_csv(TABLE/'permutation_importance.csv',index=False)
phase('EXPLAINABILITY_COMPLETE')

# Publication-oriented figures.
plt.rcParams.update({'figure.dpi':120,'savefig.dpi':180})
def savefig(name): plt.tight_layout(); plt.savefig(FIG/name,bbox_inches='tight'); plt.close()
for xcol,name,xlab in [('Pass_T','01_response_by_pass_depth.png','Sequential pass'),('Load_kN','02_response_by_load_depth.png','Vertical load (kN)'),('Speed_kmh','03_response_by_speed_depth.png','Forward speed (km/h)')]:
    plt.figure(figsize=(7,4.5))
    for dep in [5,15]:
        q=long_df[long_df.Depth_cm==dep].groupby(xcol).Response_N.agg(['mean','sem']).reset_index()
        plt.errorbar(q[xcol],q['mean'],yerr=q['sem'],marker='o',capsize=3,label=f'{dep} cm')
    plt.xlabel(xlab); plt.ylabel('Calibrated vertical response (N)'); plt.legend(title='Provisional depth'); savefig(name)

for task,num in [('A_all','04'),('B_history','05')]:
    q=summary_df[(summary_df.task==task)&(summary_df.scheme=='VW')&(summary_df.complete_oof==True)].sort_values('Group_MAE')
    plt.figure(figsize=(8,5)); plt.barh(q.model,q.Group_MAE); plt.gca().invert_yaxis(); plt.xlabel('Group-equal MAE (N)'); plt.title(task); savefig(f'{num}_model_ranking_{task}.png')
    z=pred_df[(pred_df.task==task)&(pred_df.scheme=='VW')&(pred_df.model=='Selected_inner')]
    plt.figure(figsize=(5.5,5.5)); plt.scatter(z.y_true,z.y_pred,alpha=.8); lo=min(z.y_true.min(),z.y_pred.min()); hi=max(z.y_true.max(),z.y_pred.max()); plt.plot([lo,hi],[lo,hi],'--'); plt.xlabel('Observed response (N)'); plt.ylabel('OOF predicted response (N)'); plt.title(f'{task}: inner-selected family'); savefig(f'{num}b_observed_predicted_{task}.png')

if not hist_df.empty:
    q=hist_df.dropna(subset=['history_improvement_vs_A']).sort_values('history_improvement_vs_A')
    plt.figure(figsize=(8,5)); plt.barh(q.model,q.history_improvement_vs_A); plt.axvline(0,color='black',lw=1); plt.xlabel('A_paired MAE − B_history MAE (N); positive = history helps'); savefig('06_history_improvement.png')

q=summary_df[(summary_df.scheme=='VW')&(summary_df.model=='Selected_inner')&(summary_df.task.isin(['A_all','A_depth5','A_depth15','B_history','B_depth5','B_depth15']))]
plt.figure(figsize=(8,5)); plt.barh(q.task,q.Group_MAE); plt.xlabel('Group-equal MAE (N)'); plt.title('Unified versus depth-specific'); savefig('07_unified_vs_depth_specific.png')

for task,num in [('A_all','08'),('B_history','09')]:
    q=bypass_df[(bypass_df.task==task)&(bypass_df.scheme=='VW')&(bypass_df.model=='Selected_inner')]
    plt.figure(figsize=(7,4)); plt.plot(q.Pass_T,q.MAE,marker='o'); plt.xlabel('Pass'); plt.ylabel('MAE (N)'); plt.title(f'Error by pass: {task}'); savefig(f'{num}_error_by_pass_{task}.png')

if not interval_df.empty:
    q=interval_df[(interval_df.task=='A_all')&(interval_df.scheme=='VW')]
    plt.figure(figsize=(6,4)); plt.bar(q.nominal.astype(str),q.empirical_coverage); plt.plot([-0.4,1.4],[.8,.8],'--',alpha=.4); plt.ylabel('Empirical coverage'); plt.xlabel('Nominal interval'); plt.title('GPR interval calibration'); savefig('10_gpr_coverage.png')

# Summary JSON and dynamic report.
valid=summary_df[(summary_df.scheme=='VW')&(summary_df.complete_oof==True)]
best={}
for task in core_tasks:
    q=valid[(valid.task==task)&(valid.model.isin(MODELS))]
    if not q.empty:
        r=q.sort_values(['Group_MAE','RMSE']).iloc[0]; best[task]={'model':r.model,'Group_MAE_N':float(r.Group_MAE),'RMSE_N':float(r.RMSE),'R2_pooled':None if pd.isna(r.R2_pooled) else float(r.R2_pooled)}

summary_json={'schema':'soilbin.q1.v1','response_status':RESPONSE_STATUS,'depth_mapping_status':DEPTH_MAP_STATUS,'models':MODELS,'baselines':BASELINES,'primary_metric':'Group-equal MAE','validation':'Nested grouped CV; outer leave-one-VxW-sequence-out; inner GroupKFold=4','parallel_outer_jobs':N_JOBS,'best_by_task':best,'all_metrics':df_records(summary_df),'no_external_validation':True}
jdump(ROOT/'summary.json',summary_json)

manifest={'schema':'soilbin.q1.v1','finished_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,'python':sys.version.split()[0],'cpu_count':os.cpu_count(),'n_jobs':N_JOBS,'models':MODELS,'input_hashes':{'conditions':qc['conditions_sha256'],'targets':qc['targets_sha256']},'data_freeze':{'runs':51,'long_rows':102,'history_rows':82},'depth_map':DEPTH_MAP,'depth_map_status':DEPTH_MAP_STATUS,'response_status':RESPONSE_STATUS,'external_validation':False,'evidence_files':[]}

report=[]
report.append('# SoilBin Q1/Q2 Evidence Report\n')
report.append('## Scientific status\n')
report.append(f'- Response currently modeled as **calibrated vertical force proxy (N)**. Stress conversion is pending effective sensor area/direct stress calibration.\n- Depth mapping is **provisional**: LC5→5 cm, LC1→15 cm, inferred from response magnitudes and soil-mechanics plausibility; experimental layout confirmation is required.\n- No independent external validation campaign is available.\n')
report.append('## Design\n- 51 valid controlled soil-bin runs; 3 speed levels (1,2,3 km/h), 3 vertical loads (2,3,4 kN), sequential passes T1–T6, two depths.\n- 102 depth observations are not treated as independent experiments. Sequential passes are not treated as independent replicates.\n')
report.append('## Validation\n- Primary: nested grouped CV with complete V×W sequences held out.\n- Secondary stress tests: leave-one-speed-out and leave-one-load-out.\n- Model selection uses inner folds only.\n')
report.append('## Model families\n'+'\n'.join([f'- {m}' for m in MODELS])+'\n')
report.append('## Best predictive results by task\n')
for task,r in best.items(): report.append(f"- **{task}**: {r['model']}; Group-MAE={r['Group_MAE_N']:.3f} N; RMSE={r['RMSE_N']:.3f} N; pooled R²={r['R2_pooled']}\n")
report.append('\n## Statistical inference caution\n'+stats_note+'\n')
report.append('\n## Required pre-publication confirmations\n1. Confirm LC5=5 cm and LC1=15 cm from sensor layout/log.\n2. Confirm force-to-stress conversion and report final stress units (e.g., kPa) before labeling the response “vertical soil stress”.\n3. Keep the absence of independent external validation explicit.\n')
(ROOT/'FINAL_REPORT.md').write_text('\n'.join(report),encoding='utf-8')

# Evidence sufficiency checklist and reproducibility manifest.
checklist={'data_qc':True,'physical_levels_recorded':True,'grouped_nested_cv':True,'random_split_used':False,'nine_models':True,'history_ablation':True,'persistence_baseline':True,'depth_specific_comparison':True,'leave_speed_out':True,'leave_load_out':True,'gpr_uncertainty':True,'permutation_importance':True,'statistical_effects':STATSMODELS_OK,'external_validation':False,'depth_mapping_confirmed':False,'force_to_stress_conversion_confirmed':False}
jdump(ROOT/'publication_manifest.json',{'checklist':checklist,'limitations':['Depth mapping remains provisional.','Response currently N force proxy, not verified kPa stress.','No independent external validation.','Only nine VxW sequence groups; inferential p-values require caution.'],'figure_count':len(list(FIG.glob('*.png'))),'table_count':len(list(TABLE.glob('*.csv')))})

for p in ROOT.rglob('*'):
    if p.is_file(): manifest['evidence_files'].append(str(p.relative_to(ROOT)))
jdump(ROOT/'run_manifest.json',manifest)

zip_path=Path('/kaggle/working/soilbin_q1_q2_evidence_v1.zip') if Path('/kaggle/working').exists() else Path('soilbin_q1_q2_evidence_v1.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.rglob('*'):
        if p.is_file(): z.write(p,p.relative_to(ROOT.parent))
phase('REPORT_COMPLETE')
print(json.dumps({'status':'COMPLETE','evidence_root':str(ROOT),'zip':str(zip_path),'best_by_task':best,'limitations':checklist},ensure_ascii=False),flush=True)
phase('DONE')
