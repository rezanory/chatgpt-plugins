from __future__ import annotations
import base64,gzip,hashlib,itertools,json,math,os,sys,time,traceback
from io import BytesIO,StringIO
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

SEED=20260929
np.random.seed(SEED)
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
TARGETS=['LC1_peak_magnitude_N_delta','LC5_peak_magnitude_N_delta']
EXPECTED_DATA='dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059'
V23_REF=23.104189837450967
V29_FIXED_REF=21.657327708415846
V29_T2T6_REF=20.970434406316773
V29_ORACLE_REF=14.923
OUT=Path(os.environ.get('SOILBIN_OUTPUT_ROOT','/kaggle/working/SOILBIN_V31_V50'))
OUT.mkdir(parents=True,exist_ok=True)


def _jsonable(x):
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,)): return float(x)
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,pd.Series): return x.to_dict()
    if isinstance(x,pd.DataFrame): return x.to_dict(orient='records')
    return str(x)

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=_jsonable),encoding='utf-8')

def group_weights(g):
    g=np.asarray(g); u,c=np.unique(g,return_counts=True); m=dict(zip(u,c)); w=np.array([1/m[x] for x in g],float); return w/w.mean()

def gmae1(y,p,g):
    y=np.asarray(y,float); p=np.asarray(p,float); g=np.asarray(g)
    return float(np.mean([np.mean(np.abs(y[g==x]-p[g==x])) for x in np.unique(g)]))

def force_metrics(y,p,g):
    y=np.asarray(y,float); p=np.asarray(p,float); g=np.asarray(g)
    ch=[gmae1(y[:,j],p[:,j],g) for j in range(2)]
    return {'joint_Group_MAE_N':float(np.mean(ch)),'LC1_Group_MAE_N':ch[0],'LC5_Group_MAE_N':ch[1],
            'group_errors':{str(x):float(np.abs(y[g==x]-p[g==x]).mean()) for x in sorted(set(g))}}

def paired_group_report(y,p,b,g):
    y=np.asarray(y,float); p=np.asarray(p,float); b=np.asarray(b,float); g=np.asarray(g)
    vals=[]
    for x in sorted(set(g)):
        m=g==x; vals.append(float(np.abs(y[m]-b[m]).mean()-np.abs(y[m]-p[m]).mean()))
    d=np.asarray(vals,float)
    signs=np.asarray(list(itertools.product([-1,1],repeat=len(d))),float)
    perm=(signs*d).mean(axis=1)
    rng=np.random.default_rng(SEED)
    boot=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(axis=1)
    return {'mean_improvement_N':float(d.mean()),'groups_improved':int((d>0).sum()),
            'exact_signflip_one_sided_p':float(np.mean(perm>=d.mean()-1e-12)),
            'bootstrap_95CI_N':[float(x) for x in np.quantile(boot,[.025,.975])],
            'group_improvements_N':{str(k):float(v) for k,v in zip(sorted(set(g)),d)}}

def decode_payload_csv(payload,key):
    s=payload.get(key)
    if not s: return None
    raw=gzip.decompress(base64.b64decode(s.strip()))
    return pd.read_csv(BytesIO(raw)), raw

def load(payload):
    d,raw=decode_payload_csv(payload,'model_b64')
    if d is None: raise RuntimeError('MISSING_MODEL')
    actual=hashlib.sha256(raw).hexdigest()
    if payload.get('enforce_fingerprint',True) and actual!=EXPECTED_DATA:
        raise RuntimeError('SOURCE_FINGERPRINT_MISMATCH:'+actual)
    if 'Unnamed: 0' in d.columns: d=d.drop(columns=['Unnamed: 0'])
    if 'index' in d.columns and d['index'].nunique()==len(d): d=d.drop(columns=['index'])
    if len(d)!=51 or d.Run_ID.nunique()!=51: raise RuntimeError(f'EXPECTED_51_RUNS:{len(d)}')
    d['Group']=d.Run_ID.str.extract(r'^(V\dW\d)')[0]
    d['Speed']=d.Speed_level.astype(float)
    d['Load']=d.Weight_level.astype(float)+1.0
    d['Pass_T']=d.Pass_T.astype(int)
    p=d[['Group','Pass_T']+TARGETS].copy(); p['Pass_T']+=1
    p=p.rename(columns={TARGETS[0]:'prev_LC1',TARGETS[1]:'prev_LC5'})
    h=d.merge(p,on=['Group','Pass_T'],how='left',validate='one_to_one')
    h=h[h.prev_LC1.notna() & h.prev_LC5.notna()].sort_values(['Group','Pass_T']).reset_index(drop=True)
    if len(h)!=41: raise RuntimeError(f'EXPECTED_41_TRANSITIONS:{len(h)}')
    wf=None
    if payload.get('wave_b64'):
        wf,_=decode_payload_csv(payload,'wave_b64')
        if 'Unnamed: 0' in wf.columns: wf=wf.drop(columns=['Unnamed: 0'])
        if 'index' in wf.columns and wf['index'].nunique()==len(wf): wf=wf.drop(columns=['index'])
        if len(wf)!=51: raise RuntimeError(f'EXPECTED_51_WAVE_ROWS:{len(wf)}')
    return d,h,wf,actual

def run_map(d):
    return {(str(r.Group),int(r.Pass_T)):r for _,r in d.iterrows()}

def history_frame(d,h,max_depth=4):
    mp=run_map(d); z=h.copy()
    for k in range(1,max_depth+1):
        for s,col in [('LC1',TARGETS[0]),('LC5',TARGETS[1])]:
            vals=[]; av=[]
            for _,r in z.iterrows():
                rr=mp.get((str(r.Group),int(r.Pass_T)-k))
                if rr is None: vals.append(0.0); av.append(0.0)
                else: vals.append(float(rr[col])); av.append(1.0)
            z[f'lag{k}_{s}']=vals; z[f'lag{k}_{s}_avail']=av
    for s in ('LC1','LC5'):
        for k in range(1,max_depth):
            z[f'delta_lag{k}_{s}']=z[f'lag{k}_{s}']-z[f'lag{k+1}_{s}']
    for s in ('LC1','LC5'):
        vals=[]; slopes=[]; means=[]; cum=[]
        for _,r in z.iterrows():
            seq=[]
            for k in range(max_depth,0,-1):
                if r[f'lag{k}_{s}_avail']>0: seq.append(float(r[f'lag{k}_{s}']))
            if seq:
                means.append(float(np.mean(seq))); vals.append(float(seq[-1]))
                slopes.append(float(np.polyfit(np.arange(len(seq)),seq,1)[0]) if len(seq)>=2 else 0.0)
                cum.append(float(np.sum(np.abs(np.diff(seq)))) if len(seq)>=2 else 0.0)
            else: means.append(0.0); vals.append(0.0); slopes.append(0.0); cum.append(0.0)
        z[f'hist_mean_{s}']=means; z[f'hist_slope_{s}']=slopes; z[f'hist_cumabs_{s}']=cum
    return z

def attach_prior_wave(h,wf):
    if wf is None: return h.copy(),[]
    idcols={'Run_ID','V_code','Speed_level','W_code','Weight_level','Pass_T'}
    feat=[c for c in wf.columns if c not in idcols]
    mp={(str(r.V_code)+str(r.W_code),int(r.Pass_T)):r for _,r in wf.iterrows()}
    # Prefer Group derivation directly from Run_ID if available.
    mp2={(str(r.Run_ID)[:-2] if False else None):r for _,r in wf.iterrows()}
    by={}
    for _,r in wf.iterrows():
        group=str(r.Run_ID).split('T')[0]
        by[(group,int(r.Pass_T))]=r
    z=h.copy()
    for c in feat:
        arr=[]
        for _,r in z.iterrows():
            q=by.get((str(r.Group),int(r.Pass_T)-1))
            v=np.nan if q is None else q[c]
            arr.append(float(v) if pd.notna(v) else 0.0)
        z['pw_'+c]=arr
    return z,['pw_'+c for c in feat]

def base_features(z):
    return ['Speed','Load','Pass_T','prev_LC1','prev_LC5']

def history_cols(z,depth=4,both=True,sensor=None,include_summary=True):
    cols=['Speed','Load','Pass_T']
    sensors=['LC1','LC5'] if both else [sensor]
    for k in range(1,depth+1):
        for s in sensors:
            cols += [f'lag{k}_{s}',f'lag{k}_{s}_avail']
    if include_summary:
        for s in sensors: cols += [f'hist_mean_{s}',f'hist_slope_{s}',f'hist_cumabs_{s}']
    return cols

def wave_shape_cols(pwcols):
    keys=('shape_norm_','shape_turn_','shape_dct','sample_dct','cross_shape_dct','xcorr_lag_ms','xcorr_r','aligned_scale','aligned_r2','traj3d_','phase_area','aligned_norm_','active_duration')
    return [c for c in pwcols if any(k in c for k in keys) and not c.endswith('shape_peak')]

# ----- nested grouped regression -----
def model_catalog(n_features):
    # Deliberately low-capacity for n=41; model-family search is not the scientific variable.
    return [
      ('RIDGE_1',make_pipeline(StandardScaler(),Ridge(alpha=1.0))),
      ('RIDGE_10',make_pipeline(StandardScaler(),Ridge(alpha=10.0))),
      ('RIDGE_100',make_pipeline(StandardScaler(),Ridge(alpha=100.0))),
    ]

def fit_weighted(model,X,y,g):
    w=group_weights(g)
    if hasattr(model,'steps'):
        try:model.fit(X,y,ridge__sample_weight=w)
        except TypeError:model.fit(X,y)
    else:
        try:model.fit(X,y,sample_weight=w)
        except TypeError:model.fit(X,y)
    return model

def select_model(train,features,target):
    groups=np.asarray(train.Group)
    scores=[]
    for name,proto in model_catalog(len(features)):
        ps=np.full(len(train),np.nan)
        for vg in sorted(set(groups)):
            tr=groups!=vg; va=groups==vg
            m=clone(proto); fit_weighted(m,train.loc[tr,features].to_numpy(float),train.loc[tr,target].to_numpy(float),groups[tr])
            ps[va]=m.predict(train.loc[va,features].to_numpy(float))
        scores.append((gmae1(train[target],ps,groups),name,proto))
    scores.sort(key=lambda x:(x[0],x[1]))
    return scores[0][2],{'selected':scores[0][1],'inner_Group_MAE_N':float(scores[0][0]),'scores':[{'model':n,'Group_MAE_N':float(s)} for s,n,_ in scores]}

def grouped_oof(z,features,targets=TARGETS,model_select=True):
    g=np.asarray(z.Group); y=z[targets].to_numpy(float); p=np.full_like(y,np.nan,float); audit=[]
    for outer in sorted(set(g)):
        tr=z.Group!=outer; te=z.Group==outer
        for j,t in enumerate(targets):
            if model_select:
                proto,meta=select_model(z.loc[tr],features,t)
            else:
                proto=make_pipeline(StandardScaler(),Ridge(alpha=10.0));meta={'selected':'RIDGE_10'}
            m=clone(proto); fit_weighted(m,z.loc[tr,features].to_numpy(float),z.loc[tr,t].to_numpy(float),np.asarray(z.loc[tr,'Group']))
            p[te,j]=m.predict(z.loc[te,features].to_numpy(float))
            audit.append({'outer':outer,'target':t,**meta})
    if not np.isfinite(p).all(): raise RuntimeError('OOF_INCOMPLETE')
    return p,audit

def compare_feature_sets(z,sets):
    y=z[TARGETS].to_numpy(float);g=np.asarray(z.Group); out=[]; preds={};aud={}
    for name,cols in sets.items():
        p,a=grouped_oof(z,cols); m=force_metrics(y,p,g); preds[name]=p;aud[name]=a;out.append({'variant':name,**m})
    return out,preds,aud

# ----- latent PLS -----
def pls_oof(z,history_features,n_components,include_conditions=True):
    g=np.asarray(z.Group); y=z[TARGETS].to_numpy(float); p=np.full_like(y,np.nan); states=np.full((len(z),n_components),np.nan); audits=[]
    for outer in sorted(set(g)):
        tr=g!=outer;te=g==outer
        Xtr=z.loc[tr,history_features].to_numpy(float);Xte=z.loc[te,history_features].to_numpy(float)
        scaler=StandardScaler().fit(Xtr); A=scaler.transform(Xtr);B=scaler.transform(Xte)
        nc=max(1,min(n_components,A.shape[1],A.shape[0]-1))
        pls=PLSRegression(n_components=nc,scale=False,max_iter=1000)
        pls.fit(A,y[tr]); p[te]=pls.predict(B); states[te,:nc]=pls.transform(B)[:,:nc]
        audits.append({'outer':outer,'n_components':nc})
    return p,states,audits

def choose_pls_dim(z,features,dims=(1,2,3,4)):
    rows=[];cache={}
    for k in dims:
        p,s,a=pls_oof(z,features,k);sc=force_metrics(z[TARGETS].to_numpy(float),p,np.asarray(z.Group))['joint_Group_MAE_N']; rows.append((sc,k));cache[k]=(p,s,a)
    rows.sort(); sc,k=rows[0]
    return k,cache[k], [{'n_components':int(kk),'joint_Group_MAE_N':float(ss)} for ss,kk in rows]

# ----- Lane implementations -----
def lane_v31(d,h,wf):
    z=history_frame(d,h);z,pw=attach_prior_wave(z,wf); shape=wave_shape_cols(pw)
    sets={'BASE_PREV_PEAKS':base_features(z),'ALIGNED_GEOMETRY_ONLY':['Speed','Load','Pass_T']+shape,
          'BASE_PLUS_ALIGNED_GEOMETRY':base_features(z)+shape}
    rows,preds,aud=compare_feature_sets(z,sets); y=z[TARGETS].to_numpy(float);g=np.asarray(z.Group)
    base=preds['BASE_PREV_PEAKS']; champ=min(rows,key=lambda x:x['joint_Group_MAE_N'])
    for r in rows:r['paired_vs_base']=paired_group_report(y,preds[r['variant']],base,g)
    return {'question':'Can leakage-safe previous-pass lag-aligned shape/geometry improve amplitude/state prediction?',
            'status':'COMPLETE','scientific_status':'REPEATED_DEVELOPMENT_GROUP_OOF','summary':rows,'champion':champ,'n_shape_features':len(shape),
            'reference_locked':{'V23':V23_REF,'V29_FIXED':V29_FIXED_REF,'V29_T2_T6':V29_T2T6_REF},'audit':aud,
            'guards':{'target_pass_waveform_used':False,'prior_pass_waveform_only':True,'whole_speed_load_outer_holdout':True}}

def lane_v32(d,h,wf):
    # Sensor baseline-noise lower-bound proxy in force units; not a full irreducible-error estimate.
    rows=[]
    for s in ('LC1','LC5'):
        sig=d[f'{s}_baseline_noise_sigma_raw_kg'].to_numpy(float)
        raw=np.abs(d[f'{s}_peak_delta_raw_kg'].to_numpy(float));force=np.abs(d[f'{s}_peak_magnitude_N_delta'].to_numpy(float))
        cal=np.divide(force,raw,out=np.zeros_like(force),where=raw>1e-9)
        noise=sig*cal
        rows.append({'sensor':s,'median_baseline_noise_force_proxy_N':float(np.nanmedian(noise)),'p95_proxy_N':float(np.nanquantile(noise,.95))})
    return {'question':'How much current error is plausibly reducible versus measurement/noise floor?','status':'COMPLETE','scientific_status':'DIAGNOSTIC_ONLY',
            'sensor_noise_floor_proxy':rows,'development_headroom':{'V29_T2_T6_N':V29_T2T6_REF,'historical_oracle_N':V29_ORACLE_REF,'gap_N':float(V29_T2T6_REF-V29_ORACLE_REF)},
            'interpretation_guard':'Baseline-noise and oracle gaps are proxies/bounds, not a proven irreducible Bayes error.'}

def lane_v33(d,h,wf):
    z=history_frame(d,h); feats=history_cols(z,4,both=True,include_summary=True)
    k,(p,s,a),curve=choose_pls_dim(z,feats);m=force_metrics(z[TARGETS].to_numpy(float),p,np.asarray(z.Group))
    return {'question':'What is the minimum low-dimensional predictive latent state supported by grouped OOF data?','status':'COMPLETE','scientific_status':'REPEATED_DEVELOPMENT_GROUP_OOF',
            'selected_dim':k,'dimension_curve':curve,'metrics':m,'state_variance':np.nanvar(s,axis=0).tolist(),'audit':a,'feature_count':len(feats)}

def lane_v34(d,h,wf):
    z=history_frame(d,h)
    # Restrict to rows with lag2 available to test incremental older-history value.
    q=z[(z.lag2_LC1_avail>0)&(z.lag2_LC5_avail>0)].reset_index(drop=True)
    immediate=['Speed','Load','Pass_T','lag1_LC1','lag1_LC5']
    older=immediate+['lag2_LC1','lag2_LC5','hist_slope_LC1','hist_slope_LC5','hist_cumabs_LC1','hist_cumabs_LC5']
    rows,preds,aud=compare_feature_sets(q,{'IMMEDIATE_STATE':immediate,'STATE_PLUS_OLDER_HISTORY':older})
    y=q[TARGETS].to_numpy(float);g=np.asarray(q.Group);base=preds['IMMEDIATE_STATE'];full=preds['STATE_PLUS_OLDER_HISTORY']
    return {'question':'After conditioning on immediate state, does older history retain predictive information?','status':'COMPLETE','scientific_status':'MARKOV_SUFFICIENCY_DIAGNOSTIC',
            'summary':rows,'paired_incremental_older_history':paired_group_report(y,full,base,g),'n_rows':len(q),'audit':aud}

def lane_v35(d,h,wf):
    z=history_frame(d,h);rows=[];preds={}
    for dep in (1,2,3,4):
        cols=history_cols(z,dep,both=True,include_summary=False);p,a=grouped_oof(z,cols);m=force_metrics(z[TARGETS].to_numpy(float),p,np.asarray(z.Group));rows.append({'depth':dep,**m});preds[dep]=p
    best=min(rows,key=lambda x:x['joint_Group_MAE_N']); tol=best['joint_Group_MAE_N']+0.5
    sufficient=min(r['depth'] for r in rows if r['joint_Group_MAE_N']<=tol)
    return {'question':'What minimum history depth is sufficient within 0.5 N of the best grouped OOF depth?','status':'COMPLETE','scientific_status':'REPEATED_DEVELOPMENT_MEMORY_SUFFICIENCY',
            'depth_curve':rows,'best_depth':best['depth'],'minimum_sufficient_depth_0p5N':sufficient}

def lane_v36(d,h,wf):
    z=history_frame(d,h)
    sets={'LC1_HISTORY':history_cols(z,4,both=False,sensor='LC1'),'LC5_HISTORY':history_cols(z,4,both=False,sensor='LC5'),'BOTH_HISTORY':history_cols(z,4,both=True)}
    rows,preds,aud=compare_feature_sets(z,sets)
    y=z[TARGETS].to_numpy(float);g=np.asarray(z.Group)
    return {'question':'Are LC1 and LC5 separately observable views or is joint history materially better?','status':'COMPLETE','scientific_status':'CROSS_DEPTH_OBSERVABILITY_DIAGNOSTIC',
            'summary':rows,'joint_gain_over_LC1':paired_group_report(y,preds['BOTH_HISTORY'],preds['LC1_HISTORY'],g),
            'joint_gain_over_LC5':paired_group_report(y,preds['BOTH_HISTORY'],preds['LC5_HISTORY'],g),'audit':aud}

def lane_v37(d,h,wf):
    z=history_frame(d,h)
    results=[]
    # Predict LC1: own history vs own+LC5. Predict LC5: own vs own+LC1.
    for target,same,other in [(TARGETS[0],'LC1','LC5'),(TARGETS[1],'LC5','LC1')]:
        own=['Speed','Load','Pass_T']+[f'lag{k}_{same}' for k in range(1,5)]+[f'lag{k}_{same}_avail' for k in range(1,5)]
        cross=own+[f'lag{k}_{other}' for k in range(1,5)]+[f'lag{k}_{other}_avail' for k in range(1,5)]
        p0=np.full(len(z),np.nan);p1=np.full(len(z),np.nan);g=np.asarray(z.Group)
        for outer in sorted(set(g)):
            tr=g!=outer;te=g==outer
            for cols,out in [(own,p0),(cross,p1)]:
                proto,meta=select_model(z.loc[tr],cols,target);m=clone(proto);fit_weighted(m,z.loc[tr,cols].to_numpy(float),z.loc[tr,target].to_numpy(float),g[tr]);out[te]=m.predict(z.loc[te,cols].to_numpy(float))
        gain=gmae1(z[target],p0,g)-gmae1(z[target],p1,g)
        results.append({'target':target,'source_added':other,'own_only_Group_MAE_N':gmae1(z[target],p0,g),'with_cross_Group_MAE_N':gmae1(z[target],p1,g),'conditional_predictive_gain_N':float(gain)})
    return {'question':'After controlling own history and conditions, is cross-sensor history directionally informative?','status':'COMPLETE','scientific_status':'CONDITIONAL_PREDICTIVE_INFORMATION_NOT_CAUSALITY','directional_results':results}

def lane_v38(d,h,wf):
    z=history_frame(d,h)
    q=z[(z.lag2_LC1_avail>0)&(z.lag2_LC5_avail>0)].reset_index(drop=True)
    state=['Speed','Load','Pass_T','lag1_LC1','lag1_LC5']
    path=state+['delta_lag1_LC1','delta_lag1_LC5','hist_slope_LC1','hist_slope_LC5','hist_cumabs_LC1','hist_cumabs_LC5']
    rows,preds,aud=compare_feature_sets(q,{'STATE_ONLY':state,'STATE_PLUS_PATH':path})
    # trajectory closed-area diagnostic per group
    areas=[]
    for gr,s in d.sort_values('Pass_T').groupby('Group'):
        x=s[TARGETS[0]].to_numpy(float);y=s[TARGETS[1]].to_numpy(float)
        if len(x)>=3:
            area=.5*abs(float(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1))))
            areas.append({'Group':gr,'closed_phase_area_N2':area,'n':len(x)})
    return {'question':'Does path information improve prediction beyond immediate state, consistent with path dependence/hysteresis?','status':'COMPLETE','scientific_status':'PATH_DEPENDENCE_DIAGNOSTIC',
            'summary':rows,'paired_path_gain':paired_group_report(q[TARGETS].to_numpy(float),preds['STATE_PLUS_PATH'],preds['STATE_ONLY'],np.asarray(q.Group)),
            'closed_LC1_LC5_phase_area':areas,'interpretation_guard':'Closed phase area is geometric, not directly dissipated energy.'}

def lane_v39(d,h,wf):
    z=history_frame(d,h);cols=base_features(z);p,a=grouped_oof(z,cols);y=z[TARGETS].to_numpy(float);g=np.asarray(z.Group)
    rows=[]
    for j,s in enumerate(('LC1','LC5')):
        for t in sorted(z.Pass_T.unique()):
            m=z.Pass_T.to_numpy()==t
            rows.append({'sensor':s,'pass':int(t),'n':int(m.sum()),'MAE_N':float(np.mean(np.abs(y[m,j]-p[m,j]))),'bias_N':float(np.mean(p[m,j]-y[m,j])),'target_sd_N':float(np.std(y[m,j]))})
    return {'question':'How does predictability vary by sensor and pass?','status':'COMPLETE','scientific_status':'DESCRIPTIVE_OOF_PREDICTABILITY_PROFILE','profile':rows,'overall':force_metrics(y,p,g),'audit':a}

def lane_v40(d,h,wf):
    z=history_frame(d,h)
    feats=[c for c in history_cols(z,4,both=True) if c!='Pass_T']
    g=np.asarray(z.Group); true=z.Pass_T.to_numpy(int);pred=np.full(len(z),np.nan)
    # Ridge regression for ordinal pass, no Pass_T input.
    for outer in sorted(set(g)):
        tr=g!=outer;te=g==outer;m=make_pipeline(StandardScaler(),Ridge(alpha=10));m.fit(z.loc[tr,feats],true[tr],ridge__sample_weight=group_weights(g[tr]));pred[te]=m.predict(z.loc[te,feats])
    rounded=np.clip(np.rint(pred),2,6).astype(int)
    # Force prediction with and without Pass_T
    withp=history_cols(z,4,both=True);nop=[c for c in withp if c!='Pass_T']
    rows,preds,aud=compare_feature_sets(z,{'WITH_PASS':withp,'PASS_BLIND':nop})
    return {'question':'Can physical history reconstruct soil-age/pass and retain force prediction without explicit pass number?','status':'COMPLETE','scientific_status':'PASS_BLIND_STATE_DIAGNOSTIC',
            'pass_reconstruction':{'MAE_pass':float(np.mean(np.abs(pred-true))),'rounded_accuracy':float(np.mean(rounded==true)),'spearman_r':float(spearmanr(pred,true).statistic)},'force_summary':rows,'audit':aud}

def lane_v41(d,h,wf):
    # Empirical invariants only because Speed/Load are ordinal levels, not dimensional measurements.
    q=d.copy(); eps=1e-9
    candidates={
      'LC5_over_LC1':q[TARGETS[1]]/(q[TARGETS[0]]+eps),
      'sum_over_LoadLevel':(q[TARGETS[0]]+q[TARGETS[1]])/(q.Load+eps),
      'geom_mean_over_Load':np.sqrt(q[TARGETS[0]]*q[TARGETS[1]])/(q.Load+eps),
      'LC5_over_LC1_times_speed':q[TARGETS[1]]/(q[TARGETS[0]]+eps)*q.Speed,
      'log_ratio':np.log((q[TARGETS[1]]+eps)/(q[TARGETS[0]]+eps)),
    }
    rows=[]
    for name,v in candidates.items():
        v=np.asarray(v,float);rows.append({'candidate':name,'CV_pct':float(100*np.std(v)/max(abs(np.mean(v)),eps)),'MAD':float(np.median(np.abs(v-np.median(v)))),'spearman_with_pass':float(spearmanr(v,q.Pass_T).statistic)})
    rows.sort(key=lambda x:x['CV_pct'])
    return {'question':'Are there low-variation empirical combinations across conditions?','status':'COMPLETE','scientific_status':'EMPIRICAL_INVARIANT_SEARCH_NOT_PHYSICAL_DIMENSIONLESS_LAW','candidates':rows,'best':rows[0]}

def lane_v42(d,h,wf):
    q=d.copy(); eps=1e-9
    transforms={
      'RAW':np.column_stack([q[TARGETS[0]],q[TARGETS[1]]]),
      'DIV_LOAD_LEVEL':np.column_stack([q[TARGETS[0]]/q.Load,q[TARGETS[1]]/q.Load]),
      'DIV_SPEED_LEVEL':np.column_stack([q[TARGETS[0]]/q.Speed,q[TARGETS[1]]/q.Speed]),
      'DIV_LOAD_SPEED':np.column_stack([q[TARGETS[0]]/(q.Load*q.Speed),q[TARGETS[1]]/(q.Load*q.Speed)]),
      'RATIO_TO_COMBINED':np.column_stack([q[TARGETS[0]]/(q[TARGETS[0]]+q[TARGETS[1]]+eps),q[TARGETS[1]]/(q[TARGETS[0]]+q[TARGETS[1]]+eps)])
    }
    rows=[]
    for name,A in transforms.items():
        # collapse score: residual variance after pass-wise centering, normalized by total variance
        resid=[];total=[]
        for j in range(2):
            x=A[:,j]; total.append(float(np.var(x)))
            r=x.copy()
            for t in sorted(q.Pass_T.unique()):
                m=q.Pass_T.to_numpy()==t;r[m]-=np.mean(x[m])
            resid.append(float(np.var(r)))
        score=float(np.mean([resid[j]/max(total[j],eps) for j in range(2)]))
        rows.append({'transform':name,'collapse_score_lower_better':score,'residual_var':resid,'total_var':total})
    rows.sort(key=lambda x:x['collapse_score_lower_better'])
    return {'question':'Which empirical normalization most collapses Speed×Load differences after pass trend removal?','status':'COMPLETE','scientific_status':'EMPIRICAL_NORMALIZED_COLLAPSE_NOT_DIMENSIONAL_ANALYSIS','summary':rows,'best':rows[0]}

def lane_v43(d,h,wf):
    z=history_frame(d,h);features=base_features(z);g=np.asarray(z.Group);sens=[]
    # Fit all data exploratory model; scenario only, not causal.
    mods=[]
    for t in TARGETS:
        m=ExtraTreesRegressor(n_estimators=500,max_depth=3,min_samples_leaf=2,random_state=SEED,n_jobs=1);m.fit(z[features],z[t],sample_weight=group_weights(g));mods.append(m)
    states=z[['prev_LC1','prev_LC5','Pass_T']].median().to_dict()
    for speed in (1.,2.,3.):
        for load in (2.,3.,4.):
            x=np.array([[speed,load,states['Pass_T'],states['prev_LC1'],states['prev_LC5']]],float)
            sens.append({'Speed_level':speed,'Load_encoded':load,'LC1_pred_N':float(mods[0].predict(x)[0]),'LC5_pred_N':float(mods[1].predict(x)[0])})
    return {'question':'How would the learned transition surface respond to changes in Speed/Load for a fixed representative prior state?','status':'COMPLETE','scientific_status':'MODEL_BASED_SCENARIO_NOT_CAUSAL_COUNTERFACTUAL','representative_state':states,'scenario_grid':sens}

def lane_v44(d,h,wf):
    z=history_frame(d,h)
    # Controlled linear / lifted state transition from prev state to current state.
    y=z[TARGETS].to_numpy(float);g=np.asarray(z.Group)
    Xlin=z[['prev_LC1','prev_LC5','Speed','Load','Pass_T']].to_numpy(float)
    Xlift=np.column_stack([Xlin,Xlin[:,0]*Xlin[:,1],Xlin[:,0]**2,Xlin[:,1]**2,Xlin[:,0]*Xlin[:,2],Xlin[:,1]*Xlin[:,2]])
    def oof(A):
        p=np.full_like(y,np.nan)
        for outer in sorted(set(g)):
            tr=g!=outer;te=g==outer
            for j in range(2):
                m=make_pipeline(StandardScaler(),Ridge(alpha=10));m.fit(A[tr],y[tr,j],ridge__sample_weight=group_weights(g[tr]));p[te,j]=m.predict(A[te])
        return p
    persist=z[['prev_LC1','prev_LC5']].to_numpy(float);pl=oof(Xlin);pe=oof(Xlift)
    return {'question':'Can a controlled linear or lifted Koopman-style operator describe one-pass dynamics?','status':'COMPLETE','scientific_status':'KOOPMAN_STYLE_OPERATOR_CHALLENGE',
            'summary':[{'variant':'PERSIST',**force_metrics(y,persist,g)},{'variant':'LINEAR_CONTROLLED_STATE',**force_metrics(y,pl,g)},{'variant':'POLY_LIFTED_OPERATOR',**force_metrics(y,pe,g)}]}

def lane_v45(d,h,wf):
    z=history_frame(d,h);g=np.asarray(z.Group);base=['prev_LC1','prev_LC5','Speed','Load','Pass_T']
    poly=PolynomialFeatures(degree=2,include_bias=False);A=poly.fit_transform(z[base].to_numpy(float));names=poly.get_feature_names_out(base);y=z[TARGETS].to_numpy(float)-z[['prev_LC1','prev_LC5']].to_numpy(float)
    alphas=[0.01,0.1,1.0,5.0];out=[]
    for j,s in enumerate(('LC1','LC5')):
        best=None
        for a in alphas:
            p=np.full(len(z),np.nan)
            for outer in sorted(set(g)):
                tr=g!=outer;te=g==outer;m=make_pipeline(StandardScaler(),Lasso(alpha=a,max_iter=20000,random_state=SEED));m.fit(A[tr],y[tr,j]);p[te]=m.predict(A[te])
            sc=gmae1(y[:,j],p,g)
            if best is None or sc<best[0]:best=(sc,a,p)
        sc,a,p=best;fit=make_pipeline(StandardScaler(),Lasso(alpha=a,max_iter=20000,random_state=SEED));fit.fit(A,y[:,j]);coef=fit[-1].coef_; idx=np.argsort(np.abs(coef))[::-1][:8]
        out.append({'sensor':s,'alpha':a,'delta_Group_MAE_N':float(sc),'top_terms':[{'term':str(names[i]),'coef_scaled':float(coef[i])} for i in idx if abs(coef[i])>1e-8]})
    return {'question':'Can a sparse low-order equation approximate the state transition law?','status':'COMPLETE','scientific_status':'SPARSE_EQUATION_DISCOVERY_REPEATED_DEVELOPMENT','equations':out,'guard':'Coefficients are standardized-model coefficients; physical equation requires dimensional metadata.'}

def lane_v46(d,h,wf):
    z=history_frame(d,h);base=base_features(z);hist=history_cols(z,4,both=True);k,(pp,states,a),curve=choose_pls_dim(z,hist)
    pb,ab=grouped_oof(z,base);ph,ah=grouped_oof(z,hist);y=z[TARGETS].to_numpy(float);g=np.asarray(z.Group)
    rows=[{'representation':'A_CONVENTIONAL',**force_metrics(y,pb,g),'dimension':len(base)},
          {'representation':'B_FULL_HISTORY',**force_metrics(y,ph,g),'dimension':len(hist)},
          {'representation':'C_NATIVE_LATENT_PLS',**force_metrics(y,pp,g),'dimension':k}]
    return {'question':'Which representation—conventional, explicit history, or compressed learned state—generalizes best?','status':'COMPLETE','scientific_status':'VASI_SNSR_REPRESENTATION_CHALLENGE','summary':rows,'latent_dimension_curve':curve,'selected_latent_dim':k}

def surface_geometry(speed,load,z):
    X=np.column_stack([speed,load]);poly=PolynomialFeatures(2,include_bias=True);A=poly.fit_transform(X);m=Ridge(alpha=1e-6,fit_intercept=False).fit(A,z)
    # fine grid surface area and curvature proxy from coefficients
    sx=np.linspace(1,3,25);ly=np.linspace(2,4,25);S,L=np.meshgrid(sx,ly,indexing='ij');G=poly.transform(np.column_stack([S.ravel(),L.ravel()]));Z=m.predict(G).reshape(S.shape)
    area=0.0
    for i in range(S.shape[0]-1):
      for j in range(S.shape[1]-1):
        p00=np.array([S[i,j],L[i,j],Z[i,j]]);p10=np.array([S[i+1,j],L[i+1,j],Z[i+1,j]]);p01=np.array([S[i,j+1],L[i,j+1],Z[i,j+1]]);p11=np.array([S[i+1,j+1],L[i+1,j+1],Z[i+1,j+1]])
        area+=.5*np.linalg.norm(np.cross(p10-p00,p01-p00))+.5*np.linalg.norm(np.cross(p11-p10,p01-p10))
    # numerical curvature proxy from second differences
    curv=float(np.nanmean(np.abs(np.diff(Z,n=2,axis=0)))+np.nanmean(np.abs(np.diff(Z,n=2,axis=1))))
    grad=float(np.nanmean(np.sqrt(np.gradient(Z,axis=0)**2+np.gradient(Z,axis=1)**2)))
    vol=float(np.trapz(np.trapz(Z,ly,axis=1),sx,axis=0))
    return {'surface_area':float(area),'curvature_proxy':curv,'mean_gradient_proxy':grad,'volume_under_surface_proxy':vol,'coef':m.coef_.tolist()}

def lane_v47(d,h,wf):
    rows=[]
    for t in sorted(d.Pass_T.unique()):
        q=d[d.Pass_T==t]
        if len(q)<5: continue
        for s,col in [('LC1',TARGETS[0]),('LC5',TARGETS[1])]:
            rows.append({'pass':int(t),'sensor':s,'n':len(q),**surface_geometry(q.Speed.to_numpy(),q.Load.to_numpy(),q[col].to_numpy())})
    return {'question':'How do 3D Speed×Load response-surface geometry, area, curvature and volume proxies evolve by pass?','status':'COMPLETE','scientific_status':'DESCRIPTIVE_3D_RESPONSE_SURFACE','surfaces':rows}

def trajectory_descriptors(q):
    q=q.sort_values('Pass_T');A=q[TARGETS].to_numpy(float);t=q.Pass_T.to_numpy(float)
    # z-score across project scales to avoid LC5 dominating; pass normalized.
    B=np.column_stack([A[:,0]/max(d_global_sd[0],1e-9),A[:,1]/max(d_global_sd[1],1e-9),(t-1)/5])
    seg=np.diff(B,axis=0);lens=np.linalg.norm(seg,axis=1);turn=[]
    for i in range(1,len(seg)):
        a,b=seg[i-1],seg[i];den=max(np.linalg.norm(a)*np.linalg.norm(b),1e-12);turn.append(float(np.arccos(np.clip(np.dot(a,b)/den,-1,1))))
    tors=[]
    for i in range(2,len(seg)):
        tors.append(float(abs(np.dot(seg[i],np.cross(seg[i-1],seg[i-2])))))
    return {'path_length':float(lens.sum()),'mean_turn_angle':float(np.mean(turn) if turn else 0),'max_turn_angle':float(np.max(turn) if turn else 0),
            'torsion_proxy':float(np.mean(tors) if tors else 0),'net_displacement':float(np.linalg.norm(B[-1]-B[0])),'tortuosity':float(lens.sum()/max(np.linalg.norm(B[-1]-B[0]),1e-9))}

def lane_v48(d,h,wf):
    global d_global_sd; d_global_sd=d[TARGETS].std().to_numpy(float)
    rows=[{'Group':g,'n':len(q),**trajectory_descriptors(q)} for g,q in d.groupby('Group')]
    return {'question':'What geometric trajectories do T1→T6 follow in LC1–LC5–pass state space?','status':'COMPLETE','scientific_status':'DESCRIPTIVE_3D_STATE_TRAJECTORY','trajectories':rows}

def lane_v49(d,h,wf):
    if wf is None: raise RuntimeError('V49_REQUIRES_WAVE_FEATURES')
    cols=['traj3d_path_length','traj3d_mean_turn','traj3d_max_turn','traj3d_torsion_proxy','lc1_lc5_phase_area','aligned_norm_corr','aligned_norm_mae','xcorr_r','aligned_r2']
    cols=[c for c in cols if c in wf.columns]
    summary=[]
    for c in cols:
        summary.append({'feature':c,'mean':float(wf[c].mean()),'median':float(wf[c].median()),'sd':float(wf[c].std()),'spearman_with_pass':float(spearmanr(wf[c],wf.Pass_T).statistic)})
    bypass=wf.groupby('Pass_T')[cols].median(numeric_only=True).reset_index().to_dict(orient='records')
    return {'question':'What 3D waveform-geometry descriptors vary systematically with pass/condition?','status':'COMPLETE','scientific_status':'DESCRIPTIVE_3D_WAVEFORM_GEOMETRY','feature_summary':summary,'median_by_pass':bypass}

def lane_v50(d,h,wf):
    z=history_frame(d,h);z,pw=attach_prior_wave(z,wf)
    geom=[c for c in pw if any(k in c for k in ('traj3d_','phase_area','turn_angle','arc_length','aligned_norm_','xcorr_r','aligned_r2'))]
    sets={'BASE':base_features(z),'GEOMETRY_ONLY':['Speed','Load','Pass_T']+geom,'BASE_PLUS_GEOMETRY':base_features(z)+geom}
    rows,preds,aud=compare_feature_sets(z,sets);y=z[TARGETS].to_numpy(float);g=np.asarray(z.Group)
    base=preds['BASE']
    for r in rows:r['paired_vs_base']=paired_group_report(y,preds[r['variant']],base,g)
    return {'question':'Do measurable geometric descriptors add grouped out-of-sample predictive value?','status':'COMPLETE','scientific_status':'GEOMETRIC_FEATURE_ABLATION_GROUP_OOF','summary':rows,'n_geometry_features':len(geom),'audit':aud}

LANES={
 'V31':lane_v31,'V32':lane_v32,'V33':lane_v33,'V34':lane_v34,'V35':lane_v35,'V36':lane_v36,'V37':lane_v37,'V38':lane_v38,'V39':lane_v39,'V40':lane_v40,
 'V41':lane_v41,'V42':lane_v42,'V43':lane_v43,'V44':lane_v44,'V45':lane_v45,'V46':lane_v46,'V47':lane_v47,'V48':lane_v48,'V49':lane_v49,'V50':lane_v50}

def main(payload):
    lane=str(payload['lane']).upper(); d,h,wf,sha=load(payload);t=time.time()
    res=LANES[lane](d,h,wf);res.update({'lane':lane,'campaign':payload.get('campaign'),'data_sha256':sha,'elapsed_seconds':time.time()-t,
      'guards':{**res.get('guards',{}),'random_row_split_used':False,'whole_speed_load_outer_holdout_for_predictive_models':True,'future_target_used':False}})
    save('RESULTS.json',res);print('SOILBIN_LANE_COMPLETE',lane,json.dumps({'status':res['status'],'elapsed':res['elapsed_seconds']}),flush=True);return res

if __name__=='__main__':
    if 'PAYLOAD' in globals(): main(globals()['PAYLOAD'])
    elif len(sys.argv)>1:
        main(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
