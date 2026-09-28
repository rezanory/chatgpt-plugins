"""SoilBin V10-V15: frozen-data parallel research campaign.
All performance estimates are exploratory. V5 and the measured targets are immutable.
"""
from __future__ import annotations
import os, sys, json, math, hashlib, base64, gzip, lzma, struct, itertools, traceback, time
from pathlib import Path
from functools import lru_cache
from importlib.metadata import version, PackageNotFoundError
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
from scipy.linalg import cho_factor, cho_solve
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score

SEED=20260914
EXPECTED_DATA='dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059'
EXPECTED_WAVE_XZ='6ccca93e5e146e67908e37cf07575bdf1ed44b0f3b84fcf0ee3198cd2ffa02c9'
EXPECTED_WAVE_BLOB='786cd72be8a6e37aa0e34cee044f70b047c9a84f5a9230abc9028117db765677'
FROZEN=24.28878365273039
TARGETS=['LC1_peak_magnitude_N_delta','LC5_peak_magnitude_N_delta']
BASE=['Load','Speed','Pass_T','prev_LC1','prev_LC5']
OUT=Path(os.environ.get('SOILBIN_OUTPUT_ROOT','/kaggle/working/SOILBIN_PARALLEL'))
OUT.mkdir(parents=True,exist_ok=True)
START=time.time()
FIT_AUDIT=[]

def sha(data): return hashlib.sha256(data).hexdigest()
def pack(obj): return base64.b64encode(gzip.compress(json.dumps(obj,separators=(',',':'),allow_nan=False).encode(),mtime=0)).decode()
def unpack(s): return json.loads(gzip.decompress(base64.b64decode(s)))
def save(name,obj):
    p=OUT/name
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
    return p

def load_data(payload):
    raw=gzip.decompress(base64.b64decode(payload['model_b64']))
    if sha(raw)!=EXPECTED_DATA: raise ValueError('INPUT_HASH_MISMATCH')
    from io import BytesIO
    d=pd.read_csv(BytesIO(raw))
    if len(d)!=51 or d.Run_ID.nunique()!=51: raise ValueError('RUN_COUNT')
    d['Group']=d.Run_ID.str.extract(r'^(V\dW\d)')[0]
    d['Speed']=d.Speed_level.astype(float)
    d['Load']=d.Weight_level.astype(float)+1.0
    d['Pass_T']=d.Pass_T.astype(int)
    allids={f'V{v}W{w}T{t}' for v in range(1,4) for w in range(1,4) for t in range(1,7)}
    if allids-set(d.Run_ID)!={'V1W1T1','V2W3T2','V3W2T1'}: raise ValueError('MISSING_SLOT_MISMATCH')
    p=d[['Group','Pass_T']+TARGETS].copy(); p.Pass_T+=1
    p=p.rename(columns=dict(zip(TARGETS,['prev_LC1','prev_LC5'])))
    h=d.merge(p,on=['Group','Pass_T'],how='left',validate='one_to_one')
    h=h[h.prev_LC1.notna()&h.prev_LC5.notna()].reset_index(drop=True)
    if len(h)!=41: raise ValueError('TRANSITION_COUNT')
    return d,h

def split_contract(h):
    g=h.Group.to_numpy(); folds=[]
    for outer in sorted(set(g)):
        tr=np.flatnonzero(g!=outer); te=np.flatnonzero(g==outer)
        inner=[]
        for a,b in GroupKFold(n_splits=4).split(tr,groups=g[tr]):
            inner.append({'train':tr[a].tolist(),'valid':tr[b].tolist()})
        folds.append({'outer':outer,'train':tr.tolist(),'test':te.tolist(),'inner':inner})
    obj={'schema':'soilbin.parallel.splits.v1','source_sha256':EXPECTED_DATA,
         'run_ids':h.Run_ID.tolist(),'groups':h.Group.tolist(),'folds':folds,
         'target_cols':TARGETS,'metric':'mean_channels(mean_groups(mean_rows(abs(y-p))))',
         'history_policy':'measured observations strictly before prediction pass; no test-target fitting'}
    digest=sha(json.dumps(obj,sort_keys=True,separators=(',',':')).encode())
    return obj,digest

def weights(g):
    g=np.asarray(g); vals,counts=np.unique(g,return_counts=True)
    w=np.array([1/counts[np.flatnonzero(vals==v)[0]] for v in g])
    return w/w.mean()

def score(y,p,g):
    y=np.asarray(y,float); p=np.asarray(p,float); g=np.asarray(g)
    if y.shape!=p.shape or not np.isfinite(p).all(): raise ValueError('BAD_PREDICTIONS')
    return float(np.mean([np.abs(y[g==v]-p[g==v]).mean() for v in np.unique(g)]))

def metrics(y,p,g):
    gm=[float(np.mean([np.abs(y[g==v,j]-p[g==v,j]).mean() for v in np.unique(g)])) for j in range(2)]
    return {'joint_Group_MAE_N':float(np.mean(gm)),'LC1_Group_MAE_N':gm[0],'LC5_Group_MAE_N':gm[1],
            'sample_MAE_N':np.abs(y-p).mean(axis=0).tolist(),'RMSE_N':np.sqrt(((y-p)**2).mean(axis=0)).tolist(),
            'bias_N':(p-y).mean(axis=0).tolist(),'R2':[float(r2_score(y[:,j],p[:,j])) for j in range(2)],
            'group_errors':{v:float(np.abs(y[g==v]-p[g==v]).mean()) for v in sorted(set(g))}}

class QuantileForest:
    """Conditional response-distribution forest; NOT the median of tree means.
    All training observations are retained (bootstrap=False). The CDF uses
    the average of normalized, group-weighted terminal-node memberships.
    """
    def fit(self,X,y,w):
        self.X=np.asarray(X); self.y=np.asarray(y); self.w=np.asarray(w)
        self.forest=ExtraTreesRegressor(n_estimators=250,max_depth=4,min_samples_leaf=2,max_features=1.0,random_state=SEED,n_jobs=1,bootstrap=False)
        self.forest.fit(X,y,sample_weight=w)
        self.leaves=self.forest.apply(X); return self
    def predict(self,X):
        leaves=self.forest.apply(X); order=np.argsort(self.y); out=[]
        for row in leaves:
            match=(self.leaves==row[None,:])*self.w[:,None]
            den=match.sum(axis=0)
            if np.any(den<=0): raise ValueError('EMPTY_LEAF')
            mass=(match/den).mean(axis=1)
            out.append(self.y[order[np.searchsorted(np.cumsum(mass[order]),.5,side='left')]])
        return np.asarray(out)

class KalmanSoil:
    """Actual probabilistic latent state, fitted by training-sequence likelihood.
    Rank 1: two observation channels share one latent dynamic state.
    Rank 2: coupled process innovations with separate observation noise.
    Filtering is forward only; no smoothing with future observations.
    Latent states are effective residual states, not measured density/moisture.
    """
    def __init__(self,rank=1,logpass=True): self.rank=rank; self.logpass=logpass
    def design(self,z):
        cols=[np.ones(len(z)),(z.Load.to_numpy()-3)/1.,z.Speed.to_numpy()-2]
        if self.logpass: cols.append(np.log(z.Pass_T.to_numpy()))
        return np.column_stack(cols)
    def matrices(self,t):
        rho=float(1/(1+np.exp(-t[0])))*.995
        if self.rank==1:
            H=np.array([[1.],[t[1]]]); Q=np.array([[np.exp(t[2])]])
            R=np.diag(np.exp(t[3:5])); P=np.eye(1)*2
        else:
            H=np.eye(2); q=np.exp(t[1:3]); cor=.9*np.tanh(t[3])
            Q=np.array([[q[0],cor*np.sqrt(q.prod())],[cor*np.sqrt(q.prod()),q[1]]])
            R=np.diag(np.exp(t[4:6])); P=np.eye(2)*2
        return rho,H,Q,R,P
    def step(self,z,P,gap,t):
        rho,H,Q,R,_=self.matrices(t); r=rho**gap
        mult=(1-rho**(2*gap))/max(1-rho*rho,1e-12)
        return z*r, P*r*r+Q*mult
    def likelihood(self,t,sequences):
        rho,H,Q,R,P0=self.matrices(t); loss=0.
        for times,obs in sequences:
            z=np.zeros(self.rank); P=P0.copy(); previous=int(times[0])
            for tm,y in zip(times,obs):
                gap=int(tm)-previous
                if gap: z,P=self.step(z,P,gap,t)
                S=H@P@H.T+R
                try: cf=cho_factor(S,lower=True,check_finite=False)
                except Exception: return 1e12
                err=y-H@z; inv=cho_solve(cf,err,check_finite=False)
                loss+=.5*(2*np.log(np.diag(cf[0])).sum()+err@inv)
                K=cho_solve(cf,H@P,check_finite=False).T
                z=z+K@err; P=P-K@H@P; P=(P+P.T)/2
                previous=int(tm)
        return float(loss+.025*np.square(t).sum())
    def fit(self,d):
        self.scale=np.maximum(d[TARGETS].std(ddof=0).to_numpy(),1.)
        A=self.design(d); yy=d[TARGETS].to_numpy()/self.scale
        w=weights(d.Group.to_numpy()); penalty=np.eye(A.shape[1])*.1; penalty[0,0]=1e-8
        self.beta=np.linalg.solve(A.T@(w[:,None]*A)+penalty,A.T@(w[:,None]*yy))
        seq=[]
        for _,z in d.groupby('Group',sort=True):
            z=z.sort_values('Pass_T'); seq.append((z.Pass_T.to_numpy(),z[TARGETS].to_numpy()/self.scale-self.design(z)@self.beta))
        if self.rank==1: t0=np.array([1.,1.,-2.,-2.,-2.]); bounds=[(-5,5),(-3,3),(-7,2),(-7,2),(-7,2)]
        else: t0=np.array([1.,-2.,-2.,.5,-2.,-2.]); bounds=[(-5,5),(-7,2),(-7,2),(-2,2),(-7,2),(-7,2)]
        opt=minimize(self.likelihood,t0,args=(seq,),method='L-BFGS-B',bounds=bounds,options={'maxiter':90,'ftol':1e-7})
        if not np.isfinite(opt.fun): raise ValueError('STATE_OPTIMIZATION_NONFINITE')
        self.theta=opt.x; self.fit_status={'converged':bool(opt.success),'message':str(opt.message),'objective':float(opt.fun)}
        return self
    def forecast(self,sequence,targets):
        seq=sequence.sort_values('Pass_T'); wanted=set(targets); out={}
        _,H,Q,R,P=self.matrices(self.theta); z=np.zeros(self.rank); previous=int(seq.Pass_T.iloc[0])
        for _,row in seq.iterrows():
            tm=int(row.Pass_T); gap=tm-previous
            if gap: z,P=self.step(z,P,gap,self.theta)
            mean=(self.design(pd.DataFrame([row]))@self.beta)[0]
            if row.Run_ID in wanted:
                # Record prediction BEFORE assimilating this pass measurement.
                out[row.Run_ID]=(mean+H@z)*self.scale
            y=row[TARGETS].to_numpy(float)/self.scale-mean
            S=H@P@H.T+R; K=np.linalg.solve(S,H@P).T
            z=z+K@(y-H@z); P=P-K@H@P; P=(P+P.T)/2; previous=tm
        return np.vstack([out[k] for k in targets])

class RelaxationPrior:
    """Physics-inspired relaxation proxy; not a calibrated soil digital twin.
    Predicted next force = measured prior force * relaxation-profile ratio.
    Unknown contact area, moisture and modulus are NOT invented observations.
    """
    def __init__(self,shape='exp',discrepancy=False): self.shape=shape; self.discrepancy=discrepancy
    def ratio(self,x,t):
        k=np.exp(t[0]+t[1]*(x[:,0]-3)+t[2]*(x[:,1]-2)); floor=1/(1+np.exp(-t[3])); p=x[:,2]
        if self.shape=='exp':
            f=lambda n: floor+(1-floor)*np.exp(-k*(n-1))
        else: f=lambda n: floor+(1-floor)*np.power(np.maximum(n,1),-k)
        return f(p)/np.maximum(f(p-1),1e-9)
    def fit(self,X,y,g):
        self.theta=[]; self.gp=[]; self.scaler=StandardScaler().fit(X); w=weights(g)
        for j in range(2):
            s=max(float(np.std(y[:,j])),1.)
            def residual(t):
                e=(X[:,3+j]*self.ratio(X,t)-y[:,j])*np.sqrt(w)/s
                return np.r_[e,.1*t]
            fit=least_squares(residual,[-1.,0.,0.,0.],bounds=([-5,-2,-2,-5],[2,2,2,5]),loss='soft_l1',max_nfev=150)
            self.theta.append(fit.x)
            if self.discrepancy:
                resid=y[:,j]-X[:,3+j]*self.ratio(X,fit.x)
                kernel=ConstantKernel(1.,(.01,100.))*Matern(length_scale=np.ones(5),length_scale_bounds=(.1,100.),nu=1.5)+WhiteKernel(.2,(.01,10.))
                gp=GaussianProcessRegressor(kernel=kernel,normalize_y=True,alpha=1e-6,random_state=SEED,n_restarts_optimizer=0)
                gp.fit(self.scaler.transform(X),resid); self.gp.append(gp)
        return self
    def predict(self,X):
        p=np.column_stack([X[:,3+j]*self.ratio(X,self.theta[j]) for j in range(2)])
        if self.discrepancy:
            for j in range(2): p[:,j]+=self.gp[j].predict(self.scaler.transform(X))
        return p

class Engine:
    def __init__(self,d,h):
        self.d=d; self.h=h; self.g=h.Group.to_numpy(); self.y=h[TARGETS].to_numpy(float)
        self.x=h[BASE].to_numpy(float); self.previous=h[['prev_LC1','prev_LC5']].to_numpy(float)
        self.cache={}; self.selections=[]; self.warn=[]
    def predict(self,model,tr,te):
        tr=np.asarray(tr,int); te=np.asarray(te,int)
        a=set(self.g[tr]); b=set(self.g[te])
        if a&b: raise ValueError('GROUP_LEAKAGE')
        key=(model,tuple(tr.tolist()),tuple(te.tolist()))
        if key in self.cache: return self.cache[key].copy()
        FIT_AUDIT.append({'model':model,'train_groups':sorted(a),'test_groups':sorted(b)})
        X=self.x[tr]; Xt=self.x[te]; yd=self.y[tr]-self.previous[tr]; yt=self.y[tr]
        w=weights(self.g[tr]); out=None
        if model=='PERSISTENCE': out=self.previous[te].copy()
        elif model=='V5':
            choices=['RF_DEEP','RF_SHALLOW','ET_DEEP','ET_SHALLOW']; best=None
            for c in choices:
                po=np.full((len(tr),2),np.nan)
                for ai,bi in GroupKFold(n_splits=min(4,len(a))).split(tr,groups=self.g[tr]):
                    po[bi]=self.predict(c,tr[ai],tr[bi])
                s=score(yt,po,self.g[tr])
                if best is None or s<best[0]-1e-12: best=(s,c)
            self.selections.append({'train_groups':sorted(a),'test_groups':sorted(b),'selected':best[1],'inner_mae':best[0]})
            out=self.predict(best[1],tr,te)
        elif model.startswith(('RF_','ET_','ABS_','QRF_')):
            out=np.zeros((len(te),2))
            for j in range(2):
                if model.startswith('QRF_'):
                    m=QuantileForest().fit(X,yd[:,j],w); pred=m.predict(Xt)
                else:
                    is_et=model.startswith('ET_') or model=='ABS_ET_WEIGHTED'
                    cls=ExtraTreesRegressor if is_et else RandomForestRegressor
                    shallow=('SHALLOW' in model) or model.startswith('ABS_')
                    criterion='absolute_error' if model.startswith('ABS_') else 'squared_error'
                    m=cls(n_estimators=250,max_depth=4 if shallow else None,min_samples_leaf=2 if shallow else 1,
                          criterion=criterion,random_state=SEED,n_jobs=1,max_features=1.)
                    m.fit(X,yd[:,j],sample_weight=w if 'WEIGHTED' in model else None)
                    pred=m.predict(Xt)
                out[:,j]=pred+self.previous[te,j]
        elif model.startswith('TABPFN_'):
            from tabpfn import TabPFNRegressor
            out=np.zeros((len(te),2)); delta='DELTA' in model
            for j in range(2):
                m=TabPFNRegressor(device='cpu',n_estimators=2,random_state=SEED)
                m.fit(X,yd[:,j] if delta else yt[:,j])
                pred=m.predict(Xt,output_type='median')
                out[:,j]=np.asarray(pred).ravel()+(self.previous[te,j] if delta else 0.)
        elif model.startswith('STATE_'):
            rank=1 if 'RANK1' in model else 2
            md=KalmanSoil(rank=rank,logpass=('LOG' in model)).fit(self.d[self.d.Group.isin(a)])
            if not md.fit_status['converged']: self.warn.append({'model':model,'training_groups':sorted(a),**md.fit_status})
            out=np.zeros((len(te),2))
            for group in sorted(b):
                local=np.flatnonzero(self.g[te]==group); ids=self.h.iloc[te[local]].Run_ID.tolist()
                out[local]=md.forecast(self.d[self.d.Group==group],ids)
        elif model.startswith('PHYS_'):
            m=RelaxationPrior(shape='power' if 'POWER' in model else 'exp',discrepancy=model.endswith('_GP')).fit(X,yt,self.g[tr])
            out=m.predict(Xt)
        else: raise ValueError('UNKNOWN_MODEL:'+model)
        out=np.asarray(out,float)
        if out.shape!=(len(te),2) or not np.isfinite(out).all(): raise ValueError('INVALID_PREDICTION:'+model)
        self.cache[key]=out.copy(); return out

LANE_MODELS={
 'V11':['V5','RF_SHALLOW','RF_SHALLOW_WEIGHTED','ET_SHALLOW_WEIGHTED','ABS_RF_UNWEIGHTED','ABS_RF_WEIGHTED','ABS_ET_WEIGHTED','QRF_MEDIAN_WEIGHTED'],
 'V12':['TABPFN_DIRECT','TABPFN_DELTA'],
 'V13':['STATE_RANK1','STATE_RANK1_LOG','STATE_RANK2_LOG'],
 'V14':['PHYS_EXP','PHYS_POWER','PHYS_EXP_GP','PHYS_POWER_GP'],
}

def audit_v10(payload,d,h,contract,digest):
    report={'lane':'V10','scope':'calibrated, baseline-corrected waveform snapshot plus feature table; not original ADC voltage',
            'target_modified':False,'dropped_runs':[],'valid_runs':51,'transitions':41,'groups':9,'split_sha256':digest,
            'calibration_certificate_available':False,'sensor_area_available':False,'independent_repeat_noise_floor_identifiable':False}
    checks=[]
    if 'wave_b64' not in payload: report['raw_waveform_status']='BLOCKED_MISSING_SNAPSHOT'
    else:
        xz=base64.b64decode(payload['wave_b64']); blob=lzma.decompress(xz)
        if sha(xz)!=EXPECTED_WAVE_XZ or sha(blob)!=EXPECTED_WAVE_BLOB: raise ValueError('WAVE_HASH')
        nhead=struct.unpack('<I',blob[:4])[0]; meta=json.loads(blob[4:4+nhead]); n=meta['n']; off=4+nhead
        run=np.frombuffer(blob,dtype='u1',count=n,offset=off); off+=n
        t=np.frombuffer(blob,dtype='<u4',count=n,offset=off); off+=4*n
        f1=np.frombuffer(blob,dtype='<f8',count=n,offset=off); off+=8*n
        f5=np.frombuffer(blob,dtype='<f8',count=n,offset=off); off+=8*n
        if off!=len(blob) or n!=80958: raise ValueError('WAVE_FORMAT')
        for i,rid in enumerate(meta['runs']):
            mask=run==i; tt=t[mask].astype(float); rr=d[d.Run_ID==rid].iloc[0]
            for ch,signal in [('LC1',f1[mask]),('LC5',f5[mask])]:
                edge=max(10,min(200,len(signal)//8)); bg=np.r_[signal[:edge],signal[-edge:]]
                sigma=1.4826*np.median(np.abs(bg-np.median(bg))); peak=float(np.max(np.abs(signal)))
                ratio=float(rr[ch+'_peak_magnitude_N_delta']/max(abs(rr[ch+'_peak_delta_raw_kg']),1e-12))
                checks.append({'Run_ID':rid,'channel':ch,'n':len(signal),'time_strictly_increasing':bool(np.all(np.diff(tt)>0)),
                 'finite':bool(np.isfinite(signal).all()),'median_dt_ms':float(np.median(np.diff(tt))),
                 'recomputed_peak_N':peak,'frozen_peak_N':float(rr[ch+'_peak_magnitude_N_delta']),
                 'peak_abs_difference_N':abs(peak-float(rr[ch+'_peak_magnitude_N_delta'])),
                 'edge_mad_sigma_N':float(sigma),'edge_drift_N':float(np.median(signal[-edge:])-np.median(signal[:edge])),
                 'peak_to_edge_noise':None if sigma<=0 else peak/float(sigma),
                 'max_magnitude_tie_count':int(np.sum(np.abs(signal)==peak)),
                 'N_per_raw_unit':ratio,'hardware_clipping_diagnosed':False})
        pd.DataFrame(checks).to_csv(OUT/'waveform_audit.csv',index=False)
        report['raw_waveform_status']='AUDITED_102_CHANNEL_TRACES'
        report['trace_count']=len(checks); report['waveform_samples']=n
        report['max_peak_reproduction_difference_N']=max(r['peak_abs_difference_N'] for r in checks)
        report['nonfinite_trace_count']=sum(not r['finite'] for r in checks)
        report['nonmonotonic_timestamp_traces']=sum(not r['time_strictly_increasing'] for r in checks)
        report['physical_parameters_not_inferred_from_other_papers']=True
    report['signal_checks']=checks
    report['limitations']=['Corrected traces cannot establish full ADC saturation or original zero calibration.','Noise during no-load edges is not the independent experimental reproducibility floor.','No new sensor values are fabricated.']
    return report

def paired_report(y,p,baseline,g):
    diff=np.array([np.abs(y[g==v]-baseline[g==v]).mean()-np.abs(y[g==v]-p[g==v]).mean() for v in sorted(set(g))])
    perm=(np.array(list(itertools.product([-1,1],repeat=len(diff))))*diff).mean(axis=1)
    rng=np.random.default_rng(20260928); boot=diff[rng.integers(0,len(diff),size=(10000,len(diff)))].mean(axis=1)
    return {'delta_MAE_N':float(diff.mean()),'groups_improved':int((diff>1e-10).sum()),'groups_tied':int((np.abs(diff)<=1e-10).sum()),
            'signflip_one_sided_p':float(np.mean(perm>=diff.mean()-1e-12)),
            'bootstrap_95CI_N':np.quantile(boot,[.025,.975]).tolist(),
            'inference_status':'exploratory; repeated development on same groups; no new independent test'}

def run_predictive(lane,payload,d,h,contract,digest):
    engine=Engine(d,h); y=engine.y; g=engine.g
    models=LANE_MODELS[lane]; bank={'schema':'soilbin.parallel.bank.v1','lane':lane,'split_sha256':digest,'source_sha256':EXPECTED_DATA,
                                 'run_ids':h.Run_ID.tolist(),'groups':h.Group.tolist(),'models':{},'folds':contract['folds']}
    rows=[]
    for k,model in enumerate(models):
        print(f'CGP_PHASE:{lane}_MODEL_{k+1}_{model}',flush=True)
        p=np.full_like(y,np.nan); inner={}
        for fold in contract['folds']:
            tr=np.array(fold['train']); te=np.array(fold['test']); p[te]=engine.predict(model,tr,te)
            # Every meta-training prediction excludes its validation group from
            # both model fitting AND any hyperparameter/model-family selection.
            pin=np.full_like(y,np.nan)
            for inn in fold['inner']:
                it=np.array(inn['train']); iv=np.array(inn['valid']); pin[iv]=engine.predict(model,it,iv)
            if not np.isfinite(pin[tr]).all(): raise ValueError('INCOMPLETE_META_OOF')
            inner[fold['outer']]={'indices':tr.tolist(),'pred':pin[tr].tolist()}
        bank['models'][model]={'outer_pred':p.tolist(),'inner_oof':inner}
        rows.append({'model':model,**metrics(y,p,g)})
        save('progress.json',{'lane':lane,'completed_models':k+1,'total_models':len(models)})
        # Durable incremental evidence survives interruption; no partial model is scored.
        save('BANK_PARTIAL.json',{'complete':False,'split_sha256':digest,'payload_gzip_b64':pack(bank)})
    if lane=='V11':
        reproduced=next(r for r in rows if r['model']=='V5')['joint_Group_MAE_N']
        if abs(reproduced-FROZEN)>1e-6: raise ValueError(f'V5_REPRODUCTION_FAILED:{reproduced}')
    save('selection_audit.json',engine.selections); save('fit_warnings.json',engine.warn)
    wrapper={'complete':True,'split_sha256':digest,'source_sha256':EXPECTED_DATA,'payload_gzip_b64':pack(bank)}
    bp=save('BANK.json',wrapper)
    if bp.stat().st_size>262144: raise ValueError('BANK_EXCEEDS_BROKER_LIMIT')
    result={'lane':lane,'models_completed':len(models),'models_expected':len(models),'summary':rows,
            'split_sha256':digest,'prediction_bank_sha256':sha(bp.read_bytes()),'warnings_count':len(engine.warn)}
    if lane=='V14':
        result['validated_multifidelity_simulator']={'status':'BLOCKED_MISSING_PHYSICAL_METADATA',
            'missing':['effective_sensor_area_or_direct_stress_calibration','contact_geometry_or_tire_geometry','soil_constitutive_parameters','moisture_and_initial_density_records'],
            'executed_instead':'four explicitly physics-inspired relaxation/discrepancy models; no verified synthetic physical measurements',
            'not_equivalent_to_validated_multifidelity':True}
    return result

def run_ensemble(payload,d,h,contract,digest):
    y=h[TARGETS].to_numpy(float); g=h.Group.to_numpy(); candidates={}; source_status={}
    for lane,wrapper in payload['banks'].items():
        b=unpack(wrapper['payload_gzip_b64'])
        if not wrapper['complete'] or b['split_sha256']!=digest or b['run_ids']!=h.Run_ID.tolist(): raise ValueError('INCOMPATIBLE_BANK:'+lane)
        candidates.update(b['models']); source_status[lane]='COMPLETE_BANK'
    if 'V5' not in candidates: raise ValueError('NO_VALID_BASELINE')
    # One prospectively fixed representative from each independent family.
    designated=['V5','QRF_MEDIAN_WEIGHTED','TABPFN_DELTA','STATE_RANK1_LOG','PHYS_EXP_GP','PERSISTENCE']
    for n in designated:
        if n=='PERSISTENCE':
            prev=h[['prev_LC1','prev_LC5']].to_numpy(float)
            candidates[n]={'outer_pred':prev.tolist(),'inner_oof':{f['outer']:{'indices':f['train'],'pred':prev[f['train']].tolist()} for f in contract['folds']}}
    available=[n for n in designated if n in candidates]
    missing=[n for n in designated if n not in candidates]
    specs={'FIXED_EQUAL':('equal',available),'ANCHORED_BLEND':('blend',available)}
    for name in available:
        if name!='V5': specs['REMOVE_'+name]=('blend',[n for n in available if n!=name])
    allpred={}; weight_rows=[]
    for method,(kind,members) in specs.items():
        pred=np.full_like(y,np.nan)
        for f in contract['folds']:
            tr=np.array(f['train']); te=np.array(f['test'])
            if any(candidates[n]['inner_oof'][f['outer']]['indices']!=f['train'] for n in members): raise ValueError('META_ORDER')
            pin=np.stack([np.array(candidates[n]['inner_oof'][f['outer']]['pred']) for n in members],axis=2)
            pout=np.stack([np.array(candidates[n]['outer_pred'])[te] for n in members],axis=2)
            if kind=='equal': w=np.ones(len(members))/len(members)
            else:
                # Capacity-constrained pool: V5 >= 0.5, each other expert <=0.25.
                # The finite weight grid and penalties are frozen before any test result.
                K=len(members)-1; best=None
                for tail in itertools.product((0.,.125,.25),repeat=K):
                    if sum(tail)>.5+1e-12: continue
                    ww=np.array([1-sum(tail),*tail]); pp=np.tensordot(pin,ww,axes=(2,0))
                    objective=score(y[tr],pp,g[tr])+.10*np.count_nonzero(ww[1:])
                    if best is None or objective<best[0]-1e-12: best=(objective,ww)
                w=best[1]
            pred[te]=np.tensordot(pout,w,axes=(2,0))
            weight_rows.append({'method':method,'outer_group':f['outer'],'weights':dict(zip(members,w.tolist()))})
        allpred[method]=pred
    baseline=np.array(candidates['V5']['outer_pred'])
    if abs(score(y,baseline,g)-FROZEN)>1e-6: raise ValueError('BASELINE_GATE')
    summary=[{'model':'V5',**metrics(y,baseline,g)}]
    for model,p in allpred.items(): summary.append({'model':model,**metrics(y,p,g),'paired_vs_v5':paired_report(y,p,baseline,g)})
    tests=[x for x in summary if 'paired_vs_v5' in x]
    order=sorted(range(len(tests)),key=lambda i:tests[i]['paired_vs_v5']['signflip_one_sided_p'])
    running=0.
    for rank,idx in enumerate(order):
        running=max(running,min(1.,(len(tests)-rank)*tests[idx]['paired_vs_v5']['signflip_one_sided_p']))
        tests[idx]['paired_vs_v5']['holm_adjusted_p']=running
        tests[idx]['paired_vs_v5']['comparison_family_size']=len(tests)
    full=allpred['ANCHORED_BLEND']; effects=[]
    for name in available:
        if name=='V5': continue
        reduced=allpred['REMOVE_'+name]
        effects.append({'removed':name,'refit_weights':True,'full_MAE_N':score(y,full,g),'without_MAE_N':score(y,reduced,g),
                        'removal_harm_N':score(y,reduced,g)-score(y,full,g)})
    save('ensemble_weights.json',weight_rows); save('member_removal_ablation.json',effects)
    return {'lane':'V15','summary':summary,'member_contributions':effects,'designated_members':designated,'available_members':available,
            'missing_members':missing,'complete_planned_ensemble':not missing,'source_status':source_status,
            'unavailable_lanes':payload.get('unavailable_lanes',{}),'split_sha256':digest,
            'members_selected_using_outer_results':False,'target_specific_weights':False,
            'description':'predeclared fixed equal and anchored low-capacity convex blend, plus refitted member-removal ablation'}

def main(payload):
    lane=payload['lane']; result={}; status='COMPLETE'; reason=None
    d,h=load_data(payload); contract,digest=split_contract(h)
    save('SPLIT_CONTRACT.json',contract)
    if lane=='V10': result=audit_v10(payload,d,h,contract,digest)
    elif lane=='V15': result=run_ensemble(payload,d,h,contract,digest)
    else:
        result=run_predictive(lane,payload,d,h,contract,digest)
        if lane=='V14': status='COMPLETE_PROXY_MODELS_PHYSICAL_MULTIFIDELITY_BLOCKED'
    result.update({'schema':'soilbin.parallel.result.v1','status':status,'campaign':payload['campaign'],'lane':lane,
        'source_sha256':EXPECTED_DATA,'elapsed_seconds':time.time()-START,'post_lock_exploratory':True,'does_not_supersede_v5':True,
        'guardrails':{'random_split_used':False,'targets_modified':False,'current_pass_waveforms_used_for_prediction':False,
                     'inner_prediction_model_selection_excludes_validation_groups':True,'independent_new_experiment':False},
        'environment':{p:version(p) for p in ('numpy','pandas','scipy','scikit-learn')},
        'fit_calls':len(FIT_AUDIT)})
    save('fit_isolation_audit.json',FIT_AUDIT); save('RESULTS.json',result)
    import shutil
    shutil.make_archive(str(OUT.parent/(lane+'_soilbin_parallel')),'zip',OUT)
    print('SOILBIN_PARALLEL_COMPLETE',json.dumps({'lane':lane,'status':status,'split_sha256':digest,'summary':result.get('summary',[])}),flush=True)

if __name__=='__main__':
    try:
        main(PAYLOAD)
    except Exception as exc:
        save('RESULTS.json',{'schema':'soilbin.parallel.result.v1','lane':globals().get('PAYLOAD',{}).get('lane'),
             'campaign':globals().get('PAYLOAD',{}).get('campaign'),'status':'FAILED','error_type':type(exc).__name__,
             'error':str(exc)[:1500],'traceback':traceback.format_exc()[-6000:],'elapsed_seconds':time.time()-START,
             'source_sha256':EXPECTED_DATA,'models_complete':False})
        print('SOILBIN_PARALLEL_FAILED',type(exc).__name__,str(exc)[:800],flush=True)
        raise
