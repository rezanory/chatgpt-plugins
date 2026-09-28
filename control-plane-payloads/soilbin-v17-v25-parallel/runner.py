"""SoilBin V17-V25 parallel adaptive-state research campaign.
All runs are post-lock exploratory. Frozen data and measured targets are immutable.
"""
from __future__ import annotations
import base64,gzip,hashlib,itertools,json,math,os,random,time,traceback
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import binomtest,spearmanr
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,brier_score_loss
from sklearn.model_selection import GroupKFold

SEED=20260928
random.seed(SEED); np.random.seed(SEED)
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[k]="1"
EXPECTED_DATA="dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"
TARGETS=["LC1_peak_magnitude_N_delta","LC5_peak_magnitude_N_delta"]
FORCE_BASE=["Load","Speed","Pass_T","prev_LC1","prev_LC5"]
V11_LOCK=23.86044078619803
V5_LOCK=24.28878365273039
EPS=1e-9
EXPERTS=["QLOG","REBOUND","SATEXP","PERSIST"]
OUT=Path(os.environ.get("SOILBIN_OUTPUT_ROOT","/kaggle/working/SOILBIN_V17_V25"))
OUT.mkdir(parents=True,exist_ok=True)
START=time.time()

def phase(x): print("CGP_PHASE:"+x,flush=True)
def save(name,obj):
    p=OUT/name
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if hasattr(x,"item") else str(x)),encoding="utf-8")
    return p
def sha(b): return hashlib.sha256(b).hexdigest()
def unpack_b64_gz(s):
    return gzip.decompress(base64.b64decode(s.strip()))
def group_weights(g):
    g=np.asarray(g); u,c=np.unique(g,return_counts=True); m=dict(zip(u,c))
    w=np.array([1/m[x] for x in g],float); return w/w.mean()
def gmae1(y,p,g):
    y=np.asarray(y,float);p=np.asarray(p,float);g=np.asarray(g)
    return float(np.mean([np.mean(np.abs(y[g==x]-p[g==x])) for x in np.unique(g)]))
def force_metrics(y,p,g):
    y=np.asarray(y,float);p=np.asarray(p,float);g=np.asarray(g)
    ch=[float(np.mean([np.abs(y[g==x,j]-p[g==x,j]).mean() for x in np.unique(g)])) for j in range(2)]
    return {"joint_Group_MAE_N":float(np.mean(ch)),"LC1_Group_MAE_N":ch[0],"LC5_Group_MAE_N":ch[1],
            "group_errors":{str(x):float(np.abs(y[g==x]-p[g==x]).mean()) for x in sorted(set(g))}}
def paired_group_report(y,p,b,g):
    vals=[]
    for x in sorted(set(g)):
        m=g==x; vals.append(float(np.abs(y[m]-b[m]).mean()-np.abs(y[m]-p[m]).mean()))
    d=np.array(vals); signs=np.asarray(list(itertools.product([-1,1],repeat=len(d))),float)
    perm=(signs*d).mean(axis=1); rng=np.random.default_rng(SEED)
    boot=d[rng.integers(0,len(d),size=(20000,len(d)))].mean(axis=1)
    return {"mean_improvement_N":float(d.mean()),"groups_improved":int((d>0).sum()),
            "exact_signflip_one_sided_p":float(np.mean(perm>=d.mean()-1e-12)),
            "bootstrap_95CI_N":[float(x) for x in np.quantile(boot,[.025,.975])]}
def load_data(payload):
    raw=unpack_b64_gz(payload["model_b64"])
    if sha(raw)!=EXPECTED_DATA: raise RuntimeError("SOURCE_FINGERPRINT_MISMATCH:"+sha(raw))
    d=pd.read_csv(BytesIO(raw))
    if len(d)!=51 or d.Run_ID.nunique()!=51: raise RuntimeError("EXPECTED_51_RUNS")
    d["Group"]=d.Run_ID.str.extract(r"^(V\dW\d)")[0]
    d["Speed"]=d.Speed_level.astype(float); d["Load"]=d.Weight_level.astype(float)+1.; d["Pass_T"]=d.Pass_T.astype(int)
    # force-transition table exactly matching V11
    p=d[["Group","Pass_T"]+TARGETS].copy();p.Pass_T+=1
    p=p.rename(columns={TARGETS[0]:"prev_LC1",TARGETS[1]:"prev_LC5"})
    h=d.merge(p,on=["Group","Pass_T"],how="left",validate="one_to_one")
    h=h[h.prev_LC1.notna()&h.prev_LC5.notna()].reset_index(drop=True)
    if len(h)!=41: raise RuntimeError("EXPECTED_41_FORCE_TRANSITIONS")
    # physical transition magnitudes
    tr=[]
    for g,z in d.groupby("Group"):
        z=z.sort_values("Pass_T"); by={int(r.Pass_T):r for _,r in z.iterrows()}
        for t in range(2,7):
            if t in by and t-1 in by:
                a,b=by[t-1],by[t]
                d1=float(b[TARGETS[0]]-a[TARGETS[0]]);d5=float(b[TARGETS[1]]-a[TARGETS[1]])
                tr.append({"Group":g,"Pass":t,"Speed":float(b.Speed),"Load":float(b.Load),
                           "D_LC1":abs(d1),"D_LC5":abs(d5),"D_joint":.5*(abs(d1)+abs(d5)),
                           "S_LC1":d1,"S_LC5":d5})
    tr=pd.DataFrame(tr).sort_values(["Group","Pass"]).reset_index(drop=True)
    if len(tr)!=41: raise RuntimeError("EXPECTED_41_MAGNITUDES")
    dyn=[]
    for g,z in tr.groupby("Group"):
        z=z.sort_values("Pass").reset_index(drop=True)
        for j in range(1,len(z)):
            a,r=z.iloc[j-1],z.iloc[j]
            if int(r.Pass)-int(a.Pass)!=1: continue
            pre=z.iloc[j-2] if j>=2 and int(a.Pass)-int(z.iloc[j-2].Pass)==1 else None
            row={"Group":g,"BasePass":int(a.Pass),"TargetPass":int(r.Pass),
                 "Speed":float(r.Speed),"Load":float(r.Load)}
            for m in ("D_joint","D_LC1","D_LC5"):
                row["Base_"+m]=float(a[m]);row[m]=float(r[m])
                row["PrevRatio_"+m]=1.0 if pre is None else float(a[m]/max(float(pre[m]),EPS))
            row["HasPrevPrev"]=0 if pre is None else 1
            dyn.append(row)
    dyn=pd.DataFrame(dyn).sort_values(["Group","TargetPass"]).reset_index(drop=True)
    if len(dyn)!=32: raise RuntimeError("EXPECTED_32_ONE_STEP_DYNAMICS")
    return d,h,tr,dyn

# ---------- functional experts ----------
def _ratio(expert,p,t,a,bp=None):
    t=np.asarray(t,float);a=np.asarray(a,float)
    if expert=="PERSIST": return np.ones_like(t)
    if expert=="QLOG":
        b1,b2=p; return np.exp(b1*(t-a)+b2*(t*t-a*a))
    if expert=="SATEXP":
        q,k=p
        ft=q+(1-q)*np.exp(-k*(t-1));fa=q+(1-q)*np.exp(-k*(a-1))
        return ft/np.maximum(fa,1e-12)
    if expert=="REBOUND":
        b1,b2=p;bp=float(bp)
        before=np.minimum(t,bp)-np.minimum(a,bp)
        after=np.maximum(0,t-bp)-np.maximum(0,a-bp)
        return np.exp(b1*before+b2*after)
    raise KeyError(expert)
def _spec(expert):
    if expert=="QLOG": return np.array([-.2,.02]),np.array([-5.,-2.]),np.array([5.,2.])
    if expert=="SATEXP": return np.array([.2,.5]),np.array([.001,0.]),np.array([.999,20.])
    if expert=="REBOUND": return np.array([-.3,.1]),np.array([-5.,0.]),np.array([0.,5.])
    if expert=="PERSIST": return np.array([]),np.array([]),np.array([])
    raise KeyError(expert)
def fit_expert(df,expert,metric,bp=None):
    if expert=="PERSIST": return {"p":np.array([]),"bp":None}
    x0,lo,hi=_spec(expert); y=np.maximum(df[metric].to_numpy(float),1e-8)
    base=np.maximum(df["Base_"+metric].to_numpy(float),1e-8)
    ratio=y/base;t=df.TargetPass.to_numpy(float);a=df.BasePass.to_numpy(float);w=np.sqrt(group_weights(df.Group))
    def res(p):
        pred=np.maximum(_ratio(expert,p,t,a,bp),1e-8)
        return w*(np.log(pred)-np.log(ratio))
    opt=least_squares(res,x0,bounds=(lo,hi),loss="soft_l1",f_scale=.35,max_nfev=2500)
    return {"p":opt.x,"bp":bp}
def choose_bp(df,metric):
    best=None
    for bp in (3,4,5):
        fit=fit_expert(df,"REBOUND",metric,bp)
        pred=predict_expert(df,"REBOUND",metric,fit)
        sc=gmae1(df[metric],pred,df.Group)
        if best is None or (sc,bp)<best[:2]: best=(sc,bp,fit)
    return best[2]
def predict_expert(df,expert,metric,fit):
    if expert=="PERSIST": return df["Base_"+metric].to_numpy(float)
    rr=_ratio(expert,fit["p"],df.TargetPass,df.BasePass,fit.get("bp"))
    return df["Base_"+metric].to_numpy(float)*rr
def fit_predict(df_train,df_test,expert,metric):
    fit=choose_bp(df_train,metric) if expert=="REBOUND" else fit_expert(df_train,expert,metric)
    return predict_expert(df_test,expert,metric,fit),fit
def expert_bundle(data,outer,metric):
    train=data[data.Group!=outer];test=data[data.Group==outer]
    Pte=np.zeros((len(test),len(EXPERTS)));Pin=np.full((len(train),len(EXPERTS)),np.nan)
    train=train.copy();test=test.copy()
    for k,e in enumerate(EXPERTS):
        Pte[:,k],_=fit_predict(train,test,e,metric)
        for vg in sorted(train.Group.unique()):
            tr=train[train.Group!=vg];va=train[train.Group==vg]
            pred,_=fit_predict(tr,va,e,metric)
            Pin[train.Group.to_numpy()==vg,k]=pred
    if not np.isfinite(Pin).all(): raise RuntimeError("INCOMPLETE_INNER_EXPERT_OOF")
    return train,test,Pin,Pte

def comps(n,k,prefix=()):
    if k==1:
        yield prefix+(n,);return
    for i in range(n+1): yield from comps(n-i,k-1,prefix+(i,))
def opt_weights(y,P,g,step=.05):
    n=int(round(1/step));best=None
    for c in comps(n,P.shape[1]):
        w=np.asarray(c,float)/n;pred=P@w;sc=gmae1(y,pred,g)
        if best is None or sc<best[0]-1e-12: best=(sc,w)
    return best[1],float(best[0])
def state_X(df,metric="D_joint"):
    base=np.maximum(df["Base_"+metric].to_numpy(float),EPS)
    pr=np.maximum(df["PrevRatio_"+metric].to_numpy(float),EPS)
    return np.column_stack([
        df.TargetPass.to_numpy(float),df.BasePass.to_numpy(float),
        np.log1p(base),np.log(pr),df.Speed.to_numpy(float),df.Load.to_numpy(float),
        df.HasPrevPrev.to_numpy(float)
    ])
def arm_weights(train,test,Pin,Pte,arm,metric="D_joint"):
    K=len(EXPERTS);y=train[metric].to_numpy(float);g=train.Group.to_numpy()
    if arm=="A_QLOG":
        w=np.zeros(K);w[EXPERTS.index("QLOG")]=1;return np.tile(w,(len(test),1)),np.tile(w,(len(train),1)),{"arm":arm}
    if arm=="B_EQUAL":
        w=np.ones(K)/K;return np.tile(w,(len(test),1)),np.tile(w,(len(train),1)),{"arm":arm}
    global_w,sc=opt_weights(y,Pin,g,.05)
    if arm=="C_CONVEX":
        return np.tile(global_w,(len(test),1)),np.tile(global_w,(len(train),1)),{"arm":arm,"global_w":global_w.tolist(),"inner_mae":sc}
    if arm=="D_PASS_GATE":
        maps={}
        for t in sorted(train.TargetPass.unique()):
            m=train.TargetPass.to_numpy()==t
            maps[int(t)]=opt_weights(y[m],Pin[m],g[m],.1)[0] if m.sum()>=3 else global_w
        Wte=np.vstack([maps.get(int(t),global_w) for t in test.TargetPass])
        Wtr=np.vstack([maps.get(int(t),global_w) for t in train.TargetPass])
        return Wte,Wtr,{"arm":arm,"pass_w":{str(k):v.tolist() for k,v in maps.items()}}
    if arm=="E_STATE_GATE":
        labels=np.argmin(np.abs(Pin-y[:,None]),axis=1)
        uniq=np.unique(labels)
        if len(uniq)==1:
            Wte=np.zeros((len(test),K));Wte[:,uniq[0]]=1
            Wtr=np.zeros((len(train),K));Wtr[:,uniq[0]]=1
        else:
            clf=RandomForestClassifier(n_estimators=200,max_depth=2,min_samples_leaf=2,
                class_weight="balanced",random_state=SEED,n_jobs=1)
            clf.fit(state_X(train,metric),labels)
            def probs(df):
                q=np.zeros((len(df),K));pr=clf.predict_proba(state_X(df,metric))
                for j,c in enumerate(clf.classes_): q[:,int(c)]=pr[:,j]
                q+=.02;return q/q.sum(axis=1,keepdims=True)
            Wte=probs(test);Wtr=probs(train)
        return Wte,Wtr,{"arm":arm,"label_counts":{EXPERTS[int(k)]:int(v) for k,v in zip(*np.unique(labels,return_counts=True))}}
    raise KeyError(arm)

def normalize_rows(W):
    W=np.maximum(np.asarray(W,float),1e-12);return W/W.sum(axis=1,keepdims=True)

# ---------- adaptive reward/penalty ----------
def update_factor(adv,eta,shape):
    a=np.asarray(adv,float)
    if shape=="EXP": return np.exp(np.clip(eta*a,-5,5))
    if shape=="POWER":
        return np.exp(np.sign(a)*eta*np.log1p(np.abs(a)))
    if shape=="HYPERBOLIC":
        return np.where(a>=0,1+eta*a,1/(1+eta*(-a)))
    if shape=="TANH":
        return np.clip(1+np.tanh(eta*a),.05,1.95)
    if shape=="LINEAR":
        return np.clip(1+eta*a,.05,3.)
    raise KeyError(shape)
def eta_factor(row,law,scale):
    if law=="CONST": return 1.
    t=float(row.TargetPass)
    if law=="PASS_INC": return .5+.5*max(0,min(1,(t-3)/3))
    if law=="PASS_DEC": return 1/(.5+.5*max(0,min(1,(t-3)/3)))
    ratio=max(float(row.D_joint)/max(float(row.Base_D_joint),EPS),EPS)
    if law=="HISTORY_RATIO": return float(np.clip(.5+abs(math.log(ratio)),.5,2.5))
    if law=="STATE_MAG": return float(np.clip(float(row.D_joint)/max(scale,EPS),.5,2.5))
    if law=="HYBRID":
        a=np.clip(.5+abs(math.log(ratio)),.5,2.5);b=np.clip(float(row.D_joint)/max(scale,EPS),.5,2.5)
        return float(np.clip(math.sqrt(a*b),.5,2.5))
    raise KeyError(law)
def simulate_online(df,P,baseW,mode="BOTH",eta=.5,law="CONST",shape="EXP",policy="KEEP",threshold=1.5,t6prior=None,scale=None):
    df=df.reset_index(drop=True);P=np.asarray(P,float);baseW=normalize_rows(baseW)
    pred=np.zeros(len(df));Wused=np.zeros_like(baseW)
    scale=float(scale if scale is not None else np.median(df.D_joint))
    for group in sorted(df.Group.unique()):
        ids=np.flatnonzero(df.Group.to_numpy()==group)
        ids=ids[np.argsort(df.TargetPass.to_numpy()[ids])]
        mult=np.ones(P.shape[1])
        for ii in ids:
            row=df.iloc[ii];bw=baseW[ii].copy()
            if policy=="T6_PRIOR" and int(row.TargetPass)==6 and t6prior is not None: bw=np.asarray(t6prior,float)
            w=np.maximum(bw*mult,1e-12);w=w/w.sum()
            Wused[ii]=w;pred[ii]=float(P[ii]@w)
            losses=np.abs(P[ii]-float(row.D_joint));ref=float(w@losses)
            adv=np.clip((ref-losses)/max(scale,1.),-2.5,2.5)
            if mode=="NONE": a=np.zeros_like(adv)
            elif mode=="REWARD_ONLY": a=np.maximum(adv,0)
            elif mode=="PENALTY_ONLY": a=np.minimum(adv,0)
            elif mode=="BOTH": a=adv
            else: raise KeyError(mode)
            ef=eta*eta_factor(row,law,scale)
            mult*=update_factor(a,ef,shape)
            mult=np.clip(mult,1e-5,1e5);mult/=max(np.mean(mult),EPS)
            rebound=float(row.D_joint)/max(float(row.Base_D_joint),EPS)>threshold
            if rebound and policy=="SOFTEN": mult=np.sqrt(mult)
            elif rebound and policy=="RESET": mult=np.ones_like(mult)
    return pred,Wused
def tune_eta(train,Pin,baseW,mode,law="CONST",shape="EXP",policy="KEEP",threshold=1.5,t6prior=None):
    scale=float(np.median(train.D_joint));best=None
    for eta in (.1,.25,.5,1.,2.):
        p,_=simulate_online(train,Pin,baseW,mode,eta,law,shape,policy,threshold,t6prior,scale)
        sc=gmae1(train.D_joint,p,train.Group)
        if best is None or (sc,eta)<best[:2]:best=(sc,eta)
    return float(best[1]),float(best[0])
def t6_prior_from_inner(train,Pin):
    m=train.TargetPass.to_numpy()==6
    if m.sum()<3:return np.ones(len(EXPERTS))/len(EXPERTS)
    return opt_weights(train.D_joint.to_numpy()[m],Pin[m],train.Group.to_numpy()[m],.1)[0]

# ---------- V17 ensemble ----------
def run_v17e(dyn):
    arms=["A_QLOG","B_EQUAL","C_CONVEX","D_PASS_GATE","E_STATE_GATE"]
    preds={a:np.full(len(dyn),np.nan) for a in arms}; audits=[]
    for outer in sorted(dyn.Group.unique()):
        train,test,Pin,Pte=expert_bundle(dyn,outer,"D_joint")
        teidx=test.index.to_numpy()
        for arm in arms:
            Wte,Wtr,meta=arm_weights(train,test,Pin,Pte,arm)
            preds[arm][teidx]=np.sum(Pte*Wte,axis=1)
            audits.append({"outer":outer,"arm":arm,**meta})
    summary=[]
    for arm in arms: summary.append({"arm":arm,"Group_MAE_N":gmae1(dyn.D_joint,preds[arm],dyn.Group)})
    best=min(summary,key=lambda x:x["Group_MAE_N"])
    # descriptive oracle headroom
    mat=np.column_stack([preds[a] for a in arms])
    oracle=np.min(np.abs(mat-dyn.D_joint.to_numpy()[:,None]),axis=1)
    oracle_g=float(np.mean([oracle[dyn.Group.to_numpy()==g].mean() for g in dyn.Group.unique()]))
    return {"lane":"V17E","question":"five-arm functional ensemble/gating challenge","summary":summary,
            "champion":best,"oracle_Group_MAE_N_descriptive":oracle_g,"audits":audits,
            "experts":EXPERTS,"outer_prediction_selection_leakage":False}

# ---------- V17 transition ----------
def run_v17t(dyn,tr):
    # T6 paired rebound
    piv=tr.pivot(index="Group",columns="Pass",values="D_joint")
    pair=pd.DataFrame({"D5":piv[5],"D6":piv[6]}).dropna();diff=(pair.D6-pair.D5).to_numpy(float)
    signs=np.asarray(list(itertools.product([-1,1],repeat=len(diff))),float)
    signp=float(np.mean((signs*diff).mean(axis=1)>=diff.mean()-1e-12))
    rebound={"groups":len(diff),"groups_D6_gt_D5":int((diff>0).sum()),
             "groups_D6_gt_2x_D5":int((pair.D6>2*pair.D5).sum()),
             "mean_increase_N":float(diff.mean()),"signflip_one_sided_p":signp,
             "directional_binomial_p":float(binomtest(int((diff>0).sum()),len(diff),.5,alternative="greater").pvalue)}
    # breakpoint stability on training-only outer folds
    bps=[]
    for outer in sorted(dyn.Group.unique()):
        fit=choose_bp(dyn[dyn.Group!=outer],"D_joint");bps.append({"outer":outer,"breakpoint":int(fit["bp"])})
    # prospective rebound classification, features contain history only
    X=state_X(dyn);y=(dyn.D_joint.to_numpy()>dyn.Base_D_joint.to_numpy()).astype(int);g=dyn.Group.to_numpy()
    models={"PASS_ONLY":np.full(len(dyn),np.nan),"STATE_LOGIT":np.full(len(dyn),np.nan),"STATE_RF":np.full(len(dyn),np.nan)}
    for outer in sorted(set(g)):
        trn=g!=outer;te=g==outer
        # pass-only
        lp=LogisticRegression(C=.5,class_weight="balanced",max_iter=1000,random_state=SEED)
        lp.fit(X[trn,:1],y[trn]);models["PASS_ONLY"][te]=lp.predict_proba(X[te,:1])[:,1]
        lg=LogisticRegression(C=.25,class_weight="balanced",max_iter=1000,random_state=SEED)
        lg.fit(X[trn],y[trn]);models["STATE_LOGIT"][te]=lg.predict_proba(X[te])[:,1]
        rf=RandomForestClassifier(n_estimators=250,max_depth=2,min_samples_leaf=2,class_weight="balanced",random_state=SEED,n_jobs=1)
        rf.fit(X[trn],y[trn]);models["STATE_RF"][te]=rf.predict_proba(X[te])[:,1]
    cls=[]
    for name,p in models.items():
        hard=(p>=.5).astype(int)
        row={"model":name,"balanced_accuracy":float(balanced_accuracy_score(y,hard)),
             "brier":float(brier_score_loss(y,p))}
        if len(np.unique(y))==2:row["AUROC"]=float(roc_auc_score(y,p))
        cls.append(row)
    best=min(cls,key=lambda x:x["brier"])
    return {"lane":"V17T","t6_rebound":rebound,"breakpoints":bps,
            "breakpoint_counts":{str(k):int(v) for k,v in zip(*np.unique([x["breakpoint"] for x in bps],return_counts=True))},
            "prospective_rebound_models":cls,"champion":best,
            "guard":"candidate regime transition only; no collapse/fatigue/shakedown class asserted"}

# ---------- V18-V21 generic ----------
def online_lane(dyn,lane):
    if lane=="V18":
        candidates=[{"name":x,"mode":x,"law":"CONST","shape":"EXP","policy":"KEEP"} for x in ("NONE","REWARD_ONLY","PENALTY_ONLY","BOTH")]
    elif lane=="V19":
        candidates=[{"name":x,"mode":"BOTH","law":x,"shape":"EXP","policy":"KEEP"} for x in ("CONST","PASS_INC","PASS_DEC","HISTORY_RATIO","STATE_MAG","HYBRID")]
    elif lane=="V20":
        candidates=[{"name":x,"mode":"BOTH","law":"CONST","shape":x,"policy":"KEEP"} for x in ("EXP","POWER","HYPERBOLIC","TANH","LINEAR")]
    elif lane=="V21":
        candidates=[{"name":x,"mode":"BOTH","law":"CONST","shape":"EXP","policy":x} for x in ("KEEP","SOFTEN","RESET","T6_PRIOR")]
    else: raise KeyError(lane)
    out={c["name"]:np.full(len(dyn),np.nan) for c in candidates};audit=[]
    for outer in sorted(dyn.Group.unique()):
        train,test,Pin,Pte=expert_bundle(dyn,outer,"D_joint")
        prior,_=opt_weights(train.D_joint,Pin,train.Group,.05)
        baseTr=np.tile(prior,(len(train),1));baseTe=np.tile(prior,(len(test),1))
        t6p=t6_prior_from_inner(train,Pin)
        for c in candidates:
            best=None
            thresholds=(1.25,1.5,2.0) if c["policy"] in ("SOFTEN","RESET") else (1.5,)
            for th in thresholds:
                if c["mode"]=="NONE":
                    eta=0.;inner=float(gmae1(train.D_joint,Pin@prior,train.Group))
                else:
                    eta,inner=tune_eta(train,Pin,baseTr,c["mode"],c["law"],c["shape"],c["policy"],th,t6p)
                key=(inner,th,eta)
                if best is None or key<best[0]:best=(key,eta,th,inner)
            _,eta,th,inner=best
            p,_=simulate_online(test,Pte,baseTe,c["mode"],eta,c["law"],c["shape"],c["policy"],th,t6p,float(np.median(train.D_joint)))
            out[c["name"]][test.index.to_numpy()]=p
            audit.append({"outer":outer,"candidate":c["name"],"eta":eta,"threshold":th,"inner_Group_MAE_N":inner})
    summary=[{"candidate":n,"Group_MAE_N":gmae1(dyn.D_joint,p,dyn.Group)} for n,p in out.items()]
    champion=min(summary,key=lambda x:x["Group_MAE_N"])
    medeta=float(np.median([x["eta"] for x in audit if x["candidate"]==champion["candidate"]]))
    return {"lane":lane,"summary":summary,"champion":champion,"champion_median_eta":medeta,"selection_audit":audit,
            "causal_update_guard":"held-out group weights update only after each observed prior transition; never current/future target"}

# ---------- V22 channel-specific ----------
def run_v22(dyn):
    candidates=("SHARED_STATIC","CHANNEL_STATIC","SHARED_DYNAMIC","CHANNEL_DYNAMIC","HIERARCHICAL_DYNAMIC")
    pred1={x:np.full(len(dyn),np.nan) for x in candidates};pred5={x:np.full(len(dyn),np.nan) for x in candidates};audit=[]
    for outer in sorted(dyn.Group.unique()):
        tj,te,pinj,ptej=expert_bundle(dyn,outer,"D_joint")
        t1,te1,pin1,pte1=expert_bundle(dyn,outer,"D_LC1")
        t5,te5,pin5,pte5=expert_bundle(dyn,outer,"D_LC5")
        pj,_=opt_weights(tj.D_joint,pinj,tj.Group,.1);p1,_=opt_weights(t1.D_LC1,pin1,t1.Group,.1);p5,_=opt_weights(t5.D_LC5,pin5,t5.Group,.1)
        eta=.5
        # static
        pred1["SHARED_STATIC"][te.index]=pte1@pj;pred5["SHARED_STATIC"][te.index]=pte5@pj
        pred1["CHANNEL_STATIC"][te.index]=pte1@p1;pred5["CHANNEL_STATIC"][te.index]=pte5@p5
        # shared dynamic updates from joint loss
        _,Wj=simulate_online(te,ptej,np.tile(pj,(len(te),1)),"BOTH",eta,"CONST","EXP","KEEP",1.5,None,float(np.median(tj.D_joint)))
        pred1["SHARED_DYNAMIC"][te.index]=np.sum(pte1*Wj,axis=1);pred5["SHARED_DYNAMIC"][te.index]=np.sum(pte5*Wj,axis=1)
        # channel-specific dynamic (simulate using temporary renamed target)
        def simch(teX,P,W,metric):
            q=teX.copy();q["D_joint"]=q[metric];q["Base_D_joint"]=q["Base_"+metric]
            return simulate_online(q,P,np.tile(W,(len(q),1)),"BOTH",eta,"CONST","EXP","KEEP",1.5,None,float(np.median(q.D_joint)))[1]
        W1=simch(te,pte1,p1,"D_LC1");W5=simch(te,pte5,p5,"D_LC5")
        pred1["CHANNEL_DYNAMIC"][te.index]=np.sum(pte1*W1,axis=1);pred5["CHANNEL_DYNAMIC"][te.index]=np.sum(pte5*W5,axis=1)
        Wh1=normalize_rows(.5*Wj+.5*W1);Wh5=normalize_rows(.5*Wj+.5*W5)
        pred1["HIERARCHICAL_DYNAMIC"][te.index]=np.sum(pte1*Wh1,axis=1);pred5["HIERARCHICAL_DYNAMIC"][te.index]=np.sum(pte5*Wh5,axis=1)
        audit.append({"outer":outer,"shared_prior":pj.tolist(),"lc1_prior":p1.tolist(),"lc5_prior":p5.tolist()})
    summary=[]
    for n in candidates:
        m1=gmae1(dyn.D_LC1,pred1[n],dyn.Group);m5=gmae1(dyn.D_LC5,pred5[n],dyn.Group)
        summary.append({"candidate":n,"joint_Group_MAE_N":.5*(m1+m5),"LC1_Group_MAE_N":m1,"LC5_Group_MAE_N":m5})
    return {"lane":"V22","summary":summary,"champion":min(summary,key=lambda x:x["joint_Group_MAE_N"]),"audit":audit}

# ---------- composite dynamics for V23+ ----------
def upstream_choice(up,lane,default):
    try:return str(up[lane]["champion"]["candidate"] if "candidate" in up[lane]["champion"] else up[lane]["champion"]["arm"])
    except Exception:
        try:return str(up[lane]["champion"]["model"])
        except Exception:return default

def composite_cfg(up):
    return {"arm":upstream_choice(up,"V17E","E_STATE_GATE"),
            "mode":upstream_choice(up,"V18","BOTH"),
            "law":upstream_choice(up,"V19","CONST"),
            "shape":upstream_choice(up,"V20","EXP"),
            "policy":upstream_choice(up,"V21","KEEP"),
            "channel":upstream_choice(up,"V22","SHARED_DYNAMIC")}

_DYN_CACHE={}
def dyn_predict_split(data,outer,metric,cfg):
    key=(tuple(sorted(map(str,data.Group.unique()))),str(outer),str(metric),json.dumps(cfg,sort_keys=True,separators=(",",":")))
    if key in _DYN_CACHE:
        test,p,meta=_DYN_CACHE[key]
        return test.copy(),p.copy(),dict(meta)
    train,test,Pin,Pte=expert_bundle(data,outer,metric)
    # base functional ensemble arm
    arm=cfg.get("arm","E_STATE_GATE")
    if arm not in ("A_QLOG","B_EQUAL","C_CONVEX","D_PASS_GATE","E_STATE_GATE"):arm="E_STATE_GATE"
    Wte,Wtr,_=arm_weights(train,test,Pin,Pte,arm,metric)
    mode=cfg.get("mode","BOTH");mode=mode if mode in ("NONE","REWARD_ONLY","PENALTY_ONLY","BOTH") else "BOTH"
    law=cfg.get("law","CONST");law=law if law in ("CONST","PASS_INC","PASS_DEC","HISTORY_RATIO","STATE_MAG","HYBRID") else "CONST"
    shape=cfg.get("shape","EXP");shape=shape if shape in ("EXP","POWER","HYPERBOLIC","TANH","LINEAR") else "EXP"
    policy=cfg.get("policy","KEEP");policy=policy if policy in ("KEEP","SOFTEN","RESET","T6_PRIOR") else "KEEP"
    # tune eta on cross-fitted training predictions using training-only base weights
    qtrain=train.copy()
    if metric!="D_joint":
        qtrain["D_joint"]=qtrain[metric];qtrain["Base_D_joint"]=qtrain["Base_"+metric]
        qtest=test.copy();qtest["D_joint"]=qtest[metric];qtest["Base_D_joint"]=qtest["Base_"+metric]
    else:qtest=test
    t6p=t6_prior_from_inner(train.assign(D_joint=train[metric]),Pin) if metric=="D_joint" else np.ones(len(EXPERTS))/len(EXPERTS)
    eta,_=tune_eta(qtrain,Pin,Wtr,mode,law,shape,policy,1.5,t6p) if mode!="NONE" else (0.,0.)
    p,W=simulate_online(qtest,Pte,Wte,mode,eta,law,shape,policy,1.5,t6p,float(np.median(qtrain.D_joint)))
    meta={"eta":eta,"W":W}
    _DYN_CACHE[key]=(test.copy(),p.copy(),dict(meta))
    return test,p,meta

# ---------- V11 force baseline and V23 integration ----------
def fit_force_rf(train,test,extra_train=None,extra_test=None):
    X=train[FORCE_BASE].to_numpy(float);Xt=test[FORCE_BASE].to_numpy(float)
    if extra_train is not None:
        X=np.column_stack([X,np.asarray(extra_train,float)]);Xt=np.column_stack([Xt,np.asarray(extra_test,float)])
    y=train[TARGETS].to_numpy(float);prev=train[["prev_LC1","prev_LC5"]].to_numpy(float);delta=y-prev
    w=group_weights(train.Group);out=np.zeros((len(test),2))
    for j in range(2):
        m=RandomForestRegressor(n_estimators=250,max_depth=4,min_samples_leaf=2,criterion="squared_error",
            random_state=20260914,n_jobs=1,max_features=1.)
        m.fit(X,delta[:,j],sample_weight=w);out[:,j]=m.predict(Xt)+test[["prev_LC1","prev_LC5"]].to_numpy(float)[:,j]
    return out
def map_dyn_features(hrows,dynrows,pred):
    mp={(str(r.Group),int(r.TargetPass)):(float(pp),float(r.Base_D_joint),float(r.PrevRatio_D_joint))
        for (_,r),pp in zip(dynrows.iterrows(),pred)}
    vals=[]
    for _,r in hrows.iterrows():
        key=(str(r.Group),int(r.Pass_T))
        if key in mp:
            dh,bd,pr=mp[key];vals.append([dh,dh/max(bd,EPS),math.log(max(pr,EPS)),1.,1. if int(r.Pass_T)==6 else 0.])
        else: vals.append([0.,0.,0.,0.,1. if int(r.Pass_T)==6 else 0.])
    return np.asarray(vals,float)
def scale_force(pred,hrows,dhat,lam):
    out=np.asarray(pred,float).copy();prev=hrows[["prev_LC1","prev_LC5"]].to_numpy(float);dv=out-prev
    mag=.5*(np.abs(dv[:,0])+np.abs(dv[:,1]))
    for i in range(len(out)):
        if dhat[i,3]<.5 or mag[i]<EPS:continue
        target=(1-lam)*mag[i]+lam*dhat[i,0];out[i]=prev[i]+dv[i]*(target/max(mag[i],EPS))
    return out
def force_inner_oof(train_h,train_dyn,cfg,aug=True):
    pred=np.full((len(train_h),2),np.nan);feat=np.zeros((len(train_h),5))
    for vg in sorted(train_h.Group.unique()):
        th=train_h[train_h.Group!=vg];vh=train_h[train_h.Group==vg]
        dd=train_dyn[train_dyn.Group.isin(train_h.Group.unique())]
        if vg in set(dd.Group):
            dtest,dp,_=dyn_predict_split(dd,vg,"D_joint",cfg);f=map_dyn_features(vh,dtest,dp)
        else:f=np.zeros((len(vh),5))
        # training features for th: group-crossfit within remaining force training universe
        ftr=np.zeros((len(th),5))
        for gg in sorted(th.Group.unique()):
            vh2=th[th.Group==gg]
            dd2=train_dyn[train_dyn.Group.isin(th.Group.unique())]
            if gg in set(dd2.Group):
                dt2,dp2,_=dyn_predict_split(dd2,gg,"D_joint",cfg);fm=map_dyn_features(vh2,dt2,dp2)
                ftr[th.Group.to_numpy()==gg]=fm
        p=fit_force_rf(th,vh,ftr,f) if aug else fit_force_rf(th,vh)
        pred[train_h.Group.to_numpy()==vg]=p;feat[train_h.Group.to_numpy()==vg]=f
    return pred,feat

def run_force_integration(d,h,dyn,cfg):
    variants=("BASE_V11","STATE_FEATURE_RF","SCALE_BASE","SCALE_AUG")
    P={v:np.full((len(h),2),np.nan) for v in variants};audit=[]
    for outer in sorted(h.Group.unique()):
        train=h[h.Group!=outer].copy();test=h[h.Group==outer].copy()
        # dynamic feature for test
        dtest,dp,dm=dyn_predict_split(dyn,outer,"D_joint",cfg);fte=map_dyn_features(test,dtest,dp)
        # cross-fitted dynamic features for outer-training force rows
        ftr=np.zeros((len(train),5))
        for gg in sorted(train.Group.unique()):
            dd=dyn[dyn.Group.isin(train.Group.unique())]
            dt,pp,_=dyn_predict_split(dd,gg,"D_joint",cfg);fm=map_dyn_features(train[train.Group==gg],dt,pp)
            ftr[train.Group.to_numpy()==gg]=fm
        base=fit_force_rf(train,test);aug=fit_force_rf(train,test,ftr,fte)
        # training OOF force predictions for lambda selection
        boof,_=force_inner_oof(train,dyn[dyn.Group.isin(train.Group.unique())],cfg,aug=False)
        aoof,foof=force_inner_oof(train,dyn[dyn.Group.isin(train.Group.unique())],cfg,aug=True)
        ytr=train[TARGETS].to_numpy(float);gtr=train.Group.to_numpy()
        bestb=min((gmae_force_scalar(ytr,scale_force(boof,train,foof,l),gtr),l) for l in (0,.25,.5,.75,1.))
        besta=min((gmae_force_scalar(ytr,scale_force(aoof,train,foof,l),gtr),l) for l in (0,.25,.5,.75,1.))
        idx=test.index.to_numpy()
        P["BASE_V11"][idx]=base;P["STATE_FEATURE_RF"][idx]=aug
        P["SCALE_BASE"][idx]=scale_force(base,test,fte,bestb[1])
        P["SCALE_AUG"][idx]=scale_force(aug,test,fte,besta[1])
        audit.append({"outer":outer,"lambda_base":bestb[1],"lambda_aug":besta[1],"dyn_eta":dm["eta"]})
    y=h[TARGETS].to_numpy(float);g=h.Group.to_numpy()
    summary=[{"variant":v,**force_metrics(y,p,g)} for v,p in P.items()]
    base=next(x for x in summary if x["variant"]=="BASE_V11")
    if abs(base["joint_Group_MAE_N"]-V11_LOCK)>1e-6: raise RuntimeError("V11_REPRODUCTION_FAILED:"+str(base["joint_Group_MAE_N"]))
    best=min(summary,key=lambda x:x["joint_Group_MAE_N"])
    for x in summary:
        x["paired_vs_V11"]=paired_group_report(y,P[x["variant"]],P["BASE_V11"],g)
    predictions={v:P[v].tolist() for v in variants}
    return {"summary":summary,"champion":best,"audit":audit,"predictions":predictions,"config":cfg}
def gmae_force_scalar(y,p,g):
    return force_metrics(y,p,g)["joint_Group_MAE_N"]

def run_v23(d,h,dyn,payload):
    cfg=composite_cfg(payload.get("upstream_results",{}))
    r=run_force_integration(d,h,dyn,cfg)
    r.update({"lane":"V23","question":"Does adaptive state/regime dynamics improve the actual V11 force predictor?",
              "v11_reference_N":V11_LOCK,"config_source":"V17-V22 descriptive champions; repeated-development exploratory"})
    return r

def run_v24(d,h,dyn,payload):
    up=payload.get("upstream_results",{});full=composite_cfg(up)
    configs={"FULL":full,
      "NO_ENSEMBLE":{**full,"arm":"A_QLOG"},
      "NO_REWARD_PENALTY":{**full,"mode":"NONE"},
      "FIXED_INTENSITY":{**full,"law":"CONST"},
      "EXP_UPDATE":{**full,"shape":"EXP"},
      "NO_TRANSITION_POLICY":{**full,"policy":"KEEP"}}
    rows=[]
    for name,cfg in configs.items():
        rr=run_force_integration(d,h,dyn,cfg);ch=rr["champion"]
        rows.append({"ablation":name,"best_variant":ch["variant"],"joint_Group_MAE_N":ch["joint_Group_MAE_N"],
                     "LC1_Group_MAE_N":ch["LC1_Group_MAE_N"],"LC5_Group_MAE_N":ch["LC5_Group_MAE_N"]})
    fullrow=next(x for x in rows if x["ablation"]=="FULL")
    for x in rows:x["removal_harm_N"]=float(x["joint_Group_MAE_N"]-fullrow["joint_Group_MAE_N"])
    return {"lane":"V24","component_ablation":rows,"full_config":full,"question":"Which adaptive components contribute to force-level performance?"}

def decode_csv_gz_b64(s):
    return pd.read_csv(BytesIO(gzip.decompress(base64.b64decode(s.strip()))))
def run_v25(d,h,dyn,payload):
    up=payload.get("upstream_results",{});cfg=composite_cfg(up)
    internal=run_force_integration(d,h,dyn,cfg)
    champion=internal["champion"]
    # bootstrap group uncertainty using already cross-fitted champion predictions
    pred=np.asarray(internal["predictions"][champion["variant"]],float);y=h[TARGETS].to_numpy(float);g=h.Group.to_numpy()
    groups=np.array(sorted(set(g)));rng=np.random.default_rng(SEED);vals=[]
    for _ in range(20000):
        sample=rng.choice(groups,size=len(groups),replace=True)
        vals.append(float(np.mean([np.abs(y[g==x]-pred[g==x]).mean() for x in sample])))
    # external ERDC repeated-pass scale-free dynamics only
    ext={}
    if payload.get("erdc_b64"):
        e=decode_csv_gz_b64(payload["erdc_b64"])
        seq=e[(e.vehicle=="MDT")&(e["mode"]=="rolling")].copy()
        seq=seq.sort_values(["soil","depth_cm","trial_pass"])
        rows=[]
        for (soil,depth),z in seq.groupby(["soil","depth_cm"]):
            z=z.sort_values("trial_pass")
            if len(z)>=3:
                v=z.measured_center_pressure_kpa.to_numpy(float);passes=z.trial_pass.to_numpy(int)
                changes=np.abs(np.diff(v))
                rows.append({"soil":str(soil),"depth_cm":float(depth),"n_passes":len(z),
                             "first_change_kpa":float(changes[0]),"last_change_kpa":float(changes[-1]),
                             "change_ratio_last_first":float(changes[-1]/max(changes[0],EPS)),
                             "monotone_response":bool(np.all(np.diff(v)>=0) or np.all(np.diff(v)<=0))})
        ext["ERDC_repeated_pass_scale_free"]=rows
    if payload.get("canada_b64"):
        c=decode_csv_gz_b64(payload["canada_b64"])
        rho,p=spearmanr(c.wheel_load_kn,c.peak_mean_stress_kpa)
        ext["Canada_directional_corroboration"]={"rows":len(c),"spearman_load_vs_peak_stress":float(rho),"p":float(p),
          "direct_model_transfer_valid":False,"reason":"different response units, sensors, design and no repeated-pass sequence"}
    return {"lane":"V25","frozen_internal_candidate":champion,
            "group_bootstrap_95CI_MAE_N":[float(x) for x in np.quantile(vals,[.025,.975])],
            "external":ext,"config":cfg,
            "guard":"External datasets corroborate dynamics/physics only where compatible; no unit-mismatched direct force-model claim."}

def main(payload):
    lane=payload["lane"];phase("LOAD_"+lane);d,h,tr,dyn=load_data(payload)
    if lane=="V17E":result=run_v17e(dyn)
    elif lane=="V17T":result=run_v17t(dyn,tr)
    elif lane in ("V18","V19","V20","V21"):result=online_lane(dyn,lane)
    elif lane=="V22":result=run_v22(dyn)
    elif lane=="V23":result=run_v23(d,h,dyn,payload)
    elif lane=="V24":result=run_v24(d,h,dyn,payload)
    elif lane=="V25":result=run_v25(d,h,dyn,payload)
    else:raise KeyError(lane)
    result.update({"schema":"soilbin.v17-v25.parallel.v1","status":"COMPLETE","lane":lane,
        "campaign":payload.get("campaign"),"source_sha256":EXPECTED_DATA,"elapsed_seconds":time.time()-START,
        "post_lock_exploratory":True,"does_not_supersede_v5":True,
        "guardrails":{"random_split_used":False,"targets_modified":False,"current_or_future_target_used_for_prediction":False,
                      "outer_speed_load_group_held_out":True,"independent_new_experiment":False}})
    save("RESULTS.json",result);phase("COMPLETE_"+lane)
    print("SOILBIN_LANE_COMPLETE",json.dumps({"lane":lane,"champion":result.get("champion"),
      "status":result["status"]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    try:main(PAYLOAD)
    except Exception as exc:
        save("RESULTS.json",{"schema":"soilbin.v17-v25.parallel.v1","status":"FAILED",
          "lane":globals().get("PAYLOAD",{}).get("lane"),"error_type":type(exc).__name__,
          "error":str(exc)[:1600],"traceback":traceback.format_exc()[-6000:],
          "source_sha256":EXPECTED_DATA,"elapsed_seconds":time.time()-START})
        print("SOILBIN_LANE_FAILED",type(exc).__name__,str(exc)[:1200],flush=True)
        raise
