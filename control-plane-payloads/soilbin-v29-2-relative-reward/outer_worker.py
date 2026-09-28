from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
META=ROOT.parent/"soilbin-v29-1-meta-reliability"

src=(META/"outer_worker.py").read_text(encoding="utf-8")
defs=src[:src.rindex("\nouter=sys.argv[1]")]
E={"__name__":"soilbin_v292_defs","__file__":str(META/"outer_worker.py")}
exec(compile(defs,str(META/"outer_worker.py"),"exec"),E)

H=E["H"]; dyn=E["dyn"]; N=E["N"]; EXPERTS=E["EXPERTS"]
ETA_GRID=(0.1,0.25,0.5,1.0,2.0)
MODES=("V18_REFERENCE","RELATIVE_RANK","RELATIVE_PAIRWISE","RELATIVE_HIST_PERCENTILE",
       "RELATIVE_TREND","RELATIVE_RANK_TREND","RELATIVE_PAIRWISE_HIST")
VARIANTS=("V23_BASE","STATIC_INV_MAE","TRAIN_PAIRWISE_PRIOR")+MODES

def normalize(w):
    w=np.asarray(w,float)
    w=np.maximum(w,1e-12)
    return w/w.sum()

def inverse_mae_weights(y,P):
    mae=np.mean(np.abs(y[:,None]-P),axis=0)
    return normalize(1.0/np.maximum(mae,1e-6)),mae

def pairwise_prior_weights(y,P):
    loss=np.abs(y[:,None]-P)
    n=P.shape[1]
    wins=np.zeros(n,float);games=np.zeros(n,float)
    for a in range(n):
        for b in range(a+1,n):
            wins[a]+=np.sum(loss[:,a]<loss[:,b]); wins[b]+=np.sum(loss[:,b]<loss[:,a])
            ties=np.sum(np.isclose(loss[:,a],loss[:,b]))
            wins[a]+=.5*ties;wins[b]+=.5*ties
            games[a]+=len(y);games[b]+=len(y)
    rate=np.divide(wins,np.maximum(games,1.0))
    w=np.exp(3.0*(rate-.5))
    return normalize(w),rate

def hist_distributions(df,y,P):
    loss=np.abs(y[:,None]-P)
    out={}
    for t in sorted(df.Pass_T.unique()):
        m=df.Pass_T.to_numpy(int)==int(t)
        for e in range(P.shape[1]):
            vals=np.asarray(loss[m,e],float)
            out[(int(t),e)]=vals
    return out

def rank_score(loss):
    order=np.argsort(loss)
    ranks=np.empty(len(loss),int);ranks[order]=np.arange(len(loss))
    if len(loss)==1:return np.zeros(1)
    return 1.0-2.0*ranks/(len(loss)-1)

def pairwise_score(loss):
    n=len(loss);s=np.zeros(n,float)
    for a in range(n):
        for b in range(n):
            if a==b:continue
            if loss[a]<loss[b]:s[a]+=1
            elif loss[a]>loss[b]:s[a]-=1
    return s/max(n-1,1)

def hist_score(loss,t,hist):
    s=[]
    for e,l in enumerate(loss):
        vals=hist.get((int(t),e),np.asarray([l],float))
        # low loss -> positive reward; high loss -> penalty
        pct=float(np.mean(vals<=l))
        s.append(1.0-2.0*pct)
    return np.asarray(s,float)

def simulate(df,y,P,basew,mode,eta,hist):
    df=df.reset_index(drop=True);y=np.asarray(y,float);P=np.asarray(P,float)
    pred=np.full(len(df),np.nan);W=np.zeros((len(df),P.shape[1]))
    scale=max(float(np.median(np.abs(y[:,None]-P))),1.0)
    for group in sorted(df.Group.unique()):
        ids=np.flatnonzero(df.Group.to_numpy()==group)
        ids=ids[np.argsort(df.Pass_T.to_numpy()[ids])]
        w=normalize(basew.copy())
        prev_loss=None
        for ii in ids:
            W[ii]=w
            pred[ii]=float(P[ii]@w)
            loss=np.abs(P[ii]-y[ii])
            if mode=="V18_REFERENCE":
                ref=float(w@loss);score=(ref-loss)/scale
            elif mode=="RELATIVE_RANK":
                score=rank_score(loss)
            elif mode=="RELATIVE_PAIRWISE":
                score=pairwise_score(loss)
            elif mode=="RELATIVE_HIST_PERCENTILE":
                score=hist_score(loss,int(df.iloc[ii].Pass_T),hist)
            elif mode=="RELATIVE_TREND":
                if prev_loss is None:score=np.zeros_like(loss)
                else:
                    raw=(prev_loss-loss)/scale
                    score=raw-np.mean(raw)
            elif mode=="RELATIVE_RANK_TREND":
                rs=rank_score(loss)
                if prev_loss is None:ts=np.zeros_like(loss)
                else:
                    raw=(prev_loss-loss)/scale;ts=raw-np.mean(raw)
                score=.5*rs+.5*np.clip(ts,-1,1)
            elif mode=="RELATIVE_PAIRWISE_HIST":
                score=.5*pairwise_score(loss)+.5*hist_score(loss,int(df.iloc[ii].Pass_T),hist)
            else:
                raise KeyError(mode)
            w=normalize(w*np.exp(np.clip(float(eta)*score,-5,5)))
            prev_loss=loss.copy()
    return pred,W

def gmae(y,p,g):
    return N["gmae1"](np.asarray(y,float),np.asarray(p,float),np.asarray(g))

outer=sys.argv[1]
train=H[H.Group!=outer].copy().reset_index(drop=True)
test=H[H.Group==outer].copy().reset_index(drop=True)
dd=dyn[dyn.Group.isin(set(train.Group.unique()))].copy()
t0=time.time()

# Strict outer-training OOF expert predictions for tuning relative reward/penalty.
Ptr=np.full((len(train),2,3),np.nan)
for gg in sorted(train.Group.unique()):
    trn=train[train.Group!=gg].copy().reset_index(drop=True)
    va=train[train.Group==gg].copy().reset_index(drop=True)
    Pva,Ftr,diag,meta=E["fit_experts"](trn,va,dd,gg)
    Ptr[train.Group.to_numpy()==gg]=Pva
if not np.isfinite(Ptr).all():raise RuntimeError("INCOMPLETE_TRAIN_OOF_EXPERTS")

# Untouched outer-test expert predictions.
Pte,Ffull,diagfull,fullmeta=E["fit_experts"](train,test,dyn,outer)

PV={v:np.zeros((len(test),2),float) for v in VARIANTS}
PV["V23_BASE"][:]=Pte[:,:,0]
audit={"outer_group":outer,"channels":{}}

for j in range(2):
    ytr=train[N["TARGETS"][j]].to_numpy(float);gtr=train.Group.to_numpy()
    yte=test[N["TARGETS"][j]].to_numpy(float)
    invw,maes=inverse_mae_weights(ytr,Ptr[:,j,:])
    pairw,winrate=pairwise_prior_weights(ytr,Ptr[:,j,:])
    hist=hist_distributions(train,ytr,Ptr[:,j,:])

    # Static priors.
    PV["STATIC_INV_MAE"][:,j]=Pte[:,j,:]@invw
    PV["TRAIN_PAIRWISE_PRIOR"][:,j]=Pte[:,j,:]@pairw

    tuned={}
    for mode in MODES:
        scored=[]
        for eta in ETA_GRID:
            p,_=simulate(train,ytr,Ptr[:,j,:],invw,mode,eta,hist)
            scored.append((gmae(ytr,p,gtr),float(eta)))
        scored.sort(key=lambda x:(x[0],x[1]))
        best_sc,best_eta=scored[0]
        pte,Wte=simulate(test,yte,Pte[:,j,:],invw,mode,best_eta,hist)
        PV[mode][:,j]=pte
        tuned[mode]={
            "selected_eta":best_eta,
            "inner_Group_MAE_N":best_sc,
            "eta_scores":[{"eta":eta,"Group_MAE_N":sc} for sc,eta in scored],
            "outer_weights_used_before_each_result":Wte.tolist()
        }

    audit["channels"][str(j)]={
        "target":N["TARGETS"][j],
        "static_inverse_MAE_weights":{EXPERTS[e]:float(invw[e]) for e in range(3)},
        "training_OOF_expert_MAE_N":{EXPERTS[e]:float(maes[e]) for e in range(3)},
        "training_pairwise_win_rate":{EXPERTS[e]:float(winrate[e]) for e in range(3)},
        "training_pairwise_prior_weights":{EXPERTS[e]:float(pairw[e]) for e in range(3)},
        "tuned_modes":tuned
    }

out={
    "outer_group":outer,
    "row_indices":H[H.Group==outer].index.to_numpy(int).tolist(),
    "pass_T":test.Pass_T.astype(int).tolist(),
    "groups":test.Group.tolist(),
    "y":test[N["TARGETS"]].to_numpy(float).tolist(),
    "expert_predictions":{EXPERTS[e]:Pte[:,:,e].tolist() for e in range(3)},
    "variant_predictions":{v:PV[v].tolist() for v in VARIANTS},
    "audit":audit,
    "elapsed_seconds":time.time()-t0,
    "guards":{
        "current_test_result_affects_current_prediction":False,
        "observed_test_result_may_affect_only_later_passes_same_group":True,
        "outer_test_used_to_tune_eta_or_base_weights":False,
        "outer_training_OOF_used_for_eta_and_priors":True,
        "future_target_leakage":False,
        "random_split_used":False
    }
}
wd=ROOT/"workers";wd.mkdir(exist_ok=True)
(wd/f"{outer}.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
print("V292_RELATIVE_REWARD_OUTER_COMPLETE",outer,json.dumps({"n":len(test),"elapsed":out["elapsed_seconds"]},ensure_ascii=False),flush=True)
