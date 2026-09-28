from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
V28F=BASE/"soilbin-v28-force-reverse-ensemble"
V27=BASE/"soilbin-v27-probabilistic"

# Leakage-safe definitions from prior verified runners.
src=(V28F/"outer_worker.py").read_text(encoding="utf-8")
defs=src[:src.index("outer=sys.argv[1]")]
E={"__name__":"soilbin_v291_defs","__file__":str(V28F/"outer_worker.py")}
exec(compile(defs,str(V28F/"outer_worker.py"),"exec"),E)
N=E["ns"]; d=E["d"]; h0=E["h"]; dyn=E["dyn"]
CFG=dict(N["V27_CFG"])
EXPERTS=("V23","HIERARCHICAL","REVERSE_STACK")
RIDGE_ALPHA=10.0
DEPTHS={
    0:{2:1,3:2,4:3,5:2,6:3},  # LC1
    1:{2:1,3:2,4:2,5:2,6:4},  # LC5
}

def augment(d,h):
    z=h.copy()
    mp={}
    for _,r in d.iterrows():
        mp[(str(r.Group),int(r.Pass_T))]=(float(r[N["TARGETS"][0]]),float(r[N["TARGETS"][1]]))
    for k in range(1,5):
        for j,s in enumerate(("LC1","LC5")):
            vals=[]; avail=[]
            for _,r in z.iterrows():
                key=(str(r.Group),int(r.Pass_T)-k)
                if key in mp:
                    vals.append(mp[key][j]); avail.append(1.0)
                else:
                    vals.append(0.0); avail.append(0.0)
            z[f"lag{k}_{s}"]=vals
            z[f"lag{k}_{s}_avail"]=avail
    return z

H=augment(d,h0)

def hX(df,j,depth):
    load=df.Load.to_numpy(float); speed=df.Speed.to_numpy(float); pt=df.Pass_T.to_numpy(float)
    same="LC1" if j==0 else "LC5"
    other="LC5" if j==0 else "LC1"
    parts=[load,speed,pt,load*speed,pt*pt]
    for k in range(1,depth+1):
        sv=df[f"lag{k}_{same}"].to_numpy(float)
        sa=df[f"lag{k}_{same}_avail"].to_numpy(float)
        ov=df[f"lag{k}_{other}"].to_numpy(float)
        oa=df[f"lag{k}_{other}_avail"].to_numpy(float)
        parts.extend([sv,sa,np.log1p(np.maximum(sv,0.0)),ov,oa,np.log1p(np.maximum(ov,0.0)),sv-ov])
    return np.column_stack(parts)

def fit_hier_models(train,test,j):
    out=np.full(len(test),np.nan)
    fit=np.full(len(train),np.nan)
    meta={}
    target=N["TARGETS"][j]
    same="LC1" if j==0 else "LC5"
    for t in sorted(test.Pass_T.unique()):
        t=int(t); depth=DEPTHS[j][t]
        trm=train.Pass_T.to_numpy(int)==t
        tem=test.Pass_T.to_numpy(int)==t
        fitdf=train[trm] if int(trm.sum())>=4 and len(np.unique(train.loc[trm,"Group"]))>=4 else train
        ydelta=fitdf[target].to_numpy(float)-fitdf[f"lag1_{same}"].to_numpy(float)
        m=make_pipeline(StandardScaler(),Ridge(alpha=RIDGE_ALPHA))
        m.fit(hX(fitdf,j,depth),ydelta)
        out[tem]=np.maximum(test.loc[tem,f"lag1_{same}"].to_numpy(float)+m.predict(hX(test.loc[tem],j,depth)),0.0)
        if np.any(trm):
            fit[trm]=np.maximum(train.loc[trm,f"lag1_{same}"].to_numpy(float)+m.predict(hX(train.loc[trm],j,depth)),0.0)
        meta[str(t)]={"depth":depth,"alpha":RIDGE_ALPHA,"n_fit":int(len(fitdf))}
    # Fill any unusual missing pass fit values with a global depth-2 model.
    miss=~np.isfinite(fit)
    if np.any(miss):
        depth=2
        ydelta=train[target].to_numpy(float)-train[f"lag1_{same}"].to_numpy(float)
        m=make_pipeline(StandardScaler(),Ridge(alpha=RIDGE_ALPHA))
        m.fit(hX(train,j,depth),ydelta)
        fit[miss]=np.maximum(train.loc[miss,f"lag1_{same}"].to_numpy(float)+m.predict(hX(train.loc[miss],j,depth)),0.0)
    return out,fit,meta

def reverse_meta_X(df,pf,pr,slope):
    pt=df.Pass_T.to_numpy(float); load=df.Load.to_numpy(float); speed=df.Speed.to_numpy(float)
    p1=df.prev_LC1.to_numpy(float); p5=df.prev_LC5.to_numpy(float)
    return np.column_stack([pf,pr,np.abs(pf-pr),slope,pt,load,speed,p1,p5,pf*pt,pr*pt,load*speed])

def fit_experts(train,test,dyn_universe,holdout_group):
    # V23 forward expert and strict inner OOF predictions inside this universe.
    p23,poof,fte,foof=N["_v23_fold"](train,test,dyn_universe,holdout_group,CFG)
    # Apparent train-fit diagnostic for V23 using training-fitted force RF on cross-fitted state features.
    p23_fit=N["fit_force_rf"](train,train,foof,foof)

    P=np.zeros((len(test),2,3),float)
    F=np.zeros((len(train),2,3),float)
    P[:,:,0]=p23; F[:,:,0]=p23_fit

    hiermeta={}
    for j in range(2):
        ph,fh,hm=fit_hier_models(train,test,j)
        P[:,j,1]=ph; F[:,j,1]=fh
        hiermeta[str(j)]=hm

        rev_oof,slope_oof=E["reverse_oof"](train,j)
        rev_test,slope_test,rmeta=E["reverse_predict"](train,test,j)
        ytr=train[N["TARGETS"][j]].to_numpy(float)

        meta=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
        meta.fit(reverse_meta_X(train,poof[:,j],rev_oof,slope_oof),ytr,
                 ridge__sample_weight=N["group_weights"](train.Group))
        P[:,j,2]=np.maximum(meta.predict(reverse_meta_X(test,p23[:,j],rev_test,slope_test)),0.0)
        # train-side fit result of the second-level stacker on OOF base predictions.
        F[:,j,2]=np.maximum(meta.predict(reverse_meta_X(train,poof[:,j],rev_oof,slope_oof)),0.0)

    diag={}
    for j in range(2):
        target=N["TARGETS"][j]
        ytr=train[target].to_numpy(float)
        diag[str(j)]={}
        for e,name in enumerate(EXPERTS):
            err=np.abs(ytr-F[:,j,e])
            pass_mae={}
            for t in sorted(train.Pass_T.unique()):
                m=train.Pass_T.to_numpy(int)==int(t)
                pass_mae[str(int(t))]=float(np.mean(err[m])) if np.any(m) else float(np.mean(err))
            diag[str(j)][name]={
                "train_fit_MAE_N":float(np.mean(err)),
                "train_fit_error_SD_N":float(np.std(err,ddof=0)),
                "pass_train_fit_MAE_N":pass_mae,
            }
    return P,F,diag,{"hierarchical":hiermeta}

def sample_features(df,P,diag,j):
    # Sample-level pre-target features + all expert predictions/disagreement + training-result diagnostics.
    n=len(df); pt=df.Pass_T.to_numpy(float)
    rows=[]
    for i in range(n):
        preds=P[i,j,:]
        f=[
            float(df.iloc[i].Speed),float(df.iloc[i].Load),float(df.iloc[i].Pass_T),
            float(df.iloc[i].prev_LC1),float(df.iloc[i].prev_LC5),
            float(preds[0]),float(preds[1]),float(preds[2]),
            float(np.std(preds)),float(np.max(preds)-np.min(preds)),
            float(abs(preds[0]-preds[1])),float(abs(preds[0]-preds[2])),float(abs(preds[1]-preds[2])),
            float(j),
        ]
        t=str(int(pt[i]))
        for name in EXPERTS:
            q=diag[str(j)][name]
            f.extend([q["train_fit_MAE_N"],q["train_fit_error_SD_N"],q["pass_train_fit_MAE_N"].get(t,q["train_fit_MAE_N"])])
        rows.append(f)
    return np.asarray(rows,float)

def expert_features(sampleX,e):
    # Add expert identity one-hot to a shared sample representation.
    one=np.zeros((len(sampleX),3),float);one[:,e]=1.0
    return np.column_stack([sampleX,one])

outer=sys.argv[1]
if outer not in set(H.Group):
    raise SystemExit("UNKNOWN_OUTER:"+outer)

t0=time.time()
train=H[H.Group!=outer].copy()
test=H[H.Group==outer].copy()
outer_train_groups=set(train.Group.unique())
dd=dyn[dyn.Group.isin(outer_train_groups)].copy()

# Build strict meta-training examples: each inner validation group is predicted by experts trained without that group.
meta_parts=[]
for gg in sorted(train.Group.unique()):
    trn=train[train.Group!=gg].copy()
    va=train[train.Group==gg].copy()
    Pva,Ftr,diag,emeta=fit_experts(trn,va,dd,gg)
    part={
        "group":gg,
        "indices":va.index.to_numpy(int),
        "P":Pva,
        "diag":diag,
        "sampleX":[sample_features(va,Pva,diag,j) for j in range(2)],
        "y":va[N["TARGETS"]].to_numpy(float),
    }
    meta_parts.append(part)

# Full outer-training expert fit, applied once to untouched outer test.
Pte,Ffull,diagfull,fullmeta=fit_experts(train,test,dyn,outer)
sampleXte=[sample_features(test,Pte,diagfull,j) for j in range(2)]

variants=("V23_BASE","HIERARCHICAL_BASE","REVERSE_STACK_BASE","ERROR_ROUTER","WINNER_ROUTER","GAP_ROUTER","META_ROUTER")
PV={v:np.zeros((len(test),2),float) for v in variants}
PV["V23_BASE"][:]=Pte[:,:,0]
PV["HIERARCHICAL_BASE"][:]=Pte[:,:,1]
PV["REVERSE_STACK_BASE"][:]=Pte[:,:,2]

audit={"outer_group":outer,"channels":{}}

for j in range(2):
    # Concatenate meta-training data in original inner-fold order.
    Xs=np.vstack([p["sampleX"][j] for p in meta_parts])
    ys=np.concatenate([p["y"][:,j] for p in meta_parts])
    Ps=np.vstack([p["P"][:,j,:] for p in meta_parts])
    groups=np.concatenate([[p["group"]]*len(p["indices"]) for p in meta_parts])
    abs_err=np.abs(ys[:,None]-Ps)

    # Training-fit baseline used by the gap target is generated only from the model's own inner-training universe.
    train_baseline=np.zeros_like(abs_err)
    offset=0
    for p in meta_parts:
        n=len(p["indices"])
        tt=np.asarray([int(x) for x in train.loc[p["indices"],"Pass_T"]])
        for e,name in enumerate(EXPERTS):
            q=p["diag"][str(j)][name]
            train_baseline[offset:offset+n,e]=np.asarray([
                q["pass_train_fit_MAE_N"].get(str(int(t)),q["train_fit_MAE_N"]) for t in tt
            ])
        offset+=n
    gaps=abs_err-train_baseline

    # 1) Error Predictor — separate low-capacity Ridge per expert.
    pred_err_te=np.zeros((len(test),3),float)
    err_cv_mae=[]
    for e in range(3):
        m=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
        m.fit(expert_features(Xs,e),abs_err[:,e])
        pred_err_te[:,e]=np.maximum(m.predict(expert_features(sampleXte[j],e)),0.0)
        # diagnostic fit MAE only; not used for test selection tuning.
        err_cv_mae.append(float(np.mean(np.abs(abs_err[:,e]-np.maximum(m.predict(expert_features(Xs,e)),0.0)))))
    sel_err=np.argmin(pred_err_te,axis=1)
    PV["ERROR_ROUTER"][:,j]=Pte[np.arange(len(test)),j,sel_err]

    # 2) Winner Classifier — multiclass expert winner.
    winner=np.argmin(abs_err,axis=1)
    classes=np.unique(winner)
    if len(classes)>=2:
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=2000,class_weight="balanced",multi_class="auto"))
        clf.fit(Xs,winner)
        raw=clf.predict_proba(sampleXte[j])
        probs=np.zeros((len(test),3),float)
        for k,c in enumerate(clf.named_steps["logisticregression"].classes_):
            probs[:,int(c)]=raw[:,k]
        sel_win=np.argmax(probs,axis=1)
        train_acc=float(np.mean(clf.predict(Xs)==winner))
    else:
        probs=np.zeros((len(test),3),float); probs[:,int(classes[0])]=1.0
        sel_win=np.full(len(test),int(classes[0]),int); train_acc=1.0
    PV["WINNER_ROUTER"][:,j]=Pte[np.arange(len(test)),j,sel_win]

    # 3) Generalization-Gap Predictor.
    pred_gap_te=np.zeros((len(test),3),float)
    gap_fit_mae=[]
    baseline_te=np.zeros((len(test),3),float)
    tte=test.Pass_T.to_numpy(int)
    for e,name in enumerate(EXPERTS):
        q=diagfull[str(j)][name]
        baseline_te[:,e]=np.asarray([q["pass_train_fit_MAE_N"].get(str(int(t)),q["train_fit_MAE_N"]) for t in tte])
        m=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
        m.fit(expert_features(Xs,e),gaps[:,e])
        pred_gap_te[:,e]=m.predict(expert_features(sampleXte[j],e))
        gap_fit_mae.append(float(np.mean(np.abs(gaps[:,e]-m.predict(expert_features(Xs,e))))))
    expected_gap_error=np.maximum(baseline_te+pred_gap_te,0.0)
    sel_gap=np.argmin(expected_gap_error,axis=1)
    PV["GAP_ROUTER"][:,j]=Pte[np.arange(len(test)),j,sel_gap]

    # 4) Combined Meta-Router — fixed rank aggregation of the three independent meta signals.
    # Lower error/gap ranks are better; higher winner probability is better.
    rank_err=np.argsort(np.argsort(pred_err_te,axis=1),axis=1)
    rank_gap=np.argsort(np.argsort(expected_gap_error,axis=1),axis=1)
    rank_win=np.argsort(np.argsort(-probs,axis=1),axis=1)
    rank_sum=rank_err+rank_gap+rank_win
    # Tie-breaker is Error Predictor score; no outer-test label involved.
    sel_meta=np.zeros(len(test),int)
    for i in range(len(test)):
        best=np.flatnonzero(rank_sum[i]==rank_sum[i].min())
        sel_meta[i]=int(best[np.argmin(pred_err_te[i,best])])
    PV["META_ROUTER"][:,j]=Pte[np.arange(len(test)),j,sel_meta]

    yte=test[N["TARGETS"][j]].to_numpy(float)
    true_err=np.abs(yte[:,None]-Pte[:,j,:])
    true_win=np.argmin(true_err,axis=1)

    audit["channels"][str(j)]={
        "target":N["TARGETS"][j],
        "n_meta_train_rows":int(len(Xs)),
        "winner_class_counts":{EXPERTS[e]:int(np.sum(winner==e)) for e in range(3)},
        "error_predictor_train_fit_MAE_N":{EXPERTS[e]:err_cv_mae[e] for e in range(3)},
        "gap_predictor_train_fit_MAE_N":{EXPERTS[e]:gap_fit_mae[e] for e in range(3)},
        "winner_classifier_train_accuracy":train_acc,
        "outer_test_diagnostics_not_used_for_training":{
            "error_predictor_error_MAE_N":float(np.mean(np.abs(pred_err_te-true_err))),
            "winner_classifier_accuracy":float(np.mean(sel_win==true_win)),
            "gap_expected_error_MAE_N":float(np.mean(np.abs(expected_gap_error-true_err))),
        },
        "selection_counts":{
            "ERROR_ROUTER":{EXPERTS[e]:int(np.sum(sel_err==e)) for e in range(3)},
            "WINNER_ROUTER":{EXPERTS[e]:int(np.sum(sel_win==e)) for e in range(3)},
            "GAP_ROUTER":{EXPERTS[e]:int(np.sum(sel_gap==e)) for e in range(3)},
            "META_ROUTER":{EXPERTS[e]:int(np.sum(sel_meta==e)) for e in range(3)},
        },
        "outer_test_expected_error_error_predictor":pred_err_te.tolist(),
        "outer_test_expected_error_gap_predictor":expected_gap_error.tolist(),
        "outer_test_winner_probabilities":probs.tolist(),
    }

out={
    "outer_group":outer,
    "row_indices":test.index.to_numpy(int).tolist(),
    "pass_T":test.Pass_T.astype(int).tolist(),
    "groups":test.Group.tolist(),
    "y":test[N["TARGETS"]].to_numpy(float).tolist(),
    "expert_predictions":{EXPERTS[e]:Pte[:,:,e].tolist() for e in range(3)},
    "variant_predictions":{v:PV[v].tolist() for v in variants},
    "audit":audit,
    "elapsed_seconds":time.time()-t0,
    "guards":{
        "outer_test_labels_used_for_meta_training":False,
        "inner_group_cross_fitting":True,
        "current_or_future_target_used_as_meta_feature":False,
        "random_split_used":False,
    }
}
wd=ROOT/"workers";wd.mkdir(exist_ok=True)
(wd/f"{outer}.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
print("V291_META_OUTER_COMPLETE",outer,json.dumps({"n":len(test),"elapsed":out["elapsed_seconds"]},ensure_ascii=False),flush=True)
