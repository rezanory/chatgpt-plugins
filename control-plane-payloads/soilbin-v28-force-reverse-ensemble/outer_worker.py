from __future__ import annotations
import base64, gzip, json, sys, time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/"soilbin-v27-probabilistic"
runner=(PREV/"runner.py").read_text(encoding="utf-8")
defs=runner[:runner.index("\ndef main(payload):\n")]
ns={"__name__":"soilbin_v28_force_defs"}
exec(compile(defs,str(PREV/"runner.py"),"exec"),ns)

ALPHAS=(0.1,1.0,10.0,100.0)
WGRID=np.linspace(0,1,11)
TAU_Q=(0.0,0.25,0.5,0.75,1.0)

encoded=(PREV/"all_features_51.csv.gz.b64").read_text(encoding="utf-8").strip()
payload={"lane":"V28_FORCE_REVERSE_ENSEMBLE","campaign":"v28-force-reverse","model_b64":encoded}
d,h,tr,dyn=ns["load_data"](payload)

def group_mae(y,p,g):
    return ns["gmae1"](np.asarray(y,float),np.asarray(p,float),np.asarray(g))

def rev_X(df,j,current_vals):
    load=df.Load.to_numpy(float)
    speed=df.Speed.to_numpy(float)
    pt=df.Pass_T.to_numpy(float)
    other=(df.prev_LC5 if j==0 else df.prev_LC1).to_numpy(float)
    cur=np.asarray(current_vals,float)
    return np.column_stack([
        load,speed,pt,load*speed,pt*pt,other,cur,cur*pt,
        np.log1p(np.maximum(other,0.0))
    ])

def fit_reverse(df,j,alpha):
    y=(df.prev_LC1 if j==0 else df.prev_LC5).to_numpy(float)
    cur=df[ns["TARGETS"][j]].to_numpy(float)
    m=make_pipeline(StandardScaler(),Ridge(alpha=float(alpha)))
    m.fit(rev_X(df,j,cur),y)
    return m

def select_alpha_reverse(df,j):
    rows=[]
    for a in ALPHAS:
        yy=[];pp=[];gg=[]
        for vg in sorted(df.Group.unique()):
            trn=df[df.Group!=vg];va=df[df.Group==vg]
            if len(trn)<4: continue
            m=fit_reverse(trn,j,a)
            cur=va[ns["TARGETS"][j]].to_numpy(float)
            pred=m.predict(rev_X(va,j,cur))
            yy.extend((va.prev_LC1 if j==0 else va.prev_LC5).to_numpy(float))
            pp.extend(pred);gg.extend(va.Group)
        sc=group_mae(yy,pp,gg) if yy else float("inf")
        rows.append((sc,float(a)))
    rows.sort(key=lambda x:(x[0],x[1]))
    return rows[0][1],[{"alpha":a,"reverse_prev_Group_MAE_N":s} for s,a in rows]

def invert_reverse(model,train_df,test_df,j):
    z0=np.zeros(len(test_df),float)
    z1=np.ones(len(test_df),float)
    p0=model.predict(rev_X(test_df,j,z0))
    p1=model.predict(rev_X(test_df,j,z1))
    slope=p1-p0
    observed=(test_df.prev_LC1 if j==0 else test_df.prev_LC5).to_numpy(float)
    vals=train_df[ns["TARGETS"][j]].to_numpy(float)
    lo=float(vals.min());hi=float(vals.max());span=max(hi-lo,1.0)
    lo-=0.25*span;hi+=0.25*span
    out=np.full(len(test_df),float(np.median(vals)))
    good=np.abs(slope)>=1e-6
    out[good]=(observed[good]-p0[good])/slope[good]
    out=np.clip(out,lo,hi)
    return out,np.abs(slope)

def reverse_predict(train,test,j):
    a,meta=select_alpha_reverse(train,j)
    m=fit_reverse(train,j,a)
    pred,slope=invert_reverse(m,train,test,j)
    return pred,slope,{"alpha":a,"alpha_scores":meta}

def reverse_oof(train,j):
    p=np.full(len(train),np.nan);slope=np.full(len(train),np.nan)
    for vg in sorted(train.Group.unique()):
        trn=train[train.Group!=vg];va=train[train.Group==vg]
        pr,sl,_=reverse_predict(trn,va,j)
        mask=train.Group.to_numpy()==vg
        p[mask]=pr;slope[mask]=sl
    if not (np.isfinite(p).all() and np.isfinite(slope).all()):
        raise RuntimeError("REVERSE_OOF_INCOMPLETE")
    return p,slope

def choose_w(y,pf,pr,g,mask=None):
    if mask is None: mask=np.ones(len(y),bool)
    if int(np.sum(mask))<4: mask=np.ones(len(y),bool)
    rows=[]
    for w in WGRID:
        p=w*pf+(1-w)*pr
        rows.append((group_mae(y[mask],p[mask],g[mask]),float(w)))
    rows.sort(key=lambda x:(x[0],-x[1]))
    return rows[0][1],[{"w_forward":w,"Group_MAE_N":s} for s,w in rows]

def choose_gate(y,pf,pr,g):
    dis=np.abs(pf-pr)
    taus=sorted(set(float(x) for x in np.quantile(dis,TAU_Q)))
    rows=[]
    for tau in taus:
        for w in WGRID:
            blend=w*pf+(1-w)*pr
            p=np.where(dis<=tau,blend,pf)
            rows.append((group_mae(y,p,g),float(tau),float(w),float(np.mean(dis<=tau))))
    rows.sort(key=lambda x:(x[0],-x[2],x[1]))
    s,t,w,c=rows[0]
    return t,w,[{"tau":tt,"w_forward":ww,"Group_MAE_N":ss,"coverage":cc} for ss,tt,ww,cc in rows[:25]]

outer=sys.argv[1]
if outer not in set(h.Group): raise SystemExit("UNKNOWN_OUTER:"+outer)
t0=time.time()
train=h[h.Group!=outer].copy()
test=h[h.Group==outer].copy()
cfg=dict(ns["V27_CFG"])
p23,poof,fte,foof=ns["_v23_fold"](train,test,dyn,outer,cfg)

variants=("V23_BASE","REVERSE_ONLY","GLOBAL_CONVEX","SELECTIVE","LC5_ONLY_CONVEX","LC5_T2T6","PASSWISE")
P={v:p23.copy() for v in variants}
audits={"outer":outer,"channels":{}}

for j in range(2):
    ytr=train[ns["TARGETS"][j]].to_numpy(float)
    gtr=train.Group.to_numpy()
    rev_oof,slope_oof=reverse_oof(train,j)
    rev_test,slope_test,rmeta=reverse_predict(train,test,j)
    pf_oof=poof[:,j]
    pf_test=p23[:,j]

    wg,wmeta=choose_w(ytr,pf_oof,rev_oof,gtr)
    tau,ws,gmeta=choose_gate(ytr,pf_oof,rev_oof,gtr)

    P["REVERSE_ONLY"][:,j]=rev_test
    P["GLOBAL_CONVEX"][:,j]=wg*pf_test+(1-wg)*rev_test
    blend=ws*pf_test+(1-ws)*rev_test
    P["SELECTIVE"][:,j]=np.where(np.abs(pf_test-rev_test)<=tau,blend,pf_test)

    # Pass-wise ensemble weights selected on outer-training OOF only.
    pp=pf_test.copy()
    passmeta={}
    for t in sorted(train.Pass_T.unique()):
        mtr=train.Pass_T.to_numpy()==t
        if int(mtr.sum())>=4 and len(np.unique(gtr[mtr]))>=3:
            wt,_=choose_w(ytr,pf_oof,rev_oof,gtr,mtr)
        else:
            wt=wg
        mte=test.Pass_T.to_numpy()==t
        pp[mte]=wt*pf_test[mte]+(1-wt)*rev_test[mte]
        passmeta[str(int(t))]=float(wt)
    P["PASSWISE"][:,j]=pp

    if j==0:
        # LC1 remains V23 for LC5-only arms.
        P["LC5_ONLY_CONVEX"][:,0]=pf_test
        P["LC5_T2T6"][:,0]=pf_test
    else:
        P["LC5_ONLY_CONVEX"][:,1]=wg*pf_test+(1-wg)*rev_test
        hard=train.Pass_T.isin([2,6]).to_numpy()
        wh,hmeta=choose_w(ytr,pf_oof,rev_oof,gtr,hard)
        out=pf_test.copy()
        mh=test.Pass_T.isin([2,6]).to_numpy()
        out[mh]=wh*pf_test[mh]+(1-wh)*rev_test[mh]
        P["LC5_T2T6"][:,1]=out

    audits["channels"][str(j)]={
        "target":ns["TARGETS"][j],"reverse":rmeta,
        "global_w_forward":float(wg),"global_weight_scores":wmeta,
        "selective_tau":float(tau),"selective_w_forward":float(ws),"selective_scores_top25":gmeta,
        "passwise_w_forward":passmeta,
        "test_mean_reverse_slope_abs":float(np.mean(slope_test)),
        "test_mean_forward_reverse_disagreement_N":float(np.mean(np.abs(pf_test-rev_test))),
    }
    if j==1:
        audits["channels"][str(j)]["hard_T2T6_w_forward"]=float(wh)
        audits["channels"][str(j)]["hard_T2T6_scores"]=hmeta

idx=test.index.astype(int).tolist()
out={
    "outer":outer,"row_indices":idx,"pass_T":test.Pass_T.astype(int).tolist(),"groups":test.Group.tolist(),
    "y":test[ns["TARGETS"]].to_numpy(float).tolist(),
    "predictions":{k:v.tolist() for k,v in P.items()},
    "audit":audits,"elapsed_seconds":time.time()-t0
}
wd=ROOT/"workers";wd.mkdir(exist_ok=True)
(wd/f"{outer}.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
print("V28_FORCE_OUTER_COMPLETE",outer,json.dumps({"n":len(test),"elapsed":out["elapsed_seconds"]}),flush=True)
