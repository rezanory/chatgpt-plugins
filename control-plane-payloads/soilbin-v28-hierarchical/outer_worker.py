from __future__ import annotations
import json,sys,time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/"soilbin-v27-probabilistic"
runner=(PREV/"runner.py").read_text(encoding="utf-8")
defs=runner[:runner.index("\ndef main(payload):\n")]
ns={"__name__":"soilbin_v28_hier_defs"}
exec(compile(defs,str(PREV/"runner.py"),"exec"),ns)

ALPHAS=(0.1,1.0,10.0,100.0)
DEPTHS=(1,2,3,4)
WGRID=np.linspace(0,1,11)
TAU_Q=(0.0,0.25,0.5,0.75,1.0)

encoded=(PREV/"all_features_51.csv.gz.b64").read_text(encoding="utf-8").strip()
payload={"lane":"V28_HIER","campaign":"v28-hierarchical","model_b64":encoded}
d,h,tr,dyn=ns["load_data"](payload)

def augment(d,h):
    z=h.copy()
    mp={}
    for _,r in d.iterrows():
        mp[(str(r.Group),int(r.Pass_T))]=(float(r[ns["TARGETS"][0]]),float(r[ns["TARGETS"][1]]))
    for k in DEPTHS:
        for j,s in enumerate(("LC1","LC5")):
            vals=[];avail=[]
            for _,r in z.iterrows():
                key=(str(r.Group),int(r.Pass_T)-k)
                if key in mp:
                    vals.append(mp[key][j]);avail.append(1.0)
                else:
                    vals.append(0.0);avail.append(0.0)
            z[f"lag{k}_{s}"]=vals;z[f"lag{k}_{s}_avail"]=avail
    return z

H=augment(d,h)

def gmae(y,p,g):
    return ns["gmae1"](np.asarray(y,float),np.asarray(p,float),np.asarray(g))

def mem_X(df,j,depth,cross):
    load=df.Load.to_numpy(float);speed=df.Speed.to_numpy(float);pt=df.Pass_T.to_numpy(float)
    parts=[load,speed,pt,load*speed,pt*pt]
    same="LC1" if j==0 else "LC5";other="LC5" if j==0 else "LC1"
    for k in range(1,depth+1):
        sv=df[f"lag{k}_{same}"].to_numpy(float);sa=df[f"lag{k}_{same}_avail"].to_numpy(float)
        parts.extend([sv,sa,np.log1p(np.maximum(sv,0.0))])
        if cross:
            ov=df[f"lag{k}_{other}"].to_numpy(float);oa=df[f"lag{k}_{other}_avail"].to_numpy(float)
            parts.extend([ov,oa,np.log1p(np.maximum(ov,0.0)),sv-ov])
    return np.column_stack(parts)

def fit_mem(df,j,depth,cross,alpha):
    y=df[ns["TARGETS"][j]].to_numpy(float)-df[f"lag1_{'LC1' if j==0 else 'LC5'}"].to_numpy(float)
    m=make_pipeline(StandardScaler(),Ridge(alpha=float(alpha)))
    m.fit(mem_X(df,j,depth,cross),y,ridge__sample_weight=ns["group_weights"](df.Group))
    return m

def predict_mem(m,df,j,depth,cross):
    base=df[f"lag1_{'LC1' if j==0 else 'LC5'}"].to_numpy(float)
    return np.maximum(base+m.predict(mem_X(df,j,depth,cross)),0.0)

def select_mem(df,j,cross,pass_value=None):
    sub=df if pass_value is None else df[df.Pass_T==pass_value]
    if len(sub)<4 or len(np.unique(sub.Group))<4:
        if pass_value is not None:
            return select_mem(df,j,cross,None)
    scored=[]
    for depth in DEPTHS:
        for a in ALPHAS:
            yy=[];pp=[];gg=[]
            for vg in sorted(sub.Group.unique()):
                trn=sub[sub.Group!=vg];va=sub[sub.Group==vg]
                if len(trn)<3: continue
                m=fit_mem(trn,j,depth,cross,a)
                yy.extend(va[ns["TARGETS"][j]].to_numpy(float))
                pp.extend(predict_mem(m,va,j,depth,cross))
                gg.extend(va.Group)
            sc=gmae(yy,pp,gg) if yy else float("inf")
            scored.append((sc,int(depth),float(a)))
    scored.sort(key=lambda x:(x[0],x[1],x[2]))
    s,dpth,a=scored[0]
    return {"depth":dpth,"alpha":a,"inner_Group_MAE_N":s,
            "top":[{"Group_MAE_N":ss,"depth":dd,"alpha":aa} for ss,dd,aa in scored[:12]]}

def mem_predict(train,test,j,cross,pass_value=None):
    meta=select_mem(train,j,cross,pass_value)
    fitdf=train if pass_value is None else train[train.Pass_T==pass_value]
    if len(fitdf)<4 or len(np.unique(fitdf.Group))<4: fitdf=train
    m=fit_mem(fitdf,j,meta["depth"],cross,meta["alpha"])
    return predict_mem(m,test,j,meta["depth"],cross),meta

def mem_oof(train,j,cross,passwise=False):
    p=np.full(len(train),np.nan)
    for vg in sorted(train.Group.unique()):
        trn=train[train.Group!=vg];va=train[train.Group==vg]
        mask=train.Group.to_numpy()==vg
        if not passwise:
            pr,_=mem_predict(trn,va,j,cross,None);p[mask]=pr
        else:
            vv=np.full(len(va),np.nan)
            for t in sorted(va.Pass_T.unique()):
                mt=va.Pass_T.to_numpy()==t
                pr,_=mem_predict(trn,va[mt],j,cross,int(t))
                vv[mt]=pr
            p[mask]=vv
    if not np.isfinite(p).all(): raise RuntimeError("MEM_OOF_INCOMPLETE")
    return p

def choose_w(y,pbase,phead,g,mask=None):
    if mask is None:mask=np.ones(len(y),bool)
    if int(mask.sum())<4:mask=np.ones(len(y),bool)
    rows=[]
    for w in WGRID:
        p=w*pbase+(1-w)*phead
        rows.append((gmae(y[mask],p[mask],g[mask]),float(w)))
    rows.sort(key=lambda x:(x[0],-x[1]))
    return rows[0][1],[{"w_v23":w,"Group_MAE_N":s} for s,w in rows]

def choose_selective(y,pbase,phead,g,mask=None):
    if mask is None:mask=np.ones(len(y),bool)
    if int(mask.sum())<4:mask=np.ones(len(y),bool)
    dis=np.abs(pbase-phead)
    taus=sorted(set(float(x) for x in np.quantile(dis[mask],TAU_Q)))
    rows=[]
    for tau in taus:
        for w in WGRID:
            blend=w*pbase+(1-w)*phead
            p=np.where(dis<=tau,blend,pbase)
            rows.append((gmae(y[mask],p[mask],g[mask]),float(tau),float(w),float(np.mean(dis[mask]<=tau))))
    rows.sort(key=lambda x:(x[0],-x[2],x[1]))
    s,t,w,c=rows[0]
    return t,w,[{"tau":tt,"w_v23":ww,"Group_MAE_N":ss,"coverage":cc} for ss,tt,ww,cc in rows[:20]]

outer=sys.argv[1]
train=H[H.Group!=outer].copy();test=H[H.Group==outer].copy()
cfg=dict(ns["V27_CFG"])
p23,poof,fte,foof=ns["_v23_fold"](train,test,dyn,outer,cfg)
ftr_t,fte_t=ns["_transition_features_outer"](train,test,dyn,outer)

variants=("V23_BASE","SAME_GLOBAL","CROSS_GLOBAL","SAME_PASS","CROSS_PASS",
          "SPARSE_FUSION","HIER_CONVEX","HIER_SELECTIVE","HIER_PROB_GATE")
P={v:p23.copy() for v in variants}
audit={"outer":outer,"channels":{}}
t0=time.time()

for j in range(2):
    ytr=train[ns["TARGETS"][j]].to_numpy(float);gtr=train.Group.to_numpy()
    same_oof=mem_oof(train,j,False,False)
    cross_oof=mem_oof(train,j,True,False)
    samep_oof=mem_oof(train,j,False,True)
    crossp_oof=mem_oof(train,j,True,True)

    same_test,same_meta=mem_predict(train,test,j,False,None)
    cross_test,cross_meta=mem_predict(train,test,j,True,None)
    samep_test=np.full(len(test),np.nan);crossp_test=np.full(len(test),np.nan)
    passmeta={}
    for t in sorted(test.Pass_T.unique()):
        mt=test.Pass_T.to_numpy()==t
        ps,ms=mem_predict(train,test[mt],j,False,int(t))
        pc,mc=mem_predict(train,test[mt],j,True,int(t))
        samep_test[mt]=ps;crossp_test[mt]=pc
        passmeta[str(int(t))]={"same":ms,"cross":mc}

    P["SAME_GLOBAL"][:,j]=same_test
    P["CROSS_GLOBAL"][:,j]=cross_test
    P["SAME_PASS"][:,j]=samep_test
    P["CROSS_PASS"][:,j]=crossp_test

    # Sparse fusion gate chooses same- vs cross-sensor pass head using training-only OOF per pass.
    sparse=np.full(len(test),np.nan);sparse_oof=np.full(len(train),np.nan);which={}
    for t in sorted(train.Pass_T.unique()):
        mtr=train.Pass_T.to_numpy()==t
        scs=gmae(ytr[mtr],samep_oof[mtr],gtr[mtr])
        scc=gmae(ytr[mtr],crossp_oof[mtr],gtr[mtr])
        use_cross=scc<scs
        sparse_oof[mtr]=crossp_oof[mtr] if use_cross else samep_oof[mtr]
        mte=test.Pass_T.to_numpy()==t
        sparse[mte]=crossp_test[mte] if use_cross else samep_test[mte]
        which[str(int(t))]={"selected":"CROSS" if use_cross else "SAME","same_OOF_MAE_N":scs,"cross_OOF_MAE_N":scc}
    P["SPARSE_FUSION"][:,j]=sparse

    # Hierarchical convex blend: V23 backbone + selected pass/sensor head.
    hier=np.full(len(test),np.nan);sel=np.full(len(test),np.nan)
    hmeta={}
    for t in sorted(train.Pass_T.unique()):
        mtr=train.Pass_T.to_numpy()==t;mte=test.Pass_T.to_numpy()==t
        wt,wrows=choose_w(ytr,poof[:,j],sparse_oof,gtr,mtr)
        hier[mte]=wt*p23[mte,j]+(1-wt)*sparse[mte]
        hmeta[str(int(t))]={"w_v23":wt,"scores":wrows}
        tau,ws,srows=choose_selective(ytr,poof[:,j],sparse_oof,gtr,mtr)
        blend=ws*p23[mte,j]+(1-ws)*sparse[mte]
        sel[mte]=np.where(np.abs(p23[mte,j]-sparse[mte])<=tau,blend,p23[mte,j])
        hmeta[str(int(t))]["selective_tau"]=tau
        hmeta[str(int(t))]["selective_w_v23"]=ws
        hmeta[str(int(t))]["selective_scores"]=srows
    P["HIER_CONVEX"][:,j]=hier
    P["HIER_SELECTIVE"][:,j]=sel

    # Probabilistic transition-aware gate for LC5; LC1 reuses hierarchical convex.
    if j==0:
        P["HIER_PROB_GATE"][:,j]=hier
    else:
        probout=np.full(len(test),np.nan)
        ptrain=ftr_t[:,0];ptest=fte_t[:,0];availtr=ftr_t[:,1]>0;availte=fte_t[:,1]>0
        probmeta={}
        for t in sorted(train.Pass_T.unique()):
            mtr=train.Pass_T.to_numpy()==t;mte=test.Pass_T.to_numpy()==t
            basewt=hmeta[str(int(t))]["w_v23"]
            probout[mte]=basewt*p23[mte,j]+(1-basewt)*sparse[mte]
            for label,binmask in (("LOW",ptrain<.5),("HIGH",ptrain>=.5)):
                mm=mtr&binmask&availtr
                if int(mm.sum())>=4 and len(np.unique(gtr[mm]))>=3:
                    wb,_=choose_w(ytr,poof[:,j],sparse_oof,gtr,mm)
                    tt=mte&(ptest<.5 if label=="LOW" else ptest>=.5)&availte
                    probout[tt]=wb*p23[tt,j]+(1-wb)*sparse[tt]
                    probmeta[f"T{int(t)}_{label}"]={"n_train":int(mm.sum()),"w_v23":wb}
        P["HIER_PROB_GATE"][:,j]=probout
        audit["prob_gate_meta"]=probmeta

    audit["channels"][str(j)]={
        "same_global":same_meta,"cross_global":cross_meta,
        "pass_heads":passmeta,"sparse_fusion":which,"hierarchical":hmeta
    }

out={
    "outer":outer,"row_indices":test.index.astype(int).tolist(),"pass_T":test.Pass_T.astype(int).tolist(),
    "groups":test.Group.tolist(),"y":test[ns["TARGETS"]].to_numpy(float).tolist(),
    "predictions":{k:v.tolist() for k,v in P.items()},"audit":audit,
    "elapsed_seconds":time.time()-t0
}
wd=ROOT/"workers";wd.mkdir(exist_ok=True)
(wd/f"{outer}.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
print("V28_HIER_OUTER_COMPLETE",outer,json.dumps({"n":len(test),"elapsed":out["elapsed_seconds"]}),flush=True)
