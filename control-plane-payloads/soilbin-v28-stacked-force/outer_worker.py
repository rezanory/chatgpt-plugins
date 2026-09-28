from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/"soilbin-v28-force-reverse-ensemble"
src=(BASE/"outer_worker.py").read_text(encoding="utf-8")
defs=src[:src.index("outer=sys.argv[1]")]
ns={"__name__":"soilbin_v281_stacked_defs","__file__":str(BASE/"outer_worker.py")}
exec(compile(defs,str(BASE/"outer_worker.py"),"exec"),ns)

h=ns["h"];dyn=ns["dyn"];N=ns["ns"]
META_ALPHA=10.0

def meta_X(df,pf,pr,slope):
    pt=df.Pass_T.to_numpy(float);load=df.Load.to_numpy(float);speed=df.Speed.to_numpy(float)
    p1=df.prev_LC1.to_numpy(float);p5=df.prev_LC5.to_numpy(float)
    return np.column_stack([
        pf,pr,np.abs(pf-pr),slope,pt,load,speed,p1,p5,
        pf*pt,pr*pt,load*speed
    ])

def fit_meta(df,y,pf,pr,slope):
    m=make_pipeline(StandardScaler(),Ridge(alpha=META_ALPHA))
    m.fit(meta_X(df,pf,pr,slope),np.asarray(y,float),ridge__sample_weight=N["group_weights"](df.Group))
    return m

def meta_oof(df,y,pf,pr,slope):
    out=np.full(len(df),np.nan)
    for vg in sorted(df.Group.unique()):
        tr=df.Group.to_numpy()!=vg;va=~tr
        m=fit_meta(df[tr],np.asarray(y)[tr],np.asarray(pf)[tr],np.asarray(pr)[tr],np.asarray(slope)[tr])
        out[va]=m.predict(meta_X(df[va],np.asarray(pf)[va],np.asarray(pr)[va],np.asarray(slope)[va]))
    if not np.isfinite(out).all():raise RuntimeError("META_OOF_INCOMPLETE")
    return out

outer=sys.argv[1]
train=h[h.Group!=outer].copy();test=h[h.Group==outer].copy()
cfg=dict(N["V27_CFG"])
p23,poof,fte,foof=N["_v23_fold"](train,test,dyn,outer,cfg)

variants=("V23_BASE","STACKED_BOTH","LC5_STACKED","LC5_STACKED_GATED",
          "LC5_T2_STACKED","LC5_T2T6_STACKED","STACKED_GATED_BOTH")
P={v:p23.copy() for v in variants}
audit={"outer":outer,"channels":{}}
t0=time.time()

for j in range(2):
    ytr=train[N["TARGETS"][j]].to_numpy(float)
    gtr=train.Group.to_numpy()
    rev_oof,slope_oof=ns["reverse_oof"](train,j)
    rev_test,slope_test,rmeta=ns["reverse_predict"](train,test,j)
    pf_oof=poof[:,j];pf_test=p23[:,j]

    meta=fit_meta(train,ytr,pf_oof,rev_oof,slope_oof)
    pm=meta.predict(meta_X(test,pf_test,rev_test,slope_test))
    pm_oof=meta_oof(train,ytr,pf_oof,rev_oof,slope_oof)

    P["STACKED_BOTH"][:,j]=pm
    if j==0:
        P["LC5_STACKED"][:,0]=pf_test
        P["LC5_STACKED_GATED"][:,0]=pf_test
        P["LC5_T2_STACKED"][:,0]=pf_test
        P["LC5_T2T6_STACKED"][:,0]=pf_test
    else:
        P["LC5_STACKED"][:,1]=pm
        q=pf_test.copy();m2=test.Pass_T.to_numpy()==2;q[m2]=pm[m2];P["LC5_T2_STACKED"][:,1]=q
        q=pf_test.copy();mh=np.isin(test.Pass_T.to_numpy(),[2,6]);q[mh]=pm[mh];P["LC5_T2T6_STACKED"][:,1]=q

    gated=pf_test.copy();passmeta={}
    for t in sorted(train.Pass_T.unique()):
        mtr=train.Pass_T.to_numpy()==t
        base=N["gmae1"](ytr[mtr],pf_oof[mtr],gtr[mtr])
        sm=N["gmae1"](ytr[mtr],pm_oof[mtr],gtr[mtr])
        use=sm<base
        mte=test.Pass_T.to_numpy()==t
        if use:gated[mte]=pm[mte]
        passmeta[str(int(t))]={"V23_OOF_MAE_N":float(base),"STACKED_OOF_MAE_N":float(sm),"use_stacked":bool(use)}
    P["STACKED_GATED_BOTH"][:,j]=gated
    if j==1:P["LC5_STACKED_GATED"][:,1]=gated

    audit["channels"][str(j)]={
        "reverse":rmeta,"meta_alpha":META_ALPHA,"pass_gate":passmeta,
        "test_mean_disagreement_N":float(np.mean(np.abs(pf_test-rev_test)))
    }

out={
 "outer":outer,"row_indices":test.index.astype(int).tolist(),"pass_T":test.Pass_T.astype(int).tolist(),
 "groups":test.Group.tolist(),"y":test[N["TARGETS"]].to_numpy(float).tolist(),
 "predictions":{k:v.tolist() for k,v in P.items()},"audit":audit,"elapsed_seconds":time.time()-t0
}
wd=ROOT/"workers";wd.mkdir(exist_ok=True)
(wd/f"{outer}.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
print("V281_STACK_OUTER_COMPLETE",outer,json.dumps({"n":len(test),"elapsed":out["elapsed_seconds"]}),flush=True)
