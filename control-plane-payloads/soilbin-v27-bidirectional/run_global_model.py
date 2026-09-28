from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import run_layer_a as A

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"; OUT.mkdir(parents=True,exist_ok=True)
SEED=A.SEED

def pooled(w):
    rows=[]
    for _,r in w.iterrows():
        for sp in range(1,7):
          for ss in ("LC1","LC5"):
            src=A.node_id(sp,ss)
            if pd.isna(r[src]): continue
            for tp in range(1,7):
              for ts in ("LC1","LC5"):
                if sp==tp and ss==ts: continue
                tgt=A.node_id(tp,ts)
                if pd.isna(r[tgt]): continue
                rows.append({
                  "Group":r.Group,"Speed":float(r.Speed),"Load":float(r.Load),
                  "source_node":src,"target_node":tgt,"source_pass":sp,"target_pass":tp,
                  "source_sensor":ss,"target_sensor":ts,
                  "source_response":float(r[src]),"target_response":float(r[tgt]),
                })
    return pd.DataFrame(rows)

def feats(z):
    s=z.Speed.to_numpy(float); l=z.Load.to_numpy(float)
    sp=z.source_pass.to_numpy(float); tp=z.target_pass.to_numpy(float)
    ss=(z.source_sensor=="LC5").to_numpy(float); ts=(z.target_sensor=="LC5").to_numpy(float)
    same=(z.source_sensor==z.target_sensor).to_numpy(float)
    sr=z.source_response.to_numpy(float)
    dist=tp-sp
    return np.column_stack([
      s,l,s*s,l*l,s*l,sr,np.log1p(np.maximum(sr,0)),
      sp,tp,dist,np.abs(dist),ss,ts,same,(sp==tp).astype(float),
      sr*ts,sr*ss,sr*same,dist*ts
    ])

def weights(g):
    g=np.asarray(g); u,c=np.unique(g,return_counts=True); m=dict(zip(u,c))
    w=np.asarray([1/m[x] for x in g],float); return w/w.mean()

def fit_pred(kind,tr,te):
    X=feats(tr); y=tr.target_response.to_numpy(float); sw=weights(tr.Group)
    if kind=="RIDGE":
        m=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
        m.fit(X,y,ridge__sample_weight=sw)
    elif kind=="EXTRATREES":
        m=ExtraTreesRegressor(n_estimators=500,max_depth=6,min_samples_leaf=2,max_features=.75,
            random_state=SEED,n_jobs=1)
        m.fit(X,y,sample_weight=sw)
    else: raise KeyError(kind)
    return m.predict(feats(te))

def group_mae(y,p,g):
    return float(np.mean([np.mean(np.abs(y[g==x]-p[g==x])) for x in np.unique(g)]))

def choose(train):
    candidates=["RIDGE","EXTRATREES"]; scores=[]
    for kind in candidates:
      yy=[];pp=[];gg=[]
      for vg in sorted(train.Group.unique()):
        tr=train[train.Group!=vg]; va=train[train.Group==vg]
        pr=fit_pred(kind,tr,va)
        yy.extend(va.target_response);pp.extend(pr);gg.extend(va.Group)
      sc=group_mae(np.asarray(yy,float),np.asarray(pp,float),np.asarray(gg))
      scores.append((sc,kind))
    scores.sort()
    return scores[0][1],[{"model":k,"inner_Group_MAE_N":float(s)} for s,k in scores]

def main():
    t0=time.time(); d,_=A.decode_data(); w=A.make_wide(d); q=pooled(w)
    preds=np.full(len(q),np.nan); audit=[]
    for outer in sorted(q.Group.unique()):
      tr=q[q.Group!=outer]; te=q[q.Group==outer]
      kind,scores=choose(tr)
      preds[te.index.to_numpy()]=fit_pred(kind,tr,te)
      audit.append({"outer_group":outer,"selected":kind,"candidate_scores":scores,
                    "train_rows":len(tr),"test_rows":len(te)})
    if not np.isfinite(preds).all(): raise RuntimeError("GLOBAL_OOF_INCOMPLETE")
    q["global_pred"]=preds
    edge=A.pd.read_csv(OUT/"02_base_132_directed_edges.csv")
    lookup=edge.set_index("edge_id")
    rows=[]
    for (src,tgt),z in q.groupby(["source_node","target_node"]):
      y=z.target_response.to_numpy(float);p=z.global_pred.to_numpy(float)
      m=A.metrics(y,p); eid=f"{src}__TO__{tgt}"
      es=float(lookup.loc[eid,"A1_MAE_N"])
      rows.append({"edge_id":eid,"source_node":src,"target_node":tgt,"n":len(z),"n_groups":z.Group.nunique(),
          "global_MAE_N":m["MAE_N"],"global_RMSE_N":m["RMSE_N"],"global_R2":m["R2"],
          "global_NMAE_sd":m["NMAE_sd"],"edge_specific_A1_MAE_N":es,
          "global_improvement_vs_edge_specific_N":es-m["MAE_N"]})
    res=pd.DataFrame(rows).sort_values("edge_id")
    if len(res)!=132: raise RuntimeError(f"EXPECTED_132_GLOBAL_EDGES_{len(res)}")
    res.to_csv(OUT/"05_global_directed_model_results.csv",index=False)
    summary={
      "status":"COMPLETE","pooled_rows":len(q),"edges":len(res),
      "outer_groups":q.Group.nunique(),
      "mean_global_MAE_N":float(res.global_MAE_N.mean()),
      "median_global_MAE_N":float(res.global_MAE_N.median()),
      "mean_edge_specific_MAE_N":float(res.edge_specific_A1_MAE_N.mean()),
      "global_better_edges":int((res.global_improvement_vs_edge_specific_N>0).sum()),
      "global_worse_edges":int((res.global_improvement_vs_edge_specific_N<0).sum()),
      "mean_improvement_vs_edge_specific_N":float(res.global_improvement_vs_edge_specific_N.mean()),
      "selection_audit":audit,"elapsed_seconds":time.time()-t0
    }
    (OUT/"LAYER_C_SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(summary))

if __name__=="__main__":main()
