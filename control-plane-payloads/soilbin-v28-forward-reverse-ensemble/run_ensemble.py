from __future__ import annotations
import itertools, json, math, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"soilbin-v27-bidirectional"
sys.path.insert(0,str(PREV))
import run_layer_a as A

OUT=HERE/"results"
OUT.mkdir(parents=True,exist_ok=True)
SEED=A.SEED
ALPHAS=(0.1,1.0,10.0,100.0)
WGRID=np.linspace(0,1,11)
META_ALPHA=10.0

def design(df: pd.DataFrame, source_col: str | None) -> np.ndarray:
    x=A.cond_features(df)
    if source_col is not None:
        x=np.column_stack([x,df[source_col].to_numpy(float)])
    return np.asarray(x,float)

def fit_model(df,target,source,alpha):
    m=make_pipeline(StandardScaler(),Ridge(alpha=float(alpha)))
    m.fit(design(df,source),df[target].to_numpy(float))
    return m

def group_mae(y,p,g):
    y=np.asarray(y,float);p=np.asarray(p,float);g=np.asarray(g)
    return float(np.mean([np.abs(y[g==x]-p[g==x]).mean() for x in np.unique(g)]))

def select_alpha(df,target,source):
    scored=[]
    groups=sorted(df.Group.unique())
    for a in ALPHAS:
        yy=[];pp=[];gg=[]
        for vg in groups:
            tr=df[df.Group!=vg];va=df[df.Group==vg]
            if len(tr)<3: continue
            m=fit_model(tr,target,source,a)
            yy.extend(va[target].to_numpy(float));pp.extend(m.predict(design(va,source)));gg.extend(va.Group)
        sc=group_mae(np.asarray(yy),np.asarray(pp),np.asarray(gg)) if yy else float("inf")
        scored.append((sc,float(a)))
    scored.sort(key=lambda x:(x[0],x[1]))
    return scored[0][1], [{"alpha":a,"inner_Group_MAE_N":s} for s,a in scored]

def inverse_from_reverse_model(model,df,reverse_input_col,observed_source_col,train_target_vals):
    # Reverse model predicts observed source from candidate target B.
    q=df.copy()
    q[reverse_input_col]=0.0
    y0=model.predict(design(q,reverse_input_col))
    q[reverse_input_col]=1.0
    y1=model.predict(design(q,reverse_input_col))
    slope=y1-y0
    obs=df[observed_source_col].to_numpy(float)
    vals=np.asarray(train_target_vals,float)
    lo=float(vals.min());hi=float(vals.max());span=max(hi-lo,1.0)
    lo-=0.25*span;hi+=0.25*span
    cand=np.full(len(df),np.nan)
    good=np.abs(slope)>=1e-6
    cand[good]=(obs[good]-y0[good])/slope[good]
    cand[~good]=float(np.median(vals))
    cand=np.clip(cand,lo,hi)
    return cand,np.abs(slope)

def base_predictions(train,test,src,tgt):
    af,afmeta=select_alpha(train,tgt,src)
    mf=fit_model(train,tgt,src,af)
    pf=mf.predict(design(test,src))

    ar,armeta=select_alpha(train,src,tgt)
    mr=fit_model(train,src,tgt,ar)
    pinv,slope=inverse_from_reverse_model(mr,test,tgt,src,train[tgt].to_numpy(float))
    return pf,pinv,slope,{"forward_alpha":af,"reverse_alpha":ar,
                           "forward_alpha_scores":afmeta,"reverse_alpha_scores":armeta}

def inner_base_oof(train,src,tgt):
    n=len(train)
    pf=np.full(n,np.nan);pi=np.full(n,np.nan);sl=np.full(n,np.nan)
    for vg in sorted(train.Group.unique()):
        tr=train[train.Group!=vg].copy();va=train[train.Group==vg].copy()
        # Hyperparameters are selected only inside this inner-training universe.
        p1,p2,slope,_=base_predictions(tr,va,src,tgt)
        mask=train.Group.to_numpy()==vg
        pf[mask]=p1;pi[mask]=p2;sl[mask]=slope
    if not (np.isfinite(pf).all() and np.isfinite(pi).all() and np.isfinite(sl).all()):
        raise RuntimeError(f"INCOMPLETE_INNER_OOF:{src}->{tgt}")
    return pf,pi,sl

def choose_convex(y,pf,pi,g):
    rows=[]
    for w in WGRID:
        p=w*pf+(1-w)*pi
        rows.append((group_mae(y,p,g),float(w)))
    rows.sort(key=lambda x:(x[0],-x[1]))  # prefer more forward on ties
    return rows[0][1],[{"w_forward":w,"Group_MAE_N":s} for s,w in rows]

def choose_gate(y,pf,pi,g):
    disagreement=np.abs(pf-pi)
    qs=sorted(set(float(x) for x in np.quantile(disagreement,[0,.25,.5,.75,1.])))
    candidates=[]
    for tau in qs:
        for w in WGRID:
            blend=w*pf+(1-w)*pi
            p=np.where(disagreement<=tau,blend,pf)
            candidates.append((group_mae(y,p,g),float(tau),float(w),float(np.mean(disagreement<=tau))))
    candidates.sort(key=lambda x:(x[0],-x[2],x[1]))
    best=candidates[0]
    return best[1],best[2],[{"tau":t,"w_forward":w,"Group_MAE_N":s,"coverage":c} for s,t,w,c in candidates[:25]]

def fit_meta(y,pf,pi,slope,train_df,src):
    X=np.column_stack([
        pf,pi,np.abs(pf-pi),slope,
        train_df[src].to_numpy(float),
        A.cond_features(train_df)
    ])
    m=make_pipeline(StandardScaler(),Ridge(alpha=META_ALPHA))
    m.fit(X,y)
    return m

def meta_predict(m,pf,pi,slope,test_df,src):
    X=np.column_stack([
        pf,pi,np.abs(pf-pi),slope,
        test_df[src].to_numpy(float),
        A.cond_features(test_df)
    ])
    return m.predict(X)

def run_edge(wide,sp,ss,tp,ts):
    src=A.node_id(sp,ss);tgt=A.node_id(tp,ts)
    z=wide[["Group","Speed","Load",src,tgt]].dropna().reset_index(drop=True)
    if len(z)<6: raise RuntimeError(f"TOO_SMALL:{src}->{tgt}:{len(z)}")
    methods=("FORWARD","REVERSE_INVERTED","EQUAL_AVG","CONVEX","GATED","STACKED")
    P={m:np.full(len(z),np.nan) for m in methods}
    audit=[]
    for outer in sorted(z.Group.unique()):
        train=z[z.Group!=outer].copy().reset_index(drop=True)
        test=z[z.Group==outer].copy()
        pf,pi,slope,base_meta=base_predictions(train,test,src,tgt)

        # Outer-training OOF predictions used for all ensemble choices.
        pf_oof,pi_oof,sl_oof=inner_base_oof(train,src,tgt)
        ytr=train[tgt].to_numpy(float);gtr=train.Group.to_numpy()

        wf,conv_meta=choose_convex(ytr,pf_oof,pi_oof,gtr)
        tau,wg,gate_meta=choose_gate(ytr,pf_oof,pi_oof,gtr)
        meta=fit_meta(ytr,pf_oof,pi_oof,sl_oof,train,src)

        idx=test.index.to_numpy()
        P["FORWARD"][idx]=pf
        P["REVERSE_INVERTED"][idx]=pi
        P["EQUAL_AVG"][idx]=0.5*(pf+pi)
        P["CONVEX"][idx]=wf*pf+(1-wf)*pi
        blend=wg*pf+(1-wg)*pi
        P["GATED"][idx]=np.where(np.abs(pf-pi)<=tau,blend,pf)
        P["STACKED"][idx]=meta_predict(meta,pf,pi,slope,test,src)
        audit.append({
            "outer_group":outer,**base_meta,
            "selected_convex_w_forward":wf,
            "selected_gate_tau":tau,"selected_gate_w_forward":wg,
            "test_disagreement_N":float(np.abs(pf-pi).mean()),
            "test_reverse_slope_abs":float(np.mean(slope)),
            "convex_inner":conv_meta,"gate_inner_top25":gate_meta
        })

    y=z[tgt].to_numpy(float);g=z.Group.to_numpy()
    rows=[]
    for m in methods:
        mt=A.metrics(y,P[m])
        rows.append({"method":m,**mt})
    base=next(r for r in rows if r["method"]=="FORWARD")
    oracle=np.minimum(np.abs(y-P["FORWARD"]),np.abs(y-P["REVERSE_INVERTED"]))
    oracle_mae=float(oracle.mean())
    for r in rows:
        r["gain_vs_FORWARD_N"]=float(base["MAE_N"]-r["MAE_N"])
        r["gain_vs_FORWARD_pct"]=float(100*(base["MAE_N"]-r["MAE_N"])/base["MAE_N"]) if base["MAE_N"]>0 else np.nan
    return {
        "edge_id":f"{src}__TO__{tgt}",
        "source_node":src,"target_node":tgt,
        "source_pass":sp,"target_pass":tp,
        "source_sensor":ss,"target_sensor":ts,
        "n_groups":len(z),
        "temporal_distance":tp-sp,
        "methods":rows,
        "oracle_two_candidate_MAE_N":oracle_mae,
        "oracle_headroom_vs_FORWARD_N":float(base["MAE_N"]-oracle_mae),
        "audit":audit,
    }

def summarize(results):
    flat=[]
    for r in results:
        for m in r["methods"]:
            flat.append({
                "edge_id":r["edge_id"],"source_pass":r["source_pass"],"target_pass":r["target_pass"],
                "source_sensor":r["source_sensor"],"target_sensor":r["target_sensor"],
                "n_groups":r["n_groups"],"method":m["method"],
                "MAE_N":m["MAE_N"],"RMSE_N":m["RMSE_N"],"R2":m["R2"],
                "NMAE_sd":m["NMAE_sd"],"gain_vs_FORWARD_N":m["gain_vs_FORWARD_N"],
                "gain_vs_FORWARD_pct":m["gain_vs_FORWARD_pct"],
                "oracle_headroom_vs_FORWARD_N":r["oracle_headroom_vs_FORWARD_N"],
            })
    df=pd.DataFrame(flat)
    df.to_csv(OUT/"01_forward_reverse_ensemble_edge_results.csv",index=False)

    methods=[m for m in df.method.unique() if m!="FORWARD"]
    rows=[]
    rng=np.random.default_rng(SEED+28)
    for m in methods:
        z=df[df.method==m].copy()
        gains=z.gain_vs_FORWARD_N.to_numpy(float)
        boot=gains[rng.integers(0,len(gains),size=(20000,len(gains)))].mean(axis=1)
        for subset_name,mask in [
            ("ALL_60",np.ones(len(z),dtype=bool)),
            ("LC5_TARGET",z.target_sensor.to_numpy()=="LC5"),
            ("LC5_T2_T6",(z.target_sensor.to_numpy()=="LC5") & np.isin(z.target_pass.to_numpy(),[2,6])),
            ("TARGET_T2",z.target_pass.to_numpy()==2),
            ("TARGET_T6",z.target_pass.to_numpy()==6),
        ]:
            zz=z[mask]
            if len(zz)==0: continue
            g=zz.gain_vs_FORWARD_N.to_numpy(float)
            rows.append({
                "method":m,"subset":subset_name,"n_edges":len(zz),
                "edges_improved":int((g>0).sum()),"edges_equal":int(np.isclose(g,0).sum()),
                "mean_gain_N":float(g.mean()),"median_gain_N":float(np.median(g)),
                "mean_MAE_N":float(zz.MAE_N.mean()),
                "sign_binomial_p_two_sided":float(binomtest(int((g>0).sum()),int(np.sum(~np.isclose(g,0))),.5).pvalue)
                    if np.sum(~np.isclose(g,0))>0 else 1.0,
            })
        rows[-5 if len(rows)>=5 else 0:]
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/"02_forward_reverse_ensemble_summary.csv",index=False)

    # Edge-level best legal method and oracle headroom.
    legal=df[df.method.isin(["FORWARD","EQUAL_AVG","CONVEX","GATED","STACKED"])].copy()
    best=legal.sort_values("MAE_N").groupby("edge_id").first().reset_index()
    best.to_csv(OUT/"03_best_legal_method_per_edge.csv",index=False)

    overall=summary[summary.subset=="ALL_60"].sort_values("mean_gain_N",ascending=False)
    lc5hard=summary[summary.subset=="LC5_T2_T6"].sort_values("mean_gain_N",ascending=False)
    winner=overall.iloc[0].to_dict()
    hardwinner=lc5hard.iloc[0].to_dict()
    out={
        "schema":"soilbin.v28.forward-reverse-ensemble.v1",
        "status":"COMPLETE",
        "data_sha256":A.EXPECTED_SHA,
        "true_forward_edges":60,
        "leakage_guard":{
            "actual_future_target_used_at_forward_inference":False,
            "reverse_model_inverted_from_known_source_and_conditions_only":True,
            "ensemble_weights_selected_outer_training_only":True,
            "gate_threshold_selected_outer_training_only":True,
            "stacker_trained_on_outer_training_OOF_predictions_only":True,
            "random_split_used":False,
        },
        "methods":["FORWARD","REVERSE_INVERTED","EQUAL_AVG","CONVEX","GATED","STACKED"],
        "overall_ranked":overall.to_dict("records"),
        "hard_LC5_T2_T6_ranked":lc5hard.to_dict("records"),
        "overall_best":winner,
        "hard_LC5_T2_T6_best":hardwinner,
        "best_method_edge_counts":best.method.value_counts().to_dict(),
        "oracle_mean_headroom_N":float(df[df.method=="FORWARD"].oracle_headroom_vs_FORWARD_N.mean()),
        "interpretation_guard":"Reverse inversion is a legal inverse-model estimate; it never receives the actual future target at forward inference."
    }
    (OUT/"RESULTS.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    return out

def main():
    t0=time.time()
    d,_=A.decode_data();wide=A.make_wide(d)
    results=[]
    for sp in range(1,7):
        for tp in range(sp+1,7):
            for ss in ("LC1","LC5"):
                for ts in ("LC1","LC5"):
                    results.append(run_edge(wide,sp,ss,tp,ts))
                    print("EDGE_DONE",results[-1]["edge_id"],flush=True)
    if len(results)!=60: raise RuntimeError(f"EXPECTED_60_FORWARD_EDGES:{len(results)}")
    out=summarize(results)
    out["elapsed_seconds"]=time.time()-t0
    (OUT/"RESULTS.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    (OUT/"04_full_audit.json").write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print("V28_ENSEMBLE_COMPLETE",json.dumps({"best":out["overall_best"],"hard":out["hard_LC5_T2_T6_best"],"elapsed":out["elapsed_seconds"]},ensure_ascii=False),flush=True)

if __name__=="__main__": main()
