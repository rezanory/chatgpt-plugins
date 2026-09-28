from __future__ import annotations
import itertools,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import run_layer_a as A

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"; OUT.mkdir(parents=True,exist_ok=True)
ALPHAS=(0.1,1.0,10.0,100.0)
GAMMAS=(-1.0,-0.5,-0.25,0.0,0.25,0.5,1.0)

def feat(speed,load,source,direction):
    s=np.asarray(speed,float); l=np.asarray(load,float); x=np.asarray(source,float); d=np.asarray(direction,float)
    return np.column_stack([s,l,s*s,l*l,s*l,x,d,x*d,s*d,l*d])

def aug(z,a,b):
    n=len(z)
    f=pd.DataFrame({
      "Speed":np.r_[z.Speed,z.Speed],"Load":np.r_[z.Load,z.Load],
      "source":np.r_[z[a],z[b]],"direction":np.r_[np.zeros(n),np.ones(n)],
      "target":np.r_[z[b],z[a]],"Group":np.r_[z.Group,z.Group]})
    return f

def fit_aug(tr,a,b,alpha):
    q=aug(tr,a,b)
    m=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
    m.fit(feat(q.Speed,q.Load,q.source,q.direction),q.target.to_numpy(float))
    return m

def predict_dir(m,z,src_col,direction):
    return m.predict(feat(z.Speed,z.Load,z[src_col],np.full(len(z),direction,float)))

def choose_alpha(z,a,b):
    scores=[]
    for alpha in ALPHAS:
      errs=[]
      for vg in sorted(z.Group.unique()):
        tr=z[z.Group!=vg]; va=z[z.Group==vg]
        m=fit_aug(tr,a,b,alpha)
        pb=predict_dir(m,va,a,0.0); pa=predict_dir(m,va,b,1.0)
        errs.append(0.5*(np.abs(va[b].to_numpy(float)-pb).mean()+np.abs(va[a].to_numpy(float)-pa).mean()))
      scores.append((float(np.mean(errs)),float(alpha)))
    scores.sort()
    return scores[0][1],[{"alpha":a0,"inner_bidirectional_MAE_N":s} for s,a0 in scores]

def train_oof_cycle(z,a,b,alpha):
    p_ab=np.full(len(z),np.nan);p_ba=np.full(len(z),np.nan)
    c_a=np.full(len(z),np.nan);c_b=np.full(len(z),np.nan)
    for vg in sorted(z.Group.unique()):
      tr=z[z.Group!=vg];va=z[z.Group==vg]
      m=fit_aug(tr,a,b,alpha)
      idx=va.index.to_numpy()
      pb=predict_dir(m,va,a,0.0); pa=predict_dir(m,va,b,1.0)
      p_ab[idx]=pb;p_ba[idx]=pa
      # A -> Bhat -> Acycle
      c_a[idx]=m.predict(feat(va.Speed,va.Load,pb,np.ones(len(va))))
      # B -> Ahat -> Bcycle
      c_b[idx]=m.predict(feat(va.Speed,va.Load,pa,np.zeros(len(va))))
    return p_ab,p_ba,c_a,c_b

def choose_gamma(y,p,disc):
    rows=[]
    for g in GAMMAS:
      pc=p+g*disc
      sc=float(np.mean(np.abs(y-pc)))
      rows.append((sc,float(g)))
    rows.sort()
    return rows[0][1],[{"gamma":g,"MAE_N":s} for s,g in rows]

def pair_run(w,a,b):
    z=w[["Group","Speed","Load",a,b]].dropna().reset_index(drop=True)
    n=len(z)
    pred_joint_ab=np.full(n,np.nan);pred_joint_ba=np.full(n,np.nan)
    pred_cycle_ab=np.full(n,np.nan);pred_cycle_ba=np.full(n,np.nan)
    audits=[]
    for outer in sorted(z.Group.unique()):
      tr=z[z.Group!=outer].copy().reset_index(drop=True)
      te=z[z.Group==outer].copy()
      alpha,ascores=choose_alpha(tr,a,b)
      # inner OOF cycle information for gamma selection
      pab,pba,acyc,bcyc=train_oof_cycle(tr,a,b,alpha)
      ya=tr[a].to_numpy(float); yb=tr[b].to_numpy(float)
      # A->B: discrepancy is known A minus A reconstructed from Bhat
      gab,gs_ab=choose_gamma(yb,pab,ya-acyc)
      # B->A: discrepancy is known B minus B reconstructed from Ahat
      gba,gs_ba=choose_gamma(ya,pba,yb-bcyc)
      m=fit_aug(tr,a,b,alpha)
      pb=predict_dir(m,te,a,0.0);pa=predict_dir(m,te,b,1.0)
      ac=m.predict(feat(te.Speed,te.Load,pb,np.ones(len(te))))
      bc=m.predict(feat(te.Speed,te.Load,pa,np.zeros(len(te))))
      idx=te.index.to_numpy()
      pred_joint_ab[idx]=pb;pred_joint_ba[idx]=pa
      pred_cycle_ab[idx]=pb+gab*(te[a].to_numpy(float)-ac)
      pred_cycle_ba[idx]=pa+gba*(te[b].to_numpy(float)-bc)
      audits.append({"outer_group":outer,"selected_alpha":alpha,"alpha_scores":ascores,
        "gamma_A_to_B":gab,"gamma_B_to_A":gba,"gamma_scores_A_to_B":gs_ab,"gamma_scores_B_to_A":gs_ba})
    if not all(np.isfinite(x).all() for x in (pred_joint_ab,pred_joint_ba,pred_cycle_ab,pred_cycle_ba)):
      raise RuntimeError(f"INCOMPLETE_PAIR:{a}:{b}")
    return z,pred_joint_ab,pred_joint_ba,pred_cycle_ab,pred_cycle_ba,audits

def main():
    t0=time.time(); d,_=A.decode_data();w=A.make_wide(d)
    edge=pd.read_csv(OUT/"02_base_132_directed_edges.csv").set_index("edge_id")
    rows=[];audit={}
    for i,a in enumerate(A.NODES):
      for b in A.NODES[i+1:]:
        z,jab,jba,cab,cba,au=pair_run(w,a,b)
        indep_ab=float(edge.loc[f"{a}__TO__{b}","A1_MAE_N"])
        indep_ba=float(edge.loc[f"{b}__TO__{a}","A1_MAE_N"])
        jabm=float(np.mean(np.abs(z[b]-jab))); jbam=float(np.mean(np.abs(z[a]-jba)))
        cabm=float(np.mean(np.abs(z[b]-cab))); cbam=float(np.mean(np.abs(z[a]-cba)))
        rows.append({
          "node_A":a,"node_B":b,"n_groups":len(z),
          "independent_A_to_B_MAE_N":indep_ab,"independent_B_to_A_MAE_N":indep_ba,
          "joint_A_to_B_MAE_N":jabm,"joint_B_to_A_MAE_N":jbam,
          "cycle_A_to_B_MAE_N":cabm,"cycle_B_to_A_MAE_N":cbam,
          "independent_bidirectional_mean_MAE_N":0.5*(indep_ab+indep_ba),
          "joint_bidirectional_mean_MAE_N":0.5*(jabm+jbam),
          "cycle_bidirectional_mean_MAE_N":0.5*(cabm+cbam),
          "joint_gain_vs_independent_N":0.5*(indep_ab+indep_ba-jabm-jbam),
          "cycle_gain_vs_joint_N":0.5*(jabm+jbam-cabm-cbam),
          "cycle_gain_vs_independent_N":0.5*(indep_ab+indep_ba-cabm-cbam),
          "reverse_target_used_as_forward_inference_feature":False,
        })
        audit[f"{a}__{b}"]=au
    res=pd.DataFrame(rows)
    if len(res)!=66:raise RuntimeError(f"EXPECTED_66_GOT_{len(res)}")
    res.to_csv(OUT/"04_cycle_consistency_results.csv",index=False)
    (OUT/"layerB_nested_audit.json").write_text(json.dumps(audit,indent=2,allow_nan=False),encoding="utf-8")
    summary={
      "status":"COMPLETE","pairs":66,
      "joint_better_than_independent_pairs":int((res.joint_gain_vs_independent_N>0).sum()),
      "cycle_better_than_joint_pairs":int((res.cycle_gain_vs_joint_N>0).sum()),
      "cycle_better_than_independent_pairs":int((res.cycle_gain_vs_independent_N>0).sum()),
      "mean_joint_gain_vs_independent_N":float(res.joint_gain_vs_independent_N.mean()),
      "mean_cycle_gain_vs_independent_N":float(res.cycle_gain_vs_independent_N.mean()),
      "median_cycle_gain_vs_independent_N":float(res.cycle_gain_vs_independent_N.median()),
      "method":"shared bidirectional Ridge + training-only cycle-discrepancy calibrator",
      "elapsed_seconds":time.time()-t0}
    (OUT/"LAYER_B_SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(summary))

if __name__=="__main__":main()
