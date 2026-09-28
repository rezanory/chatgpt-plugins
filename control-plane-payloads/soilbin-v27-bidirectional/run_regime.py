from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import run_layer_a as A

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"; OUT.mkdir(parents=True,exist_ok=True)

def main():
    t0=time.time(); d,_=A.decode_data(); w=A.make_wide(d)
    edges=pd.read_csv(OUT/"02_base_132_directed_edges.csv")
    spec=pd.read_csv(ROOT/"spectral_pass_summary_from_library.csv")
    rows=[]
    for p in range(1,7):
      zp=d[d.Pass_T==p]
      for s in ("LC1","LC5"):
        y=zp[A.TGT[s]].to_numpy(float)
        src=edges[(edges.source_pass==p)&(edges.source_sensor==s)]
        tgt=edges[(edges.target_pass==p)&(edges.target_sensor==s)]
        ss=spec[(spec.Pass_T==p)&(spec.sensor==s)].iloc[0]
        # actual adjacent pass change, only groups where both passes exist
        changes=[]
        if p>1:
          a=A.node_id(p-1,s);b=A.node_id(p,s)
          z=w[[a,b]].dropna()
          changes=np.abs(z[b]-z[a]).to_list()
        rows.append({
          "pass":p,"sensor":s,"n":len(y),"peak_mean_N":float(np.mean(y)),
          "peak_sd_N":float(np.std(y,ddof=1)) if len(y)>1 else np.nan,
          "peak_cv":float(np.std(y,ddof=1)/np.mean(np.abs(y))) if len(y)>1 and np.mean(np.abs(y))>0 else np.nan,
          "mean_abs_change_from_previous_N":float(np.mean(changes)) if changes else np.nan,
          "sd_abs_change_from_previous_N":float(np.std(changes,ddof=1)) if len(changes)>1 else np.nan,
          "source_edge_mean_gain_N":float(src.gain_MAE_N.mean()),
          "source_edge_positive_gain_fraction":float((src.gain_MAE_N>0).mean()),
          "target_edge_mean_MAE_N":float(tgt.A1_MAE_N.mean()),
          "target_edge_mean_NMAE_sd":float(tgt.A1_NMAE_sd.mean()),
          "hf10_energy_ratio_mean":float(ss.hf10_mean),
          "hf20_energy_ratio_mean":float(ss.hf20_mean),
          "spectral_centroid_mean_Hz":float(ss.centroid_mean_Hz),
          "timeseries_signal_sd_mean_N":float(ss.signal_sd_mean_N),
          "timeseries_signal_mean_abs_mean_N":float(ss.signal_mean_abs_mean_N),
        })
    out=pd.DataFrame(rows)

    # Six pass-level vectors pooling both sensors; descriptive clustering only.
    pass_rows=[]
    for p in range(1,7):
      z=out[out["pass"]==p].set_index("sensor")
      pass_rows.append({
        "pass":p,
        "LC1_peak_cv":z.loc["LC1","peak_cv"],"LC5_peak_cv":z.loc["LC5","peak_cv"],
        "LC1_target_NMAE":z.loc["LC1","target_edge_mean_NMAE_sd"],"LC5_target_NMAE":z.loc["LC5","target_edge_mean_NMAE_sd"],
        "LC1_hf10":z.loc["LC1","hf10_energy_ratio_mean"],"LC5_hf10":z.loc["LC5","hf10_energy_ratio_mean"],
        "LC1_change":0 if pd.isna(z.loc["LC1","mean_abs_change_from_previous_N"]) else z.loc["LC1","mean_abs_change_from_previous_N"],
        "LC5_change":0 if pd.isna(z.loc["LC5","mean_abs_change_from_previous_N"]) else z.loc["LC5","mean_abs_change_from_previous_N"],
    })
    pf=pd.DataFrame(pass_rows)
    feature_cols=[c for c in pf.columns if c!="pass"]
    X=StandardScaler().fit_transform(pf[feature_cols])
    cand=[]
    labels_by_k={}
    for k in (2,3,4):
      km=KMeans(n_clusters=k,n_init=50,random_state=A.SEED).fit(X)
      lab=km.labels_; labels_by_k[k]=lab
      sil=float(silhouette_score(X,lab)) if 1<len(np.unique(lab))<len(lab) else -1.
      cand.append((sil,k))
    cand.sort(reverse=True)
    best_k=cand[0][1]; labels=labels_by_k[best_k]
    pf["cluster"]=labels
    cmap=dict(zip(pf["pass"],pf["cluster"]))
    out["pass_cluster"]=out["pass"].map(cmap)

    # simple change-point scan on LC5 adjacent change magnitude across passes 2..6
    lc5=out[out.sensor=="LC5"].sort_values("pass")
    seq=lc5[lc5["pass"]>=2]["mean_abs_change_from_previous_N"].to_numpy(float)
    passes=lc5[lc5["pass"]>=2]["pass"].to_numpy(int)
    cps=[]
    for j in range(1,len(seq)):
      left,right=seq[:j],seq[j:]
      sse=float(((left-left.mean())**2).sum()+((right-right.mean())**2).sum())
      cps.append((sse,int(passes[j-1])))
    cps.sort()
    best_cp=cps[0][1] if cps else None

    out.to_csv(OUT/"10_regime_analysis.csv",index=False)
    pf.to_csv(OUT/"10_regime_pass_clusters.csv",index=False)
    summary={
      "status":"COMPLETE","rows":len(out),"best_descriptive_cluster_k":best_k,
      "silhouette_candidates":[{"k":k,"silhouette":s} for s,k in cand],
      "pass_clusters":{str(int(p)):int(c) for p,c in zip(pf["pass"],pf["cluster"])},
      "LC5_best_change_point_after_pass":best_cp,
      "LC5_change_point_SSE_candidates":[{"after_pass":p,"SSE":s} for s,p in cps],
      "spectral_source":"timeseries_long_lc1_lc5.csv from Library",
      "spectral_method":"per-run FFT after mean removal; energy ratio >=10Hz and >=20Hz, spectral centroid",
      "inference_guard":"Six pass-level observations make clustering/change-point evidence descriptive, not population-level proof.",
      "elapsed_seconds":time.time()-t0}
    (OUT/"REGIME_SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(summary))

if __name__=="__main__":main()
