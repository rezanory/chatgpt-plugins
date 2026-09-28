from __future__ import annotations
import itertools, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_layer_a as A

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"; OUT.mkdir(parents=True,exist_ok=True)
ALPHA=10.0  # predeclared fixed capacity; no test-side selection

def source_cols(subset, mode):
    cols=[]
    for p in subset:
        if mode=="LC1_ONLY": cols.append(A.node_id(p,"LC1"))
        elif mode=="LC5_ONLY": cols.append(A.node_id(p,"LC5"))
        else: cols.extend([A.node_id(p,"LC1"),A.node_id(p,"LC5")])
    return cols

def X(df, cols):
    base=A.cond_features(df)
    if cols: base=np.column_stack([base,df[cols].to_numpy(float)])
    return base

def oof(z,target,cols):
    pred=np.full(len(z),np.nan)
    for gg in sorted(z.Group.unique()):
        tr=z[z.Group!=gg]; te=z[z.Group==gg]
        m=make_pipeline(StandardScaler(),Ridge(alpha=ALPHA))
        m.fit(X(tr,cols),tr[target].to_numpy(float))
        pred[te.index.to_numpy()]=m.predict(X(te,cols))
    if not np.isfinite(pred).all(): raise RuntimeError("INCOMPLETE_OOF")
    return pred

def task_kind(subset,tp):
    if all(p<tp for p in subset): return "FORECAST"
    if all(p>tp for p in subset): return "RECONSTRUCTION"
    return "SMOOTHING"

def run_task(w,subset,mode,tp,ts):
    cols=source_cols(subset,mode); target=A.node_id(tp,ts)
    need=["Group","Speed","Load",target]+cols
    z=w[need].dropna().reset_index(drop=True)
    if len(z)<6:
        return None
    p=oof(z,target,cols); p0=oof(z,target,[])
    m=A.metrics(z[target],p); m0=A.metrics(z[target],p0)
    kind=task_kind(subset,tp)
    return {
        "task_id":f"{kind}__SRC_{'-'.join(map(str,subset))}__{mode}__TO_T{tp}_{ts}",
        "kind":kind,"source_passes":"+".join(map(str,subset)),
        "source_count":len(subset),"sensor_mode":mode,
        "target_pass":tp,"target_sensor":ts,"target_node":target,
        "feature_count":len(cols),"n":len(z),"n_groups":z.Group.nunique(),
        "model":f"Ridge_fixed_alpha_{ALPHA:g}","random_split":False,
        **m,
        "conditions_only_MAE_N":m0["MAE_N"],
        "gain_vs_conditions_N":m0["MAE_N"]-m["MAE_N"],
        "gain_vs_conditions_pct":100*(m0["MAE_N"]-m["MAE_N"])/m0["MAE_N"] if m0["MAE_N"]>0 else np.nan,
    }

def memory_depth(rows):
    out=[]
    for ts in ("LC1","LC5"):
      for mode in ("LC1_ONLY","LC5_ONLY","LC1_LC5"):
        for tp in range(2,7):
          for depth in range(1,tp):
            subset=tuple(range(tp-depth,tp))
            key="+".join(map(str,subset))
            m=(rows.target_sensor==ts)&(rows.sensor_mode==mode)&(rows.target_pass==tp)&(rows.kind=="FORECAST")&(rows.source_passes==key)
            if m.any():
                r=rows[m].iloc[0]
                out.append({"target_pass":tp,"target_sensor":ts,"sensor_mode":mode,
                            "history_depth":depth,"history_passes":key,"n_groups":int(r.n_groups),
                            "MAE_N":float(r.MAE_N),"NMAE_sd":float(r.NMAE_sd),
                            "gain_vs_conditions_N":float(r.gain_vs_conditions_N)})
    return pd.DataFrame(out)

def fusion(rows):
    keys=["kind","source_passes","target_pass","target_sensor"]
    rec=[]
    for k,z in rows.groupby(keys,dropna=False):
        zz=z.set_index("sensor_mode")
        if not all(x in zz.index for x in ("LC1_ONLY","LC5_ONLY","LC1_LC5")): continue
        joint=float(zz.loc["LC1_LC5","MAE_N"]); a=float(zz.loc["LC1_ONLY","MAE_N"]); b=float(zz.loc["LC5_ONLY","MAE_N"])
        best_single=min(a,b)
        rec.append(dict(zip(keys,k),LC1_only_MAE_N=a,LC5_only_MAE_N=b,LC1_LC5_MAE_N=joint,
            fusion_gain_vs_best_single_N=best_single-joint,
            fusion_gain_vs_best_single_pct=100*(best_single-joint)/best_single if best_single>0 else np.nan,
            n_groups=int(zz.loc["LC1_LC5","n_groups"])))
    return pd.DataFrame(rec)

def missing_recon(rows):
    rec=[]
    for tp in range(1,7):
      past=tuple(range(1,tp)); future=tuple(range(tp+1,7)); mixed=tuple(p for p in range(1,7) if p!=tp)
      specs=[]
      if past: specs.append(("PAST_ONLY_FORECAST",past))
      if future: specs.append(("FUTURE_ONLY_RECONSTRUCTION",future))
      if past and future: specs.append(("PAST_FUTURE_SMOOTHING",mixed))
      for label,sub in specs:
        key="+".join(map(str,sub))
        for mode in ("LC1_ONLY","LC5_ONLY","LC1_LC5"):
          for ts in ("LC1","LC5"):
            m=(rows.source_passes==key)&(rows.sensor_mode==mode)&(rows.target_pass==tp)&(rows.target_sensor==ts)
            if m.any():
              r=rows[m].iloc[0]
              rec.append({"target_pass":tp,"target_sensor":ts,"reconstruction_mode":label,
                "source_passes":key,"sensor_mode":mode,"n_groups":int(r.n_groups),
                "MAE_N":float(r.MAE_N),"NMAE_sd":float(r.NMAE_sd),
                "gain_vs_conditions_N":float(r.gain_vs_conditions_N)})
    return pd.DataFrame(rec)

def main():
    t0=time.time(); d,_=A.decode_data(); w=A.make_wide(d)
    rows=[]; skipped=[]
    for tp in range(1,7):
      others=[p for p in range(1,7) if p!=tp]
      for r in range(1,6):
        for subset in itertools.combinations(others,r):
          for mode in ("LC1_ONLY","LC5_ONLY","LC1_LC5"):
            for ts in ("LC1","LC5"):
              x=run_task(w,subset,mode,tp,ts)
              if x is None:
                skipped.append((subset,mode,tp,ts))
              else: rows.append(x)
    df=pd.DataFrame(rows)
    if len(df)!=1116:
        raise RuntimeError(f"EXPECTED_1116_EXECUTABLE_TASKS_GOT_{len(df)}_SKIPPED_{len(skipped)}")
    if not (df.n_groups>=6).all(): raise RuntimeError("GROUP_COUNT_GUARD")
    df.to_csv(OUT/"06_multi_history_1116_tasks.csv",index=False)

    mem=memory_depth(df); mem.to_csv(OUT/"07_memory_depth_results.csv",index=False)
    fus=fusion(df); fus.to_csv(OUT/"08_sensor_fusion_results.csv",index=False)
    mis=missing_recon(df); mis.to_csv(OUT/"09_missing_pass_reconstruction.csv",index=False)

    forward=df[df.kind=="FORECAST"]; rev=df[df.kind=="RECONSTRUCTION"]; smooth=df[df.kind=="SMOOTHING"]
    summary={
      "status":"COMPLETE","tasks_executed":len(df),"skipped":len(skipped),
      "kind_counts":df.kind.value_counts().to_dict(),
      "n_groups_distribution":df.n_groups.value_counts().sort_index().to_dict(),
      "fixed_model":f"Ridge(alpha={ALPHA})","hyperparameter_selected_on_test":False,
      "forward_median_MAE_N":float(forward.MAE_N.median()),
      "reconstruction_median_MAE_N":float(rev.MAE_N.median()),
      "smoothing_median_MAE_N":float(smooth.MAE_N.median()),
      "LC1_target_median_MAE_N":float(df[df.target_sensor=="LC1"].MAE_N.median()),
      "LC5_target_median_MAE_N":float(df[df.target_sensor=="LC5"].MAE_N.median()),
      "fusion_tasks":len(fus),"fusion_helped_count":int((fus.fusion_gain_vs_best_single_N>0).sum()),
      "memory_depth_rows":len(mem),"missing_reconstruction_rows":len(mis),
      "elapsed_seconds":time.time()-t0
    }
    (OUT/"LAYER_D_SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(summary))

if __name__=="__main__": main()
