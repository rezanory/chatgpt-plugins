from __future__ import annotations
import base64,gzip,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
runner=(ROOT/"runner.py").read_text(encoding="utf-8")
defs=runner[:runner.index("\ndef main(payload):\n")]
ns={"__name__":"soilbin_v27_defs"}
exec(compile(defs,str(ROOT/"runner.py"),"exec"),ns)
encoded=(ROOT/"all_features_51.csv.gz.b64").read_text(encoding="utf-8").strip()
payload={"lane":"V27","campaign":"v27-parallel-outer","model_b64":encoded}
d,h,tr,dyn=ns["load_data"](payload)
outer=sys.argv[1]
if outer not in set(h.Group): raise SystemExit("UNKNOWN_OUTER:"+outer)
t0=time.time()
train=h[h.Group!=outer].copy();test=h[h.Group==outer].copy()
cfg=dict(ns["V27_CFG"])
p23,poof,fte,foof=ns["_v23_fold"](train,test,dyn,outer,cfg)
comp=ns["_v27_components"](train,poof,foof,test,p23,fte)
dyn_outer=dyn[dyn.Group.isin(train.Group.unique())].copy()
tau,tmeta=ns["_v27_select_tau"](train,dyn_outer,cfg)
preds={
 "V23_BASE":p23.copy(),
 "TWO_STAGE_MEDIAN":p23.copy(),
 "QUANTILE_P50":p23.copy(),
 "MIXTURE_REGIMES":p23.copy(),
 "SELECTIVE_CORRECTION":p23.copy(),
}
preds["TWO_STAGE_MEDIAN"][:,1]=ns["np"].maximum(p23[:,1]+comp["corr_B"],0)
preds["QUANTILE_P50"][:,1]=ns["np"].maximum(p23[:,1]+comp["corr_C"],0)
preds["MIXTURE_REGIMES"][:,1]=ns["np"].maximum(p23[:,1]+comp["corr_D"],0)
scorr=ns["np"].where(comp["confidence"]>=tau,comp["corr_D"],0.)
preds["SELECTIVE_CORRECTION"][:,1]=ns["np"].maximum(p23[:,1]+scorr,0)
out={
 "outer":outer,"row_indices":test.index.astype(int).tolist(),"groups":test.Group.tolist(),
 "pass_T":test.Pass_T.astype(int).tolist(),"y":test[ns["TARGETS"]].to_numpy(float).tolist(),
 "predictions":{k:v.tolist() for k,v in preds.items()},
 "qlo":ns["np"].maximum(p23[:,1]+comp["q25"],0).tolist(),
 "qhi":ns["np"].maximum(p23[:,1]+comp["q75"],0).tolist(),
 "prob":comp["ptrans"].tolist(),"true_event":comp["test_event"].astype(int).tolist(),
 "confidence":comp["confidence"].tolist(),"mixcorr":comp["corr_D"].tolist(),
 "selected_tau":float(tau),"tau_inner_scores":tmeta,
 "event_threshold_N":float(comp["event_threshold_N"]),"train_event_rate":float(comp["train_event_rate"]),
 "elapsed_seconds":time.time()-t0,
}
wd=ROOT/"parallel_workers";wd.mkdir(exist_ok=True)
(wd/f"{outer}.json").write_text(json.dumps(out,indent=2,allow_nan=False),encoding="utf-8")
print("V27_OUTER_COMPLETE",outer,json.dumps({"tau":tau,"n":len(test),"elapsed_seconds":out["elapsed_seconds"]}),flush=True)
