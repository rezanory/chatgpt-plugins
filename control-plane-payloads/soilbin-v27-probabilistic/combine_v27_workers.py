from __future__ import annotations
import json,base64,gzip,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
runner=(ROOT/"runner.py").read_text(encoding="utf-8")
defs=runner[:runner.index("\ndef main(payload):\n")]
ns={"__name__":"soilbin_v27_defs"}
exec(compile(defs,str(ROOT/"runner.py"),"exec"),ns)
encoded=(ROOT/"all_features_51.csv.gz.b64").read_text(encoding="utf-8").strip()
payload={"lane":"V27","campaign":"v27-parallel-outer","model_b64":encoded}
d,h,tr,dyn=ns["load_data"](payload)
variants=("V23_BASE","TWO_STAGE_MEDIAN","QUANTILE_P50","MIXTURE_REGIMES","SELECTIVE_CORRECTION")
P={v:np.full((len(h),2),np.nan) for v in variants}
qlo=np.full(len(h),np.nan);qhi=np.full(len(h),np.nan);prob=np.full(len(h),np.nan)
true_event=np.full(len(h),np.nan);confidence=np.full(len(h),np.nan);mixcorr=np.full(len(h),np.nan)
audits=[];worker_elapsed={}
wd=ROOT/"parallel_workers"
for outer in sorted(h.Group.unique()):
    f=wd/f"{outer}.json"
    if not f.exists(): raise RuntimeError("MISSING_WORKER:"+outer)
    x=json.loads(f.read_text(encoding="utf-8"))
    idx=np.asarray(x["row_indices"],int)
    for v in variants: P[v][idx]=np.asarray(x["predictions"][v],float)
    qlo[idx]=np.asarray(x["qlo"],float);qhi[idx]=np.asarray(x["qhi"],float)
    prob[idx]=np.asarray(x["prob"],float);true_event[idx]=np.asarray(x["true_event"],float)
    confidence[idx]=np.asarray(x["confidence"],float);mixcorr[idx]=np.asarray(x["mixcorr"],float)
    audits.append({"outer":outer,"selected_tau":x["selected_tau"],"tau_inner_scores":x["tau_inner_scores"],
        "event_threshold_N":x["event_threshold_N"],"train_event_rate":x["train_event_rate"],
        "test_mean_transition_probability":float(np.mean(x["prob"])),
        "test_selective_coverage":float(np.mean(np.asarray(x["confidence"])>=x["selected_tau"]))})
    worker_elapsed[outer]=x["elapsed_seconds"]
for v in variants:
    if not np.isfinite(P[v]).all(): raise RuntimeError("INCOMPLETE_PRED:"+v)
if not all(np.isfinite(x).all() for x in (qlo,qhi,prob,true_event,confidence,mixcorr)):
    raise RuntimeError("INCOMPLETE_DIAGNOSTICS")
y=h[ns["TARGETS"]].to_numpy(float);g=h.Group.to_numpy()
base_metrics=ns["force_metrics"](y,P["V23_BASE"],g)
if abs(base_metrics["joint_Group_MAE_N"]-ns["V23_LOCK"])>1e-6: raise RuntimeError("V23_LOCK_FAIL")
if abs(base_metrics["LC1_Group_MAE_N"]-ns["V23_LC1_LOCK"])>1e-6 or abs(base_metrics["LC5_Group_MAE_N"]-ns["V23_LC5_LOCK"])>1e-6:
    raise RuntimeError("CHANNEL_LOCK_FAIL")
summary=[]
for v in variants:
    m=ns["force_metrics"](y,P[v],g)
    row={"variant":v,**m}
    row["joint_improvement_vs_V23_N"]=float(base_metrics["joint_Group_MAE_N"]-m["joint_Group_MAE_N"])
    row["LC5_improvement_vs_V23_N"]=float(base_metrics["LC5_Group_MAE_N"]-m["LC5_Group_MAE_N"])
    row["LC5_improvement_vs_V23_pct"]=float(100*(base_metrics["LC5_Group_MAE_N"]-m["LC5_Group_MAE_N"])/base_metrics["LC5_Group_MAE_N"])
    row["paired_LC5_vs_V23"]=ns["_lc5_pair_report"](y[:,1],P[v][:,1],P["V23_BASE"][:,1],g)
    row["LC5_pass_MAE_N"]={str(int(t)):float(np.mean(np.abs(y[h.Pass_T.to_numpy()==t,1]-P[v][h.Pass_T.to_numpy()==t,1])))
        for t in sorted(h.Pass_T.unique())}
    summary.append(row)
coverage=float(np.mean((y[:,1]>=qlo)&(y[:,1]<=qhi)))
brier=float(ns["brier_score_loss"](true_event.astype(int),prob))
auroc=float(ns["roc_auc_score"](true_event.astype(int),prob)) if len(np.unique(true_event))==2 else None
curves=[]
for tau in ns["V27_TAU_GRID"]:
    pp=P["V23_BASE"][:,1]+np.where(confidence>=tau,mixcorr,0.)
    curves.append({"confidence_threshold":float(tau),"coverage":float(np.mean(confidence>=tau)),
        "LC5_Group_MAE_N":ns["gmae1"](y[:,1],pp,g)})
champion=min(summary,key=lambda x:x["joint_Group_MAE_N"])
result={"schema":"soilbin.v27.probabilistic.v1","status":"COMPLETE","lane":"V27","campaign":"v27-parallel-outer",
    "question":"Can probabilistic occurrence/amplitude separation improve LC5 without changing frozen V23 LC1?",
    "arms":list(variants),"summary":summary,"champion":champion,
    "transition_classifier":{"Brier":brier,"AUROC":auroc,
      "event_definition":"outer-training 60th percentile of |LC5_target-prev_LC5|; threshold never learned from test",
      "OOF_event_rate":float(np.mean(true_event))},
    "quantile_diagnostics":{"P25_P75_nominal_coverage":0.50,"observed_coverage":coverage,
      "mean_interval_width_N":float(np.mean(qhi-qlo))},
    "selective_coverage_error_curve":curves,"audit":audits,"worker_elapsed_seconds":worker_elapsed,
    "source_sha256":ns["EXPECTED_DATA"],"post_lock_exploratory":True,"does_not_supersede_v5":True,
    "design_guards":{"LC1_frozen_to_V23_for_all_arms":True,
      "V23_residual_targets_cross_fitted_inside_outer_training":True,
      "selective_tau_nested_training_only":True,
      "quantile_levels_predeclared":[.25,.5,.75],
      "mixture_weight_is_predicted_transition_probability":True,
      "current_or_future_target_in_inference_features":False,
      "parallelization_changed_scientific_folds":False},
    "guardrails":{"random_split_used":False,"targets_modified":False,
      "current_or_future_target_used_for_prediction":False,"outer_speed_load_group_held_out":True}}
out=ROOT/"parallel_V27_RESULTS.json"
out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
print("V27_PARALLEL_FANIN_COMPLETE",json.dumps({"champion":champion["variant"],"joint":champion["joint_Group_MAE_N"],
 "LC5":champion["LC5_Group_MAE_N"],"brier":brier,"auroc":auroc,"coverage":coverage}),flush=True)
