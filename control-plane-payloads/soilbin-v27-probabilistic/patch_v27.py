from pathlib import Path
p=Path(r"C:\Users\Radlina\workspace\soilbin-v27-bidirectional-worktree\control-plane-payloads\soilbin-v27-probabilistic\runner.py")
s=p.read_text(encoding="utf-8")
s=s.replace("from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor",
            "from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor")
marker="\ndef main(payload):\n"
if marker not in s:
    raise SystemExit("MAIN_MARKER_MISSING")
block=r'''
# ---------- V27 LC5 two-stage probabilistic transition model ----------
V27_CFG={"arm":"A_QLOG","mode":"REWARD_ONLY","law":"STATE_MAG","shape":"POWER","policy":"T6_PRIOR","channel":"SHARED_STATIC"}
V27_TAU_GRID=(0.0,0.2,0.4,0.6,0.8)

def _v27_state_X(hrows,dynfeat):
    b=hrows[FORCE_BASE].to_numpy(float)
    d=np.asarray(dynfeat,float)
    p1=np.maximum(hrows.prev_LC1.to_numpy(float),0)
    p5=np.maximum(hrows.prev_LC5.to_numpy(float),0)
    ratio=p5/np.maximum(p1,1.0)
    extra=np.column_stack([np.log1p(p1),np.log1p(p5),ratio,np.abs(p5-p1)])
    return np.column_stack([b,d,extra])

def _v27_logit_prob(X,y,Xte):
    X=np.asarray(X,float);Xte=np.asarray(Xte,float);y=np.asarray(y,int)
    prior=float(np.mean(y)) if len(y) else .5
    if len(np.unique(y))<2:
        return np.full(len(Xte),prior)
    mu=X.mean(axis=0);sd=X.std(axis=0);sd=np.where(sd<1e-8,1.,sd)
    clf=LogisticRegression(C=.25,class_weight="balanced",max_iter=2000,random_state=SEED)
    clf.fit((X-mu)/sd,y)
    return clf.predict_proba((Xte-mu)/sd)[:,1]

def _v27_qpred(X,y,g,Xte,q,seedadd=0):
    X=np.asarray(X,float);y=np.asarray(y,float);Xte=np.asarray(Xte,float);g=np.asarray(g)
    if len(y)<6 or len(np.unique(g))<3:
        return np.full(len(Xte),float(np.quantile(y,q)) if len(y) else 0.)
    m=GradientBoostingRegressor(loss="quantile",alpha=float(q),n_estimators=90,
        learning_rate=.05,max_depth=2,min_samples_leaf=3,random_state=SEED+seedadd)
    m.fit(X,y,sample_weight=group_weights(g))
    pr=m.predict(Xte)
    lo,hi=np.quantile(y,[.02,.98])
    return np.clip(pr,lo,hi)

def _v27_components(train,poof,foof,test,p23,fte):
    Xtr=_v27_state_X(train,foof);Xte=_v27_state_X(test,fte)
    y5=train[TARGETS[1]].to_numpy(float)
    resid=y5-np.asarray(poof,float)[:,1]
    g=train.Group.to_numpy()
    jump=np.abs(y5-train.prev_LC5.to_numpy(float))
    event_threshold=float(np.quantile(jump,.60))
    event=(jump>=event_threshold).astype(int)
    ptrans=_v27_logit_prob(Xtr,event,Xte)
    stable=event==0; trans=event==1
    qevent=_v27_qpred(Xtr[trans],resid[trans],g[trans],Xte,.5,2701) if np.any(trans) else np.zeros(len(test))
    qstable=_v27_qpred(Xtr[stable],resid[stable],g[stable],Xte,.5,2702) if np.any(stable) else np.zeros(len(test))
    qa=_v27_qpred(Xtr,resid,g,Xte,.25,2711)
    qm=_v27_qpred(Xtr,resid,g,Xte,.50,2712)
    qz=_v27_qpred(Xtr,resid,g,Xte,.75,2713)
    qs=np.sort(np.column_stack([qa,qm,qz]),axis=1)
    qa,qm,qz=qs[:,0],qs[:,1],qs[:,2]
    corr_b=np.where(ptrans>=.5,qevent,0.)
    corr_c=qm
    corr_d=(1-ptrans)*qstable+ptrans*qevent
    conf=2*np.abs(ptrans-.5)
    test_jump=np.abs(test[TARGETS[1]].to_numpy(float)-test.prev_LC5.to_numpy(float))
    test_event=(test_jump>=event_threshold).astype(int)
    return {"ptrans":ptrans,"confidence":conf,"corr_B":corr_b,"corr_C":corr_c,"corr_D":corr_d,
            "q25":qa,"q50":qm,"q75":qz,"event_threshold_N":event_threshold,
            "train_event_rate":float(event.mean()),"test_event":test_event}

def _v27_select_tau(train,dyn_outer,cfg):
    y=[];base=[];corr=[];conf=[];groups=[]
    for vg in sorted(train.Group.unique()):
        tr=train[train.Group!=vg].copy();va=train[train.Group==vg].copy()
        pva,poof,fte,foof=_v23_fold(tr,va,dyn_outer,vg,cfg)
        comp=_v27_components(tr,poof,foof,va,pva,fte)
        y.extend(va[TARGETS[1]].to_numpy(float))
        base.extend(pva[:,1]);corr.extend(comp["corr_D"]);conf.extend(comp["confidence"]);groups.extend(va.Group)
    y=np.asarray(y,float);base=np.asarray(base,float);corr=np.asarray(corr,float);conf=np.asarray(conf,float);groups=np.asarray(groups)
    rows=[]
    for tau in V27_TAU_GRID:
        p=base+np.where(conf>=tau,corr,0.)
        sc=gmae1(y,p,groups)
        rows.append((float(sc),float(tau),float(np.mean(conf>=tau))))
    rows.sort(key=lambda x:(x[0],x[1]))
    return rows[0][1],[{"tau":t,"inner_LC5_Group_MAE_N":s,"coverage":c} for s,t,c in rows]

def run_v27(d,h,dyn,payload):
    variants=("V23_BASE","TWO_STAGE_MEDIAN","QUANTILE_P50","MIXTURE_REGIMES","SELECTIVE_CORRECTION")
    P={v:np.full((len(h),2),np.nan) for v in variants}
    qlo=np.full(len(h),np.nan);qhi=np.full(len(h),np.nan);prob=np.full(len(h),np.nan)
    true_event=np.full(len(h),np.nan);confidence=np.full(len(h),np.nan);mixcorr=np.full(len(h),np.nan)
    audits=[];cfg=dict(V27_CFG)
    for outer in sorted(h.Group.unique()):
        train=h[h.Group!=outer].copy();test=h[h.Group==outer].copy()
        p23,poof,fte,foof=_v23_fold(train,test,dyn,outer,cfg)
        comp=_v27_components(train,poof,foof,test,p23,fte)
        dyn_outer=dyn[dyn.Group.isin(train.Group.unique())].copy()
        tau,tmeta=_v27_select_tau(train,dyn_outer,cfg)
        idx=test.index.to_numpy()
        for v in variants:
            P[v][idx,0]=p23[:,0]
        P["V23_BASE"][idx,1]=p23[:,1]
        P["TWO_STAGE_MEDIAN"][idx,1]=np.maximum(p23[:,1]+comp["corr_B"],0)
        P["QUANTILE_P50"][idx,1]=np.maximum(p23[:,1]+comp["corr_C"],0)
        P["MIXTURE_REGIMES"][idx,1]=np.maximum(p23[:,1]+comp["corr_D"],0)
        scorr=np.where(comp["confidence"]>=tau,comp["corr_D"],0.)
        P["SELECTIVE_CORRECTION"][idx,1]=np.maximum(p23[:,1]+scorr,0)
        qlo[idx]=np.maximum(p23[:,1]+comp["q25"],0)
        qhi[idx]=np.maximum(p23[:,1]+comp["q75"],0)
        prob[idx]=comp["ptrans"];true_event[idx]=comp["test_event"]
        confidence[idx]=comp["confidence"];mixcorr[idx]=comp["corr_D"]
        audits.append({"outer":outer,"selected_tau":tau,"tau_inner_scores":tmeta,
            "event_threshold_N":comp["event_threshold_N"],"train_event_rate":comp["train_event_rate"],
            "test_mean_transition_probability":float(np.mean(comp["ptrans"])),
            "test_selective_coverage":float(np.mean(comp["confidence"]>=tau))})
    y=h[TARGETS].to_numpy(float);g=h.Group.to_numpy()
    base_metrics=force_metrics(y,P["V23_BASE"],g)
    if abs(base_metrics["joint_Group_MAE_N"]-V23_LOCK)>1e-6:
        raise RuntimeError("V27_V23_REPRODUCTION_FAILED:"+str(base_metrics["joint_Group_MAE_N"]))
    if abs(base_metrics["LC1_Group_MAE_N"]-V23_LC1_LOCK)>1e-6 or abs(base_metrics["LC5_Group_MAE_N"]-V23_LC5_LOCK)>1e-6:
        raise RuntimeError("V27_CHANNEL_REPRODUCTION_FAILED")
    summary=[]
    for v in variants:
        m=force_metrics(y,P[v],g)
        row={"variant":v,**m}
        row["joint_improvement_vs_V23_N"]=float(base_metrics["joint_Group_MAE_N"]-m["joint_Group_MAE_N"])
        row["LC5_improvement_vs_V23_N"]=float(base_metrics["LC5_Group_MAE_N"]-m["LC5_Group_MAE_N"])
        row["LC5_improvement_vs_V23_pct"]=float(100*(base_metrics["LC5_Group_MAE_N"]-m["LC5_Group_MAE_N"])/base_metrics["LC5_Group_MAE_N"])
        row["paired_LC5_vs_V23"]=_lc5_pair_report(y[:,1],P[v][:,1],P["V23_BASE"][:,1],g)
        row["LC5_pass_MAE_N"]={str(int(t)):float(np.mean(np.abs(y[h.Pass_T.to_numpy()==t,1]-P[v][h.Pass_T.to_numpy()==t,1])))
            for t in sorted(h.Pass_T.unique())}
        summary.append(row)
    coverage=float(np.mean((y[:,1]>=qlo)&(y[:,1]<=qhi)))
    brier=float(brier_score_loss(true_event.astype(int),prob))
    auroc=float(roc_auc_score(true_event.astype(int),prob)) if len(np.unique(true_event))==2 else None
    curves=[]
    for tau in V27_TAU_GRID:
        pp=P["V23_BASE"][:,1]+np.where(confidence>=tau,mixcorr,0.)
        curves.append({"confidence_threshold":float(tau),"coverage":float(np.mean(confidence>=tau)),
            "LC5_Group_MAE_N":gmae1(y[:,1],pp,g)})
    champion=min(summary,key=lambda x:x["joint_Group_MAE_N"])
    return {"lane":"V27","question":"Can probabilistic occurrence/amplitude separation improve LC5 without changing frozen V23 LC1?",
        "arms":list(variants),"summary":summary,"champion":champion,
        "transition_classifier":{"Brier":brier,"AUROC":auroc,
            "event_definition":"outer-training 60th percentile of |LC5_target-prev_LC5|; threshold never learned from test",
            "OOF_event_rate":float(np.mean(true_event))},
        "quantile_diagnostics":{"P25_P75_nominal_coverage":0.50,"observed_coverage":coverage,
            "mean_interval_width_N":float(np.mean(qhi-qlo))},
        "selective_coverage_error_curve":curves,"audit":audits,
        "design_guards":{"LC1_frozen_to_V23_for_all_arms":True,
            "V23_residual_targets_cross_fitted_inside_outer_training":True,
            "selective_tau_nested_training_only":True,
            "quantile_levels_predeclared":[.25,.5,.75],
            "mixture_weight_is_predicted_transition_probability":True,
            "current_or_future_target_in_inference_features":False}}
'''
s=s.replace(marker,"\n"+block+marker)
s=s.replace('elif lane=="V26":result=run_v26(d,h,dyn,payload)\n    else:raise KeyError(lane)',
            'elif lane=="V26":result=run_v26(d,h,dyn,payload)\n    elif lane=="V27":result=run_v27(d,h,dyn,payload)\n    else:raise KeyError(lane)')
s=s.replace('("soilbin.v26.lc5-transition-aware.v1" if lane=="V26" else "soilbin.v17-v25.parallel.v1")',
            '("soilbin.v27.probabilistic.v1" if lane=="V27" else ("soilbin.v26.lc5-transition-aware.v1" if lane=="V26" else "soilbin.v17-v25.parallel.v1"))')
s=s.replace('("soilbin.v26.lc5-transition-aware.v1" if globals().get("PAYLOAD",{}).get("lane")=="V26" else "soilbin.v17-v25.parallel.v1")',
            '("soilbin.v27.probabilistic.v1" if globals().get("PAYLOAD",{}).get("lane")=="V27" else ("soilbin.v26.lc5-transition-aware.v1" if globals().get("PAYLOAD",{}).get("lane")=="V26" else "soilbin.v17-v25.parallel.v1"))')
p.write_text(s,encoding="utf-8")
print("PATCHED_V27",len(s))
