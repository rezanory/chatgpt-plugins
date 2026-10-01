integrity_receipt = {
    "schema": "m07.external.rsna_pediatric.inference_adapt.pre_inference_integrity.v1",
    "status": "PASS_PREINFERENCE_INTEGRITY",
    "resolution": IMAGE_SIZE,
    "manifest_sha256": manifest_meta["manifest_sha256"],
    "counts": manifest_meta["counts"],
    "external_exact_duplicate_image_sha": 0,
    "internal_external_exact_sha_overlap": manifest_meta["exact_internal_external_sha_overlap"],
    "training_performed": False,
    "hpo_performed": False,
    "external_threshold_tuning": False,
    "external_calibration_fitting": False,
    "lung_focused_inference": True,
    "test_time_adaptation": True,
}
(OUT / "M07_RSNA_R224_INFERENCE_ADAPT_PREINFERENCE_INTEGRITY_RECEIPT.json").write_text(
    json.dumps(safe_json(integrity_receipt), indent=2, ensure_ascii=False), encoding="utf-8"
)

LUNG_CROP_FRACTIONS=(0.90,0.84)
TTA_LR=1e-4
TTA_GRAD_CLIP_NORM=1.0

def decode_method(path,label,mode="canonical"):
    data=tf.io.read_file(path)
    image=tf.io.decode_image(data,channels=3,expand_animations=False)
    image.set_shape([None,None,3])
    image=tf.cast(image,tf.float32)
    if mode=="lung_crop_90":
        image=tf.image.central_crop(image,central_fraction=LUNG_CROP_FRACTIONS[0])
    elif mode=="lung_crop_84":
        image=tf.image.central_crop(image,central_fraction=LUNG_CROP_FRACTIONS[1])
    elif mode!="canonical":
        raise ValueError("UNKNOWN_INFERENCE_MODE="+str(mode))
    image=tf.image.resize_with_pad(image,IMAGE_SIZE,IMAGE_SIZE,method="bilinear",antialias=True)
    return tf.clip_by_value(image,0.0,255.0),tf.cast(label,tf.float32)

def build_method_dataset(frame,batch_size,mode="canonical"):
    opts=tf.data.Options(); opts.experimental_deterministic=True
    ds=tf.data.Dataset.from_tensor_slices(
        (frame.filepath.astype(str).to_numpy(),frame.label.astype(np.float32).to_numpy())
    ).with_options(opts)
    ds=ds.map(lambda p,y: decode_method(p,y,mode),num_parallel_calls=AUTOTUNE,deterministic=True)
    return ds.batch(int(batch_size),drop_remainder=False).prefetch(AUTOTUNE)

def episodic_headnorm_tta(model,frame,batch_size):
    layer=model.get_layer("head_norm")
    variables=list(layer.trainable_variables)
    if len(variables)!=2:
        raise RuntimeError("HEAD_NORM_VARIABLE_COUNT="+str(len(variables)))
    anchors=[tf.identity(v) for v in variables]
    preds=[]; ent0=[]; ent1=[]; norms=[]
    for images,_ in build_method_dataset(frame,batch_size,"canonical"):
        for v,a in zip(variables,anchors): v.assign(a)
        with tf.GradientTape() as tape:
            p=tf.clip_by_value(tf.cast(model(images,training=False),tf.float32),1e-6,1.0-1e-6)
            e=-(p*tf.math.log(p)+(1.0-p)*tf.math.log(1.0-p))
            loss=tf.reduce_mean(e)
        grads=tape.gradient(loss,variables)
        if any(g is None for g in grads): raise RuntimeError("HEAD_NORM_TTA_GRADIENT_MISSING")
        grads,norm=tf.clip_by_global_norm(grads,TTA_GRAD_CLIP_NORM)
        for v,g in zip(variables,grads):
            v.assign_sub(tf.cast(TTA_LR,v.dtype)*tf.cast(g,v.dtype))
        q=tf.clip_by_value(tf.cast(model(images,training=False),tf.float32),1e-6,1.0-1e-6)
        e2=-(q*tf.math.log(q)+(1.0-q)*tf.math.log(1.0-q))
        preds.append(q.numpy().reshape(-1))
        ent0.append(float(loss.numpy())); ent1.append(float(tf.reduce_mean(e2).numpy())); norms.append(float(norm.numpy()))
    for v,a in zip(variables,anchors): v.assign(a)
    out=np.concatenate(preds).astype(float)
    if len(out)!=len(frame) or not np.isfinite(out).all(): raise RuntimeError("EPISODIC_TTA_PREDICTIONS_INVALID")
    return out,{
        "method":"episodic_one_step_entropy_minimization",
        "adapted_variables":["head_norm/gamma","head_norm/beta"],
        "learning_rate":TTA_LR,
        "gradient_clip_norm":TTA_GRAD_CLIP_NORM,
        "batches":len(ent0),
        "mean_entropy_before":float(np.mean(ent0)),
        "mean_entropy_after":float(np.mean(ent1)),
        "mean_gradient_global_norm":float(np.mean(norms)),
        "persistent_weight_change":False,
        "labels_used_for_adaptation":False,
    }

def aggregate(fold_probs,thresholds):
    P=np.vstack(fold_probs)
    N=np.vstack([logit_np(P[i])-logit_np(thresholds[i+1]) for i in range(5)])
    score=N.mean(axis=0); prob=P.mean(axis=0); pred=(score>=0).astype(int)
    return prob,score,pred,N

def frame_arrays(base,prob,score,pred):
    f=base.copy()
    f["mean_probability"]=np.asarray(prob,float)
    f["normalized_ensemble_score"]=np.asarray(score,float)
    f["prediction_primary_normalized"]=np.asarray(pred,int)
    return f

print("CGP_PHASE:M07_RSNA_R224_INFERENCE_ADAPT_INFERENCE",flush=True)
canon_f=[]; c90_f=[]; c84_f=[]; adapt_f=[]; diagnostics={}
for fold in range(1,6):
    print(f"CGP_PHASE:M07_RSNA_R224_INFERENCE_ADAPT_FOLD {fold}/5",flush=True)
    tf.keras.backend.clear_session(); gc.collect()
    model=build_m07(params,SEED+fold); model.load_weights(weights[fold])
    canon=model.predict(build_method_dataset(manifest,EVAL_BATCH_SIZE,"canonical"),verbose=0).reshape(-1).astype(float)
    c90=model.predict(build_method_dataset(manifest,EVAL_BATCH_SIZE,"lung_crop_90"),verbose=0).reshape(-1).astype(float)
    c84=model.predict(build_method_dataset(manifest,EVAL_BATCH_SIZE,"lung_crop_84"),verbose=0).reshape(-1).astype(float)
    ada,diag=episodic_headnorm_tta(model,manifest,EVAL_BATCH_SIZE)
    canon_f.append(canon); c90_f.append(c90); c84_f.append(c84); adapt_f.append(ada); diagnostics[str(fold)]=diag
    del model; tf.keras.backend.clear_session(); gc.collect()

cp,cs,cy,cn=aggregate(canon_f,thresholds)
p90,s90,y90,n90=aggregate(c90_f,thresholds)
p84,s84,y84,n84=aggregate(c84_f,thresholds)
ap,ass,ay,an=aggregate(adapt_f,thresholds)
ce_prob=np.vstack([np.vstack(c90_f),np.vstack(c84_f)]).mean(axis=0)
ce_score=np.vstack([n90,n84]).mean(axis=0); ce_pred=(ce_score>=0).astype(int)

canonical_frame=frame_arrays(manifest,cp,cs,cy)
crop90_frame=frame_arrays(manifest,p90,s90,y90)
crop84_frame=frame_arrays(manifest,p84,s84,y84)
crop_equal_frame=frame_arrays(manifest,ce_prob,ce_score,ce_pred)
adapt_frame=frame_arrays(manifest,ap,ass,ay)

canonical_eval=evaluate(canonical_frame,"R224_CANONICAL_REPLAY_EXPANDED",RSNA_POSITIVE_LABEL)
crop90_eval=evaluate(crop90_frame,"R224_LUNG_CROP_90_EXPANDED",RSNA_POSITIVE_LABEL)
crop84_eval=evaluate(crop84_frame,"R224_LUNG_CROP_84_EXPANDED",RSNA_POSITIVE_LABEL)
crop_equal_eval=evaluate(crop_equal_frame,"R224_LUNG_CROP_EQUAL_EXPANDED",RSNA_POSITIVE_LABEL)
adapt_eval=evaluate(adapt_frame,"R224_EPISODIC_HEADNORM_TTA_EXPANDED",RSNA_POSITIVE_LABEL)

expected={"tn":400,"fp":178,"fn":37,"tp":484}
observed={k:int(canonical_eval["metrics"][k]) for k in expected}
if observed!=expected: raise RuntimeError("R224_CANONICAL_REPLAY_DRIFT="+json.dumps({"expected":expected,"actual":observed},sort_keys=True))

def delta(candidate,baseline):
    cm=candidate["metrics"]; bm=baseline["metrics"]
    keys=("accuracy","balanced_accuracy","precision_positive","recall_positive","f1_positive","mcc","auroc","auprc_positive")
    out={k:float(cm[k]-bm[k]) for k in keys}
    out.update({k:int(cm[k]-bm[k]) for k in ("tn","fp","fn","tp")})
    return out

primary=manifest.primary_peds_lt10.to_numpy(bool)
primary_results={}
for name,frame in (
    ("canonical",canonical_frame),("lung_crop_90",crop90_frame),("lung_crop_84",crop84_frame),
    ("lung_crop_equal",crop_equal_frame),("episodic_headnorm_tta",adapt_frame)
):
    primary_results[name]=evaluate(frame.loc[primary].copy(),"R224_"+name.upper()+"_PRIMARY_LT10",RSNA_POSITIVE_LABEL)

manifest["canonical_probability"]=cp
manifest["lung_crop_equal_probability"]=ce_prob
manifest["episodic_tta_probability"]=ap
manifest["canonical_score"]=cs
manifest["lung_crop_equal_score"]=ce_score
manifest["episodic_tta_score"]=ass
manifest.to_csv(OUT/"M07_RSNA_R224_INFERENCE_ADAPT_PREDICTIONS.csv",index=False)

report={
    "schema":"m07.external.rsna_pediatric.r224_inference_adapt.v1",
    "status":"PASS_INFERENCE_ADAPT_ANALYSIS",
    "analysis_classification":"SECONDARY_EXTERNAL_EXPLORATORY_INFERENCE_ADAPTATION",
    "analysis_is_pure_external_validation":False,
    "raw_external_result_preserved":True,
    "resolution":IMAGE_SIZE,
    "source_state":STATE_HANDLE,
    "split_fingerprint":EXPECTED_SPLIT,
    "hpo_recipe_fingerprint":EXPECTED_RECIPE,
    "external_dataset":RSNA_DATASET_REF,
    "external_manifest_sha256":EXPECTED_RSNA_MANIFEST_SHA256,
    "training_performed":False,
    "hpo_performed":False,
    "external_threshold_tuning":False,
    "external_calibration_fitting":False,
    "canonical_replay_expanded":canonical_eval,
    "lung_focused_inference":{
        "method":"deterministic_center_crop_before_canonical_resize",
        "crop_fractions":list(LUNG_CROP_FRACTIONS),"labels_used":False,"model_weights_changed":False,
        "crop90_expanded":crop90_eval,"crop84_expanded":crop84_eval,"equal_mean_expanded":crop_equal_eval,
        "delta_equal_minus_canonical":delta(crop_equal_eval,canonical_eval),
    },
    "test_time_adaptation":{
        "method":"episodic_one_step_head_layernorm_entropy_minimization",
        "labels_used":False,"adaptation_is_transductive":True,"persistent_model_weight_change":False,
        "fold_diagnostics":diagnostics,"expanded":adapt_eval,"delta_minus_canonical":delta(adapt_eval,canonical_eval),
    },
    "primary_lt10":primary_results,
    "limitations":[
        "External positive endpoint is adjudicated Lung Opacity rather than exact pneumonia diagnosis.",
        "These are secondary exploratory analyses and do not replace the untouched raw external result.",
        "Lung-focused inference uses deterministic centered crops, not a learned lung segmentation model.",
        "Test-time adaptation uses no external labels and changes only head LayerNorm gamma/beta for one batch-local entropy step before immediate reset.",
    ],
}
(OUT/"M07_RSNA_R224_INFERENCE_ADAPT_REPORT.json").write_text(json.dumps(safe_json(report),indent=2,ensure_ascii=False),encoding="utf-8")
hashes={p.relative_to(OUT).as_posix():sha256_file(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
receipt={
    "schema":"m07.external.rsna_pediatric.r224_inference_adapt.terminal.v1",
    "status":"PASS_INFERENCE_ADAPT_ANALYSIS","resolution":IMAGE_SIZE,
    "expanded_n":len(manifest),"expanded_patients":int(manifest.patient_id.nunique()),"primary_n":len(primary_manifest),
    "training_performed":False,"hpo_performed":False,"external_threshold_tuning":False,"external_calibration_fitting":False,
    "lung_focused_inference":True,"test_time_adaptation":True,"adaptation_labels_used":False,
    "persistent_model_weight_change":False,"raw_external_result_preserved":True,"canonical_replay_confusion":observed,
    "source_state":STATE_HANDLE,"split_fingerprint":EXPECTED_SPLIT,"hpo_recipe_fingerprint":EXPECTED_RECIPE,
    "external_dataset":RSNA_DATASET_REF,"external_manifest_sha256":EXPECTED_RSNA_MANIFEST_SHA256,"artifact_sha256":hashes,
}
body=json.dumps(safe_json(receipt),sort_keys=True,separators=(",",":")).encode()
receipt["receipt_sha256"]=hashlib.sha256(body).hexdigest()
(OUT/"M07_RSNA_R224_INFERENCE_ADAPT_TERMINAL_RECEIPT.json").write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding="utf-8")
zip_path=Path(shutil.make_archive(str(WORK/"M07_RSNA_PEDIATRIC_R224_INFERENCE_ADAPT_V1_COMPLETE"),"zip",root_dir=OUT))
print("CGP_PHASE:M07_RSNA_R224_INFERENCE_ADAPT_COMPLETE",flush=True)
print(json.dumps({"status":"PASS_INFERENCE_ADAPT_ANALYSIS","canonical":canonical_eval["metrics"],"lung_crop_equal":crop_equal_eval["metrics"],"episodic_headnorm_tta":adapt_eval["metrics"],"zip":str(zip_path)},sort_keys=True),flush=True)
