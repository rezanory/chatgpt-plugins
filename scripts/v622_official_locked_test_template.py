from __future__ import annotations
import base64, hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path
import numpy as np

PROJECT='PNEUMONIA V6.2.2'
FINAL_TEST_TOKEN='I_HAVE_FROZEN_THE_CHAMPION'
FINAL_MANIFEST_SHA='2313023819c9eec669d835b99dcc2524b03bb0da236c2ec8f98c4e1221948414'
RECONCILIATION_SHA='__RECONCILIATION_SHA__'
ARTIFACT_DATASET='trickermark/pneumonia-v6-2-2-finalization-artifacts'
RAW_DATASET='paultimothymooney/chest-xray-pneumonia'
SELECTED=['M06__convnext_tiny','M06__densenet121','M06__resnet50v2']
CONFIG_SHA={
'M06__convnext_tiny':'167cb7b1d17b3978f2b4f47f55e315a1d4fae293006d897d25d13b5074dadcea',
'M06__densenet121':'c9b3107715849a633d90689902298c110bbb7c9c1bc1c8a6ac9ff1693f55386e',
'M06__resnet50v2':'dce7845aa7c07ccb913cded8e9ff368e7061f3a43723ded21f8a94dba9fab697'}
SOURCE_ZIP_B64='__SOURCE_ZIP_B64__'
INPUT=Path('/kaggle/input')
WORK=Path('/kaggle/working/PNEUMONIA_V62_2_OFFICIAL_LOCKED_TEST_WORK')
OUT=Path('/kaggle/working/PNEUMONIA_V62_2_OFFICIAL_LOCKED_TEST')
WORK.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
    return h.hexdigest()

def unique(name:str)->Path:
    xs=[p for p in INPUT.rglob(name) if p.is_file()]
    if len(xs)!=1: raise RuntimeError(f'{name}: expected exactly one mounted file, found {len(xs)}')
    return xs[0]

def exclusive_json(path:Path,payload:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(payload,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
        f.flush(); os.fsync(f.fileno())

# Immutable gates are verified before benchmark labels are opened.
mpath=unique('FINAL_FREEZE_MANIFEST.json'); msha=sha(mpath)
if msha!=FINAL_MANIFEST_SHA: raise RuntimeError(f'FINAL_FREEZE_MANIFEST SHA drift: {msha}')
manifest=json.loads(mpath.read_text(encoding='utf-8'))
if manifest.get('project')!=PROJECT or manifest.get('stage')!='FINAL_FREEZE_MANIFEST' or manifest.get('status')!='FROZEN_IN_KAGGLE': raise RuntimeError('final freeze identity/status mismatch')
if manifest.get('selected_candidate_ids')!=SELECTED or manifest.get('selected_artifact_count')!=6: raise RuntimeError('frozen selected set mismatch')
if manifest.get('locked_test_used') is not False or manifest.get('external_data_used') is not False: raise RuntimeError('final freeze already consumed protected evidence')

rpath=unique('FINAL_FREEZE_RECONCILIATION.json'); rsha=sha(rpath)
if rsha!=RECONCILIATION_SHA: raise RuntimeError(f'reconciliation SHA drift: {rsha}')
recon=json.loads(rpath.read_text(encoding='utf-8'))
if recon.get('project')!=PROJECT or recon.get('stage')!='FINAL_FREEZE_RECONCILIATION' or recon.get('status')!='PASS' or recon.get('independent_reconciliation') is not True: raise RuntimeError('reconciliation identity/status mismatch')
if recon.get('final_freeze_manifest_sha256')!=FINAL_MANIFEST_SHA or recon.get('selection_locked') is not True or recon.get('policy_frozen') is not True: raise RuntimeError('reconciliation freeze binding mismatch')
if len(recon.get('verified_artifacts') or [])!=6: raise RuntimeError('reconciliation did not independently verify six artifacts')
if recon.get('locked_test_used') is not False or recon.get('external_data_used') is not False: raise RuntimeError('reconciliation indicates protected evidence consumption')

rows={(x.get('candidate_id'),x.get('role')):x for x in (manifest.get('artifacts') or [])}
files={}
for cid in SELECTED:
    cfg=unique(cid+'__config.json'); ck=unique(cid+'__final_selected.keras')
    if sha(cfg)!=CONFIG_SHA[cid]: raise RuntimeError(cid+': canonical config SHA mismatch')
    for role,p in [('config',cfg),('checkpoint',ck)]:
        row=rows.get((cid,role))
        if not row or sha(p)!=str(row.get('sha256','')).lower() or p.stat().st_size!=int(row.get('bytes',-1)): raise RuntimeError(f'{cid} {role}: final manifest artifact binding mismatch')
    files[cid]=(cfg,ck)

fit=manifest.get('deployment_fit') or {}
calibration=fit.get('calibration') or []; weights=np.asarray(fit.get('weights') or [],dtype=float); threshold=float(fit.get('threshold'))
if len(calibration)!=3 or weights.shape!=(3,) or not np.isclose(weights.sum(),1.0,atol=1e-9): raise RuntimeError('invalid frozen deployment_fit')

# One-shot authorization is written before test_locked is loaded.
from datetime import datetime, timezone
auth={
 'schema_version':1,'status':'AUTHORIZED_ONCE','project':PROJECT,'execution_mode':'FINAL_LOCKBOX',
 'authorized_at':datetime.now(timezone.utc).isoformat(),'confirmation_token_sha256':hashlib.sha256(FINAL_TEST_TOKEN.encode()).hexdigest(),
 'final_freeze_manifest_sha256':FINAL_MANIFEST_SHA,'final_freeze_reconciliation_sha256':RECONCILIATION_SHA,
 'artifact_dataset':ARTIFACT_DATASET,'internal_benchmark_dataset':RAW_DATASET,'selected_candidate_ids':SELECTED,
 'selection_locked':True,'policy_frozen':True,'reselection_after_results_allowed':False,
 'lockbox_access_started':True,'lockbox_results_available':False,
 'freeze_sha256':FINAL_MANIFEST_SHA,'dataset_identity':RAW_DATASET}
raw=json.dumps(auth,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8'); auth['record_sha256']=hashlib.sha256(raw).hexdigest()
auth_path=OUT/'FINAL_LOCKBOX_CONSUMPTION_RECORD.json'; exclusive_json(auth_path,auth)

# Exact W16 source package is embedded by the bridge after source-SHA verification.
source_zip=WORK/'source_python.zip'; source_zip.write_bytes(base64.b64decode(SOURCE_ZIP_B64))
source_dir=WORK/'source'; source_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(source_zip) as z: z.extractall(source_dir)
PROJECT_CODE=source_dir/'SOURCE_PYTHON'; sys.path.insert(0,str(PROJECT_CODE))
import pandas as pd
import tensorflow as tf
import pneumonia_runner_base as runner
from auto_ensemble_selection import apply_platt_calibrator
from project_model_loader import load_project_model_for_inference
from final_lockbox_guard import build_result_manifest, write_completion_record

# Reconstruct W16's deterministic split, then open the 624-image lockbox exactly once.
def looks(p:Path): return all((p/x).is_dir() for x in ('train','val','test'))
cands=[p for p in [INPUT]+[p for p in INPUT.rglob('*') if p.is_dir()] if looks(p) and '__MACOSX' not in p.parts]
top=[p for p in cands if not any(p!=q and q in p.parents for q in cands)]
if len(top)!=1: raise RuntimeError(f'Expected one raw train/val/test dataset root, found {top}')
effective=WORK/'effective_split'
subprocess.run([sys.executable,str(PROJECT_CODE/'build_effective_split.py'),'--data-root',str(top[0]),'--output-root',str(effective),'--target-val-total','600','--seed','42','--mode','symlink','--overwrite'],check=True)
_,_,test_df=runner.load_manifests(effective,smoke_test=False,seed=42,include_test=True,require_locked_manifest=True)
if test_df is None or len(test_df)!=624: raise RuntimeError(f'Official locked benchmark must contain exactly 624 images; found {0 if test_df is None else len(test_df)}')
labels=test_df['label'].to_numpy(dtype=int)

raw_probs=[]; signatures=[]
for cid in SELECTED:
    _,ck=files[cid]
    row=rows[(cid,'checkpoint')]
    model,sig=load_project_model_for_inference(checkpoint_path=ck,model_id='M06',checkpoint_sha256=str(row['sha256']),expected_image_size=224)
    ds=runner.build_dataset(test_df,224,16,False,42,'none',0.2,2,1)
    p=runner.predict_dataset(model,ds)
    if len(p)!=624: raise RuntimeError(cid+': prediction count mismatch')
    raw_probs.append(np.asarray(p,dtype=float)); sig['candidate_id']=cid; signatures.append(sig)
    tf.keras.backend.clear_session()
raw_matrix=np.stack(raw_probs,axis=1)
cal=np.empty_like(raw_matrix)
for i,params in enumerate(calibration): cal[:,i]=apply_platt_calibrator(raw_matrix[:,i],params)
ensemble=np.clip(cal@weights,1e-7,1-1e-7)
pred=(ensemble>=threshold).astype(int)
metrics=runner.calculate_metrics(labels,ensemble,threshold)
metrics.update({'status':'PASS','project':PROJECT,'stage':'OFFICIAL_LOCKED_TEST','n_samples':624,'selected_candidate_ids':SELECTED,'weights':[float(x) for x in weights],'threshold':threshold,'threshold_source':'FROZEN_DEVELOPMENT_ONLY','calibration_source':'FROZEN_DEVELOPMENT_ONLY','final_freeze_manifest_sha256':FINAL_MANIFEST_SHA,'final_freeze_reconciliation_sha256':RECONCILIATION_SHA,'reselection_after_results_allowed':False,'external_data_used':False})
(OUT/'OFFICIAL_LOCKED_TEST_METRICS.json').write_text(json.dumps(metrics,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
(OUT/'MODEL_SIGNATURES.json').write_text(json.dumps(signatures,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
pred_df=test_df[['sample_id','class_name','label']].copy()
for i,cid in enumerate(SELECTED): pred_df['raw_'+cid]=raw_matrix[:,i]; pred_df['calibrated_'+cid]=cal[:,i]
pred_df['ensemble_probability_pneumonia']=ensemble; pred_df['ensemble_prediction']=pred
pred_df.to_csv(OUT/'OFFICIAL_LOCKED_TEST_PREDICTIONS.csv',index=False)

result_manifest=build_result_manifest(output_root=OUT,authorization_record=auth_path,output_record=OUT/'FINAL_LOCKBOX_EVALUATION_MANIFEST.json')
completion=write_completion_record(authorization_record=auth_path,result_manifest=OUT/'FINAL_LOCKBOX_EVALUATION_MANIFEST.json',output_record=OUT/'FINAL_LOCKBOX_COMPLETION_RECORD.json')
receipt={'status':'PASS','project':PROJECT,'stage':'OFFICIAL_LOCKED_TEST','n_samples':624,'metrics':metrics,'authorization_record_sha256':auth['record_sha256'],'evaluation_manifest_sha256':result_manifest['manifest_sha256'],'completion_sha256':completion['completion_sha256'],'reselection_after_results_allowed':False,'next_permitted_stage':'EXTERNAL_VALIDATION'}
(OUT/'OFFICIAL_LOCKED_TEST_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','stage':'OFFICIAL_LOCKED_TEST','n_samples':624,'next_permitted_stage':'EXTERNAL_VALIDATION'},indent=2))
