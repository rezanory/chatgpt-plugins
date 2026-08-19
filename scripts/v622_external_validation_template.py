from __future__ import annotations
import base64, hashlib, json, os, subprocess, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT='PNEUMONIA V6.2.2'
FINAL_MANIFEST_SHA='2313023819c9eec669d835b99dcc2524b03bb0da236c2ec8f98c4e1221948414'
RECONCILIATION_SHA='99a319c8d397342aaefd3c15e0eb8f968a973738f55ef97925a2a16da8c485d8'
LOCKED_COMPLETION_SHA='2ecc67f4268f812604e51db5b46d2ece36f9de12ff7baf3cf0bbbc8df4a58ada'
LOCKED_EVALUATION_MANIFEST_SHA='a87ac30bef825f01657d9705293f431d58f59e2d325f7d96ba1026586c59ec25'
ARTIFACT_DATASET='trickermark/pneumonia-v6-2-2-finalization-artifacts'
DEV_DATASET='paultimothymooney/chest-xray-pneumonia'
CHEXPERT_DATASET='ashery/chexpert'
NIH_DATASET='nih-chest-xrays/sample'
SELECTED=['M06__convnext_tiny','M06__densenet121','M06__resnet50v2']
CONFIG_SHA={
'M06__convnext_tiny':'167cb7b1d17b3978f2b4f47f55e315a1d4fae293006d897d25d13b5074dadcea',
'M06__densenet121':'c9b3107715849a633d90689902298c110bbb7c9c1bc1c8a6ac9ff1693f55386e',
'M06__resnet50v2':'dce7845aa7c07ccb913cded8e9ff368e7061f3a43723ded21f8a94dba9fab697'}
SOURCE_ZIP_B64='__SOURCE_ZIP_B64__'
INPUT=Path('/kaggle/input')
WORK=Path('/kaggle/working/PNEUMONIA_V62_2_EXTERNAL_VALIDATION_WORK')
OUT=Path('/kaggle/working/PNEUMONIA_V62_2_EXTERNAL_VALIDATION')
WORK.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def unique(name:str)->Path:
    matches=[p for p in INPUT.rglob(name) if p.is_file()]
    if len(matches)!=1: raise RuntimeError(f'{name}: expected exactly one mounted file, found {len(matches)}')
    return matches[0]

mpath=unique('FINAL_FREEZE_MANIFEST.json')
rpath=unique('FINAL_FREEZE_RECONCILIATION.json')
lpath=unique('OFFICIAL_LOCKED_TEST_RECEIPT.json')
if sha(mpath)!=FINAL_MANIFEST_SHA: raise RuntimeError('FINAL_FREEZE_MANIFEST SHA drift')
if sha(rpath)!=RECONCILIATION_SHA: raise RuntimeError('FINAL_FREEZE_RECONCILIATION SHA drift')
manifest=json.loads(mpath.read_text(encoding='utf-8'))
recon=json.loads(rpath.read_text(encoding='utf-8'))
locked=json.loads(lpath.read_text(encoding='utf-8'))
if manifest.get('project')!=PROJECT or manifest.get('stage')!='FINAL_FREEZE_MANIFEST': raise RuntimeError('final freeze identity drift')
if recon.get('project')!=PROJECT or recon.get('stage')!='FINAL_FREEZE_RECONCILIATION' or recon.get('status')!='PASS': raise RuntimeError('reconciliation identity drift')
if locked.get('project')!=PROJECT or locked.get('stage')!='OFFICIAL_LOCKED_TEST' or locked.get('status')!='PASS': raise RuntimeError('locked-test receipt identity/status drift')
if locked.get('completion_sha256')!=LOCKED_COMPLETION_SHA or locked.get('evaluation_manifest_sha256')!=LOCKED_EVALUATION_MANIFEST_SHA: raise RuntimeError('locked-test immutable completion binding mismatch')
if int(locked.get('n_samples',-1))!=624 or locked.get('reselection_after_results_allowed') is not False or locked.get('next_permitted_stage')!='EXTERNAL_VALIDATION': raise RuntimeError('locked-test governance gate mismatch')
if manifest.get('selected_candidate_ids')!=SELECTED or recon.get('selected_candidate_ids')!=SELECTED: raise RuntimeError('selected ensemble drift')
if recon.get('final_freeze_manifest_sha256')!=FINAL_MANIFEST_SHA: raise RuntimeError('reconciliation/final-freeze binding mismatch')

rows={(x.get('candidate_id'),x.get('role')):x for x in (manifest.get('artifacts') or [])}
files={}
for cid in SELECTED:
    cfg=unique(cid+'__config.json'); ck=unique(cid+'__final_selected.keras')
    if sha(cfg)!=CONFIG_SHA[cid]: raise RuntimeError(cid+': canonical config SHA mismatch')
    for role,p in [('config',cfg),('checkpoint',ck)]:
        row=rows.get((cid,role))
        if not row or sha(p)!=str(row.get('sha256','')).lower() or p.stat().st_size!=int(row.get('bytes',-1)):
            raise RuntimeError(f'{cid} {role}: final manifest artifact binding mismatch')
    files[cid]=(cfg,ck)
fit=manifest.get('deployment_fit') or {}
calibration=fit.get('calibration') or []
weights=np.asarray(fit.get('weights') or [],dtype=float)
threshold=float(fit.get('threshold'))
if len(calibration)!=3 or weights.shape!=(3,) or not np.allclose(weights,[1/3,1/3,1/3],rtol=0,atol=1e-12): raise RuntimeError('frozen deployment_fit drift')
if not 0.0 < threshold < 1.0: raise RuntimeError('invalid frozen threshold')

source_zip=WORK/'source_python.zip'; source_zip.write_bytes(base64.b64decode(SOURCE_ZIP_B64))
source_dir=WORK/'source'; source_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(source_zip) as z: z.extractall(source_dir)
PROJECT_CODE=source_dir/'SOURCE_PYTHON'; sys.path.insert(0,str(PROJECT_CODE))
import tensorflow as tf
import pneumonia_runner_base as runner
from auto_ensemble_selection import apply_platt_calibrator
from project_model_loader import load_project_model_for_inference
from external_validation import ExternalCohortSpec, load_external_cohort, audit_external_independence, cohort_descriptive_report, subgroup_metric_table

def is_dev_root(p:Path)->bool:
    return all((p/x).is_dir() for x in ('train','val','test')) and all((p/'test'/x).is_dir() for x in ('NORMAL','PNEUMONIA'))
dev_candidates=[]
for p in [INPUT,*[q for q in INPUT.rglob('*') if q.is_dir()]]:
    if is_dev_root(p) and '__MACOSX' not in p.parts: dev_candidates.append(p.resolve())
dev_roots=[p for p in dev_candidates if not any(p!=q and q in p.parents for q in dev_candidates)]
if len(dev_roots)!=1: raise RuntimeError(f'expected one development root for overlap audit, found {dev_roots}')
dev_root=dev_roots[0]
image_ext={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
dev_rows=[]
for part in ('train','val','test'):
    for cls,label in (('NORMAL',0),('PNEUMONIA',1)):
        d=dev_root/part/cls
        if not d.is_dir(): continue
        for p in sorted(d.rglob('*')):
            if p.is_file() and p.suffix.lower() in image_ext:
                token=hashlib.sha256(f'dev:{p.as_posix()}'.encode()).hexdigest()[:24]
                dev_rows.append({'filepath':str(p.resolve()),'label':label,'patient_id':token,'patient_id_source':'sample_id_fallback','sample_id':token})
development=pd.DataFrame(dev_rows)
if development.empty: raise RuntimeError('development frame empty during external independence audit')

chex_csvs=sorted(INPUT.rglob('valid.csv'), key=lambda p:(0 if 'chexpert' in str(p).lower() else 1,len(p.parts),str(p)))
if not chex_csvs: raise RuntimeError('CheXpert valid.csv not found in Kaggle mounts')
chex_csv=chex_csvs[0]
chex_root=chex_csv.parent
for parent in [chex_csv.parent,*chex_csv.parents]:
    if parent==INPUT.parent: break
    if 'chexpert' in str(parent).lower(): chex_root=parent
chex_spec=ExternalCohortSpec(name='chexpert_valid',adapter='chexpert',root=chex_root,manifest_csv=chex_csv,site_id='stanford_chexpert',task_definition='Pneumonia positive vs explicit negative; uncertain excluded',metadata={'chexpert_uncertain_policy':'exclude'})
chex_frame,chex_load_report=load_external_cohort(chex_spec,WORK/'external_cache')

nih_labels=next(iter(sorted(INPUT.rglob('sample_labels.csv'))),None)
nih_zip=next(iter(sorted(INPUT.rglob('sample.zip'))),None)
if nih_labels is None or nih_zip is None: raise RuntimeError('NIH sample labels/zip not found in Kaggle mounts')
nih_extract=WORK/'nih_sample_images'; nih_extract.mkdir(exist_ok=True)
with zipfile.ZipFile(nih_zip) as z: z.extractall(nih_extract)
nih_raw=pd.read_csv(nih_labels)
required={'Image Index','Finding Labels'}
if not required.issubset(nih_raw.columns): raise RuntimeError(f'NIH sample columns missing: {sorted(required-set(nih_raw.columns))}')
index={p.name:p.resolve() for p in nih_extract.rglob('*') if p.is_file() and p.suffix.lower() in image_ext}
nih_rows=[]
for _,row in nih_raw.iterrows():
    labels={x.strip().lower() for x in str(row['Finding Labels']).split('|') if x.strip()}
    if 'pneumonia' in labels: label=1
    elif labels=={'no finding'}: label=0
    else: continue
    path=index.get(Path(str(row['Image Index'])).name)
    if path is None: continue
    pid=f"nih_patient_{row['Patient ID']}" if 'Patient ID' in nih_raw.columns else hashlib.sha256(f"nih:{path.name}".encode()).hexdigest()[:24]
    item={'filepath':str(path),'label':label,'patient_id':pid,'patient_id_source':'dataset_metadata' if 'Patient ID' in nih_raw.columns else 'sample_id_fallback','study_id':f'nih_study_{path.stem}','class_name':'PNEUMONIA' if label else 'NORMAL','partition':'external','external_cohort':'nih_pneumonia_vs_no_finding','external_adapter':'nih_sample','site_id':'nih'}
    if 'Patient Gender' in nih_raw.columns: item['sex']=row['Patient Gender']
    if 'Patient Age' in nih_raw.columns: item['age']=row['Patient Age']
    if 'View Position' in nih_raw.columns: item['view_position']=row['View Position']
    item['sample_id']=hashlib.sha256(f"nih:{path.as_posix()}".encode()).hexdigest()[:24]
    nih_rows.append(item)
nih_frame=pd.DataFrame(nih_rows).drop_duplicates('sample_id').reset_index(drop=True)
if nih_frame.empty or nih_frame['label'].nunique()!=2: raise RuntimeError('NIH external cohort must contain both classes')
nih_load_report={'name':'nih_pneumonia_vs_no_finding','adapter':'nih_sample','root':str(nih_zip.parent),'manifest_csv':str(nih_labels),'n_images':int(len(nih_frame)),'n_patients':int(nih_frame['patient_id'].nunique()),'class_counts':{str(k):int(v) for k,v in nih_frame['label'].value_counts().sort_index().to_dict().items()},'site_counts':{'nih':int(len(nih_frame))},'has_real_patient_metadata':bool((nih_frame['patient_id_source']=='dataset_metadata').any()),'task_definition':'Pneumonia-containing vs No Finding only','label_policy':{'nih_positive_policy':'contains_pneumonia','nih_negative_policy':'no_finding_only'}}

cohorts={'chexpert_valid':chex_frame,'nih_pneumonia_vs_no_finding':nih_frame}
combined=pd.concat(list(cohorts.values()),ignore_index=True)
independence=audit_external_independence(combined,development,policy='fail')
(OUT/'EXTERNAL_INDEPENDENCE_AUDIT.json').write_text(json.dumps(independence,indent=2,sort_keys=True)+'\n',encoding='utf-8')

models=[]; signatures=[]
for cid in SELECTED:
    _,ck=files[cid]; row=rows[(cid,'checkpoint')]
    model,sig=load_project_model_for_inference(checkpoint_path=ck,model_id='M06',checkpoint_sha256=str(row['sha256']),expected_image_size=224)
    models.append(model); sig['candidate_id']=cid; signatures.append(sig)
(OUT/'MODEL_SIGNATURES.json').write_text(json.dumps(signatures,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')

summary_cohorts={}
for name,frame in cohorts.items():
    raw_probs=[]
    for model in models:
        ds=runner.build_dataset(frame,224,16,False,42,'none',0.2,2,1)
        probs=np.asarray(runner.predict_dataset(model,ds),dtype=float)
        if len(probs)!=len(frame) or not np.all(np.isfinite(probs)): raise RuntimeError(f'{name}: invalid prediction vector')
        raw_probs.append(probs)
    raw_matrix=np.stack(raw_probs,axis=1)
    cal=np.empty_like(raw_matrix)
    for i,params in enumerate(calibration): cal[:,i]=apply_platt_calibrator(raw_matrix[:,i],params)
    ensemble=np.clip(cal@weights,1e-7,1-1e-7)
    labels=frame['label'].to_numpy(dtype=int)
    metrics=runner.calculate_metrics(labels,ensemble,threshold)
    metrics.update({'n_samples':int(len(frame)),'threshold':threshold,'threshold_source':'FROZEN_DEVELOPMENT_ONLY','calibration_source':'FROZEN_DEVELOPMENT_ONLY','weights':[float(x) for x in weights]})
    preds=frame.copy()
    for i,cid in enumerate(SELECTED): preds['raw_'+cid]=raw_matrix[:,i]; preds['calibrated_'+cid]=cal[:,i]
    preds['ensemble_probability_pneumonia']=ensemble; preds['ensemble_prediction']=(ensemble>=threshold).astype(int)
    pred_path=OUT/f'{name}__PREDICTIONS.csv'; preds.to_csv(pred_path,index=False)
    subgroup=subgroup_metric_table(frame,ensemble,threshold,runner.calculate_metrics,min_n=30)
    subgroup_path=OUT/f'{name}__SUBGROUP_METRICS.csv'; subgroup.to_csv(subgroup_path,index=False)
    description=cohort_descriptive_report(frame)
    report={'name':name,'load_report':chex_load_report if name=='chexpert_valid' else nih_load_report,'descriptive':description,'metrics':metrics,'predictions_sha256':sha(pred_path),'subgroup_metrics_sha256':sha(subgroup_path),'n_subgroup_rows':int(len(subgroup))}
    (OUT/f'{name}__REPORT.json').write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
    summary_cohorts[name]={'n_samples':int(len(frame)),'metrics':metrics,'report_sha256':sha(OUT/f'{name}__REPORT.json'),'predictions_sha256':report['predictions_sha256']}
try: tf.keras.backend.clear_session()
except Exception: pass

summary={'schema_version':1,'project':PROJECT,'stage':'EXTERNAL_VALIDATION','status':'PASS','master_account':'azadka','kernel_ref':'azadka/pneumonia-v6-2-2-external-validation','final_freeze_manifest_sha256':FINAL_MANIFEST_SHA,'final_freeze_reconciliation_sha256':RECONCILIATION_SHA,'official_locked_test_completion_sha256':LOCKED_COMPLETION_SHA,'official_locked_test_evaluation_manifest_sha256':LOCKED_EVALUATION_MANIFEST_SHA,'selected_candidate_ids':SELECTED,'external_datasets':[CHEXPERT_DATASET,NIH_DATASET],'cohorts':summary_cohorts,'independence_audit':independence,'deployment_fit':fit,'selection_locked':True,'threshold_tuning_performed':False,'calibration_fitting_performed':False,'weight_fitting_performed':False,'training_performed':False,'hpo_or_confirmation_repeated':False,'reselection_performed':False,'official_locked_test_feedback_used':False,'external_data_used':True,'external_results_feed_back_forbidden':True,'validated_at':datetime.now(timezone.utc).isoformat(),'next_permitted_stage':'FINAL_CLOSURE'}
spath=OUT/'EXTERNAL_VALIDATION_SUMMARY.json'; spath.write_text(json.dumps(summary,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
sdigest=sha(spath); (OUT/'EXTERNAL_VALIDATION_SUMMARY.sha256').write_text(sdigest+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','stage':'EXTERNAL_VALIDATION','summary_sha256':sdigest,'cohorts':{k:{'n_samples':v['n_samples'],'balanced_accuracy':v['metrics'].get('balanced_accuracy'),'mcc':v['metrics'].get('mcc')} for k,v in summary_cohorts.items()},'next':'FINAL_CLOSURE'},indent=2))
