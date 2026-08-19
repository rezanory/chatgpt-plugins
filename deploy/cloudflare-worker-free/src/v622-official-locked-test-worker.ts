import { kernelStatus, listKernels, type WorkerEnv } from "./kaggle";

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };
type Rec = Record<string, unknown>;
const API_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService";
const OWNER = "azadka";
const TARGET_SLUG = "pneumonia-v6-2-2-official-locked-test";
const TARGET_REF = `${OWNER}/${TARGET_SLUG}`;
const TARGET_TITLE = "PNEUMONIA V6.2.2 OFFICIAL LOCKED TEST";
const ARTIFACT_DATASET = "trickermark/pneumonia-v6-2-2-finalization-artifacts";
const RAW_DATASET = "paultimothymooney/chest-xray-pneumonia";
const FINALIZATION_REF = "azadka/pneumonia-v6-2-2-finalization-master";
const RECONCILIATION_REF = "azadka/pneumonia-v6-2-2-final-freeze-reconciliation";
const FINAL_MANIFEST_SHA = "2313023819c9eec669d835b99dcc2524b03bb0da236c2ec8f98c4e1221948414";
const RECONCILIATION_SHA = "99a319c8d397342aaefd3c15e0eb8f968a973738f55ef97925a2a16da8c485d8";
const POLICY_FILE_SHA = "7e23bbed9246d5588c67a46380b570e1c1a3e869062613b9dcdb298e2e822861";
const ACTIVE = new Set(["RUNNING", "QUEUED", "STARTING", "PENDING"]);

const LOCKED_TEST_SCRIPT = String.raw`from __future__ import annotations
import hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

PROJECT='PNEUMONIA V6.2.2'
FINAL_TEST_CONFIRM='I_HAVE_FROZEN_THE_CHAMPION'
FINAL_MANIFEST_SHA='2313023819c9eec669d835b99dcc2524b03bb0da236c2ec8f98c4e1221948414'
RECONCILIATION_SHA='99a319c8d397342aaefd3c15e0eb8f968a973738f55ef97925a2a16da8c485d8'
POLICY_FILE_SHA='7e23bbed9246d5588c67a46380b570e1c1a3e869062613b9dcdb298e2e822861'
SELECTED=['M06__convnext_tiny','M06__densenet121','M06__resnet50v2']
INPUT=Path('/kaggle/input')
OUT=Path('/kaggle/working/PNEUMONIA_V62_2_OFFICIAL_LOCKED_TEST'); OUT.mkdir(parents=True,exist_ok=True)
START=time.perf_counter()
if FINAL_TEST_CONFIRM!='I_HAVE_FROZEN_THE_CHAMPION': raise RuntimeError('final lockbox confirmation token mismatch')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
 return h.hexdigest()

def unique(name):
 xs=[p for p in INPUT.rglob(name) if p.is_file()]
 if len(xs)!=1: raise RuntimeError(f'{name}: expected exactly one file, found {len(xs)}')
 return xs[0]

mpath=unique('FINAL_FREEZE_MANIFEST.json'); msha=unique('FINAL_FREEZE_MANIFEST.sha256')
rpath=unique('FINAL_FREEZE_RECONCILIATION.json'); rsha=unique('FINAL_FREEZE_RECONCILIATION.sha256')
if sha(mpath)!=FINAL_MANIFEST_SHA or msha.read_text(encoding='utf-8').strip().lower()!=FINAL_MANIFEST_SHA: raise RuntimeError('final freeze binding mismatch')
if sha(rpath)!=RECONCILIATION_SHA or rsha.read_text(encoding='utf-8').strip().lower()!=RECONCILIATION_SHA: raise RuntimeError('reconciliation binding mismatch')
manifest=json.loads(mpath.read_text(encoding='utf-8')); recon=json.loads(rpath.read_text(encoding='utf-8'))
if manifest.get('locked_test_used') is not False or manifest.get('external_data_used') is not False: raise RuntimeError('final freeze already consumed forbidden evidence')
if recon.get('locked_test_used') is not False or recon.get('external_data_used') is not False: raise RuntimeError('reconciliation contains forbidden evidence')
if manifest.get('selected_candidate_ids')!=SELECTED or recon.get('selected_candidate_ids')!=SELECTED: raise RuntimeError('selected ensemble identity drift')

ppath=unique('FROZEN_BACKBONE_ENSEMBLE_POLICY.json')
if sha(ppath)!=POLICY_FILE_SHA: raise RuntimeError('frozen ensemble policy SHA mismatch')
policy=json.loads(ppath.read_text(encoding='utf-8'))
if policy.get('status')!='FROZEN_FROM_DEVELOPMENT_ONLY' or policy.get('selected_candidates')!=SELECTED: raise RuntimeError('frozen policy identity drift')
if policy.get('locked_test_used') is not False or policy.get('external_data_used') is not False: raise RuntimeError('frozen policy already touched lockbox/external')
fit=policy.get('deployment_fit') or {}
weights=np.asarray(fit.get('weights') or [],dtype=float); calibrators=fit.get('calibration') or []; threshold=float(fit.get('threshold'))
if len(weights)!=3 or len(calibrators)!=3 or not np.allclose(weights,np.asarray([1/3,1/3,1/3],float),rtol=0,atol=1e-12): raise RuntimeError('deployment weights/calibration drift')
if not (0.0 < threshold < 1.0): raise RuntimeError('invalid frozen threshold')
if manifest.get('deployment_fit')!=fit: raise RuntimeError('manifest/policy deployment fit mismatch')

@tf.keras.utils.register_keras_serializable(package='PneumoniaAI')
class EdgeBlock(tf.keras.layers.Layer):
 def __init__(self,filters=16,max_gate=0.25,**kwargs):
  super().__init__(**kwargs); self.filters=int(filters); self.max_gate=float(max_gate)
  self.conv1=tf.keras.layers.Conv2D(self.filters,3,padding='same',activation='gelu',dtype='float32')
  self.conv2=tf.keras.layers.Conv2D(3,1,padding='same',activation='tanh',dtype='float32')
 def build(self,input_shape):
  self.raw_gate=self.add_weight(name='raw_gate',shape=(),initializer=tf.keras.initializers.Constant(-4.0),trainable=True,dtype='float32'); super().build(input_shape)
 @staticmethod
 def _normalize_per_image(t):
  mn=tf.stop_gradient(tf.reduce_min(t,axis=[1,2,3],keepdims=True)); mx=tf.stop_gradient(tf.reduce_max(t,axis=[1,2,3],keepdims=True)); return tf.math.divide_no_nan(t-mn,tf.maximum(mx-mn,tf.constant(1e-3,tf.float32)))
 def call(self,inputs):
  od=inputs.dtype; x=tf.cast(inputs,tf.float32); gray=tf.image.rgb_to_grayscale(x)/255.0; sobel=tf.image.sobel_edges(gray); sy=tf.abs(sobel[...,0]); sx=tf.abs(sobel[...,1]); k=tf.constant([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]],tf.float32); k=tf.reshape(k,[3,3,1,1]); lap=tf.abs(tf.nn.conv2d(gray,k,strides=1,padding='SAME')); e=tf.concat([self._normalize_per_image(sx),self._normalize_per_image(sy),self._normalize_per_image(lap)],axis=-1); delta=tf.cast(self.conv2(self.conv1(e)),tf.float32)*255.0; gate=tf.nn.sigmoid(tf.cast(self.raw_gate,tf.float32))*tf.constant(self.max_gate,tf.float32); return tf.cast(tf.clip_by_value(x+gate*delta,0.,255.),od)
 def get_config(self):
  c=super().get_config(); c.update({'filters':self.filters,'max_gate':self.max_gate}); return c

@tf.keras.utils.register_keras_serializable(package='PneumoniaAI')
class ApplicationPreprocessing(tf.keras.layers.Layer):
 def __init__(self,backbone_name,**kwargs): super().__init__(**kwargs); self.backbone_name=str(backbone_name)
 def call(self,inputs):
  if self.backbone_name=='densenet121': return tf.keras.applications.densenet.preprocess_input(inputs)
  if self.backbone_name=='resnet50v2': return tf.keras.applications.resnet_v2.preprocess_input(inputs)
  return inputs
 def get_config(self):
  c=super().get_config(); c.update({'backbone_name':self.backbone_name}); return c

@tf.keras.utils.register_keras_serializable(package='PneumoniaAI')
class ClipPixels(tf.keras.layers.Layer):
 def call(self,inputs): return tf.clip_by_value(inputs,0.0,255.0)

def looks_root(p):
 try:
  return all((p/x).is_dir() for x in ('train','val','test')) and all((p/'test'/x).is_dir() for x in ('NORMAL','PNEUMONIA'))
 except Exception: return False
roots=[]
for d in [INPUT,*[p for p in INPUT.rglob('*') if p.is_dir()]]:
 if looks_root(d): roots.append(d.resolve())
roots=sorted({str(p):p for p in roots}.values(),key=lambda p:(len(p.parts),str(p)))
top=[p for p in roots if not any(p!=q and q in p.parents for q in roots)]
if len(top)!=1: raise RuntimeError(f'expected one canonical chest-xray root; found {top}')
root=top[0]; exts={'.jpg','.jpeg','.png','.bmp'}; records=[]
for cls,label in [('NORMAL',0),('PNEUMONIA',1)]:
 for p in sorted((root/'test'/cls).iterdir()):
  if p.is_file() and p.suffix.lower() in exts: records.append({'sample_id':f'test/{cls}/{p.name}','filepath':str(p),'label':label,'class_name':cls})
test=pd.DataFrame(records)
if len(test)!=624: raise RuntimeError(f'official locked test must contain 624 images; found {len(test)}')
if test['sample_id'].duplicated().any() or set(test['label'].unique())!={0,1}: raise RuntimeError('locked test identity/classes invalid')

def preprocess(path):
 raw=tf.io.read_file(path); image=tf.io.decode_image(raw,channels=3,expand_animations=False); image.set_shape([None,None,3]); image=tf.image.resize_with_pad(image,224,224,method=tf.image.ResizeMethod.BILINEAR,antialias=True); return tf.clip_by_value(tf.cast(image,tf.float32),0.0,255.0)
def make_ds():
 ds=tf.data.Dataset.from_tensor_slices(test['filepath'].astype(str).tolist()); o=tf.data.Options(); o.experimental_deterministic=True; ds=ds.with_options(o).map(preprocess,num_parallel_calls=2,deterministic=True); return ds.batch(16,drop_remainder=False).prefetch(1)
def platt(prob,params):
 eps=1e-7; p=np.clip(np.asarray(prob,dtype=float),eps,1-eps); logit=np.log(p)-np.log1p(-p); score=float(params['coefficient'])*logit+float(params['intercept']); return np.clip(1.0/(1.0+np.exp(-np.clip(score,-40.,40.))),eps,1-eps)

raw_probs=[]; calibrated=[]; model_meta=[]
for i,cid in enumerate(SELECTED):
 cp=unique(cid+'__final_selected.keras'); expected=(manifest.get('artifacts') or []); row=next((r for r in expected if r.get('candidate_id')==cid and r.get('role')=='checkpoint'),None)
 if not row or sha(cp)!=str(row.get('sha256','')).lower(): raise RuntimeError(cid+': checkpoint SHA mismatch before locked inference')
 model=tf.keras.models.load_model(cp,compile=False,custom_objects={'EdgeBlock':EdgeBlock,'PneumoniaAI>EdgeBlock':EdgeBlock,'ApplicationPreprocessing':ApplicationPreprocessing,'PneumoniaAI>ApplicationPreprocessing':ApplicationPreprocessing,'ClipPixels':ClipPixels,'PneumoniaAI>ClipPixels':ClipPixels})
 ish=tuple(model.input_shape); osh=tuple(model.output_shape)
 if len(ish)!=4 or ish[1:4]!=(224,224,3) or len(osh)!=2 or osh[-1]!=1: raise RuntimeError(cid+f': serialized model signature drift input={ish} output={osh}')
 probs=np.asarray(model.predict(make_ds(),verbose=1)).reshape(-1).astype(np.float64)
 if len(probs)!=624 or not np.all(np.isfinite(probs)): raise RuntimeError(cid+': invalid prediction vector')
 raw_probs.append(probs); calibrated.append(platt(probs,calibrators[i])); model_meta.append({'candidate_id':cid,'checkpoint_sha256':sha(cp),'input_shape':list(ish),'output_shape':list(osh)})
 tf.keras.backend.clear_session()

matrix=np.stack(calibrated,axis=1); ensemble=np.clip(matrix@weights,1e-7,1-1e-7); y=test['label'].to_numpy(int); pred=(ensemble>=threshold).astype(int)
tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); metrics={
 'accuracy':float(accuracy_score(y,pred)), 'balanced_accuracy':float(balanced_accuracy_score(y,pred)), 'mcc':float(matthews_corrcoef(y,pred)),
 'precision_pneumonia':float(precision_score(y,pred,pos_label=1,zero_division=0)), 'recall_pneumonia':float(recall_score(y,pred,pos_label=1,zero_division=0)),
 'precision_normal':float(precision_score(y,pred,pos_label=0,zero_division=0)), 'recall_normal':float(recall_score(y,pred,pos_label=0,zero_division=0)),
 'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)), 'roc_auc':float(roc_auc_score(y,ensemble)), 'average_precision_pneumonia':float(average_precision_score(y,ensemble)), 'brier':float(brier_score_loss(y,ensemble)),
 'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'threshold':threshold
}
preds=test[['sample_id','label','class_name']].copy()
for i,cid in enumerate(SELECTED): preds[cid+'__raw_probability']=raw_probs[i]; preds[cid+'__calibrated_probability']=calibrated[i]
preds['ensemble_probability_pneumonia']=ensemble; preds['ensemble_prediction']=pred; pred_path=OUT/'OFFICIAL_LOCKED_TEST_PREDICTIONS.csv'; preds.to_csv(pred_path,index=False)
result={'schema_version':1,'project':PROJECT,'stage':'OFFICIAL_LOCKED_TEST','status':'PASS','master_account':'azadka','kernel_ref':'azadka/pneumonia-v6-2-2-official-locked-test','final_test_confirmation':'I_HAVE_FROZEN_THE_CHAMPION','final_freeze_manifest_sha256':FINAL_MANIFEST_SHA,'final_freeze_reconciliation_sha256':RECONCILIATION_SHA,'frozen_policy_file_sha256':POLICY_FILE_SHA,'selected_candidate_ids':SELECTED,'sample_count':624,'source_partition':'official test -> test_locked; untouched by effective split','data_access_scope':'LOCKED_TEST_ONLY','deployment_fit':fit,'models':model_meta,'metrics':metrics,'predictions_sha256':sha(pred_path),'selection_locked':True,'threshold_tuning_performed':False,'calibration_fitting_performed':False,'weight_fitting_performed':False,'training_performed':False,'hpo_or_confirmation_repeated':False,'reselection_performed':False,'external_data_used':False,'feedback_to_training_hpo_selection_forbidden':True,'next_permitted_stage':'EXTERNAL_VALIDATION','evaluated_at':datetime.now(timezone.utc).isoformat(),'total_seconds':round(time.perf_counter()-START,3)}
out=OUT/'OFFICIAL_LOCKED_TEST.json'; out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8'); digest=sha(out); (OUT/'OFFICIAL_LOCKED_TEST.sha256').write_text(digest+'\n',encoding='utf-8'); print(json.dumps({'status':'PASS','stage':'OFFICIAL_LOCKED_TEST','sha256':digest,'sample_count':624,'metrics':metrics,'next':'EXTERNAL_VALIDATION'},indent=2))
`;

function rec(v: unknown): Rec { return v && typeof v === "object" && !Array.isArray(v) ? v as Rec : {}; }
function json(v: unknown, status = 200): Response { return new Response(JSON.stringify(v), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } }); }
function authorized(request: Request, env: Env): boolean { const t=env.CGP_PROJECT_CONTROL_TOKEN?.trim(); return Boolean(t&&t.length>=32&&request.headers.get("authorization")===`Bearer ${t}`); }
function authHeader(token: string): string { return token.startsWith("KGAT_") ? `Bearer ${token}` : `Basic ${btoa(`${OWNER}:${token}`)}`; }
function normalizedStatus(raw: Rec): string { const s=rec(raw.session); for(const v of [raw.status,raw.statusName,raw.state,raw.sessionStatus,s.status,s.statusName,s.state]) if(typeof v==="string"&&v.trim()) return v.trim().toUpperCase(); return "UNKNOWN"; }
function allowedOutputUrl(raw: string): URL { const u=new URL(raw),h=u.hostname.toLowerCase(); if(u.protocol!=="https:"||!(h==="api.kaggle.com"||h==="www.kaggle.com"||h==="storage.googleapis.com"||h.endsWith(".kaggleusercontent.com")||h.endsWith(".googleusercontent.com"))) throw new Error("output URL host not allowlisted"); return u; }
async function sha256Text(text:string):Promise<string>{const d=await crypto.subtle.digest("SHA-256",new TextEncoder().encode(text));return Array.from(new Uint8Array(d),b=>b.toString(16).padStart(2,"0")).join("");}
async function saveKernel(env: Env): Promise<Rec> {
 const token=env.CGP_KAGGLE_MASTER_TOKEN?.trim(); if(!token) throw new Error("master Kaggle token missing");
 const request:Rec={slug:TARGET_REF,newTitle:TARGET_TITLE,text:LOCKED_TEST_SCRIPT,language:"python",kernelType:"script",kernelExecutionType:"SAVE_AND_RUN_ALL",isPrivate:true,enableGpu:true,enableTpu:false,enableInternet:false,kernelDataSources:[FINALIZATION_REF,RECONCILIATION_REF],datasetDataSources:[ARTIFACT_DATASET,RAW_DATASET],competitionDataSources:[],modelDataSources:[]};
 const response=await fetch(`${API_ROOT}/SaveKernel`,{method:"POST",headers:{Authorization:authHeader(token),"Content-Type":"application/json","User-Agent":"chatgpt-v622-official-locked-test/1.0"},body:JSON.stringify(request)}); const text=await response.text(); let value:Rec={}; try{value=rec(text?JSON.parse(text):{});}catch{throw new Error(`SaveKernel non-JSON HTTP ${response.status}`);} const code=typeof value.code==="number"?value.code:undefined; if(!response.ok||(code!==undefined&&code>=400)) throw new Error(String(value.message??`SaveKernel HTTP ${response.status}`).slice(0,1200)); const badD=Array.isArray(value.invalidDatasetSources)?value.invalidDatasetSources:[]; const badK=Array.isArray(value.invalidKernelSources)?value.invalidKernelSources:[]; if(badD.length||badK.length) throw new Error(`SaveKernel rejected sources: datasets=${JSON.stringify(badD)} kernels=${JSON.stringify(badK)}`); return value;
}
async function outputReceipt(env:Env):Promise<Rec|null>{
 const token=env.CGP_KAGGLE_MASTER_TOKEN?.trim(); if(!token) throw new Error("master Kaggle token missing"); const response=await fetch(`${API_ROOT}/ListKernelSessionOutput`,{method:"POST",headers:{Authorization:authHeader(token),"Content-Type":"application/json","User-Agent":"chatgpt-v622-official-locked-test/1.0"},body:JSON.stringify({userName:OWNER,kernelSlug:TARGET_SLUG,pageSize:100})}); const payload=rec(await response.json()); if(!response.ok) throw new Error(`ListKernelSessionOutput HTTP ${response.status}`); const files=Array.isArray(payload.files)?payload.files.map(rec):[]; const rr=files.find(r=>String(r.fileName??"").endsWith("OFFICIAL_LOCKED_TEST.json")&&typeof r.url==="string"), sr=files.find(r=>String(r.fileName??"").endsWith("OFFICIAL_LOCKED_TEST.sha256")&&typeof r.url==="string"); if(!rr||!sr) return null; const text=await(await fetch(allowedOutputUrl(String(rr.url)),{redirect:"follow"})).text(); const declared=(await(await fetch(allowedOutputUrl(String(sr.url)),{redirect:"follow"})).text()).trim().toLowerCase(); const actual=await sha256Text(text); if(actual!==declared) throw new Error("official locked test SHA mismatch"); const r=rec(JSON.parse(text)); if(r.project!=="PNEUMONIA V6.2.2"||r.stage!=="OFFICIAL_LOCKED_TEST"||r.status!=="PASS") throw new Error("official locked test identity/status mismatch"); if(r.final_freeze_manifest_sha256!==FINAL_MANIFEST_SHA||r.final_freeze_reconciliation_sha256!==RECONCILIATION_SHA||r.frozen_policy_file_sha256!==POLICY_FILE_SHA) throw new Error("official locked test frozen binding mismatch"); if(Number(r.sample_count)!==624||r.external_data_used!==false||r.training_performed!==false||r.reselection_performed!==false) throw new Error("official locked test governance mismatch"); return {official_locked_test_sha256:actual,sample_count:r.sample_count,metrics:r.metrics,next_permitted_stage:r.next_permitted_stage};
}
async function launch(env:Env):Promise<Rec>{
 const existing=await listKernels(env,"master",TARGET_SLUG,100); const exact=existing.some(i=>String(rec(i).ref??"").toLowerCase()===TARGET_REF.toLowerCase());
 if(exact){const status=normalizedStatus(await kernelStatus(env,"master",TARGET_REF)); if(ACTIVE.has(status)) return {action:"existing_active",target_kernel_ref:TARGET_REF,status,submitted:false}; if(status==="COMPLETE"){const receipt=await outputReceipt(env); if(receipt) return {action:"existing_complete",target_kernel_ref:TARGET_REF,status,submitted:false,receipt}; throw new Error("Official locked-test kernel is COMPLETE without a validated receipt; automatic rerun is forbidden");} throw new Error(`Official locked-test kernel already exists with terminal status ${status}; automatic rerun is forbidden`);}
 const result=await saveKernel(env); return {action:"created_and_submitted",target_kernel_ref:TARGET_REF,result,submitted:true,kaggle_compute_kind:"OFFICIAL_LOCKED_TEST_ONLY",gpu:true,training:false,hpo:false,confirmation:false,reselection:false,external_validation:false};
}
export default {async fetch(request:Request,env:Env):Promise<Response>{const url=new URL(request.url); if(url.pathname==="/healthz"&&request.method==="GET") return json({service:"v622-official-locked-test",status:"ready",protected:true}); if(!authorized(request,env)) return new Response("Forbidden",{status:403}); try{if(url.pathname==="/control/v6-2-2/official-locked-test/launch"&&request.method==="POST") return json({project:"PNEUMONIA V6.2.2",...(await launch(env))}); if(url.pathname==="/control/v6-2-2/official-locked-test/status"&&request.method==="POST"){const status=normalizedStatus(await kernelStatus(env,"master",TARGET_REF)); const receipt=status==="COMPLETE"?await outputReceipt(env):null; return json({project:"PNEUMONIA V6.2.2",target_kernel_ref:TARGET_REF,status,receipt});} return new Response("Not found",{status:404});}catch(error){return json({project:"PNEUMONIA V6.2.2",ok:false,error:error instanceof Error?error.message.slice(0,1400):"unknown error"},502);}}} satisfies ExportedHandler<Env>;
