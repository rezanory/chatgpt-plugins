import { kernelStatus, listKernels, type WorkerEnv } from "./kaggle";

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };
type Rec = Record<string, unknown>;
const API_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService";
const OWNER = "azadka";
const TARGET_SLUG = "pneumonia-v6-2-2-final-freeze-reconciliation";
const TARGET_REF = `${OWNER}/${TARGET_SLUG}`;
const TARGET_TITLE = "PNEUMONIA V6.2.2 FINAL FREEZE RECONCILIATION";
const FINALIZATION_REF = "azadka/pneumonia-v6-2-2-finalization-master";
const DATASET_REF = "trickermark/pneumonia-v6-2-2-finalization-artifacts";
const BACKBONE_REF = "trickermark/pneumonia-v6-2-2-backbone-m06-r224";
const FINAL_MANIFEST_SHA = "2313023819c9eec669d835b99dcc2524b03bb0da236c2ec8f98c4e1221948414";
const RECIPE_SHA = "27cae8c59e06b609f9dfd02b526d807dc359b2bb14ae7bc5ee77c68c7553b08f";
const POLICY_FILE_SHA = "7e23bbed9246d5588c67a46380b570e1c1a3e869062613b9dcdb298e2e822861";
const POLICY_SEMANTIC_SHA = "7f5817f3429bb86668718cac9158a7061c6e730cd14bc6137402be82baf576d7";
const ACTIVE = new Set(["RUNNING", "QUEUED", "STARTING", "PENDING"]);

const SCRIPT = String.raw`from __future__ import annotations
import hashlib,json
from pathlib import Path
INPUT=Path('/kaggle/input'); OUT=Path('/kaggle/working/PNEUMONIA_V62_2_FINAL_RECONCILIATION'); OUT.mkdir(parents=True,exist_ok=True)
PROJECT='PNEUMONIA V6.2.2'; TARGET='azadka/pneumonia-v6-2-2-final-freeze-reconciliation'
DATASET_REF='trickermark/pneumonia-v6-2-2-finalization-artifacts'; BACKBONE_REF='trickermark/pneumonia-v6-2-2-backbone-m06-r224'; FINALIZATION_REF='azadka/pneumonia-v6-2-2-finalization-master'
FINAL_MANIFEST_SHA='2313023819c9eec669d835b99dcc2524b03bb0da236c2ec8f98c4e1221948414'; RECIPE_SHA='27cae8c59e06b609f9dfd02b526d807dc359b2bb14ae7bc5ee77c68c7553b08f'; POLICY_FILE_SHA='7e23bbed9246d5588c67a46380b570e1c1a3e869062613b9dcdb298e2e822861'; POLICY_SEMANTIC_SHA='7f5817f3429bb86668718cac9158a7061c6e730cd14bc6137402be82baf576d7'
SELECTED=['M06__convnext_tiny','M06__densenet121','M06__resnet50v2']; CONFIG_SHA={'M06__convnext_tiny':'167cb7b1d17b3978f2b4f47f55e315a1d4fae293006d897d25d13b5074dadcea','M06__densenet121':'c9b3107715849a633d90689902298c110bbb7c9c1bc1c8a6ac9ff1693f55386e','M06__resnet50v2':'dce7845aa7c07ccb913cded8e9ff368e7061f3a43723ded21f8a94dba9fab697'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def unique(name):
 xs=[p for p in INPUT.rglob(name) if p.is_file()]
 if len(xs)!=1:raise RuntimeError(f'{name}: expected one file, found {len(xs)}')
 return xs[0]
mp=unique('FINAL_FREEZE_MANIFEST.json'); sp=unique('FINAL_FREEZE_MANIFEST.sha256'); qp=unique('QUALIFICATION_NON_CONSUMPTION.json'); pp=unique('FROZEN_BACKBONE_ENSEMBLE_POLICY.json')
actual=sha(mp); declared=sp.read_text().strip().lower()
if actual!=FINAL_MANIFEST_SHA or declared!=FINAL_MANIFEST_SHA:raise RuntimeError('final manifest SHA mismatch')
m=json.loads(mp.read_text()); q=json.loads(qp.read_text()); policy=json.loads(pp.read_text())
if m.get('project')!=PROJECT or m.get('stage')!='FINAL_FREEZE_MANIFEST' or m.get('status')!='FROZEN_IN_KAGGLE':raise RuntimeError('final manifest identity mismatch')
if m.get('master_account')!='azadka' or m.get('source_artifact_dataset')!=DATASET_REF or m.get('provenance_source_kernel')!=BACKBONE_REF:raise RuntimeError('final provenance mismatch')
if m.get('recipe_sha256')!=RECIPE_SHA or m.get('frozen_policy_file_sha256')!=POLICY_FILE_SHA or m.get('frozen_policy_semantic_sha256')!=POLICY_SEMANTIC_SHA:raise RuntimeError('recipe/policy binding mismatch')
if m.get('selected_candidate_ids')!=SELECTED or m.get('selected_artifact_count')!=6:raise RuntimeError('selected set mismatch')
for k in ['locked_test_used','external_data_used','training_performed','hpo_or_confirmation_repeated','reselection_allowed_after_freeze']:
 if m.get(k) is not False:raise RuntimeError('forbidden final flag '+k)
qsha=sha(qp)
if m.get('qualification_non_consumption_sha256')!=qsha or q.get('project')!=PROJECT:raise RuntimeError('qualification binding mismatch')
for k in ['qualification_performed','used_for_selection','final_lockbox_used','external_data_used']:
 if q.get(k) is not False:raise RuntimeError('qualification flag mismatch '+k)
if sha(pp)!=POLICY_FILE_SHA or policy.get('policy_sha256')!=POLICY_SEMANTIC_SHA or policy.get('selected_candidates')!=SELECTED:raise RuntimeError('policy identity mismatch')
if policy.get('locked_test_used') is not False or policy.get('external_data_used') is not False or m.get('deployment_fit')!=policy.get('deployment_fit'):raise RuntimeError('policy evidence/fit mismatch')
fit=m.get('deployment_fit') or {}
if fit.get('weights')!=[1/3,1/3,1/3] or len(fit.get('calibration') or [])!=3 or not isinstance(fit.get('threshold'),(int,float)):raise RuntimeError('deployment fit structure mismatch')
rows={(r.get('candidate_id'),r.get('role')):r for r in (m.get('artifacts') or [])}
if len(rows)!=6:raise RuntimeError('artifact row count mismatch')
verified=[]
for cid in SELECTED:
 for role,suffix in [('config','__config.json'),('checkpoint','__final_selected.keras')]:
  f=unique(cid+suffix); digest=sha(f); row=rows.get((cid,role))
  if not row or digest!=str(row.get('sha256','')).lower() or f.stat().st_size!=int(row.get('bytes',-1)) or row.get('dataset_file')!=f.name:raise RuntimeError(f'artifact mismatch {cid} {role}')
  if role=='config' and digest!=CONFIG_SHA[cid]:raise RuntimeError('config canonical mismatch '+cid)
  verified.append({'candidate_id':cid,'role':role,'dataset_file':f.name,'bytes':f.stat().st_size,'sha256':digest})
record={'schema_version':1,'project':PROJECT,'stage':'FINAL_FREEZE_RECONCILIATION','status':'PASS','independent_reconciliation':True,'reconciliation_kernel':TARGET,'finalization_kernel':FINALIZATION_REF,'final_freeze_manifest_sha256':actual,'qualification_non_consumption_sha256':qsha,'recipe_sha256':RECIPE_SHA,'frozen_policy_file_sha256':POLICY_FILE_SHA,'frozen_policy_semantic_sha256':POLICY_SEMANTIC_SHA,'selected_candidate_ids':SELECTED,'verified_artifacts':verified,'selection_locked':True,'policy_frozen':True,'used_for_selection':False,'qualification_performed':False,'locked_test_used':False,'external_data_used':False,'lockbox_access_authorized':False,'next_required_gate':'HARDEN_DISABLE_OBSOLETE_CLOUDFLARE_MUTATING_WORKFLOWS'}
rp=OUT/'FINAL_FREEZE_RECONCILIATION.json'; rp.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n'); rsha=sha(rp); (OUT/'FINAL_FREEZE_RECONCILIATION.sha256').write_text(rsha+'\n')
print(json.dumps({'status':'PASS','stage':'FINAL_FREEZE_RECONCILIATION','reconciliation_sha256':rsha,'final_freeze_manifest_sha256':actual,'verified_artifacts':len(verified),'locked_test_used':False,'external_data_used':False,'lockbox_access_authorized':False},indent=2))
`;
function rec(v:unknown):Rec{return v&&typeof v==="object"&&!Array.isArray(v)?v as Rec:{}}
function json(v:unknown,status=200){return new Response(JSON.stringify(v),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}})}
function authorized(r:Request,e:Env){const t=e.CGP_PROJECT_CONTROL_TOKEN?.trim();return Boolean(t&&t.length>=32&&r.headers.get("authorization")===`Bearer ${t}`)}
function authHeader(t:string){return t.startsWith("KGAT_")?`Bearer ${t}`:`Basic ${btoa(`${OWNER}:${t}`)}`}
function norm(raw:Rec){const s=rec(raw.session);for(const v of [raw.status,raw.statusName,raw.state,raw.sessionStatus,s.status,s.statusName,s.state])if(typeof v==="string"&&v.trim())return v.trim().toUpperCase();return "UNKNOWN"}
function allowed(raw:string){const u=new URL(raw),h=u.hostname.toLowerCase();if(u.protocol!=="https:"||!(h==="api.kaggle.com"||h==="www.kaggle.com"||h==="storage.googleapis.com"||h.endsWith(".kaggleusercontent.com")||h.endsWith(".googleusercontent.com")))throw new Error("output URL host not allowed");return u}
async function shaText(t:string){const d=await crypto.subtle.digest("SHA-256",new TextEncoder().encode(t));return Array.from(new Uint8Array(d),b=>b.toString(16).padStart(2,"0")).join("")}
async function save(env:Env){const t=env.CGP_KAGGLE_MASTER_TOKEN?.trim();if(!t)throw new Error("master token missing");const body:Rec={slug:TARGET_REF,newTitle:TARGET_TITLE,text:SCRIPT,language:"python",kernelType:"script",kernelExecutionType:"SAVE_AND_RUN_ALL",isPrivate:true,enableGpu:false,enableTpu:false,enableInternet:false,kernelDataSources:[FINALIZATION_REF],datasetDataSources:[DATASET_REF],competitionDataSources:[],modelDataSources:[]};const r=await fetch(`${API_ROOT}/SaveKernel`,{method:"POST",headers:{Authorization:authHeader(t),"Content-Type":"application/json","User-Agent":"chatgpt-v622-final-reconcile-v2/1.0"},body:JSON.stringify(body)});const text=await r.text();let v:Rec={};try{v=rec(text?JSON.parse(text):{})}catch{throw new Error(`SaveKernel non-JSON ${r.status}`)}if(!r.ok)throw new Error(String(v.message??text).slice(0,1000));const bd=Array.isArray(v.invalidDatasetSources)?v.invalidDatasetSources:[],bk=Array.isArray(v.invalidKernelSources)?v.invalidKernelSources:[];if(bd.length||bk.length)throw new Error(`rejected sources datasets=${JSON.stringify(bd)} kernels=${JSON.stringify(bk)}`);return v}
async function receipt(env:Env):Promise<Rec|null>{const t=env.CGP_KAGGLE_MASTER_TOKEN?.trim();if(!t)throw new Error("master token missing");const r=await fetch(`${API_ROOT}/ListKernelSessionOutput`,{method:"POST",headers:{Authorization:authHeader(t),"Content-Type":"application/json","User-Agent":"chatgpt-v622-final-reconcile-v2/1.0"},body:JSON.stringify({userName:OWNER,kernelSlug:TARGET_SLUG,pageSize:100})});const p=rec(await r.json());if(!r.ok)throw new Error(`output HTTP ${r.status}`);const fs=Array.isArray(p.files)?p.files.map(rec):[];const rr=fs.find(x=>String(x.fileName??"").endsWith("FINAL_FREEZE_RECONCILIATION.json")&&typeof x.url==="string"),sr=fs.find(x=>String(x.fileName??"").endsWith("FINAL_FREEZE_RECONCILIATION.sha256")&&typeof x.url==="string");if(!rr||!sr)return null;const text=await(await fetch(allowed(String(rr.url)),{redirect:"follow"})).text();const declared=(await(await fetch(allowed(String(sr.url)),{redirect:"follow"})).text()).trim().toLowerCase(),actual=await shaText(text);if(actual!==declared)throw new Error("reconciliation SHA mismatch");const v=rec(JSON.parse(text));if(v.project!=="PNEUMONIA V6.2.2"||v.stage!=="FINAL_FREEZE_RECONCILIATION"||v.status!=="PASS"||v.independent_reconciliation!==true||v.reconciliation_kernel!==TARGET_REF)throw new Error("reconciliation identity mismatch");if(v.final_freeze_manifest_sha256!==FINAL_MANIFEST_SHA||v.locked_test_used!==false||v.external_data_used!==false||v.lockbox_access_authorized!==false)throw new Error("reconciliation frozen/evidence mismatch");return {reconciliation_sha256:actual,final_freeze_manifest_sha256:v.final_freeze_manifest_sha256,verified_artifact_count:Array.isArray(v.verified_artifacts)?v.verified_artifacts.length:0,next_required_gate:v.next_required_gate,lockbox_access_authorized:false}}
async function launch(env:Env):Promise<Rec>{const xs=await listKernels(env,"master",TARGET_SLUG,100),exact=xs.some(x=>String(rec(x).ref??"").toLowerCase()===TARGET_REF.toLowerCase());if(exact){const s=norm(await kernelStatus(env,"master",TARGET_REF));if(ACTIVE.has(s))return{action:"existing_active",target_kernel_ref:TARGET_REF,status:s,submitted:false};if(s==="COMPLETE"){try{const rc=await receipt(env);if(rc)return{action:"existing_complete",target_kernel_ref:TARGET_REF,status:s,submitted:false,receipt:rc}}catch{/* old v1 receipt has wrong self-ref: replace with corrected version */}}}const result=await save(env);return{action:exact?"submitted_corrected_version":"created_and_submitted",target_kernel_ref:TARGET_REF,result,submitted:true,kaggle_compute_kind:"INDEPENDENT_FINAL_FREEZE_RECONCILIATION_REPAIR",training:false,hpo:false,confirmation:false,locked_test:false,external_validation:false}}
export default{async fetch(request:Request,env:Env):Promise<Response>{const u=new URL(request.url);if(u.pathname==="/healthz"&&request.method==="GET")return json({service:"v622-finalization-reconcile-v2",status:"ready",protected:true});if(!authorized(request,env))return new Response("Forbidden",{status:403});try{if(u.pathname==="/control/v6-2-2/finalization-reconcile-v2/launch"&&request.method==="POST")return json({project:"PNEUMONIA V6.2.2",...(await launch(env))});if(u.pathname==="/control/v6-2-2/finalization-reconcile-v2/status"&&request.method==="POST"){const s=norm(await kernelStatus(env,"master",TARGET_REF));const rc=s==="COMPLETE"?await receipt(env):null;return json({project:"PNEUMONIA V6.2.2",target_kernel_ref:TARGET_REF,status:s,receipt:rc})}return new Response("Not found",{status:404})}catch(e){return json({project:"PNEUMONIA V6.2.2",ok:false,error:e instanceof Error?e.message.slice(0,1400):"unknown",locked_test:false,external_validation:false},502)}}} satisfies ExportedHandler<Env>;
