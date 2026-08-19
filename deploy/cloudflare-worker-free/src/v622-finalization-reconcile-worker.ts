import { kernelStatus, listKernels, type WorkerEnv } from "./kaggle";

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };
type Rec = Record<string, unknown>;
const API_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService";
const OWNER = "azadka";
const TARGET_SLUG = "pneumonia-v6-2-2-finalization-reconcile";
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

const RECONCILE_SCRIPT = String.raw`from __future__ import annotations
import hashlib, json
from pathlib import Path
INPUT=Path('/kaggle/input')
OUT=Path('/kaggle/working/PNEUMONIA_V62_2_FINAL_RECONCILIATION'); OUT.mkdir(parents=True,exist_ok=True)
PROJECT='PNEUMONIA V6.2.2'
DATASET_REF='trickermark/pneumonia-v6-2-2-finalization-artifacts'
BACKBONE_REF='trickermark/pneumonia-v6-2-2-backbone-m06-r224'
FINALIZATION_REF='azadka/pneumonia-v6-2-2-finalization-master'
FINAL_MANIFEST_SHA='2313023819c9eec669d835b99dcc2524b03bb0da236c2ec8f98c4e1221948414'
RECIPE_SHA='27cae8c59e06b609f9dfd02b526d807dc359b2bb14ae7bc5ee77c68c7553b08f'
POLICY_FILE_SHA='7e23bbed9246d5588c67a46380b570e1c1a3e869062613b9dcdb298e2e822861'
POLICY_SEMANTIC_SHA='7f5817f3429bb86668718cac9158a7061c6e730cd14bc6137402be82baf576d7'
SELECTED=['M06__convnext_tiny','M06__densenet121','M06__resnet50v2']
CONFIG_SHA={'M06__convnext_tiny':'167cb7b1d17b3978f2b4f47f55e315a1d4fae293006d897d25d13b5074dadcea','M06__densenet121':'c9b3107715849a633d90689902298c110bbb7c9c1bc1c8a6ac9ff1693f55386e','M06__resnet50v2':'dce7845aa7c07ccb913cded8e9ff368e7061f3a43723ded21f8a94dba9fab697'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
 return h.hexdigest()
def unique(name):
 xs=[p for p in INPUT.rglob(name) if p.is_file()]
 if len(xs)!=1: raise RuntimeError(f'{name}: expected exactly one file, found {len(xs)}')
 return xs[0]
mpath=unique('FINAL_FREEZE_MANIFEST.json'); spath=unique('FINAL_FREEZE_MANIFEST.sha256'); qpath=unique('QUALIFICATION_NON_CONSUMPTION.json'); ppath=unique('FROZEN_BACKBONE_ENSEMBLE_POLICY.json')
actual_manifest_sha=sha(mpath); declared=spath.read_text(encoding='utf-8').strip().lower()
if actual_manifest_sha!=FINAL_MANIFEST_SHA or declared!=FINAL_MANIFEST_SHA: raise RuntimeError('FINAL_FREEZE_MANIFEST frozen SHA mismatch')
m=json.loads(mpath.read_text(encoding='utf-8')); q=json.loads(qpath.read_text(encoding='utf-8')); policy=json.loads(ppath.read_text(encoding='utf-8'))
if m.get('project')!=PROJECT or m.get('stage')!='FINAL_FREEZE_MANIFEST' or m.get('status')!='FROZEN_IN_KAGGLE': raise RuntimeError('final freeze identity/status mismatch')
if m.get('master_account')!='azadka' or m.get('source_artifact_dataset')!=DATASET_REF or m.get('provenance_source_kernel')!=BACKBONE_REF: raise RuntimeError('final freeze provenance mismatch')
if m.get('recipe_sha256')!=RECIPE_SHA or m.get('frozen_policy_file_sha256')!=POLICY_FILE_SHA or m.get('frozen_policy_semantic_sha256')!=POLICY_SEMANTIC_SHA: raise RuntimeError('final freeze recipe/policy mismatch')
if m.get('selected_candidate_ids')!=SELECTED or m.get('selected_artifact_count')!=6: raise RuntimeError('selected set/cardinality mismatch')
for k in ['locked_test_used','external_data_used','training_performed','hpo_or_confirmation_repeated','reselection_allowed_after_freeze']:
 if m.get(k) is not False: raise RuntimeError(f'forbidden final-freeze flag: {k}')
qsha=sha(qpath)
if m.get('qualification_non_consumption_sha256')!=qsha: raise RuntimeError('qualification SHA binding mismatch')
if q.get('project')!=PROJECT: raise RuntimeError('qualification project mismatch')
for k in ['qualification_performed','used_for_selection','final_lockbox_used','external_data_used']:
 if q.get(k) is not False: raise RuntimeError(f'qualification non-consumption mismatch: {k}')
if sha(ppath)!=POLICY_FILE_SHA: raise RuntimeError('policy file SHA mismatch')
if policy.get('status')!='FROZEN_FROM_DEVELOPMENT_ONLY' or policy.get('recipe_sha256')!=RECIPE_SHA or policy.get('policy_sha256')!=POLICY_SEMANTIC_SHA: raise RuntimeError('policy identity mismatch')
if policy.get('selected_candidates')!=SELECTED or policy.get('locked_test_used') is not False or policy.get('external_data_used') is not False: raise RuntimeError('policy selected/evidence mismatch')
if m.get('deployment_fit')!=policy.get('deployment_fit'): raise RuntimeError('deployment fit mismatch between manifest and policy')
fit=m.get('deployment_fit') or {}
if fit.get('weights')!=[1/3,1/3,1/3] or len(fit.get('calibration') or [])!=3 or not isinstance(fit.get('threshold'),(int,float)): raise RuntimeError('deployment fit structure mismatch')
rows={(r.get('candidate_id'),r.get('role')):r for r in (m.get('artifacts') or [])}
if len(rows)!=6: raise RuntimeError('manifest artifact row cardinality mismatch')
verified=[]
for cid in SELECTED:
 for role,suffix in [('config','__config.json'),('checkpoint','__final_selected.keras')]:
  f=unique(cid+suffix); digest=sha(f); row=rows.get((cid,role))
  if not row: raise RuntimeError(f'missing manifest artifact row: {cid} {role}')
  if digest!=str(row.get('sha256','')).lower() or f.stat().st_size!=int(row.get('bytes',-1)): raise RuntimeError(f'independent artifact mismatch: {cid} {role}')
  if row.get('dataset_file')!=f.name: raise RuntimeError(f'dataset filename mismatch: {cid} {role}')
  if role=='config' and digest!=CONFIG_SHA[cid]: raise RuntimeError(f'canonical config mismatch: {cid}')
  verified.append({'candidate_id':cid,'role':role,'dataset_file':f.name,'bytes':f.stat().st_size,'sha256':digest})
record={'schema_version':1,'project':PROJECT,'stage':'FINAL_FREEZE_RECONCILIATION','status':'PASS','independent_reconciliation':True,'reconciliation_kernel':'azadka/pneumonia-v6-2-2-finalization-reconcile','finalization_kernel':FINALIZATION_REF,'final_freeze_manifest_sha256':actual_manifest_sha,'qualification_non_consumption_sha256':qsha,'recipe_sha256':RECIPE_SHA,'frozen_policy_file_sha256':POLICY_FILE_SHA,'frozen_policy_semantic_sha256':POLICY_SEMANTIC_SHA,'selected_candidate_ids':SELECTED,'verified_artifacts':verified,'selection_locked':True,'policy_frozen':True,'used_for_selection':False,'qualification_performed':False,'locked_test_used':False,'external_data_used':False,'lockbox_access_authorized':False,'next_required_gate':'HARDEN_DISABLE_OBSOLETE_CLOUDFLARE_MUTATING_WORKFLOWS'}
rpath=OUT/'FINAL_FREEZE_RECONCILIATION.json'; rpath.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8'); rsha=sha(rpath); (OUT/'FINAL_FREEZE_RECONCILIATION.sha256').write_text(rsha+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','stage':'FINAL_FREEZE_RECONCILIATION','reconciliation_sha256':rsha,'final_freeze_manifest_sha256':actual_manifest_sha,'verified_artifacts':len(verified),'locked_test_used':False,'external_data_used':False,'lockbox_access_authorized':False},indent=2))
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
 const request:Rec={slug:TARGET_REF,newTitle:TARGET_TITLE,text:RECONCILE_SCRIPT,language:"python",kernelType:"script",kernelExecutionType:"SAVE_AND_RUN_ALL",isPrivate:true,enableGpu:false,enableTpu:false,enableInternet:false,kernelDataSources:[FINALIZATION_REF],datasetDataSources:[DATASET_REF],competitionDataSources:[],modelDataSources:[]};
 const response=await fetch(`${API_ROOT}/SaveKernel`,{method:"POST",headers:{Authorization:authHeader(token),"Content-Type":"application/json","User-Agent":"chatgpt-v622-final-reconcile/1.0"},body:JSON.stringify(request)});
 const text=await response.text(); let value:Rec={}; try{value=rec(text?JSON.parse(text):{});}catch{throw new Error(`SaveKernel non-JSON HTTP ${response.status}`);} const code=typeof value.code==="number"?value.code:undefined; if(!response.ok||(code!==undefined&&code>=400)) throw new Error(String(value.message??`SaveKernel HTTP ${response.status}`).slice(0,1200)); const badD=Array.isArray(value.invalidDatasetSources)?value.invalidDatasetSources:[]; const badK=Array.isArray(value.invalidKernelSources)?value.invalidKernelSources:[]; if(badD.length||badK.length) throw new Error(`SaveKernel rejected sources: datasets=${JSON.stringify(badD)} kernels=${JSON.stringify(badK)}`); return value;
}
async function outputReceipt(env:Env):Promise<Rec|null>{
 const token=env.CGP_KAGGLE_MASTER_TOKEN?.trim(); if(!token) throw new Error("master Kaggle token missing"); const response=await fetch(`${API_ROOT}/ListKernelSessionOutput`,{method:"POST",headers:{Authorization:authHeader(token),"Content-Type":"application/json","User-Agent":"chatgpt-v622-final-reconcile/1.0"},body:JSON.stringify({userName:OWNER,kernelSlug:TARGET_SLUG,pageSize:100})}); const payload=rec(await response.json()); if(!response.ok) throw new Error(`ListKernelSessionOutput HTTP ${response.status}`); const files=Array.isArray(payload.files)?payload.files.map(rec):[]; const rr=files.find(r=>String(r.fileName??"").endsWith("FINAL_FREEZE_RECONCILIATION.json")&&typeof r.url==="string"), sr=files.find(r=>String(r.fileName??"").endsWith("FINAL_FREEZE_RECONCILIATION.sha256")&&typeof r.url==="string"); if(!rr||!sr) return null; const text=await(await fetch(allowedOutputUrl(String(rr.url)),{redirect:"follow"})).text(); const declared=(await(await fetch(allowedOutputUrl(String(sr.url)),{redirect:"follow"})).text()).trim().toLowerCase(); const actual=await sha256Text(text); if(actual!==declared) throw new Error("reconciliation SHA mismatch"); const r=rec(JSON.parse(text)); if(r.project!=="PNEUMONIA V6.2.2"||r.stage!=="FINAL_FREEZE_RECONCILIATION"||r.status!=="PASS"||r.independent_reconciliation!==true) throw new Error("reconciliation identity/status mismatch"); if(r.final_freeze_manifest_sha256!==FINAL_MANIFEST_SHA||r.recipe_sha256!==RECIPE_SHA||r.frozen_policy_file_sha256!==POLICY_FILE_SHA||r.frozen_policy_semantic_sha256!==POLICY_SEMANTIC_SHA) throw new Error("reconciliation canonical binding mismatch"); if(r.locked_test_used!==false||r.external_data_used!==false||r.lockbox_access_authorized!==false) throw new Error("forbidden reconciliation authorization/evidence"); return {reconciliation_sha256:actual,final_freeze_manifest_sha256:r.final_freeze_manifest_sha256,verified_artifact_count:Array.isArray(r.verified_artifacts)?r.verified_artifacts.length:0,next_required_gate:r.next_required_gate,lockbox_access_authorized:r.lockbox_access_authorized};
}
async function launch(env:Env):Promise<Rec>{ const existing=await listKernels(env,"master",TARGET_SLUG,100); const exact=existing.some(i=>String(rec(i).ref??"").toLowerCase()===TARGET_REF.toLowerCase()); if(exact){const status=normalizedStatus(await kernelStatus(env,"master",TARGET_REF)); if(ACTIVE.has(status)) return {action:"existing_active",target_kernel_ref:TARGET_REF,status,submitted:false}; if(status==="COMPLETE"){const receipt=await outputReceipt(env); if(receipt) return {action:"existing_complete",target_kernel_ref:TARGET_REF,status,submitted:false,receipt};}} const result=await saveKernel(env); return {action:exact?"submitted_new_version":"created_and_submitted",target_kernel_ref:TARGET_REF,result,submitted:true,kaggle_compute_kind:"INDEPENDENT_FINAL_FREEZE_RECONCILIATION_ONLY",training:false,hpo:false,confirmation:false,locked_test:false,external_validation:false}; }
export default {async fetch(request:Request,env:Env):Promise<Response>{const url=new URL(request.url); if(url.pathname==="/healthz"&&request.method==="GET") return json({service:"v622-finalization-reconcile",status:"ready",protected:true}); if(!authorized(request,env)) return new Response("Forbidden",{status:403}); try{if(url.pathname==="/control/v6-2-2/finalization-reconcile/launch"&&request.method==="POST") return json({project:"PNEUMONIA V6.2.2",...(await launch(env))}); if(url.pathname==="/control/v6-2-2/finalization-reconcile/status"&&request.method==="POST"){const status=normalizedStatus(await kernelStatus(env,"master",TARGET_REF)); const receipt=status==="COMPLETE"?await outputReceipt(env):null; return json({project:"PNEUMONIA V6.2.2",target_kernel_ref:TARGET_REF,status,receipt});} return new Response("Not found",{status:404});}catch(error){return json({project:"PNEUMONIA V6.2.2",ok:false,error:error instanceof Error?error.message.slice(0,1400):"unknown error"},502);}}} satisfies ExportedHandler<Env>;
