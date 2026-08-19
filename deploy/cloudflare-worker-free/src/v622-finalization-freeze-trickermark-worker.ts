import { kernelStatus, listKernels, type WorkerEnv } from "./kaggle";

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };
type Rec = Record<string, unknown>;

const API_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService";
const ACCOUNT_ID = "kg-05";
const OWNER = "trickermark";
const TARGET_SLUG = "pneumonia-v6-2-2-finalization-master";
const TARGET_REF = `${OWNER}/${TARGET_SLUG}`;
const TARGET_TITLE = "PNEUMONIA V6.2.2 FINALIZATION MASTER";
const DATASET_REF = "trickermark/pneumonia-v6-2-2-finalization-artifacts";
const BACKBONE_REF = "trickermark/pneumonia-v6-2-2-backbone-m06-r224";
const RECIPE_SHA256 = "27cae8c59e06b609f9dfd02b526d807dc359b2bb14ae7bc5ee77c68c7553b08f";
const POLICY_FILE_SHA256 = "7e23bbed9246d5588c67a46380b570e1c1a3e869062613b9dcdb298e2e822861";
const POLICY_SEMANTIC_SHA256 = "7f5817f3429bb86668718cac9158a7061c6e730cd14bc6137402be82baf576d7";
const SELECTED = ["M06__convnext_tiny", "M06__densenet121", "M06__resnet50v2"] as const;
const ACTIVE = new Set(["RUNNING", "QUEUED", "STARTING", "PENDING"]);

const FREEZE_SCRIPT = String.raw`from __future__ import annotations
import hashlib, json
from pathlib import Path
INPUT=Path('/kaggle/input')
OUT=Path('/kaggle/working/PNEUMONIA_V62_2_FINALIZATION_MASTER'); OUT.mkdir(parents=True,exist_ok=True)
DATASET_REF='trickermark/pneumonia-v6-2-2-finalization-artifacts'
BACKBONE_REF='trickermark/pneumonia-v6-2-2-backbone-m06-r224'
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
policy_path=unique('FROZEN_BACKBONE_ENSEMBLE_POLICY.json')
if sha(policy_path)!=POLICY_FILE_SHA: raise RuntimeError('policy file SHA drift')
policy=json.loads(policy_path.read_text(encoding='utf-8'))
if policy.get('status')!='FROZEN_FROM_DEVELOPMENT_ONLY': raise RuntimeError('policy status drift')
if policy.get('recipe_sha256')!=RECIPE_SHA or policy.get('policy_sha256')!=POLICY_SEMANTIC_SHA: raise RuntimeError('policy identity drift')
if policy.get('selected_candidates')!=SELECTED: raise RuntimeError('selected candidates drift')
if policy.get('locked_test_used') is not False or policy.get('external_data_used') is not False: raise RuntimeError('forbidden policy evidence')
fit=policy.get('deployment_fit') or {}
if fit.get('weights')!=[1/3,1/3,1/3] or len(fit.get('calibration') or [])!=3 or not isinstance(fit.get('threshold'),(int,float)): raise RuntimeError('deployment fit drift')
rows=[]
for cid in SELECTED:
 config=unique(cid+'__config.json'); checkpoint=unique(cid+'__final_selected.keras')
 csha=sha(config)
 if csha!=CONFIG_SHA[cid]: raise RuntimeError(cid+': config SHA mismatch')
 rows.extend([{'candidate_id':cid,'role':'config','basename':'config.json','bytes':config.stat().st_size,'sha256':csha,'dataset_file':config.name},{'candidate_id':cid,'role':'checkpoint','basename':'final_selected.keras','bytes':checkpoint.stat().st_size,'sha256':sha(checkpoint),'dataset_file':checkpoint.name}])
qualification={'schema_version':1,'project':'PNEUMONIA V6.2.2','qualification_performed':False,'used_for_selection':False,'final_lockbox_used':False,'external_data_used':False}
qpath=OUT/'QUALIFICATION_NON_CONSUMPTION.json'; qpath.write_text(json.dumps(qualification,indent=2,sort_keys=True)+'\n',encoding='utf-8'); qsha=sha(qpath)
manifest={'schema_version':1,'project':'PNEUMONIA V6.2.2','stage':'FINAL_FREEZE_MANIFEST','status':'FROZEN_IN_KAGGLE','master_account':'trickermark','source_artifact_dataset':DATASET_REF,'provenance_source_kernel':BACKBONE_REF,'recipe_sha256':RECIPE_SHA,'frozen_policy_file_sha256':POLICY_FILE_SHA,'frozen_policy_semantic_sha256':POLICY_SEMANTIC_SHA,'selected_candidate_ids':SELECTED,'selected_artifact_count':6,'artifacts':rows,'deployment_fit':fit,'qualification_non_consumption_sha256':qsha,'qualification_performed':False,'used_for_selection':False,'locked_test_used':False,'external_data_used':False,'training_performed':False,'hpo_or_confirmation_repeated':False,'reselection_allowed_after_freeze':False,'next_permitted_stage':'INDEPENDENT_FREEZE_RECONCILIATION'}
mpath=OUT/'FINAL_FREEZE_MANIFEST.json'; mpath.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8'); msha=sha(mpath)
(OUT/'FINAL_FREEZE_MANIFEST.sha256').write_text(msha+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','stage':'FINAL_FREEZE_MANIFEST','manifest_sha256':msha,'selected_artifacts':6,'qualification_performed':False,'used_for_selection':False,'locked_test_used':False,'external_data_used':False,'next_permitted_stage':'INDEPENDENT_FREEZE_RECONCILIATION'},indent=2))
`;

function rec(v: unknown): Rec { return v && typeof v === "object" && !Array.isArray(v) ? v as Rec : {}; }
function json(v: unknown, status = 200): Response { return new Response(JSON.stringify(v), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } }); }
function authorized(request: Request, env: Env): boolean { const token=env.CGP_PROJECT_CONTROL_TOKEN?.trim(); return Boolean(token&&token.length>=32&&request.headers.get("authorization")===`Bearer ${token}`); }
function authHeader(token: string): string { return token.startsWith("KGAT_") ? `Bearer ${token}` : `Basic ${btoa(`${OWNER}:${token}`)}`; }
function normalizedStatus(raw: Rec): string { const s=rec(raw.session); for(const v of [raw.status,raw.statusName,raw.state,raw.sessionStatus,s.status,s.statusName,s.state]) if(typeof v==="string"&&v.trim()) return v.trim().toUpperCase(); return "UNKNOWN"; }
function allowedOutputUrl(raw: string): URL { const u=new URL(raw),h=u.hostname.toLowerCase(); if(u.protocol!=="https:"||!(h==="api.kaggle.com"||h==="www.kaggle.com"||h==="storage.googleapis.com"||h.endsWith(".kaggleusercontent.com")||h.endsWith(".googleusercontent.com"))) throw new Error("output URL host not allowlisted"); return u; }
async function sha256Text(text:string):Promise<string>{const d=await crypto.subtle.digest("SHA-256",new TextEncoder().encode(text));return Array.from(new Uint8Array(d),b=>b.toString(16).padStart(2,"0")).join("");}
async function saveKernel(env: Env): Promise<Rec> {
 const token=env.CGP_KAGGLE_KG05_TOKEN?.trim(); if(!token) throw new Error("kg-05 Kaggle token missing");
 const request:Rec={slug:TARGET_REF,newTitle:TARGET_TITLE,text:FREEZE_SCRIPT,language:"python",kernelType:"script",kernelExecutionType:"SAVE_AND_RUN_ALL",isPrivate:true,enableGpu:false,enableTpu:false,enableInternet:false,kernelDataSources:[],datasetDataSources:[DATASET_REF],competitionDataSources:[],modelDataSources:[]};
 const response=await fetch(`${API_ROOT}/SaveKernel`,{method:"POST",headers:{Authorization:authHeader(token),"Content-Type":"application/json","User-Agent":"chatgpt-v622-finalization-freeze-same-owner/1.0"},body:JSON.stringify(request)});
 const text=await response.text(); let value:Rec={}; try{value=rec(text?JSON.parse(text):{});}catch{throw new Error(`SaveKernel non-JSON HTTP ${response.status}`);}
 const code=typeof value.code==="number"?value.code:undefined; if(!response.ok||(code!==undefined&&code>=400)) throw new Error(String(value.message??`SaveKernel HTTP ${response.status}`).slice(0,1000));
 const badDatasets=Array.isArray(value.invalidDatasetSources)?value.invalidDatasetSources:[]; const badKernels=Array.isArray(value.invalidKernelSources)?value.invalidKernelSources:[];
 if(badDatasets.length||badKernels.length) throw new Error(`SaveKernel rejected data sources: datasets=${JSON.stringify(badDatasets)} kernels=${JSON.stringify(badKernels)}`);
 return value;
}
async function outputReceipt(env:Env):Promise<Rec|null>{
 const token=env.CGP_KAGGLE_KG05_TOKEN?.trim(); if(!token) throw new Error("kg-05 Kaggle token missing");
 const response=await fetch(`${API_ROOT}/ListKernelSessionOutput`,{method:"POST",headers:{Authorization:authHeader(token),"Content-Type":"application/json","User-Agent":"chatgpt-v622-finalization-freeze-same-owner/1.0"},body:JSON.stringify({userName:OWNER,kernelSlug:TARGET_SLUG,pageSize:100})});
 const payload=rec(await response.json()); if(!response.ok) throw new Error(`ListKernelSessionOutput HTTP ${response.status}`); const files=Array.isArray(payload.files)?payload.files.map(rec):[];
 const mr=files.find(r=>String(r.fileName??"").endsWith("FINAL_FREEZE_MANIFEST.json")&&typeof r.url==="string"), sr=files.find(r=>String(r.fileName??"").endsWith("FINAL_FREEZE_MANIFEST.sha256")&&typeof r.url==="string"); if(!mr||!sr) return null;
 const manifestResponse=await fetch(allowedOutputUrl(String(mr.url)),{redirect:"follow"}); if(!manifestResponse.ok) throw new Error(`manifest download HTTP ${manifestResponse.status}`); const manifestText=await manifestResponse.text();
 const shaResponse=await fetch(allowedOutputUrl(String(sr.url)),{redirect:"follow"}); if(!shaResponse.ok) throw new Error(`manifest SHA download HTTP ${shaResponse.status}`); const declared=(await shaResponse.text()).trim().toLowerCase(); const actual=await sha256Text(manifestText); if(actual!==declared) throw new Error("FINAL_FREEZE_MANIFEST SHA mismatch");
 const m=rec(JSON.parse(manifestText)); if(m.project!=="PNEUMONIA V6.2.2"||m.stage!=="FINAL_FREEZE_MANIFEST"||m.status!=="FROZEN_IN_KAGGLE") throw new Error("FINAL_FREEZE identity/status mismatch");
 if(m.master_account!==OWNER||m.source_artifact_dataset!==DATASET_REF||m.provenance_source_kernel!==BACKBONE_REF||m.recipe_sha256!==RECIPE_SHA256||m.frozen_policy_file_sha256!==POLICY_FILE_SHA256||m.frozen_policy_semantic_sha256!==POLICY_SEMANTIC_SHA256) throw new Error("FINAL_FREEZE canonical identity mismatch");
 if(m.qualification_performed!==false||m.used_for_selection!==false||m.locked_test_used!==false||m.external_data_used!==false) throw new Error("forbidden evidence in FINAL_FREEZE");
 return {manifest_sha256:actual,selected_artifact_count:m.selected_artifact_count,selected_candidate_ids:m.selected_candidate_ids,next_permitted_stage:m.next_permitted_stage,deployment_fit:m.deployment_fit,artifacts:m.artifacts,qualification_non_consumption_sha256:m.qualification_non_consumption_sha256};
}
async function launch(env:Env):Promise<Rec>{
 const existing=await listKernels(env,ACCOUNT_ID,TARGET_SLUG,100); const exact=existing.some(i=>String(rec(i).ref??"").toLowerCase()===TARGET_REF.toLowerCase());
 if(exact){const status=normalizedStatus(await kernelStatus(env,ACCOUNT_ID,TARGET_REF)); if(ACTIVE.has(status)) return {action:"existing_active",target_kernel_ref:TARGET_REF,status,submitted:false}; if(status==="COMPLETE"){const receipt=await outputReceipt(env); if(receipt) return {action:"existing_complete",target_kernel_ref:TARGET_REF,status,submitted:false,receipt};}}
 const result=await saveKernel(env); return {action:exact?"submitted_new_version":"created_and_submitted",target_kernel_ref:TARGET_REF,artifact_dataset_ref:DATASET_REF,provenance_source_kernel:BACKBONE_REF,result,submitted:true,kaggle_compute_kind:"FINALIZATION_FREEZE_ONLY",training:false,hpo:false,confirmation:false,locked_test:false,external_validation:false};
}

export default {async fetch(request:Request,env:Env):Promise<Response>{const url=new URL(request.url); if(url.pathname==="/healthz"&&request.method==="GET")return json({service:"v622-finalization-freeze-same-owner",status:"ready",protected:true}); if(!authorized(request,env))return new Response("Forbidden",{status:403}); try{if(url.pathname==="/control/v6-2-2/finalization-freeze-same-owner/launch"&&request.method==="POST")return json({project:"PNEUMONIA V6.2.2",...(await launch(env))}); if(url.pathname==="/control/v6-2-2/finalization-freeze-same-owner/status"&&request.method==="POST"){const status=normalizedStatus(await kernelStatus(env,ACCOUNT_ID,TARGET_REF)); const receipt=status==="COMPLETE"?await outputReceipt(env):null; return json({project:"PNEUMONIA V6.2.2",target_kernel_ref:TARGET_REF,status,receipt});} return new Response("Not found",{status:404});}catch(error){return json({project:"PNEUMONIA V6.2.2",ok:false,error:error instanceof Error?error.message.slice(0,1200):"unknown error"},502);}}} satisfies ExportedHandler<Env>;
