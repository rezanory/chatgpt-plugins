import { kernelStatus, listKernels, type WorkerEnv } from "./kaggle";

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };
type Rec = Record<string, unknown>;

const API_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService";
const ACCOUNT_ID = "kg-05";
const OWNER = "trickermark";
const SOURCE_REF = "trickermark/pneumonia-v6-2-2-backbone-m06-r224";
const TARGET_SLUG = "pneumonia-v6-2-2-publish-finalization-artifacts";
const TARGET_REF = `${OWNER}/${TARGET_SLUG}`;
const TARGET_TITLE = "PNEUMONIA V6.2.2 PUBLISH FINALIZATION ARTIFACTS";
const DATASET_REF = "trickermark/pneumonia-v6-2-2-finalization-artifacts";
const ACTIVE = new Set(["RUNNING", "QUEUED", "STARTING", "PENDING"]);

const PUBLISH_SCRIPT = String.raw`from __future__ import annotations
import hashlib, json, shutil, time
from pathlib import Path

INPUT=Path('/kaggle/input')
STAGE=Path('/kaggle/working/v622_finalization_artifacts')
STAGE.mkdir(parents=True, exist_ok=True)
DATASET_REF='trickermark/pneumonia-v6-2-2-finalization-artifacts'
SOURCE_REF='trickermark/pneumonia-v6-2-2-backbone-m06-r224'
SELECTED=['M06__convnext_tiny','M06__densenet121','M06__resnet50v2']
CONFIG_SHA={
 'M06__convnext_tiny':'167cb7b1d17b3978f2b4f47f55e315a1d4fae293006d897d25d13b5074dadcea',
 'M06__densenet121':'c9b3107715849a633d90689902298c110bbb7c9c1bc1c8a6ac9ff1693f55386e',
 'M06__resnet50v2':'dce7845aa7c07ccb913cded8e9ff368e7061f3a43723ded21f8a94dba9fab697'}
POLICY_SHA='7e23bbed9246d5588c67a46380b570e1c1a3e869062613b9dcdb298e2e822861'

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
 return h.hexdigest()

def unique_suffix(suffix:str)->Path:
 xs=[p for p in INPUT.rglob('*') if p.is_file() and p.as_posix().endswith(suffix)]
 if len(xs)!=1: raise RuntimeError(f'{suffix}: expected exactly one source file, found {len(xs)}')
 return xs[0]

rows=[]
for cid in SELECTED:
 run=cid+'__seed-42'
 for role,base in [('config','config.json'),('checkpoint','final_selected.keras')]:
  src=unique_suffix(f'/backbone_comparison/{run}/{base}')
  dst=STAGE/(cid+'__'+('config.json' if role=='config' else 'final_selected.keras'))
  shutil.copy2(src,dst)
  digest=sha(dst)
  if role=='config' and digest!=CONFIG_SHA[cid]: raise RuntimeError(cid+': config SHA mismatch')
  if dst.stat().st_size<=0: raise RuntimeError(cid+': empty '+role)
  rows.append({'candidate_id':cid,'role':role,'file':dst.name,'bytes':dst.stat().st_size,'sha256':digest})
policy_src=unique_suffix('/FROZEN_BACKBONE_ENSEMBLE_POLICY.json')
policy_dst=STAGE/'FROZEN_BACKBONE_ENSEMBLE_POLICY.json'
shutil.copy2(policy_src,policy_dst)
if sha(policy_dst)!=POLICY_SHA: raise RuntimeError('frozen policy SHA mismatch')

import kagglehub
print('Kaggle identity:', kagglehub.whoami(), flush=True)
kagglehub.dataset_upload(DATASET_REF, str(STAGE), version_notes='PNEUMONIA V6.2.2 frozen finalization artifacts from backbone output')

from kagglehub.clients import build_kaggle_client
from kagglesdk.datasets.types.dataset_api_service import ApiUpdateDatasetMetadataRequest
from kagglesdk.datasets.types.dataset_types import DatasetSettings, DatasetCollaborator
from kagglesdk.users.types.users_enums import CollaboratorType
settings=DatasetSettings()
settings.title='PNEUMONIA V6.2.2 Finalization Artifacts'
settings.description='Private frozen selected artifacts for PNEUMONIA V6.2.2 finalization.'
settings.is_private=True
collab=DatasetCollaborator(); collab.username='azadka'; collab.role=CollaboratorType.READER
settings.collaborators=[collab]
req=ApiUpdateDatasetMetadataRequest(); req.owner_slug='trickermark'; req.dataset_slug='pneumonia-v6-2-2-finalization-artifacts'; req.settings=settings
with build_kaggle_client() as client:
 resp=client.datasets.dataset_api_client.update_dataset_metadata(req)
 errors=list(getattr(resp,'errors',[]) or [])
 if errors: raise RuntimeError('collaborator update failed: '+str(errors)[:1000])

receipt={
 'project':'PNEUMONIA V6.2.2','stage':'KAGGLE_NATIVE_ARTIFACT_PUBLISH','status':'PASS',
 'source_kernel':SOURCE_REF,'dataset_ref':DATASET_REF,'dataset_owner':'trickermark',
 'reader':'azadka','artifact_count':7,'selected_artifact_count':6,'artifacts':rows,
 'policy_sha256':POLICY_SHA,'github_artifact_storage_used':False,
 'github_checkpoint_transport_used':False,'model_compute_performed':False,
 'training_hpo_confirmation':False,'locked_test_used':False,'external_validation_used':False,
 'next_stage':'FINAL_FREEZE_MANIFEST'}
Path('/kaggle/working/KAGGLE_NATIVE_ARTIFACT_PUBLISH_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','dataset_ref':DATASET_REF,'reader':'azadka','artifact_count':7},indent=2),flush=True)
`;

function rec(v: unknown): Rec { return v && typeof v === "object" && !Array.isArray(v) ? v as Rec : {}; }
function json(v: unknown, status = 200): Response { return new Response(JSON.stringify(v), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } }); }
function authorized(request: Request, env: Env): boolean { const token=env.CGP_PROJECT_CONTROL_TOKEN?.trim(); return Boolean(token&&token.length>=32&&request.headers.get("authorization")===`Bearer ${token}`); }
function authHeader(token: string): string { return token.startsWith("KGAT_") ? `Bearer ${token}` : `Basic ${btoa(`${OWNER}:${token}`)}`; }
function normalizedStatus(raw: Rec): string { const s=rec(raw.session); for(const v of [raw.status,raw.statusName,raw.state,raw.sessionStatus,s.status,s.statusName,s.state]) if(typeof v==="string"&&v.trim()) return v.trim().toUpperCase(); return "UNKNOWN"; }
function allowedOutputUrl(raw:string):URL{const u=new URL(raw),h=u.hostname.toLowerCase(); if(u.protocol!=="https:"||!(h==="api.kaggle.com"||h==="www.kaggle.com"||h==="storage.googleapis.com"||h.endsWith(".kaggleusercontent.com")||h.endsWith(".googleusercontent.com"))) throw new Error("output URL host not allowlisted"); return u;}

async function saveKernel(env: Env): Promise<Rec> {
 const token=env.CGP_KAGGLE_KG05_TOKEN?.trim(); if(!token) throw new Error("trickermark Kaggle token missing");
 const body:Rec={slug:TARGET_REF,newTitle:TARGET_TITLE,text:PUBLISH_SCRIPT,language:"python",kernelType:"script",kernelExecutionType:"SAVE_AND_RUN_ALL",isPrivate:true,enableGpu:false,enableTpu:false,enableInternet:true,kernelDataSources:[SOURCE_REF],datasetDataSources:[],competitionDataSources:[],modelDataSources:[]};
 const response=await fetch(`${API_ROOT}/SaveKernel`,{method:"POST",headers:{Authorization:authHeader(token),"Content-Type":"application/json","User-Agent":"chatgpt-v622-kaggle-native-publish/1.0"},body:JSON.stringify(body)});
 const text=await response.text(); let value:Rec={}; try{value=rec(text?JSON.parse(text):{});}catch{throw new Error(`SaveKernel non-JSON HTTP ${response.status}`);}
 const code=typeof value.code==="number"?value.code:undefined; if(!response.ok||(code!==undefined&&code>=400)) throw new Error(String(value.message??`SaveKernel HTTP ${response.status}`).slice(0,1200));
 const bad=Array.isArray(value.invalidKernelSources)?value.invalidKernelSources:[]; if(bad.length) throw new Error(`SaveKernel rejected source: ${JSON.stringify(bad)}`);
 return value;
}

async function receipt(env:Env):Promise<Rec|null>{
 const token=env.CGP_KAGGLE_KG05_TOKEN?.trim(); if(!token) throw new Error("trickermark Kaggle token missing");
 const response=await fetch(`${API_ROOT}/ListKernelSessionOutput`,{method:"POST",headers:{Authorization:authHeader(token),"Content-Type":"application/json","User-Agent":"chatgpt-v622-kaggle-native-publish/1.0"},body:JSON.stringify({userName:OWNER,kernelSlug:TARGET_SLUG,pageSize:100})});
 const payload=rec(await response.json()); if(!response.ok) throw new Error(`ListKernelSessionOutput HTTP ${response.status}`); const files=Array.isArray(payload.files)?payload.files.map(rec):[];
 const row=files.find(r=>String(r.fileName??"").endsWith("KAGGLE_NATIVE_ARTIFACT_PUBLISH_RECEIPT.json")&&typeof r.url==="string"); if(!row) return null;
 const r=await fetch(allowedOutputUrl(String(row.url)),{redirect:"follow"}); if(!r.ok) throw new Error(`receipt download HTTP ${r.status}`); const v=rec(JSON.parse(await r.text()));
 if(v.project!=="PNEUMONIA V6.2.2"||v.stage!=="KAGGLE_NATIVE_ARTIFACT_PUBLISH"||v.status!=="PASS"||v.dataset_ref!==DATASET_REF||v.reader!=="azadka"||v.artifact_count!==7) throw new Error("Kaggle-native publish receipt mismatch");
 return v;
}

async function launch(env:Env):Promise<Rec>{
 const existing=await listKernels(env,ACCOUNT_ID,TARGET_SLUG,100); const exact=existing.some(i=>String(rec(i).ref??"").toLowerCase()===TARGET_REF.toLowerCase());
 if(exact){const status=normalizedStatus(await kernelStatus(env,ACCOUNT_ID,TARGET_REF)); if(ACTIVE.has(status)) return {action:"existing_active",target_kernel_ref:TARGET_REF,status,submitted:false}; if(status==="COMPLETE"){const r=await receipt(env); if(r) return {action:"existing_complete",target_kernel_ref:TARGET_REF,status,submitted:false,receipt:r};}}
 const result=await saveKernel(env); return {action:exact?"submitted_new_version":"created_and_submitted",target_kernel_ref:TARGET_REF,source_kernel_ref:SOURCE_REF,dataset_ref:DATASET_REF,result,submitted:true,kaggle_compute_kind:"ARTIFACT_PUBLISH_ONLY",model_compute:false,training:false,hpo:false,confirmation:false,locked_test:false,external_validation:false};
}

export default {async fetch(request:Request,env:Env):Promise<Response>{const url=new URL(request.url); if(url.pathname==="/healthz"&&request.method==="GET")return json({service:"v622-kaggle-native-artifact-publish",status:"ready",protected:true}); if(!authorized(request,env))return new Response("Forbidden",{status:403}); try{if(url.pathname==="/control/v6-2-2/kaggle-native-artifact-publish/launch"&&request.method==="POST")return json({project:"PNEUMONIA V6.2.2",...(await launch(env))}); if(url.pathname==="/control/v6-2-2/kaggle-native-artifact-publish/status"&&request.method==="POST"){const status=normalizedStatus(await kernelStatus(env,ACCOUNT_ID,TARGET_REF)); const r=status==="COMPLETE"?await receipt(env):null; return json({project:"PNEUMONIA V6.2.2",target_kernel_ref:TARGET_REF,status,receipt:r});} return new Response("Not found",{status:404});}catch(error){return json({project:"PNEUMONIA V6.2.2",ok:false,error:error instanceof Error?error.message.slice(0,1400):"unknown error",model_compute:false},502);}}} satisfies ExportedHandler<Env>;
