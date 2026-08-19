import { type WorkerEnv } from "./kaggle";

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };
type Rec = Record<string, unknown>;
const OWNER = "trickermark";
const DATASET_SLUG = "pneumonia-v6-2-2-finalization-artifacts";
const DATASET_REF = `${OWNER}/${DATASET_SLUG}`;
const READER = "azadka";
const API_ROOT = "https://api.kaggle.com/v1/datasets.DatasetApiService";

function rec(v: unknown): Rec { return v && typeof v === "object" && !Array.isArray(v) ? v as Rec : {}; }
function json(v: unknown, status=200){ return new Response(JSON.stringify(v), {status, headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}}); }
function authorized(req:Request,env:Env){const t=env.CGP_PROJECT_CONTROL_TOKEN?.trim(); return Boolean(t&&t.length>=32&&req.headers.get("authorization")===`Bearer ${t}`);}
function authHeader(token:string){return token.startsWith("KGAT_")?`Bearer ${token}`:`Basic ${btoa(`${OWNER}:${token}`)}`;}
async function call(env:Env,method:string,body:Rec):Promise<Rec>{const token=env.CGP_KAGGLE_KG05_TOKEN?.trim();if(!token)throw new Error("trickermark Kaggle token missing");const r=await fetch(`${API_ROOT}/${method}`,{method:"POST",headers:{Authorization:authHeader(token),"Content-Type":"application/json","User-Agent":"chatgpt-v622-dataset-permission/1.0"},body:JSON.stringify(body)});const text=await r.text();let v:Rec={};try{v=rec(text?JSON.parse(text):{});}catch{throw new Error(`${method} non-JSON HTTP ${r.status}: ${text.slice(0,500)}`);}if(!r.ok)throw new Error(`${method} HTTP ${r.status}: ${String(v.message??text).slice(0,900)}`);return v;}
async function grant(env:Env){
  const before = await call(env,"GetDataset",{ownerSlug:OWNER,datasetSlug:DATASET_SLUG});
  const update = await call(env,"UpdateDatasetMetadata",{ownerSlug:OWNER,datasetSlug:DATASET_SLUG,settings:{collaborators:[{username:READER,role:1}]}});
  const after = await call(env,"GetDataset",{ownerSlug:OWNER,datasetSlug:DATASET_SLUG});
  return {project:"PNEUMONIA V6.2.2",status:"PASS",dataset_ref:DATASET_REF,reader:READER,role:"READER",role_value:1,before,update,after,model_compute:false,training:false,hpo:false,confirmation:false,locked_test:false,external_validation:false};
}
export default {async fetch(request:Request,env:Env):Promise<Response>{const u=new URL(request.url);if(u.pathname==="/healthz")return json({service:"v622-dataset-permission",status:"ready",protected:true});if(u.pathname!=="/control/v6-2-2/dataset-permission/grant"||request.method!=="POST")return new Response("Not found",{status:404});if(!authorized(request,env))return new Response("Forbidden",{status:403});try{return json(await grant(env));}catch(error){return json({project:"PNEUMONIA V6.2.2",ok:false,error:error instanceof Error?error.message.slice(0,1400):"unknown error",dataset_ref:DATASET_REF,model_compute:false},502);}}} satisfies ExportedHandler<Env>;
