import { kernelLogs, kernelStatus, type WorkerEnv } from "./kaggle";

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };
const ACCOUNT_ID = "kg-05";
const KERNEL_REF = "trickermark/pneumonia-v6-2-2-publish-finalization-artifacts";
function authorized(request: Request, env: Env): boolean { const t=env.CGP_PROJECT_CONTROL_TOKEN?.trim(); return Boolean(t&&t.length>=32&&request.headers.get("authorization")===`Bearer ${t}`); }
function json(v:unknown,status=200){return new Response(JSON.stringify(v),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}});}
export default {async fetch(request:Request,env:Env):Promise<Response>{const u=new URL(request.url);if(u.pathname==="/healthz")return json({status:"ready",read_only:true});if(u.pathname!=="/control/v6-2-2/publisher-log-probe"||request.method!=="POST")return new Response("Not found",{status:404});if(!authorized(request,env))return new Response("Forbidden",{status:403});try{const [status,log]=await Promise.all([kernelStatus(env,ACCOUNT_ID,KERNEL_REF),kernelLogs(env,ACCOUNT_ID,KERNEL_REF)]);return json({project:"PNEUMONIA V6.2.2",kernel_ref:KERNEL_REF,status,log_tail:log.slice(-30000),read_only:true,model_compute:false});}catch(error){return json({ok:false,error:error instanceof Error?error.message.slice(0,1200):"unknown error",read_only:true},502);}}} satisfies ExportedHandler<Env>;
