import fs from 'node:fs';

const [workerPathArg]=process.argv.slice(2);
if(!workerPathArg) throw new Error('usage: node v622_patch_external_worker_repair_v2.mjs <worker.ts>');
const workerPath=workerPathArg;
let text=fs.readFileSync(workerPath,'utf8');
const oldClause="if(TERMINAL_BAD.has(status))throw new Error(`external validation already terminal ${status}; automatic rerun forbidden`);";
const newClause="if(TERMINAL_BAD.has(status)){if(!EXTERNAL_VALIDATION_SCRIPT.includes(\"'incident_repair':'NIH_MOUNT_PACKAGING_V2'\")||!EXTERNAL_VALIDATION_SCRIPT.includes(\"'prior_failed_kernel_version':1\"))throw new Error('terminal external repair requested without immutable v2 incident lineage');const result=await saveKernel(env);return{action:'incident_repair_v2_submitted',target_kernel_ref:TARGET_REF,prior_status:status,prior_failed_kernel_version:1,incident_repair:'NIH_MOUNT_PACKAGING_V2',repair_scope:'NIH input packaging discovery only; frozen model/policy unchanged',result,submitted:true,kaggle_compute_kind:'EXTERNAL_VALIDATION_REPAIR_V2',gpu:true,internet:false,cohorts:[CHEXPERT_DATASET,NIH_DATASET],training:false,hpo:false,confirmation:false,reselection:false,threshold_tuning:false,script_sha256:EXTERNAL_VALIDATION_SCRIPT_SHA256,w16_notebook_sha256:W16_NOTEBOOK_SHA256,w16_source_zip_sha256:W16_SOURCE_ZIP_SHA256}};";
if(!text.includes(oldClause)) throw new Error('expected v1 terminal-rerun guard not found');
if(text.indexOf(oldClause)!==text.lastIndexOf(oldClause)) throw new Error('v1 terminal-rerun guard is not unique');
text=text.replace(oldClause,newClause);
const oldStatus="automatic_rerun_forbidden:true";
if(!text.includes(oldStatus)) throw new Error('expected status governance marker missing');
text=text.replaceAll(oldStatus,"automatic_rerun_forbidden:true,incident_repair_v2_authorized:true");
fs.writeFileSync(workerPath,text,'utf8');
console.log(JSON.stringify({stage:'EXTERNAL_VALIDATION_WORKER_REPAIR',incident_repair:'NIH_MOUNT_PACKAGING_V2',prior_failed_kernel_version:1,repair_scope:'NIH input packaging discovery only; frozen model/policy unchanged'},null,2));
