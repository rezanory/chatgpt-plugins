import fs from 'node:fs';

const [workerPath]=process.argv.slice(2);
if(!workerPath) throw new Error('usage: node v622_patch_external_worker_repair_v3.mjs <worker.ts>');
let text=fs.readFileSync(workerPath,'utf8');
const oldClause="if(TERMINAL_BAD.has(status))throw new Error(`external validation already terminal ${status}; automatic rerun forbidden`);";
const newClause="if(TERMINAL_BAD.has(status)){if(!EXTERNAL_VALIDATION_SCRIPT.includes(\"'incident_repair':'NIH_LABELED_ALIAS_V3'\")||!EXTERNAL_VALIDATION_SCRIPT.includes(\"'prior_failed_kernel_version':2\"))throw new Error('terminal external repair requested without immutable v3 incident lineage');const result=await saveKernel(env);return{action:'incident_repair_v3_submitted',target_kernel_ref:TARGET_REF,prior_status:status,prior_failed_kernel_version:2,incident_repair:'NIH_LABELED_ALIAS_V3',repair_scope:'NIH labeled Image Index alias resolution only; byte-identical aliases accepted deterministically; frozen model/policy unchanged',result,submitted:true,kaggle_compute_kind:'EXTERNAL_VALIDATION_REPAIR_V3',gpu:true,internet:false,cohorts:[CHEXPERT_DATASET,NIH_DATASET],training:false,hpo:false,confirmation:false,reselection:false,threshold_tuning:false,script_sha256:EXTERNAL_VALIDATION_SCRIPT_SHA256,w16_notebook_sha256:W16_NOTEBOOK_SHA256,w16_source_zip_sha256:W16_SOURCE_ZIP_SHA256}};";
if(!text.includes(oldClause)) throw new Error('expected terminal-rerun guard not found');
if(text.indexOf(oldClause)!==text.lastIndexOf(oldClause)) throw new Error('terminal-rerun guard is not unique');
text=text.replace(oldClause,newClause);
const oldReceipt="return{status:'PASS',stage:'EXTERNAL_VALIDATION',summary_sha256:actual,cohorts:v.cohorts,independence_audit:v.independence_audit,next_permitted_stage:v.next_permitted_stage};";
const newReceipt="const alias=rec(v.nih_alias_resolution);if(v.incident_repair!=='NIH_LABELED_ALIAS_V3'||Number(v.prior_failed_kernel_version)!==2)throw new Error('external validation repair-v3 lineage mismatch');if(Number(alias.missing_count)!==0||Number(alias.conflict_count)!==0||Number(alias.indexed_images)!==Number(alias.labeled_image_index_count))throw new Error('NIH alias-resolution receipt mismatch');return{status:'PASS',stage:'EXTERNAL_VALIDATION',summary_sha256:actual,cohorts:v.cohorts,independence_audit:v.independence_audit,incident_repair:v.incident_repair,prior_failed_kernel_version:v.prior_failed_kernel_version,nih_alias_resolution:v.nih_alias_resolution,next_permitted_stage:v.next_permitted_stage};";
if(!text.includes(oldReceipt)) throw new Error('expected external receipt return not found');
text=text.replace(oldReceipt,newReceipt);
text=text.replaceAll('automatic_rerun_forbidden:true','automatic_rerun_forbidden:true,incident_repair_v3_authorized:true');
fs.writeFileSync(workerPath,text,'utf8');
console.log(JSON.stringify({stage:'EXTERNAL_VALIDATION_WORKER_REPAIR',incident_repair:'NIH_LABELED_ALIAS_V3',prior_failed_kernel_version:2,receipt_requires_zero_missing_and_conflicts:true},null,2));
