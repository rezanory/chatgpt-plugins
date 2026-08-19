import { kernelStatus, listKernels, type WorkerEnv } from "./kaggle";

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };

const API_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService";
const OWNER = "azadka";
const TARGET_SLUG = "pneumonia-v6-2-2-finalization-master";
const TARGET_REF = `${OWNER}/${TARGET_SLUG}`;
const TARGET_TITLE = "PNEUMONIA V6.2.2 FINALIZATION MASTER";
const BACKBONE_REF = "trickermark/pneumonia-v6-2-2-backbone-m06-r224";
const RECIPE_SHA256 = "27cae8c59e06b609f9dfd02b526d807dc359b2bb14ae7bc5ee77c68c7553b08f";
const POLICY_FILE_SHA256 = "7e23bbed9246d5588c67a46380b570e1c1a3e869062613b9dcdb298e2e822861";
const POLICY_SEMANTIC_SHA256 = "7f5817f3429bb86668718cac9158a7061c6e730cd14bc6137402be82baf576d7";
const SELECTED = ["M06__convnext_tiny", "M06__densenet121", "M06__resnet50v2"] as const;
const ACTIVE = new Set(["RUNNING", "QUEUED", "STARTING", "PENDING"]);

const FREEZE_SCRIPT = String.raw`from __future__ import annotations
import hashlib, json
from pathlib import Path

INPUT = Path('/kaggle/input')
OUT = Path('/kaggle/working/PNEUMONIA_V62_2_FINALIZATION_MASTER')
OUT.mkdir(parents=True, exist_ok=True)
BACKBONE_REF = 'trickermark/pneumonia-v6-2-2-backbone-m06-r224'
RECIPE_SHA = '27cae8c59e06b609f9dfd02b526d807dc359b2bb14ae7bc5ee77c68c7553b08f'
POLICY_FILE_SHA = '7e23bbed9246d5588c67a46380b570e1c1a3e869062613b9dcdb298e2e822861'
POLICY_SEMANTIC_SHA = '7f5817f3429bb86668718cac9158a7061c6e730cd14bc6137402be82baf576d7'
SELECTED = ['M06__convnext_tiny','M06__densenet121','M06__resnet50v2']
CONFIG_SHA = {
  'M06__convnext_tiny':'167cb7b1d17b3978f2b4f47f55e315a1d4fae293006d897d25d13b5074dadcea',
  'M06__densenet121':'c9b3107715849a633d90689902298c110bbb7c9c1bc1c8a6ac9ff1693f55386e',
  'M06__resnet50v2':'dce7845aa7c07ccb913cded8e9ff368e7061f3a43723ded21f8a94dba9fab697',
}

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def unique_policy() -> Path:
    matches=[]
    for p in INPUT.rglob('FROZEN_BACKBONE_ENSEMBLE_POLICY.json'):
        try:
            if sha(p) == POLICY_FILE_SHA:
                matches.append(p)
        except OSError:
            pass
    if len(matches) != 1:
        raise RuntimeError(f'expected exactly one frozen policy with canonical hash, found {len(matches)}')
    return matches[0]

policy_path=unique_policy()
policy=json.loads(policy_path.read_text(encoding='utf-8'))
if policy.get('status') != 'FROZEN_FROM_DEVELOPMENT_ONLY': raise RuntimeError('policy status drift')
if policy.get('recipe_sha256') != RECIPE_SHA: raise RuntimeError('recipe SHA drift')
if policy.get('policy_sha256') != POLICY_SEMANTIC_SHA: raise RuntimeError('policy semantic SHA drift')
if policy.get('selected_candidates') != SELECTED: raise RuntimeError('selected candidates drift')
if policy.get('locked_test_used') is not False or policy.get('external_data_used') is not False: raise RuntimeError('forbidden evidence in policy')
fit=policy.get('deployment_fit') or {}
if fit.get('weights') != [1/3,1/3,1/3]: raise RuntimeError('deployment weights drift')
if not isinstance(fit.get('threshold'), (int,float)): raise RuntimeError('deployment threshold missing')
if len(fit.get('calibration') or []) != 3: raise RuntimeError('deployment calibration missing')

rows=[]
for cid in SELECTED:
    candidates=[]
    for config in INPUT.rglob('config.json'):
        if not config.parent.name.startswith(cid + '__seed-'):
            continue
        checkpoint=config.parent/'final_selected.keras'
        if checkpoint.is_file():
            candidates.append((config,checkpoint))
    if len(candidates) != 1:
        raise RuntimeError(f'{cid}: expected exactly one config/checkpoint pair, found {len(candidates)}')
    config, checkpoint = candidates[0]
    config_sha=sha(config)
    if config_sha != CONFIG_SHA[cid]: raise RuntimeError(f'{cid}: canonical config SHA mismatch')
    checkpoint_sha=sha(checkpoint)
    rows.extend([
      {'candidate_id':cid,'role':'config','basename':'config.json','bytes':config.stat().st_size,'sha256':config_sha,'source_path':str(config)},
      {'candidate_id':cid,'role':'checkpoint','basename':'final_selected.keras','bytes':checkpoint.stat().st_size,'sha256':checkpoint_sha,'source_path':str(checkpoint)},
    ])

qualification={
  'schema_version':1,
  'project':'PNEUMONIA V6.2.2',
  'qualification_performed':False,
  'used_for_selection':False,
  'final_lockbox_used':False,
  'external_data_used':False,
  'note':'Final ensemble was frozen from development-only OOF evidence; no qualification dataset was consumed for selection.'
}
qpath=OUT/'QUALIFICATION_NON_CONSUMPTION.json'
qpath.write_text(json.dumps(qualification,indent=2,sort_keys=True)+'\n',encoding='utf-8')
qsha=sha(qpath)

manifest={
  'schema_version':1,
  'project':'PNEUMONIA V6.2.2',
  'stage':'FINAL_FREEZE_MANIFEST',
  'status':'FROZEN_IN_KAGGLE',
  'master_account':'azadka',
  'source_kernel':BACKBONE_REF,
  'recipe_sha256':RECIPE_SHA,
  'frozen_policy_file_sha256':POLICY_FILE_SHA,
  'frozen_policy_semantic_sha256':POLICY_SEMANTIC_SHA,
  'selected_candidate_ids':SELECTED,
  'selected_artifact_count':6,
  'artifacts':rows,
  'deployment_fit':fit,
  'qualification_non_consumption_sha256':qsha,
  'locked_test_used':False,
  'external_data_used':False,
  'training_performed':False,
  'hpo_or_confirmation_repeated':False,
  'reselection_allowed_after_freeze':False,
  'next_permitted_stage':'OFFICIAL_LOCKED_TEST',
}
mpath=OUT/'FINAL_FREEZE_MANIFEST.json'
mpath.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
msha=sha(mpath)
(OUT/'FINAL_FREEZE_MANIFEST.sha256').write_text(msha+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','stage':'FINAL_FREEZE_MANIFEST','manifest_sha256':msha,'selected_artifacts':6,'locked_test_used':False,'external_data_used':False},indent=2))
`;

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}
function authorized(request: Request, env: Env): boolean {
  const token = env.CGP_PROJECT_CONTROL_TOKEN?.trim();
  return Boolean(token && token.length >= 32 && request.headers.get("authorization") === `Bearer ${token}`);
}
function authHeader(token: string): string {
  return token.startsWith("KGAT_") ? `Bearer ${token}` : `Basic ${btoa(`${OWNER}:${token}`)}`;
}
function rec(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
function normalizedStatus(raw: Record<string, unknown>): string {
  const session = rec(raw.session);
  for (const value of [raw.status, raw.statusName, raw.state, raw.sessionStatus, session.status, session.statusName, session.state]) {
    if (typeof value === "string" && value.trim()) return value.trim().toUpperCase();
  }
  return "UNKNOWN";
}
function allowedOutputUrl(raw: string): URL {
  const url = new URL(raw);
  const host = url.hostname.toLowerCase();
  if (url.protocol !== "https:" || !(host === "api.kaggle.com" || host === "www.kaggle.com" || host === "storage.googleapis.com" || host.endsWith(".kaggleusercontent.com") || host.endsWith(".googleusercontent.com"))) throw new Error("output URL host is not allowlisted");
  return url;
}
async function sha256Text(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, "0")).join("");
}
async function saveKernel(env: Env): Promise<Record<string, unknown>> {
  const token = env.CGP_KAGGLE_MASTER_TOKEN?.trim();
  if (!token) throw new Error("master Kaggle token missing");
  const request: Record<string, unknown> = {
    slug: TARGET_REF,
    newTitle: TARGET_TITLE,
    text: FREEZE_SCRIPT,
    language: "python",
    kernelType: "script",
    kernelExecutionType: "SAVE_AND_RUN_ALL",
    isPrivate: true,
    enableGpu: false,
    enableTpu: false,
    enableInternet: false,
    kernelDataSources: [BACKBONE_REF],
    datasetDataSources: [],
    competitionDataSources: [],
    modelDataSources: [],
  };
  const response = await fetch(`${API_ROOT}/SaveKernel`, { method: "POST", headers: { Authorization: authHeader(token), "Content-Type": "application/json", "User-Agent": "chatgpt-v622-finalization-freeze/1.0" }, body: JSON.stringify(request) });
  const text = await response.text();
  let value: Record<string, unknown> = {};
  try { value = rec(text ? JSON.parse(text) : {}); } catch { throw new Error(`SaveKernel non-JSON HTTP ${response.status}`); }
  const code = typeof value.code === "number" ? value.code : undefined;
  if (!response.ok || (code !== undefined && code >= 400)) throw new Error(String(value.message ?? `SaveKernel HTTP ${response.status}`).slice(0, 1000));
  return value;
}
async function outputReceipt(env: Env): Promise<Record<string, unknown> | null> {
  const token = env.CGP_KAGGLE_MASTER_TOKEN?.trim();
  if (!token) throw new Error("master Kaggle token missing");
  const response = await fetch(`${API_ROOT}/ListKernelSessionOutput`, { method: "POST", headers: { Authorization: authHeader(token), "Content-Type": "application/json", "User-Agent": "chatgpt-v622-finalization-freeze/1.0" }, body: JSON.stringify({ userName: OWNER, kernelSlug: TARGET_SLUG, pageSize: 100 }) });
  const payload = rec(await response.json());
  if (!response.ok) throw new Error(`ListKernelSessionOutput HTTP ${response.status}`);
  const files = Array.isArray(payload.files) ? payload.files.map(rec) : [];
  const manifestRow = files.find(row => String(row.fileName ?? "").endsWith("FINAL_FREEZE_MANIFEST.json") && typeof row.url === "string");
  const shaRow = files.find(row => String(row.fileName ?? "").endsWith("FINAL_FREEZE_MANIFEST.sha256") && typeof row.url === "string");
  if (!manifestRow || !shaRow) return null;
  const manifestText = await (await fetch(allowedOutputUrl(String(manifestRow.url)).toString(), { redirect: "follow" })).text();
  const declaredSha = (await (await fetch(allowedOutputUrl(String(shaRow.url)).toString(), { redirect: "follow" })).text()).trim().toLowerCase();
  const actualSha = await sha256Text(manifestText);
  if (actualSha !== declaredSha) throw new Error("FINAL_FREEZE_MANIFEST output SHA mismatch");
  const manifest = rec(JSON.parse(manifestText));
  if (manifest.project !== "PNEUMONIA V6.2.2" || manifest.stage !== "FINAL_FREEZE_MANIFEST" || manifest.status !== "FROZEN_IN_KAGGLE") throw new Error("FINAL_FREEZE_MANIFEST identity/status mismatch");
  if (manifest.source_kernel !== BACKBONE_REF || manifest.recipe_sha256 !== RECIPE_SHA256 || manifest.frozen_policy_file_sha256 !== POLICY_FILE_SHA256 || manifest.frozen_policy_semantic_sha256 !== POLICY_SEMANTIC_SHA256) throw new Error("FINAL_FREEZE_MANIFEST canonical hash mismatch");
  if (manifest.locked_test_used !== false || manifest.external_data_used !== false) throw new Error("forbidden evidence in FINAL_FREEZE_MANIFEST");
  return { manifest_sha256: actualSha, selected_artifact_count: manifest.selected_artifact_count, selected_candidate_ids: manifest.selected_candidate_ids, next_permitted_stage: manifest.next_permitted_stage, deployment_fit: manifest.deployment_fit };
}
async function launch(env: Env): Promise<Record<string, unknown>> {
  const existing = await listKernels(env, "master", TARGET_SLUG, 100);
  const exact = existing.some(item => String(rec(item).ref ?? "").toLowerCase() === TARGET_REF.toLowerCase());
  if (exact) {
    const statusPayload = await kernelStatus(env, "master", TARGET_REF);
    const status = normalizedStatus(statusPayload);
    if (ACTIVE.has(status)) return { action: "existing_active", target_kernel_ref: TARGET_REF, status, submitted: false };
    if (status === "COMPLETE") {
      const receipt = await outputReceipt(env);
      if (receipt) return { action: "existing_complete", target_kernel_ref: TARGET_REF, status, submitted: false, receipt };
    }
  }
  const result = await saveKernel(env);
  return { action: exact ? "submitted_new_version" : "created_and_submitted", target_kernel_ref: TARGET_REF, source_kernel_ref: BACKBONE_REF, result, submitted: true, kaggle_compute_kind: "FINALIZATION_FREEZE_ONLY", training: false, hpo: false, confirmation: false, locked_test: false, external_validation: false };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/healthz" && request.method === "GET") return json({ service: "v622-finalization-freeze-bridge", status: "ready", protected: true });
    if (!authorized(request, env)) return new Response("Forbidden", { status: 403 });
    try {
      if (url.pathname === "/control/v6-2-2/finalization-freeze/launch" && request.method === "POST") return json({ project: "PNEUMONIA V6.2.2", ...(await launch(env)) });
      if (url.pathname === "/control/v6-2-2/finalization-freeze/status" && request.method === "POST") {
        const statusPayload = await kernelStatus(env, "master", TARGET_REF);
        const status = normalizedStatus(statusPayload);
        const receipt = status === "COMPLETE" ? await outputReceipt(env) : null;
        return json({ project: "PNEUMONIA V6.2.2", target_kernel_ref: TARGET_REF, status, receipt });
      }
      return new Response("Not found", { status: 404 });
    } catch (error) {
      return json({ project: "PNEUMONIA V6.2.2", ok: false, error: error instanceof Error ? error.message.slice(0, 1000) : "unknown error" }, 502);
    }
  },
} satisfies ExportedHandler<Env>;
